"""GPU SCRFD decode + filter + NMS using the native CUDA extension.

No raw detector tensor leaves the device. Candidate count after threshold is
bounded; only the final NMS keep mask and small control-plane counters are
read back.
"""
from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass

import numpy as np
from cuda.bindings import runtime as cuda_runtime

from app.ml.gpu.buffer_arena import BufferArena
from app.ml.gpu.device_tensor import DeviceTensor, check_cuda
from mergenvision_gpu import argsort_descending, nms, scale_clip_compact, scrfd_decode_level

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuDetections:
    boxes: DeviceTensor          # [N, 4]
    scores: DeviceTensor         # [N]
    landmarks: DeviceTensor      # [N, 10]
    count: int                   # N (candidates after threshold, before NMS)
    order: DeviceTensor          # [N] argsort indices
    keep: DeviceTensor           # [N] uint8 keep mask (post-NMS)


class ScrfdGpuPostprocess:
    """Decode SCRFD outputs on device."""

    def __init__(
        self,
        input_size: int = 640,
        strides: tuple[int, ...] = (8, 16, 32),
        device_id: int = 0,
        anchors_per_location: int = 2,
        max_candidates: int = 2000,
    ) -> None:
        self._input_size = int(input_size)
        self._strides = tuple(strides)
        self._device_id = int(device_id)
        self._anchors_per_location = int(anchors_per_location)
        self._max_candidates = int(max_candidates)
        self._arena = BufferArena(device_id=device_id)
        self._scaled_boxes = self._arena.reserve(
            (self._max_candidates, 4), ctypes.c_float, stream=0
        )
        self._scaled_landmarks = self._arena.reserve(
            (self._max_candidates, 10), ctypes.c_float, stream=0
        )
        self._scaled_scores = self._arena.reserve(
            (self._max_candidates,), ctypes.c_float, stream=0
        )
        self._scaled_count = self._arena.reserve((1,), ctypes.c_int32, stream=0)
        self._anchors = self._build_anchors()

    def _build_anchors(self) -> DeviceTensor:
        # Official SCRFD anchors are top-left grid corners; no +0.5 offset.
        anchor_list = []
        for stride in self._strides:
            grid = self._input_size // stride
            for y in range(grid):
                for x in range(grid):
                    for _ in range(self._anchors_per_location):
                        anchor_list.append((float(x * stride), float(y * stride)))
        arr = np.array(anchor_list, dtype=np.float32).reshape(-1, 2)
        nbytes = arr.nbytes
        shape = (arr.shape[0], 2)

        err, ptr = cuda_runtime.cudaMalloc(nbytes)
        check_cuda(err, "anchor cudaMalloc")
        err = cuda_runtime.cudaMemcpy(
            ptr,
            arr.ctypes.data,
            nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        check_cuda(err, "anchor H2D")
        # Anchors live for the lifetime of the postprocessor.
        return DeviceTensor(
            ptr=int(ptr),
            shape=shape,
            dtype=ctypes.c_float,
            device_id=self._device_id,
            owner=self,
            stream=None,
        )

    def _group_outputs(
        self,
        outputs: dict[str, DeviceTensor],
    ) -> list[tuple[int, DeviceTensor, DeviceTensor, DeviceTensor]]:
        """Group detector outputs by FPN level (stride)."""
        scores: list[DeviceTensor] = []
        bboxes: list[DeviceTensor] = []
        landmarks: list[DeviceTensor] = []
        for tensor in outputs.values():
            if len(tensor.shape) == 3:
                _, a, c = tensor.shape
            elif len(tensor.shape) == 2:
                a, c = tensor.shape
            else:
                raise ValueError(f"Unexpected detector output shape {tensor.shape}")
            if c == 1:
                scores.append(tensor)
            elif c == 4:
                bboxes.append(tensor)
            elif c == 10:
                landmarks.append(tensor)
            else:
                raise ValueError(f"Unexpected detector output channel {c}")

        def count(t: DeviceTensor) -> int:
            return t.shape[1]

        scores = sorted(scores, key=count, reverse=True)
        bboxes = sorted(bboxes, key=count, reverse=True)
        landmarks = sorted(landmarks, key=count, reverse=True)

        if not (len(scores) == len(bboxes) == len(landmarks) == len(self._strides)):
            raise ValueError(
                f"Output count mismatch: scores={len(scores)} bboxes={len(bboxes)} "
                f"landmarks={len(landmarks)} strides={len(self._strides)}"
            )

        return [
            (stride, scores[i], bboxes[i], landmarks[i])
            for i, stride in enumerate(self._strides)
        ]

    def decode(
        self,
        outputs: dict[str, DeviceTensor],
        *,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        stream: int | None = None,
    ) -> GpuDetections:
        active_stream = stream if stream is not None else 0

        levels = self._group_outputs(outputs)
        total_anchors = sum(s.shape[0] for _, s, _, _ in levels)
        if total_anchors != self._anchors.shape[0]:
            raise ValueError(
                f"Anchor mismatch: generated {self._anchors.shape[0]}, "
                f"outputs sum {total_anchors}"
            )

        # Preallocate candidate buffers.
        cand_boxes = self._arena.reserve(
            (self._max_candidates, 4), ctypes.c_float, stream=active_stream
        )
        cand_scores = self._arena.reserve(
            (self._max_candidates,), ctypes.c_float, stream=active_stream
        )
        cand_landmarks = self._arena.reserve(
            (self._max_candidates, 10), ctypes.c_float, stream=active_stream
        )
        counter = self._arena.reserve(
            (1,), ctypes.c_int32, stream=active_stream
        )
        err = cuda_runtime.cudaMemsetAsync(
            counter.ptr, 0, 4, active_stream
        )
        check_cuda(err, "counter memset")

        anchor_offset = 0
        for stride, score_t, bbox_t, kps_t in levels:
            num = score_t.shape[1] if len(score_t.shape) == 3 else score_t.shape[0]
            scrfd_decode_level(
                score_t.ptr,
                bbox_t.ptr,
                kps_t.ptr,
                self._anchors.ptr + anchor_offset * 2 * ctypes.sizeof(ctypes.c_float),
                num,
                stride,
                conf_threshold,
                cand_boxes.ptr,
                cand_scores.ptr,
                cand_landmarks.ptr,
                counter.ptr,
                self._max_candidates,
                active_stream,
            )
            anchor_offset += num

        # Read candidate count.
        count_arr = np.empty(1, dtype=np.int32)
        err = cuda_runtime.cudaMemcpyAsync(
            count_arr.ctypes.data,
            counter.ptr,
            4,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            active_stream,
        )
        check_cuda(err, "counter D2H")
        err = cuda_runtime.cudaStreamSynchronize(active_stream)
        check_cuda(err, "sync count")
        count = int(count_arr[0])
        if count > self._max_candidates:
            logger.warning(
                "SCRFD candidate overflow: %d > %d", count, self._max_candidates
            )
            count = self._max_candidates

        # Argsort (descending) on scores. Thrust sorts in-place, so copy first.
        sort_scores = self._arena.reserve(
            (count,), ctypes.c_float, stream=active_stream
        )
        err = cuda_runtime.cudaMemcpyAsync(
            sort_scores.ptr,
            cand_scores.ptr,
            count * ctypes.sizeof(ctypes.c_float),
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            active_stream,
        )
        check_cuda(err, "scores D2D for sort")
        order = self._arena.reserve((count,), ctypes.c_int32, stream=active_stream)
        argsort_descending(
            sort_scores.ptr, order.ptr, count, active_stream
        )

        keep = self._arena.reserve(
            (count,), ctypes.c_uint8, stream=active_stream
        )
        nms(cand_boxes.ptr, order.ptr, count, nms_threshold, keep.ptr, active_stream)

        return GpuDetections(
            boxes=cand_boxes,
            scores=cand_scores,
            landmarks=cand_landmarks,
            count=count,
            order=order,
            keep=keep,
        )

    def scale_and_compact(
        self,
        detections: GpuDetections,
        *,
        original_height: int,
        original_width: int,
        stream: int | None = None,
    ) -> tuple[DeviceTensor, DeviceTensor, DeviceTensor, DeviceTensor]:
        """Scale NMS-surviving detections back to original image coordinates.

        Everything stays on device; only the compacted count is read back
        by the caller.
        """
        active_stream = stream if stream is not None else 0
        scale = self._input_size / max(original_height, original_width)
        inv_scale = 1.0 / scale

        scale_clip_compact(
            detections.boxes.ptr,
            detections.landmarks.ptr,
            detections.scores.ptr,
            detections.order.ptr,
            detections.keep.ptr,
            detections.count,
            inv_scale,
            original_width,
            original_height,
            self._scaled_boxes.ptr,
            self._scaled_landmarks.ptr,
            self._scaled_scores.ptr,
            self._scaled_count.ptr,
            active_stream,
        )
        return (
            self._scaled_boxes,
            self._scaled_landmarks,
            self._scaled_scores,
            self._scaled_count,
        )

    def close(self) -> None:
        self._arena.close()
        # Anchor buffer was allocated separately and is owned by self; free it.
        err = cuda_runtime.cudaFree(self._anchors.ptr)
        check_cuda(err, "anchor cudaFree")
