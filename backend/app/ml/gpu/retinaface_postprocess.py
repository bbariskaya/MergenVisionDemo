"""RetinaFace R50 decode/NMS/scaling implemented entirely on the GPU.

Only small control-plane counters and the final selected landmarks are copied
back to the host.  This avoids the large device-to-host transfer of raw
``loc/conf/landms`` tensors.
"""
from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from itertools import product
from math import ceil

import numpy as np
from cuda.bindings import runtime as cuda_runtime

from app.ml.gpu.buffer_arena import BufferArena
from app.ml.gpu.device_tensor import DeviceTensor, check_cuda
from mergenvision_gpu import (
    argsort_descending,
    nms,
    retinaface_decode_batch,
    scale_clip_compact_xy,
)

logger = logging.getLogger(__name__)

_MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
_STEPS = [8, 16, 32]
_VARIANCE = np.array([0.1, 0.2], dtype=np.float32)
_CLIP = False


def _build_priors(image_size: int = 640) -> np.ndarray:
    """Return (16800, 4) prior box tensor in center-size form [cx,cy,w,h]."""
    anchors = []
    for k, step in enumerate(_STEPS):
        f_h = ceil(image_size / step)
        f_w = ceil(image_size / step)
        min_sizes = _MIN_SIZES[k]
        for i, j in product(range(f_h), range(f_w)):
            for min_size in min_sizes:
                s_kx = min_size / image_size
                s_ky = min_size / image_size
                cx = (j + 0.5) * step / image_size
                cy = (i + 0.5) * step / image_size
                anchors += [cx, cy, s_kx, s_ky]
    priors = np.array(anchors, dtype=np.float32).reshape(-1, 4)
    if _CLIP:
        np.clip(priors, 0.0, 1.0, out=priors)
    return priors


@dataclass(frozen=True)
class RetinaFaceDetections:
    """Surviving detections for one image, still normalized to 640-space."""

    boxes: DeviceTensor          # [N, 4]
    scores: DeviceTensor         # [N]
    landmarks: DeviceTensor      # [N, 10]
    count: int
    order: DeviceTensor          # [N]
    keep: DeviceTensor           # [N]


@dataclass(frozen=True)
class RetinaFaceScaledDetections:
    """Detections scaled back to original image coordinates."""

    boxes: DeviceTensor          # [K, 4]
    scores: DeviceTensor         # [K]
    landmarks: DeviceTensor      # [K, 10]
    count: DeviceTensor          # [1]


class RetinaFacePostprocess:
    """Decode RetinaFace R50 outputs, NMS, and scale on device."""

    def __init__(
        self,
        input_size: int = 640,
        device_id: int = 0,
        max_candidates: int = 2000,
    ) -> None:
        self._input_size = int(input_size)
        self._device_id = int(device_id)
        self._max_candidates = int(max_candidates)
        self._arena = BufferArena(device_id=device_id)
        self._priors = self._upload_priors(_build_priors(input_size))

    def _upload_priors(self, priors: np.ndarray) -> DeviceTensor:
        err, ptr = cuda_runtime.cudaMalloc(priors.nbytes)
        check_cuda(err, "retinaface priors cudaMalloc")
        err = cuda_runtime.cudaMemcpy(
            ptr,
            priors.ctypes.data,
            priors.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        check_cuda(err, "retinaface priors H2D")
        return DeviceTensor(
            ptr=int(ptr),
            shape=priors.shape,
            dtype=ctypes.c_float,
            device_id=self._device_id,
            owner=self,
            stream=None,
        )

    def decode(
        self,
        outputs: dict[str, DeviceTensor],
        *,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        stream: int | None = None,
    ) -> list[RetinaFaceDetections]:
        active_stream = stream if stream is not None else 0
        if set(outputs.keys()) != {"loc", "conf", "landms"}:
            raise ValueError(
                f"RetinaFace requires outputs loc/conf/landms, got {set(outputs.keys())}"
            )
        loc_t = outputs["loc"]
        conf_t = outputs["conf"]
        landms_t = outputs["landms"]
        if len(loc_t.shape) != 3 or len(conf_t.shape) != 3 or len(landms_t.shape) != 3:
            raise ValueError(
                "RetinaFace outputs must be 3D (batch x anchors x channels); "
                f"got loc={loc_t.shape}, conf={conf_t.shape}, landms={landms_t.shape}"
            )

        batch = loc_t.shape[0]
        num_anchors = loc_t.shape[1]
        if (
            conf_t.shape[0] != batch
            or conf_t.shape[1] != num_anchors
            or landms_t.shape[0] != batch
            or landms_t.shape[1] != num_anchors
        ):
            raise ValueError("RetinaFace output batch/anchor mismatch")
        if num_anchors != self._priors.shape[0]:
            raise ValueError(
                f"Anchor mismatch: priors={self._priors.shape[0]}, outputs={num_anchors}"
            )

        cand_boxes = self._arena.reserve(
            (batch, self._max_candidates, 4), ctypes.c_float, stream=active_stream
        )
        cand_scores = self._arena.reserve(
            (batch, self._max_candidates), ctypes.c_float, stream=active_stream
        )
        cand_landmarks = self._arena.reserve(
            (batch, self._max_candidates, 10), ctypes.c_float, stream=active_stream
        )
        counters = self._arena.reserve((batch,), ctypes.c_int32, stream=active_stream)
        err = cuda_runtime.cudaMemsetAsync(
            counters.ptr, 0, batch * 4, active_stream
        )
        check_cuda(err, "retinaface counters memset")

        retinaface_decode_batch(
            loc_t.ptr,
            conf_t.ptr,
            landms_t.ptr,
            self._priors.ptr,
            batch,
            num_anchors,
            conf_threshold,
            float(_VARIANCE[0]),
            float(_VARIANCE[1]),
            self._max_candidates,
            cand_boxes.ptr,
            cand_scores.ptr,
            cand_landmarks.ptr,
            counters.ptr,
            active_stream,
        )

        count_arr = np.empty(batch, dtype=np.int32)
        err = cuda_runtime.cudaMemcpyAsync(
            count_arr.ctypes.data,
            counters.ptr,
            batch * 4,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            active_stream,
        )
        check_cuda(err, "retinaface counters D2H")
        err = cuda_runtime.cudaStreamSynchronize(active_stream)
        check_cuda(err, "retinaface decode sync")

        per_image: list[RetinaFaceDetections] = []
        for b in range(batch):
            count = int(count_arr[b])
            if count > self._max_candidates:
                logger.warning(
                    "RetinaFace candidate overflow: image %d %d > %d", b, count, self._max_candidates
                )
                count = self._max_candidates
            if count == 0:
                per_image.append(
                    RetinaFaceDetections(
                        boxes=self._empty((0, 4), active_stream),
                        scores=self._empty((0,), active_stream),
                        landmarks=self._empty((0, 10), active_stream),
                        count=0,
                        order=self._empty((0,), active_stream, dtype=ctypes.c_int32),
                        keep=self._empty((0,), active_stream, dtype=ctypes.c_uint8),
                    )
                )
                continue

            # Per-image argsort.
            b_scores = DeviceTensor(
                ptr=cand_scores.ptr + b * self._max_candidates * 4,
                shape=(self._max_candidates,),
                dtype=ctypes.c_float,
                device_id=self._device_id,
                owner=cand_scores,
                stream=active_stream,
            )
            sort_scores = self._arena.reserve((count,), ctypes.c_float, stream=active_stream)
            err = cuda_runtime.cudaMemcpyAsync(
                sort_scores.ptr,
                b_scores.ptr,
                count * 4,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                active_stream,
            )
            check_cuda(err, "retinaface scores D2D")
            order = self._arena.reserve((count,), ctypes.c_int32, stream=active_stream)
            argsort_descending(sort_scores.ptr, order.ptr, count, active_stream)

            b_boxes = DeviceTensor(
                ptr=cand_boxes.ptr + b * self._max_candidates * 4 * 4,
                shape=(self._max_candidates, 4),
                dtype=ctypes.c_float,
                device_id=self._device_id,
                owner=cand_boxes,
                stream=active_stream,
            )
            keep = self._arena.reserve((count,), ctypes.c_uint8, stream=active_stream)
            nms(b_boxes.ptr, order.ptr, count, nms_threshold, keep.ptr, active_stream)

            per_image.append(
                RetinaFaceDetections(
                    boxes=b_boxes,
                    scores=b_scores,
                    landmarks=DeviceTensor(
                        ptr=cand_landmarks.ptr + b * self._max_candidates * 10 * 4,
                        shape=(self._max_candidates, 10),
                        dtype=ctypes.c_float,
                        device_id=self._device_id,
                        owner=cand_landmarks,
                        stream=active_stream,
                    ),
                    count=count,
                    order=order,
                    keep=keep,
                )
            )
        return per_image

    def scale_and_compact(
        self,
        detections: list[RetinaFaceDetections],
        *,
        original_heights: list[int],
        original_widths: list[int],
        stream: int | None = None,
    ) -> list[RetinaFaceScaledDetections]:
        active_stream = stream if stream is not None else 0
        scaled: list[RetinaFaceScaledDetections] = []
        for b, det in enumerate(detections):
            count = det.count
            out_boxes = self._arena.reserve((count, 4), ctypes.c_float, stream=active_stream)
            out_landmarks = self._arena.reserve((count, 10), ctypes.c_float, stream=active_stream)
            out_scores = self._arena.reserve((count,), ctypes.c_float, stream=active_stream)
            out_count = self._arena.reserve((1,), ctypes.c_int32, stream=active_stream)
            if count == 0:
                err = cuda_runtime.cudaMemsetAsync(out_count.ptr, 0, 4, active_stream)
                check_cuda(err, "retinaface zero count memset")
                scaled.append(
                    RetinaFaceScaledDetections(
                        boxes=out_boxes,
                        landmarks=out_landmarks,
                        scores=out_scores,
                        count=out_count,
                    )
                )
                continue

            scale_x = float(original_widths[b])
            scale_y = float(original_heights[b])
            scale_clip_compact_xy(
                det.boxes.ptr,
                det.landmarks.ptr,
                det.scores.ptr,
                det.order.ptr,
                det.keep.ptr,
                count,
                scale_x,
                scale_y,
                original_widths[b],
                original_heights[b],
                out_boxes.ptr,
                out_landmarks.ptr,
                out_scores.ptr,
                out_count.ptr,
                active_stream,
            )
            scaled.append(
                RetinaFaceScaledDetections(
                    boxes=out_boxes,
                    landmarks=out_landmarks,
                    scores=out_scores,
                    count=out_count,
                )
            )
        return scaled

    def _empty(
        self,
        shape: tuple[int, ...],
        stream: int,
        dtype: type = ctypes.c_float,
    ) -> DeviceTensor:
        if any(s == 0 for s in shape):
            return DeviceTensor(
                ptr=0,
                shape=shape,
                dtype=dtype,
                device_id=self._device_id,
                owner=self._arena,
                stream=stream,
            )
        return self._arena.reserve(shape, dtype, stream=stream)

    def close(self) -> None:
        self._arena.close()
        try:
            cuda_runtime.cudaFree(self._priors.ptr)
        except Exception:
            logger.exception("retinaface priors cudaFree failed")
