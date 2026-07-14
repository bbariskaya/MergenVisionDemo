"""End-to-end GPU-only face extraction pipeline.

encoded JPEG bytes -> nvImageCodec decode -> CV-CUDA detector preprocess
-> TensorRT SCRFD -> native CUDA decode/NMS -> scale/clamp -> GPU alignment
-> TensorRT ArcFace -> native CUDA L2 normalize -> CPU result
"""
from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cuda.bindings import runtime as cuda_runtime

from app.core.config import Settings, settings as default_settings
from app.ml.gpu.alignment import GpuFaceAligner
from app.ml.gpu.buffer_arena import BufferArena
from app.ml.gpu.decoder import DecodeInfo, JpegGpuDecoder
from app.ml.gpu.device_tensor import DeviceTensor, check_cuda
from app.ml.gpu.preprocess import GpuDetectorPreprocessor
from app.ml.gpu.recognizer import GpuRecognizer
from app.ml.gpu.retinaface_postprocess import RetinaFacePostprocess
from app.ml.gpu.retinaface_preprocessor import RetinaFacePreprocessor
from app.ml.gpu.scrfd_postprocess import ScrfdGpuPostprocess
from app.ml.gpu.trt_device_engine import TrtDeviceEngine

logger = logging.getLogger(__name__)


@dataclass
class GpuFaceExtraction:
    bbox: np.ndarray  # [x1, y1, x2, y2] in original image space
    landmarks: np.ndarray  # [5, 2] in original image space
    embedding: np.ndarray  # L2-normalized
    score: float


class GpuFacePipeline:
    """Single-GPU face detection + alignment + recognition pipeline."""

    def __init__(
        self,
        cfg: Settings = default_settings,
        device_id: int = 0,
    ) -> None:
        self._cfg = cfg
        self._device_id = device_id
        self._set_device()
        self._model_pack = cfg.model_pack
        self._decoder = JpegGpuDecoder(device_id=device_id)

        if self._model_pack == "retinaface_r50":
            self._preprocessor = RetinaFacePreprocessor(
                input_size=cfg.detector_input_size, device_id=device_id
            )
            self._detector_engine = TrtDeviceEngine(
                cfg.detector_engine_path.with_name("retinaface_r50_dynamic.engine"),
                device_id=device_id,
            )
            self._postprocess = RetinaFacePostprocess(
                input_size=cfg.detector_input_size,
                device_id=device_id,
            )
        else:
            self._preprocessor = GpuDetectorPreprocessor(
                input_size=cfg.detector_input_size, device_id=device_id
            )
            self._detector_engine = TrtDeviceEngine(
                cfg.detector_engine_path, device_id=device_id
            )
            self._postprocess = ScrfdGpuPostprocess(
                input_size=cfg.detector_input_size,
                device_id=device_id,
            )

        self._aligner = GpuFaceAligner(device_id=device_id)
        self._recognizer = GpuRecognizer(
            engine_path=cfg.embedder_engine_path,
            device_id=device_id,
            settings=cfg,
        )
        self._arena = BufferArena(device_id=device_id)
        self._align_status = self._arena.reserve((1,), ctypes.c_int32, stream=0)
        self._l2_status = self._arena.reserve((1,), ctypes.c_int32, stream=0)
        err, self._stream = cuda_runtime.cudaStreamCreate()
        check_cuda(err, "pipeline stream create")

    def _set_device(self) -> None:
        err = cuda_runtime.cudaSetDevice(self._device_id)
        check_cuda(err, f"cudaSetDevice({self._device_id})")

    def warmup(self) -> None:
        """Warm up all engine contexts and CUDA allocations."""
        self._set_device()
        input_name = self._detector_engine._input_names[0]
        self._detector_engine.warmup(
            {input_name: (1, 3, self._cfg.detector_input_size, self._cfg.detector_input_size)}
        )
        self._recognizer.warmup()
        logger.info("GPU face pipeline warmup complete")

    def close(self) -> None:
        try:
            self._set_device()
        except Exception as exc:
            logger.warning("set device before close failed: %s", exc)
        try:
            self._recognizer.close()
        except Exception as exc:
            logger.warning("recognizer close failed: %s", exc)
        try:
            self._aligner.close()
        except Exception as exc:
            logger.warning("aligner close failed: %s", exc)
        try:
            self._postprocess.close()
        except Exception as exc:
            logger.warning("postprocess close failed: %s", exc)
        try:
            self._detector_engine.close()
        except Exception as exc:
            logger.warning("detector close failed: %s", exc)
        try:
            self._preprocessor.close()
        except Exception as exc:
            logger.warning("preprocessor close failed: %s", exc)
        try:
            self._decoder.close()
        except Exception as exc:
            logger.warning("decoder close failed: %s", exc)
        try:
            self._arena.close()
        except Exception as exc:
            logger.warning("arena close failed: %s", exc)
        if hasattr(self, "_stream"):
            try:
                cuda_runtime.cudaStreamDestroy(self._stream)
            except Exception as exc:
                logger.warning("stream destroy failed: %s", exc)
            del self._stream

    def __enter__(self) -> "GpuFacePipeline":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _scaled_to_host(
        self,
        scaled,
        stream: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Copy a scaled detection result to host and return count."""
        count_arr = np.empty(1, dtype=np.int32)
        err = cuda_runtime.cudaMemcpyAsync(
            count_arr.ctypes.data,
            scaled.count.ptr,
            4,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        )
        check_cuda(err, "scaled count D2H")
        err = cuda_runtime.cudaStreamSynchronize(stream)
        check_cuda(err, "scaled count sync")
        n_faces = int(count_arr[0])
        if n_faces == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 10), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                0,
            )
        boxes_host = np.empty((n_faces, 4), dtype=np.float32)
        landmarks_host = np.empty((n_faces, 10), dtype=np.float32)
        scores_host = np.empty((n_faces,), dtype=np.float32)
        err = cuda_runtime.cudaMemcpyAsync(
            boxes_host.ctypes.data,
            scaled.boxes.ptr,
            boxes_host.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        )
        check_cuda(err, "scaled boxes D2H")
        err = cuda_runtime.cudaMemcpyAsync(
            landmarks_host.ctypes.data,
            scaled.landmarks.ptr,
            landmarks_host.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        )
        check_cuda(err, "scaled landmarks D2H")
        err = cuda_runtime.cudaMemcpyAsync(
            scores_host.ctypes.data,
            scaled.scores.ptr,
            scores_host.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        )
        check_cuda(err, "scaled scores D2H")
        err = cuda_runtime.cudaStreamSynchronize(stream)
        check_cuda(err, "scaled sync")
        return boxes_host, landmarks_host, scores_host, n_faces

    def extract_bytes(self, image_bytes: bytes) -> list[GpuFaceExtraction]:
        """Extract all faces from a JPEG/PNG byte buffer."""
        self._set_device()
        if not image_bytes:
            return []

        d_image, info = self._decoder.decode(
            image_bytes, stream=int(self._stream)
        )
        d_input = self._preprocessor.preprocess(
            d_image, stream=int(self._stream)
        )

        det_input_name = self._detector_engine._input_names[0]
        det_outputs = self._detector_engine.infer_device(
            {det_input_name: d_input},
            stream=int(self._stream),
        )
        detections = self._postprocess.decode(
            det_outputs,
            conf_threshold=self._cfg.detector_confidence_threshold,
            nms_threshold=self._cfg.detector_nms_iou,
            stream=int(self._stream),
        )

        if self._model_pack == "retinaface_r50":
            scaled_list = self._postprocess.scale_and_compact(
                detections,
                original_heights=[info.height],
                original_widths=[info.width],
                stream=int(self._stream),
            )
            scaled = scaled_list[0]
        else:
            d_scaled_boxes, d_scaled_landmarks, d_scaled_scores, d_count = \
                self._postprocess.scale_and_compact(
                    detections,
                    original_height=info.height,
                    original_width=info.width,
                    stream=int(self._stream),
                )
            from app.ml.gpu.retinaface_postprocess import RetinaFaceScaledDetections
            scaled = RetinaFaceScaledDetections(
                boxes=d_scaled_boxes,
                landmarks=d_scaled_landmarks,
                scores=d_scaled_scores,
                count=d_count,
            )

        del det_outputs

        boxes_host, landmarks_host, scores_host, n_faces = self._scaled_to_host(
            scaled, stream=int(self._stream)
        )
        if n_faces == 0:
            return []

        d_landmarks = DeviceTensor(
            scaled.landmarks.ptr,
            (n_faces, 10),
            ctypes.c_float,
            self._device_id,
            scaled.landmarks,
            stream=int(self._stream),
        )
        d_chips = self._aligner.align(
            d_image, d_landmarks, stream=int(self._stream), status=self._align_status
        )
        err = cuda_runtime.cudaMemsetAsync(
            self._l2_status.ptr, 0, 4, int(self._stream)
        )
        check_cuda(err, "l2 status memset")
        d_embeddings = self._recognizer.embed(
            d_chips, stream=int(self._stream), status=self._l2_status
        )

        embeddings_host = np.empty(d_embeddings.shape, dtype=np.float32)
        status_host = np.empty(2, dtype=np.int32)
        err = cuda_runtime.cudaMemcpyAsync(
            embeddings_host.ctypes.data,
            d_embeddings.ptr,
            embeddings_host.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            int(self._stream),
        )
        check_cuda(err, "embeddings D2H")
        err = cuda_runtime.cudaMemcpyAsync(
            status_host[0:1].ctypes.data,
            self._align_status.ptr,
            4,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            int(self._stream),
        )
        check_cuda(err, "align status D2H")
        err = cuda_runtime.cudaMemcpyAsync(
            status_host[1:2].ctypes.data,
            self._l2_status.ptr,
            4,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            int(self._stream),
        )
        check_cuda(err, "l2 status D2H")
        err = cuda_runtime.cudaStreamSynchronize(int(self._stream))
        check_cuda(err, "final extract sync")

        if status_host[0] != 0:
            raise ValueError(
                f"similarity_transform failed (status={status_host[0]}); "
                "non-finite or degenerate landmarks"
            )
        if status_host[1] & 1:
            raise ValueError("l2_normalize input contains non-finite values")
        if status_host[1] & 2:
            raise ValueError("l2_normalize encountered a zero-norm row")

        results: list[GpuFaceExtraction] = []
        for bbox, lms, emb, score in zip(
            boxes_host, landmarks_host.reshape(-1, 5, 2), embeddings_host, scores_host
        ):
            results.append(
                GpuFaceExtraction(
                    bbox=bbox,
                    landmarks=lms,
                    embedding=emb,
                    score=float(score),
                )
            )
        return results

    def extract_batch(
        self,
        image_bytes_list: list[bytes],
        *,
        pick_largest: bool = True,
        max_batch: int = 256,
    ) -> list[GpuFaceExtraction | None]:
        """Batch extraction optimised for RetinaFace R50 dynamic batch."""
        self._set_device()
        if self._model_pack != "retinaface_r50":
            return [
                self._pick_largest(self.extract_bytes(b))
                if pick_largest
                else None
                for b in image_bytes_list
            ]

        n = len(image_bytes_list)
        if n == 0:
            return []

        batch_size = min(n, max_batch)

        results: list[GpuFaceExtraction | None] = [None] * n
        for chunk_start in range(0, n, batch_size):
            chunk = image_bytes_list[chunk_start : chunk_start + batch_size]
            b = len(chunk)

            d_images, infos = self._decoder.decode_batch(
                chunk, stream=int(self._stream)
            )
            d_input = self._preprocessor.preprocess_batch(
                d_images, stream=int(self._stream)
            )

            input_name = self._detector_engine._input_names[0]
            det_outputs = self._detector_engine.infer_device(
                {input_name: d_input},
                stream=int(self._stream),
            )
            per_image = self._postprocess.decode(
                det_outputs,
                conf_threshold=self._cfg.detector_confidence_threshold,
                nms_threshold=self._cfg.detector_nms_iou,
                stream=int(self._stream),
            )
            scaled_list = self._postprocess.scale_and_compact(
                per_image,
                original_heights=[info.height for info in infos],
                original_widths=[info.width for info in infos],
                stream=int(self._stream),
            )

            batch_selections = self._postprocess.pick_largest_device(
                scaled_list,
                stream=int(self._stream),
            )

            valid_host = np.empty(b, dtype=np.int32)
            err = cuda_runtime.cudaMemcpyAsync(
                valid_host.ctypes.data,
                batch_selections.valid.ptr,
                valid_host.nbytes,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                int(self._stream),
            )
            check_cuda(err, "batch selections valid D2H")
            err = cuda_runtime.cudaStreamSynchronize(int(self._stream))
            check_cuda(err, "valid sync")

            pending_host: list[tuple[int, list[np.ndarray]]] = []
            for i in range(b):
                if not valid_host[i]:
                    continue
                host_row = [
                    np.empty((4,), dtype=np.float32),
                    np.empty((10,), dtype=np.float32),
                    np.empty((1,), dtype=np.float32),
                ]
                err = cuda_runtime.cudaMemcpyAsync(
                    host_row[0].ctypes.data,
                    batch_selections.boxes.ptr + i * 4 * 4,
                    4 * 4,
                    cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    int(self._stream),
                )
                check_cuda(err, "selected box D2H")
                err = cuda_runtime.cudaMemcpyAsync(
                    host_row[1].ctypes.data,
                    batch_selections.landmarks.ptr + i * 10 * 4,
                    10 * 4,
                    cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    int(self._stream),
                )
                check_cuda(err, "selected landmarks D2H")
                err = cuda_runtime.cudaMemcpyAsync(
                    host_row[2].ctypes.data,
                    batch_selections.scores.ptr + i * 4,
                    1 * 4,
                    cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    int(self._stream),
                )
                check_cuda(err, "selected score D2H")
                pending_host.append((i, host_row))

            err = cuda_runtime.cudaStreamSynchronize(int(self._stream))
            check_cuda(err, "selected rows sync")

            selected_meta: list[tuple[int, int, np.ndarray, np.ndarray, float]] = [
                (
                    chunk_start + i,
                    i,
                    host_row[0],
                    host_row[1],
                    float(host_row[2][0]),
                )
                for i, host_row in pending_host
            ]
            selected_indices = [i for i, _ in pending_host]

            if not selected_meta:
                continue

            m = len(selected_meta)
            chip_batch = self._arena.reserve(
                (m, 3, self._cfg.embedder_input_size, self._cfg.embedder_input_size),
                ctypes.c_float,
                stream=int(self._stream),
            )
            chip_plane_size = (
                self._cfg.embedder_input_size
                * self._cfg.embedder_input_size
                * 3
                * 4
            )

            for j, (_, img_idx, _, _, _) in enumerate(selected_meta):
                d_landmarks = DeviceTensor(
                    batch_selections.landmarks.ptr + selected_indices[j] * 10 * 4,
                    (1, 10),
                    ctypes.c_float,
                    self._device_id,
                    batch_selections.landmarks,
                    stream=int(self._stream),
                )
                chip = self._aligner.align(
                    d_images[img_idx],
                    d_landmarks,
                    stream=int(self._stream),
                    status=self._align_status,
                )
                err = cuda_runtime.cudaMemcpyAsync(
                    chip_batch.ptr + j * chip_plane_size,
                    chip.ptr,
                    chip_plane_size,
                    cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    int(self._stream),
                )
                check_cuda(err, "extract_batch chip D2D")

            d_embeddings = self._recognizer.embed(
                DeviceTensor(
                    ptr=chip_batch.ptr,
                    shape=(m, 3, self._cfg.embedder_input_size, self._cfg.embedder_input_size),
                    dtype=ctypes.c_float,
                    device_id=self._device_id,
                    owner=chip_batch,
                    stream=int(self._stream),
                ),
                stream=int(self._stream),
                status=self._l2_status,
            )

            embeddings_host = np.empty(d_embeddings.shape, dtype=np.float32)
            status_host = np.empty(2, dtype=np.int32)
            err = cuda_runtime.cudaMemcpyAsync(
                embeddings_host.ctypes.data,
                d_embeddings.ptr,
                embeddings_host.nbytes,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                int(self._stream),
            )
            check_cuda(err, "extract_batch embeddings D2H")
            err = cuda_runtime.cudaMemcpyAsync(
                status_host[0:1].ctypes.data,
                self._align_status.ptr,
                4,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                int(self._stream),
            )
            check_cuda(err, "align status D2H")
            err = cuda_runtime.cudaMemcpyAsync(
                status_host[1:2].ctypes.data,
                self._l2_status.ptr,
                4,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                int(self._stream),
            )
            check_cuda(err, "l2 status D2H")
            err = cuda_runtime.cudaStreamSynchronize(int(self._stream))
            check_cuda(err, "extract_batch final sync")

            if status_host[0] != 0:
                raise ValueError(
                    f"similarity_transform failed (status={status_host[0]}); "
                    "non-finite or degenerate landmarks"
                )
            if status_host[1] & 1:
                raise ValueError("l2_normalize input contains non-finite values")
            if status_host[1] & 2:
                raise ValueError("l2_normalize encountered a zero-norm row")

            for j, (global_idx, _, bbox, landmarks, score) in enumerate(selected_meta):
                results[global_idx] = GpuFaceExtraction(
                    bbox=bbox,
                    landmarks=landmarks.reshape(5, 2),
                    embedding=embeddings_host[j],
                    score=score,
                )

        return results

    def _pick_largest(self, faces: list[GpuFaceExtraction]) -> GpuFaceExtraction | None:
        if not faces:
            return None
        if len(faces) == 1:
            return faces[0]
        areas = [
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces
        ]
        return faces[int(np.argmax(areas))]
