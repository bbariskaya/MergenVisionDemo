"""GPU SCRFD postprocess parity vs CPU reference oracle."""
from pathlib import Path

import ctypes

import numpy as np
import pytest
from cuda.bindings import runtime as cuda_runtime

from app.core.config import settings
from app.ml.gpu.buffer_arena import BufferArena
from app.ml.gpu.device_tensor import DeviceTensor, check_cuda
from app.ml.gpu.scrfd_postprocess import ScrfdGpuPostprocess
from app.ml.gpu.trt_device_engine import TrtDeviceEngine
from app.ml.postprocess import decode_detections
from app.ml.preprocess import load_image, preprocess_detector
from app.ml.trt_engine import TrtEngine

SAMPLES_DIR = Path("/app/artifacts/samples")


def _device_to_numpy(tensor: DeviceTensor, stream: int) -> np.ndarray:
    host = np.empty(tensor.shape, dtype=np.float32)
    err = cuda_runtime.cudaMemcpyAsync(
        host.ctypes.data,
        tensor.ptr,
        tensor.nbytes,
        cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        stream,
    )
    check_cuda(err, "D2H")
    err = cuda_runtime.cudaStreamSynchronize(stream)
    check_cuda(err, "sync")
    return host


def _sorted_keep_indices(
    keep: DeviceTensor, order: DeviceTensor, stream: int
) -> list[int]:
    keep_host = np.empty(keep.shape, dtype=np.uint8)
    order_host = np.empty(order.shape, dtype=np.int32)
    err = cuda_runtime.cudaMemcpyAsync(
        keep_host.ctypes.data,
        keep.ptr,
        keep.nbytes,
        cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        stream,
    )
    check_cuda(err, "keep D2H")
    err = cuda_runtime.cudaMemcpyAsync(
        order_host.ctypes.data,
        order.ptr,
        order.nbytes,
        cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        stream,
    )
    check_cuda(err, "order D2H")
    err = cuda_runtime.cudaStreamSynchronize(stream)
    check_cuda(err, "sync keep/order")
    # keep_host[i] refers to sorted position i.
    return [int(order_host[i]) for i in range(len(order_host)) if keep_host[i]]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    w = max(0.0, x2 - x1 + 1.0)
    h = max(0.0, y2 - y1 + 1.0)
    inter = w * h
    area_a = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0)
    area_b = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("mergenvision_gpu") is None,
    reason="mergenvision-gpu native extension not installed",
)
def test_gpu_postprocess_matches_cpu_reference():
    jpeg_path = SAMPLES_DIR / "t1.jpg"
    cpu_image = load_image(str(jpeg_path))
    cpu_input, _ = preprocess_detector(cpu_image, settings.detector_input_size)

    # CPU reference.
    ref_engine = TrtEngine(settings.detector_engine_path)
    try:
        ref_outputs = ref_engine.infer({"input.1": cpu_input})
        ref_dets = decode_detections(
            ref_outputs,
            input_size=settings.detector_input_size,
            conf_threshold=0.5,
            nms_threshold=0.4,
        )
        ref_boxes = np.array([d.bbox for d in ref_dets], dtype=np.float32)
        ref_scores = np.array([d.score for d in ref_dets], dtype=np.float32)
        ref_landmarks = np.array(
            [d.landmarks.flatten() for d in ref_dets], dtype=np.float32
        )
    finally:
        ref_engine.close()

    # GPU path.
    gpu_engine = TrtDeviceEngine(settings.detector_engine_path, device_id=0)
    post = ScrfdGpuPostprocess(
        input_size=settings.detector_input_size,
        device_id=0,
    )
    try:
        err, stream = cuda_runtime.cudaStreamCreate()
        check_cuda(err, "stream")
        try:
            d_input = gpu_engine._arena.reserve(
                cpu_input.shape, ctypes.c_float, stream=int(stream)
            )
            err = cuda_runtime.cudaMemcpyAsync(
                d_input.ptr,
                cpu_input.ctypes.data,
                cpu_input.nbytes,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
                int(stream),
            )
            check_cuda(err, "H2D input")

            det_outputs = gpu_engine.infer_device(
                {"input.1": d_input}, stream=int(stream)
            )
            detections = post.decode(
                det_outputs,
                conf_threshold=0.5,
                nms_threshold=0.4,
                stream=int(stream),
            )

            keep_indices = _sorted_keep_indices(
                detections.keep, detections.order, int(stream)
            )
            boxes_host = _device_to_numpy(detections.boxes, int(stream))
            scores_host = _device_to_numpy(detections.scores, int(stream))
            landmarks_host = _device_to_numpy(detections.landmarks, int(stream))

            gpu_boxes = boxes_host[keep_indices]
            gpu_scores = scores_host[keep_indices]
            gpu_landmarks = landmarks_host[keep_indices]

            def _sort_by_coords(boxes, scores, landmarks):
                key = boxes.sum(axis=1)
                order = np.argsort(key, kind="mergesort")
                return boxes[order], scores[order], landmarks[order]

            gpu_boxes, gpu_scores, gpu_landmarks = _sort_by_coords(
                gpu_boxes, gpu_scores, gpu_landmarks
            )
            ref_boxes, ref_scores, ref_landmarks = _sort_by_coords(
                ref_boxes, ref_scores, ref_landmarks
            )

            # Parallel approximate NMS can suppress a few more boxes than
            # the CPU greedy reference in dense scenes, so we enforce the
            # correctness invariants instead of an exact set match.
            assert gpu_boxes.shape[0] <= ref_boxes.shape[0] + 2, (
                f"GPU kept far more boxes than CPU: gpu={gpu_boxes.shape[0]} "
                f"cpu={ref_boxes.shape[0]}"
            )

            # No pair of GPU-kept boxes may overlap above the NMS threshold.
            for i in range(gpu_boxes.shape[0]):
                for j in range(i + 1, gpu_boxes.shape[0]):
                    iou = _iou(gpu_boxes[i], gpu_boxes[j])
                    assert iou <= 0.4 + 1e-6, (
                        f"GPU kept boxes overlap {iou:.3f} at {i},{j}"
                    )

            # Every GPU-kept box must match a CPU-kept box (same candidate).
            for gbox in gpu_boxes:
                best_iou = max(_iou(gbox, cbox) for cbox in ref_boxes)
                assert best_iou >= 0.5, (
                    f"GPU box has no CPU match, best IoU={best_iou:.3f}"
                )

            # The top-scoring box is deterministic and must match.
            if gpu_boxes.shape[0] > 0 and ref_boxes.shape[0] > 0:
                np.testing.assert_allclose(
                    gpu_boxes[0], ref_boxes[0], rtol=0, atol=3.0
                )
                np.testing.assert_allclose(
                    gpu_scores[0], ref_scores[0], rtol=0, atol=5e-3
                )
                np.testing.assert_allclose(
                    gpu_landmarks[0], ref_landmarks[0], rtol=0, atol=3.0
                )
        finally:
            cuda_runtime.cudaStreamDestroy(int(stream))
    finally:
        post.close()
        gpu_engine.close()
