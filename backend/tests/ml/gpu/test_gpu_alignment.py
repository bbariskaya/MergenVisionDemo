"""GPU face-alignment parity vs CPU OpenCV oracle."""
import ctypes

import numpy as np
import pytest
from cuda.bindings import runtime as cuda_runtime

from app.ml.alignment import align_face, ARC_FACE_SRC
from app.ml.gpu.alignment import GpuFaceAligner
from app.ml.gpu.buffer_arena import BufferArena
from app.ml.gpu.device_tensor import DeviceTensor, check_cuda


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


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("mergenvision_gpu") is None,
    reason="mergenvision-gpu native extension not installed",
)
def test_gpu_alignment_matches_cpu_oracle():
    h, w = 300, 400
    cpu_image = np.zeros((h, w, 3), dtype=np.uint8)
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    cpu_image[:, :, 0] = (x * 255 // w).astype(np.uint8)
    cpu_image[:, :, 1] = (y * 255 // h).astype(np.uint8)
    cpu_image[:, :, 2] = ((x + y) * 255 // (w + h)).astype(np.uint8)

    base_landmarks = ARC_FACE_SRC.copy()
    base_landmarks += np.array([120.0, 80.0], dtype=np.float32)
    landmarks = np.tile(base_landmarks.reshape(1, 10), (3, 1)).astype(np.float32)
    for i in range(1, 3):
        landmarks[i, 0::2] += i * 80.0

    cpu_chips = np.array(
        [align_face(cpu_image, lm.reshape(5, 2)) for lm in landmarks],
        dtype=np.float32,
    )
    cpu_chips = np.transpose(cpu_chips, (0, 3, 1, 2))
    cpu_chips = (cpu_chips - 127.5) / 127.5

    aligner = GpuFaceAligner(device_id=0)
    arena = BufferArena(device_id=0)
    err, stream = cuda_runtime.cudaStreamCreate()
    check_cuda(err, "stream")
    try:
        d_image = arena.reserve(
            (1, h, w, 3), ctypes.c_uint8, stream=int(stream)
        )
        cuda_runtime.cudaMemcpyAsync(
            d_image.ptr,
            cpu_image.ctypes.data,
            cpu_image.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
            int(stream),
        )
        d_landmarks = arena.reserve(
            landmarks.shape, ctypes.c_float, stream=int(stream)
        )
        cuda_runtime.cudaMemcpyAsync(
            d_landmarks.ptr,
            landmarks.ctypes.data,
            landmarks.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
            int(stream),
        )
        gpu_chips = aligner.align(
            d_image, d_landmarks, stream=int(stream)
        )
        gpu_chips_host = _device_to_numpy(gpu_chips, int(stream))
    finally:
        cuda_runtime.cudaStreamDestroy(int(stream))
        aligner.close()
        arena.close()

    assert gpu_chips_host.shape == cpu_chips.shape
    np.testing.assert_allclose(gpu_chips_host, cpu_chips, rtol=0, atol=0.1)
