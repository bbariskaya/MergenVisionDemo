"""Tests for TrtDeviceEngine direct device inference."""
import ctypes

import numpy as np
import pytest
from cuda.bindings import runtime as cuda_runtime

from app.core.config import settings
from app.ml.gpu.buffer_arena import BufferArena
from app.ml.gpu.device_tensor import DeviceTensor, check_cuda
from app.ml.gpu.trt_device_engine import TrtDeviceEngine


@pytest.fixture(scope="module")
def arena():
    a = BufferArena(device_id=0)
    yield a
    a.close()


@pytest.fixture
def recognition_engine(tmp_path_factory):
    # Module-scoped engines would be nicer, but we keep fixture per-test to
    # avoid cross-test state; this is acceptable for small smoke tests.
    engine = TrtDeviceEngine(settings.embedder_engine_path, device_id=0)
    yield engine
    engine.close()


def _device_to_numpy(tensor: DeviceTensor, stream: int) -> np.ndarray:
    """Copy a device tensor to host for test assertion only."""
    host = np.empty(tensor.shape, dtype=np.float32)
    err = cuda_runtime.cudaMemcpyAsync(
        host.ctypes.data,
        tensor.ptr,
        tensor.nbytes,
        cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        stream,
    )
    check_cuda(err, "D2H test copy")
    err = cuda_runtime.cudaStreamSynchronize(stream)
    check_cuda(err, "stream sync")
    return host


def test_device_tensor_spec():
    arena = BufferArena(device_id=0)
    try:
        t = arena.reserve((2, 3, 112, 112), ctypes.c_float, stream=0)
        assert t.shape == (2, 3, 112, 112)
        assert t.dtype is ctypes.c_float
        assert t.nbytes == 2 * 3 * 112 * 112 * 4
        assert t.ptr != 0
        assert t.owner is arena
    finally:
        arena.close()


def test_recognizer_device_infer_parity(recognition_engine):
    engine = recognition_engine
    batch = 2
    shape = (batch, 3, 112, 112)
    np_input = np.zeros(shape, dtype=np.float32)

    err, stream = cuda_runtime.cudaStreamCreate()
    check_cuda(err, "test stream create")
    stream_handle = int(stream)
    try:
        d_input = engine._arena.reserve(shape, ctypes.c_float, stream=stream_handle)
        err = cuda_runtime.cudaMemcpyAsync(
            d_input.ptr,
            np_input.ctypes.data,
            np_input.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream_handle,
        )
        check_cuda(err, "H2D test copy")

        outputs = engine.infer_device({"input.1": d_input}, stream=stream_handle)
        out_name = next(iter(outputs))
        out_tensor = outputs[out_name]
        assert out_tensor.device_id == 0
        assert out_tensor.owner is engine._arena

        host_out = _device_to_numpy(out_tensor, stream_handle)
        assert host_out.shape == (batch, 512)

        # Reference CPU inference to verify parity of raw unnormalized outputs.
        from app.ml.trt_engine import TrtEngine
        ref = TrtEngine(settings.embedder_engine_path)
        try:
            ref_outputs = ref.infer({"input.1": np_input})
            ref_out_name = next(iter(ref_outputs))
            ref_out = ref_outputs[ref_out_name]
            assert np.isfinite(host_out).all()
            assert np.isfinite(ref_out).all()
            max_abs_err = float(np.max(np.abs(host_out - ref_out)))
            # Tolerate FP16/FP32 execution-order differences.
            assert max_abs_err < 1e-3, f"max_abs_err={max_abs_err}"
        finally:
            ref.close()
    finally:
        cuda_runtime.cudaStreamDestroy(stream_handle)


def test_recognizer_device_infer_reuses_arena(recognition_engine):
    engine = recognition_engine
    shape = (1, 3, 112, 112)
    np_input = np.zeros(shape, dtype=np.float32)

    err, stream = cuda_runtime.cudaStreamCreate()
    check_cuda(err, "test stream create")
    stream_handle = int(stream)
    try:
        d_input = engine._arena.reserve(shape, ctypes.c_float, stream=stream_handle)
        err = cuda_runtime.cudaMemcpyAsync(
            d_input.ptr,
            np_input.ctypes.data,
            np_input.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream_handle,
        )
        check_cuda(err, "H2D test copy")

        out1 = engine.infer_device({"input.1": d_input}, stream=stream_handle)
        out2 = engine.infer_device({"input.1": d_input}, stream=stream_handle)
        # Output should come from the same arena allocation for the same shape.
        name1 = next(iter(out1))
        name2 = next(iter(out2))
        assert out1[name1].ptr == out2[name2].ptr
    finally:
        cuda_runtime.cudaStreamDestroy(stream_handle)
