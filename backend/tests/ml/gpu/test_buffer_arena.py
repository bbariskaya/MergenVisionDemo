import gc
from typing import Any

import pytest
from cuda.bindings import runtime as cuda_runtime
from mergenvision_gpu import spin_wait_cycles

from app.ml.gpu.buffer_arena import BufferArena, BufferLease


def _check(err: Any, msg: str) -> None:
    if isinstance(err, tuple):
        err = err[0]
    if err != cuda_runtime.cudaError_t.cudaSuccess:
        name = cuda_runtime.cudaGetErrorName(err)
        text = cuda_runtime.cudaGetErrorString(err)
        raise RuntimeError(f"{msg}: {name} - {text}")


@pytest.fixture
def stream():
    _check(cuda_runtime.cudaSetDevice(0), "set device")
    err, s = cuda_runtime.cudaStreamCreate()
    _check(err, "create stream")
    yield s
    _check(cuda_runtime.cudaStreamDestroy(s), "destroy stream")


@pytest.fixture
def arena():
    a = BufferArena(device_id=0)
    yield a
    a.close()


def spin(stream: Any, cycles: int = 200_000_000) -> None:
    spin_wait_cycles(cycles, int(stream))


def test_release_event_fences_reuse(arena, stream):
    type_byte = __import__("ctypes").c_uint8
    lease_a = arena.acquire((1024,), type_byte, stream=int(stream))
    ptr_a = lease_a.ptr
    spin(stream)
    lease_a.release(int(stream))

    # Lease_a is pending and must not be handed out again while its event is
    # incomplete.  acquire is non-blocking, so this assertion races the GPU;
    # use a long enough spin that Python overhead is negligible in comparison.
    assert ptr_a in arena.pending_ptrs()
    lease_b = arena.acquire((1024,), type_byte, stream=int(stream))
    assert lease_b.ptr != ptr_a
    assert arena.unique_allocation_count() == 2

    _check(cuda_runtime.cudaStreamSynchronize(stream), "sync")

    # After synchronization the same allocation can be reused.
    lease_c = arena.acquire((1024,), type_byte, stream=int(stream))
    assert lease_c.ptr == ptr_a
    lease_c.release(int(stream))
    _check(cuda_runtime.cudaStreamSynchronize(stream), "sync")


def test_release_while_active_view_raises(arena, stream):
    lease = arena.acquire((64,), __import__("ctypes").c_float, stream=int(stream))
    tensor = lease.as_tensor(stream=int(stream))
    assert tensor is not None
    with pytest.raises(RuntimeError, match="active DeviceTensor views"):
        lease.release(int(stream))


def test_reserve_tensor_release_on_gc_allows_reuse(arena, stream):
    tensor = arena.reserve((256,), __import__("ctypes").c_float, stream=int(stream))
    ptr = tensor.ptr
    del tensor
    gc.collect()
    _check(cuda_runtime.cudaStreamSynchronize(stream), "sync")
    assert arena.unique_allocation_count() == 1
    tensor2 = arena.reserve((256,), __import__("ctypes").c_float, stream=int(stream))
    assert tensor2.ptr == ptr
