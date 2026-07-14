import subprocess
import sys

import pytest


_SCRIPT = """
import ctypes
from cuda.bindings import runtime as cuda_runtime
from app.ml.gpu.buffer_arena import BufferArena

code = cuda_runtime.cudaSetDevice(0)
if isinstance(code, tuple):
    code = code[0]
assert code == cuda_runtime.cudaError_t.cudaSuccess, "failed to set device"

arena = BufferArena(device_id=0)
for _ in range(5):
    tensor = arena.reserve((1024,), ctypes.c_float, stream=0)
    assert tensor.ptr != 0
    del tensor

lease = arena.acquire((512,), ctypes.c_uint8, stream=0)
tensor = lease.as_tensor(stream=0)
del tensor
lease.release(0)

arena.close()
print("ok")
"""


@pytest.mark.parametrize("iteration", range(20))
def test_arena_subprocess_lifecycle(iteration: int) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "ok", result.stderr
