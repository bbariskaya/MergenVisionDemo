"""Reusable device buffer arena to avoid per-request cudaMalloc."""
from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from typing import Any

from cuda.bindings import runtime as cuda_runtime

from app.ml.gpu.device_tensor import DeviceTensor, check_cuda

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BufferKey:
    shape: tuple[int, ...]
    dtype: type


class BufferArena:
    """Owns preallocated device buffers and reuses them by (shape, dtype).

    All allocations live until the arena is closed. No defragmentation is
    implemented; this is acceptable because the production pipeline uses a
    bounded set of shapes (detector input, recognizer chunks, scratch tensors).
    """

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = int(device_id)
        self._buffers: dict[_BufferKey, int] = {}
        self._closed = False

    def device_id(self) -> int:
        return self._device_id

    def reserve(
        self,
        shape: tuple[int, ...],
        dtype: type,
        *,
        stream: int | None = None,
    ) -> DeviceTensor:
        if self._closed:
            raise RuntimeError("BufferArena is closed")
        key = _BufferKey(shape=shape, dtype=dtype)
        itemsize: int = {
            ctypes.c_uint8: 1,
            ctypes.c_int8: 1,
            ctypes.c_uint16: 2,
            ctypes.c_int16: 2,
            ctypes.c_float: 4,
            ctypes.c_int32: 4,
            ctypes.c_int64: 8,
        }.get(dtype, ctypes.sizeof(dtype))
        nbytes = itemsize * int(__import__("functools").reduce(int.__mul__, shape, 1))
        # Zero-byte allocations return NULL, which DeviceTensor rejects.
        alloc_nbytes = nbytes if nbytes > 0 else 1

        ptr = self._buffers.get(key)
        if ptr is not None and ptr != 0:
            return DeviceTensor(ptr, shape, dtype, self._device_id, self, stream=stream)

        err, ptr = cuda_runtime.cudaMalloc(alloc_nbytes)
        check_cuda(err, f"BufferArena cudaMalloc({alloc_nbytes})")
        self._buffers[key] = int(ptr)
        logger.debug("Arena allocated shape=%s dtype=%s nbytes=%d", shape, dtype, nbytes)
        return DeviceTensor(int(ptr), shape, dtype, self._device_id, self, stream=stream)

    def close(self) -> None:
        if self._closed:
            return
        for key, ptr in list(self._buffers.items()):
            try:
                check_cuda(cuda_runtime.cudaFree(ptr), f"BufferArena cudaFree {key}")
            except Exception as exc:
                logger.warning("BufferArena cudaFree failed for %s: %s", key, exc)
        self._buffers.clear()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
