"""Reusable device buffer arena to avoid per-request cudaMalloc."""
from __future__ import annotations

import ctypes
import logging
import weakref
from collections import defaultdict
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
    """Device buffer pool keyed by (shape, dtype) with weak-ref reclamation.

    Each ``reserve`` returns a buffer that is guaranteed not to alias any other
    currently-live ``DeviceTensor``.  When a returned tensor is garbage
    collected its buffer goes back to the free list for reuse.
    """

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = int(device_id)
        self._free: dict[_BufferKey, list[int]] = defaultdict(list)
        self._allocated: set[int] = set()
        self._closed = False

    def device_id(self) -> int:
        return self._device_id

    def _itemsize(self, dtype: type) -> int:
        return {
            ctypes.c_uint8: 1,
            ctypes.c_int8: 1,
            ctypes.c_uint16: 2,
            ctypes.c_int16: 2,
            ctypes.c_float: 4,
            ctypes.c_int32: 4,
            ctypes.c_int64: 8,
        }.get(dtype, ctypes.sizeof(dtype))

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
        itemsize = self._itemsize(dtype)
        nbytes = itemsize * int(__import__("functools").reduce(int.__mul__, shape, 1))
        # Zero-byte allocations return NULL, which DeviceTensor rejects.
        alloc_nbytes = nbytes if nbytes > 0 else 1

        ptr: int | None = None
        free_list = self._free[key]
        while free_list:
            candidate = free_list.pop()
            if candidate in self._allocated:
                ptr = candidate
                break

        if ptr is None:
            err, raw_ptr = cuda_runtime.cudaMalloc(alloc_nbytes)
            check_cuda(err, f"BufferArena cudaMalloc({alloc_nbytes})")
            ptr = int(raw_ptr)
            self._allocated.add(ptr)
            logger.debug("Arena allocated shape=%s dtype=%s nbytes=%d", shape, dtype, nbytes)
        else:
            logger.debug("Arena reused shape=%s dtype=%s nbytes=%d", shape, dtype, nbytes)

        tensor = DeviceTensor(ptr, shape, dtype, self._device_id, self, stream=stream)

        def release(t_ptr: int = ptr, t_key: _BufferKey = key) -> None:
            if self._closed:
                try:
                    cuda_runtime.cudaFree(t_ptr)
                except Exception:
                    pass
            else:
                self._free[t_key].append(t_ptr)

        weakref.finalize(tensor, release)
        return tensor

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for ptr in list(self._allocated):
            try:
                check_cuda(cuda_runtime.cudaFree(ptr), f"BufferArena cudaFree {ptr}")
            except Exception as exc:
                logger.warning("BufferArena cudaFree failed for %s: %s", ptr, exc)
        self._allocated.clear()
        self._free.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
