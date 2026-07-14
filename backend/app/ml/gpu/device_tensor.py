"""Device tensor lifetime/ownership contract for the GPU data plane."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

from cuda.bindings import runtime as cuda_runtime


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: type

    @property
    def size(self) -> int:
        return int(__import__("functools").reduce(int.__mul__, self.shape, 1))


class DeviceTensor:
    """Immutable device pointer wrapper with explicit owner lifetime.

    Rules:
    - The owner object keeps the allocation alive.
    - NumPy or host conversion is not offered in production code.
    - Stream ownership is explicit; no implicit synchronization.
    """

    _DTYPE_TO_ITEMSIZE: dict[type, int] = {
        ctypes.c_uint8: 1,
        ctypes.c_int8: 1,
        ctypes.c_uint16: 2,
        ctypes.c_int16: 2,
        ctypes.c_float: 4,
        ctypes.c_int32: 4,
        ctypes.c_int64: 8,
    }

    def __init__(
        self,
        ptr: int,
        shape: tuple[int, ...],
        dtype: type,
        device_id: int,
        owner: Any,
        stream: int | None = None,
        *,
        lease: Any = None,
    ) -> None:
        if ptr == 0:
            raise ValueError("DeviceTensor requires a non-null device pointer")
        self._ptr = int(ptr)
        self._shape = tuple(shape)
        self._dtype = dtype
        self._device_id = int(device_id)
        self._owner = owner
        self._lease = lease
        self._stream = stream
        itemsize = self._DTYPE_TO_ITEMSIZE.get(dtype)
        if itemsize is None:
            raise TypeError(f"Unsupported DeviceTensor dtype: {dtype}")
        self._itemsize = itemsize
        # Lease-backed tensors report the allocation size, which may exceed the
        # requested view shape (e.g., scratch buffers reused with smaller shapes).
        if self._lease is not None:
            self._nbytes = self._lease.ptr_nbytes
        else:
            self._nbytes = self._itemsize * int(
                __import__("functools").reduce(int.__mul__, self._shape, 1)
            )

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def owner(self) -> Any:
        return self._owner

    @property
    def stream(self) -> int | None:
        return self._stream

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def strides(self) -> tuple[int, ...]:
        # C-contiguous strides only.
        return self._c_contiguous_strides(self._shape, self._dtype)

    @staticmethod
    def _c_contiguous_strides(
        shape: tuple[int, ...], dtype: type
    ) -> tuple[int, ...]:
        itemsize: int = {
            ctypes.c_uint8: 1,
            ctypes.c_int8: 1,
            ctypes.c_uint16: 2,
            ctypes.c_int16: 2,
            ctypes.c_float: 4,
            ctypes.c_int32: 4,
            ctypes.c_int64: 8,
        }.get(dtype, ctypes.sizeof(dtype))
        strides = []
        prod = 1
        for dim in reversed(shape):
            strides.append(prod * itemsize)
            prod *= dim
        return tuple(reversed(strides))

    @property
    def layout(self) -> str:
        return "C"

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DeviceTensor(ptr={self._ptr}, shape={self._shape}, "
            f"dtype={self._dtype.__name__}, device={self._device_id}, "
            f"nbytes={self._nbytes})"
        )

    def bind_stream(self, stream: int) -> "DeviceTensor":
        """Return a new view tied to the provided stream; owner unchanged."""
        return DeviceTensor(
            self._ptr,
            self._shape,
            self._dtype,
            self._device_id,
            self._owner,
            stream=stream,
            lease=self._lease,
        )


_CUDA_CHECK_ERRORS = True


def check_cuda(err: cuda_runtime.cudaError_t | tuple, msg: str) -> None:
    # cuda.bindings may return the error code directly or as a tuple.
    if isinstance(err, tuple):
        err = err[0]
    name = cuda_runtime.cudaGetErrorName(err)
    if isinstance(name, tuple):
        name = name[1] if len(name) > 1 else name[0]
    text = cuda_runtime.cudaGetErrorString(err)
    if isinstance(text, tuple):
        text = text[1] if len(text) > 1 else text[0]
    if err != cuda_runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{msg}: {name} ({text})")
