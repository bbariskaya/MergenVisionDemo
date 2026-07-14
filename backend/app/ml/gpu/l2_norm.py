"""GPU L2 normalization wrapper for embedding vectors."""
from __future__ import annotations

import ctypes
import logging

import numpy as np
from cuda.bindings import runtime as cuda_runtime

from app.ml.gpu.buffer_arena import BufferArena
from app.ml.gpu.device_tensor import DeviceTensor, check_cuda
from mergenvision_gpu import l2_normalize as _l2_normalize

logger = logging.getLogger(__name__)


def l2_normalize_device(
    input_tensor: DeviceTensor,
    *,
    arena: BufferArena,
    stream: int | None = None,
    epsilon: float = 1e-12,
    output: DeviceTensor | None = None,
    status: DeviceTensor | None = None,
) -> DeviceTensor:
    """L2-normalize ``input_tensor``.

    Parameters
    ----------
    input_tensor:
        [rows, cols] float32 device tensor.
    arena:
        Buffer arena used for the output buffer and status scratch (unless
        both are supplied by the caller).
    stream:
        CUDA stream handle (0 for the default stream).
    epsilon:
        Small constant added to the squared norm before sqrt for stability.
    output:
        Optional output buffer. If provided, normalization is done in-place
        when ``output`` aliases ``input_tensor``.
    status:
        Optional ``[1]`` int32 device tensor for deferred error checking.

    Returns
    -------
    A [rows, cols] float32 device tensor with unit row norms.
    """
    if input_tensor.dtype is not ctypes.c_float:
        raise TypeError(f"l2_normalize expects float32, got {input_tensor.dtype}")
    if len(input_tensor.shape) != 2:
        raise ValueError(f"l2_normalize expects [rows, cols], got {input_tensor.shape}")

    active_stream = stream if stream is not None else 0
    rows, cols = input_tensor.shape

    if output is None:
        output = arena.reserve((rows, cols), ctypes.c_float, stream=active_stream)
    else:
        if output.shape != (rows, cols) or output.dtype is not ctypes.c_float:
            raise ValueError(
                f"output shape/dtype mismatch: {output.shape}/{output.dtype}"
            )

    owns_status = False
    if status is None:
        status = arena.reserve((1,), ctypes.c_int32, stream=active_stream)
        owns_status = True
    else:
        if status.shape != (1,) or status.dtype is not ctypes.c_int32:
            raise ValueError(
                f"status must be [1] int32, got {status.shape} {status.dtype}"
            )

    if owns_status:
        err = cuda_runtime.cudaMemsetAsync(status.ptr, 0, 4, active_stream)
        check_cuda(err, "l2_normalize status memset")

    _l2_normalize(
        input_tensor.ptr,
        output.ptr,
        rows,
        cols,
        epsilon,
        status.ptr,
        active_stream,
    )

    if owns_status:
        status_host = np.empty(1, dtype=np.int32)
        err = cuda_runtime.cudaMemcpyAsync(
            status_host.ctypes.data,
            status.ptr,
            4,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            active_stream,
        )
        check_cuda(err, "l2_normalize status D2H")
        err = cuda_runtime.cudaStreamSynchronize(active_stream)
        check_cuda(err, "l2_normalize status sync")

        if status_host[0] & 1:
            raise ValueError("l2_normalize input contains non-finite values")
        if status_host[0] & 2:
            raise ValueError("l2_normalize encountered a zero-norm row")

    return output
