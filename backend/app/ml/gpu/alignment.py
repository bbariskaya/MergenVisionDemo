"""GPU five-point face alignment using native similarity transform + warp."""
from __future__ import annotations

import ctypes
import logging

import numpy as np
from cuda.bindings import runtime as cuda_runtime

from app.ml.gpu.buffer_arena import BufferArena
from app.ml.gpu.device_tensor import DeviceTensor, check_cuda
from mergenvision_gpu import similarity_transform, warp_align

logger = logging.getLogger(__name__)


class GpuFaceAligner:
    """Compute ArcFace 112x112 alignment matrices and warp chips on GPU."""

    OUTPUT_SIZE = 112

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = int(device_id)
        self._arena = BufferArena(device_id=device_id)

    def _set_device(self) -> None:
        err = cuda_runtime.cudaSetDevice(self._device_id)
        check_cuda(err, f"aligner cudaSetDevice({self._device_id})")

    def compute_matrices(
        self,
        landmarks: DeviceTensor,
        *,
        stream: int | None = None,
        status: DeviceTensor | None = None,
    ) -> DeviceTensor:
        """Return [N, 2, 3] affine matrices from source landmarks.

        If ``status`` is provided it must be a ``[1]`` int32 device tensor.
        The caller is then responsible for reading it after the next sync.
        Otherwise the status is read back and validated immediately.
        """
        self._set_device()
        if landmarks.dtype is not ctypes.c_float:
            raise TypeError(f"landmarks must be float32, got {landmarks.dtype}")
        if len(landmarks.shape) != 2 or landmarks.shape[1] != 10:
            raise ValueError(f"landmarks must be [N, 10], got {landmarks.shape}")

        active_stream = stream if stream is not None else 0
        n = landmarks.shape[0]
        matrices = self._arena.reserve(
            (n, 6), ctypes.c_float, stream=active_stream
        )

        owns_status = False
        if status is None:
            status = self._arena.reserve(
                (1,), ctypes.c_int32, stream=active_stream
            )
            owns_status = True
        else:
            if status.shape != (1,) or status.dtype is not ctypes.c_int32:
                raise ValueError(
                    f"status must be [1] int32, got {status.shape} {status.dtype}"
                )

        err = cuda_runtime.cudaMemsetAsync(status.ptr, 0, 4, active_stream)
        check_cuda(err, "status memset")

        similarity_transform(
            landmarks.ptr,
            matrices.ptr,
            n,
            self.OUTPUT_SIZE,
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
            check_cuda(err, "status D2H")
            err = cuda_runtime.cudaStreamSynchronize(active_stream)
            check_cuda(err, "status sync")
            if status_host[0] != 0:
                raise ValueError(
                    f"similarity_transform failed (status={status_host[0]}); "
                    "non-finite or degenerate landmarks"
                )

        return DeviceTensor(
            ptr=matrices.ptr,
            shape=(n, 2, 3),
            dtype=ctypes.c_float,
            device_id=self._device_id,
            owner=matrices,
            stream=active_stream,
        )

    def align(
        self,
        image: DeviceTensor,
        landmarks: DeviceTensor,
        *,
        stream: int | None = None,
        status: DeviceTensor | None = None,
    ) -> DeviceTensor:
        """Warp N 112x112 face chips from a single source image.

        Parameters
        ----------
        image:
            [1, H, W, 3] uint8 source image in NHWC layout.
        landmarks:
            [N, 10] source landmarks in the **source image** coordinate space.
        status:
            Optional ``[1]`` int32 device tensor for deferred error checking.
        """
        self._set_device()
        if image.dtype is not ctypes.c_uint8:
            raise TypeError(f"image must be uint8, got {image.dtype}")
        if len(image.shape) != 4 or image.shape[0] != 1 or image.shape[3] != 3:
            raise ValueError(f"image must be [1, H, W, 3], got {image.shape}")
        if landmarks.dtype is not ctypes.c_float:
            raise TypeError(f"landmarks must be float32, got {landmarks.dtype}")

        active_stream = stream if stream is not None else 0
        _, h, w, _ = image.shape
        n = landmarks.shape[0]

        matrices = self.compute_matrices(landmarks, stream=active_stream, status=status)
        output = self._arena.reserve(
            (n, 3, self.OUTPUT_SIZE, self.OUTPUT_SIZE),
            ctypes.c_float,
            stream=active_stream,
        )

        warp_align(
            image.ptr,
            h,
            w,
            matrices.ptr,
            n,
            output.ptr,
            active_stream,
        )

        return DeviceTensor(
            ptr=output.ptr,
            shape=(n, 3, self.OUTPUT_SIZE, self.OUTPUT_SIZE),
            dtype=ctypes.c_float,
            device_id=self._device_id,
            owner=output,
            stream=active_stream,
        )

    def close(self) -> None:
        self._arena.close()
