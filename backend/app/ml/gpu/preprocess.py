"""CV-CUDA GPU preprocessing for detector input.

encoded JPEG bytes -> decode -> resize -> pad -> normalize -> NCHW
All operations stay on GPU; no PIL/OpenCV/NumPy image processing.
"""
from __future__ import annotations

import ctypes
import logging

import cvcuda
from cuda.bindings import runtime as cuda_runtime

from app.ml.gpu.buffer_arena import BufferArena
from app.ml.gpu.device_tensor import DeviceTensor, check_cuda

logger = logging.getLogger(__name__)

MEAN = 127.5
STD = 128.0


class GpuDetectorPreprocessor:
    """Preprocess decoded JPEG into detector NCHW input."""

    def __init__(self, input_size: int, device_id: int = 0) -> None:
        self._input_size = int(input_size)
        self._device_id = int(device_id)
        self._arena = BufferArena(device_id=device_id)

    @staticmethod
    def _scale_for_size(h: int, w: int, target: int) -> float:
        return target / max(h, w)

    def preprocess(
        self,
        decoded: DeviceTensor,
        *,
        stream: int | None = None,
    ) -> DeviceTensor:
        if decoded.dtype is not ctypes.c_uint8:
            raise TypeError(
                f"GpuDetectorPreprocessor expects uint8 input, got {decoded.dtype}"
            )
        if len(decoded.shape) != 4:
            raise ValueError(f"Expected NHWC decoded tensor, got {decoded.shape}")
        _, h, w, c = decoded.shape
        if c != 3:
            raise ValueError(f"Expected 3 channels, got {c}")

        active_stream = stream if stream is not None else 0
        cvcuda_stream = cvcuda.as_stream(active_stream)

        # Wrap the decoder-owned GPU buffer with CV-CUDA and add a batch dim.
        # The decoder Image object is kept alive through ``decoded.owner`` and
        # the intermediate list below.
        hwc = cvcuda.as_tensor(decoded.owner, cvcuda.TensorLayout.HWC)
        own_nhwc = cvcuda.stack([hwc], stream=cvcuda_stream)

        scale = self._scale_for_size(h, w, self._input_size)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        if new_h == 0:
            new_h = 1
        if new_w == 0:
            new_w = 1

        resized = cvcuda.resize(
            own_nhwc,
            (1, new_h, new_w, 3),
            cvcuda.Interp.LINEAR,
            stream=cvcuda_stream,
        )

        pad_bottom = self._input_size - new_h
        pad_right = self._input_size - new_w
        padded = cvcuda.copymakeborder(
            resized,
            cvcuda.Border.CONSTANT,
            [0.0, 0.0, 0.0],
            top=0,
            left=0,
            bottom=pad_bottom,
            right=pad_right,
            stream=cvcuda_stream,
        )

        normalized = cvcuda.convertto(
            padded,
            cvcuda.Type.F32,
            scale=1.0 / STD,
            offset=-MEAN / STD,
            stream=cvcuda_stream,
        )

        nchw = cvcuda.reformat(
            normalized,
            cvcuda.TensorLayout.NCHW,
            stream=cvcuda_stream,
        )

        # CV-CUDA may return pitched/strided memory. TensorRT expects
        # row-major contiguous NCHW, so copy to a contiguous arena buffer.
        out = self._make_contiguous_nchw(nchw, active_stream)

        # Keep every CV-CUDA intermediate (and the underlying decoder image)
        # alive while the async graph runs.
        intermediates = [decoded.owner, hwc, own_nhwc, resized, padded, normalized, nchw, out]
        return DeviceTensor(
            ptr=out.ptr,
            shape=out.shape,
            dtype=ctypes.c_float,
            device_id=self._device_id,
            owner=intermediates,
            stream=active_stream,
        )

    def _make_contiguous_nchw(
        self,
        tensor: cvcuda.Tensor,
        stream: int,
    ) -> DeviceTensor:
        """Copy a possibly pitched CV-CUDA NCHW tensor into a contiguous buffer."""
        cai = tensor.cuda().__cuda_array_interface__
        shape = tuple(cai["shape"])
        strides = tuple(cai["strides"])
        src_ptr = int(cai["data"][0])

        if strides == DeviceTensor._c_contiguous_strides(shape, ctypes.c_float):
            return DeviceTensor(
                ptr=src_ptr,
                shape=shape,
                dtype=ctypes.c_float,
                device_id=self._device_id,
                owner=tensor,
                stream=stream,
            )

        itemsize = 4
        n, c, h, w = shape
        dst = self._arena.reserve(shape, ctypes.c_float, stream=stream)
        row_pitch_src = strides[-2]
        row_pitch_dst = w * itemsize
        for ni in range(n):
            for ci in range(c):
                src_plane = src_ptr + ni * strides[0] + ci * strides[1]
                dst_plane = dst.ptr + (
                    (ni * c + ci) * h * w * itemsize
                )
                err = cuda_runtime.cudaMemcpy2DAsync(
                    dst_plane,
                    row_pitch_dst,
                    src_plane,
                    row_pitch_src,
                    w * itemsize,
                    h,
                    cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    stream,
                )
                check_cuda(err, "contiguous NCHW D2D")
        return dst

    def close(self) -> None:
        self._arena.close()
