"""RetinaFace R50 GPU preprocessing.

Decoded RGB image -> 640x640 resize -> RGB->BGR swap + mean subtraction
-> NCHW float32.  All work stays on the GPU.
"""
from __future__ import annotations

import ctypes
import logging
from typing import Any

import cvcuda
import numpy as np
from cuda.bindings import runtime as cuda_runtime

from app.ml.gpu.buffer_arena import BufferArena
from app.ml.gpu.device_tensor import DeviceTensor, check_cuda

logger = logging.getLogger(__name__)


class _CudaArrayInterface:
    """Lightweight wrapper that exposes a device pointer to CV-CUDA."""

    def __init__(
        self,
        ptr: int,
        shape: tuple[int, ...],
        dtype: type,
        strides: tuple[int, ...] | None = None,
    ) -> None:
        self.__cuda_array_interface__: dict[str, Any] = {
            "shape": tuple(shape),
            "typestr": np.dtype(dtype).str,
            "data": (int(ptr), False),
            "version": 2,
            "strides": strides,
        }


MEANS_BGR = np.array(
    [
        [0.0, 0.0, 1.0, -104.0],  # B = R
        [0.0, 1.0, 0.0, -117.0],  # G = G
        [1.0, 0.0, 0.0, -123.0],  # R = B
    ],
    dtype=np.float32,
)


class RetinaFacePreprocessor:
    """Build a BGR mean-subtracted NCHW TensorRT input for RetinaFace R50."""

    def __init__(self, input_size: int = 640, device_id: int = 0) -> None:
        self._input_size = int(input_size)
        self._device_id = int(device_id)
        self._arena = BufferArena(device_id=device_id)
        self._twist = self._build_twist()

    def _build_twist(self) -> cvcuda.Tensor:
        """Upload the 3x4 color-twist matrix that does RGB->BGR + mean subtract."""
        buf = self._arena.reserve(MEANS_BGR.shape, ctypes.c_float, stream=0)
        err = cuda_runtime.cudaMemcpyAsync(
            buf.ptr,
            MEANS_BGR.ctypes.data,
            MEANS_BGR.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
            0,
        )
        check_cuda(err, "retinaface twist H2D")
        wrapper = _CudaArrayInterface(buf.ptr, MEANS_BGR.shape, ctypes.c_float)
        return cvcuda.as_tensor(wrapper)

    def preprocess_batch(
        self,
        decoded: list[DeviceTensor],
        *,
        stream: int | None = None,
    ) -> DeviceTensor:
        """Batch squish-resize an entire chunk for RetinaFace R50.

        Returns an ``[N, 3, input_size, input_size]`` contiguous float32 tensor.
        """
        self._set_device()
        active_stream = stream if stream is not None else 0
        cvcuda_stream = cvcuda.as_stream(active_stream)
        n = len(decoded)
        if n == 0:
            raise ValueError("preprocess_batch called with empty decoded list")

        batch = cvcuda.ImageBatchVarShape(n)
        images: list[Any] = []
        tensors: list[Any] = []
        for dt in decoded:
            if dt.dtype is not ctypes.c_uint8:
                raise TypeError(f"expected uint8 decoded image, got {dt.dtype}")
            t = cvcuda.as_tensor(dt.owner, cvcuda.TensorLayout.HWC)
            images.append(cvcuda.as_image(t.cuda()))
            batch.pushback(images[-1])
            tensors.append(t)

        sizes = [(self._input_size, self._input_size)] * n
        resized = cvcuda.resize(
            batch, sizes, cvcuda.Interp.LINEAR, stream=cvcuda_stream
        )

        plane_size = self._input_size * self._input_size * 3
        tmp_nhwc = self._arena.reserve(
            (n, self._input_size, self._input_size, 3),
            ctypes.c_uint8,
            stream=active_stream,
        )
        resized_images: list[Any] = []
        for i, img in enumerate(resized):
            cai = img.cuda().__cuda_array_interface__
            src_ptr = int(cai["data"][0])
            err = cuda_runtime.cudaMemcpyAsync(
                tmp_nhwc.ptr + i * plane_size,
                src_ptr,
                plane_size,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                active_stream,
            )
            check_cuda(err, "preprocess_batch resize D2D")
            resized_images.append(img)

        tmp_wrapper = _CudaArrayInterface(
            tmp_nhwc.ptr, tmp_nhwc.shape, ctypes.c_uint8
        )
        uint8_nhwc = cvcuda.as_tensor(tmp_wrapper, cvcuda.TensorLayout.NHWC)
        f32 = cvcuda.convertto(
            uint8_nhwc, cvcuda.Type.F32, scale=1.0, offset=0.0, stream=cvcuda_stream
        )
        twisted = cvcuda.color_twist(f32, self._twist, stream=cvcuda_stream)
        nchw = cvcuda.reformat(twisted, cvcuda.TensorLayout.NCHW, stream=cvcuda_stream)
        nchw_cai = nchw.cuda().__cuda_array_interface__
        out_ptr = int(nchw_cai["data"][0])
        out_shape = tuple(nchw_cai["shape"])

        return DeviceTensor(
            ptr=out_ptr,
            shape=out_shape,
            dtype=ctypes.c_float,
            device_id=self._device_id,
            owner=[
                decoded,
                batch,
                images,
                tensors,
                resized,
                resized_images,
                tmp_nhwc,
                uint8_nhwc,
                f32,
                twisted,
                nchw,
                self._twist,
            ],
            stream=active_stream,
        )

    def _set_device(self) -> None:
        err = cuda_runtime.cudaSetDevice(self._device_id)
        check_cuda(err, f"retinaface preprocessor cudaSetDevice({self._device_id})")

    def preprocess(
        self,
        decoded: DeviceTensor,
        *,
        stream: int | None = None,
    ) -> DeviceTensor:
        self._set_device()
        if decoded.dtype is not ctypes.c_uint8:
            raise TypeError(
                f"RetinaFacePreprocessor expects uint8 input, got {decoded.dtype}"
            )
        if len(decoded.shape) != 4:
            raise ValueError(f"Expected NHWC decoded tensor, got {decoded.shape}")
        _, h, w, c = decoded.shape
        if c != 3:
            raise ValueError(f"Expected 3 channels, got {c}")

        active_stream = stream if stream is not None else 0
        cvcuda_stream = cvcuda.as_stream(active_stream)

        # Add explicit batch dimension: decoder returns (1, H, W, 3).
        hwc = cvcuda.as_tensor(decoded.owner, cvcuda.TensorLayout.HWC)
        nhwc = cvcuda.stack([hwc], stream=cvcuda_stream)

        # RetinaFace expects the input to be exactly 640x640 (squish, no padding).
        resized = cvcuda.resize(
            nhwc,
            (1, self._input_size, self._input_size, 3),
            cvcuda.Interp.LINEAR,
            stream=cvcuda_stream,
        )

        # Convert to float32, then color-twist for RGB->BGR + mean subtraction.
        float_nhwc = cvcuda.convertto(
            resized,
            cvcuda.Type.F32,
            scale=1.0,
            offset=0.0,
            stream=cvcuda_stream,
        )
        twisted = cvcuda.color_twist(
            float_nhwc,
            self._twist,
            stream=cvcuda_stream,
        )

        nchw = cvcuda.reformat(
            twisted,
            cvcuda.TensorLayout.NCHW,
            stream=cvcuda_stream,
        )

        out = self._make_contiguous_nchw(nchw, active_stream)
        intermediates = [
            decoded.owner,
            hwc,
            nhwc,
            resized,
            float_nhwc,
            twisted,
            nchw,
            out,
        ]
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
        cai = tensor.cuda().__cuda_array_interface__
        shape = tuple(cai["shape"])
        strides = tuple(cai["strides"]) if cai["strides"] else None
        src_ptr = int(cai["data"][0])

        if strides is None:
            return DeviceTensor(
                ptr=src_ptr,
                shape=shape,
                dtype=ctypes.c_float,
                device_id=self._device_id,
                owner=tensor,
                stream=stream,
            )

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
                check_cuda(err, "retinaface contiguous NCHW D2D")
        return dst

    def close(self) -> None:
        self._arena.close()
