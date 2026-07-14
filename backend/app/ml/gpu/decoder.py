"""nvImageCodec JPEG GPU decoder with no silent CPU fallback."""
from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from typing import Any

import nvidia.nvimgcodec as nvimgcodec
from cuda.bindings import runtime as cuda_runtime

from app.ml.gpu.device_tensor import DeviceTensor, check_cuda

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodeInfo:
    width: int
    height: int
    channels: int
    backend_kind: str
    backend_name: str
    device_id: int


class JpegGpuDecoder:
    """Single-GPU JPEG decoder backed by nvImageCodec.

    The decoder is created once at startup and reused. Production mode never
    silently falls back to CPU; CPU backend raises an explicit error.
    """

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = int(device_id)
        # Empty backends list allows all backends; we enforce GPU-only by
        # inspecting the per-image decode backend info after decode.
        self._decoder = nvimgcodec.Decoder(
            device_id=self._device_id,
            backends=[],
            options=":fancy_upsampling=0",
        )
        self._closed = False

    def _tensor_from_image(self, image: Any, stream: int | None) -> DeviceTensor:
        cuda_view = image.cuda()
        cai = cuda_view.__cuda_array_interface__
        ptr = int(cai["data"][0])
        shape = tuple(cai["shape"])
        if len(shape) != 3:
            raise RuntimeError(f"Decoded image shape {shape} is not HWC")
        height, width, channels = shape
        if channels != 3:
            raise RuntimeError(f"Decoded image channels {channels} != 3")
        if ptr == 0:
            raise RuntimeError("Decoded image device pointer is null")
        return DeviceTensor(
            ptr=ptr,
            shape=(1, height, width, channels),
            dtype=ctypes.c_uint8,
            device_id=self._device_id,
            owner=image,
            stream=stream,
        )

    def decode(
        self,
        encoded: bytes,
        *,
        stream: int | None = None,
    ) -> tuple[DeviceTensor, DecodeInfo]:
        err = cuda_runtime.cudaSetDevice(self._device_id)
        check_cuda(err, f"jpeg decoder cudaSetDevice({self._device_id})")
        if self._closed:
            raise RuntimeError("JpegGpuDecoder is closed")
        if not encoded:
            raise ValueError("empty encoded JPEG buffer")

        image = self._decoder.decode(
            nvimgcodec.CodeStream(encoded),
            cuda_stream=stream or 0,
        )
        if image is None:
            raise RuntimeError("nvImageCodec decode returned None")

        info = self._decode_info(image)
        if info.backend_kind != "GPU_ONLY":
            raise RuntimeError(
                f"Unexpected decode backend {info.backend_kind}; "
                "CPU fallback is not allowed in production"
            )

        return self._tensor_from_image(image, stream), info

    def decode_batch(
        self,
        encoded_list: list[bytes],
        *,
        stream: int | None = None,
    ) -> tuple[list[DeviceTensor], list[DecodeInfo]]:
        """Decode a list of JPEG buffers in parallel on the GPU."""
        err = cuda_runtime.cudaSetDevice(self._device_id)
        check_cuda(err, f"jpeg decoder cudaSetDevice({self._device_id})")
        if self._closed:
            raise RuntimeError("JpegGpuDecoder is closed")
        if not encoded_list:
            return [], []

        code_streams = [nvimgcodec.CodeStream(b) for b in encoded_list]
        images = self._decoder.decode(code_streams, cuda_stream=stream or 0)
        if images is None:
            raise RuntimeError("nvImageCodec batch decode returned None")
        if not isinstance(images, (list, tuple)):
            images = [images]
        if len(images) != len(encoded_list):
            raise RuntimeError(
                f"nvImageCodec batch decode returned {len(images)} images for "
                f"{len(encoded_list)} inputs"
            )

        tensors: list[DeviceTensor] = []
        infos: list[DecodeInfo] = []
        for image in images:
            if image is None:
                raise RuntimeError("nvImageCodec batch decode returned None for an image")
            info = self._decode_info(image)
            if info.backend_kind != "GPU_ONLY":
                raise RuntimeError(
                    f"Unexpected decode backend {info.backend_kind}; "
                    "CPU fallback is not allowed in production"
                )
            tensors.append(self._tensor_from_image(image, stream))
            infos.append(info)
        return tensors, infos

    def _decode_info(self, image: Any) -> DecodeInfo:
        # buffer_kind is the official evidence of where the decoded buffer lives.
        buffer_kind = image.buffer_kind
        kind_name = (
            "GPU_ONLY"
            if buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE
            else str(buffer_kind)
        )
        cuda_view = image.cuda()
        cai = cuda_view.__cuda_array_interface__
        h, w, c = cai["shape"]
        return DecodeInfo(
            width=int(w),
            height=int(h),
            channels=int(c),
            backend_kind=kind_name,
            backend_name=str(buffer_kind),
            device_id=self._device_id,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            # Release the nvImageCodec decoder while the CUDA context is still
            # active. Letting it live until process teardown causes a fatal
            # crash when the destructor touches a destroyed context.
            err = cuda_runtime.cudaSetDevice(self._device_id)
            check_cuda(err, "decoder close set device")
            self._decoder = None
        except Exception:
            logger.exception("JpegGpuDecoder.close failed")
        finally:
            self._closed = True

    def __del__(self) -> None:
        pass
