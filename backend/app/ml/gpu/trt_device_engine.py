"""TensorRT execution context with direct device-pointer binding."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import tensorrt as trt
from cuda.bindings import runtime as cuda_runtime

from app.ml.gpu.buffer_arena import BufferArena
from app.ml.gpu.device_tensor import DeviceTensor, check_cuda

logger = logging.getLogger(__name__)


class TrtDeviceEngine:
    """Device-only TensorRT inference wrapper.

    - `infer()` remains host-in/host-out for the CPU reference oracle.
    - `infer_device()` accepts and returns `DeviceTensor`s without H2D/D2H.
    """

    def __init__(self, engine_path: Path | str, device_id: int = 0) -> None:
        self.engine_path = Path(engine_path)
        self._device_id = int(device_id)
        err = cuda_runtime.cudaSetDevice(device_id)
        check_cuda(err, f"trt engine init cudaSetDevice({device_id})")
        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._trt_logger)
        self._engine = self._deserialize()
        self._context = self._engine.create_execution_context()
        err, self._stream = cuda_runtime.cudaStreamCreate()
        check_cuda(err, "cudaStreamCreate")
        self._arena = BufferArena(device_id=device_id)
        self._output_buffers: dict[str, DeviceTensor] = {}
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self._output_names.append(name)
        self._lock = threading.Lock()
        self._closed = False

    @property
    def engine(self) -> trt.ICudaEngine:
        return self._engine

    def _deserialize(self) -> trt.ICudaEngine:
        if not self.engine_path.exists():
            raise FileNotFoundError(self.engine_path)
        data = self.engine_path.read_bytes()
        engine = self._runtime.deserialize_cuda_engine(data)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize {self.engine_path}")
        logger.info("Deserialized TensorRT device engine: %s", self.engine_path)
        return engine

    def _ensure_output_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        ctype: type,
        stream: int,
    ) -> DeviceTensor:
        """Return a buffer large enough for ``shape``, growing it if needed.

        Buffers are keyed by output name and reused across inferences. This
        avoids a `cudaMalloc` for every call once the typical shapes are seen.
        """
        buf = self._output_buffers.get(name)
        needed = int(__import__("functools").reduce(int.__mul__, shape, 1))
        if buf is not None and buf.dtype == ctype:
            capacity = int(
                __import__("functools").reduce(int.__mul__, buf.shape, 1)
            )
            if capacity >= needed:
                return buf

        # Grow capacity to the current requirement; first call allocates.
        buf = self._arena.reserve(shape, ctype, stream=stream)
        self._output_buffers[name] = buf
        logger.debug(
            "Allocated/reallocated output buffer %s shape=%s ctype=%s",
            name, shape, ctype.__name__
        )
        return buf

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("TrtDeviceEngine is closed")

    def _set_device(self) -> None:
        err = cuda_runtime.cudaSetDevice(self._device_id)
        check_cuda(err, f"trt engine cudaSetDevice({self._device_id})")

    def infer_device(
        self,
        inputs: dict[str, DeviceTensor],
        *,
        stream: int | None = None,
    ) -> dict[str, DeviceTensor]:
        """Execute with device-resident inputs and outputs.

        No host upload/download is performed inside this method.
        """
        with self._lock:
            self._check_open()
            self._set_device()
            provided = set(inputs.keys())
            expected = set(self._input_names)
            if provided != expected:
                missing = expected - provided
                extra = provided - expected
                raise ValueError(
                    f"Input name mismatch. Missing: {sorted(missing)}, "
                    f"Extra: {sorted(extra)}"
                )

            active_stream = stream if stream is not None else int(self._stream)

            for name, tensor in inputs.items():
                if tensor.device_id != self._device_id:
                    raise ValueError(
                        f"Input '{name}' is on device {tensor.device_id}, "
                        f"expected {self._device_id}"
                    )
                shape = tuple(tensor.shape)
                if not self._context.set_input_shape(name, shape):
                    raise RuntimeError(
                        f"set_input_shape failed for '{name}' with shape {shape}"
                    )
                if not self._context.set_tensor_address(name, tensor.ptr):
                    raise RuntimeError(
                        f"set_tensor_address failed for input '{name}'"
                    )

            outputs: dict[str, DeviceTensor] = {}
            for name in self._output_names:
                shape = tuple(self._context.get_tensor_shape(name))
                if any(s <= 0 for s in shape):
                    raise RuntimeError(
                        f"Output '{name}' has invalid shape {shape}; "
                        "input shapes may not be set"
                    )
                dtype = self._engine.get_tensor_dtype(name)
                ctype = self._trt_dtype_to_ctype(dtype)
                buf = self._ensure_output_buffer(name, shape, ctype, active_stream)
                tensor = DeviceTensor(
                    ptr=buf.ptr,
                    shape=shape,
                    dtype=ctype,
                    device_id=self._device_id,
                    owner=self._arena,
                    stream=active_stream,
                )
                if not self._context.set_tensor_address(name, tensor.ptr):
                    raise RuntimeError(
                        f"set_tensor_address failed for output '{name}'"
                    )
                outputs[name] = tensor

            if not self._context.execute_async_v3(active_stream):
                raise RuntimeError("execute_async_v3 failed")

            return outputs

    @staticmethod
    def _trt_dtype_to_ctype(dtype: trt.DataType) -> type:
        mapping = {
            trt.DataType.FLOAT: __import__("ctypes").c_float,
            trt.DataType.INT32: __import__("ctypes").c_int32,
            trt.DataType.INT8: __import__("ctypes").c_int8,
            trt.DataType.UINT8: __import__("ctypes").c_uint8,
            trt.DataType.HALF: __import__("ctypes").c_uint16,
            trt.DataType.BF16: __import__("ctypes").c_uint16,
        }
        ctype = mapping.get(dtype)
        if ctype is None:
            raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
        return ctype

    def input_profile(self, name: str, profile_index: int = 0) -> tuple[list[int], list[int], list[int]]:
        return self._engine.get_tensor_profile_shape(name, profile_index)

    def warmup(self, input_shapes: dict[str, tuple[int, ...]]) -> None:
        self._set_device()
        dummy: dict[str, DeviceTensor] = {}
        for name, shape in input_shapes.items():
            if name not in self._input_names:
                raise ValueError(f"Unknown input tensor '{name}'")
            dtype = self._engine.get_tensor_dtype(name)
            ctype = self._trt_dtype_to_ctype(dtype)
            tensor = self._arena.reserve(shape, ctype, stream=int(self._stream))
            dummy[name] = tensor
        self.infer_device(dummy)
        err = cuda_runtime.cudaStreamSynchronize(self._stream)
        check_cuda(err, "warmup cudaStreamSynchronize")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._output_buffers.clear()
            self._arena.close()
            try:
                check_cuda(
                    cuda_runtime.cudaStreamDestroy(self._stream),
                    "cudaStreamDestroy",
                )
            except Exception as exc:
                logger.warning("cudaStreamDestroy failed: %s", exc)
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
