from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cuda_runtime

logger = logging.getLogger(__name__)


class TrtEngine:
    def __init__(self, engine_path: Path | str) -> None:
        self.engine_path = Path(engine_path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._deserialize()
        self.context = self.engine.create_execution_context()
        err, self.stream = cuda_runtime.cudaStreamCreate()
        if err != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamCreate failed: {err}")
        self._buffers: dict[str, Any] = {}

    def _deserialize(self) -> trt.ICudaEngine:
        if not self.engine_path.exists():
            raise FileNotFoundError(self.engine_path)
        data = self.engine_path.read_bytes()
        engine = self.runtime.deserialize_cuda_engine(data)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize {self.engine_path}")
        return engine

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Set dynamic input shapes and upload inputs
        device_inputs: dict[str, int] = {}
        for name, arr in inputs.items():
            shape = tuple(arr.shape)
            self.context.set_input_shape(name, shape)
            nbytes = arr.nbytes
            if name not in self._buffers or self._buffers[name]["nbytes"] < nbytes:
                if name in self._buffers:
                    cuda_runtime.cudaFree(self._buffers[name]["ptr"])
                err, d_ptr = cuda_runtime.cudaMalloc(nbytes)
                if err != cuda_runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaMalloc failed for {name}: {err}")
                self._buffers[name] = {"ptr": int(d_ptr), "nbytes": nbytes}
            d_ptr = self._buffers[name]["ptr"]
            err = cuda_runtime.cudaMemcpyAsync(
                d_ptr,
                arr.ctypes.data,
                nbytes,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )
            if err[0] != cuda_runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpyAsync H2D failed for {name}: {err}")
            self.context.set_tensor_address(name, d_ptr)
            device_inputs[name] = d_ptr

        # Allocate outputs
        outputs: dict[str, np.ndarray] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            arr = np.empty(shape, dtype=dtype)
            nbytes = arr.nbytes
            key = f"__out__:{name}"
            if key not in self._buffers or self._buffers[key]["nbytes"] < nbytes:
                if key in self._buffers:
                    cuda_runtime.cudaFree(self._buffers[key]["ptr"])
                err, d_ptr = cuda_runtime.cudaMalloc(nbytes)
                if err != cuda_runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaMalloc failed for {name}: {err}")
                self._buffers[key] = {"ptr": int(d_ptr), "nbytes": nbytes}
            d_ptr = self._buffers[key]["ptr"]
            self.context.set_tensor_address(name, d_ptr)
            outputs[name] = arr

        success = self.context.execute_async_v3(self.stream)
        if not success:
            raise RuntimeError("execute_async_v3 failed")

        err = cuda_runtime.cudaStreamSynchronize(self.stream)
        if err[0] != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamSynchronize failed: {err}")

        for name, arr in outputs.items():
            key = f"__out__:{name}"
            d_ptr = self._buffers[key]["ptr"]
            err = cuda_runtime.cudaMemcpyAsync(
                arr.ctypes.data,
                d_ptr,
                arr.nbytes,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
            if err[0] != cuda_runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpyAsync D2H failed for {name}: {err}")
        err = cuda_runtime.cudaStreamSynchronize(self.stream)
        if err[0] != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamSynchronize failed: {err}")
        return outputs

    def warmup(self) -> None:
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                if any(s <= 0 for s in shape):
                    shape = tuple(
                        max(1, s) if s > 0 else 1 for s in shape
                    )
                arr = np.zeros(shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
                self.infer({name: arr})
                break

    def __del__(self) -> None:
        try:
            for key in list(getattr(self, "_buffers", {}).keys()):
                cuda_runtime.cudaFree(self._buffers[key]["ptr"])
            if hasattr(self, "stream"):
                cuda_runtime.cudaStreamDestroy(self.stream)
        except Exception:
            pass
