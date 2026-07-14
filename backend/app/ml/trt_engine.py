from __future__ import annotations

import logging
import threading
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
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self._output_names.append(name)
        self._lock = threading.Lock()
        self._closed = False

    def _deserialize(self) -> trt.ICudaEngine:
        if not self.engine_path.exists():
            raise FileNotFoundError(self.engine_path)
        data = self.engine_path.read_bytes()
        engine = self.runtime.deserialize_cuda_engine(data)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize {self.engine_path}")
        logger.info("Deserialized TensorRT engine: %s", self.engine_path)
        return engine

    def _ensure_host_buffer(self, arr: np.ndarray, name: str) -> np.ndarray:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Input '{name}' must be a numpy array, got {type(arr)}")
        if arr.dtype != trt.nptype(self.engine.get_tensor_dtype(name)):
            raise TypeError(
                f"Input '{name}' dtype {arr.dtype} does not match tensor dtype "
                f"{trt.nptype(self.engine.get_tensor_dtype(name))}"
            )
        if arr.ndim == 0:
            raise ValueError(f"Input '{name}' has zero dimensions")
        if not arr.flags["C_CONTIGUOUS"] or not arr.flags["ALIGNED"]:
            logger.warning("Input '%s' is not C-contiguous; making a contiguous copy", name)
            arr = np.ascontiguousarray(arr, dtype=arr.dtype)
        return arr

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot infer on a closed TrtEngine")

            provided = set(inputs.keys())
            expected = set(self._input_names)
            if provided != expected:
                missing = expected - provided
                extra = provided - expected
                raise ValueError(
                    f"Input name mismatch. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
                )

            # Upload inputs and bind
            for name, arr in inputs.items():
                arr = self._ensure_host_buffer(arr, name)
                shape = tuple(arr.shape)
                ok = self.context.set_input_shape(name, shape)
                if not ok:
                    raise RuntimeError(
                        f"set_input_shape failed for '{name}' with shape {shape}"
                    )
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
                if not self.context.set_tensor_address(name, d_ptr):
                    raise RuntimeError(f"set_tensor_address failed for input '{name}'")

            # Allocate and bind outputs
            outputs: dict[str, np.ndarray] = {}
            for name in self._output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(s <= 0 for s in shape):
                    raise RuntimeError(
                        f"Output '{name}' has invalid shape {shape}; input shapes may not be set"
                    )
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                arr = np.empty(shape, dtype=dtype)
                arr = np.ascontiguousarray(arr)
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
                if not self.context.set_tensor_address(name, d_ptr):
                    raise RuntimeError(f"set_tensor_address failed for output '{name}'")
                outputs[name] = arr

            success = self.context.execute_async_v3(self.stream)
            if not success:
                raise RuntimeError("execute_async_v3 failed")

            err = cuda_runtime.cudaStreamSynchronize(self.stream)
            if err[0] != cuda_runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaStreamSynchronize after execute failed: {err}")

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
                raise RuntimeError(f"cudaStreamSynchronize after D2H failed: {err}")
            return outputs

    def warmup(self, input_shapes: dict[str, tuple[int, ...]] | None = None) -> None:
        shapes = input_shapes or {}
        for name in self._input_names:
            if name in shapes:
                shape = shapes[name]
            else:
                shape = tuple(self.engine.get_tensor_shape(name))
                if any(s <= 0 for s in shape):
                    shape = tuple(max(1, s) if s > 0 else 1 for s in shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            arr = np.zeros(shape, dtype=dtype)
            self.infer({name: arr})
            break

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for key in list(self._buffers.keys()):
                try:
                    cuda_runtime.cudaFree(self._buffers[key]["ptr"])
                except Exception as exc:
                    logger.warning("cudaFree failed for %s: %s", key, exc)
                self._buffers.pop(key, None)
            if hasattr(self, "stream"):
                try:
                    cuda_runtime.cudaStreamDestroy(self.stream)
                except Exception as exc:
                    logger.warning("cudaStreamDestroy failed: %s", exc)
                del self.stream
            self._closed = True

    # Backward compatibility / alias
    destroy = close

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
