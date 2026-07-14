"""GPU ArcFace recognizer using a TensorRT engine and native L2 normalize."""
from __future__ import annotations

import ctypes
import logging
from pathlib import Path

from cuda.bindings import runtime as cuda_runtime

from app.core.config import Settings
from app.ml.gpu.buffer_arena import BufferArena
from app.ml.gpu.device_tensor import DeviceTensor, check_cuda
from app.ml.gpu.l2_norm import l2_normalize_device
from app.ml.gpu.trt_device_engine import TrtDeviceEngine

logger = logging.getLogger(__name__)


class GpuRecognizer:
    """Device-only ArcFace feature extractor.

    Expects aligned RGB face crops as ``[N, 3, 112, 112]`` float32 with
    pixel values already normalized (the caller, e.g. ``GpuFaceAligner``,
    is responsible for (x - 127.5) / 127.5). The returned embeddings are
    L2-normalized on device and copied to host only at the pipeline boundary.
    """

    def __init__(
        self,
        engine_path: Path | str | None = None,
        device_id: int = 0,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings
        if engine_path is None:
            if settings is None:
                raise ValueError("Provide engine_path or settings")
            engine_path = settings.embedder_engine_path
        self._engine = TrtDeviceEngine(engine_path, device_id=device_id)
        self._device_id = device_id
        self._arena = BufferArena(device_id=device_id)

    @property
    def engine(self) -> TrtDeviceEngine:
        return self._engine

    def max_batch(self) -> int:
        """Return the recognizer engine's maximum batch size."""
        input_name = self._engine._input_names[0]
        try:
            _, _, max_shape = self._engine.engine.get_tensor_profile_shape(
                input_name, 0
            )
            return int(max_shape[0])
        except Exception:
            return 64

    def embed(
        self,
        faces: DeviceTensor,
        *,
        stream: int | None = None,
        status: DeviceTensor | None = None,
    ) -> DeviceTensor:
        """Return L2-normalized embeddings for ``faces``.

        ``faces`` must be a float32 ``[N, 3, 112, 112]`` device tensor.
        The result is a float32 ``[N, D]`` device tensor.

        If ``status`` is a ``[1]`` int32 device tensor, it is written by the
        L2 kernel and the caller is responsible for reading it. The caller
        must zero it before calling ``embed`` if chunking is expected.
        """
        err = cuda_runtime.cudaSetDevice(self._device_id)
        check_cuda(err, f"recognizer cudaSetDevice({self._device_id})")
        if faces.dtype is not ctypes.c_float:
            raise TypeError(f"GpuRecognizer expects float32 faces, got {faces.dtype}")
        if len(faces.shape) != 4 or faces.shape[1:] != (3, 112, 112):
            raise ValueError(
                f"GpuRecognizer expects [N,3,112,112], got {faces.shape}"
            )

        active_stream = stream if stream is not None else 0
        n = faces.shape[0]
        if n == 0:
            return self._arena.reserve((0, self.embedding_dim), ctypes.c_float)

        max_batch = self.max_batch()
        if n <= max_batch:
            return self._embed_chunk(faces, active_stream, status=status)

        embeddings: list[DeviceTensor] = []
        for offset in range(0, n, max_batch):
            chunk = self._slice_faces(faces, offset, min(max_batch, n - offset))
            embeddings.append(self._embed_chunk(chunk, active_stream, status=status))

        return self._concat_embeddings(embeddings, active_stream)

    def _slice_faces(
        self,
        faces: DeviceTensor,
        offset: int,
        count: int,
        stream: int,
    ) -> DeviceTensor:
        """Create a lightweight view of a batch slice.

        This copies by device pointer offset; it assumes ``faces`` is contiguous.
        """
        if count <= 0:
            raise ValueError("slice count must be positive")
        elem_bytes = 4 * 3 * 112 * 112
        ptr = faces.ptr + offset * elem_bytes
        return DeviceTensor(
            ptr=ptr,
            shape=(count, 3, 112, 112),
            dtype=ctypes.c_float,
            device_id=self._device_id,
            owner=faces,
            stream=stream,
        )

    def _embed_chunk(
        self,
        faces: DeviceTensor,
        stream: int,
        *,
        status: DeviceTensor | None = None,
    ) -> DeviceTensor:
        input_name = self._engine._input_names[0]
        outputs = self._engine.infer_device({input_name: faces}, stream=stream)
        # ArcFace engines have a single output embedding tensor.
        embedding_tensor = next(iter(outputs.values()))
        if embedding_tensor.dtype is not ctypes.c_float:
            raise TypeError(
                f"Recognizer output is not float32: {embedding_tensor.dtype}"
            )
        # Normalize in place on the engine output buffer; no D2D copy.
        return l2_normalize_device(
            embedding_tensor,
            arena=self._arena,
            stream=stream,
            output=embedding_tensor,
            status=status,
        )

    def _concat_embeddings(
        self,
        embeddings: list[DeviceTensor],
        stream: int,
    ) -> DeviceTensor:
        if len(embeddings) == 1:
            return embeddings[0]
        total = sum(t.shape[0] for t in embeddings)
        dim = embeddings[0].shape[1]
        out = self._arena.reserve((total, dim), ctypes.c_float, stream=stream)
        offset = 0
        for tensor in embeddings:
            n = tensor.shape[0]
            if n == 0:
                continue
            row_bytes = dim * 4
            err = cuda_runtime.cudaMemcpyAsync(
                out.ptr + offset * row_bytes,
                tensor.ptr,
                n * row_bytes,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                stream,
            )
            check_cuda(err, "concat embeddings D2D")
            offset += n
        return out

    @property
    def embedding_dim(self) -> int:
        if self._settings is not None:
            return self._settings.embedding_dim
        # Infer from engine output shape if possible.
        for name in self._engine._output_names:
            shape = self._engine.engine.get_tensor_shape(name)
            if len(shape) == 2:
                return int(shape[1])
        return 512

    def warmup(self) -> None:
        input_name = self._engine._input_names[0]
        self._engine.warmup({input_name: (1, 3, 112, 112)})

    def close(self) -> None:
        self._engine.close()
        self._arena.close()
