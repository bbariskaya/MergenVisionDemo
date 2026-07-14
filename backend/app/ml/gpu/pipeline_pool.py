"""Multi-GPU pool of face pipelines for bulk extraction."""
from __future__ import annotations

import asyncio
import logging

from cuda.bindings import runtime as cuda_runtime

from app.core.config import Settings, settings as default_settings
from app.ml.gpu.face_pipeline import GpuFacePipeline

logger = logging.getLogger(__name__)


def _detect_gpu_count() -> int:
    """Return the number of CUDA devices visible to this process."""
    try:
        err, count = cuda_runtime.cudaGetDeviceCount()
        if err == 0 and count > 0:
            return int(count)
    except Exception as exc:
        logger.warning("could not get CUDA device count: %s", exc)
    return 1


class GpuPipelinePool:
    """Owns one pipeline per visible GPU."""

    def __init__(
        self,
        cfg: Settings = default_settings,
        *,
        num_gpus: int | None = None,
    ) -> None:
        self._cfg = cfg
        self._num_gpus = num_gpus if num_gpus is not None else _detect_gpu_count()
        self._pipelines: list[GpuFacePipeline] = []
        self._locks: list[asyncio.Lock] = []
        for i in range(self._num_gpus):
            logger.info("creating face pipeline on GPU %d/%d", i + 1, self._num_gpus)
            pipeline = GpuFacePipeline(cfg=cfg, device_id=i)
            self._pipelines.append(pipeline)
            self._locks.append(asyncio.Lock())

    def __len__(self) -> int:
        return self._num_gpus

    def __getitem__(self, idx: int) -> tuple[GpuFacePipeline, asyncio.Lock]:
        return self._pipelines[idx], self._locks[idx]

    def warmup(self) -> None:
        for pipeline in self._pipelines:
            pipeline.warmup()

    def close(self) -> None:
        for pipeline in self._pipelines:
            try:
                pipeline.close()
            except Exception as exc:
                logger.warning("pipeline close failed: %s", exc)
