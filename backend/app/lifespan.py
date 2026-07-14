import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.infrastructure import db as db_module
from app.infrastructure.minio import PhotoStorage
from app.infrastructure.qdrant import FaceVectorStore
from app.ml.gpu.face_pipeline import GpuFacePipeline
from app.services.readiness import ReadinessService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_module.configure_engine()
    storage: PhotoStorage | None = None
    vector_store: FaceVectorStore | None = None
    face_pipeline: GpuFacePipeline | None = None

    try:
        storage = PhotoStorage()
        await storage.initialize()

        vector_store = FaceVectorStore()
        await vector_store.initialize()

        face_pipeline = GpuFacePipeline(device_id=0)
        face_pipeline.warmup()

        app.state.storage = storage
        app.state.vector_store = vector_store
        app.state.face_pipeline = face_pipeline
        app.state.face_pipeline_lock = asyncio.Lock()
        app.state.readiness_service = ReadinessService(
            engine=db_module.engine,
            storage=storage,
            vector_store=vector_store,
        )

        yield
    finally:
        # NOTE: We intentionally do not call face_pipeline.close(). NVIDIA's
        # nvImageCodec global cleanup can throw on explicit close during Uvicorn
        # shutdown; simply exiting the process releases GPU resources cleanly.
        if vector_store is not None:
            try:
                await vector_store.close()
            except Exception as exc:
                logger.warning(
                    "Qdrant client close failed during shutdown: %s",
                    exc.__class__.__name__,
                )
        await db_module.dispose_engine()
