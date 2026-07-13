from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.infrastructure import db as db_module
from app.infrastructure.minio import PhotoStorage
from app.infrastructure.qdrant import FaceVectorStore
from app.services.readiness import ReadinessService


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_module.configure_engine()

    storage = PhotoStorage()
    await storage.initialize()

    vector_store = FaceVectorStore()
    await vector_store.initialize()

    app.state.storage = storage
    app.state.vector_store = vector_store
    app.state.readiness_service = ReadinessService(
        engine=db_module.engine,
        storage=storage,
        vector_store=vector_store,
    )

    yield

    await db_module.dispose_engine()
