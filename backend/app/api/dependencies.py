"""FastAPI dependency functions for the API layer."""
from __future__ import annotations

import asyncio

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure import db as db_module
from app.infrastructure.minio import PhotoStorage
from app.infrastructure.qdrant import FaceVectorStore
from app.ml.gpu.face_pipeline import GpuFacePipeline


async def get_db() -> AsyncSession:
    """Yield an async SQLAlchemy session."""
    if db_module.AsyncSessionLocal is None:
        raise RuntimeError("Database engine not configured")
    async with db_module.AsyncSessionLocal() as session:
        yield session


def get_face_pipeline_lock(request: Request) -> asyncio.Lock:
    """Return the global lock that serializes GPU pipeline access."""
    lock = getattr(request.app.state, "face_pipeline_lock", None)
    if lock is None:
        raise RuntimeError("Face pipeline lock not initialized")
    return lock


def get_storage(request: Request) -> PhotoStorage:
    """Return the initialized MinIO photo storage client."""
    storage = getattr(request.app.state, "storage", None)
    if storage is None:
        raise RuntimeError("PhotoStorage not initialized")
    return storage


def get_vector_store(request: Request) -> FaceVectorStore:
    """Return the initialized Qdrant vector store client."""
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is None:
        raise RuntimeError("FaceVectorStore not initialized")
    return vector_store


def get_face_pipeline(request: Request) -> GpuFacePipeline:
    """Return the warmed-up GPU face pipeline."""
    pipeline = getattr(request.app.state, "face_pipeline", None)
    if pipeline is None:
        raise RuntimeError("GpuFacePipeline not initialized")
    return pipeline
