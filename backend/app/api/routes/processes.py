"""Process log query API route.

This router only reads recognition process records that the face service
persists during recognition calls.
"""
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.controllers.face_controller import FaceController
from app.api.dependencies import (
    get_db,
    get_face_pipeline,
    get_face_pipeline_lock,
    get_storage,
    get_vector_store,
)
from app.infrastructure.minio import PhotoStorage
from app.infrastructure.qdrant import FaceVectorStore
from app.ml.gpu.face_pipeline import GpuFacePipeline
from app.schemas.face import ProcessDetail
from app.services.face_service import FaceService

router = APIRouter(prefix="/processes", tags=["processes"])


def _build_service(
    db: AsyncSession,
    storage: PhotoStorage,
    vector_store: FaceVectorStore,
    pipeline: GpuFacePipeline,
    pipeline_lock,
) -> FaceService:
    return FaceService(
        db=db,
        storage=storage,
        vector_store=vector_store,
        pipeline=pipeline,
        pipeline_lock=pipeline_lock,
    )


@router.get(
    "/{process_id}",
    response_model=ProcessDetail,
    summary="Get recognition process details",
)
async def get_process(
    process_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[PhotoStorage, Depends(get_storage)],
    vector_store: Annotated[FaceVectorStore, Depends(get_vector_store)],
    pipeline: Annotated[GpuFacePipeline, Depends(get_face_pipeline)],
    pipeline_lock=Depends(get_face_pipeline_lock),
) -> ProcessDetail:
    service = _build_service(db, storage, vector_store, pipeline, pipeline_lock)
    controller = FaceController(service)
    return await controller.get_process(process_id)
