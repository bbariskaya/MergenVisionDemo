"""Face recognition and enrollment API routes.

Routers are intentionally thin. They depend on the controller layer for
request/response orchestration and the service layer for domain logic.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.controllers.face_controller import FaceController
from app.api.dependencies import (
    get_db,
    get_face_pipeline,
    get_face_pipeline_lock,
    get_storage,
    get_vector_store,
)
from app.core.config import settings
from app.core.errors import ValidationError
from app.infrastructure.minio import PhotoStorage
from app.infrastructure.qdrant import FaceVectorStore
from app.ml.gpu.face_pipeline import GpuFacePipeline
from app.schemas.face import (
    AddPhotoResponse,
    BulkEnrollItemRequest,
    BulkEnrollResponse,
    EnrollResponse,
    FaceDetail,
    FaceHistoryEntry,
    FaceListResponse,
    RecognizeResponse,
)
from app.services.face_service import FaceService

router = APIRouter(prefix="/faces", tags=["faces"])


def _build_service(
    db: AsyncSession,
    storage: PhotoStorage,
    vector_store: FaceVectorStore,
    pipeline: GpuFacePipeline,
    pipeline_lock,
) -> FaceService:
    """Factory used by routes to wire dependencies into the service layer."""
    return FaceService(
        db=db,
        storage=storage,
        vector_store=vector_store,
        pipeline=pipeline,
        pipeline_lock=pipeline_lock,
    )


@router.post(
    "/recognize",
    response_model=RecognizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Recognize faces in an image",
)
async def recognize_faces(
    image: Annotated[UploadFile, File()],
    db: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[PhotoStorage, Depends(get_storage)],
    vector_store: Annotated[FaceVectorStore, Depends(get_vector_store)],
    pipeline: Annotated[GpuFacePipeline, Depends(get_face_pipeline)],
    pipeline_lock=Depends(get_face_pipeline_lock),
    top_k: int = Query(
        default=settings.top_k_default,
        ge=1,
        le=settings.top_k_max,
    ),
    threshold: float = Query(
        default=settings.matched_threshold,
        ge=0.0,
        le=1.0,
    ),
) -> RecognizeResponse:
    image_bytes = await image.read()
    service = _build_service(db, storage, vector_store, pipeline, pipeline_lock)
    controller = FaceController(service)
    return await controller.recognize(image_bytes, top_k=top_k, threshold=threshold)


@router.post(
    "/enroll",
    response_model=EnrollResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a named face",
)
async def enroll_face(
    name: Annotated[str, Form(min_length=1)],
    national_id: Annotated[str, Form(min_length=1, alias="nationalId")],
    image: Annotated[UploadFile, File()],
    db: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[PhotoStorage, Depends(get_storage)],
    vector_store: Annotated[FaceVectorStore, Depends(get_vector_store)],
    pipeline: Annotated[GpuFacePipeline, Depends(get_face_pipeline)],
    pipeline_lock=Depends(get_face_pipeline_lock),
    metadata: Annotated[str | None, Form()] = None,
) -> EnrollResponse:
    image_bytes = await image.read()
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise ValidationError("metadata must be valid JSON") from exc
    else:
        parsed_metadata = None

    service = _build_service(db, storage, vector_store, pipeline, pipeline_lock)
    controller = FaceController(service)
    return await controller.enroll(
        image_bytes,
        name=name,
        national_id=national_id,
        metadata=parsed_metadata,
    )


@router.post(
    "/enroll/bulk",
    response_model=BulkEnrollResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bulk enroll named faces",
)
async def bulk_enroll_faces(
    images: Annotated[list[UploadFile], File()],
    entries: Annotated[str, Form()],
    db: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[PhotoStorage, Depends(get_storage)],
    vector_store: Annotated[FaceVectorStore, Depends(get_vector_store)],
    pipeline: Annotated[GpuFacePipeline, Depends(get_face_pipeline)],
    pipeline_lock=Depends(get_face_pipeline_lock),
) -> BulkEnrollResponse:
    if not images:
        raise ValidationError("at least one image is required")
    try:
        parsed_entries = [
            BulkEnrollItemRequest.model_validate(entry)
            for entry in json.loads(entries)
        ]
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValidationError("entries must be a valid JSON array of metadata objects") from exc

    if len(parsed_entries) != len(images):
        raise ValidationError(
            f"entries count ({len(parsed_entries)}) must equal images count ({len(images)})"
        )

    image_bytes_list = await asyncio.gather(*[image.read() for image in images])
    items = [
        (image_bytes, entry.name, entry.national_id, entry.metadata)
        for image_bytes, entry in zip(image_bytes_list, parsed_entries)
    ]

    service = _build_service(db, storage, vector_store, pipeline, pipeline_lock)
    controller = FaceController(service)
    return await controller.bulk_enroll(items)


@router.get(
    "",
    response_model=FaceListResponse,
    summary="List enrolled faces",
)
async def list_faces(
    db: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[PhotoStorage, Depends(get_storage)],
    vector_store: Annotated[FaceVectorStore, Depends(get_vector_store)],
    pipeline: Annotated[GpuFacePipeline, Depends(get_face_pipeline)],
    pipeline_lock=Depends(get_face_pipeline_lock),
    search: Annotated[str | None, Query()] = None,
    is_active: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> FaceListResponse:
    service = _build_service(db, storage, vector_store, pipeline, pipeline_lock)
    controller = FaceController(service)
    return await controller.list_faces(
        search=search,
        is_active=is_active,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{face_id}",
    response_model=FaceDetail,
    summary="Get face details",
)
async def get_face(
    face_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[PhotoStorage, Depends(get_storage)],
    vector_store: Annotated[FaceVectorStore, Depends(get_vector_store)],
    pipeline: Annotated[GpuFacePipeline, Depends(get_face_pipeline)],
    pipeline_lock=Depends(get_face_pipeline_lock),
) -> FaceDetail:
    service = _build_service(db, storage, vector_store, pipeline, pipeline_lock)
    controller = FaceController(service)
    return await controller.get_face(face_id)


@router.post(
    "/{face_id}/photos",
    response_model=AddPhotoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a photo to an enrolled person",
)
async def add_person_photo(
    face_id: uuid.UUID,
    image: Annotated[UploadFile, File()],
    db: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[PhotoStorage, Depends(get_storage)],
    vector_store: Annotated[FaceVectorStore, Depends(get_vector_store)],
    pipeline: Annotated[GpuFacePipeline, Depends(get_face_pipeline)],
    pipeline_lock=Depends(get_face_pipeline_lock),
) -> AddPhotoResponse:
    image_bytes = await image.read()
    service = _build_service(db, storage, vector_store, pipeline, pipeline_lock)
    controller = FaceController(service)
    return await controller.add_person_photo(face_id, image_bytes)


@router.delete(
    "/{face_id}/photos/{photo_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a photo from an enrolled person",
)
async def delete_person_photo(
    face_id: uuid.UUID,
    photo_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[PhotoStorage, Depends(get_storage)],
    vector_store: Annotated[FaceVectorStore, Depends(get_vector_store)],
    pipeline: Annotated[GpuFacePipeline, Depends(get_face_pipeline)],
    pipeline_lock=Depends(get_face_pipeline_lock),
) -> None:
    service = _build_service(db, storage, vector_store, pipeline, pipeline_lock)
    controller = FaceController(service)
    await controller.delete_person_photo(face_id, photo_id)


@router.delete(
    "/{face_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a face",
)
async def delete_face(
    face_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[PhotoStorage, Depends(get_storage)],
    vector_store: Annotated[FaceVectorStore, Depends(get_vector_store)],
    pipeline: Annotated[GpuFacePipeline, Depends(get_face_pipeline)],
    pipeline_lock=Depends(get_face_pipeline_lock),
) -> None:
    service = _build_service(db, storage, vector_store, pipeline, pipeline_lock)
    controller = FaceController(service)
    await controller.delete_face(face_id)


@router.get(
    "/{face_id}/history",
    response_model=list[FaceHistoryEntry],
    summary="Get recognition history for a face",
)
async def get_face_history(
    face_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[PhotoStorage, Depends(get_storage)],
    vector_store: Annotated[FaceVectorStore, Depends(get_vector_store)],
    pipeline: Annotated[GpuFacePipeline, Depends(get_face_pipeline)],
    pipeline_lock=Depends(get_face_pipeline_lock),
) -> list[FaceHistoryEntry]:
    service = _build_service(db, storage, vector_store, pipeline, pipeline_lock)
    controller = FaceController(service)
    return await controller.get_history(face_id)
