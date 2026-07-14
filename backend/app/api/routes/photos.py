"""Photo serving routes.

These endpoints proxy images from object storage so the browser never needs
MinIO credentials or direct bucket access.
"""
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_storage
from app.core.errors import NotFoundError
from app.domain.models import PersonPhoto
from app.infrastructure.minio import PhotoStorage

router = APIRouter(prefix="/photos", tags=["photos"])


@router.get(
    "/{photo_id}",
    responses={
        status.HTTP_200_OK: {
            "content": {
                "image/jpeg": {},
                "image/png": {},
                "application/octet-stream": {},
            }
        }
    },
    summary="Get a photo by ID",
)
async def get_photo(
    photo_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    storage: Annotated[PhotoStorage, Depends(get_storage)],
) -> Response:
    result = await db.execute(
        select(PersonPhoto).where(PersonPhoto.photo_id == photo_id)
    )
    photo = result.scalar_one_or_none()
    if photo is None:
        raise NotFoundError(f"photo {photo_id} not found")

    data = await storage.get_object(photo.object_key)
    return Response(
        content=data,
        media_type=photo.mime_type or "application/octet-stream",
        headers={"Cache-Control": "private, max-age=300"},
    )
