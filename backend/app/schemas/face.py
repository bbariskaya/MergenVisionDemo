"""Pydantic schemas for face recognition API endpoints."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _ApiModel(BaseModel):
    """Base that lets callers use either snake_case field names or camelCase aliases."""

    model_config = ConfigDict(populate_by_name=True)


class BoundingBox(_ApiModel):
    x1: float
    y1: float
    x2: float
    y2: float


class FaceCandidate(_ApiModel):
    face_id: uuid.UUID = Field(alias="faceId")
    person_id: uuid.UUID = Field(alias="personId")
    photo_id: uuid.UUID = Field(alias="photoId")
    score: float


class RecognizedFace(_ApiModel):
    face_index: int = Field(alias="faceIndex")
    face_id: uuid.UUID | None = Field(alias="faceId", default=None)
    status: str
    name: str | None = None
    metadata: dict[str, Any] | None = None
    bounding_box: BoundingBox = Field(alias="boundingBox")
    landmarks: list[list[float]]
    confidence: float | None = None
    candidates: list[FaceCandidate] = Field(default_factory=list)


class RecognizeResponse(_ApiModel):
    process_id: uuid.UUID = Field(alias="processId")
    face_count: int = Field(alias="faceCount")
    faces: list[RecognizedFace]


class EnrollResponse(_ApiModel):
    face_id: uuid.UUID = Field(alias="faceId")
    person_id: uuid.UUID = Field(alias="personId")
    photo_id: uuid.UUID = Field(alias="photoId")
    status: str
    name: str
    created_at: datetime = Field(alias="createdAt")


class FacePhotoItem(_ApiModel):
    photo_id: uuid.UUID = Field(alias="photoId")
    status: str
    created_at: datetime = Field(alias="createdAt")


class FaceDetail(_ApiModel):
    face_id: uuid.UUID = Field(alias="faceId")
    person_id: uuid.UUID = Field(alias="personId")
    photo_id: uuid.UUID = Field(alias="photoId")
    name: str
    national_id_masked: str = Field(alias="nationalIdMasked")
    status: str
    bounding_box: BoundingBox = Field(alias="boundingBox")
    landmarks: list[list[float]]
    metadata: dict[str, Any] | None = None
    created_at: datetime = Field(alias="createdAt")
    photos: list[FacePhotoItem]


class FaceHistoryEntry(_ApiModel):
    process_id: uuid.UUID = Field(alias="processId")
    status: str
    timestamp: datetime


class ProcessFace(_ApiModel):
    face_index: int = Field(alias="faceIndex")
    status: str
    face_id: uuid.UUID | None = Field(alias="faceId", default=None)
    score: float | None = None
    bounding_box: BoundingBox = Field(alias="boundingBox")


class ProcessDetail(_ApiModel):
    process_id: uuid.UUID = Field(alias="processId")
    status: str
    face_count: int | None = Field(alias="faceCount", default=None)
    created_at: datetime = Field(alias="createdAt")
    completed_at: datetime | None = Field(alias="completedAt", default=None)
    faces: list[ProcessFace]


class BulkEnrollItemRequest(_ApiModel):
    name: str
    national_id: str = Field(alias="nationalId")
    metadata: dict[str, Any] | None = None


class BulkEnrollRecord(_ApiModel):
    index: int
    face_id: uuid.UUID = Field(alias="faceId")
    person_id: uuid.UUID = Field(alias="personId")
    photo_id: uuid.UUID = Field(alias="photoId")
    status: str
    name: str


class BulkEnrollResponse(_ApiModel):
    enrolled: list[BulkEnrollRecord]
    errors: list[dict[str, Any]]


class FaceListItem(_ApiModel):
    face_id: uuid.UUID = Field(alias="faceId")
    person_id: uuid.UUID = Field(alias="personId")
    photo_id: uuid.UUID = Field(alias="photoId")
    name: str
    national_id_masked: str = Field(alias="nationalIdMasked")
    status: str
    created_at: datetime = Field(alias="createdAt")


class FaceListResponse(_ApiModel):
    items: list[FaceListItem]
    total: int
    limit: int
    offset: int
