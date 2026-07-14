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
    name: str | None = None


class RecognizedFace(_ApiModel):
    face_index: int = Field(alias="faceIndex")
    face_id: uuid.UUID | None = Field(alias="faceId", default=None)
    person_id: uuid.UUID | None = Field(alias="personId", default=None)
    photo_id: uuid.UUID | None = Field(alias="photoId", default=None)
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
    photo_id: uuid.UUID | None = Field(alias="photoId", default=None)
    name: str
    national_id_masked: str = Field(alias="nationalIdMasked")
    status: str
    bounding_box: BoundingBox | None = Field(alias="boundingBox", default=None)
    landmarks: list[list[float]] | None = Field(default=None)
    metadata: dict[str, Any] | None = None
    created_at: datetime = Field(alias="createdAt")
    photos: list[FacePhotoItem]


class AddPhotoResponse(_ApiModel):
    face_id: uuid.UUID = Field(alias="faceId")
    person_id: uuid.UUID = Field(alias="personId")
    photo_id: uuid.UUID = Field(alias="photoId")
    sample_id: uuid.UUID = Field(alias="sampleId")
    status: str
    name: str
    created_at: datetime = Field(alias="createdAt")


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
    photo_id: uuid.UUID | None = Field(alias="photoId", default=None)
    name: str
    national_id_masked: str = Field(alias="nationalIdMasked")
    status: str
    created_at: datetime = Field(alias="createdAt")
    photo_count: int = Field(alias="photoCount", default=1)


class FaceListResponse(_ApiModel):
    items: list[FaceListItem]
    total: int
    limit: int
    offset: int


class BulkJobShard(_ApiModel):
    worker_id: str = Field(alias="workerId")
    shard_index: int = Field(alias="shardIndex")
    process_id: uuid.UUID = Field(alias="processId")
    status: str
    progress: dict[str, Any] = Field(default_factory=dict)


class VggfaceBulkJobResponse(_ApiModel):
    job_id: uuid.UUID = Field(alias="jobId")
    status: str
    dataset_type: str = Field(alias="datasetType")
    assigned_workers: list[str] = Field(alias="assignedWorkers")
    target_total_active_photos: int = Field(alias="targetTotalActivePhotos")
    starting_active_photos: int = Field(alias="startingActivePhotos")
    current_active_photos: int = Field(alias="currentActivePhotos")
    photos_added_by_job: int = Field(alias="photosAddedByJob")
    requested_photos: int = Field(alias="requestedPhotos")
    total_discovered: int = Field(alias="totalDiscovered", default=0)
    total_scanned: int = Field(alias="totalScanned", default=0)
    total_processed: int = Field(alias="totalProcessed", default=0)
    total_enrolled: int = Field(alias="totalEnrolled")
    total_duplicate: int = Field(alias="totalDuplicate")
    total_no_face: int = Field(alias="totalNoFace")
    total_errors: int = Field(alias="totalErrors")
    total_in_flight: int = Field(alias="totalInFlight", default=0)
    total_rejected: int = Field(alias="totalRejected", default=0)
    total_corrupt: int = Field(alias="totalCorrupt", default=0)
    elapsed_seconds: float = Field(alias="elapsedSeconds")
    avg_photos_per_second: float = Field(alias="avgPhotosPerSecond")
    scanned_photos_per_second: float = Field(alias="scannedPhotosPerSecond", default=0.0)
    processed_photos_per_second: float = Field(alias="processedPhotosPerSecond", default=0.0)
    enrolled_photos_per_second: float = Field(alias="enrolledPhotosPerSecond", default=0.0)
    duplicate_photos_per_second: float = Field(alias="duplicatePhotosPerSecond", default=0.0)
    probe_p50_ms: float | None = Field(alias="probeP50Ms", default=None)
    probe_p95_ms: float | None = Field(alias="probeP95Ms", default=None)
    shards: list[BulkJobShard]
    created_at: datetime = Field(alias="createdAt")
    completed_at: datetime | None = Field(alias="completedAt", default=None)


class VggfaceBulkJobStartRequest(_ApiModel):
    max_photos: int | None = Field(alias="maxPhotos", default=None)
