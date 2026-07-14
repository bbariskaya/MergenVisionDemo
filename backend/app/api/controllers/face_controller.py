"""HTTP-agnostic controller layer for face API endpoints.

Controllers translate between transport-level objects (bytes, form values)
and the domain service. They do not contain business rules.
"""
from __future__ import annotations

import uuid

from app.schemas.face import (
    BulkEnrollRecord,
    BulkEnrollResponse,
    EnrollResponse,
    FaceDetail,
    FaceHistoryEntry,
    FaceListResponse,
    ProcessDetail,
    RecognizedFace,
    RecognizeResponse,
)
from app.services.face_service import BulkEnrollItem, FaceService


class FaceController:
    """Coordinates face service calls and response serialization."""

    def __init__(self, service: FaceService) -> None:
        self._service = service

    async def recognize(
        self,
        image_bytes: bytes,
        *,
        top_k: int,
        threshold: float,
    ) -> RecognizeResponse:
        process_id, face_dicts = await self._service.recognize_image(
            image_bytes,
            top_k=top_k,
            threshold=threshold,
        )
        return RecognizeResponse(
            process_id=process_id,
            face_count=len(face_dicts),
            faces=[RecognizedFace.model_validate(fd) for fd in face_dicts],
        )

    async def enroll(
        self,
        image_bytes: bytes,
        *,
        name: str,
        national_id: str,
        metadata: dict | None,
    ) -> EnrollResponse:
        result = await self._service.enroll_face(
            image_bytes,
            name=name,
            national_id=national_id,
            metadata=metadata,
        )
        return EnrollResponse.model_validate(result)

    async def bulk_enroll(
        self,
        items: list[tuple[bytes, str, str, dict | None]],
    ) -> BulkEnrollResponse:
        service_items = [
            BulkEnrollItem(
                image_bytes=image_bytes,
                name=name,
                national_id=national_id,
                metadata=metadata,
            )
            for image_bytes, name, national_id, metadata in items
        ]
        records, errors, _ = await self._service.bulk_enroll(service_items)
        return BulkEnrollResponse(
            enrolled=[BulkEnrollRecord.model_validate(r) for r in records],
            errors=[{"index": idx, "message": msg} for idx, msg in errors],
        )

    async def get_face(self, face_id: uuid.UUID) -> FaceDetail:
        result = await self._service.get_face(face_id)
        return FaceDetail.model_validate(result)

    async def delete_face(self, face_id: uuid.UUID) -> None:
        await self._service.delete_face(face_id)

    async def get_history(self, face_id: uuid.UUID) -> list[FaceHistoryEntry]:
        entries = await self._service.get_face_history(face_id)
        return [FaceHistoryEntry.model_validate(e) for e in entries]

    async def list_faces(
        self,
        *,
        search: str | None,
        is_active: bool | None,
        limit: int,
        offset: int,
    ) -> FaceListResponse:
        result = await self._service.list_faces(
            search=search,
            is_active=is_active,
            limit=limit,
            offset=offset,
        )
        return FaceListResponse.model_validate(result)

    async def get_process(self, process_id: uuid.UUID) -> ProcessDetail:
        result = await self._service.get_process(process_id)
        return ProcessDetail.model_validate(result)
