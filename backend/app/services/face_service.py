"""Domain/application service for face recognition and enrollment operations."""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import io
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import sqlalchemy as sa
from qdrant_client import models
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.errors import ConflictError, InternalError, NotFoundError, ValidationError
from app.core.ids import (
    derive_face_identity_id,
    derive_person_id,
    new_uuid7,
)
from app.core.security import hash_national_id, mask_national_id
from app.domain.models import (
    FaceIdentity,
    FaceSample,
    Person,
    PersonPhoto,
    RecognitionRequest,
    RecognitionResult,
)
from app.infrastructure.minio import PhotoStorage
from app.infrastructure.qdrant import FaceVectorStore, SearchHit
from app.ml.gpu.face_pipeline import GpuFaceExtraction, GpuFacePipeline


@dataclass
class BulkEnrollItem:
    image_bytes: bytes
    name: str
    national_id: str
    metadata: dict[str, Any] | None = None


class FaceService:
    """Orchestrates face recognition, enrollment, and process history.

    The service owns all business rules. It is deliberately HTTP-agnostic:
    callers are responsible for request/response serialization.
    """

    def __init__(
        self,
        *,
        db: AsyncSession,
        storage: PhotoStorage,
        vector_store: FaceVectorStore,
        pipeline: GpuFacePipeline,
        pipeline_lock: asyncio.Lock,
        gpu_executor: concurrent.futures.Executor | None = None,
    ) -> None:
        self._db = db
        self._storage = storage
        self._vector_store = vector_store
        self._pipeline = pipeline
        self._pipeline_lock = pipeline_lock
        self._gpu_executor = gpu_executor

    @staticmethod
    def _split_name(full_name: str) -> tuple[str, str]:
        parts = full_name.strip().split()
        if not parts:
            return "", ""
        if len(parts) == 1:
            return parts[0], ""
        # "Ali Veli Kaya" -> first="Ali Veli", last="Kaya"
        return " ".join(parts[:-1]), parts[-1]

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    async def _extract_faces(self, image_bytes: bytes) -> list[GpuFaceExtraction]:
        """Run the GPU face pipeline under the global lock.

        The pipeline reuses its output buffers, so we copy the results before
        returning. This lets the lock be released while we perform async DB /
        storage / Qdrant work without another request overwriting the buffers.
        """
        async with self._pipeline_lock:
            raw_faces = self._pipeline.extract_bytes(image_bytes)
            return [
                GpuFaceExtraction(
                    bbox=f.bbox.copy(),
                    landmarks=f.landmarks.copy(),
                    embedding=f.embedding.copy(),
                    score=f.score,
                )
                for f in raw_faces
            ]

    async def _extract_batch_faces(
        self, image_bytes_list: list[bytes]
    ) -> list[GpuFaceExtraction | None]:
        """Batch extraction under the global lock, with defensive copies."""
        async with self._pipeline_lock:
            if self._gpu_executor is not None:
                loop = asyncio.get_running_loop()
                raw = await loop.run_in_executor(
                    self._gpu_executor,
                    lambda: self._pipeline.extract_batch(
                        image_bytes_list,
                        pick_largest=True,
                        max_batch=256,
                    ),
                )
            else:
                raw = self._pipeline.extract_batch(
                    image_bytes_list,
                    pick_largest=True,
                    max_batch=256,
                )
            return [
                GpuFaceExtraction(
                    bbox=f.bbox.copy(),
                    landmarks=f.landmarks.copy(),
                    embedding=f.embedding.copy(),
                    score=f.score,
                )
                if f is not None
                else None
                for f in raw
            ]

    async def recognize_image(
        self,
        image_bytes: bytes,
        *,
        top_k: int,
        threshold: float,
    ) -> tuple[uuid.UUID, list[dict[str, Any]]]:
        """Recognize all faces in an image and log the process.

        Returns a tuple of (process_id, list_of_face_dicts).
        """
        if not image_bytes:
            raise ValidationError("empty image")

        object_key = f"queries/{new_uuid7()}"
        await self._storage.put_object(
            object_key=object_key,
            data=io.BytesIO(image_bytes),
            length=len(image_bytes),
            content_type="application/octet-stream",
        )

        request = RecognitionRequest(
            query_object_key=object_key,
            status="pending",
            top_k=top_k,
            threshold=threshold,
        )
        self._db.add(request)
        await self._db.flush()

        try:
            faces = await self._extract_faces(image_bytes)
        except Exception as exc:
            request.status = "failed"
            request.completed_at = self._utc_now()
            await self._db.commit()
            raise InternalError(f"face extraction failed: {exc}") from exc

        request.face_count = len(faces)
        face_dicts: list[dict[str, Any]] = []

        for idx, face in enumerate(faces):
            candidates = await self._vector_store.search_active(
                face.embedding.tolist(),
                model_version=settings.model_pack,
                top_k=top_k,
                active=True,
            )
            best = candidates[0] if candidates else None
            is_known = best is not None and best.score >= threshold

            known_name: str | None = None
            known_metadata: dict[str, Any] | None = None
            known_face_id: uuid.UUID | None = None
            person_name_map: dict[uuid.UUID, str] = {}
            person_identity_map: dict[uuid.UUID, uuid.UUID] = {}
            person_details_map: dict[uuid.UUID, dict[str, Any] | None] = {}

            person_ids = {c.person_id for c in candidates}
            if is_known and best is not None:
                person_ids.add(best.person_id)
            if person_ids:
                person_rows = await self._db.execute(
                    select(
                        Person.person_id,
                        Person.face_identity_id,
                        Person.first_name,
                        Person.last_name,
                        Person.details,
                    ).where(Person.person_id.in_(person_ids))
                )
                for row in person_rows:
                    pid = row.person_id
                    person_name_map[pid] = f"{row.first_name} {row.last_name}".strip()
                    person_identity_map[pid] = row.face_identity_id
                    person_details_map[pid] = row.details

            if is_known and best is not None:
                known_name = person_name_map.get(best.person_id)
                known_face_id = person_identity_map.get(best.person_id)
                known_metadata = person_details_map.get(best.person_id)

            face_dict = self._build_recognized_face(
                face_index=idx,
                extraction=face,
                is_known=is_known,
                best=best,
                candidates=candidates,
                face_id=known_face_id,
                name=known_name,
                metadata=known_metadata,
                person_name_map=person_name_map,
                person_identity_map=person_identity_map,
            )
            face_dicts.append(face_dict)

            result = RecognitionResult(
                request_id=request.request_id,
                face_index=idx,
                recognition_status="known" if is_known else "unknown",
                bbox=_bbox_to_dict(face.bbox),
                best_person_id=best.person_id if is_known else None,
                best_photo_id=best.photo_id if is_known else None,
                best_sample_id=best.sample_id if is_known else None,
                best_score=best.score if is_known else None,
                candidates=[_hit_to_dict(hit) for hit in candidates],
            )
            self._db.add(result)

        request.status = "completed"
        request.completed_at = self._utc_now()
        await self._db.commit()
        return request.request_id, face_dicts

    async def enroll_face(
        self,
        image_bytes: bytes,
        *,
        name: str,
        national_id: str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Enroll a single face for a known person.

        If the national ID already exists, the new photo/sample is added to
        the existing person.
        """
        if not image_bytes:
            raise ValidationError("empty image")
        if not name or not national_id:
            raise ValidationError("name and national_id are required")

        faces = await self._extract_faces(image_bytes)
        if len(faces) == 0:
            raise ValidationError("no face detected in enrollment image")

        # Enrollment is for one identity; if several faces are present, use the
        # largest one (by bounding-box area) as the primary subject.
        face = max(faces, key=lambda f: _bbox_area(f.bbox))
        first_name, last_name = self._split_name(name)
        national_hmac = hash_national_id(national_id, settings.hmac_key)
        national_masked = mask_national_id(national_id)
        content_sha256 = hashlib.sha256(image_bytes).hexdigest()

        face_identity_result = await self._db.execute(
            select(FaceIdentity).where(
                FaceIdentity.identity_lookup_hmac == national_hmac
            )
        )
        face_identity = face_identity_result.scalar_one_or_none()

        if face_identity is None:
            from app.core.ids import derive_face_identity_id, derive_person_id

            face_identity = FaceIdentity(
                face_identity_id=derive_face_identity_id(national_hmac),
                identity_lookup_hmac=national_hmac,
                display_name=f"{first_name} {last_name}".strip(),
                external_identity_hash=national_hmac,
                is_active=True,
            )
            self._db.add(face_identity)
            await self._db.flush()

        person_result = await self._db.execute(
            select(Person).where(Person.national_id_lookup_hmac == national_hmac)
        )
        person = person_result.scalar_one_or_none()

        if person is None:
            person = Person(
                person_id=derive_person_id(national_hmac),
                face_identity_id=face_identity.face_identity_id,
                first_name=first_name,
                last_name=last_name,
                national_id_lookup_hmac=national_hmac,
                national_id_masked=national_masked,
                details=metadata or {},
            )
            self._db.add(person)
            await self._db.flush()
        else:
            person.first_name = first_name
            person.last_name = last_name
            person.face_identity_id = face_identity.face_identity_id
            if metadata is not None:
                person.details = metadata

        duplicate = await self._db.execute(
            select(PersonPhoto).where(
                PersonPhoto.person_id == person.person_id,
                PersonPhoto.content_sha256 == content_sha256,
            )
        )
        if duplicate.scalar_one_or_none() is not None:
            raise ConflictError(
                "this exact photo is already enrolled for this person"
            )

        object_key = f"enrollments/{new_uuid7()}"
        await self._storage.put_object(
            object_key=object_key,
            data=io.BytesIO(image_bytes),
            length=len(image_bytes),
            content_type="application/octet-stream",
        )

        photo = PersonPhoto(
            person_id=person.person_id,
            object_key=object_key,
            content_sha256=content_sha256,
            mime_type="application/octet-stream",
            width=0,
            height=0,
            status="active",
        )
        self._db.add(photo)
        await self._db.flush()

        sample = FaceSample(
            person_id=person.person_id,
            photo_id=photo.photo_id,
            detector_model=settings.model_pack,
            embedding_model=settings.model_pack,
            bbox=_bbox_to_dict(face.bbox),
            landmarks=face.landmarks.tolist(),
            quality_score=None,
            status="active",
        )
        self._db.add(sample)
        await self._db.flush()

        await self._vector_store.upsert_batch(
            [
                models.PointStruct(
                    id=str(sample.sample_id),
                    vector=face.embedding.tolist(),
                    payload={
                        "sampleId": str(sample.sample_id),
                        "photoId": str(photo.photo_id),
                        "personId": str(person.person_id),
                        "active": True,
                        "modelVersion": settings.model_pack,
                    },
                )
            ]
        )

        await self._db.commit()

        return {
            "face_id": face_identity.face_identity_id,
            "person_id": person.person_id,
            "photo_id": photo.photo_id,
            "status": "known",
            "name": f"{person.first_name} {person.last_name}".strip(),
            "created_at": sample.created_at,
        }

    async def add_person_photo(
        self,
        face_id: uuid.UUID,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        """Add a new photo/sample to an existing person."""
        if not image_bytes:
            raise ValidationError("empty image")

        person_result = await self._db.execute(
            select(Person).where(Person.face_identity_id == face_id)
        )
        person = person_result.scalar_one_or_none()
        if person is None:
            raise NotFoundError(f"person with face {face_id} not found")

        faces = await self._extract_faces(image_bytes)
        if len(faces) == 0:
            raise ValidationError("no face detected in photo")
        face = max(faces, key=lambda f: _bbox_area(f.bbox))

        content_sha256 = hashlib.sha256(image_bytes).hexdigest()
        duplicate = await self._db.execute(
            select(PersonPhoto).where(
                PersonPhoto.person_id == person.person_id,
                PersonPhoto.content_sha256 == content_sha256,
            )
        )
        if duplicate.scalar_one_or_none() is not None:
            raise ConflictError(
                "this exact photo is already enrolled for this person"
            )

        object_key = f"enrollments/{new_uuid7()}"
        await self._storage.put_object(
            object_key=object_key,
            data=io.BytesIO(image_bytes),
            length=len(image_bytes),
            content_type="application/octet-stream",
        )

        photo = PersonPhoto(
            person_id=person.person_id,
            object_key=object_key,
            content_sha256=content_sha256,
            mime_type="application/octet-stream",
            width=0,
            height=0,
            status="active",
        )
        self._db.add(photo)
        await self._db.flush()

        sample = FaceSample(
            person_id=person.person_id,
            photo_id=photo.photo_id,
            detector_model=settings.model_pack,
            embedding_model=settings.model_pack,
            bbox=_bbox_to_dict(face.bbox),
            landmarks=face.landmarks.tolist(),
            quality_score=None,
            status="active",
        )
        self._db.add(sample)
        await self._db.flush()

        await self._vector_store.upsert_batch(
            [
                models.PointStruct(
                    id=str(sample.sample_id),
                    vector=face.embedding.tolist(),
                    payload={
                        "sampleId": str(sample.sample_id),
                        "photoId": str(photo.photo_id),
                        "personId": str(person.person_id),
                        "active": True,
                        "modelVersion": settings.model_pack,
                    },
                )
            ]
        )

        await self._db.commit()

        return {
            "face_id": person.face_identity_id,
            "person_id": person.person_id,
            "photo_id": photo.photo_id,
            "sample_id": sample.sample_id,
            "status": "active",
            "name": f"{person.first_name} {person.last_name}".strip(),
            "created_at": sample.created_at,
        }

    async def delete_person_photo(
        self,
        face_id: uuid.UUID,
        photo_id: uuid.UUID,
    ) -> None:
        """Soft-delete one photo and its face sample for a person."""
        person_result = await self._db.execute(
            select(Person).where(Person.face_identity_id == face_id)
        )
        person = person_result.scalar_one_or_none()
        if person is None:
            raise NotFoundError(f"person with face {face_id} not found")

        photo_result = await self._db.execute(
            select(PersonPhoto).where(
                PersonPhoto.photo_id == photo_id,
                PersonPhoto.person_id == person.person_id,
            )
        )
        photo = photo_result.scalar_one_or_none()
        if photo is None:
            raise NotFoundError(f"photo {photo_id} not found")

        now = self._utc_now()
        photo.status = "deleted"
        photo.deleted_at = now

        sample_result = await self._db.execute(
            select(FaceSample).where(
                FaceSample.photo_id == photo_id,
                FaceSample.person_id == person.person_id,
            )
        )
        sample = sample_result.scalar_one_or_none()
        if sample is not None:
            sample.status = "deleted"
            await self._vector_store.set_active(sample.sample_id, active=False)

        await self._db.commit()

    async def _persist_sub_chunk(
        self,
        indices: list[int],
        items: list[BulkEnrollItem],
        faces: list[GpuFaceExtraction | None],
    ) -> tuple[list[dict[str, Any]], list[tuple[int, str]], float]:
        """Upload, insert, and index one sub-chunk of faces.

        Returns ``(records, errors, io_ms)``.
        """
        successful: list[tuple[int, BulkEnrollItem, GpuFaceExtraction]] = []
        errors: list[tuple[int, str]] = []
        for idx, it, face in zip(indices, items, faces):
            if face is None:
                errors.append((idx, "no face detected"))
            else:
                successful.append((idx, it, face))

        if not successful:
            return [], errors, 0.0

        t_io0 = time.perf_counter()
        n = len(successful)
        person_ids = [new_uuid7() for _ in range(n)]
        photo_ids = [new_uuid7() for _ in range(n)]
        sample_ids = [new_uuid7() for _ in range(n)]
        object_keys = [f"enrollments/{photo_id}" for photo_id in photo_ids]

        # Limit Minio upload concurrency to avoid connection thrashing.
        upload_concurrency = 32
        upload_tasks = [
            self._storage.put_object(
                object_key=object_keys[i],
                data=io.BytesIO(successful[i][1].image_bytes),
                length=len(successful[i][1].image_bytes),
                content_type="application/octet-stream",
            )
            for i in range(n)
        ]
        for start in range(0, len(upload_tasks), upload_concurrency):
            await asyncio.gather(
                *upload_tasks[start : start + upload_concurrency]
            )

        now = self._utc_now()
        face_identity_rows: list[dict[str, Any]] = []
        person_rows: list[dict[str, Any]] = []
        photo_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        qdrant_points: list[models.PointStruct] = []

        for i, (original_idx, it, face) in enumerate(successful):
            first_name, last_name = self._split_name(it.name)
            national_hmac = hash_national_id(it.national_id, settings.hmac_key)
            national_masked = mask_national_id(it.national_id)
            content_sha256 = hashlib.sha256(it.image_bytes).hexdigest()
            face_identity_id = derive_face_identity_id(national_hmac)

            face_identity_rows.append(
                {
                    "face_identity_id": face_identity_id,
                    "identity_lookup_hmac": national_hmac,
                    "display_name": f"{first_name} {last_name}".strip(),
                    "external_identity_hash": national_hmac,
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            person_rows.append(
                {
                    "person_id": person_ids[i],
                    "face_identity_id": face_identity_id,
                    "first_name": first_name,
                    "last_name": last_name,
                    "national_id_lookup_hmac": national_hmac,
                    "national_id_masked": national_masked,
                    "details": it.metadata or {},
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            photo_rows.append(
                {
                    "photo_id": photo_ids[i],
                    "person_id": person_ids[i],
                    "object_key": object_keys[i],
                    "content_sha256": content_sha256,
                    "mime_type": "application/octet-stream",
                    "width": 0,
                    "height": 0,
                    "status": "active",
                    "created_at": now,
                    "updated_at": now,
                }
            )
            sample_rows.append(
                {
                    "sample_id": sample_ids[i],
                    "person_id": person_ids[i],
                    "photo_id": photo_ids[i],
                    "detector_model": settings.model_pack,
                    "embedding_model": settings.model_pack,
                    "bbox": _bbox_to_dict(face.bbox),
                    "landmarks": face.landmarks.tolist(),
                    "quality_score": float(face.score),
                    "status": "active",
                    "created_at": now,
                    "updated_at": now,
                }
            )
            qdrant_points.append(
                models.PointStruct(
                    id=str(sample_ids[i]),
                    vector=face.embedding.tolist(),
                    payload={
                        "sampleId": str(sample_ids[i]),
                        "photoId": str(photo_ids[i]),
                        "personId": str(person_ids[i]),
                        "active": True,
                        "modelVersion": settings.model_pack,
                    },
                )
            )

        from sqlalchemy.dialects.postgresql import insert as pg_insert

        if face_identity_rows:
            await self._db.execute(
                pg_insert(FaceIdentity)
                .values(face_identity_rows)
                .on_conflict_do_nothing(
                    index_elements=["identity_lookup_hmac"]
                )
            )
        await self._db.execute(sa.insert(Person), person_rows)
        await self._db.execute(sa.insert(PersonPhoto), photo_rows)
        await self._db.execute(sa.insert(FaceSample), sample_rows)

        # Parallel Qdrant upserts to hide index latency.
        qdrant_chunk_size = 256
        qdrant_chunks = [
            qdrant_points[offset : offset + qdrant_chunk_size]
            for offset in range(0, len(qdrant_points), qdrant_chunk_size)
        ]
        await asyncio.gather(
            *(self._vector_store.upsert_batch(c) for c in qdrant_chunks)
        )

        await self._db.commit()
        io_ms = (time.perf_counter() - t_io0) * 1000

        records: list[dict[str, Any]] = []
        for i, (original_idx, it, face) in enumerate(successful):
            records.append(
                {
                    "index": original_idx,
                    "face_id": face_identity_rows[i]["face_identity_id"],
                    "person_id": person_ids[i],
                    "photo_id": photo_ids[i],
                    "status": "known",
                    "name": f"{person_rows[i]['first_name']} {person_rows[i]['last_name']}".strip(),
                }
            )

        return records, errors, io_ms

    async def bulk_enroll(
        self,
        items: list[BulkEnrollItem],
        *,
        extract_chunk_size: int = 1024,
    ) -> tuple[list[dict[str, Any]], list[tuple[int, str]], dict[str, float]]:
        """Enroll many faces with extraction and IO pipelined.

        Returns ``(enrolled_records, errors)`` where errors are ``(index, message)``
        pairs aligned with the input list.
        """
        if not items:
            return [], [], {"extraction_ms": 0.0, "io_ms": 0.0}

        errors: list[tuple[int, str]] = []
        valid_indices: list[int] = []
        valid_items: list[BulkEnrollItem] = []
        for idx, it in enumerate(items):
            if not it.image_bytes:
                errors.append((idx, "empty image"))
                continue
            if not it.name or not it.national_id:
                errors.append((idx, "name and national_id are required"))
                continue
            valid_indices.append(idx)
            valid_items.append(it)

        if not valid_items:
            return [], errors, {"extraction_ms": 0.0, "io_ms": 0.0}

        queue: asyncio.Queue[
            tuple[list[int], list[BulkEnrollItem], list[GpuFaceExtraction | None]] | None
        ] = asyncio.Queue(maxsize=8)
        records_all: list[dict[str, Any]] = []
        errors_all: list[tuple[int, str]] = list(errors)
        total_extract_ms: list[float] = [0.0]
        total_io_ms: list[float] = [0.0]

        async def extractor() -> None:
            for start in range(0, len(valid_items), extract_chunk_size):
                sub_indices = valid_indices[start : start + extract_chunk_size]
                sub_items = valid_items[start : start + extract_chunk_size]
                t0 = time.perf_counter()
                faces = await self._extract_batch_faces(
                    [it.image_bytes for it in sub_items]
                )
                total_extract_ms[0] += (time.perf_counter() - t0) * 1000
                await queue.put((sub_indices, sub_items, faces))
            await queue.put(None)

        async def persister() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    break
                sub_indices, sub_items, faces = item
                recs, errs, io_ms = await self._persist_sub_chunk(
                    sub_indices, sub_items, faces
                )
                records_all.extend(recs)
                errors_all.extend(errs)
                total_io_ms[0] += io_ms

        await asyncio.gather(extractor(), persister())

        return records_all, errors_all, {
            "extraction_ms": total_extract_ms[0],
            "io_ms": total_io_ms[0],
        }

    async def get_face(self, face_id: uuid.UUID) -> dict[str, Any]:
        """Return persisted metadata for an enrolled person (face_id == face_identity_id)."""
        result = await self._db.execute(
            select(Person)
            .options(
                selectinload(Person.samples),
                selectinload(Person.photos),
            )
            .where(Person.face_identity_id == face_id)
            .where(Person.is_active.is_(True))
        )
        person = result.scalar_one_or_none()
        if person is None:
            raise NotFoundError(f"face {face_id} not found")

        primary_sample = next(
            (s for s in person.samples if s.status == "active"),
            None,
        )
        photos = [
            {
                "photo_id": photo.photo_id,
                "status": photo.status,
                "created_at": photo.created_at,
            }
            for photo in sorted(
                (p for p in person.photos if p.status != "deleted"),
                key=lambda p: p.created_at,
                reverse=True,
            )
        ]
        return {
            "face_id": person.face_identity_id,
            "person_id": person.person_id,
            "photo_id": primary_sample.photo_id if primary_sample else None,
            "name": f"{person.first_name} {person.last_name}".strip(),
            "national_id_masked": person.national_id_masked,
            "status": "active" if primary_sample else "deleted",
            "bounding_box": primary_sample.bbox if primary_sample else None,
            "landmarks": primary_sample.landmarks if primary_sample else None,
            "metadata": person.details,
            "created_at": primary_sample.created_at if primary_sample else person.created_at,
            "photos": photos,
        }

    async def list_faces(
        self,
        *,
        search: str | None,
        is_active: bool | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        """Return a paginated, searchable list of enrolled *persons*.

        Each row represents one person with a count of their enrolled photos.
        The canonical ``face_id`` is the person's ``face_identity_id``.
        """
        filters: list[sa.ColumnElement[bool]] = [
            Person.is_active.is_(True),
        ]
        if is_active is not None:
            filters.append(
                Person.is_active.is_(is_active)
            )

        if search:
            pattern = f"%{search}%"
            name_concat = sa.func.concat(Person.first_name, " ", Person.last_name)
            filters.append(
                sa.or_(
                    name_concat.ilike(pattern),
                    FaceIdentity.display_name.ilike(pattern),
                )
            )

        active_photo_join = sa.and_(
            Person.person_id == PersonPhoto.person_id,
            PersonPhoto.status == "active",
        )

        base = (
            select(
                Person.person_id,
                Person.face_identity_id,
                Person.first_name,
                Person.last_name,
                Person.national_id_masked,
                Person.created_at,
                sa.func.min(sa.cast(PersonPhoto.photo_id, sa.String)).label(
                    "primary_photo_id"
                ),
                sa.func.count(sa.func.distinct(PersonPhoto.photo_id)).label(
                    "photo_count"
                ),
            )
            .join(FaceIdentity, Person.face_identity_id == FaceIdentity.face_identity_id)
            .outerjoin(PersonPhoto, active_photo_join)
            .group_by(
                Person.person_id,
                Person.face_identity_id,
                Person.first_name,
                Person.last_name,
                Person.national_id_masked,
                Person.created_at,
            )
            .having(sa.func.count(sa.func.distinct(PersonPhoto.photo_id)) > 0)
        )
        count_stmt = (
            sa.select(sa.func.count(sa.distinct(Person.person_id)))
            .select_from(Person)
            .where(Person.is_active.is_(True))
            .where(
                sa.exists().where(
                    PersonPhoto.person_id == Person.person_id
                ).where(
                    PersonPhoto.status == "active"
                )
            )
        )

        if filters:
            predicate = sa.and_(*filters)
            base = base.where(predicate)
            count_stmt = count_stmt.where(predicate)
            if search:
                count_stmt = count_stmt.join(
                    FaceIdentity, Person.face_identity_id == FaceIdentity.face_identity_id
                )

        stmt = (
            base.order_by(Person.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await self._db.execute(stmt)
        rows = result.mappings().all()
        total_result = await self._db.execute(count_stmt)
        total = total_result.scalar() or 0

        items = [
            {
                "face_id": row["face_identity_id"],
                "person_id": row["person_id"],
                "photo_id": row["primary_photo_id"],
                "name": f"{row['first_name']} {row['last_name']}".strip(),
                "national_id_masked": row["national_id_masked"],
                "status": "active" if row["photo_count"] else "deleted",
                "created_at": row["created_at"],
                "photo_count": row["photo_count"],
            }
            for row in rows
        ]

        return {"items": items, "total": total, "limit": limit, "offset": offset}

    async def delete_face(self, face_id: uuid.UUID) -> None:
        """Soft-delete an enrolled person and all of their vectors/photos."""
        result = await self._db.execute(
            select(Person)
            .where(Person.face_identity_id == face_id)
            .where(Person.is_active.is_(True))
        )
        person = result.scalar_one_or_none()
        if person is None:
            raise NotFoundError(f"face {face_id} not found")

        now = self._utc_now()
        stmt = await self._db.execute(
            select(FaceSample, PersonPhoto)
            .join(PersonPhoto, FaceSample.photo_id == PersonPhoto.photo_id)
            .where(FaceSample.person_id == person.person_id)
            .where(FaceSample.status != "deleted")
        )
        samples_to_deactivate: list[uuid.UUID] = []
        for sample, photo in stmt.tuples().all():
            sample.status = "deleted"
            photo.status = "deleted"
            photo.deleted_at = now
            samples_to_deactivate.append(sample.sample_id)

        for sample_id in samples_to_deactivate:
            await self._vector_store.set_active(sample_id, active=False)

        person.is_active = False
        person.deleted_at = now

        await self._db.commit()

    async def get_face_history(
        self,
        face_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        """Return recognition process history where this person was the best match."""
        person_result = await self._db.execute(
            select(Person.person_id).where(Person.face_identity_id == face_id)
        )
        person_id = person_result.scalar_one_or_none()
        if person_id is None:
            return []

        result = await self._db.execute(
            select(RecognitionResult, RecognitionRequest)
            .join(
                RecognitionRequest,
                RecognitionResult.request_id == RecognitionRequest.request_id,
            )
            .where(RecognitionResult.best_person_id == person_id)
            .where(RecognitionResult.recognition_status == "known")
            .order_by(RecognitionRequest.created_at.desc())
        )
        entries: list[dict[str, Any]] = []
        for res, req in result.tuples().all():
            entries.append(
                {
                    "process_id": req.request_id,
                    "status": req.status,
                    "timestamp": req.created_at,
                }
            )
        return entries

    async def get_process(self, process_id: uuid.UUID) -> dict[str, Any]:
        """Return a recognition process and its per-face results."""
        result = await self._db.execute(
            select(RecognitionRequest).where(RecognitionRequest.request_id == process_id)
        )
        request = result.scalar_one_or_none()
        if request is None:
            raise NotFoundError(f"process {process_id} not found")

        results = await self._db.execute(
            select(RecognitionResult)
            .where(RecognitionResult.request_id == process_id)
            .order_by(RecognitionResult.face_index)
        )
        result_rows = results.scalars().all()

        person_ids = {
            res.best_person_id
            for res in result_rows
            if res.recognition_status == "known" and res.best_person_id is not None
        }
        person_identity_map: dict[uuid.UUID, uuid.UUID] = {}
        if person_ids:
            person_rows = await self._db.execute(
                select(Person.person_id, Person.face_identity_id).where(
                    Person.person_id.in_(person_ids)
                )
            )
            person_identity_map = {pid: fid for pid, fid in person_rows}

        faces: list[dict[str, Any]] = []
        for res in result_rows:
            face_id = None
            if res.recognition_status == "known" and res.best_person_id is not None:
                face_id = person_identity_map.get(res.best_person_id)
            faces.append(
                {
                    "face_index": res.face_index,
                    "status": res.recognition_status,
                    "face_id": face_id,
                    "score": res.best_score,
                    "bounding_box": res.bbox,
                }
            )

        return {
            "process_id": request.request_id,
            "status": request.status,
            "face_count": request.face_count,
            "created_at": request.created_at,
            "completed_at": request.completed_at,
            "faces": faces,
        }

    def _build_recognized_face(
        self,
        *,
        face_index: int,
        extraction: Any,
        is_known: bool,
        best: SearchHit | None,
        candidates: list[SearchHit],
        face_id: uuid.UUID | None = None,
        name: str | None,
        metadata: dict[str, Any] | None,
        person_name_map: dict[uuid.UUID, str] | None = None,
        person_identity_map: dict[uuid.UUID, uuid.UUID] | None = None,
    ) -> dict[str, Any]:
        if person_name_map is None:
            person_name_map = {}
        if person_identity_map is None:
            person_identity_map = {}
        return {
            "face_index": face_index,
            "face_id": face_id,
            "person_id": best.person_id if is_known else None,
            "photo_id": best.photo_id if is_known else None,
            "status": "known" if is_known else "unknown",
            "name": name,
            "metadata": metadata,
            "bounding_box": _bbox_to_dict(extraction.bbox),
            "landmarks": extraction.landmarks.tolist(),
            "confidence": best.score if is_known else None,
            "candidates": [
                {
                    **_hit_to_dict(hit),
                    "face_id": person_identity_map.get(hit.person_id, hit.sample_id),
                    "name": person_name_map.get(hit.person_id),
                }
                for hit in candidates
            ],
        }


def _bbox_to_dict(bbox: np.ndarray) -> dict[str, float]:
    arr = np.asarray(bbox, dtype=float).ravel()
    return {"x1": float(arr[0]), "y1": float(arr[1]), "x2": float(arr[2]), "y2": float(arr[3])}


def _bbox_area(bbox: np.ndarray) -> float:
    arr = np.asarray(bbox, dtype=float).ravel()
    return max(0.0, float(arr[2]) - float(arr[0])) * max(
        0.0, float(arr[3]) - float(arr[1])
    )


def _hit_to_dict(hit: SearchHit) -> dict[str, Any]:
    return {
        "face_id": str(hit.sample_id),
        "person_id": str(hit.person_id),
        "photo_id": str(hit.photo_id),
        "score": float(hit.score),
    }
