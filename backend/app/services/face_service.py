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
from app.core.ids import new_uuid7
from app.core.security import hash_national_id, mask_national_id
from app.domain.models import (
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
            if is_known:
                person_result = await self._db.execute(
                    select(Person).where(Person.person_id == best.person_id)
                )
                person = person_result.scalar_one_or_none()
                if person is not None:
                    known_name = f"{person.first_name} {person.last_name}".strip()
                    known_metadata = person.details or None

            face_dict = self._build_recognized_face(
                face_index=idx,
                extraction=face,
                is_known=is_known,
                best=best,
                candidates=candidates,
                name=known_name,
                metadata=known_metadata,
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

        person_result = await self._db.execute(
            select(Person).where(Person.national_id_lookup_hmac == national_hmac)
        )
        person = person_result.scalar_one_or_none()

        if person is None:
            person = Person(
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
            "face_id": sample.sample_id,
            "person_id": person.person_id,
            "photo_id": photo.photo_id,
            "status": "known",
            "name": f"{person.first_name} {person.last_name}".strip(),
            "created_at": sample.created_at,
        }

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
        person_rows: list[dict[str, Any]] = []
        photo_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        qdrant_points: list[models.PointStruct] = []

        for i, (original_idx, it, face) in enumerate(successful):
            first_name, last_name = self._split_name(it.name)
            national_hmac = hash_national_id(it.national_id, settings.hmac_key)
            national_masked = mask_national_id(it.national_id)
            content_sha256 = hashlib.sha256(it.image_bytes).hexdigest()

            person_rows.append(
                {
                    "person_id": person_ids[i],
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
                    "face_id": sample_ids[i],
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
        """Return persisted metadata for an enrolled face."""
        result = await self._db.execute(
            select(FaceSample)
            .options(
                selectinload(FaceSample.person).selectinload(Person.photos)
            )
            .where(FaceSample.sample_id == face_id)
            .where(FaceSample.status != "deleted")
        )
        sample = result.scalar_one_or_none()
        if sample is None:
            raise NotFoundError(f"face {face_id} not found")

        person = sample.person
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
            "face_id": sample.sample_id,
            "person_id": sample.person_id,
            "photo_id": sample.photo_id,
            "name": f"{person.first_name} {person.last_name}".strip(),
            "national_id_masked": person.national_id_masked,
            "status": sample.status,
            "bounding_box": sample.bbox,
            "landmarks": sample.landmarks,
            "metadata": person.details,
            "created_at": sample.created_at,
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
        """Return a paginated, searchable list of enrolled faces."""
        filters: list[sa.ColumnElement[bool]] = []
        if search:
            pattern = f"%{search}%"
            filters.append(
                sa.func.concat(Person.first_name, " ", Person.last_name).ilike(
                    pattern
                )
            )
        if is_active is not None:
            filters.append(
                FaceSample.status == ("active" if is_active else "deleted")
            )

        base = select(FaceSample).join(FaceSample.person)
        count_stmt = (
            sa.select(sa.func.count(FaceSample.sample_id))
            .select_from(FaceSample)
            .join(FaceSample.person)
        )
        if filters:
            base = base.where(sa.and_(*filters))
            count_stmt = count_stmt.where(sa.and_(*filters))

        stmt = (
            base.options(selectinload(FaceSample.person))
            .order_by(FaceSample.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await self._db.execute(stmt)
        samples = result.scalars().all()
        total = (await self._db.execute(count_stmt)).scalar_one()

        items: list[dict[str, Any]] = []
        for sample in samples:
            person = sample.person
            items.append(
                {
                    "face_id": sample.sample_id,
                    "person_id": sample.person_id,
                    "photo_id": sample.photo_id,
                    "name": f"{person.first_name} {person.last_name}".strip(),
                    "national_id_masked": person.national_id_masked,
                    "status": sample.status,
                    "created_at": sample.created_at,
                }
            )

        return {"items": items, "total": total, "limit": limit, "offset": offset}

    async def delete_face(self, face_id: uuid.UUID) -> None:
        """Soft-delete an enrolled face and deactivate its vector."""
        result = await self._db.execute(
            select(FaceSample)
            .options(selectinload(FaceSample.person))
            .where(FaceSample.sample_id == face_id)
        )
        sample = result.scalar_one_or_none()
        if sample is None:
            raise NotFoundError(f"face {face_id} not found")

        sample.status = "deleted"
        photo = sample.photo
        photo.status = "deleted"
        photo.deleted_at = self._utc_now()

        await self._vector_store.set_active(sample.sample_id, active=False)

        # If the person has no active samples left, deactivate the person.
        active_count_result = await self._db.execute(
            select(FaceSample)
            .where(FaceSample.person_id == sample.person_id)
            .where(FaceSample.status != "deleted")
        )
        if active_count_result.scalar_one_or_none() is None:
            person = sample.person
            person.is_active = False

        await self._db.commit()

    async def get_face_history(
        self,
        face_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        """Return recognition process history where this face was the best match."""
        result = await self._db.execute(
            select(RecognitionResult, RecognitionRequest)
            .join(
                RecognitionRequest,
                RecognitionResult.request_id == RecognitionRequest.request_id,
            )
            .where(RecognitionResult.best_sample_id == face_id)
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

        faces: list[dict[str, Any]] = []
        for res in results.scalars().all():
            faces.append(
                {
                    "face_index": res.face_index,
                    "status": res.recognition_status,
                    "face_id": res.best_sample_id,
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
        name: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "face_index": face_index,
            "face_id": best.sample_id if is_known else None,
            "status": "known" if is_known else "unknown",
            "name": name,
            "metadata": metadata,
            "bounding_box": _bbox_to_dict(extraction.bbox),
            "landmarks": extraction.landmarks.tolist(),
            "confidence": best.score if is_known else None,
            "candidates": [_hit_to_dict(hit) for hit in candidates],
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
