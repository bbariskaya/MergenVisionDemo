"""Idempotent, person-grouped bulk enrollment used by worker processes.

The service:
- builds a deterministic manifest from LFW-style folders
- shards by ``person_id`` so each GPU process owns disjoint identities
- extracts faces in batches, then persists each photo only if its deterministic
  ``sample_id`` is not already present
- records durable Qdrant upserts with ``wait=True``
- writes a ``process_record`` / ``process_event`` audit trail with no raw PII
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert

import numpy as np
import sqlalchemy as sa
from qdrant_client import models
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.ids import (
    derive_face_identity_id,
    derive_person_id,
    derive_photo_id,
    derive_sample_id,
    identity_hmac,
    new_uuid7,
)
from app.domain.models import (
    FaceIdentity,
    FaceSample,
    Person,
    PersonPhoto,
    ProcessEvent,
    ProcessRecord,
)
from app.infrastructure.minio import PhotoStorage
from app.infrastructure.qdrant import FaceVectorStore
from app.ml.gpu.face_pipeline import GpuFaceExtraction, GpuFacePipeline
from app.services.bulk_manifest import EnrollmentIdentity, EnrollmentPhoto

logger = logging.getLogger(__name__)


class EnrollmentCancelled(Exception):
    """Raised when the running enrollment shard is cooperatively cancelled."""


@dataclass(frozen=True)
class _PhotoTask:
    identity: EnrollmentIdentity
    photo: EnrollmentPhoto
    index: int


@dataclass(frozen=True)
class _StagedFace:
    identity: EnrollmentIdentity
    photo: EnrollmentPhoto
    person_id: uuid.UUID
    photo_id: uuid.UUID
    sample_id: uuid.UUID
    face: GpuFaceExtraction


@dataclass
class _ShardResult:
    identities: int = 0
    photos: int = 0
    faces_enrolled: int = 0
    faces_duplicate: int = 0
    no_face: int = 0
    errors: int = 0
    total_scanned: int = 0
    total_processed: int = 0
    extraction_ms: float = 0.0
    io_ms: float = 0.0
    messages: list[str] = field(default_factory=list)


class BulkEnrollmentService:
    """Enroll a disjoint shard of identities with idempotent cross-store writes."""

    def __init__(
        self,
        db: AsyncSession,
        storage: PhotoStorage,
        vector_store: FaceVectorStore,
        pipeline: GpuFacePipeline,
        pipeline_lock: asyncio.Lock,
        *,
        gpu_executor: concurrent.futures.Executor | None = None,
        extract_batch_size: int | None = None,
        qdrant_wait: bool = False,
        max_persistence_concurrency: int | None = None,
        activation_batch_size: int | None = None,
    ) -> None:
        self._db = db
        self._storage = storage
        self._vector_store = vector_store
        self._pipeline = pipeline
        self._pipeline_lock = pipeline_lock
        self._gpu_executor = gpu_executor
        self._extract_batch_size = extract_batch_size or settings.bulk_extract_batch_size
        self._activation_batch_size = activation_batch_size or settings.bulk_activation_batch_size
        self._qdrant_wait = qdrant_wait
        self._model_pack = settings.model_pack
        self._persist_semaphore = asyncio.Semaphore(
            max_persistence_concurrency or settings.bulk_max_persistence_concurrency
        )

    def _split_name(self, full_name: str) -> tuple[str, str]:
        parts = full_name.strip().split()
        if not parts:
            return "", ""
        if len(parts) == 1:
            return parts[0], ""
        return " ".join(parts[:-1]), parts[-1]

    async def _extract_batch_faces(
        self, image_bytes_list: list[bytes]
    ) -> list[GpuFaceExtraction | None]:
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

    async def _read_photo_bytes(self, path: Path) -> bytes:
        return await asyncio.to_thread(path.read_bytes)

    async def _ensure_identities(
        self,
        identities: tuple[EnrollmentIdentity, ...],
    ) -> dict[str, uuid.UUID]:
        """Insert missing FaceIdentity / Person rows and return person_id map."""
        hmac_to_identity = {
            identity.identity_hmac: identity for identity in identities
        }
        existing_face_ids = await self._db.execute(
            select(FaceIdentity.identity_lookup_hmac, FaceIdentity.face_identity_id).where(
                FaceIdentity.identity_lookup_hmac.in_(hmac_to_identity.keys())
            )
        )
        existing_hmacs = {row[0] for row in existing_face_ids.all()}

        missing = [
            identity for identity in identities
            if identity.identity_hmac not in existing_hmacs
        ]
        if missing:
            now = _utc_now()
            face_rows = [
                {
                    "face_identity_id": uuid.UUID(identity.face_identity_id),
                    "identity_lookup_hmac": identity.identity_hmac,
                    "display_name": identity.display_name,
                    "external_identity_hash": identity.identity_hmac,
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
                for identity in missing
            ]
            person_rows = [
                {
                    "person_id": uuid.UUID(identity.person_id),
                    "face_identity_id": uuid.UUID(identity.face_identity_id),
                    "first_name": self._split_name(identity.display_name)[0],
                    "last_name": self._split_name(identity.display_name)[1],
                    "national_id_lookup_hmac": identity.identity_hmac,
                    "national_id_masked": _mask_hmac(identity.identity_hmac),
                    "details": {},
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
                for identity in missing
            ]
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            await self._db.execute(
                pg_insert(FaceIdentity)
                .values(face_rows)
                .on_conflict_do_nothing(index_elements=["identity_lookup_hmac"])
            )
            await self._db.execute(
                pg_insert(Person)
                .values(person_rows)
                .on_conflict_do_nothing(index_elements=["national_id_lookup_hmac"])
            )
            await self._db.flush()

        return {identity.identity_hmac: uuid.UUID(identity.person_id) for identity in identities}

    async def _stage_face(
        self,
        identity: EnrollmentIdentity,
        photo: EnrollmentPhoto,
        person_id: uuid.UUID,
        face: GpuFaceExtraction,
    ) -> tuple[_StagedFace | None, bool]:
        """Stage one face for idempotent cross-store activation.

        Returns the staged descriptor, or ``(None, True)`` when the sample is
        already active (duplicate).
        """
        photo_id = uuid.UUID(photo.photo_id)
        sample_id = derive_sample_id(photo_id, self._model_pack)

        existing = await self._db.get(FaceSample, sample_id)
        if existing is not None and existing.status == "active":
            return None, True

        async with self._persist_semaphore:
            object_key = f"enrollments/{person_id}/{photo_id}"
            photo_exists = await self._db.get(PersonPhoto, photo_id)
            if photo_exists is None:
                if not await self._storage.object_exists(object_key):
                    data = await self._read_photo_bytes(photo.path)
                    await self._storage.put_object(
                        object_key=object_key,
                        data=__import__("io").BytesIO(data),
                        length=len(data),
                        content_type="application/octet-stream",
                    )
                now = _utc_now()
                self._db.add(
                    PersonPhoto(
                        photo_id=photo_id,
                        person_id=person_id,
                        object_key=object_key,
                        content_sha256=photo.content_sha256,
                        mime_type="application/octet-stream",
                        width=0,
                        height=0,
                        status="staged",
                        created_at=now,
                        updated_at=now,
                    )
                )

            if existing is None:
                self._db.add(
                    FaceSample(
                        sample_id=sample_id,
                        person_id=person_id,
                        photo_id=photo_id,
                        detector_model=self._model_pack,
                        embedding_model=self._model_pack,
                        bbox=_bbox_to_dict(face.bbox),
                        landmarks=face.landmarks.tolist(),
                        quality_score=float(face.score),
                        status="staged",
                        created_at=_utc_now(),
                        updated_at=_utc_now(),
                    )
                )
            else:
                existing.status = "staged"

        return (
            _StagedFace(
                identity=identity,
                photo=photo,
                person_id=person_id,
                photo_id=photo_id,
                sample_id=sample_id,
                face=face,
            ),
            False,
        )

    async def _stage_batch(
        self,
        tasks_faces: list[tuple[_PhotoTask, GpuFaceExtraction]],
        person_id_map: dict[str, uuid.UUID],
        result: _ShardResult,
        staged_buffer: list[_StagedFace],
    ) -> None:
        """Stage a chunk with one batched duplicate lookup and conflict-safe upserts."""
        work: list[tuple[_PhotoTask, GpuFaceExtraction, uuid.UUID, uuid.UUID, uuid.UUID]] = []
        for task, face in tasks_faces:
            person_id = person_id_map[task.identity.identity_hmac]
            photo_id = uuid.UUID(task.photo.photo_id)
            sample_id = derive_sample_id(photo_id, self._model_pack)
            work.append((task, face, person_id, photo_id, sample_id))

        photo_ids = [photo_id for _, _, _, photo_id, _ in work]

        # Single batched lookup: PostgreSQL is the hot-path duplicate authority.
        existing_photos: set[uuid.UUID] = set()
        active_samples: set[uuid.UUID] = set()
        existing_samples: set[uuid.UUID] = set()

        photo_rows = await self._db.execute(
            select(PersonPhoto.photo_id).where(PersonPhoto.photo_id.in_(photo_ids))
        )
        existing_photos = {r[0] for r in photo_rows.all()}

        sample_rows = await self._db.execute(
            select(FaceSample.photo_id, FaceSample.sample_id, FaceSample.status).where(
                FaceSample.photo_id.in_(photo_ids)
            )
        )
        for photo_id, sample_id, status in sample_rows.all():
            existing_samples.add(photo_id)
            if status == "active":
                active_samples.add(photo_id)

        # Partition work into duplicates and new/staged faces.
        to_stage = []
        for task, face, person_id, photo_id, sample_id in work:
            if photo_id in active_samples:
                result.faces_duplicate += 1
                continue
            to_stage.append((task, face, person_id, photo_id, sample_id))

        if not to_stage:
            return

        async def _upload_photo(
            task: _PhotoTask,
            person_id: uuid.UUID,
            photo_id: uuid.UUID,
        ) -> None:
            if photo_id in existing_photos:
                return
            async with self._persist_semaphore:
                object_key = f"enrollments/{person_id}/{photo_id}"
                data = await self._read_photo_bytes(task.photo.path)
                await self._storage.put_object(
                    object_key=object_key,
                    data=__import__("io").BytesIO(data),
                    length=len(data),
                    content_type="application/octet-stream",
                )

        await asyncio.gather(*(_upload_photo(task, person_id, photo_id) for task, _, person_id, photo_id, _ in to_stage))

        now = _utc_now()
        photo_rows = [
            {
                "photo_id": photo_id,
                "person_id": person_id,
                "object_key": f"enrollments/{person_id}/{photo_id}",
                "content_sha256": task.photo.content_sha256,
                "mime_type": "application/octet-stream",
                "width": 0,
                "height": 0,
                "status": "staged",
                "created_at": now,
                "updated_at": now,
            }
            for task, _, person_id, photo_id, _ in to_stage
        ]
        sample_rows = [
            {
                "sample_id": sample_id,
                "person_id": person_id,
                "photo_id": photo_id,
                "detector_model": self._model_pack,
                "embedding_model": self._model_pack,
                "bbox": _bbox_to_dict(face.bbox),
                "landmarks": face.landmarks.tolist(),
                "quality_score": float(face.score),
                "status": "staged",
                "created_at": now,
                "updated_at": now,
            }
            for _, face, person_id, photo_id, sample_id in to_stage
        ]

        if photo_rows:
            await self._db.execute(
                pg_insert(PersonPhoto)
                .values(photo_rows)
                .on_conflict_do_nothing(index_elements=["photo_id"])
            )
        if sample_rows:
            await self._db.execute(
                pg_insert(FaceSample)
                .values(sample_rows)
                .on_conflict_do_update(
                    index_elements=["photo_id"],
                    set_={
                        "status": "staged",
                        "detector_model": self._model_pack,
                        "embedding_model": self._model_pack,
                        "bbox": pg_insert(FaceSample).excluded.bbox,
                        "landmarks": pg_insert(FaceSample).excluded.landmarks,
                        "quality_score": pg_insert(FaceSample).excluded.quality_score,
                        "updated_at": now,
                    },
                )
            )

        for task, face, person_id, photo_id, sample_id in to_stage:
            result.faces_enrolled += 1
            staged_buffer.append(
                _StagedFace(
                    identity=task.identity,
                    photo=task.photo,
                    person_id=person_id,
                    photo_id=photo_id,
                    sample_id=sample_id,
                    face=face,
                )
            )

    async def _activate_buffer(self, staged: list[_StagedFace]) -> None:
        """Promote a buffered set of staged faces to active in PG and Qdrant.

        Uses DML and a single batched Qdrant upsert (active=True) to minimize
        per-photo round-trips.
        """
        if not staged:
            return
        now = _utc_now()
        photo_ids = [item.photo_id for item in staged]
        await self._db.execute(
            update(PersonPhoto)
            .where(PersonPhoto.photo_id.in_(photo_ids))
            .values(status="active", updated_at=now)
        )
        await self._db.execute(
            update(FaceSample)
            .where(FaceSample.photo_id.in_(photo_ids))
            .values(status="active", updated_at=now)
        )
        qdrant_points = [
            models.PointStruct(
                id=str(item.sample_id),
                vector=item.face.embedding.tolist(),
                payload={
                    "sampleId": str(item.sample_id),
                    "photoId": str(item.photo_id),
                    "personId": str(item.person_id),
                    "active": True,
                    "modelVersion": self._model_pack,
                },
            )
            for item in staged
        ]
        await self._upsert_qdrant(qdrant_points)

    async def _upsert_qdrant(
        self,
        points: list[models.PointStruct],
    ) -> None:
        if not points:
            return
        await self._vector_store.upsert_batch(points, wait=self._qdrant_wait)

    async def enroll_shard(
        self,
        identities: tuple[EnrollmentIdentity, ...],
        *,
        parent_process_id: uuid.UUID | None = None,
        idempotency_key: str = "",
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
        max_photos: int | None = None,
    ) -> _ShardResult:
        if not identities:
            return _ShardResult()

        result = _ShardResult(identities=len(identities))
        result.photos = sum(len(i.photos) for i in identities)

        summary: dict[str, Any] = {
            "identities": result.identities,
            "photos": result.photos,
        }
        if parent_process_id is not None:
            summary["parent_process_id"] = str(parent_process_id)
        if idempotency_key:
            summary["idempotency_key"] = idempotency_key
        process_record = ProcessRecord(
            process_type="bulk_enroll_shard",
            status="running",
            summary=summary,
        )
        self._db.add(process_record)
        await self._db.flush()

        seq = 0
        self._db.add(
            _event(process_record.process_id, seq, "started", "pending", "running")
        )
        seq += 1

        try:
            person_id_map = await self._ensure_identities(identities)

            identity_photo_counts = {
                identity.identity_hmac: len(identity.photos)
                for identity in identities
            }

            tasks = [
                _PhotoTask(identity=identity, photo=photo, index=idx)
                for identity in identities
                for idx, photo in enumerate(identity.photos)
            ]

            if max_photos is not None and len(tasks) > max_photos:
                tasks = tasks[:max_photos]
                result.photos = len(tasks)
                process_record.summary["photos"] = result.photos

            staged_buffer: list[_StagedFace] = []
            total_batches = (len(tasks) + self._extract_batch_size - 1) // self._extract_batch_size
            prev_no_face = 0
            scanned_counts: dict[str, int] = {}
            last_completed_identity_key: str | None = None

            for start in range(0, len(tasks), self._extract_batch_size):
                chunk = tasks[start : start + self._extract_batch_size]
                chunk_photo_ids = [uuid.UUID(t.photo.photo_id) for t in chunk]

                # Pre-filter already-active duplicates so they skip file I/O and ML.
                active_rows = await self._db.execute(
                    select(FaceSample.photo_id).where(
                        FaceSample.photo_id.in_(chunk_photo_ids),
                        FaceSample.status == "active",
                    )
                )
                active_photo_ids = {r[0] for r in active_rows.all()}
                if active_photo_ids:
                    result.faces_duplicate += len(active_photo_ids)
                    chunk = [t for t in chunk if uuid.UUID(t.photo.photo_id) not in active_photo_ids]
                    chunk_photo_ids = [uuid.UUID(t.photo.photo_id) for t in chunk]

                image_bytes = await asyncio.gather(
                    *(self._read_photo_bytes(t.photo.path) for t in chunk)
                )

                t0 = time.perf_counter()
                faces = await self._extract_batch_faces(image_bytes)
                result.extraction_ms += (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                with_faces = [
                    (task, face)
                    for task, face in zip(chunk, faces)
                    if face is not None
                ]
                result.no_face += len(chunk) - len(with_faces)
                if with_faces:
                    await self._stage_batch(
                        with_faces, person_id_map, result, staged_buffer
                    )

                # Update identity-level checkpoint after scanning this batch.
                for task in chunk:
                    key = task.identity.identity_hmac
                    scanned_counts[key] = scanned_counts.get(key, 0) + 1
                    if scanned_counts[key] >= identity_photo_counts.get(key, 1):
                        last_completed_identity_key = task.identity.identity_key

                if len(staged_buffer) >= self._activation_batch_size:
                    await self._db.flush()
                    await self._activate_buffer(staged_buffer)
                    staged_buffer.clear()

                result.total_scanned += len(chunk)
                result.total_processed += len(with_faces) + result.no_face - prev_no_face
                prev_no_face = result.no_face

                result.io_ms += (time.perf_counter() - t0) * 1000

                progress = {
                    "identities": result.identities,
                    "photos": result.photos,
                    "faces_enrolled": result.faces_enrolled,
                    "faces_duplicate": result.faces_duplicate,
                    "no_face": result.no_face,
                    "errors": result.errors,
                    "total_scanned": result.total_scanned,
                    "total_processed": result.total_processed,
                    "batches_completed": start // self._extract_batch_size + 1,
                    "total_batches": total_batches,
                    "extraction_ms": result.extraction_ms,
                    "io_ms": result.io_ms,
                    "last_completed_identity_key": last_completed_identity_key,
                }
                process_record.summary["progress"] = progress
                if progress_callback is not None:
                    try:
                        await progress_callback(progress)
                    except Exception:
                        logger.exception("progress callback failed")
                await self._db.commit()

                if cancel_check is not None and cancel_check():
                    break

            if staged_buffer:
                await self._db.flush()
                await self._activate_buffer(staged_buffer)
                staged_buffer.clear()
                await self._db.commit()

            process_record.status = "completed"
            process_record.completed_at = _utc_now()
            process_record.summary.update(
                {
                    "faces_enrolled": result.faces_enrolled,
                    "faces_duplicate": result.faces_duplicate,
                    "no_face": result.no_face,
                    "errors": result.errors,
                    "last_completed_identity_key": last_completed_identity_key,
                }
            )
            self._db.add(
                _event(
                    process_record.process_id,
                    seq,
                    "completed",
                    "running",
                    "completed",
                    message=f"enrolled {result.faces_enrolled} faces ({result.faces_duplicate} duplicates)",
                )
            )
            await self._db.commit()
            return result

        except EnrollmentCancelled:
            process_record.status = "cancelled"
            process_record.completed_at = _utc_now()
            process_record.error_message = "cancelled by operator"
            self._db.add(
                _event(
                    process_record.process_id,
                    seq,
                    "cancelled",
                    "running",
                    "cancelled",
                    message=f"cancelled after {result.faces_enrolled} enrolled, {result.faces_duplicate} duplicates",
                )
            )
            await self._db.commit()
            raise

        except Exception as exc:
            process_record.status = "failed"
            process_record.completed_at = _utc_now()
            process_record.error_message = str(exc)[:500]
            self._db.add(
                _event(
                    process_record.process_id,
                    seq,
                    "failed",
                    "running",
                    "failed",
                    message=f"failure: {exc.__class__.__name__}",
                )
            )
            await self._db.commit()
            raise

    async def create_parent_process(self, num_shards: int, total_photos: int) -> uuid.UUID:
        record = ProcessRecord(
            process_type="bulk_enroll",
            status="running",
            summary={"num_shards": num_shards, "total_photos": total_photos},
        )
        self._db.add(record)
        await self._db.flush()
        return record.process_id

    async def finalize_parent_process(
        self,
        process_id: uuid.UUID,
        shard_results: list[_ShardResult],
    ) -> None:
        record = await self._db.get(ProcessRecord, process_id)
        if record is None:
            raise RuntimeError(f"parent process {process_id} not found")
        total_enrolled = sum(r.faces_enrolled for r in shard_results)
        total_duplicate = sum(r.faces_duplicate for r in shard_results)
        total_errors = sum(r.errors for r in shard_results)
        record.status = "completed" if total_errors == 0 else "failed"
        record.completed_at = _utc_now()
        record.summary.update(
            {
                "shards": len(shard_results),
                "faces_enrolled": total_enrolled,
                "faces_duplicate": total_duplicate,
                "errors": total_errors,
            }
        )
        await self._db.commit()


def _utc_now() -> __import__("datetime").datetime:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _mask_hmac(hmac_hex: str) -> str:
    """Mask an identity HMAC to fit Person.national_id_masked (varchar(32))."""
    visible = hmac_hex[-8:] if len(hmac_hex) >= 8 else hmac_hex
    return "*" * min(24, len(hmac_hex) - len(visible)) + visible


def _bbox_to_dict(bbox: np.ndarray) -> dict[str, float]:
    arr = np.asarray(bbox, dtype=float).ravel()
    return {
        "x1": float(arr[0]),
        "y1": float(arr[1]),
        "x2": float(arr[2]),
        "y2": float(arr[3]),
    }


def _event(
    process_id: uuid.UUID,
    sequence: int,
    event_type: str,
    status_before: str | None,
    status_after: str | None,
    message: str = "",
    details: dict[str, Any] | None = None,
) -> ProcessEvent:
    return ProcessEvent(
        process_id=process_id,
        sequence=sequence,
        event_type=event_type,
        status_before=status_before,
        status_after=status_after,
        message=message,
        details=details or {},
    )
