"""Fast, person-grouped bulk enrollment used by worker processes.

The service preserves the identity contract:

* One dataset folder = exactly one ``Person`` row.
* Each ``Person`` owns one active ``FaceIdentity``.
* Each valid photo in that folder becomes one ``PersonPhoto`` + one ``FaceSample``.
* No per-photo "exists?" / "active?" SELECTs on fresh import.
* Deterministic IDs make cancel/resume idempotent without duplicates.
* Photos and embeddings land in the same MinIO namespace and Qdrant collection
  regardless of source dataset.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import io
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
from qdrant_client import models
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.ids import derive_sample_id
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


class _DecodeError(Exception):
    """Local marker for a photo that could not be read or decoded."""


@dataclass
class _PhotoTask:
    identity: EnrollmentIdentity
    photo: EnrollmentPhoto


@dataclass
class _ShardResult:
    discovered_identities: int = 0
    discovered_photos: int = 0
    processed: int = 0
    enrolled: int = 0
    no_face: int = 0
    decode_error: int = 0
    persistence_error: int = 0
    failed: int = 0
    fatal_error: bool = False
    fatal_code: str | None = None
    fatal_stage: str | None = None
    fatal_message: str | None = None
    extraction_ms: float = 0.0
    io_ms: float = 0.0
    total_batches: int = 0
    messages: list[str] = field(default_factory=list)
    status: str = "completed"
    worker_name: str | None = None


class BulkEnrollmentService:
    """Enroll a disjoint shard of identities deterministically and in bulk."""

    def __init__(
        self,
        db: AsyncSession,
        storage: PhotoStorage,
        vector_store: FaceVectorStore,
        pipeline: GpuFacePipeline,
        pipeline_lock: asyncio.Lock,
        *,
        gpu_executor: concurrent.futures.Executor | None = None,
        io_executor: concurrent.futures.Executor | None = None,
        extract_batch_size: int | None = None,
        qdrant_wait: bool = False,
        max_persistence_concurrency: int | None = None,
    ) -> None:
        self._db = db
        self._storage = storage
        self._vector_store = vector_store
        self._pipeline = pipeline
        self._pipeline_lock = pipeline_lock
        self._gpu_executor = gpu_executor
        self._io_executor = io_executor
        self._extract_batch_size = extract_batch_size or settings.bulk_extract_batch_size
        self._qdrant_wait = qdrant_wait
        self._model_pack = settings.model_pack
        self._embedding_model_version = settings.embedding_model_version
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
                        max_batch=self._extract_batch_size,
                    ),
                )
            else:
                raw = self._pipeline.extract_batch(
                    image_bytes_list,
                    pick_largest=True,
                    max_batch=self._extract_batch_size,
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

    async def _read_photo(self, photo: EnrollmentPhoto) -> bytes:
        if photo.data is not None:
            return photo.data
        loop = asyncio.get_running_loop()
        executor = self._io_executor
        if executor is not None:
            return await loop.run_in_executor(executor, photo.path.read_bytes)
        return await loop.run_in_executor(None, photo.path.read_bytes)

    async def _read_one(self, photo: EnrollmentPhoto) -> bytes:
        try:
            t0 = time.perf_counter()
            data = await self._read_photo(photo)
            return data
        except Exception as exc:
            raise _DecodeError(f"read/decode failed for {photo.path}: {exc}") from exc
        finally:
            pass

    async def _ensure_identities(self, identities: Iterable[EnrollmentIdentity]) -> None:
        """Blind bulk upsert of FaceIdentity and Person rows.

        No SELECTs for existence.  ``ON CONFLICT DO NOTHING`` makes re-runs and
        resume safe because all IDs are deterministic.
        """
        unique = {identity.identity_hmac: identity for identity in identities}
        if not unique:
            return
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
            for identity in unique.values()
        ]
        person_rows = [
            {
                "person_id": uuid.UUID(identity.person_id),
                "face_identity_id": uuid.UUID(identity.face_identity_id),
                "first_name": self._split_name(identity.display_name)[0],
                "last_name": self._split_name(identity.display_name)[1],
                "national_id_lookup_hmac": identity.identity_hmac,
                "national_id_masked": _mask_hmac(identity.identity_hmac),
                "details": {"source_dataset": identity.source_dataset},
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            }
            for identity in unique.values()
        ]
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

    async def _upload_photo(
        self,
        person_id: uuid.UUID,
        photo_id: uuid.UUID,
        data: bytes,
    ) -> str:
        object_key = f"enrollments/{person_id}/{photo_id}"
        async with self._persist_semaphore:
            await self._storage.put_object(
                object_key=object_key,
                data=io.BytesIO(data),
                length=len(data),
                content_type="application/octet-stream",
            )
        return object_key

    async def _persist_batch(
        self,
        items: list[tuple[EnrollmentIdentity, EnrollmentPhoto, GpuFaceExtraction, bytes]],
    ) -> None:
        """Persist one extracted chunk with batched upserts across all stores."""
        if not items:
            return

        now = _utc_now()

        # 1. Ensure identities for everyone in this chunk (batched, no SELECTs).
        await self._ensure_identities([identity for identity, _, _, _ in items])

        # 2. Upload photo bytes to MinIO concurrently with bounded parallelism.
        upload_args = [
            (uuid.UUID(identity.person_id), uuid.UUID(photo.photo_id), data)
            for identity, photo, _, data in items
        ]
        upload_results = await asyncio.gather(
            *(self._upload_photo(*args) for args in upload_args),
            return_exceptions=True,
        )
        object_keys: list[str] = []
        for result in upload_results:
            if isinstance(result, Exception):
                raise result
            object_keys.append(result)

        # Deduplicate within batch to avoid Postgres cardinality violation on
        # ON CONFLICT DO UPDATE when the same photo_id appears twice.
        unique_items: list[tuple[EnrollmentIdentity, EnrollmentPhoto, ExtractedFace, bytes]] = []
        unique_object_keys: list[str] = []
        seen_photo_ids: set[str] = set()
        for (identity, photo, face, raw), object_key in zip(items, object_keys):
            if photo.photo_id in seen_photo_ids:
                continue
            seen_photo_ids.add(photo.photo_id)
            unique_items.append((identity, photo, face, raw))
            unique_object_keys.append(object_key)
        items = unique_items
        object_keys = unique_object_keys

        # 3. PostgreSQL bulk upserts (active immediately).
        photo_rows = [
            {
                "photo_id": uuid.UUID(photo.photo_id),
                "person_id": uuid.UUID(identity.person_id),
                "object_key": object_key,
                "content_sha256": photo.content_sha256,
                "mime_type": "application/octet-stream",
                "width": 0,
                "height": 0,
                "status": "active",
                "created_at": now,
                "updated_at": now,
            }
            for (identity, photo, _, _), object_key in zip(items, object_keys)
        ]
        sample_rows = [
            {
                "sample_id": derive_sample_id(uuid.UUID(photo.photo_id), self._embedding_model_version),
                "person_id": uuid.UUID(identity.person_id),
                "photo_id": uuid.UUID(photo.photo_id),
                "detector_model": self._model_pack,
                "embedding_model": self._embedding_model_version,
                "bbox": _bbox_to_dict(face.bbox),
                "landmarks": face.landmarks.tolist(),
                "quality_score": float(face.score),
                "status": "active",
                "created_at": now,
                "updated_at": now,
            }
            for identity, photo, face, _ in items
        ]
        qdrant_points = [
            models.PointStruct(
                id=str(derive_sample_id(uuid.UUID(photo.photo_id), self._embedding_model_version)),
                vector=face.embedding.tolist(),
                payload={
                    "sampleId": str(derive_sample_id(uuid.UUID(photo.photo_id), self._embedding_model_version)),
                    "photoId": str(uuid.UUID(photo.photo_id)),
                    "personId": str(uuid.UUID(identity.person_id)),
                    "active": True,
                    "modelVersion": self._embedding_model_version,
                },
            )
            for identity, photo, face, _ in items
        ]

        if photo_rows:
            await self._db.execute(
                pg_insert(PersonPhoto)
                .values(photo_rows)
                .on_conflict_do_update(
                    index_elements=["photo_id"],
                    set_={
                        "person_id": pg_insert(PersonPhoto).excluded.person_id,
                        "object_key": pg_insert(PersonPhoto).excluded.object_key,
                        "content_sha256": pg_insert(PersonPhoto).excluded.content_sha256,
                        "status": "active",
                        "updated_at": now,
                    },
                )
            )
        if sample_rows:
            await self._db.execute(
                pg_insert(FaceSample)
                .values(sample_rows)
                .on_conflict_do_update(
                    index_elements=["photo_id"],
                    set_={
                        "person_id": pg_insert(FaceSample).excluded.person_id,
                        "detector_model": pg_insert(FaceSample).excluded.detector_model,
                        "embedding_model": pg_insert(FaceSample).excluded.embedding_model,
                        "bbox": pg_insert(FaceSample).excluded.bbox,
                        "landmarks": pg_insert(FaceSample).excluded.landmarks,
                        "quality_score": pg_insert(FaceSample).excluded.quality_score,
                        "status": "active",
                        "updated_at": now,
                    },
                )
            )
        await self._upsert_qdrant(qdrant_points)

    async def _upsert_qdrant(
        self,
        points: list[models.PointStruct],
    ) -> None:
        if not points:
            return
        await self._vector_store.upsert_batch(points, wait=self._qdrant_wait)

    async def _read_and_extract(
        self,
        tasks: list[tuple[EnrollmentIdentity, EnrollmentPhoto]],
        result: _ShardResult,
    ) -> list[tuple[EnrollmentIdentity, EnrollmentPhoto, GpuFaceExtraction, bytes]]:
        """Read bytes for the chunk and extract faces.

        Decode failures are counted but do not fail the whole chunk.  A fatal
        GPU/inference error aborts the shard.
        """
        if not tasks:
            return []

        # Bounded concurrent file reads.
        io_futures = [self._read_one(photo) for _, photo in tasks]
        io_results = await asyncio.gather(*io_futures, return_exceptions=True)

        valid: list[tuple[EnrollmentIdentity, EnrollmentPhoto, bytes]] = []
        for (identity, photo), maybe_data in zip(tasks, io_results):
            if isinstance(maybe_data, _DecodeError):
                result.decode_error += 1
                logger.warning("Decode/read failed for %s", photo.path)
                continue
            if isinstance(maybe_data, Exception):
                result.decode_error += 1
                logger.warning("Decode/read failed for %s: %s", photo.path, maybe_data)
                continue
            photo.content_sha256 = hashlib.sha256(maybe_data).hexdigest()
            valid.append((identity, photo, maybe_data))

        if not valid:
            return []

        image_bytes_list = [data for _, _, data in valid]
        t0 = time.perf_counter()
        try:
            faces = await self._extract_batch_faces(image_bytes_list)
        except Exception as exc:
            result.failed += len(valid)
            result.fatal_error = True
            result.fatal_code = "EXTRACTION_ERROR"
            result.fatal_stage = "gpu_extraction"
            result.fatal_message = _sanitize(str(exc))
            raise
        result.extraction_ms += (time.perf_counter() - t0) * 1000

        output: list[tuple[EnrollmentIdentity, EnrollmentPhoto, GpuFaceExtraction, bytes]] = []
        for (identity, photo, data), face in zip(valid, faces):
            if face is None:
                result.no_face += 1
                continue
            output.append((identity, photo, face, data))
        return output

    async def _produce(
        self,
        identities: Iterable[EnrollmentIdentity],
        queue: asyncio.Queue,
        batch_size: int,
        max_photos: int | None,
        result: _ShardResult,
        cancel_check: Callable[[], bool] | None,
        consumer_task: asyncio.Task | None = None,
    ) -> None:
        """Stream identities, read bytes, extract faces, push to queue."""
        pending: list[tuple[EnrollmentIdentity, EnrollmentPhoto]] = []
        seen_photos = 0
        should_stop = False

        async def _put(item: list) -> None:
            if consumer_task is None:
                await queue.put(item)
                return
            try:
                await asyncio.wait_for(queue.put(item), timeout=120)
            except asyncio.TimeoutError:
                if consumer_task.done() and consumer_task.exception():
                    raise consumer_task.exception()
                raise RuntimeError("queue.put timed out; consumer may be stalled")

        logger.info("producer starting for %s identities (max_photos=%s)", "?", max_photos)
        try:
            for identity in identities:
                result.discovered_identities += 1
                result.discovered_photos += len(identity.photos)
                logger.info(
                    "producer identity %s photos=%d total_seen_identities=%d total_seen_photos=%d",
                    identity.identity_key,
                    len(identity.photos),
                    result.discovered_identities,
                    result.discovered_photos,
                )
                for photo in identity.photos:
                    if max_photos is not None and seen_photos >= max_photos:
                        should_stop = True
                        break
                    pending.append((identity, photo))
                    seen_photos += 1
                    if len(pending) >= batch_size:
                        extracted = await self._read_and_extract(pending, result)
                        await _put(extracted)
                        result.total_batches += 1
                        logger.info("producer queued batch %d size %d", result.total_batches, len(extracted))
                        pending = []
                        if cancel_check is not None and cancel_check():
                            return
                if should_stop:
                    break
        finally:
            if pending:
                extracted = await self._read_and_extract(pending, result)
                await _put(extracted)
                result.total_batches += 1
                logger.info("producer queued final batch %d size %d", result.total_batches, len(extracted))
            await queue.put(None)

    async def _consume(
        self,
        queue: asyncio.Queue,
        result: _ShardResult,
        progress_callback: Callable[[dict[str, Any]], Any] | None,
        process_record: ProcessRecord,
    ) -> None:
        """Persist extracted chunks, commit after each activation batch."""
        logger.info("consumer starting")
        seq = 0
        while True:
            logger.info("consumer waiting on queue")
            item = await queue.get()
            logger.info("consumer got item size=%s", len(item) if item is not None else None)
            if item is None:
                return
            if not item:
                continue

            last_identity_key = item[-1][0].identity_key
            try:
                await self._persist_batch(item)
            except Exception as exc:
                result.persistence_error += len(item)
                result.failed += len(item)
                if not result.fatal_error:
                    result.fatal_error = True
                    result.fatal_code = "PERSISTENCE_ERROR"
                    result.fatal_stage = "persist_batch"
                    result.fatal_message = _sanitize(str(exc))
                await self._commit_progress(process_record, result, last_identity_key, seq)
                raise

            result.processed += len(item)
            result.enrolled += len(item)
            seq += 1
            logger.info("consumer persisted batch %d enrolled=%d", seq, result.enrolled)
            await self._commit_progress(process_record, result, last_identity_key, seq)

    async def _commit_progress(
        self,
        process_record: ProcessRecord,
        result: _ShardResult,
        last_identity_key: str,
        seq: int,
    ) -> None:
        error_count = result.no_face + result.decode_error + result.persistence_error
        soft_error_rate = (
            (result.no_face + result.decode_error) / max(result.discovered_photos, 1)
        )
        progress = {
            "discovered_identities": result.discovered_identities,
            "discovered_photos": result.discovered_photos,
            "processed": result.processed,
            "enrolled": result.enrolled,
            "no_face": result.no_face,
            "decode_error": result.decode_error,
            "persistence_error": result.persistence_error,
            "failed": result.failed,
            "soft_error_rate": round(soft_error_rate, 6),
            "total_batches": result.total_batches,
            "extraction_ms": round(result.extraction_ms, 2),
            "io_ms": round(result.io_ms, 2),
            "last_completed_identity_key": last_identity_key,
        }
        process_record.summary["progress"] = progress
        self._db.add(
            _event(
                process_record.process_id,
                seq,
                "progress",
                process_record.status,
                process_record.status,
                message=f"discovered={result.discovered_photos} enrolled={result.enrolled} no_face={result.no_face}",
                details={k: v for k, v in progress.items() if k in ("enrolled", "no_face", "decode_error", "persistence_error", "failed", "processed")},
            )
        )
        await self._db.commit()

    async def enroll_shard(
        self,
        identities: Iterable[EnrollmentIdentity],
        *,
        parent_process_id: uuid.UUID | None = None,
        idempotency_key: str = "",
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
        max_photos: int | None = None,
        worker_name: str | None = None,
    ) -> _ShardResult:
        result = _ShardResult(worker_name=worker_name or os.environ.get("WORKER_ID"))

        summary: dict[str, Any] = {
            "worker_name": result.worker_name,
            "identities": 0,
            "photos": 0,
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

        queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        consumer_task = asyncio.create_task(
            self._consume(queue, result, progress_callback, process_record)
        )

        def _on_consumer_done(t: asyncio.Task) -> None:
            if t.done() and not t.cancelled() and t.exception() is not None:
                exc = t.exception()
                logger.error("consumer task died unexpectedly: %s", exc, exc_info=exc)

        consumer_task.add_done_callback(_on_consumer_done)

        try:
            await self._produce(
                identities,
                queue,
                self._extract_batch_size,
                max_photos,
                result,
                cancel_check,
                consumer_task=consumer_task,
            )
            if not consumer_task.done():
                await queue.put(None)
            await consumer_task
        except EnrollmentCancelled:
            await self._mark_cancelled(process_record, result)
            raise
        except Exception as exc:
            if not result.fatal_error:
                result.fatal_error = True
                result.fatal_code = result.fatal_code or "SHARD_FAILURE"
                result.fatal_stage = result.fatal_stage or "unknown"
                result.fatal_message = result.fatal_message or _sanitize(str(exc))
            await self._mark_failed(process_record, result, str(exc))
            raise
        finally:
            if not consumer_task.done():
                consumer_task.cancel()
                try:
                    await consumer_task
                except asyncio.CancelledError:
                    pass

        soft_error_rate = (result.no_face + result.decode_error) / max(result.discovered_photos, 1)
        if result.fatal_error or result.failed > 0 or soft_error_rate > 0.03:
            result.status = "failed"
            process_record.status = "failed"
            process_record.summary["outcome"] = "failed"
        elif soft_error_rate > 0.0:
            process_record.status = "completed"
            process_record.summary["outcome"] = "completed_with_warnings"
        else:
            process_record.status = "completed"
            process_record.summary["outcome"] = "completed"

        process_record.completed_at = _utc_now()
        process_record.summary["progress"] = {
            "discovered_identities": result.discovered_identities,
            "discovered_photos": result.discovered_photos,
            "processed": result.processed,
            "enrolled": result.enrolled,
            "no_face": result.no_face,
            "decode_error": result.decode_error,
            "persistence_error": result.persistence_error,
            "failed": result.failed,
            "soft_error_rate": round(soft_error_rate, 6),
            "extraction_ms": round(result.extraction_ms, 2),
            "io_ms": round(result.io_ms, 2),
        }
        if progress_callback is not None:
            try:
                await progress_callback(process_record.summary["progress"])
            except Exception:
                logger.exception("progress callback failed")
        self._db.add(
            _event(
                process_record.process_id,
                1,
                "finished",
                "running",
                process_record.status,
                message=f"status={process_record.summary['outcome']} enrolled={result.enrolled}",
                details=process_record.summary["progress"],
            )
        )
        await self._db.commit()
        return result

    async def _mark_cancelled(
        self,
        process_record: ProcessRecord,
        result: _ShardResult,
    ) -> None:
        process_record.status = "cancelled"
        process_record.completed_at = _utc_now()
        process_record.error_message = "cancelled by operator"
        self._db.add(
            _event(
                process_record.process_id,
                1,
                "cancelled",
                "running",
                "cancelled",
                message=f"cancelled after {result.enrolled} enrolled",
                details={"enrolled": result.enrolled, "processed": result.processed},
            )
        )
        await self._db.commit()

    async def _mark_failed(
        self,
        process_record: ProcessRecord,
        result: _ShardResult,
        raw_message: str,
    ) -> None:
        process_record.status = "failed"
        process_record.completed_at = _utc_now()
        process_record.error_message = _sanitize(raw_message)[:500]
        process_record.summary["failure"] = {
            "code": result.fatal_code or "UNKNOWN",
            "stage": result.fatal_stage or "unknown",
            "worker": result.worker_name,
            "message": result.fatal_message or _sanitize(raw_message),
        }
        self._db.add(
            _event(
                process_record.process_id,
                1,
                "failed",
                "running",
                "failed",
                message=f"{result.fatal_code or 'UNKNOWN'}: {result.fatal_message or _sanitize(raw_message)}",
            )
        )
        await self._db.commit()

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
        total_enrolled = sum(r.enrolled for r in shard_results)
        total_failed = sum(r.failed for r in shard_results)
        total_photos = sum(r.discovered_photos for r in shard_results)
        total_no_face = sum(r.no_face for r in shard_results)
        total_decode_error = sum(r.decode_error for r in shard_results)
        any_fatal = any(r.fatal_error for r in shard_results)
        soft_rate = (total_no_face + total_decode_error) / max(total_photos, 1)
        if any_fatal or total_failed > 0 or soft_rate > 0.03:
            record.status = "failed"
            record.summary["outcome"] = "failed"
        elif soft_rate > 0.0:
            record.status = "completed"
            record.summary["outcome"] = "completed_with_warnings"
        else:
            record.status = "completed"
            record.summary["outcome"] = "completed"
        record.completed_at = _utc_now()
        record.summary.update(
            {
                "shards": len(shard_results),
                "discovered_photos": total_photos,
                "enrolled": total_enrolled,
                "no_face": total_no_face,
                "decode_error": total_decode_error,
                "failed": total_failed,
                "soft_error_rate": round(soft_rate, 6),
            }
        )
        await self._db.commit()


def _utc_now() -> __import__("datetime").datetime:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _sanitize(message: str) -> str:
    """Strip filesystem paths and long stack traces for UI display."""
    import re
    message = re.sub(r"[/\\][^\s]{4,}", "<path>", message)
    return message[:500]


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
