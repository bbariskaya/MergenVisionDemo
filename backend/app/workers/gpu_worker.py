"""Long-lived GPU worker supporting one physical GPU per container.

Each worker is a separate OS process, sees exactly one CUDA device as
internal ordinal 0, and processes sequential jobs while keeping the same
pipeline instance warm across batches.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import io
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

import logging
import traceback

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

logger = logging.getLogger(__name__)

from app.core.config import settings
from app.core.ids import (
    derive_face_identity_id,
    derive_person_id,
    derive_process_id,
    identity_hmac,
)
from app.domain.models import Person, PersonPhoto, ProcessEvent, ProcessRecord
from app.infrastructure import db as db_module
from app.infrastructure.minio import PhotoStorage
from app.infrastructure.qdrant import FaceVectorStore
from app.ml.gpu.face_pipeline import GpuFacePipeline
from app.services.bulk_enrollment import BulkEnrollmentService, EnrollmentCancelled
from app.services.bulk_manifest import (
    EnrollmentIdentity,
    EnrollmentPhoto,
    build_casia_manifest,
    build_lfw_manifest,
    shard_by_person_id,
)
from app.services.vggface_manifest import (
    stream_vggface_manifest,
    vggface_preflight,
    shard_vggface_identities,
)
from app.services.readiness import ReadinessService


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_WORKER_ID = os.environ.get("WORKER_ID", "gpu-worker-0")
_HOST_DEVICE_ID_VAR = os.environ.get("HOST_GPU_DEVICE_ID", "0")
_WORKER_ROLE = os.environ.get("WORKER_ROLE", "online")


class _ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class SourceDescriptor(_ApiModel):
    type: Literal["local_lfw", "local_vggface", "local_casia", "synthetic", "manifest"] = Field(alias="type")
    path: str | None = Field(alias="path", default=None)
    bucket: str | None = Field(alias="bucket", default=None)
    minio_prefix: str | None = Field(alias="minioPrefix", default=None)
    max_identities: int | None = Field(alias="maxIdentities", default=None)
    num_identities: int | None = Field(alias="numIdentities", default=None)
    photos_per_identity: int | None = Field(alias="photosPerIdentity", default=None)
    identities: list[dict[str, Any]] | None = Field(alias="identities", default=None)
    resume_after_identity_key: str | None = Field(alias="resumeAfterIdentityKey", default=None)


class JobCreateRequest(_ApiModel):
    job_id: str = Field(alias="jobId")
    idempotency_key: str = Field(alias="idempotencyKey")
    source: SourceDescriptor
    dataset_type: Literal["lfw", "vggface", "casia", "synthetic"] = Field(alias="datasetType")
    mode: Literal["import", "benchmark"] = Field(alias="mode")
    requested_parallelism: int = Field(alias="requestedParallelism", default=1)
    assigned_workers: list[str] | None = Field(alias="assignedWorkers", default=None)
    shard_index: int | None = Field(alias="shardIndex", default=None)
    max_photos: int | None = Field(alias="maxPhotos", default=None)


class JobStatusResponse(_ApiModel):
    job_id: str = Field(alias="jobId")
    idempotency_key: str = Field(alias="idempotencyKey")
    status: str
    worker_id: str = Field(alias="workerId")
    host_gpu_uuid: str = Field(alias="hostGpuUuid")
    internal_device: int = Field(alias="internalDevice")
    progress: dict[str, Any]
    created_at: str | None = Field(alias="createdAt", default=None)
    completed_at: str | None = Field(alias="completedAt", default=None)
    error_message: str | None = Field(alias="errorMessage", default=None)


class HealthResponse(_ApiModel):
    status: str
    worker_id: str = Field(alias="workerId")
    host_gpu_uuid: str = Field(alias="hostGpuUuid")
    internal_device: int = Field(alias="internalDevice")
    pipeline_warmed: bool = Field(alias="pipelineWarmed")


def _deterministic_process_id(idempotency_key: str) -> uuid.UUID:
    return derive_process_id("gpu_worker_job", idempotency_key.encode("utf-8"))


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


async def _ensure_single_gpu_visible() -> tuple[str, str]:
    """Return (host device id string, visible GPU UUID)."""
    from cuda.bindings import runtime as cuda_runtime

    err, visible = cuda_runtime.cudaGetDeviceCount()
    if err != 0 or visible != 1:
        raise RuntimeError(
            f"worker {_WORKER_ID} expects exactly one visible GPU, found {visible} (err={err})"
        )

    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=uuid",
        "--format=csv,noheader",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed: {stderr.decode().strip() or 'unknown error'}"
        )
    uuids = [line.strip() for line in stdout.decode().splitlines() if line.strip()]
    if len(uuids) != 1:
        raise RuntimeError(
            f"worker {_WORKER_ID} expected one GPU UUID from nvidia-smi, got {uuids}"
        )
    return _HOST_DEVICE_ID_VAR, uuids[0]


def _build_synthetic_identities(
    num_identities: int,
    photos_per_identity: int,
) -> tuple[EnrollmentIdentity, ...]:
    """Create deterministic synthetic identity/photo descriptors.

    Real JPEGs are written to /tmp because the enrollment service currently
    reads photo bytes from disk paths.
    """
    identities: list[EnrollmentIdentity] = []
    rng = np.random.default_rng(42)
    for i in range(num_identities):
        identity_key = f"Synthetic_{i:04d}"
        display_name = identity_key.replace("_", " ")
        hmac_val = identity_hmac(identity_key, settings.hmac_key)
        photos: list[EnrollmentPhoto] = []
        for j in range(photos_per_identity):
            # Deterministic 64x64 RGB image; not a real face but safe for
            # pipeline/no-face throughput tests.
            arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            tmp_path = Path(f"/tmp/synthetic_{i:04d}_{j:02d}.jpg")
            img.save(tmp_path, format="JPEG", quality=85)
            content_sha256 = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
            photos.append(EnrollmentPhoto(path=tmp_path, content_sha256=content_sha256))
        identities.append(
            EnrollmentIdentity(
                identity_key=identity_key,
                display_name=display_name,
                identity_hmac=hmac_val,
                person_id=str(derive_person_id(hmac_val)),
                face_identity_id=str(derive_face_identity_id(hmac_val)),
                source_dataset="synthetic",
                photos=tuple(photos),
            )
        )
    return tuple(identities)


def _load_manifest_identities(payload: SourceDescriptor) -> tuple[EnrollmentIdentity, ...]:
    """Load identities from an inline manifest (test helper)."""
    result: list[EnrollmentIdentity] = []
    for identity in payload.identities or []:
        photos = [
            EnrollmentPhoto(
                path=Path(photo["path"]),
                content_sha256=photo["contentSha256"],
            )
            for photo in identity.get("photos", [])
        ]
        result.append(
            EnrollmentIdentity(
                identity_key=identity["identityKey"],
                display_name=identity["displayName"],
                identity_hmac=identity["identityHmac"],
                person_id=identity["personId"],
                face_identity_id=identity["faceIdentityId"],
                source_dataset=identity.get("sourceDataset", "inline"),
                photos=tuple(photos),
            )
        )
    return tuple(result)


async def _load_identities(
    payload: SourceDescriptor,
    shard_index: int = 0,
    num_shards: int = 1,
    max_photos: int | None = None,
) -> Iterator[EnrollmentIdentity] | tuple[EnrollmentIdentity, ...]:
    if payload.type == "manifest":
        return _load_manifest_identities(payload)
    if payload.type == "synthetic":
        return await asyncio.to_thread(
            _build_synthetic_identities,
            payload.num_identities or 3,
            payload.photos_per_identity or 1,
        )
    if payload.type == "local_vggface":
        if not payload.path:
            raise ValueError("local_vggface source requires path")
        root = Path(payload.path)
        if not root.is_dir():
            raise ValueError(f"vggface root not found: {root}")
        identities = stream_vggface_manifest(
            root,
            shard_index=shard_index if num_shards > 1 else None,
            num_shards=num_shards if num_shards > 1 else None,
            resume_after_identity_key=payload.resume_after_identity_key,
            max_photos=max_photos,
        )
    elif payload.type == "local_lfw":
        if not payload.path:
            raise ValueError("local_lfw source requires path")
        root = Path(payload.path)
        if not root.is_dir():
            raise ValueError(f"lfw root not found: {root}")
        identities = build_lfw_manifest(root)
        if payload.max_identities is not None:
            identities = identities[: payload.max_identities]
    elif payload.type == "local_casia":
        if not payload.path:
            raise ValueError("local_casia source requires path")
        root = Path(payload.path)
        if not root.is_dir():
            raise ValueError(f"casia root not found: {root}")
        identities = build_casia_manifest(root)
        if num_shards > 1:
            shards = shard_by_person_id(identities, num_shards)
            identities = shards[shard_index]
        if payload.max_identities is not None:
            identities = identities[: payload.max_identities]
    else:
        raise ValueError(f"unsupported source type: {payload.type}")

    # stream_vggface_manifest already applies sharding; do not shard again.
    return iter(identities)


def _job_status_response(
    process: ProcessRecord,
    request: JobCreateRequest,
    worker_info: dict[str, Any],
) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=str(process.process_id),
        idempotency_key=request.idempotency_key,
        status=process.status,
        worker_id=worker_info["worker_id"],
        host_gpu_uuid=worker_info["host_gpu_uuid"],
        internal_device=0,
        progress=process.summary.get("progress", {}),
        created_at=process.created_at.isoformat() if process.created_at else None,
        completed_at=process.completed_at.isoformat() if process.completed_at else None,
        error_message=process.error_message,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    # One dedicated thread owns the TensorRT context; a separate bounded pool
    # handles file reads and synchronous MinIO SDK calls.  Do not expose a
    # global 128-thread default executor to unrelated operations.
    io_workers = min(32, (os.cpu_count() or 4) * 2)
    gpu_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="gpu-"
    )
    io_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=io_workers, thread_name_prefix="io-"
    )
    loop.set_default_executor(io_executor)
    db_module.configure_engine()

    host_device_id, host_gpu_uuid = await _ensure_single_gpu_visible()

    storage: PhotoStorage | None = None
    vector_store: FaceVectorStore | None = None
    pipeline: GpuFacePipeline | None = None

    try:
        storage = PhotoStorage()
        await storage.initialize()

        vector_store = FaceVectorStore()
        await vector_store.initialize()

        pipeline = GpuFacePipeline(device_id=0)
        pipeline.warmup()

        app.state.worker_info = {
            "worker_id": _WORKER_ID,
            "host_device_id": host_device_id,
            "host_gpu_uuid": host_gpu_uuid,
            "internal_device": 0,
            "pipeline_warmed": True,
        }
        app.state.storage = storage
        app.state.vector_store = vector_store
        app.state.pipeline = pipeline
        app.state.pipeline_lock = asyncio.Lock()
        app.state.gpu_executor = gpu_executor
        app.state.io_executor = io_executor
        app.state.readiness_service = ReadinessService(
            engine=db_module.engine,
            storage=storage,
            vector_store=vector_store,
        )
        app.state.current_job_id: str | None = None
        app.state.cancel_requested = False

        yield
    finally:
        if pipeline is not None:
            try:
                if hasattr(pipeline, "close"):
                    pipeline.close()
            except Exception:
                pass
        if vector_store is not None:
            try:
                await vector_store.close()
            except Exception:
                pass
        gpu_executor.shutdown(wait=True)
        io_executor.shutdown(wait=True)
        await db_module.dispose_engine()


def create_worker_app() -> FastAPI:
    app = FastAPI(
        title="MergenVision GPU Worker",
        version="0.2.0",
        lifespan=_lifespan,
    )

    @app.get("/health/live")
    async def health_live(request: Request) -> dict[str, Any]:
        info = request.app.state.worker_info
        return {
            "status": "alive",
            "worker_id": info["worker_id"],
            "timestamp": _utc_now().isoformat(),
        }

    @app.get("/health/ready")
    async def health_ready(request: Request) -> HealthResponse:
        readiness = request.app.state.readiness_service
        _, ok = await readiness.check()
        busy = request.app.state.current_job_id is not None
        info = request.app.state.worker_info
        status_text = "ready" if (ok and not busy) else ("busy" if ok else "not_ready")
        return HealthResponse(
            status=status_text,
            worker_id=info["worker_id"],
            host_gpu_uuid=info["host_gpu_uuid"],
            internal_device=info["internal_device"],
            pipeline_warmed=info["pipeline_warmed"],
        )

    @app.post("/internal/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def create_job(
        request: Request,
        job: JobCreateRequest,
    ) -> JobStatusResponse:
        if not job.idempotency_key:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="idempotency_key is required",
            )

        if _WORKER_ROLE == "online" and job.dataset_type in {"lfw", "vggface", "casia"}:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="bulk enrollment is prohibited on the online-only worker",
            )

        process_id = _deterministic_process_id(job.idempotency_key)

        async with db_module.AsyncSessionLocal() as session:
            existing = await session.get(ProcessRecord, process_id)
            if existing is not None and existing.status in ("completed", "failed"):
                return _job_status_response(existing, job, request.app.state.worker_info)

            if existing is None:
                record = ProcessRecord(
                    process_id=process_id,
                    process_type="gpu_worker_job",
                    status="pending",
                    summary={
                        "worker_id": _WORKER_ID,
                        "host_gpu_uuid": request.app.state.worker_info["host_gpu_uuid"],
                        "requested_parallelism": job.requested_parallelism,
                        "dataset_type": job.dataset_type,
                        "mode": job.mode,
                        "idempotency_key": job.idempotency_key,
                        "input_job_id": job.job_id,
                    },
                )
                session.add(record)
                await session.flush()
                session.add(
                    _event(
                        process_id,
                        0,
                        "accepted",
                        None,
                        "pending",
                        message=f"accepted by {_WORKER_ID}",
                    )
                )
                await session.commit()
            else:
                record = existing

        if request.app.state.current_job_id is not None:
            # Sequential processing: one job at a time per worker.
            async with db_module.AsyncSessionLocal() as session:
                record.status = "failed"
                record.error_message = "worker is already processing a job"
                record.completed_at = _utc_now()
                session.add(record)
                session.add(
                    _event(
                        process_id,
                        1,
                        "rejected_busy",
                        "pending",
                        "failed",
                        message="rejected: worker busy",
                    )
                )
                await session.commit()
            return _job_status_response(record, job, request.app.state.worker_info)

        # Start work in background so POST returns immediately and health/cancel
        # endpoints stay responsive during long GPU inference.
        app_state = request.app.state
        request.app.state.current_job_id = str(process_id)

        async def _job_runner() -> None:
            await _run_job(app_state, job, process_id)

        asyncio.create_task(_job_runner())
        return _job_status_response(record, job, request.app.state.worker_info)

    @app.post("/internal/v1/jobs/{job_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
    async def cancel_job(
        job_id: uuid.UUID,
        request: Request,
    ) -> JobStatusResponse:
        async with db_module.AsyncSessionLocal() as session:
            record = await session.get(ProcessRecord, job_id)
            if record is None or record.process_type != "gpu_worker_job":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="job not found"
                )
            if request.app.state.current_job_id != str(job_id):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="job is not running on this worker",
                )
            request.app.state.cancel_requested = True
            if record.status in ("pending", "queued", "running"):
                record.status = "cancel_requested"
                record.updated_at = _utc_now()
                session.add(
                    _event(
                        job_id,
                        100,
                        "cancel_requested",
                        record.status,
                        "cancel_requested",
                        message=f"cancel requested on {_WORKER_ID}",
                    )
                )
            await session.commit()
            return _job_status_response(record, JobCreateRequest(
                job_id=str(job_id),
                idempotency_key=record.summary.get("idempotency_key", str(job_id)),
                source=SourceDescriptor(type="manifest"),
                dataset_type=record.summary.get("dataset_type", "lfw"),
                mode=record.summary.get("mode", "import"),
            ), request.app.state.worker_info)

    async def _run_job(
        app_state: Any,
        job: JobCreateRequest,
        process_id: uuid.UUID,
    ) -> JobStatusResponse | None:
        app_state.cancel_requested = False
        app_state.current_job_id = str(job.job_id)
        try:
            async with db_module.AsyncSessionLocal() as session:
                record = await session.get(ProcessRecord, process_id)
                if record is None:
                    raise RuntimeError("process record disappeared")
                if record.status == "running":
                    # Another concurrent request started it; wait/return.
                    return _job_status_response(record, job, app_state.worker_info)
                record.status = "running"
                record.started_at = _utc_now()
                session.add(
                    _event(
                        process_id,
                        1,
                        "started",
                        "pending",
                        "running",
                        message=f"{_WORKER_ID} starts processing",
                    )
                )
                await session.commit()

            identities = await _load_identities(
                job.source,
                shard_index=job.shard_index or 0,
                num_shards=job.requested_parallelism or 1,
                max_photos=job.max_photos,
            )

            async def _report_progress(progress: dict[str, Any]) -> None:
                async with db_module.AsyncSessionLocal() as session:
                    rec = await session.get(ProcessRecord, process_id)
                    if rec is not None:
                        rec.summary["progress"] = progress
                        flag_modified(rec, "summary")
                        await session.commit()

            async with db_module.AsyncSessionLocal() as session:
                service = BulkEnrollmentService(
                    db=session,
                    storage=app_state.storage,
                    vector_store=app_state.vector_store,
                    pipeline=app_state.pipeline,
                    pipeline_lock=app_state.pipeline_lock,
                    gpu_executor=app_state.gpu_executor,
                    io_executor=app_state.io_executor,
                    qdrant_wait=False,
                )
                result = await service.enroll_shard(
                    identities,
                    parent_process_id=process_id,
                    idempotency_key=job.idempotency_key,
                    cancel_check=lambda: app_state.cancel_requested,
                    progress_callback=_report_progress,
                    max_photos=job.max_photos,
                    worker_name=_WORKER_ID,
                )

            async with db_module.AsyncSessionLocal() as session:
                record = await session.get(ProcessRecord, process_id)
                # Service already finalized the shard process record; keep the
                # same decisions but expose the new result-shape fields.
                record.status = result.status or record.status
                record.completed_at = _utc_now()
                record.summary["progress"] = {
                    "worker_name": result.worker_name,
                    "discovered_identities": result.discovered_identities,
                    "discovered_photos": result.discovered_photos,
                    "processed": result.processed,
                    "enrolled": result.enrolled,
                    "no_face": result.no_face,
                    "decode_error": result.decode_error,
                    "persistence_error": result.persistence_error,
                    "failed": result.failed,
                    "fatal_error": result.fatal_error,
                    "extraction_ms": result.extraction_ms,
                    "io_ms": result.io_ms,
                }
                if result.fatal_code:
                    record.summary["failure"] = {
                        "code": result.fatal_code,
                        "stage": result.fatal_stage,
                        "worker": result.worker_name,
                        "message": result.fatal_message,
                    }
                flag_modified(record, "summary")
                session.add(
                    _event(
                        process_id,
                        2,
                        "completed" if record.status == "completed" else record.status,
                        "running",
                        record.status,
                        message=(
                            f"enrolled {result.enrolled} faces, "
                            f"no_face={result.no_face}, "
                            f"decode_error={result.decode_error}, "
                            f"failed={result.failed}"
                        ),
                    )
                )
                await session.commit()
                return _job_status_response(record, job, app_state.worker_info)

        except EnrollmentCancelled:
            async with db_module.AsyncSessionLocal() as session:
                record = await session.get(ProcessRecord, process_id)
                if record is not None and record.status != "cancelled":
                    record.status = "cancelled"
                    record.completed_at = _utc_now()
                    flag_modified(record, "summary")
                    session.add(
                        _event(
                            process_id,
                            2,
                            "cancelled",
                            "running",
                            "cancelled",
                            message=f"job cancelled on {_WORKER_ID}",
                        )
                    )
                    await session.commit()
                return _job_status_response(record, job, app_state.worker_info)

        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception("shard %s failed: %s", process_id, exc)
            record = None
            try:
                async with db_module.AsyncSessionLocal() as session:
                    record = await session.get(ProcessRecord, process_id)
                    if record is not None:
                        record.status = "failed"
                        record.completed_at = _utc_now()
                        record.error_message = str(exc)[:500]
                        record.summary["error_traceback"] = tb[-4000:]
                        flag_modified(record, "summary")
                        session.add(
                            _event(
                                process_id,
                                2,
                                "failed",
                                "running",
                                "failed",
                                message=f"failure: {exc.__class__.__name__}",
                                details={"traceback": tb[-2000:]},
                            )
                        )
                        await session.commit()
            except Exception as persist_exc:
                logger.exception("could not persist failure for %s: %s", process_id, persist_exc)
            return _job_status_response(record, job, app_state.worker_info)
        finally:
            app_state.current_job_id = None
            app_state.cancel_requested = False

    @app.get("/internal/v1/jobs/{job_id}")
    async def get_job(
        job_id: uuid.UUID,
        request: Request,
    ) -> JobStatusResponse:
        async with db_module.AsyncSessionLocal() as session:
            record = await session.get(ProcessRecord, job_id)
            if record is None or record.process_type != "gpu_worker_job":
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
            return JobStatusResponse(
                job_id=str(record.process_id),
                idempotency_key=record.summary.get("idempotency_key", ""),
                status=record.status,
                worker_id=record.summary.get("worker_id", ""),
                host_gpu_uuid=record.summary.get("host_gpu_uuid", ""),
                internal_device=0,
                progress=record.summary.get("progress", {}),
                created_at=record.created_at.isoformat() if record.created_at else None,
                completed_at=record.completed_at.isoformat() if record.completed_at else None,
                error_message=record.error_message,
            )

    def _coro_chain(coro: Any) -> str:
        parts: list[str] = []
        seen: set[int] = set()
        while coro is not None and id(coro) not in seen:
            seen.add(id(coro))
            frame = getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None)
            name = getattr(coro, "__name__", type(coro).__name__)
            if frame is not None:
                parts.append(
                    f"  coro={name} at {frame.f_code.co_filename}:{frame.f_lineno} "
                    f"locals={ {k: repr(v)[:80] for k, v in list(frame.f_locals.items())[:5]} }"
                )
            else:
                parts.append(f"  coro={name} (no frame)")
            coro = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)
        return "\n".join(parts)

    @app.get("/internal/v1/debug/tasks")
    async def debug_tasks() -> dict[str, Any]:
        stacks: list[str] = []
        for task in asyncio.all_tasks():
            if task.get_name() in {"Task-1", "Task-2"}:
                continue
            try:
                buf = io.StringIO()
                task.print_stack(file=buf)
                chain = _coro_chain(task.get_coro()) if task.get_coro() else "no coro"
                stacks.append(
                    f"--- {task.get_name()} ---\n"
                    + buf.getvalue()
                    + "\nawait chain:\n"
                    + chain
                )
            except Exception as exc:
                stacks.append(f"--- {task.get_name()} err {exc} ---")
        return {"tasks": stacks, "current_job_id": app.state.current_job_id}

    return app


app = create_worker_app()
