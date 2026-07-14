"""Public control-plane endpoints for durable VGGFace bulk enrollment."""
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.domain.models import ProcessRecord
from app.schemas.face import (
    VggfaceBulkJobResponse,
    VggfaceBulkJobStartRequest,
)
from app.services.bulk_orchestrator import (
    dispatch_shards,
    get_casia_job,
    get_lfw_job,
    get_vggface_job,
    request_cancellation,
    resume_vggface_job,
    start_casia_job,
    start_lfw_job,
    start_vggface_job,
)

router = APIRouter(prefix="/bulk-jobs", tags=["bulk-jobs"])


def _to_response(record: __import__("app.domain.models").ProcessRecord) -> VggfaceBulkJobResponse:
    summary = record.summary or {}
    shards = summary.get("shards", [])
    return VggfaceBulkJobResponse(
        job_id=record.process_id,
        status=record.status,
        dataset_type=summary.get("dataset_type", "vggface"),
        assigned_workers=summary.get("assigned_workers", []),
        target_total_active_photos=summary.get("target_total_active_photos", 0),
        starting_active_photos=summary.get("starting_active_photos", 0),
        current_active_photos=summary.get("current_active_photos", 0),
        photos_added_by_job=summary.get("photos_added_by_job", 0),
        requested_photos=summary.get("requested_photos", 0),
        total_discovered=summary.get("total_discovered", 0),
        total_scanned=summary.get("total_scanned", 0),
        total_processed=summary.get("total_processed", 0),
        total_enrolled=summary.get("total_enrolled", 0),
        total_duplicate=summary.get("total_duplicate", 0),
        total_no_face=summary.get("total_no_face", 0),
        total_errors=summary.get("total_errors", 0),
        total_in_flight=summary.get("total_in_flight", 0),
        total_rejected=summary.get("total_rejected", 0),
        total_corrupt=summary.get("total_corrupt", 0),
        elapsed_seconds=summary.get("elapsed_seconds", 0.0),
        avg_photos_per_second=summary.get("avg_photos_per_second", 0.0),
        scanned_photos_per_second=summary.get("scanned_photos_per_second", 0.0),
        processed_photos_per_second=summary.get("processed_photos_per_second", 0.0),
        enrolled_photos_per_second=summary.get("enrolled_photos_per_second", 0.0),
        duplicate_photos_per_second=summary.get("duplicate_photos_per_second", 0.0),
        probe_p50_ms=summary.get("probe_p50_ms"),
        probe_p95_ms=summary.get("probe_p95_ms"),
        shards=[
            {
                "worker_id": shard.get("worker_id", ""),
                "shard_index": shard.get("shard_index", 0),
                "process_id": uuid.UUID(shard["process_id"]),
                "status": shard.get("status", "queued"),
                "progress": shard.get("progress", {}),
            }
            for shard in shards
        ],
        created_at=record.created_at,
        completed_at=record.completed_at,
    )


@router.post(
    "/vggface",
    response_model=VggfaceBulkJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start durable VGGFace bulk enrollment",
)
async def start_vggface(
    request: Annotated[VggfaceBulkJobStartRequest, ...],
    background_tasks: BackgroundTasks,
) -> VggfaceBulkJobResponse:
    result = await start_vggface_job(max_photos=request.max_photos)
    background_tasks.add_task(dispatch_shards, uuid.UUID(result.job_id))
    record = await get_vggface_job(uuid.UUID(result.job_id))
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="failed to create job",
        )
    return _to_response(record)


@router.get(
    "/latest",
    response_model=VggfaceBulkJobResponse,
    summary="Get the latest VGGFace bulk enrollment job",
)
async def get_latest_job(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> VggfaceBulkJobResponse:
    result = await db.execute(
        select(ProcessRecord)
        .where(ProcessRecord.process_type == "vggface_bulk")
        .order_by(desc(ProcessRecord.created_at))
        .limit(1)
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no job found")
    refreshed = await get_vggface_job(record.process_id)
    if refreshed is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return _to_response(refreshed)


@router.post(
    "/lfw",
    response_model=VggfaceBulkJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start durable LFW bulk enrollment",
)
async def start_lfw(
    request: Annotated[VggfaceBulkJobStartRequest, ...],
    background_tasks: BackgroundTasks,
) -> VggfaceBulkJobResponse:
    result = await start_lfw_job(max_photos=request.max_photos)
    background_tasks.add_task(dispatch_shards, uuid.UUID(result.job_id))
    record = await get_lfw_job(uuid.UUID(result.job_id))
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="failed to create job",
        )
    return _to_response(record)


@router.post(
    "/casia",
    response_model=VggfaceBulkJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start durable CASIA-WebFace bulk enrollment",
)
async def start_casia(
    request: Annotated[VggfaceBulkJobStartRequest, ...],
    background_tasks: BackgroundTasks,
) -> VggfaceBulkJobResponse:
    result = await start_casia_job(max_photos=request.max_photos)
    background_tasks.add_task(dispatch_shards, uuid.UUID(result.job_id))
    record = await get_casia_job(uuid.UUID(result.job_id))
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="failed to create job",
        )
    return _to_response(record)


@router.get(
    "/{job_id}",
    response_model=VggfaceBulkJobResponse,
    summary="Get durable bulk enrollment status",
)
async def get_job(job_id: uuid.UUID) -> VggfaceBulkJobResponse:
    record = await get_vggface_job(job_id)
    if record is None:
        record = await get_lfw_job(job_id)
    if record is None:
        record = await get_casia_job(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return _to_response(record)


@router.post(
    "/{job_id}/cancel",
    response_model=VggfaceBulkJobResponse,
    summary="Request graceful cancellation of a bulk enrollment job",
)
async def cancel_job(job_id: uuid.UUID) -> VggfaceBulkJobResponse:
    record = await request_cancellation(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return _to_response(record)


@router.post(
    "/{job_id}/resume",
    response_model=VggfaceBulkJobResponse,
    summary="Resume a cancelled or failed VGGFace bulk enrollment job",
)
async def resume_job(job_id: uuid.UUID) -> VggfaceBulkJobResponse:
    record = await resume_vggface_job(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return _to_response(record)
