"""Public control plane for durable VGGFace bulk enrollment.

The orchestrator runs inside the API container.  It only transfers compact job
descriptors to the long-lived GPU worker containers; it never proxies image
bytes.  Each worker owns its own ``ProcessRecord`` so progress, cancellation
and idempotent resume survive API container restarts.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import sqlalchemy as sa
from sqlalchemy import func, select
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import settings
from app.core.ids import derive_process_id
from app.domain.models import PersonPhoto, ProcessEvent, ProcessRecord
from app.infrastructure import db as db_module
from app.services.bulk_manifest import build_casia_manifest, expected_cardinality
from app.services.vggface_manifest_cache import get_vggface_preflight

logger = logging.getLogger(__name__)

_BULK_WORKERS = [
    w.strip()
    for w in os.environ.get("BULK_WORKERS", "gpu-worker-1,gpu-worker-2").split(",")
    if w.strip()
]
_RECOGNITION_PROBE_URL = "http://localhost:8000/faces/recognize"
_PROBE_INTERVAL_SECONDS = 5.0
_PROBE_TIMEOUT_SECONDS = 10.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def _active_photo_count() -> int:
    async with db_module.AsyncSessionLocal() as db:
        result = await db.execute(
            select(func.count(PersonPhoto.photo_id)).where(PersonPhoto.status == "active")
        )
        return int(result.scalar_one())


def _shard_process_id(idempotency_key: str) -> __import__("uuid").UUID:
    return derive_process_id("gpu_worker_job", idempotency_key.encode("utf-8"))


@dataclass(frozen=True)
class VggfaceJobStartResult:
    job_id: str
    starting_active_photos: int
    target_total_active_photos: int
    remaining_budget: int
    assigned_workers: list[str]


async def start_vggface_job(
    *,
    max_photos: int | None = None,
) -> VggfaceJobStartResult:
    """Create the durable parent job record and queue shard work.

    Actual dispatch to workers is performed by :func:`dispatch_shards` so that
    the HTTP endpoint can return immediately.
    """
    active_photos = await _active_photo_count()
    target = settings.vggface_target_active_photos
    remaining_budget = max(0, target - active_photos)

    preflight = await asyncio.to_thread(get_vggface_preflight, settings.vggface_dataset_path)
    requested_budget = max_photos if max_photos is not None else preflight.photo_count
    budget = min(requested_budget, remaining_budget, preflight.photo_count)
    if budget <= 0:
        raise ValueError(
            f"VGGFace photo budget exhausted (active={active_photos}, target={target})"
        )

    async with db_module.AsyncSessionLocal() as db:
        parent = ProcessRecord(
            process_type="vggface_bulk",
            status="queued",
            summary={
                "dataset_type": "vggface",
                "assigned_workers": _BULK_WORKERS,
                "requested_parallelism": len(_BULK_WORKERS),
                "target_total_active_photos": target,
                "starting_active_photos": active_photos,
                "remaining_budget": remaining_budget,
                "requested_photos": budget,
                "preflight_identities": preflight.identity_count,
                "preflight_photos": preflight.photo_count,
                "preflight_duplicates": preflight.duplicate_photo_count,
                "preflight_corrupt": preflight.corrupt_paths_count,
                "shards": [],
                "probe_latencies_ms": [],
            },
        )
        db.add(parent)
        await db.flush()

        shards: list[dict[str, Any]] = []
        per_shard_budget = math.ceil(budget / len(_BULK_WORKERS))
        for idx, worker in enumerate(_BULK_WORKERS):
            idempotency = f"vggface-bulk:{parent.process_id}:shard:{idx}"
            shard_process_id = str(_shard_process_id(idempotency))
            shards.append(
                {
                    "worker_id": worker,
                    "shard_index": idx,
                    "idempotency_key": idempotency,
                    "process_id": shard_process_id,
                    "status": "queued",
                    "max_photos": per_shard_budget,
                }
            )
        parent.summary["shards"] = shards
        flag_modified(parent, "summary")
        await db.commit()

    return VggfaceJobStartResult(
        job_id=str(parent.process_id),
        starting_active_photos=active_photos,
        target_total_active_photos=target,
        remaining_budget=remaining_budget,
        assigned_workers=_BULK_WORKERS,
    )


async def start_lfw_job(
    *,
    max_photos: int | None = None,
) -> VggfaceJobStartResult:
    """Create a durable LFW parent job and queue shard work."""
    active_photos = await _active_photo_count()
    target = settings.lfw_target_active_photos
    remaining_budget = max(0, target - active_photos)

    root = settings.lfw_dataset_path
    num_identities, num_photos = await asyncio.to_thread(expected_cardinality, root)
    requested_budget = max_photos if max_photos is not None else num_photos
    budget = min(requested_budget, remaining_budget, num_photos)
    if budget <= 0:
        raise ValueError(
            f"LFW photo budget exhausted (active={active_photos}, target={target})"
        )

    async with db_module.AsyncSessionLocal() as db:
        parent = ProcessRecord(
            process_type="lfw_bulk",
            status="queued",
            summary={
                "dataset_type": "lfw",
                "assigned_workers": _BULK_WORKERS,
                "requested_parallelism": len(_BULK_WORKERS),
                "target_total_active_photos": target,
                "starting_active_photos": active_photos,
                "remaining_budget": remaining_budget,
                "requested_photos": budget,
                "preflight_identities": num_identities,
                "preflight_photos": num_photos,
                "shards": [],
                "probe_latencies_ms": [],
            },
        )
        db.add(parent)
        await db.flush()

        shards: list[dict[str, Any]] = []
        per_shard_budget = math.ceil(budget / len(_BULK_WORKERS))
        for idx, worker in enumerate(_BULK_WORKERS):
            idempotency = f"lfw-bulk:{parent.process_id}:shard:{idx}"
            shard_process_id = str(_shard_process_id(idempotency))
            shards.append(
                {
                    "worker_id": worker,
                    "shard_index": idx,
                    "idempotency_key": idempotency,
                    "process_id": shard_process_id,
                    "status": "queued",
                    "max_photos": per_shard_budget,
                    "dataset_type": "lfw",
                }
            )
        parent.summary["shards"] = shards
        flag_modified(parent, "summary")
        await db.commit()

    return VggfaceJobStartResult(
        job_id=str(parent.process_id),
        starting_active_photos=active_photos,
        target_total_active_photos=target,
        remaining_budget=remaining_budget,
        assigned_workers=_BULK_WORKERS,
    )


async def start_casia_job(
    *,
    max_photos: int | None = None,
) -> VggfaceJobStartResult:
    """Create a durable CASIA-WebFace parent job and queue shard work."""
    active_photos = await _active_photo_count()
    target = settings.casia_target_active_photos
    remaining_budget = max(0, target - active_photos)

    root = settings.casia_dataset_path
    identities = await asyncio.to_thread(build_casia_manifest, root)
    num_identities = len(identities)
    num_photos = sum(len(i.photos) for i in identities)
    requested_budget = max_photos if max_photos is not None else num_photos
    budget = min(requested_budget, remaining_budget, num_photos)
    if budget <= 0:
        raise ValueError(
            f"CASIA photo budget exhausted (active={active_photos}, target={target})"
        )

    async with db_module.AsyncSessionLocal() as db:
        parent = ProcessRecord(
            process_type="casia_bulk",
            status="queued",
            summary={
                "dataset_type": "casia",
                "assigned_workers": _BULK_WORKERS,
                "requested_parallelism": len(_BULK_WORKERS),
                "target_total_active_photos": target,
                "starting_active_photos": active_photos,
                "remaining_budget": remaining_budget,
                "requested_photos": budget,
                "preflight_identities": num_identities,
                "preflight_photos": num_photos,
                "shards": [],
                "probe_latencies_ms": [],
            },
        )
        db.add(parent)
        await db.flush()

        shards: list[dict[str, Any]] = []
        per_shard_budget = math.ceil(budget / len(_BULK_WORKERS))
        for idx, worker in enumerate(_BULK_WORKERS):
            idempotency = f"casia-bulk:{parent.process_id}:shard:{idx}"
            shard_process_id = str(_shard_process_id(idempotency))
            shards.append(
                {
                    "worker_id": worker,
                    "shard_index": idx,
                    "idempotency_key": idempotency,
                    "process_id": shard_process_id,
                    "status": "queued",
                    "max_photos": per_shard_budget,
                    "dataset_type": "casia",
                }
            )
        parent.summary["shards"] = shards
        flag_modified(parent, "summary")
        await db.commit()

    return VggfaceJobStartResult(
        job_id=str(parent.process_id),
        starting_active_photos=active_photos,
        target_total_active_photos=target,
        remaining_budget=remaining_budget,
        assigned_workers=_BULK_WORKERS,
    )


def _build_worker_payload(
    shard: dict[str, Any],
) -> dict[str, Any]:
    dataset_type = shard.get("dataset_type", "vggface")
    if dataset_type == "lfw":
        source: dict[str, Any] = {"type": "local_lfw", "path": str(settings.lfw_dataset_path)}
    elif dataset_type == "casia":
        source = {"type": "local_casia", "path": str(settings.casia_dataset_path)}
    else:
        source = {"type": "local_vggface", "path": str(settings.vggface_dataset_path)}
    if shard.get("resume_after_identity_key"):
        source["resumeAfterIdentityKey"] = shard["resume_after_identity_key"]
    return {
        "jobId": shard["process_id"],
        "idempotencyKey": shard["idempotency_key"],
        "source": source,
        "datasetType": dataset_type,
        "mode": "import",
        "requestedParallelism": len(_BULK_WORKERS),
        "assignedWorkers": _BULK_WORKERS,
        "shardIndex": shard["shard_index"],
        "maxPhotos": shard["max_photos"],
    }


_TERMINAL_SHARD_STATUSES = {"completed", "failed", "cancelled"}
_POLL_INTERVAL_SECONDS = 2.0


async def _dispatch_one_shard(
    client: httpx.AsyncClient,
    parent_id: __import__("uuid").UUID,
    shard: dict[str, Any],
) -> None:
    worker = shard["worker_id"]
    post_url = f"http://{worker}:8001/internal/v1/jobs"
    process_id = shard["process_id"]
    poll_url = f"http://{worker}:8001/internal/v1/jobs/{process_id}"
    payload = _build_worker_payload(shard)
    logger.info("dispatching shard %s to %s", shard["shard_index"], worker)

    try:
        resp = await client.post(post_url, json=payload, timeout=None)
        resp.raise_for_status()
        body = resp.json()
        shard["status"] = body.get("status", "unknown")
        shard["worker_response"] = {
            "status": body.get("status"),
            "progress": body.get("progress", {}),
        }
    except Exception as exc:
        logger.exception("shard %s failed to start on %s", shard["shard_index"], worker)
        shard["status"] = "failed"
        shard["worker_response"] = {"error": f"{exc.__class__.__name__}: {exc}"}
        await _persist_shard_update(parent_id, shard)
        return

    # The worker runs the job in the background; poll its status endpoint
    # until it reaches a terminal state.
    while shard["status"] not in _TERMINAL_SHARD_STATUSES:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        try:
            resp = await client.get(poll_url, timeout=10.0)
            resp.raise_for_status()
            body = resp.json()
            shard["status"] = body.get("status", shard["status"])
            shard["worker_response"] = {
                "status": body.get("status"),
                "progress": body.get("progress", {}),
            }
        except Exception as exc:
            logger.warning(
                "shard %s poll on %s failed: %s",
                shard["shard_index"],
                worker,
                exc.__class__.__name__,
            )
        await _persist_shard_update(parent_id, shard)

    logger.info(
        "shard %s finished on %s: %s",
        shard["shard_index"],
        worker,
        shard["status"],
    )


async def _persist_shard_update(
    parent_id: __import__("uuid").UUID,
    shard: dict[str, Any],
) -> None:
    async with db_module.AsyncSessionLocal() as db:
        parent = await db.get(ProcessRecord, parent_id)
        if parent is not None:
            parent.summary["shards"] = [
                s if s["shard_index"] != shard["shard_index"] else shard
                for s in parent.summary["shards"]
            ]
            flag_modified(parent, "summary")
            await db.commit()


async def _probe_recognition_latency(parent_id: __import__("uuid").UUID) -> None:
    """Poll GPU 0 recognition latency while bulk enrollment runs.

    Uses the public /faces/recognize endpoint, which routes through the online
    GPU worker.  Latencies are stored in the parent summary for the UI.
    """
    probe_image_path = Path("/app/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
    image_bytes: bytes | None = None
    try:
        image_bytes = probe_image_path.read_bytes()
    except Exception:
        logger.warning("recognition probe image not available")
        return

    latencies: list[float] = []
    while True:
        await asyncio.sleep(_PROBE_INTERVAL_SECONDS)
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
                files = {"image": ("probe.jpg", image_bytes, "image/jpeg")}
                resp = await client.post(_RECOGNITION_PROBE_URL, files=files)
                _ = resp.status_code
        except Exception as exc:
            logger.warning("recognition probe failed: %s", exc.__class__.__name__)
            continue
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)
        # keep last 60 samples
        latencies = latencies[-60:]

        async with db_module.AsyncSessionLocal() as db:
            parent = await db.get(ProcessRecord, parent_id)
            if parent is None:
                break
            if parent.status in ("cancelled", "completed", "failed"):
                break
            parent.summary["probe_latencies_ms"] = latencies
            if latencies:
                parent.summary["probe_p50_ms"] = _percentile(latencies, 0.5)
                parent.summary["probe_p95_ms"] = _percentile(latencies, 0.95)
            flag_modified(parent, "summary")
            await db.commit()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


async def dispatch_shards(parent_id: __import__("uuid").UUID) -> None:
    """Dispatch all shards to GPU workers and monitor recognition latency."""
    async with db_module.AsyncSessionLocal() as db:
        parent = await db.get(ProcessRecord, parent_id)
        if parent is None:
            raise RuntimeError(f"parent job {parent_id} not found")
        parent.status = "running"
        parent.started_at = _utc_now()
        await db.commit()
        shards = parent.summary.get("shards", [])

    probe_task = asyncio.create_task(_probe_recognition_latency(parent_id))
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            await asyncio.gather(
                *(_dispatch_one_shard(client, parent_id, shard) for shard in shards)
            )
    finally:
        probe_task.cancel()
        try:
            await probe_task
        except asyncio.CancelledError:
            pass

    async with db_module.AsyncSessionLocal() as db:
        parent = await db.get(ProcessRecord, parent_id)
        if parent is None:
            return
        await _aggregate_parent(parent, db)
        parent.completed_at = _utc_now()
        if parent.summary.get("cancelled", False):
            parent.status = "cancelled"
        elif parent.summary.get("failed_shards", 0):
            parent.status = "failed"
        else:
            parent.status = "completed"
        await db.commit()


async def _aggregate_shards(
    parent: ProcessRecord,
    db: db_module.AsyncSession,
) -> dict[str, Any]:
    """Read-only aggregation of shard progress.  Returns computed summary fields."""
    shards = parent.summary.get("shards", [])
    total_enrolled = 0
    total_duplicate = 0
    total_no_face = 0
    total_errors = 0
    total_scanned = 0
    total_processed = 0
    total_discovered = 0
    failed_shards = 0
    cancelled_shards = 0
    running_shards = 0

    for shard in shards:
        process_id = _shard_process_id(shard["idempotency_key"])
        record = await db.get(ProcessRecord, process_id)
        if record is not None:
            shard["status"] = record.status
            shard["progress"] = record.summary.get("progress", {})
            progress = record.summary.get("progress", {})
            total_enrolled += progress.get("enrolled", progress.get("faces_enrolled", 0))
            total_duplicate += progress.get("faces_duplicate", 0)
            total_no_face += progress.get("no_face", 0)
            decode_err = progress.get("decode_error", 0)
            persist_err = progress.get("persistence_error", 0)
            failed_err = progress.get("failed", 0)
            total_errors += decode_err + persist_err + failed_err + progress.get("errors", 0)
            total_scanned += progress.get("processed", progress.get("total_scanned", 0))
            total_processed += progress.get("processed", progress.get("total_processed", 0))
            total_discovered += progress.get("discovered_photos", progress.get("photos", shard.get("max_photos", 0)))
            if record.status == "failed":
                failed_shards += 1
            elif record.status == "cancelled":
                cancelled_shards += 1
            elif record.status == "running":
                running_shards += 1
        else:
            shard["status"] = "queued"

    # Freeze terminal duration; keep running duration live.
    elapsed_seconds = 0.0
    if parent.started_at:
        if parent.status in ("completed", "cancelled", "failed") and parent.completed_at:
            elapsed_seconds = (parent.completed_at - parent.started_at).total_seconds()
        else:
            elapsed_seconds = (_utc_now() - parent.started_at).total_seconds()

    scanned_rate = round(total_scanned / elapsed_seconds, 2) if elapsed_seconds > 0 else 0.0
    processed_rate = round(total_processed / elapsed_seconds, 2) if elapsed_seconds > 0 else 0.0
    enrolled_rate = round(total_enrolled / elapsed_seconds, 2) if elapsed_seconds > 0 else 0.0
    duplicate_rate = round(total_duplicate / elapsed_seconds, 2) if elapsed_seconds > 0 else 0.0

    active_now = await _active_photo_count()
    starting = parent.summary.get("starting_active_photos", 0)

    return {
        "total_enrolled": total_enrolled,
        "total_duplicate": total_duplicate,
        "total_no_face": total_no_face,
        "total_errors": total_errors,
        "total_scanned": total_scanned,
        "total_processed": total_processed,
        "total_discovered": total_discovered,
        "total_in_flight": max(0, total_discovered - total_scanned),
        "total_rejected": 0,
        "total_corrupt": 0,
        "failed_shards": failed_shards,
        "cancelled_shards": cancelled_shards,
        "running_shards": running_shards,
        "completed_shards": len(shards) - failed_shards - cancelled_shards - running_shards,
        "cancelled": cancelled_shards > 0 and running_shards == 0 and failed_shards == 0,
        "elapsed_seconds": elapsed_seconds,
        "avg_photos_per_second": enrolled_rate,
        "scanned_photos_per_second": scanned_rate,
        "processed_photos_per_second": processed_rate,
        "enrolled_photos_per_second": enrolled_rate,
        "duplicate_photos_per_second": duplicate_rate,
        "current_active_photos": active_now,
        "photos_added_by_job": max(0, active_now - starting),
    }


async def _aggregate_parent(parent: ProcessRecord, db: db_module.AsyncSession) -> None:
    """Write the aggregated summary back to the parent record (used by the controller)."""
    computed = await _aggregate_shards(parent, db)
    parent.summary.update(computed)
    active_now = await _active_photo_count()
    parent.summary["current_active_photos"] = active_now
    parent.summary["photos_added_by_job"] = max(
        0, active_now - parent.summary.get("starting_active_photos", 0)
    )
    flag_modified(parent, "summary")


async def _get_job(job_id: __import__("uuid").UUID, process_type: str) -> ProcessRecord | None:
    async with db_module.AsyncSessionLocal() as db:
        parent = await db.get(ProcessRecord, job_id)
        if parent is None or parent.process_type != process_type:
            return None
        computed = await _aggregate_shards(parent, db)
        parent.summary = {**parent.summary, **computed}
        # Refresh events for the response
        await db.refresh(parent, attribute_names=["events"])
        return parent


async def get_vggface_job(job_id: __import__("uuid").UUID) -> ProcessRecord | None:
    return await _get_job(job_id, "vggface_bulk")


async def get_lfw_job(job_id: __import__("uuid").UUID) -> ProcessRecord | None:
    return await _get_job(job_id, "lfw_bulk")


async def get_casia_job(job_id: __import__("uuid").UUID) -> ProcessRecord | None:
    return await _get_job(job_id, "casia_bulk")


_BULK_PROCESS_TYPES = {"vggface_bulk", "lfw_bulk", "casia_bulk"}


async def request_cancellation(job_id: __import__("uuid").UUID) -> ProcessRecord | None:
    async with db_module.AsyncSessionLocal() as db:
        parent = await db.get(ProcessRecord, job_id)
        if parent is None or parent.process_type not in _BULK_PROCESS_TYPES:
            return None
        if parent.status in ("cancelled", "completed", "failed"):
            return parent
        parent.status = "cancel_requested"
        parent.updated_at = _utc_now()
        await db.commit()
        shards = parent.summary.get("shards", [])

    async with httpx.AsyncClient(timeout=30.0) as client:
        await asyncio.gather(
            *(
                _send_cancel(client, shard)
                for shard in shards
            )
        )

    async with db_module.AsyncSessionLocal() as db:
        parent = await db.get(ProcessRecord, job_id)
        if parent is not None:
            parent.status = "cancelling"
            parent.updated_at = _utc_now()
            await _aggregate_parent(parent, db)
            await db.commit()
        return parent


async def _send_cancel(client: httpx.AsyncClient, shard: dict[str, Any]) -> None:
    worker = shard["worker_id"]
    process_id = shard["process_id"]
    url = f"http://{worker}:8001/internal/v1/jobs/{process_id}/cancel"
    try:
        await client.post(url)
    except Exception as exc:
        logger.warning("cancel request to %s failed: %s", worker, exc.__class__.__name__)


async def resume_vggface_job(job_id: __import__("uuid").UUID) -> ProcessRecord | None:
    async with db_module.AsyncSessionLocal() as db:
        parent = await db.get(ProcessRecord, job_id)
        if parent is None or parent.process_type != "vggface_bulk":
            return None
        if parent.status == "running":
            return parent
        parent.status = "queued"
        parent.completed_at = None
        parent.updated_at = _utc_now()

        # Reset terminal child shards to pending and attach identity-level
        # resume checkpoints from their last durable progress.
        shards = parent.summary.get("shards", [])
        for shard in shards:
            shard["status"] = "queued"
            shard["worker_response"] = {}
            process_id = _shard_process_id(shard["idempotency_key"])
            child = await db.get(ProcessRecord, process_id)
            if child is not None and child.status in ("completed", "cancelled", "failed"):
                child.status = "pending"
                child.completed_at = None
                child.error_message = None
                child.updated_at = _utc_now()
            progress = ({} if child is None else child.summary.get("progress", {}))
            last_key = progress.get("last_completed_identity_key")
            if last_key:
                shard["resume_after_identity_key"] = last_key
            else:
                shard.pop("resume_after_identity_key", None)
        parent.summary["shards"] = shards
        flag_modified(parent, "summary")
        await db.commit()

    asyncio.create_task(dispatch_shards(job_id))
    return await get_vggface_job(job_id)
