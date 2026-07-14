#!/usr/bin/env python3
"""Offline CASIA-WebFace bulk enrollment starter + CLI monitor.

Runs inside a worker container while the API is stopped.  It creates the parent
job record, dispatches shards to gpu-worker-{0,1,2}, and prints live progress
(enrolled count, photos/sec, ETA) until the job finishes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone

import httpx
from sqlalchemy import desc, select
from sqlalchemy.orm.attributes import flag_modified

sys.path.insert(0, "/app")

from app.core.config import settings
from app.domain.models import ProcessRecord
from app.infrastructure import db as db_module
from app.services.bulk_orchestrator import (
    _aggregate_parent,
    dispatch_shards,
    get_casia_job,
    start_casia_job,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("casia_offline")

_LOG_FILE = "/tmp/casia_offline.log"
_WORKER_PORT = int(os.environ.get("API_PORT", "8001"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_duration(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}s")
    if m or h:
        parts.append(f"{m}d")
    parts.append(f"{s}sn")
    return " ".join(parts)


def _fmt_eta(remaining: int, rate: float) -> str:
    if not rate or remaining <= 0:
        return "—"
    return _fmt_duration(remaining / rate)


async def _wait_for_workers(workers: list[str], timeout: float = 300.0) -> bool:
    logger.info("Worker'lar hazır olana kadar bekleniyor: %s", ", ".join(workers))
    async with httpx.AsyncClient(timeout=10.0) as client:
        deadline = time.time() + timeout
        while time.time() < deadline:
            all_ready = True
            for w in workers:
                url = f"http://{w}:{_WORKER_PORT}/health/ready"
                try:
                    r = await client.get(url)
                    data = r.json()
                    if data.get("status") != "ready":
                        logger.info("%s durum=%s", w, data.get("status"))
                        all_ready = False
                except Exception as exc:
                    logger.debug("%s erişilemiyor: %s", w, exc)
                    all_ready = False
            if all_ready:
                logger.info("Tüm worker'lar hazır.")
                return True
            await asyncio.sleep(2.0)
    logger.error("Worker'lar %s saniye içinde hazır olmadı", int(timeout))
    return False


async def _finalize(parent_id: uuid.UUID) -> ProcessRecord | None:
    async with db_module.AsyncSessionLocal() as db:
        parent = await db.get(ProcessRecord, parent_id)
        if parent is None:
            return None
        await _aggregate_parent(parent, db)
        if parent.status in ("completed", "failed", "cancelled"):
            parent.completed_at = _now()
        await db.commit()
        return parent


async def _monitor(job_id: uuid.UUID) -> None:
    db_module.configure_engine()
    samples: deque[tuple[float, int]] = deque()
    window_seconds = 60.0

    with open(_LOG_FILE, "a", encoding="utf-8") as fh:
        while True:
            try:
                record = await get_casia_job(job_id)
                if record is None:
                    await asyncio.sleep(5.0)
                    continue

                summary = record.summary or {}
                active = summary.get("current_active_photos", 0)
                start_active = summary.get("starting_active_photos", 0)
                requested = summary.get("requested_photos", 0)
                added = max(0, active - start_active)
                elapsed = summary.get("elapsed_seconds", 0.0)
                shards = summary.get("shards", [])
                shard_line = ", ".join(
                    f"{sh.get('worker_id','?')}:{sh.get('status','?')}"
                    for sh in shards
                )

                now = time.time()
                samples.append((now, added))
                while samples and samples[0][0] < now - window_seconds:
                    samples.popleft()
                if len(samples) >= 2:
                    dt = samples[-1][0] - samples[0][0]
                    d_added = samples[-1][1] - samples[0][1]
                    rate = d_added / dt if dt > 0 else 0.0
                else:
                    rate = added / elapsed if elapsed > 0 else 0.0

                eta = _fmt_eta(max(0, requested - added), rate)

                line = (
                    f"[{_now().strftime('%H:%M:%S')}] "
                    f"aktif={active:,} "
                    f"bu işlemde eklendi={added:,}/{requested:,} "
                    f"hız={rate:.1f} foto/sn "
                    f"Kalan~{eta} "
                    f"geçen={_fmt_duration(elapsed)} | "
                    f"{shard_line}"
                )
                print(line)
                fh.write(line + "\n")
                fh.flush()

                if record.status in ("completed", "failed", "cancelled"):
                    final = await _finalize(job_id)
                    final_status = final.status if final else record.status
                    final_line = f"CASIA job SON DURUM: {final_status.upper()}"
                    print(final_line)
                    fh.write(final_line + "\n")
                    fh.flush()
                    return

            except Exception as exc:
                logger.exception("monitor döngüsünde hata")
                fh.write(f"monitor hatası: {exc}\n")
                fh.flush()

            await asyncio.sleep(5.0)


async def main() -> int:
    workers = [
        w.strip()
        for w in os.environ.get("BULK_WORKERS", "gpu-worker-0,gpu-worker-1,gpu-worker-2").split(",")
        if w.strip()
    ]

    db_module.configure_engine()
    os.makedirs(os.path.dirname(_LOG_FILE) or "/tmp", exist_ok=True)

    with open(_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== CASIA offline başlatıldı {_now().isoformat()} ===\n")
        fh.flush()

    async with db_module.AsyncSessionLocal() as db:
        result = await db.execute(
            select(ProcessRecord)
            .where(ProcessRecord.process_type == "casia_bulk")
            .order_by(desc(ProcessRecord.created_at))
            .limit(1)
        )
        existing = result.scalar_one_or_none()

    if existing and existing.status == "running":
        job_id = existing.process_id
        print(f"Mevcut CASIA job devam ediyor: {job_id}; izleniyor.")
    elif existing and existing.status == "queued":
        job_id = existing.process_id
        print(f"Mevcut CASIA job sırada: {job_id}; shard'lar dağıtılıyor...")
        asyncio.create_task(dispatch_shards(job_id))
    else:
        if not await _wait_for_workers(workers):
            return 1
        print("Yeni CASIA bulk job oluşturuluyor...")
        result = await start_casia_job()
        job_id = uuid.UUID(result.job_id)
        print(f"Job oluşturuldu: {job_id}, shard'lar dağıtılıyor...")
        asyncio.create_task(dispatch_shards(job_id))

    await _monitor(job_id)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
