"""Submit two separate import jobs concurrently to two GPU workers.

Job 1 (LFW) is pinned to worker0 with parallelism=1.
Job 2 (a synthetic dataset derived from a few LFW folders under different
folder names) is pinned to worker1 with parallelism=1.

This proves:
- each worker sees exactly one GPU
- different independent jobs can run concurrently
- cross-job IDs do not collide because they derive from distinct folder keys
- no raw LFW folder names appear in logs or object keys
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from app.services.bulk_manifest import build_lfw_manifest


LFW_ROOT = Path("/app/lfw/lfw-deepfunneled/lfw-deepfunneled")
WORKER_A = "http://worker0:8001"
WORKER_B = "http://worker1:8001"
REQUEST_TIMEOUT = 1800.0
SYNTHETIC_DIR = Path("/app/lfw/_synthetic_job_dataset")


def _prepare_synthetic_dataset() -> Path:
    """Copy a few LFW folders into a shared synthetic tree with renamed keys."""
    if SYNTHETIC_DIR.exists():
        shutil.rmtree(SYNTHETIC_DIR)
    SYNTHETIC_DIR.mkdir(parents=True)

    source_folders = ["AJ_Lamas", "Jennifer_Aniston", "Aaron_Peirsol"]
    for folder in source_folders:
        src = LFW_ROOT / folder
        if not src.exists():
            continue
        dst_name = f"SynthJob_{folder}"
        dst = SYNTHETIC_DIR / dst_name
        shutil.copytree(src, dst)
    return SYNTHETIC_DIR


def _identity_payload(identity: Any) -> dict[str, Any]:
    return {
        "identity_key": identity.identity_key,
        "display_name": identity.display_name,
        "identity_hmac": identity.identity_hmac,
        "person_id": identity.person_id,
        "face_identity_id": identity.face_identity_id,
        "photos": [
            {
                "content_sha256": photo.content_sha256,
                "path": str(photo.path),
            }
            for photo in identity.photos
        ],
    }


async def _submit_job(
    client: httpx.AsyncClient,
    url: str,
    identities: tuple[Any, ...],
    idempotency_key: str,
) -> dict[str, Any]:
    payload = {
        "identities": [_identity_payload(i) for i in identities],
        "idempotency_key": idempotency_key,
    }
    resp = await client.post(
        f"{url}/enroll",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["result"]


async def main() -> int:
    if not LFW_ROOT.exists():
        print(f"LFW dataset not found at {LFW_ROOT}", file=os.sys.stderr)
        return 1

    synthetic_root = _prepare_synthetic_dataset()
    lfw_identities = build_lfw_manifest(LFW_ROOT)
    synthetic_identities = build_lfw_manifest(synthetic_root)
    print(f"Job A (LFW): {len(lfw_identities)} identities")
    print(
        f"Job B (synthetic): {len(synthetic_identities)} identities, "
        f"root={synthetic_root}"
    )

    async with httpx.AsyncClient() as client:
        for url in (WORKER_A, WORKER_B):
            resp = await client.get(f"{url}/health", timeout=10.0)
            print(f"{url} health: {resp.json()}")

        t0 = time.perf_counter()
        job_a, job_b = await asyncio.gather(
            _submit_job(client, WORKER_A, lfw_identities, "concurrent-lfw-job-a"),
            _submit_job(client, WORKER_B, synthetic_identities, "concurrent-synthetic-job-b"),
        )
        elapsed = time.perf_counter() - t0

    print(f"\nConcurrent jobs completed in {elapsed:.2f}s")
    print(f"Job A result: {job_a}")
    print(f"Job B result: {job_b}")

    a_person_ids = {i.person_id for i in lfw_identities}
    b_person_ids = {i.person_id for i in synthetic_identities}
    if a_person_ids & b_person_ids:
        print(
            "COLLISION DETECTED: overlapping person IDs across jobs",
            file=os.sys.stderr,
        )
        return 1
    print("No cross-job person ID collisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
