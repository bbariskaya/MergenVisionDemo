"""Durable GPU-worker LFW bulk enrollment benchmark.

Default mode pins the entire LFW import to a single long-lived worker
(parallelism=1).  Use ``--parallelism N`` to opt-in to multi-GPU sharding,
partitioned by stable hash of ``person_id`` across N assigned workers.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings
from app.services.bulk_manifest import (
    build_lfw_manifest,
    expected_cardinality,
    shard_by_person_id,
)


LFW_ROOT = Path("/app/lfw/lfw-deepfunneled/lfw-deepfunneled")
REQUEST_TIMEOUT = 1800.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument(
        "--worker-urls",
        type=str,
        default=None,
        help="comma-separated worker URLs (default: http://worker0:8001,...http://workerN-1:8001)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--idempotency-key",
        type=str,
        default="lfw-bulk-v1",
        help="idempotency key written to process_record metadata",
    )
    return parser.parse_args()


def _worker_urls(parallelism: int, override: str | None) -> list[str]:
    if override:
        return [url.strip() for url in override.split(",")]
    return [f"http://worker{i}:8001" for i in range(parallelism)]


def _limit_identities(
    identities: tuple[Any, ...],
    limit: int,
) -> tuple[Any, ...]:
    kept: list[Any] = []
    count = 0
    for identity in identities:
        if count >= limit:
            break
        take = min(len(identity.photos), limit - count)
        kept.append(
            identity.__class__(
                identity_key=identity.identity_key,
                display_name=identity.display_name,
                identity_hmac=identity.identity_hmac,
                person_id=identity.person_id,
                face_identity_id=identity.face_identity_id,
                photos=identity.photos[:take],
            )
        )
        count += take
    return tuple(kept)


async def _health_check(client: httpx.AsyncClient, url: str) -> bool:
    try:
        resp = await client.get(f"{url}/health", timeout=10.0)
        return resp.status_code == 200 and resp.json().get("status") == "ready"
    except Exception:
        return False


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


async def _enroll_shard(
    client: httpx.AsyncClient,
    url: str,
    shard: tuple[Any, ...],
    *,
    idempotency_key: str,
) -> dict[str, Any]:
    payload = {
        "identities": [_identity_payload(i) for i in shard],
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
    args = parse_args()
    if not LFW_ROOT.exists():
        print(f"LFW dataset not found at {LFW_ROOT}", file=os.sys.stderr)
        return 1

    print(f"Model pack: {settings.model_pack}")
    persons, photos = expected_cardinality(LFW_ROOT)
    print(f"Expected: {persons} persons, {photos} photos")

    if args.limit:
        print(f"Limit: approx first {args.limit} photos")

    if args.dry_run:
        print("Dry run; exiting without enrolling.")
        return 0

    identities = build_lfw_manifest(LFW_ROOT)
    if args.limit:
        identities = _limit_identities(identities, args.limit)

    parallelism = max(1, args.parallelism)
    urls = _worker_urls(parallelism, args.worker_urls)
    if len(urls) < parallelism:
        print(
            f"warning: only {len(urls)} worker URLs for parallelism={parallelism}",
            file=os.sys.stderr,
        )
        parallelism = len(urls)

    if parallelism == 1:
        shards = (identities,)
    else:
        shards = shard_by_person_id(identities, parallelism)
    print(f"Parallelism: {parallelism}; shards: {[len(s) for s in shards]} identities")

    async with httpx.AsyncClient() as client:
        print("Waiting for worker health...")
        healthy = await asyncio.gather(
            *(_health_check(client, url) for url in urls[:parallelism])
        )
        for url, ok in zip(urls[:parallelism], healthy):
            print(f"  {url}: {'ready' if ok else 'NOT READY'}")
        if not all(healthy):
            print("Not all workers are healthy", file=os.sys.stderr)
            return 1

        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(
                _enroll_shard(
                    client,
                    url,
                    shard,
                    idempotency_key=args.idempotency_key,
                )
                for url, shard in zip(urls[:parallelism], shards)
            )
        )
        total_t = time.perf_counter() - t0

    total_enrolled = sum(int(r.get("faces_enrolled", 0)) for r in results)
    total_duplicate = sum(int(r.get("faces_duplicate", 0)) for r in results)
    total_no_face = sum(int(r.get("no_face", 0)) for r in results)
    total_photos = sum(int(r.get("photos", 0)) for r in results)

    print(f"\nTotal enrolled: {total_enrolled} faces from {total_photos} photos")
    print(f"Duplicates: {total_duplicate}")
    print(f"No face: {total_no_face}")
    print(f"Wall time: {total_t:.2f}s")
    print(f"Photos/sec: {total_photos / total_t:.1f}" if total_t else "n/a")
    print(f"Faces/sec: {total_enrolled / total_t:.1f}" if total_t else "n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
