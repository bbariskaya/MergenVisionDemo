"""Process-per-GPU bulk enrollment benchmark for the full LFW dataset.

Each GPU runs in a separate OS process that sees exactly one physical GPU
(device_id=0 inside its own container/context).  All workers share the same
PostgreSQL, MinIO and Qdrant services.  This avoids the shared Python-process
contention problems of the thread-based benchmark.

Examples
--------
Single GPU:
    MODEL_PACK=retinaface_r50 python scripts/benchmark_bulk_enroll_multiprocess.py --num-gpus 1

Three GPUs:
    MODEL_PACK=retinaface_r50 python scripts/benchmark_bulk_enroll_multiprocess.py --num-gpus 3
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import multiprocessing as mp
import os
import secrets
import time
from pathlib import Path
from typing import Any

LFW_ROOT = Path("/app/lfw/lfw-deepfunneled/lfw-deepfunneled")
CHUNK_SIZE = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=3)
    return parser.parse_args()


def load_paths(limit: int | None = None) -> list[Path]:
    paths: list[Path] = []
    for person_dir in sorted(LFW_ROOT.iterdir()):
        if not person_dir.is_dir():
            continue
        paths.extend(sorted(person_dir.glob("*.jpg")))
        if limit and len(paths) >= limit:
            break
    return paths[:limit] if limit else paths


class BulkEnrollItem:
    def __init__(self, image_bytes: bytes, name: str, national_id: str, metadata: dict[str, Any] | None):
        self.image_bytes = image_bytes
        self.name = name
        self.national_id = national_id
        self.metadata = metadata


def worker(gpu_id: int, paths: list[Path], run_id: str) -> dict[str, float | int]:
    """Run one GPU's share in its own process with a single visible GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Delayed imports so the CUDA runtime only sees the visible device.
    from app.infrastructure import db as db_module
    from app.infrastructure.minio import PhotoStorage
    from app.infrastructure.qdrant import FaceVectorStore
    from app.ml.gpu.face_pipeline import GpuFacePipeline
    from app.services.face_service import BulkEnrollItem, FaceService

    db_module.configure_engine()

    async def _run() -> dict[str, float | int]:
        storage = PhotoStorage()
        await storage.initialize()
        vector_store = FaceVectorStore()
        await vector_store.initialize()

        pipeline = GpuFacePipeline(device_id=0)
        gpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"gpu{gpu_id}_extract"
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(gpu_executor, pipeline.warmup)
        lock = asyncio.Lock()

        items = [
            BulkEnrollItem(
                image_bytes=path.read_bytes(),
                name=f"Person_gpu{gpu_id}_{i:05d}",
                national_id=f"{run_id}G{gpu_id}N{i:08d}",
                metadata={"source": str(path.name), "gpu": gpu_id},
            )
            for i, path in enumerate(paths)
        ]

        total_images = 0
        total_faces = 0
        total_errors = 0
        total_extract_ms = 0.0
        total_io_ms = 0.0

        async with db_module.AsyncSessionLocal() as session:
            service = FaceService(
                db=session,
                storage=storage,
                vector_store=vector_store,
                pipeline=pipeline,
                pipeline_lock=lock,
                gpu_executor=gpu_executor,
            )
            block = 0
            for chunk_start in range(0, len(items), CHUNK_SIZE):
                block += 1
                chunk = items[chunk_start : chunk_start + CHUNK_SIZE]
                records, errors, timings = await service.bulk_enroll(chunk)
                total_images += len(chunk)
                total_faces += len(records)
                total_errors += len(errors)
                total_extract_ms += timings.get("extraction_ms", 0.0)
                total_io_ms += timings.get("io_ms", 0.0)
                print(
                    f"  gpu{gpu_id} block{block}: {len(records)} enrolled, "
                    f"{len(errors)} errors, "
                    f"extract={timings.get('extraction_ms', 0.0):.0f}ms "
                    f"io={timings.get('io_ms', 0.0):.0f}ms",
                    flush=True,
                )

        await vector_store.close()
        await db_module.dispose_engine()
        gpu_executor.shutdown(wait=True)
        return {
            "gpu": gpu_id,
            "images": total_images,
            "faces": total_faces,
            "errors": total_errors,
            "extract_ms": total_extract_ms,
            "io_ms": total_io_ms,
        }

    return asyncio.run(_run())


def main() -> int:
    args = parse_args()
    paths = load_paths(limit=args.limit)
    print(f"LFW images: {len(paths)}")
    print(f"Model pack: {os.environ.get('MODEL_PACK', 'default')}")
    print(f"GPUs used: {args.num_gpus} (separate processes)")

    run_id = secrets.token_hex(8)
    print(f"Run id: {run_id}")

    n = len(paths)
    chunk_size = (n + args.num_gpus - 1) // args.num_gpus
    chunks = [paths[i : i + chunk_size] for i in range(0, n, chunk_size)]

    mp.set_start_method("spawn", force=True)
    t0 = time.perf_counter()
    with mp.Pool(processes=args.num_gpus) as pool:
        async_results = [
            pool.apply_async(worker, (gpu_id, chunk, run_id))
            for gpu_id, chunk in enumerate(chunks[: args.num_gpus])
        ]
        results = [r.get() for r in async_results]
    total_t = time.perf_counter() - t0

    total_images = sum(int(r["images"]) for r in results)
    total_faces = sum(int(r["faces"]) for r in results)
    total_errors = sum(int(r["errors"]) for r in results)
    total_extract_ms = sum(r["extract_ms"] for r in results)
    total_io_ms = sum(r["io_ms"] for r in results)

    print(f"\nTotal: {total_faces} faces enrolled from {total_images} images")
    print(f"Errors: {total_errors}")
    print(f"Wall time: {total_t:.2f}s")
    print(f"GPU extraction: {total_extract_ms / 1000:.2f}s")
    print(f"IO (upload + DB + Qdrant): {total_io_ms / 1000:.2f}s")
    print(f"Images/sec: {total_images / total_t:.1f}")
    print(f"Faces/sec: {total_faces / total_t:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
