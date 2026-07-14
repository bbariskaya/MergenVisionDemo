"""Benchmark bulk enrollment of the full LFW dataset through FaceService.

Run inside the API container with the RetinaFace model pack:
    MODEL_PACK=retinaface_r50 python scripts/benchmark_bulk_enroll.py
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import os
import secrets
import time
from pathlib import Path

from app.core.config import settings
from app.infrastructure import db as db_module
from app.infrastructure.minio import PhotoStorage
from app.infrastructure.qdrant import FaceVectorStore
from app.ml.gpu.face_pipeline import GpuFacePipeline
from app.services.face_service import BulkEnrollItem, FaceService


LFW_ROOT = Path("/app/lfw/lfw-deepfunneled/lfw-deepfunneled")
CHUNK_SIZE = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="limit number of images")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=int(os.environ.get("NUM_GPUS", 0)),
        help="number of GPUs to use (0=auto-detect; 1-3=explicit)",
    )
    return parser.parse_args()


def detect_gpu_count() -> int:
    try:
        import nvidia_ml_py as nvml
        nvml.nvmlInit()
        count = int(nvml.nvmlDeviceGetCount())
        nvml.nvmlShutdown()
        return max(1, count)
    except Exception:
        pass
    try:
        from cuda.bindings import runtime as cuda_runtime

        err, count = cuda_runtime.cudaGetDeviceCount()
        if err == 0 and count > 0:
            return int(count)
    except Exception:
        pass
    return 1


def load_lfw_paths(limit: int | None = None) -> list[Path]:
    paths: list[Path] = []
    for person_dir in sorted(LFW_ROOT.iterdir()):
        if not person_dir.is_dir():
            continue
        paths.extend(sorted(person_dir.glob("*.jpg")))
        if limit and len(paths) >= limit:
            break
    return paths[:limit] if limit else paths


def build_items(paths: list[Path], run_id: str) -> list[BulkEnrollItem]:
    items: list[BulkEnrollItem] = []
    for i, path in enumerate(paths):
        items.append(
            BulkEnrollItem(
                image_bytes=path.read_bytes(),
                name=f"Person_{i:05d}",
                national_id=f"{run_id}N{i:08d}",
                metadata={"source": str(path.name)},
            )
        )
    return items


async def enroll_on_gpu(
    gpu_id: int,
    items: list[BulkEnrollItem],
) -> dict[str, float | int]:
    """Run one GPU's share in its own thread + event loop (isolated CUDA context)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _enroll_on_gpu_thread, gpu_id, items)


def _enroll_on_gpu_thread(
    gpu_id: int,
    items: list[BulkEnrollItem],
) -> dict[str, float | int]:
    return asyncio.run(_enroll_on_gpu_async(gpu_id, items))


async def _enroll_on_gpu_async(
    gpu_id: int,
    items: list[BulkEnrollItem],
) -> dict[str, float | int]:
    """Async worker inside a dedicated thread for one GPU."""
    db_module.configure_engine()
    storage = PhotoStorage()
    await storage.initialize()
    vector_store = FaceVectorStore()
    await vector_store.initialize()

    pipeline = GpuFacePipeline(device_id=gpu_id)
    gpu_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"gpu{gpu_id}_extract"
    )
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(gpu_executor, pipeline.warmup)
    lock = asyncio.Lock()

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
                f"io={timings.get('io_ms', 0.0):.0f}ms"
            )

        # Pipelines are intentionally not closed here; the per-thread CUDA
        # contexts live until the process exits to avoid cross-thread teardown
        # races in nvImageCodec/TensorRT.
        await vector_store.close()
        await db_module.dispose_engine()
    return {
        "images": total_images,
        "faces": total_faces,
        "errors": total_errors,
        "extract_ms": total_extract_ms,
        "io_ms": total_io_ms,
    }


async def main() -> int:
    args = parse_args()
    db_module.configure_engine()

    num_gpus = args.num_gpus if args.num_gpus > 0 else detect_gpu_count()

    paths = load_lfw_paths(limit=args.limit)
    print(f"LFW images: {len(paths)}")
    print(f"Model pack: {settings.model_pack}")
    print(f"GPUs used: {num_gpus}")

    run_id = secrets.token_hex(8)
    print(f"Run id: {run_id}")

    items = build_items(paths, run_id)
    splits = [items[i::num_gpus] for i in range(num_gpus)]

    t0 = time.perf_counter()
    results = await asyncio.gather(
        *(enroll_on_gpu(i, splits[i]) for i in range(num_gpus))
    )
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

    await db_module.dispose_engine()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
