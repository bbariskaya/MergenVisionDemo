"""Compute-only face-pipeline benchmark for the full LFW dataset.

This does **not** touch PostgreSQL, MinIO or Qdrant.  It measures only the
ML path:

    JPEG bytes -> decode -> preprocess -> detector -> postprocess ->
    largest-face selection -> alignment -> recognizer -> L2 normalize

Examples
--------
Single GPU (uses cuda device 0 by default):
    MODEL_PACK=retinaface_r50 python scripts/profile_compute_only.py

Process-per-GPU (each child sees exactly one physical GPU):
    MODEL_PACK=retinaface_r50 python scripts/profile_compute_only.py --num-gpus 3
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

LFW_ROOT = Path("/app/lfw/lfw-deepfunneled/lfw-deepfunneled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=1, help="number of GPUs to benchmark in separate processes")
    parser.add_argument("--limit", type=int, default=None, help="limit number of images")
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


def worker(gpu_id: int, paths: list[Path]) -> dict[str, float | int]:
    """Run on a single physical GPU by hiding the others from this process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Delayed imports so the CUDA runtime only sees the visible device.
    import numpy as np
    from app.ml.gpu.face_pipeline import GpuFacePipeline

    random.seed(gpu_id)
    buffers = [p.read_bytes() for p in paths]

    pipeline = GpuFacePipeline(device_id=0)
    pipeline.warmup()
    # Warm-up batch
    _ = pipeline.extract_batch(buffers[:16])

    t0 = time.perf_counter()
    results = pipeline.extract_batch(buffers)
    wall = time.perf_counter() - t0

    faces = sum(1 for r in results if r is not None)
    pipeline.close()

    return {
        "gpu": gpu_id,
        "images": len(paths),
        "faces": faces,
        "wall": wall,
        "images_per_sec": len(paths) / wall if wall else 0.0,
    }


def main() -> int:
    args = parse_args()
    paths = load_paths(args.limit)
    if not paths:
        print(f"No images found under {LFW_ROOT}", file=os.sys.stderr)
        return 1

    print(f"LFW images: {len(paths)}")
    print(f"Model pack: {os.environ.get('MODEL_PACK', 'default')}")
    print(f"GPUs used:  {args.num_gpus} (separate processes)")

    mp.set_start_method("spawn", force=True)

    n = len(paths)
    chunk_size = (n + args.num_gpus - 1) // args.num_gpus
    chunks = [paths[i : i + chunk_size] for i in range(0, n, chunk_size)]

    t_start = time.perf_counter()
    with mp.Pool(processes=args.num_gpus) as pool:
        async_results = [
            pool.apply_async(worker, (gpu_id, chunk))
            for gpu_id, chunk in enumerate(chunks[:args.num_gpus])
        ]
        per_gpu = [r.get() for r in async_results]
    total_wall = time.perf_counter() - t_start

    for result in per_gpu:
        print(
            f"  gpu{result['gpu']}: images={result['images']} faces={result['faces']} "
            f"wall={result['wall']:.3f}s ({result['images_per_sec']:.1f} img/s)"
        )

    total_images = sum(r["images"] for r in per_gpu)
    total_faces = sum(r["faces"] for r in per_gpu)
    print(f"\nAggregated: {total_images} images, {total_faces} faces")
    print(f"Total wall time (parallel): {total_wall:.3f}s")
    print(f"Aggregate throughput: {total_images / total_wall:.1f} images/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
