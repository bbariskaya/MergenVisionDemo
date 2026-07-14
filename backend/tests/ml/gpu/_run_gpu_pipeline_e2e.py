"""Run end-to-end GPU/CPU parity check in a fresh process."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from app.ml.pipeline import FacePipeline as CpuPipeline
from app.ml.gpu.face_pipeline import GpuFacePipeline

LFW_DIR = Path("/app/lfw/lfw-deepfunneled")


def main() -> int:
    if not LFW_DIR.exists():
        print(json.dumps({"error": "LFW not mounted"}))
        return 77

    cpu_pipeline = CpuPipeline()
    gpu_pipeline = GpuFacePipeline()
    gpu_pipeline.warmup()

    similarities: list[float] = []
    try:
        paths = sorted(LFW_DIR.rglob("*.jpg"))[:20]
        if not paths:
            print(json.dumps({"error": "no images"}))
            return 77

        for path in paths:
            cpu_faces = cpu_pipeline.extract_from_path(path)
            if not cpu_faces:
                continue
            gpu_faces = gpu_pipeline.extract_bytes(path.read_bytes())
            if not gpu_faces:
                continue

            cpu_emb = cpu_faces[0].embedding
            gpu_emb = gpu_faces[0].embedding
            sim = float(
                np.dot(cpu_emb, gpu_emb)
                / (np.linalg.norm(cpu_emb) * np.linalg.norm(gpu_emb))
            )
            similarities.append(sim)
    finally:
        cpu_pipeline.close()
        gpu_pipeline.close()

    if not similarities:
        print(json.dumps({"error": "no faces"}))
        return 1

    result = {
        "count": len(similarities),
        "min": min(similarities),
        "mean": float(np.mean(similarities)),
        "max": max(similarities),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
