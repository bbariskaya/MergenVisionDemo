"""Diagnose Alicia-Silverstone recognition score end-to-end.

Run inside the GPU-worker or API container:
    python scripts/diag_alicia.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import psycopg
from minio import Minio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ml.gpu.face_pipeline import GpuFacePipeline

DATABASE_URL = os.environ["DATABASE_URL"].replace("postgresql+psycopg", "postgresql", 1)
MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
MINIO_BUCKET = os.environ.get("MINIO_BUCKET_PHOTOS", "mergenvision-photos")
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "face_samples")


def qdrant_get_vector(point_id: str) -> tuple[np.ndarray, dict]:
    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/{point_id}?with_vector=true&with_payload=true"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    vec = np.asarray(data["result"]["vector"], dtype=np.float32)
    return vec, data["result"].get("payload", {})


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    person_name = "Alicia Silverstone"
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT person_id, face_identity_id FROM person WHERE first_name || ' ' || last_name = %s",
                (person_name,),
            )
            row = cur.fetchone()
            if row is None:
                print(f"{person_name} not found in DB")
                return 1
            person_id, face_id = row
            print(f"person_id={person_id} face_identity_id={face_id}")

            cur.execute(
                "SELECT s.sample_id, p.photo_id, p.object_key "
                "FROM face_sample s JOIN person_photo p ON s.photo_id = p.photo_id "
                "WHERE s.person_id = %s AND s.status = 'active' "
                "ORDER BY p.created_at",
                (person_id,),
            )
            samples = cur.fetchall()
            print(f"active samples/photos: {len(samples)}")
            for s in samples:
                print(f"  sample={s[0]} photo={s[1]} object={s[2]}")

    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    # Recompute embeddings for each stored photo with the current production GPU pipeline.
    pipe = GpuFacePipeline(device_id=0)
    pipe.warmup()

    recomputed: dict[str, np.ndarray] = {}
    stored: dict[str, np.ndarray] = {}
    for sample_id, photo_id, object_key in samples:
        resp = minio_client.get_object(MINIO_BUCKET, object_key)
        image_bytes = resp.read()
        resp.close()
        resp.release_conn()

        faces = pipe.extract_bytes(image_bytes)
        if not faces:
            print(f"NO FACE recomputed for {object_key}")
            continue
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        recomputed[str(sample_id)] = largest.embedding

        vec, payload = qdrant_get_vector(str(sample_id))
        stored[str(sample_id)] = vec
        print(
            f"sample={sample_id} photo={photo_id} modelVersion={payload.get('modelVersion')} "
            f"recomputed_norm={np.linalg.norm(largest.embedding):.6f} "
            f"stored_norm={np.linalg.norm(vec):.6f} "
            f"self_cosine_recomputed_vs_stored={cosine(largest.embedding, vec):.6f}"
        )

    print("\nPairwise cosines (recomputed embeddings):")
    ids = list(recomputed.keys())
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            print(f"  {a[:8]} <-> {b[:8]}: {cosine(recomputed[a], recomputed[b]):.6f}")

    print("\nPairwise cosines (stored Qdrant vectors):")
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            print(f"  {a[:8]} <-> {b[:8]}: {cosine(stored[a], stored[b]):.6f}")

    # Distribution against a small negative sample.
    print("\nTop negative comparisons for first sample:")
    first_id = ids[0]
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT sample_id FROM face_sample "
                "WHERE person_id != %s AND status = 'active' "
                "ORDER BY random() LIMIT 100",
                (person_id,),
            )
            neg_samples = [r[0] for r in cur.fetchall()]
    neg_scores: list[tuple[float, str]] = []
    for sid in neg_samples:
        vec, _ = qdrant_get_vector(str(sid))
        neg_scores.append((cosine(stored[first_id], vec), str(sid)))
    neg_scores.sort(reverse=True)
    top5 = neg_scores[:5]
    p95 = neg_scores[int(0.05 * len(neg_scores))][0]
    print(f"  negative max={neg_scores[0][0]:.6f} p95={p95:.6f}")
    for score, sid in top5:
        print(f"    {sid[:8]}: {score:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
