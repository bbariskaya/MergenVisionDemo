"""Diagnose Avril query score vs enrolled samples."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ml.gpu.face_pipeline import GpuFacePipeline

DATABASE_URL = os.environ["DATABASE_URL"].replace("postgresql+psycopg", "postgresql", 1)
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "face_samples")


def get_vector(point_id: str) -> np.ndarray:
    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/{point_id}?with_vector=true&with_payload=true"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return np.asarray(data["result"]["vector"], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    img_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/avril.jpg"
    pipe = GpuFacePipeline(device_id=0)
    pipe.warmup()
    with open(img_path, "rb") as f:
        faces = pipe.extract_bytes(f.read())
    if not faces:
        print("No face detected")
        return 1
    q = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    print(f"Query face bbox: {q.bbox.tolist()}, norm={np.linalg.norm(q.embedding):.6f}")

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT person_id FROM person WHERE first_name = 'Avril' AND last_name = 'Lavigne'")
            row = cur.fetchone()
            if row is None:
                print("Avril Lavigne not enrolled")
                return 1
            person_id = row[0]
            print(f"person_id={person_id}")
            cur.execute(
                "SELECT sample_id FROM face_sample WHERE person_id = %s AND status = 'active' ORDER BY created_at",
                (person_id,),
            )
            sample_ids = [str(r[0]) for r in cur.fetchall()]

    print(f"Enrolled active samples: {len(sample_ids)}")
    scores: list[tuple[float, str]] = []
    vectors = []
    for sid in sample_ids:
        vec = get_vector(sid)
        vectors.append(vec)
        scores.append((cosine(q.embedding, vec), sid))
    scores.sort(reverse=True)
    print("Top 5 query-vs-enrolled scores:")
    for score, sid in scores[:5]:
        print(f"  {sid[:8]}: {score:.6f}")
    print(f"mean={sum(s[0] for s in scores)/len(scores):.6f} max={scores[0][0]:.6f} min={scores[-1][0]:.6f}")

    if len(vectors) >= 2:
        mat = (vectors @ np.stack(vectors).T) / (
            np.linalg.norm(vectors, axis=1, keepdims=True) @ np.linalg.norm(vectors, axis=1, keepdims=True).T
        )
        iu = np.triu_indices(len(vectors), k=1)
        same_pairs = mat[iu].tolist()
        print(f"Enrolled-sample pairwise: count={len(same_pairs)} mean={np.mean(same_pairs):.6f} max={np.max(same_pairs):.6f} min={np.min(same_pairs):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
