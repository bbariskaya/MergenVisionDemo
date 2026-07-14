"""Threshold analysis using stored face embeddings.

Computes same-person vs different-person cosine-score distributions over the
entire enrolled dataset, plus ROC/EER/FAR metrics for candidate thresholds.
Run inside the API / GPU-worker container:
    python scripts/lfw_threshold_analysis.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATABASE_URL = os.environ["DATABASE_URL"].replace("postgresql+psycopg", "postgresql", 1)
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "face_samples")
NEGATIVE_PAIR_LIMIT = int(os.environ.get("NEGATIVE_PAIR_LIMIT", "200000"))


def qdrant_scroll_all() -> dict[str, np.ndarray]:
    """Fetch every stored vector from Qdrant."""
    vectors: dict[str, np.ndarray] = {}
    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/scroll"
    offset: str | None = None
    page = 0
    while True:
        body: dict = {
            "limit": 1000,
            "with_vector": True,
            "with_payload": False,
        }
        if offset:
            body["offset"] = offset
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
        for point in result["result"]["points"]:
            vectors[point["id"]] = np.asarray(point["vector"], dtype=np.float32)
        page += 1
        print(f"  scroll page {page}: {len(vectors)} vectors so far")
        offset = result["result"].get("next_page_offset")
        if offset is None:
            break
    return vectors


def cosine_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a shape (n, d), b shape (m, d) -> (n, m) cosine matrix."""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    return (a @ b.T) / (a_norm @ b_norm.T)


def describe(name: str, scores: np.ndarray) -> None:
    print(
        f"{name}: count={len(scores):,}  "
        f"mean={scores.mean():.4f}  std={scores.std():.4f}  "
        f"min={scores.min():.4f}  max={scores.max():.4f}  "
        f"p5={np.percentile(scores, 5):.4f}  "
        f"p25={np.percentile(scores, 25):.4f}  "
        f"p50={np.percentile(scores, 50):.4f}  "
        f"p75={np.percentile(scores, 75):.4f}  "
        f"p95={np.percentile(scores, 95):.4f}  "
        f"p99={np.percentile(scores, 99):.4f}"
    )


def main() -> int:
    print("Loading vectors from Qdrant...")
    vectors = qdrant_scroll_all()
    print(f"Loaded {len(vectors):,} vectors")

    print("Loading sample metadata from Postgres...")
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT sample_id, person_id FROM face_sample WHERE status = 'active'"
            )
            sample_to_person = {str(r[0]): str(r[1]) for r in cur.fetchall()}

    print(f"Loaded {len(sample_to_person):,} active samples")

    # Keep only IDs present in both DB and Qdrant.
    ids = list(set(vectors.keys()) & set(sample_to_person.keys()))
    print(f"Intersection size: {len(ids):,}")

    person_to_ids: dict[str, list[str]] = defaultdict(list)
    for sid in ids:
        person_to_ids[sample_to_person[sid]].append(sid)

    ids_with_genuine = [p for p, s in person_to_ids.items() if len(s) >= 2]
    print(f"Identities with >=2 samples: {len(ids_with_genuine):,}")

    genuines: list[float] = []
    print("Computing genuine pairwise scores...")
    for person in ids_with_genuine:
        sids = person_to_ids[person]
        vecs = np.stack([vectors[sid] for sid in sids])
        mat = cosine_pairs(vecs, vecs)
        # upper triangle, excluding diagonal
        iu = np.triu_indices(len(sids), k=1)
        genuines.extend(mat[iu].tolist())
    genuine_scores = np.asarray(genuines, dtype=np.float32)
    describe("Genuine", genuine_scores)

    print("Sampling negative pairs...")
    negatives: list[float] = []
    person_list = list(person_to_ids.items())
    tried = 0
    while len(negatives) < NEGATIVE_PAIR_LIMIT:
        tried += 1
        if tried > NEGATIVE_PAIR_LIMIT * 10:
            break
        p1, s1 = random.choice(person_list)
        p2, s2 = random.choice(person_list)
        if p1 == p2:
            continue
        a = random.choice(s1)
        b = random.choice(s2)
        negatives.append(float(np.dot(vectors[a], vectors[b])))
    negative_scores = np.asarray(negatives, dtype=np.float32)
    describe("Impostor", negative_scores)

    # Candidate thresholds.
    print("\nThreshold analysis:")
    thresholds = np.arange(0.30, 0.75, 0.01)
    metrics = []
    for t in thresholds:
        tp = int((genuine_scores >= t).sum())
        fn = int((genuine_scores < t).sum())
        fp = int((negative_scores >= t).sum())
        tn = int((negative_scores < t).sum())
        tar = tp / (tp + fn) if (tp + fn) else 0.0
        far = fp / (fp + tn) if (fp + tn) else 0.0
        frr = fn / (tp + fn) if (tp + fn) else 0.0
        # EER-ish: absolute difference between far and frr
        metrics.append((t, tar, far, frr, abs(far - frr), tp, fn, fp, tn))

    # EER point.
    eer_row = min(metrics, key=lambda x: x[4])
    print(f"Approx EER threshold={eer_row[0]:.2f} TAR={eer_row[1]:.4f} FAR={eer_row[2]:.4f} FRR={eer_row[3]:.4f}")

    # Threshold with FAR <= 1e-4.
    rows_low_far = [m for m in metrics if m[2] <= 1e-4]
    if rows_low_far:
        best_low_far = max(rows_low_far, key=lambda x: x[1])
        print(
            f"Threshold with FAR<=1e-4: {best_low_far[0]:.2f} "
            f"TAR={best_low_far[1]:.4f} FAR={best_low_far[2]:.6f}"
        )
    else:
        print("No threshold achieves FAR <= 1e-4")

    # Threshold with FRR <= 5%.
    rows_low_frr = [m for m in metrics if m[3] <= 0.05]
    if rows_low_frr:
        lowest_far_at_low_frr = min(rows_low_frr, key=lambda x: x[2])
        print(
            f"Threshold with FRR<=5%: {lowest_far_at_low_frr[0]:.2f} "
            f"TAR={lowest_far_at_low_frr[1]:.4f} FAR={lowest_far_at_low_frr[2]:.6f}"
        )
    else:
        print("No threshold achieves FRR <= 5%")

    print("\nCandidate detail table (every 0.05):")
    for m in metrics:
        if abs((m[0] * 100) % 5) < 0.5:
            print(
                f"  t={m[0]:.2f}  TAR={m[1]:.4f}  FAR={m[2]:.6f}  FRR={m[3]:.4f}  "
                f"tp={m[5]} fn={m[6]} fp={m[7]} tn={m[8]}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
