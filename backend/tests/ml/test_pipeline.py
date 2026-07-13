from pathlib import Path

import numpy as np
import pytest

from app.ml.pipeline import FacePipeline
from app.ml.preprocess import load_image


SAMPLES_DIR = Path("/app/artifacts/samples")


@pytest.fixture(scope="session", autouse=True)
def ensure_samples():
    import subprocess

    if not (SAMPLES_DIR / "t1.jpg").exists():
        subprocess.run(
            ["python", "scripts/download_samples.py"],
            cwd="/app",
            check=True,
        )


@pytest.fixture(scope="session")
def pipeline():
    return FacePipeline()


def _first_embedding(pipeline: FacePipeline, path: Path) -> np.ndarray:
    faces = pipeline.extract_from_path(path)
    assert len(faces) >= 1, f"No face detected in {path}"
    emb = faces[0].embedding
    assert emb.shape == (512,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5
    return emb


def test_detect_real_image(pipeline: FacePipeline) -> None:
    faces = pipeline.extract_from_path(SAMPLES_DIR / "t1.jpg")
    assert len(faces) >= 1
    assert faces[0].bbox.shape == (4,)
    assert faces[0].landmarks.shape == (5, 2)


def test_embedding_shape_and_norm(pipeline: FacePipeline) -> None:
    _first_embedding(pipeline, SAMPLES_DIR / "t1.jpg")


def test_batch_parity(pipeline: FacePipeline) -> None:
    from app.ml.alignment import align_face

    faces = pipeline.extract_from_path(SAMPLES_DIR / "t1.jpg")
    image = load_image(SAMPLES_DIR / "t1.jpg")
    aligned = align_face(image, faces[0].landmarks)
    single = pipeline.embed_aligned(aligned)
    batch = pipeline.embed_batch([aligned])
    np.testing.assert_allclose(single, batch[0], rtol=1e-5, atol=1e-5)


def test_same_person_high_similarity(pipeline: FacePipeline) -> None:
    emb1 = _first_embedding(pipeline, SAMPLES_DIR / "t1.jpg")
    emb2 = _first_embedding(pipeline, SAMPLES_DIR / "Tom_Hanks_54745.png")
    sim = float(np.dot(emb1, emb2))
    print(f"same person (t1 vs Tom_Hanks) cosine: {sim:.4f}")
    assert sim > 0.6, f"expected same-person similarity >0.6, got {sim}"


def test_different_person_lower_similarity(pipeline: FacePipeline) -> None:
    emb1 = _first_embedding(pipeline, SAMPLES_DIR / "t1.jpg")
    emb2 = _first_embedding(pipeline, SAMPLES_DIR / "mask_blue.jpg")
    sim = float(np.dot(emb1, emb2))
    print(f"different person (t1 vs mask_blue) cosine: {sim:.4f}")
    assert sim < 0.6, f"expected different-person similarity <0.6, got {sim}"
