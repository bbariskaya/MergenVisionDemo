from pathlib import Path

import numpy as np
import pytest

from app.ml.alignment import align_face
from app.ml.pipeline import FaceExtraction, FacePipeline
from app.ml.preprocess import load_image


SAMPLES_DIR = Path("/app/artifacts/samples")
LFW_DIR = Path("/app/lfw/lfw-deepfunneled/lfw-deepfunneled")


def _first_extraction(pipeline: FacePipeline, path: Path) -> FaceExtraction:
    faces = pipeline.extract_from_path(path)
    assert len(faces) >= 1, f"No face detected in {path}"
    return faces[0]


def _first_embedding(pipeline: FacePipeline, path: Path) -> np.ndarray:
    face = _first_extraction(pipeline, path)
    emb = face.embedding
    assert emb.shape == (512,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5
    return emb


def test_detect_multi_face(pipeline: FacePipeline) -> None:
    # t1.jpg is a group photo; we expect multiple detections.
    faces = pipeline.extract_from_path(SAMPLES_DIR / "t1.jpg")
    assert len(faces) >= 3, f"expected >=3 faces, got {len(faces)}"
    for face in faces:
        assert face.bbox.shape == (4,)
        assert face.landmarks.shape == (5, 2)


def test_detect_no_face(pipeline: FacePipeline) -> None:
    # mask_blue.jpg should yield zero detections.
    faces = pipeline.extract_from_path(SAMPLES_DIR / "mask_blue.jpg")
    assert len(faces) == 0


def test_embedding_shape_and_norm(pipeline: FacePipeline) -> None:
    _first_embedding(pipeline, SAMPLES_DIR / "t1.jpg")


def test_extract_uses_single_recognizer_batch(pipeline: FacePipeline, monkeypatch) -> None:
    image = load_image(SAMPLES_DIR / "t1.jpg")
    faces = pipeline.extract_from_path(SAMPLES_DIR / "t1.jpg")
    assert len(faces) >= 2, f"expected >=2 faces, got {len(faces)}"

    call_count = {"recognizer": 0}
    original_infer = pipeline.recognizer.infer

    def counting_infer(inputs):
        call_count["recognizer"] += 1
        return original_infer(inputs)

    monkeypatch.setattr(pipeline.recognizer, "infer", counting_infer)

    results = pipeline.extract(image)
    assert len(results) == len(faces)
    assert call_count["recognizer"] == 1, (
        f"expected exactly one recognizer inference for {len(faces)} faces, "
        f"got {call_count['recognizer']}"
    )
    for i, (face, result) in enumerate(zip(faces, results)):
        np.testing.assert_allclose(face.embedding, result.embedding, rtol=1e-5, atol=1e-5)


def test_extract_no_face_recognizer_not_called(pipeline: FacePipeline, monkeypatch) -> None:
    image = load_image(SAMPLES_DIR / "mask_blue.jpg")
    called = {"recognizer": False}

    def failing_infer(_inputs):
        called["recognizer"] = True
        raise AssertionError("recognizer should not be called when no faces detected")

    monkeypatch.setattr(pipeline.recognizer, "infer", failing_infer)
    faces = pipeline.extract(image)
    assert len(faces) == 0
    assert called["recognizer"] is False


def test_batch_parity(pipeline: FacePipeline) -> None:
    from app.ml.alignment import align_face

    face = _first_extraction(pipeline, SAMPLES_DIR / "t1.jpg")
    image = load_image(SAMPLES_DIR / "t1.jpg")
    aligned = align_face(image, face.landmarks)

    single = pipeline.embed_aligned(aligned)
    batch_one = pipeline.embed_batch([aligned])
    np.testing.assert_allclose(single, batch_one[0], rtol=1e-5, atol=1e-5)

    # Real batch with two different faces.
    face2 = pipeline.extract_from_path(SAMPLES_DIR / "t1.jpg")[1]
    aligned2 = align_face(image, face2.landmarks)
    batch_two = pipeline.embed_batch([aligned, aligned2])
    assert batch_two.shape == (2, 512)
    np.testing.assert_allclose(single, batch_two[0], rtol=1e-5, atol=1e-5)


def test_same_person_high_similarity(pipeline: FacePipeline) -> None:
    person = "Abdullah_Gul"
    emb1 = _first_embedding(
        pipeline, LFW_DIR / person / f"{person}_0013.jpg"
    )
    emb2 = _first_embedding(
        pipeline, LFW_DIR / person / f"{person}_0014.jpg"
    )
    sim = float(np.dot(emb1, emb2))
    print(f"same person ({person}_0013 vs 0014) cosine: {sim:.4f}")
    assert sim > 0.6, f"expected same-person similarity >0.6, got {sim}"


def test_different_person_lower_similarity(pipeline: FacePipeline) -> None:
    emb1 = _first_embedding(pipeline, LFW_DIR / "AJ_Lamas" / "AJ_Lamas_0001.jpg")
    emb2 = _first_embedding(
        pipeline, LFW_DIR / "Zach_Safrin" / "Zach_Safrin_0001.jpg"
    )
    sim = float(np.dot(emb1, emb2))
    print(f"different person (AJ_Lamas vs Zach_Safrin) cosine: {sim:.4f}")
    assert sim < 0.5, f"expected different-person similarity <0.5, got {sim}"
