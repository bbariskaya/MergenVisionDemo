import numpy as np

from app.ml.alignment import (
    ARC_FACE_SRC,
    _estimate_similarity_transform,
    align_face,
)


def test_identity_transform_maps_template_to_itself():
    matrix = _estimate_similarity_transform(ARC_FACE_SRC, ARC_FACE_SRC)
    mapped = (
        np.hstack([ARC_FACE_SRC, np.ones((5, 1), dtype=np.float32)])
        @ matrix.T
    )
    np.testing.assert_allclose(mapped, ARC_FACE_SRC, atol=1e-4)


def test_estimated_transform_recovers_known_similarity():
    # Apply a known scale+rotation+translation to the template, then estimate
    # the transform and verify the inverse maps back.
    angle = np.deg2rad(15.0)
    scale = 1.3
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    translation = np.array([12.0, -8.0], dtype=np.float32)

    transformed = ARC_FACE_SRC @ rotation.T * scale + translation
    matrix = _estimate_similarity_transform(transformed, ARC_FACE_SRC)
    recovered = (
        np.hstack([transformed, np.ones((5, 1), dtype=np.float32)])
        @ matrix.T
    )
    np.testing.assert_allclose(recovered, ARC_FACE_SRC, atol=1e-3)


def test_align_face_produces_112_rgb():
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    landmarks = ARC_FACE_SRC * 2.5 + 50.0
    aligned = align_face(image, landmarks)
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8
    assert aligned.flags["C_CONTIGUOUS"]
