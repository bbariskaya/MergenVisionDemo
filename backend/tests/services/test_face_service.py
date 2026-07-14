"""Unit tests for face_service helpers."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from app.infrastructure.qdrant import SearchHit
from app.services.face_service import FaceService


class _DummyFaceService:
    pass


def _make_extraction():
    return SimpleNamespace(
        bbox=(0.1, 0.2, 0.3, 0.4),
        landmarks=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]),
    )


def test_build_recognized_face_candidate_face_id_is_person_identity():
    """Candidates must link to the person's canonical face_identity_id so the
    UI detail page (GET /faces/{face_id}) resolves correctly. ``face_id`` must
    never be the Qdrant sample_id.
    """
    person_id = uuid.uuid4()
    face_identity_id = uuid.uuid4()
    sample_id = uuid.uuid4()
    photo_id = uuid.uuid4()

    other_person_id = uuid.uuid4()
    other_face_identity_id = uuid.uuid4()
    other_sample_id = uuid.uuid4()
    other_photo_id = uuid.uuid4()

    best = SearchHit(
        sample_id=sample_id,
        photo_id=photo_id,
        person_id=person_id,
        score=0.65,
    )
    other = SearchHit(
        sample_id=other_sample_id,
        photo_id=other_photo_id,
        person_id=other_person_id,
        score=0.35,
    )

    person_identity_map = {
        person_id: face_identity_id,
        other_person_id: other_face_identity_id,
    }
    person_name_map = {
        person_id: "Alice Smith",
        other_person_id: "Bob Jones",
    }

    svc = _DummyFaceService()
    result = FaceService._build_recognized_face(
        svc,
        face_index=0,
        extraction=_make_extraction(),
        is_known=True,
        best=best,
        candidates=[best, other],
        face_id=face_identity_id,
        name="Alice Smith",
        metadata=None,
        person_name_map=person_name_map,
        person_identity_map=person_identity_map,
    )

    assert result["face_id"] == face_identity_id
    assert result["person_id"] == person_id

    candidate_ids = {c["face_id"] for c in result["candidates"]}
    assert sample_id not in candidate_ids
    assert other_sample_id not in candidate_ids
    assert face_identity_id in candidate_ids
    assert other_face_identity_id in candidate_ids

    best_candidate = result["candidates"][0]
    assert best_candidate["face_id"] == face_identity_id
    assert best_candidate["person_id"] == str(person_id)
    assert best_candidate["photo_id"] == str(photo_id)
    assert best_candidate["name"] == "Alice Smith"
