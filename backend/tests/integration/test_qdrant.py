import pytest
from qdrant_client import models

from app.core.ids import uuid7
from app.infrastructure.qdrant import FaceVectorStore


def _random_vector(dim: int = 512) -> list[float]:
    import numpy as np

    vec = np.random.random(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _make_model_version() -> str:
    return f"test-{uuid7()}"


@pytest.mark.asyncio
async def test_qdrant_active_search_lifecycle(vector_store: FaceVectorStore) -> None:
    sample_id = uuid7()
    photo_id = uuid7()
    person_id = uuid7()
    model_version = _make_model_version()
    vector = _random_vector()

    await vector_store.upsert_batch(
        [
            models.PointStruct(
                id=str(sample_id),
                vector=vector,
                payload={
                    "sampleId": str(sample_id),
                    "photoId": str(photo_id),
                    "personId": str(person_id),
                    "active": True,
                    "modelVersion": model_version,
                },
            )
        ]
    )

    try:
        hits = await vector_store.search_active(
            vector=vector,
            model_version=model_version,
            top_k=5,
        )
        assert len(hits) == 1
        assert hits[0].sample_id == sample_id
        assert hits[0].photo_id == photo_id
        assert hits[0].person_id == person_id
        assert hits[0].score > 0.99

        await vector_store.set_active(sample_id, False)

        active_hits = await vector_store.search_active(
            vector=vector,
            model_version=model_version,
            top_k=5,
        )
        assert len(active_hits) == 0

        inactive_hits = await vector_store.search_active(
            vector=vector,
            model_version=model_version,
            top_k=5,
            active=False,
        )
        assert len(inactive_hits) == 1
    finally:
        await vector_store.delete(sample_id)

    after_delete = await vector_store.search_active(
        vector=vector,
        model_version=model_version,
        top_k=5,
        active=False,
    )
    assert len(after_delete) == 0


@pytest.mark.asyncio
async def test_qdrant_payload_has_no_pii(vector_store: FaceVectorStore) -> None:
    sample_id = uuid7()
    model_version = _make_model_version()
    vector = _random_vector()
    payload = {
        "sampleId": str(sample_id),
        "photoId": str(uuid7()),
        "personId": str(uuid7()),
        "active": True,
        "modelVersion": model_version,
    }

    await vector_store.upsert_batch(
        [models.PointStruct(id=str(sample_id), vector=vector, payload=payload)]
    )

    try:
        keys = set(payload.keys())
        forbidden = {"name", "first_name", "national_id", "national_id_masked"}
        assert not keys & forbidden
    finally:
        await vector_store.delete(sample_id)
