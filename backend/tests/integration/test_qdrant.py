import re

import numpy as np
import pytest
from qdrant_client import models

from app.core.ids import uuid7
from app.infrastructure.qdrant import FaceVectorStore


def _random_vector(dim: int = 512) -> list[float]:
    vec = np.random.random(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _make_model_version() -> str:
    return f"test-{uuid7()}"


def _valid_point(sample_id=None, vector=None, **payload_overrides) -> models.PointStruct:
    sample_id = sample_id or uuid7()
    vector = vector or _random_vector()
    payload = {
        "sampleId": str(sample_id),
        "photoId": str(uuid7()),
        "personId": str(uuid7()),
        "active": True,
        "modelVersion": _make_model_version(),
    }
    payload.update(payload_overrides)
    return models.PointStruct(id=str(sample_id), vector=vector, payload=payload)


@pytest.mark.asyncio
async def test_qdrant_health_check_true_for_existing_collection(
    vector_store: FaceVectorStore,
) -> None:
    assert await vector_store.health_check() is True


@pytest.mark.asyncio
async def test_qdrant_health_check_false_for_missing_collection() -> None:
    store = FaceVectorStore()
    store._collection = f"missing-{uuid7()}"
    assert await store.health_check() is False


@pytest.mark.asyncio
async def test_qdrant_health_check_false_for_incompatible_schema() -> None:
    store = FaceVectorStore()
    bad_collection = f"bad-schema-{uuid7()}"
    store._collection = bad_collection
    store._vector_size = 13  # intentionally wrong
    await store.initialize()
    try:
        # Now correct size expects incompatible existing schema.
        correct_store = FaceVectorStore()
        correct_store._collection = bad_collection
        assert await correct_store.health_check() is False
        await correct_store.close()
    finally:
        await store._client.delete_collection(bad_collection)
        await store.close()


@pytest.mark.asyncio
async def test_qdrant_payload_forbidden_key_rejected(
    vector_store: FaceVectorStore,
) -> None:
    point = _valid_point()
    point.payload["national_id"] = "123"
    with pytest.raises(ValueError, match="Forbidden payload keys"):
        await vector_store.upsert_batch([point])


@pytest.mark.asyncio
async def test_qdrant_payload_missing_key_rejected(
    vector_store: FaceVectorStore,
) -> None:
    point = _valid_point()
    del point.payload["modelVersion"]
    with pytest.raises(ValueError, match="Missing required payload keys"):
        await vector_store.upsert_batch([point])


@pytest.mark.asyncio
async def test_qdrant_payload_point_id_mismatch_rejected(
    vector_store: FaceVectorStore,
) -> None:
    point = _valid_point()
    point.id = str(uuid7())  # different from payload sampleId
    with pytest.raises(ValueError, match="point id"):
        await vector_store.upsert_batch([point])


@pytest.mark.asyncio
async def test_qdrant_payload_invalid_uuid_rejected(
    vector_store: FaceVectorStore,
) -> None:
    point = _valid_point(sampleId=str(uuid7()), sampleId_override="not-a-uuid")
    # sampleId_override key name is invalid by allowlist; use forbidden test instead.
    point = _valid_point()
    point.payload["sampleId"] = "not-a-uuid"
    with pytest.raises(ValueError, match="UUID"):
        await vector_store.upsert_batch([point])


@pytest.mark.asyncio
async def test_qdrant_payload_vector_validation(
    vector_store: FaceVectorStore,
) -> None:
    with pytest.raises(ValueError, match="vector"):
        await vector_store.upsert_batch(
            [_valid_point(vector=[0.0] * 511)]
        )

    with pytest.raises(ValueError, match="finite"):
        bad = _random_vector()
        bad[0] = float("nan")
        await vector_store.upsert_batch([_valid_point(vector=bad)])

    with pytest.raises(ValueError, match="zero"):
        await vector_store.upsert_batch([_valid_point(vector=[0.0] * 512)])


@pytest.mark.asyncio
async def test_qdrant_client_close_idempotent() -> None:
    store = FaceVectorStore()
    await store.close()
    await store.close()  # should not raise


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
    allowed = {"sampleId", "photoId", "personId", "active", "modelVersion"}

    await vector_store.upsert_batch(
        [models.PointStruct(id=str(sample_id), vector=vector, payload=payload)]
    )

    try:
        retrieved = await vector_store._client.retrieve(
            collection_name=vector_store._collection,
            ids=[str(sample_id)],
            with_payload=True,
        )
        assert len(retrieved) == 1
        persisted_payload = retrieved[0].payload or {}
        keys = set(persisted_payload.keys())
        assert keys == allowed
        forbidden = {"name", "first_name", "national_id", "national_id_masked"}
        assert not keys & forbidden
    finally:
        await vector_store.delete(sample_id)
