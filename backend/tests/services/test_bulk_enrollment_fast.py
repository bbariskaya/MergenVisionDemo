"""Targeted tests for the deterministic fast bulk enrollment path.

These tests hit the real PostgreSQL / MinIO / Qdrant stack but operate on a
handful of synthetic files, so they are not benchmarks or dataset imports.
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import FaceIdentity, FaceSample, Person, PersonPhoto
from app.infrastructure import db as db_module
from app.infrastructure.minio import PhotoStorage
from app.infrastructure.qdrant import FaceVectorStore
from app.ml.gpu.face_pipeline import GpuFaceExtraction
from app.services.bulk_enrollment import BulkEnrollmentService
from app.services.bulk_manifest import build_lfw_manifest


class _FakeGpuPipeline:
    """CPU-only fake that returns one deterministic face per image."""

    def extract_batch(
        self,
        image_bytes_list: list[bytes],
        *,
        pick_largest: bool = True,
        max_batch: int = 256,
    ) -> list[GpuFaceExtraction]:
        dim = 512
        results: list[GpuFaceExtraction] = []
        for idx, data in enumerate(image_bytes_list):
            seed = int(hashlib.sha256(data).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed + idx)
            results.append(
                GpuFaceExtraction(
                    bbox=np.array([10.0, 20.0, 50.0, 60.0], dtype=np.float32),
                    landmarks=rng.random((5, 2), dtype=np.float32),
                    embedding=rng.standard_normal(dim).astype(np.float32),
                    score=0.95,
                )
            )
        return results


@pytest.fixture
async def test_storage():
    storage = PhotoStorage()
    await storage.initialize()
    return storage


@pytest.fixture
async def test_vector_store():
    store = FaceVectorStore()
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


def _write_jpeg(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG", quality=85)


@pytest.mark.asyncio
async def test_identity_grouping_and_idempotency(
    tmp_path: Path,
    test_storage: PhotoStorage,
    test_vector_store: FaceVectorStore,
) -> None:
    person_a = tmp_path / "Person_A"
    person_b = tmp_path / "Person_B"
    person_a.mkdir()
    person_b.mkdir()

    for idx in range(3):
        _write_jpeg(person_a / f"photo_{idx}.jpg", seed=100 + idx)
    for idx in range(2):
        _write_jpeg(person_b / f"photo_{idx}.jpg", seed=200 + idx)

    identities = build_lfw_manifest(tmp_path)
    assert len(identities) == 2
    assert sum(len(i.photos) for i in identities) == 5

    async with db_module.AsyncSessionLocal() as session:
        service = BulkEnrollmentService(
            db=session,
            storage=test_storage,
            vector_store=test_vector_store,
            pipeline=_FakeGpuPipeline(),
            pipeline_lock=asyncio.Lock(),
            qdrant_wait=True,
        )
        result = await service.enroll_shard(identities)

    assert result.discovered_identities == 2
    assert result.discovered_photos == 5
    assert result.enrolled == 5
    assert result.processed == 5
    assert result.no_face == 0
    assert result.decode_error == 0
    assert result.persistence_error == 0
    assert result.failed == 0

    async with db_module.AsyncSessionLocal() as session:
        counts = await _active_counts(session)

    assert counts["persons"] == 2
    assert counts["face_identities"] == 2
    assert counts["photos"] == 5
    assert counts["samples"] == 5

    # Idempotency run: re-enrolling the same manifest must not create duplicates.
    async with db_module.AsyncSessionLocal() as session:
        service2 = BulkEnrollmentService(
            db=session,
            storage=test_storage,
            vector_store=test_vector_store,
            pipeline=_FakeGpuPipeline(),
            pipeline_lock=asyncio.Lock(),
            qdrant_wait=True,
        )
        result2 = await service2.enroll_shard(identities)

    assert result2.enrolled == 5
    assert result2.processed == 5
    assert result2.no_face == 0
    assert result2.failed == 0

    async with db_module.AsyncSessionLocal() as session:
        counts2 = await _active_counts(session)

    assert counts2 == counts

    # Source dataset provenance is stored on Person.details.
    async with db_module.AsyncSessionLocal() as session:
        person_rows = (
            (await session.execute(select(Person.details))).scalars().all()
        )
    assert all(r.get("source_dataset") == "lfw" for r in person_rows)

    # Qdrant points land in the same collection with the active/modelVersion contract.
    qdrant_count = await test_vector_store._client.count(
        collection_name=test_vector_store._collection,
    )
    assert qdrant_count.count >= 5


async def _active_counts(session: AsyncSession) -> dict[str, int]:
    return {
        "persons": (
            await session.execute(
                select(func.count()).select_from(Person).where(Person.is_active.is_(True))
            )
        ).scalar_one(),
        "face_identities": (
            await session.execute(
                select(func.count())
                .select_from(FaceIdentity)
                .where(FaceIdentity.is_active.is_(True))
            )
        ).scalar_one(),
        "photos": (
            await session.execute(
                select(func.count())
                .select_from(PersonPhoto)
                .where(PersonPhoto.status == "active")
            )
        ).scalar_one(),
        "samples": (
            await session.execute(
                select(func.count())
                .select_from(FaceSample)
                .where(FaceSample.status == "active")
            )
        ).scalar_one(),
    }
