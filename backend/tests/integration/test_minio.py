import io

import pytest

from app.core.ids import uuid7
from app.infrastructure.minio import PhotoStorage


@pytest.mark.asyncio
async def test_minio_round_trip(storage: PhotoStorage) -> None:
    object_key = f"test/{uuid7()}.bin"
    payload = b"mergenvision test payload"

    meta = await storage.put_object(
        object_key=object_key,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="application/octet-stream",
    )
    assert meta.object_key == object_key
    assert meta.size == len(payload)

    exists = await storage.object_exists(object_key)
    assert exists is True

    downloaded = await storage.get_object(object_key)
    assert downloaded == payload

    await storage.delete_object(object_key)

    exists_after = await storage.object_exists(object_key)
    assert exists_after is False


@pytest.mark.asyncio
async def test_minio_delete_idempotent(storage: PhotoStorage) -> None:
    object_key = f"test/{uuid7()}-missing.bin"
    await storage.delete_object(object_key)
    assert await storage.object_exists(object_key) is False
