"""Destructive reset of the active project's demo enrollment/process data.

Preserves schema, Alembic history, inference configuration and model assets.
Does NOT run ``docker compose down -v``.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import sqlalchemy as sa
from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings
from app.infrastructure import db as db_module
from app.infrastructure.minio import PhotoStorage


ACTIVE_CONTAINERS = {
    "mergenvisiondemo-api-1",
    "mergenvisiondemo-worker0-1",
    "mergenvisiondemo-worker1-1",
    "mergenvisiondemo-worker2-1",
    "mergenvisiondemo-postgres-1",
    "mergenvisiondemo-minio-1",
    "mergenvisiondemo-qdrant-1",
}


def _project_belongs_to_active() -> bool:
    """Best-effort: verify we can see the active project's containers."""
    stream = os.popen("docker compose ps --format '{{.Name}}'")
    names = {line.strip() for line in stream if line.strip()}
    stream.close()
    return bool(ACTIVE_CONTAINERS & names)


async def _pg_counts(conn: Any) -> dict[str, int]:
    tables = [
        "face_sample",
        "person_photo",
        "recognition_result",
        "recognition_request",
        "person",
        "face_identity",
        "process_event",
        "process_record",
    ]
    counts: dict[str, int] = {}
    for table in tables:
        result = await conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}"))
        counts[table] = result.scalar()
    return counts


async def _reset_postgres() -> None:
    db_module.configure_engine()
    async with db_module.engine.begin() as conn:
        print("PostgreSQL before counts:", await _pg_counts(conn))
        for statement in (
            "DELETE FROM recognition_result",
            "DELETE FROM recognition_request",
            "DELETE FROM face_sample",
            "DELETE FROM person_photo",
            "DELETE FROM person",
            "DELETE FROM face_identity",
            "DELETE FROM process_event",
            "DELETE FROM process_record",
        ):
            await conn.execute(sa.text(statement))
        print("PostgreSQL after counts:", await _pg_counts(conn))
    await db_module.dispose_engine()


async def _reset_minio() -> None:
    storage = PhotoStorage()
    await storage.initialize()
    objects = await asyncio.to_thread(
        lambda: list(storage._client.list_objects(storage._bucket, recursive=True))
    )
    print(f"MinIO before count: {len(objects)} objects in {storage._bucket}")
    for obj in objects:
        await storage.delete_object(obj.object_name)
    after = await asyncio.to_thread(
        lambda: list(storage._client.list_objects(storage._bucket, recursive=True))
    )
    print(f"MinIO after count: {len(after)} objects in {storage._bucket}")


async def _reset_qdrant() -> None:
    client = AsyncQdrantClient(url=settings.qdrant_url, check_compatibility=False)
    try:
        exists = await client.collection_exists(settings.qdrant_collection)
        if not exists:
            print("Qdrant collection does not exist; nothing to clear")
            return
        info = await client.get_collection(settings.qdrant_collection)
        before = info.points_count
        print(f"Qdrant before count: {before} points")
        await client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=models.Filter(),
            wait=True,
        )
        info = await client.get_collection(settings.qdrant_collection)
        print(f"Qdrant after count: {info.points_count} points")
    finally:
        await client.close()


async def main() -> int:
    if not _project_belongs_to_active():
        print(
            "Could not confirm active project containers. Refusing destructive reset.",
            file=os.sys.stderr,
        )
        return 1

    print("Resetting active demo data for project: MergenVisionDemo")
    await _reset_qdrant()
    await _reset_minio()
    await _reset_postgres()
    print("Reset complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
