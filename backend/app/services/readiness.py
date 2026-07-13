import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.infrastructure.minio import PhotoStorage
from app.infrastructure.qdrant import FaceVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentStatus:
    name: str
    status: str  # "ok" or "unavailable"
    details: dict[str, Any] | None = None


class ReadinessService:
    def __init__(
        self,
        engine: AsyncEngine,
        storage: PhotoStorage,
        vector_store: FaceVectorStore,
    ) -> None:
        self._engine = engine
        self._storage = storage
        self._vector_store = vector_store

    async def check(self) -> tuple[list[ComponentStatus], bool]:
        statuses: list[ComponentStatus] = []
        postgres_ok = await self._check_postgres()
        minio_ok = await self._storage.health_check()
        qdrant_ok = await self._vector_store.health_check()

        statuses.append(
            ComponentStatus(
                name="postgres",
                status="ok" if postgres_ok else "unavailable",
            )
        )
        statuses.append(
            ComponentStatus(
                name="minio",
                status="ok" if minio_ok else "unavailable",
            )
        )
        statuses.append(
            ComponentStatus(
                name="qdrant",
                status="ok" if qdrant_ok else "unavailable",
            )
        )

        all_ok = postgres_ok and minio_ok and qdrant_ok
        if not all_ok:
            logger.warning("Readiness check failed: %s", statuses)
        return statuses, all_ok

    async def _check_postgres(self) -> bool:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                return True
        except Exception as exc:
            logger.warning("Postgres readiness check failed: %s", exc)
            return False
