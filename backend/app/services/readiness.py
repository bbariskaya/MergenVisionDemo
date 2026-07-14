import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.infrastructure.minio import PhotoStorage
from app.infrastructure.qdrant import FaceVectorStore

logger = logging.getLogger(__name__)
_TIMEOUT_SECONDS = 5.0

REQUIRED_TABLES = (
    "person",
    "person_photo",
    "face_sample",
    "recognition_request",
    "recognition_result",
)


@dataclass(frozen=True)
class ComponentStatus:
    name: str
    status: str
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
        coros = [
            self._check_with_timeout("postgres", self._check_postgres()),
            self._check_with_timeout("minio", self._storage.health_check()),
            self._check_with_timeout("qdrant", self._vector_store.health_check()),
        ]
        statuses = await asyncio.gather(*coros, return_exceptions=True)
        ok = all(
            not isinstance(s, Exception) and s.status == "ok" for s in statuses
        )
        if not ok:
            logger.warning(
                "Readiness check failed: %s",
                [
                    {"name": getattr(s, "name", None), "status": getattr(s, "status", "error")}
                    for s in statuses
                ],
            )
        return statuses, ok

    async def _check_with_timeout(self, name: str, coro) -> ComponentStatus:
        try:
            healthy = await asyncio.wait_for(coro, timeout=_TIMEOUT_SECONDS)
            status = "ok" if healthy else "unavailable"
            details = None
        except asyncio.TimeoutError:
            status = "unavailable"
            details = {"reason": "timeout"}
        except Exception as exc:
            status = "unavailable"
            details = {"reason": exc.__class__.__name__}
        return ComponentStatus(name=name, status=status, details=details)

    async def _check_postgres(self) -> bool:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                missing: list[str] = []
                for table in REQUIRED_TABLES:
                    result = await conn.execute(
                        text(f"SELECT to_regclass('public.{table}')")
                    )
                    if result.scalar() is None:
                        missing.append(table)
                if missing:
                    logger.warning(
                        "Postgres readiness: missing required tables: %s", missing
                    )
                    return False
                return True
        except Exception as exc:
            logger.warning(
                "Postgres readiness check failed: %s", exc.__class__.__name__
            )
            return False
