import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    sample_id: uuid.UUID
    photo_id: uuid.UUID
    person_id: uuid.UUID
    score: float


ALLOWED_PAYLOAD_KEYS = {
    "sampleId",
    "photoId",
    "personId",
    "active",
    "modelVersion",
}


class QdrantSchemaError(RuntimeError):
    pass


class FaceVectorStore:
    def __init__(self) -> None:
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url, check_compatibility=False
        )
        self._collection = settings.qdrant_collection
        self._vector_size = settings.embedding_dim
        self._distance = models.Distance.COSINE
        self._active_field = "active"
        self._model_version_field = "modelVersion"
        self._closed = False

    @staticmethod
    def _validate_uuid(value: Any, name: str) -> uuid.UUID:
        if isinstance(value, uuid.UUID):
            return value
        try:
            return uuid.UUID(str(value))
        except Exception as exc:
            raise ValueError(f"{name} is not a valid UUID: {value!r}") from exc

    def _validate_payload(self, point: models.PointStruct) -> None:
        payload = point.payload or {}
        extra = set(payload.keys()) - ALLOWED_PAYLOAD_KEYS
        if extra:
            raise ValueError(f"Forbidden payload keys: {sorted(extra)}")
        missing = ALLOWED_PAYLOAD_KEYS - set(payload.keys())
        if missing:
            raise ValueError(f"Missing required payload keys: {sorted(missing)}")

        point_id = self._validate_uuid(point.id, "point id")
        sample_id = self._validate_uuid(payload["sampleId"], "sampleId")
        if point_id != sample_id:
            raise ValueError(
                f"point id ({point_id}) must equal payload.sampleId ({sample_id})"
            )

        for key in ("photoId", "personId"):
            self._validate_uuid(payload[key], key)

        if not isinstance(payload["active"], bool):
            raise ValueError(
                f"active must be bool, got {type(payload['active']).__name__}"
            )

        model_version = payload.get("modelVersion", "")
        if not isinstance(model_version, str) or not model_version:
            raise ValueError("modelVersion must be a non-empty string")

    def _validate_vector(self, vector: list[float]) -> None:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (self._vector_size,):
            raise ValueError(
                f"vector dimension must be {self._vector_size}, got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("vector contains non-finite values (NaN or Inf)")
        norm = float(np.linalg.norm(arr))
        if norm == 0:
            raise ValueError("zero vector is not allowed")

    def _validate_search_args(
        self,
        vector: list[float],
        model_version: str,
        top_k: int,
        active: bool,
    ) -> None:
        self._validate_vector(vector)
        if not isinstance(model_version, str) or not model_version:
            raise ValueError("model_version must be a non-empty string")
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        if not isinstance(active, bool):
            raise ValueError("active must be bool")

    async def _schema_ok(self) -> bool:
        try:
            info = await self._client.get_collection(self._collection)
        except Exception:
            return False

        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            if "" not in vectors:
                return False
            vector_params = vectors[""]
        else:
            vector_params = vectors
        if vector_params.size != self._vector_size:
            return False
        if vector_params.distance != self._distance:
            return False

        payload_schema = info.payload_schema or {}
        for field, expected in (
            (self._active_field, models.PayloadSchemaType.BOOL.value),
            (self._model_version_field, models.PayloadSchemaType.KEYWORD.value),
        ):
            if field not in payload_schema:
                return False
            if payload_schema[field].data_type.value != expected:
                return False
        return True

    async def _ensure_indexes(self) -> None:
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name=self._active_field,
            field_schema=models.PayloadSchemaType.BOOL,
            wait=True,
        )
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name=self._model_version_field,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    async def initialize(self) -> None:
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                ),
            )
            await self._ensure_indexes()
            return

        # Create any missing indexes before validating so old collections can be
        # brought up to the expected schema without dropping data.
        await self._ensure_indexes()

        if not await self._schema_ok():
            logger.error(
                "Qdrant collection %r has incompatible schema/indexes. "
                "Manual migration (drop/recreate) is required.",
                self._collection,
            )
            raise QdrantSchemaError(
                f"Incompatible Qdrant schema for {self._collection}"
            )

    async def upsert_batch(
        self, points: list[models.PointStruct], *, wait: bool = False
    ) -> None:
        for point in points:
            self._validate_payload(point)
            self._validate_vector(point.vector)
        # Smaller Qdrant batches to keep request sizes bounded.
        batch_size = 256
        for i in range(0, len(points), batch_size):
            await self._client.upsert(
                collection_name=self._collection,
                points=points[i : i + batch_size],
                wait=wait,
            )

    async def search_active(
        self,
        vector: list[float],
        *,
        model_version: str,
        top_k: int,
        active: bool = True,
    ) -> list[SearchHit]:
        self._validate_search_args(vector, model_version, top_k, active)
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=self._active_field,
                        match=models.MatchValue(value=active),
                    ),
                    models.FieldCondition(
                        key=self._model_version_field,
                        match=models.MatchValue(value=model_version),
                    ),
                ]
            ),
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            try:
                hits.append(
                    SearchHit(
                        sample_id=uuid.UUID(str(payload.get("sampleId"))),
                        photo_id=uuid.UUID(str(payload.get("photoId"))),
                        person_id=uuid.UUID(str(payload.get("personId"))),
                        score=point.score,
                    )
                )
            except ValueError as exc:
                logger.warning(
                    "Malformed payload hit skipped: sample_id=%s error=%s",
                    payload.get("sampleId"),
                    exc,
                )
        return hits

    async def set_active(
        self, sample_id: uuid.UUID, active: bool
    ) -> None:
        await self.set_active_batch([sample_id], active)

    async def set_active_batch(
        self, sample_ids: list[uuid.UUID], active: bool
    ) -> None:
        if not sample_ids:
            return
        batch_size = 256
        for i in range(0, len(sample_ids), batch_size):
            await self._client.set_payload(
                collection_name=self._collection,
                points=[str(s) for s in sample_ids[i : i + batch_size]],
                payload={self._active_field: active},
                wait=True,
            )

    async def delete(self, sample_id: uuid.UUID) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.PointIdsList(points=[str(sample_id)]),
            wait=True,
        )

    async def health_check(self) -> bool:
        try:
            exists = await self._client.collection_exists(self._collection)
            if not exists:
                return False
            return await self._schema_ok()
        except Exception:
            return False

    async def close(self) -> None:
        if self._closed:
            return
        try:
            await self._client.close()
            self._closed = True
        except Exception as exc:
            logger.warning(
                "Qdrant client close failed: %s", exc.__class__.__name__
            )
