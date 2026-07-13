import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings


@dataclass(frozen=True)
class SearchHit:
    sample_id: uuid.UUID
    photo_id: uuid.UUID
    person_id: uuid.UUID
    score: float


class FaceVectorStore:
    def __init__(self) -> None:
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url, check_compatibility=False
        )
        self._collection = settings.qdrant_collection
        self._vector_size = settings.embedding_dim
        self._distance = models.Distance.COSINE
        self._active_field = "active"

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
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=self._active_field,
                field_schema=models.PayloadSchemaType.BOOL,
            )

    async def upsert_batch(self, points: list[models.PointStruct]) -> None:
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search_active(
        self,
        vector: list[float],
        *,
        model_version: str,
        top_k: int,
        active: bool = True,
    ) -> list[SearchHit]:
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
                        key="modelVersion",
                        match=models.MatchValue(value=model_version),
                    ),
                ]
            ),
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    sample_id=uuid.UUID(str(payload.get("sampleId"))),
                    photo_id=uuid.UUID(str(payload.get("photoId"))),
                    person_id=uuid.UUID(str(payload.get("personId"))),
                    score=point.score,
                )
            )
        return hits

    async def set_active(
        self, sample_id: uuid.UUID, active: bool
    ) -> None:
        await self._client.set_payload(
            collection_name=self._collection,
            points=[str(sample_id)],
            payload={self._active_field: active},
        )

    async def delete(self, sample_id: uuid.UUID) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.PointIdsList(points=[str(sample_id)]),
        )

    async def health_check(self) -> bool:
        try:
            await self._client.collection_exists(self._collection)
            return True
        except Exception:
            return False
