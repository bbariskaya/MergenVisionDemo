import asyncio
from dataclasses import dataclass
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

from app.core.config import settings


@dataclass(frozen=True)
class ObjectMeta:
    object_key: str
    content_type: str
    size: int


class PhotoStorage:
    def __init__(self) -> None:
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self._bucket = settings.minio_bucket_photos

    async def initialize(self) -> None:
        exists = await asyncio.to_thread(self._client.bucket_exists, self._bucket)
        if exists:
            return
        try:
            await asyncio.to_thread(self._client.make_bucket, self._bucket)
        except S3Error as exc:
            if exc.code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                return
            raise
        # Final verification after create/race.
        if not await asyncio.to_thread(self._client.bucket_exists, self._bucket):
            raise RuntimeError(f"MinIO bucket {self._bucket} not available after init")

    async def put_object(
        self,
        object_key: str,
        data: BinaryIO,
        length: int,
        content_type: str,
    ) -> ObjectMeta:
        await asyncio.to_thread(
            self._client.put_object,
            self._bucket,
            object_key,
            data,
            length,
            content_type=content_type,
        )
        return ObjectMeta(
            object_key=object_key,
            content_type=content_type,
            size=length,
        )

    async def get_object(self, object_key: str) -> bytes:
        response = await asyncio.to_thread(
            self._client.get_object, self._bucket, object_key
        )
        try:
            return await asyncio.to_thread(response.read)
        finally:
            await asyncio.to_thread(response.close)
            await asyncio.to_thread(response.release_conn)

    async def object_exists(self, object_key: str) -> bool:
        try:
            await asyncio.to_thread(
                self._client.stat_object, self._bucket, object_key
            )
            return True
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return False
            raise

    async def delete_object(self, object_key: str) -> None:
        try:
            await asyncio.to_thread(
                self._client.remove_object, self._bucket, object_key
            )
        except S3Error as exc:
            if exc.code != "NoSuchKey":
                raise

    async def health_check(self) -> bool:
        try:
            return bool(
                await asyncio.to_thread(self._client.bucket_exists, self._bucket)
            )
        except Exception:
            return False
