import pytest

from app.infrastructure.db import configure_engine


@pytest.fixture(autouse=True)
def _ensure_db_engine():
    configure_engine()
    yield


@pytest.fixture
async def storage():
    from app.infrastructure.minio import PhotoStorage

    s = PhotoStorage()
    await s.initialize()
    return s


@pytest.fixture
async def vector_store():
    from app.infrastructure.qdrant import FaceVectorStore

    v = FaceVectorStore()
    await v.initialize()
    return v
