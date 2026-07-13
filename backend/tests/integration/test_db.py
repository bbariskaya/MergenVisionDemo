import pytest
from sqlalchemy import text

from app.infrastructure import db as db_module


@pytest.mark.asyncio
async def test_migrations_applied():
    tables = [
        "person",
        "person_photo",
        "face_sample",
        "recognition_request",
        "recognition_result",
    ]
    async with db_module.engine.begin() as conn:
        for table in tables:
            result = await conn.execute(
                text(f"SELECT to_regclass('public.{table}')")
            )
            assert result.scalar() is not None, f"{table} not found"
