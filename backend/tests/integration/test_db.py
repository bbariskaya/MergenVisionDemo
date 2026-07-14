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


@pytest.mark.asyncio
async def test_engine_dispose_and_reconfigure():
    # Start from a known configured state.
    db_module.configure_engine()
    assert db_module.engine is not None
    assert db_module.AsyncSessionLocal is not None

    await db_module.dispose_engine()
    assert db_module.engine is None
    assert db_module.AsyncSessionLocal is None

    with pytest.raises(RuntimeError, match="Database engine not configured"):
        async for _ in db_module.get_db():
            pass

    db_module.configure_engine()
    assert db_module.engine is not None
    assert db_module.AsyncSessionLocal is not None

    # Ensure the reconfigured engine works.
    async with db_module.engine.begin() as conn:
        result = await conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
