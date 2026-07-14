import pytest
from sqlalchemy import text

from app.infrastructure import db as db_module


@pytest.mark.asyncio
async def test_approved_schema_tables_exist():
    expected = {
        "person",
        "face_identity",
        "process_record",
        "inference_profile",
        "person_photo",
        "face_sample",
        "recognition_result",
        "process_event",
    }
    async with db_module.engine.begin() as conn:
        for table in expected:
            result = await conn.execute(
                text(f"SELECT to_regclass('public.{table}')")
            )
            assert result.scalar() is not None, f"{table} not found"


@pytest.mark.asyncio
async def test_process_record_status_check_constraint():
    async with db_module.engine.begin() as conn:
        result = await conn.execute(
            text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = 'process_record'::regclass AND contype = 'c'"
            )
        )
        constraints = {row[0] for row in result.all()}
        assert any("status" in c for c in constraints), "missing process_record status check"


@pytest.mark.asyncio
async def test_process_event_has_no_raw_pii_columns():
    async with db_module.engine.begin() as conn:
        result = await conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'process_event'"
            )
        )
        columns = {row[0] for row in result.all()}
        forbidden = {"national_id", "raw_name", "file_path", "folder_name"}
        assert not columns & forbidden, f"process_event contains forbidden columns: {columns & forbidden}"
