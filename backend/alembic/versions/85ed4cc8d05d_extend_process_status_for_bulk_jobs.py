"""extend process status for bulk jobs

Revision ID: 85ed4cc8d05d
Revises: 363a56874e2e
Create Date: 2026-07-14 16:08:45.328328

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '85ed4cc8d05d'
down_revision: Union[str, None] = '363a56874e2e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_STATUSES = "'pending','queued','running','cancel_requested','cancelling','cancelled','completed','failed'"


def _recreate(table: str, constraint: str, column: str, nullable: bool) -> None:
    op.drop_constraint(constraint, table, type_="check")
    if nullable:
        sql = (
            f"ALTER TABLE {table} ADD CONSTRAINT {constraint} "
            f"CHECK ({column} IS NULL OR ({column})::text = ANY ((ARRAY[{_STATUSES}])::text[]))"
        )
    else:
        sql = (
            f"ALTER TABLE {table} ADD CONSTRAINT {constraint} "
            f"CHECK (({column})::text = ANY ((ARRAY[{_STATUSES}])::text[]))"
        )
    op.execute(sql)


def upgrade() -> None:
    _recreate("process_record", "process_record_status_check", "status", nullable=False)
    _recreate(
        "process_event",
        "process_event_status_before_check",
        "status_before",
        nullable=True,
    )
    _recreate(
        "process_event",
        "process_event_status_after_check",
        "status_after",
        nullable=True,
    )


def downgrade() -> None:
    old_statuses = "'pending','running','completed','failed'"
    for table, constraint, nullable in (
        ("process_record", "process_record_status_check", False),
        ("process_event", "process_event_status_before_check", True),
        ("process_event", "process_event_status_after_check", True),
    ):
        op.drop_constraint(constraint, table, type_="check")
        col = "status" if table == "process_record" else constraint.split("_")[2]
        if nullable:
            sql = (
                f"ALTER TABLE {table} ADD CONSTRAINT {constraint} "
                f"CHECK ({col} IS NULL OR ({col})::text = ANY ((ARRAY[{old_statuses}])::text[]))"
            )
        else:
            sql = (
                f"ALTER TABLE {table} ADD CONSTRAINT {constraint} "
                f"CHECK (({col})::text = ANY ((ARRAY[{old_statuses}])::text[]))"
            )
        op.execute(sql)
