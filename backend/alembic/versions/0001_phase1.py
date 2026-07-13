"""phase1 initial schema

Revision ID: 0001_phase1
Revises:
Create Date: 2026-07-13 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_phase1"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "person",
        sa.Column("person_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("first_name", sa.String(255), nullable=False),
        sa.Column("last_name", sa.String(255), nullable=False),
        sa.Column("national_id_lookup_hmac", sa.String(64), nullable=False),
        sa.Column("national_id_masked", sa.String(32), nullable=False),
        sa.Column(
            "details", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"
        ),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False, server_default="true"
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
        sa.PrimaryKeyConstraint("person_id"),
        sa.UniqueConstraint("national_id_lookup_hmac"),
        sa.Index("ix_person_active_created", "is_active", sa.desc("created_at")),
    )

    op.create_table(
        "person_photo",
        sa.Column("photo_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("person_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("object_key", sa.String(512), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("mime_type", sa.String(64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column(
            "status", sa.String(16), nullable=False, server_default="staged"
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
        sa.PrimaryKeyConstraint("photo_id"),
        sa.ForeignKeyConstraint(["person_id"], ["person.person_id"]),
        sa.UniqueConstraint("object_key"),
        sa.UniqueConstraint("person_id", "content_sha256"),
        sa.Index(
            "ix_person_photo_person_status_created",
            "person_id",
            "status",
            sa.desc("created_at"),
        ),
        sa.CheckConstraint("status IN ('staged','active','failed','deleted')"),
    )

    op.create_table(
        "face_sample",
        sa.Column("sample_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("person_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("photo_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("detector_model", sa.String(128), nullable=False),
        sa.Column("embedding_model", sa.String(128), nullable=False),
        sa.Column(
            "bbox", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "landmarks", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column(
            "status", sa.String(16), nullable=False, server_default="staged"
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
        sa.PrimaryKeyConstraint("sample_id"),
        sa.ForeignKeyConstraint(["person_id"], ["person.person_id"]),
        sa.ForeignKeyConstraint(["photo_id"], ["person_photo.photo_id"]),
        sa.UniqueConstraint("photo_id"),
        sa.Index("ix_face_sample_person_status", "person_id", "status"),
        sa.CheckConstraint("status IN ('staged','active','failed','deleted')"),
    )

    op.create_table(
        "recognition_request",
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query_object_key", sa.String(512), nullable=True),
        sa.Column(
            "status", sa.String(16), nullable=False, server_default="pending"
        ),
        sa.Column("face_count", sa.Integer(), nullable=True),
        sa.Column(
            "top_k", sa.SmallInteger(), nullable=False, server_default="5"
        ),
        sa.Column(
            "threshold", sa.Float(), nullable=False, server_default="0.6"
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("request_id"),
        sa.Index("ix_recognition_request_created", sa.desc("created_at")),
        sa.CheckConstraint("top_k BETWEEN 1 AND 20"),
        sa.CheckConstraint("threshold >= 0 AND threshold <= 1"),
        sa.CheckConstraint("status IN ('pending','completed','failed')"),
    )

    op.create_table(
        "recognition_result",
        sa.Column("result_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "request_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column("face_index", sa.Integer(), nullable=False),
        sa.Column("recognition_status", sa.String(16), nullable=False),
        sa.Column(
            "bbox", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("best_person_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("best_photo_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("best_sample_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("best_score", sa.Float(), nullable=True),
        sa.Column(
            "candidates",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
        sa.PrimaryKeyConstraint("result_id"),
        sa.ForeignKeyConstraint(["request_id"], ["recognition_request.request_id"]),
        sa.ForeignKeyConstraint(["best_person_id"], ["person.person_id"]),
        sa.ForeignKeyConstraint(["best_photo_id"], ["person_photo.photo_id"]),
        sa.ForeignKeyConstraint(["best_sample_id"], ["face_sample.sample_id"]),
        sa.UniqueConstraint("request_id", "face_index"),
        sa.Index("ix_recognition_result_request_face", "request_id", "face_index"),
        sa.CheckConstraint("recognition_status IN ('known','unknown')"),
    )


def downgrade() -> None:
    op.drop_table("recognition_result")
    op.drop_table("recognition_request")
    op.drop_table("face_sample")
    op.drop_table("person_photo")
    op.drop_table("person")
