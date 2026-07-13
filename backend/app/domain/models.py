import uuid
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, SmallInteger, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.ids import new_uuid7


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {
        dict[str, Any]: JSONB,
        list[dict[str, Any]]: JSONB,
    }


class Person(Base):
    __tablename__ = "person"

    person_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_uuid7
    )
    first_name: Mapped[str] = mapped_column(String(255), nullable=False)
    last_name: Mapped[str] = mapped_column(String(255), nullable=False)
    national_id_lookup_hmac: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )
    national_id_masked: Mapped[str] = mapped_column(String(32), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    photos: Mapped[list["PersonPhoto"]] = relationship(
        back_populates="person", lazy="selectin"
    )
    samples: Mapped[list["FaceSample"]] = relationship(
        back_populates="person", lazy="selectin"
    )

    __table_args__ = (
        sa.Index("ix_person_active_created", "is_active", sa.desc("created_at")),
    )


class PersonPhoto(Base):
    __tablename__ = "person_photo"

    photo_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_uuid7
    )
    person_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("person.person_id"), nullable=False
    )
    object_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="staged"
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    person: Mapped["Person"] = relationship(back_populates="photos")
    sample: Mapped["FaceSample | None"] = relationship(
        back_populates="photo", uselist=False
    )

    __table_args__ = (
        sa.UniqueConstraint("person_id", "content_sha256"),
        sa.Index(
            "ix_person_photo_person_status_created",
            "person_id",
            "status",
            sa.desc("created_at"),
        ),
        sa.CheckConstraint("status IN ('staged','active','failed','deleted')"),
    )


class FaceSample(Base):
    __tablename__ = "face_sample"

    sample_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_uuid7
    )
    person_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("person.person_id"), nullable=False
    )
    photo_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("person_photo.photo_id"), nullable=False, unique=True
    )
    detector_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    bbox: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    landmarks: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="staged"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    person: Mapped["Person"] = relationship(back_populates="samples")
    photo: Mapped["PersonPhoto"] = relationship(back_populates="sample")

    __table_args__ = (
        sa.Index("ix_face_sample_person_status", "person_id", "status"),
        sa.CheckConstraint("status IN ('staged','active','failed','deleted')"),
    )


class RecognitionRequest(Base):
    __tablename__ = "recognition_request"

    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_uuid7
    )
    query_object_key: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending"
    )
    face_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    top_k: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=5)
    threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.6)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        sa.Index("ix_recognition_request_created", sa.desc("created_at")),
        sa.CheckConstraint("top_k BETWEEN 1 AND 20"),
        sa.CheckConstraint("threshold >= 0 AND threshold <= 1"),
        sa.CheckConstraint("status IN ('pending','completed','failed')"),
    )


class RecognitionResult(Base):
    __tablename__ = "recognition_result"

    result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=new_uuid7
    )
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("recognition_request.request_id"), nullable=False
    )
    face_index: Mapped[int] = mapped_column(Integer, nullable=False)
    recognition_status: Mapped[str] = mapped_column(String(16), nullable=False)
    bbox: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    best_person_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("person.person_id"), nullable=True
    )
    best_photo_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("person_photo.photo_id"), nullable=True
    )
    best_sample_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("face_sample.sample_id"), nullable=True
    )
    best_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    candidates: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    __table_args__ = (
        sa.UniqueConstraint("request_id", "face_index"),
        sa.Index("ix_recognition_result_request_face", "request_id", "face_index"),
        sa.CheckConstraint("recognition_status IN ('known','unknown')"),
    )
