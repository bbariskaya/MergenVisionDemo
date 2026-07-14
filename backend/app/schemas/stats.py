"""Pydantic schemas for system statistics endpoints."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class EnrollmentStats(_ApiModel):
    person_count: int = Field(alias="personCount")
    face_count: int = Field(alias="faceCount")
    photo_count: int = Field(alias="photoCount")
    recognition_count: int = Field(alias="recognitionCount")
    active_person_count: int = Field(alias="activePersonCount")
