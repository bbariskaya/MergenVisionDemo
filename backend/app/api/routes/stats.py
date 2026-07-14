"""Read-only system statistics routes."""
from __future__ import annotations

import sqlalchemy as sa
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.domain.models import FaceIdentity, FaceSample, Person, PersonPhoto, RecognitionRequest
from app.schemas.stats import EnrollmentStats

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get(
    "",
    response_model=EnrollmentStats,
    summary="Enrollment statistics",
)
async def get_enrollment_stats(
    db: AsyncSession = Depends(get_db),
) -> EnrollmentStats:
    person_count = (
        await db.execute(sa.select(sa.func.count()).select_from(Person))
    ).scalar() or 0
    # A face is the canonical identity per person; count active identities.
    face_count = (
        await db.execute(
            sa.select(sa.func.count())
            .select_from(FaceIdentity)
            .where(FaceIdentity.is_active.is_(True))
            .where(
                sa.exists()
                .where(Person.face_identity_id == FaceIdentity.face_identity_id)
                .where(Person.is_active.is_(True))
            )
        )
    ).scalar() or 0
    photo_count = (
        await db.execute(
            sa.select(sa.func.count()).select_from(PersonPhoto).where(PersonPhoto.status == "active")
        )
    ).scalar() or 0
    recognition_count = (
        await db.execute(sa.select(sa.func.count()).select_from(RecognitionRequest))
    ).scalar() or 0
    active_person_count = (
        await db.execute(
            sa.select(sa.func.count()).select_from(Person).where(Person.is_active.is_(True))
        )
    ).scalar() or 0
    return EnrollmentStats(
        person_count=int(person_count),
        face_count=int(face_count),
        photo_count=int(photo_count),
        recognition_count=int(recognition_count),
        active_person_count=int(active_person_count),
    )
