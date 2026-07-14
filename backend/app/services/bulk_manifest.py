"""Build a deterministic, person-grouped manifest from an LFW-style folder tree.

A manifest groups all photos under one folder as one ``EnrollmentIdentity``.
All deterministic UUIDs and HMACs are derived from the normalized folder name,
never from filesystem order or generated placeholders.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from app.core.config import settings
from app.core.ids import (
    derive_face_identity_id,
    derive_person_id,
    derive_photo_id,
    identity_hmac,
)


@dataclass
class EnrollmentPhoto:
    path: Path
    content_sha256: str
    data: bytes | None = None

    @property
    def photo_id(self) -> str:
        return str(derive_photo_id(self.content_sha256))


@dataclass(frozen=True)
class EnrollmentIdentity:
    identity_key: str
    display_name: str
    identity_hmac: str
    person_id: str
    face_identity_id: str
    source_dataset: str
    photos: tuple[EnrollmentPhoto, ...]


def normalize_lfw_folder_name(folder_name: str) -> tuple[str, str]:
    """Return (identity_key, display_name) for an LFW folder.

    ``identity_key`` retains the original normalized token (e.g.
    ``Jennifer_Aniston``).  ``display_name`` is the human-readable rendering
    (e.g. ``Jennifer Aniston``).
    """
    key = folder_name.strip()
    display = key.replace("_", " ")
    return key, display


def _content_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_lfw_manifest(
    root: Path,
    *,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> tuple[EnrollmentIdentity, ...]:
    """Build a manifest from a ``lfw-deepfunneled`` root folder.

    Returns identities sorted by ``identity_key``.  Each identity contains its
    photos sorted by filename for deterministic ordering.
    """
    identities: list[EnrollmentIdentity] = []
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        raw_key, display_name = normalize_lfw_folder_name(person_dir.name)
        identity_key = f"lfw:{raw_key}"
        photos = sorted(
            (
                EnrollmentPhoto(path=p, content_sha256="")
                for p in person_dir.iterdir()
                if p.is_file() and p.suffix.lower() in extensions
            ),
            key=lambda photo: photo.path.name,
        )
        if not photos:
            continue
        hmac_val = identity_hmac(identity_key, settings.hmac_key)
        identities.append(
            EnrollmentIdentity(
                identity_key=identity_key,
                display_name=display_name,
                identity_hmac=hmac_val,
                person_id=str(derive_person_id(hmac_val)),
                face_identity_id=str(derive_face_identity_id(hmac_val)),
                source_dataset="lfw",
                photos=tuple(photos),
            )
        )
    return tuple(sorted(identities, key=lambda identity: identity.identity_key))


def shard_by_person_id(
    identities: tuple[EnrollmentIdentity, ...],
    num_shards: int,
) -> tuple[tuple[EnrollmentIdentity, ...], ...]:
    """Assign identities to shards deterministically by ``person_id``."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    shards: list[list[EnrollmentIdentity]] = [[] for _ in range(num_shards)]
    for identity in identities:
        shard = uuid.UUID(identity.person_id).int % num_shards
        shards[shard].append(identity)
    return tuple(tuple(sorted(s, key=lambda i: i.identity_key)) for s in shards)


def build_casia_manifest(
    root: Path,
    *,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> tuple[EnrollmentIdentity, ...]:
    """Build a manifest from a CASIA-WebFace folder tree.

    CASIA identity folders are already prefixed (e.g. ``casia_0000045``) so we
    use the folder name directly without adding a dataset namespace.
    """
    identities: list[EnrollmentIdentity] = []
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        key = person_dir.name
        photos = sorted(
            (
                EnrollmentPhoto(path=p, content_sha256="")
                for p in person_dir.iterdir()
                if p.is_file() and p.suffix.lower() in extensions
            ),
            key=lambda photo: photo.path.name,
        )
        if not photos:
            continue
        hmac_val = identity_hmac(key, settings.hmac_key)
        identities.append(
            EnrollmentIdentity(
                identity_key=key,
                display_name=key,
                identity_hmac=hmac_val,
                person_id=str(derive_person_id(hmac_val)),
                face_identity_id=str(derive_face_identity_id(hmac_val)),
                source_dataset="casia",
                photos=tuple(photos),
            )
        )
    return tuple(sorted(identities, key=lambda identity: identity.identity_key))


def expected_cardinality(root: Path, *, dataset: str = "lfw") -> tuple[int, int]:
    """Return (num_persons, num_photos) before any import."""
    if dataset == "casia":
        identities = build_casia_manifest(root)
    else:
        identities = build_lfw_manifest(root)
    return len(identities), sum(len(i.photos) for i in identities)


def manifest_iter_photos(
    identities: tuple[EnrollmentIdentity, ...],
) -> Iterator[tuple[EnrollmentIdentity, EnrollmentPhoto]]:
    for identity in identities:
        for photo in identity.photos:
            yield identity, photo
