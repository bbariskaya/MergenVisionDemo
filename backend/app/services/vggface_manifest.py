"""Deterministic, streaming VGGFace manifest builder.

The VGGFace dataset ships as identity-labelled folders such as
``faces/n000024/0001_01.jpg``.  Each folder maps to one ``Person``/
``FaceIdentity``.  Only the numeric folder identity is used transiently for
enumeration; it never appears in MinIO keys, Qdrant payloads, public logs, or
benchmark output.  Display names are controlled labels because no verified
human-readable metadata is available locally.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from app.core.config import settings
from app.core.ids import derive_face_identity_id, derive_person_id, identity_hmac
from app.services.bulk_manifest import EnrollmentIdentity, EnrollmentPhoto


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass(frozen=True)
class VggfacePreflight:
    root: Path
    identity_count: int
    photo_count: int
    duplicate_photo_count: int
    corrupt_paths_count: int


def _content_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _identity_key(folder_name: str) -> str:
    return f"vggface:{folder_name.strip()}"


def _display_name(folder_name: str) -> str:
    return f"VGGFace {folder_name.strip()}"


def _list_photo_paths(folder: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    )


def _build_identity(folder_path: Path) -> EnrollmentIdentity:
    folder_name = folder_path.name
    identity_key = _identity_key(folder_name)
    display_name = _display_name(folder_name)
    hmac_val = identity_hmac(identity_key, settings.hmac_key)
    photos = tuple(
        EnrollmentPhoto(path=p, content_sha256="")
        for p in _list_photo_paths(folder_path)
    )
    return EnrollmentIdentity(
        identity_key=identity_key,
        display_name=display_name,
        identity_hmac=hmac_val,
        person_id=str(derive_person_id(hmac_val)),
        face_identity_id=str(derive_face_identity_id(hmac_val)),
        source_dataset="vggface",
        photos=photos,
    )


def vggface_preflight(root: Path) -> VggfacePreflight:
    """Return sanitized counts without reading image bytes.

    Content SHA-256 hashing is deferred to extraction time; duplicate detection
    is skipped during preflight to avoid scanning every file on every job start.
    """
    if not root.is_dir():
        raise ValueError(f"VGGFace root not found: {root}")
    faces_root = root / "faces" if (root / "faces").is_dir() else root
    identity_count = 0
    photo_count = 0
    corrupt_count = 0

    for folder in sorted(p for p in faces_root.iterdir() if p.is_dir()):
        identity_count += 1
        try:
            photo_count += len(_list_photo_paths(folder))
        except OSError:
            corrupt_count += 1

    return VggfacePreflight(
        root=root,
        identity_count=identity_count,
        photo_count=photo_count,
        duplicate_photo_count=0,
        corrupt_paths_count=corrupt_count,
    )


def _folder_bucket(folder: Path, num_shards: int) -> int:
    folder_name = folder.name.strip()
    identity_key = _identity_key(folder_name)
    hmac_val = identity_hmac(identity_key, settings.hmac_key)
    person_id = str(derive_person_id(hmac_val))
    digest = hashlib.sha256(person_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def stream_vggface_manifest(
    root: Path,
    *,
    max_identities: int | None = None,
    max_photos: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
    resume_after_identity_key: str | None = None,
) -> Iterator[EnrollmentIdentity]:
    """Yield identities lazily with deterministic, source-namespaced IDs.

    When ``shard_index`` and ``num_shards`` are supplied, only identities whose
    deterministic shard bucket matches are built. This avoids hashing photos on
    workers that will discard the identity.

    ``resume_after_identity_key`` skips identities up to and including the
    supplied key, so a resumed job does not rescan already completed ones.
    """
    if not root.is_dir():
        raise ValueError(f"VGGFace root not found: {root}")
    faces_root = root / "faces" if (root / "faces").is_dir() else root
    folders = sorted(p for p in faces_root.iterdir() if p.is_dir())

    sharding = num_shards is not None and shard_index is not None
    if sharding:
        if num_shards <= 0:
            raise ValueError("num_shards must be positive")
        if not 0 <= shard_index < num_shards:  # type: ignore[arg-type]
            raise ValueError("shard_index out of range")

    built = 0
    photos_seen = 0
    for folder in folders:
        folder_name = folder.name.strip()
        identity_key = _identity_key(folder_name)
        if resume_after_identity_key is not None and identity_key <= resume_after_identity_key:
            continue
        if sharding and _folder_bucket(folder, num_shards) != shard_index:  # type: ignore[arg-type]
            continue
        identity = _build_identity(folder)
        if identity.photos:
            if max_photos is not None and photos_seen + len(identity.photos) > max_photos:
                remaining = max(0, max_photos - photos_seen)
                if remaining > 0:
                    identity = EnrollmentIdentity(
                        identity_key=identity.identity_key,
                        display_name=identity.display_name,
                        identity_hmac=identity.identity_hmac,
                        person_id=identity.person_id,
                        face_identity_id=identity.face_identity_id,
                        source_dataset=identity.source_dataset,
                        photos=identity.photos[:remaining],
                    )
                    yield identity
                break
            yield identity
            photos_seen += len(identity.photos)
            built += 1
            if max_identities is not None and built >= max_identities:
                break


def shard_vggface_identities(
    identities: Iterator[EnrollmentIdentity],
    shard_index: int,
    num_shards: int,
) -> Iterator[EnrollmentIdentity]:
    """Assign identities to shards using a stable hash of ``person_id``.

    Prefer :func:`stream_vggface_manifest` with sharding for large datasets.
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index out of range")
    for identity in identities:
        digest = hashlib.sha256(identity.person_id.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % num_shards
        if bucket == shard_index:
            yield identity
