"""Lightweight disk cache for VGGFace metadata.

The API request path must not SHA-256 the entire dataset on every job start.
This module caches counts, duplicate counts and the folder list keyed by the
dataset directory mtime.  Photo content hashes are intentionally *not* cached
across job boundaries; the worker streams identities within its budget and
hashes only the selected photos.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.services.vggface_manifest import VggfacePreflight, vggface_preflight

_CACHE_DIR = Path("/tmp/mergenvision_cache")
_CACHE_FILE = _CACHE_DIR / "vggface_manifest_cache.json"


@dataclass(frozen=True)
class VggfaceManifestCache:
    dataset_mtime: float
    preflight: VggfacePreflight

    def to_dict(self) -> dict:
        return {
            "dataset_mtime": self.dataset_mtime,
            "preflight": {
                "root": str(self.preflight.root),
                "identity_count": self.preflight.identity_count,
                "photo_count": self.preflight.photo_count,
                "duplicate_photo_count": self.preflight.duplicate_photo_count,
                "corrupt_paths_count": self.preflight.corrupt_paths_count,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VggfaceManifestCache":
        p = data["preflight"]
        return cls(
            dataset_mtime=float(data["dataset_mtime"]),
            preflight=VggfacePreflight(
                root=Path(p["root"]),
                identity_count=int(p["identity_count"]),
                photo_count=int(p["photo_count"]),
                duplicate_photo_count=int(p["duplicate_photo_count"]),
                corrupt_paths_count=int(p["corrupt_paths_count"]),
            ),
        )


def _dataset_mtime(path: Path) -> float:
    faces_root = path / "faces" if (path / "faces").is_dir() else path
    mtimes = [faces_root.stat().st_mtime]
    for folder in faces_root.iterdir():
        mtimes.append(folder.stat().st_mtime)
    return max(mtimes)


def get_vggface_preflight(path: Path | None = None) -> VggfacePreflight:
    """Return preflight counts, using a local disk cache when valid."""
    path = path or settings.vggface_dataset_path
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    current_mtime = _dataset_mtime(path)

    if _CACHE_FILE.exists():
        try:
            cache = VggfaceManifestCache.from_dict(json.loads(_CACHE_FILE.read_text()))
            if cache.dataset_mtime >= current_mtime:
                return cache.preflight
        except Exception:
            pass

    preflight = vggface_preflight(path)
    cache = VggfaceManifestCache(dataset_mtime=current_mtime, preflight=preflight)
    _CACHE_FILE.write_text(json.dumps(cache.to_dict()))
    return preflight
