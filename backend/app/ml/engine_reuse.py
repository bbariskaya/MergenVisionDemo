"""Pure helper for deciding whether an existing TensorRT engine can be reused."""
from __future__ import annotations

from typing import Any


DEFAULT_REQUIRED_KEYS = (
    "onnx_sha256",
    "engine_sha256",
    "trt_version",
    "cuda_version",
    "gpu",
    "precision",
    "profiles",
)


def metadata_matches(
    existing: dict[str, Any] | None,
    *,
    onnx_sha256: str,
    engine_sha256: str,
    trt_version: str,
    cuda_version: str,
    gpu_compute_capability: str,
    precision: str,
    profiles: Any,
) -> bool:
    """Return True if the stored engine metadata is compatible with current build plan.

    All fields must be present and equal. A mismatch in any version/capability
    means a rebuild is required.
    """
    if not existing:
        return False

    for key in DEFAULT_REQUIRED_KEYS:
        if key not in existing:
            return False

    if existing.get("engine_sha256") != engine_sha256:
        return False

    if existing.get("onnx_sha256") != onnx_sha256:
        return False
    if existing.get("trt_version") != trt_version:
        return False
    if existing.get("cuda_version") != cuda_version:
        return False

    gpu = existing.get("gpu") or {}
    if gpu.get("compute_capability") != gpu_compute_capability:
        return False

    if existing.get("precision") != precision:
        return False
    if existing.get("profiles") != profiles:
        return False

    return True
