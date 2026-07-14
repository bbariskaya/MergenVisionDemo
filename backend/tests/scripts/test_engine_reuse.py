import pytest

from app.ml.engine_reuse import metadata_matches


def _base_metadata():
    return {
        "onnx_sha256": "a" * 64,
        "engine_sha256": "e" * 64,
        "trt_version": "10.3.0",
        "cuda_version": "12.4.1",
        "gpu": {"name": "Quadro RTX 8000", "compute_capability": "7.5"},
        "precision": "FP16",
        "profiles": [{"input.1": {"min": [1, 3, 640, 640], "opt": [1, 3, 640, 640], "max": [1, 3, 640, 640]}}],
    }


def test_metadata_matches_exact():
    existing = _base_metadata()
    assert metadata_matches(
        existing,
        onnx_sha256=existing["onnx_sha256"],
        engine_sha256=existing["engine_sha256"],
        trt_version=existing["trt_version"],
        cuda_version=existing["cuda_version"],
        gpu_compute_capability="7.5",
        precision="FP16",
        profiles=existing["profiles"],
    ) is True


def test_metadata_matches_missing_engine():
    assert metadata_matches(
        None,
        onnx_sha256="a" * 64,
        engine_sha256="e" * 64,
        trt_version="10.3.0",
        cuda_version="12.4.1",
        gpu_compute_capability="7.5",
        precision="FP16",
        profiles=[],
    ) is False


def test_metadata_matches_onnx_sha_mismatch():
    existing = _base_metadata()
    assert metadata_matches(
        existing,
        onnx_sha256="b" * 64,
        engine_sha256=existing["engine_sha256"],
        trt_version=existing["trt_version"],
        cuda_version=existing["cuda_version"],
        gpu_compute_capability="7.5",
        precision="FP16",
        profiles=existing["profiles"],
    ) is False


def test_metadata_matches_engine_sha_mismatch():
    existing = _base_metadata()
    assert metadata_matches(
        existing,
        onnx_sha256=existing["onnx_sha256"],
        engine_sha256="f" * 64,
        trt_version=existing["trt_version"],
        cuda_version=existing["cuda_version"],
        gpu_compute_capability="7.5",
        precision="FP16",
        profiles=existing["profiles"],
    ) is False


def test_metadata_matches_trt_version_mismatch():
    existing = _base_metadata()
    assert metadata_matches(
        existing,
        onnx_sha256=existing["onnx_sha256"],
        engine_sha256=existing["engine_sha256"],
        trt_version="10.4.0",
        cuda_version=existing["cuda_version"],
        gpu_compute_capability="7.5",
        precision="FP16",
        profiles=existing["profiles"],
    ) is False


def test_metadata_matches_cuda_version_mismatch():
    existing = _base_metadata()
    assert metadata_matches(
        existing,
        onnx_sha256=existing["onnx_sha256"],
        engine_sha256=existing["engine_sha256"],
        trt_version=existing["trt_version"],
        cuda_version="12.5.1",
        gpu_compute_capability="7.5",
        precision="FP16",
        profiles=existing["profiles"],
    ) is False


def test_metadata_matches_compute_capability_mismatch():
    existing = _base_metadata()
    assert metadata_matches(
        existing,
        onnx_sha256=existing["onnx_sha256"],
        engine_sha256=existing["engine_sha256"],
        trt_version=existing["trt_version"],
        cuda_version=existing["cuda_version"],
        gpu_compute_capability="8.6",
        precision="FP16",
        profiles=existing["profiles"],
    ) is False


def test_metadata_matches_precision_mismatch():
    existing = _base_metadata()
    assert metadata_matches(
        existing,
        onnx_sha256=existing["onnx_sha256"],
        engine_sha256=existing["engine_sha256"],
        trt_version=existing["trt_version"],
        cuda_version=existing["cuda_version"],
        gpu_compute_capability="7.5",
        precision="FP32",
        profiles=existing["profiles"],
    ) is False


def test_metadata_matches_profile_mismatch():
    existing = _base_metadata()
    assert metadata_matches(
        existing,
        onnx_sha256=existing["onnx_sha256"],
        engine_sha256=existing["engine_sha256"],
        trt_version=existing["trt_version"],
        cuda_version=existing["cuda_version"],
        gpu_compute_capability="7.5",
        precision="FP16",
        profiles=[{"input.1": {"min": [1, 3, 640, 640], "opt": [1, 3, 640, 640], "max": [1, 3, 1280, 1280]}}],
    ) is False
