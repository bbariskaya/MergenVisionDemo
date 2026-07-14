import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import tensorrt as trt

from app.ml.engine_reuse import metadata_matches

DEFAULT_ARTIFACTS_DIR = Path("/app/artifacts")
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
MODELS_DIR = ARTIFACTS_DIR / "models"
ENGINES_DIR = ARTIFACTS_DIR / "engines"
METADATA_PATH = ARTIFACTS_DIR / "engine_metadata.json"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def get_gpu_info() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip()
        name, cc = out.split(",", 1)
        return {"name": name.strip(), "compute_capability": cc.strip()}
    except Exception as exc:
        print(f"Warning: could not query GPU info: {exc}", file=sys.stderr)
        return {}


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool = True,
    input_profiles: dict[str, tuple[list[int], list[int], list[int]]] | None = None,
) -> tuple[trt.ICudaEngine, dict]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    profile = builder.create_optimization_profile()
    if input_profiles:
        for name, (min_shape, opt_shape, max_shape) in input_profiles.items():
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Engine build failed: {onnx_path}")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError(f"Engine deserialization failed: {onnx_path}")

    part_path = engine_path.with_suffix(engine_path.suffix + ".part")
    with open(part_path, "wb") as f:
        f.write(serialized)
    # Deserialize the part file as a smoke test before atomically promoting it.
    smoke = runtime.deserialize_cuda_engine(part_path.read_bytes())
    if smoke is None:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(f"Deserialized smoke test failed for {engine_path}")
    del smoke
    part_path.replace(engine_path)
    # Re-read from final path for metadata hash.
    engine_path_bytes = engine_path.read_bytes()

    tensor_info: list[dict] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = str(engine.get_tensor_mode(name))
        shape = list(engine.get_tensor_shape(name))
        dtype = str(engine.get_tensor_dtype(name))
        tensor_info.append({"name": name, "mode": mode, "shape": shape, "dtype": dtype})

    profiles_info = []
    if input_profiles:
        profiles_info.append(
            {
                name: {"min": mn, "opt": opt, "max": mx}
                for name, (mn, opt, mx) in input_profiles.items()
            }
        )

    return engine, {
        "tensors": tensor_info,
        "profiles": profiles_info,
        "fp16": fp16,
        "engine_sha256": hashlib.sha256(engine_path_bytes).hexdigest(),
    }


RECOGNIZER_INPUT_NAME = "input.1"
RECOGNIZER_PRECISION = "FP16"
RECOGNIZER_PROFILES = [
    {
        RECOGNIZER_INPUT_NAME: {
            "min": [1, 3, 112, 112],
            "opt": [128, 3, 112, 112],
            "max": [256, 3, 112, 112],
        }
    }
]


def build_recognizer(onnx_path: Path, engine_path: Path) -> dict:
    input_profiles = {
        RECOGNIZER_INPUT_NAME: (
            [1, 3, 112, 112],
            [128, 3, 112, 112],
            [256, 3, 112, 112],
        )
    }
    _, info = build_engine(onnx_path, engine_path, fp16=True, input_profiles=input_profiles)
    return info


DETECTOR_INPUT_NAME = "input.1"
DETECTOR_PRECISION = "FP16"
DETECTOR_PROFILES = [
    {
        DETECTOR_INPUT_NAME: {
            "min": [1, 3, 160, 160],
            "opt": [1, 3, 640, 640],
            "max": [1, 3, 1280, 1280],
        }
    }
]


def build_detector(onnx_path: Path, engine_path: Path) -> dict:
    input_profiles = {
        DETECTOR_INPUT_NAME: (
            [1, 3, 160, 160],
            [1, 3, 640, 640],
            [1, 3, 1280, 1280],
        )
    }
    _, info = build_engine(onnx_path, engine_path, fp16=True, input_profiles=input_profiles)
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pack", default="antelopev2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not MODEL_METADATA_PATH.exists():
        print("Model metadata not found; run inspect_models.py first", file=sys.stderr)
        return 1

    model_meta = json.loads(MODEL_METADATA_PATH.read_text())
    selected = model_meta.get("selected", {})
    detector_rel = selected.get("detector")
    recognizer_rel = selected.get("recognizer")
    if not detector_rel or not recognizer_rel:
        print("Detector/recognizer not selected", file=sys.stderr)
        return 1

    detector_onnx = ARTIFACTS_DIR / detector_rel
    recognizer_onnx = ARTIFACTS_DIR / recognizer_rel
    ENGINES_DIR.mkdir(parents=True, exist_ok=True)

    gpu_info = get_gpu_info()
    cuda_version = os.environ.get("CUDA_VERSION", "unknown")
    trt_version = trt.__version__

    manifest = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    pack_manifest = manifest.setdefault(args.model_pack, {})

    artifacts = {}

    def build_if_needed(
        name: str,
        onnx_path: Path,
        builder_fn,
        expected_precision: str,
        expected_profiles: list[dict],
    ) -> dict:
        engine_path = ENGINES_DIR / f"{name}.engine"
        onnx_sha = sha256_file(onnx_path)
        existing = pack_manifest.get(name)
        cc = gpu_info.get("compute_capability") if gpu_info else ""
        engine_sha = sha256_file(engine_path) if engine_path.exists() else ""

        if not args.force and engine_path.exists():
            if not metadata_matches(
                existing,
                onnx_sha256=onnx_sha,
                engine_sha256=engine_sha,
                trt_version=trt_version,
                cuda_version=cuda_version,
                gpu_compute_capability=cc,
                precision=expected_precision,
                profiles=expected_profiles,
            ):
                raise RuntimeError(
                    f"Existing engine {engine_path} is stale or incompatible. "
                    "Use --force to rebuild."
                )
            print(f"Reusing existing {engine_path}")
            return existing

        print(f"Building {name} engine from {onnx_path}")
        start = time.time()
        info = builder_fn(onnx_path, engine_path)
        elapsed = time.time() - start
        print(f"Built {engine_path} in {elapsed:.1f}s")

        entry = {
            "onnx_path": str(onnx_path.relative_to(ARTIFACTS_DIR)),
            "onnx_sha256": onnx_sha,
            "engine_sha256": info["engine_sha256"],
            "engine_path": str(engine_path.relative_to(ARTIFACTS_DIR)),
            "trt_version": trt_version,
            "cuda_version": cuda_version,
            "gpu": gpu_info,
            "precision": expected_precision,
            "profiles": expected_profiles,
            "tensors": info["tensors"],
            "build_duration_seconds": elapsed,
        }
        pack_manifest[name] = entry
        return entry

    artifacts["detector"] = build_if_needed(
        "detector",
        detector_onnx,
        build_detector,
        DETECTOR_PRECISION,
        DETECTOR_PROFILES,
    )
    artifacts["recognizer"] = build_if_needed(
        "recognizer",
        recognizer_onnx,
        build_recognizer,
        RECOGNIZER_PRECISION,
        RECOGNIZER_PROFILES,
    )

    METADATA_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nEngine metadata written to {METADATA_PATH}")
    print(json.dumps({k: v["engine_path"] for k, v in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
