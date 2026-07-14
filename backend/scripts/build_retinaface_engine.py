import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ml.engine_reuse import metadata_matches  # noqa: E402

DEFAULT_ARTIFACTS_DIR = Path("/app/artifacts")
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
MODELS_DIR = ARTIFACTS_DIR / "models"
ENGINES_DIR = ARTIFACTS_DIR / "engines"
METADATA_PATH = ARTIFACTS_DIR / "engine_metadata.json"

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


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
    smoke = runtime.deserialize_cuda_engine(part_path.read_bytes())
    if smoke is None:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(f"Deserialized smoke test failed for {engine_path}")
    del smoke
    part_path.replace(engine_path)
    engine_path_bytes = engine_path.read_bytes()

    tensor_info = []
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


def get_gpu_info() -> dict:
    import subprocess

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    onnx_path = MODELS_DIR / "retinaface_r50_dynamic.onnx"
    engine_path = ENGINES_DIR / "retinaface_r50_dynamic.engine"
    if not onnx_path.exists():
        print(f"ONNX not found: {onnx_path}", file=sys.stderr)
        return 1

    ENGINES_DIR.mkdir(parents=True, exist_ok=True)
    gpu_info = get_gpu_info()
    cuda_version = os.environ.get("CUDA_VERSION", "unknown")
    trt_version = trt.__version__
    onnx_sha = sha256_file(onnx_path)

    manifest = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    pack_manifest = manifest.setdefault("retinaface_r50", {})
    existing = pack_manifest.get("detector")

    input_profiles = {
        "input": (
            [1, 3, 640, 640],
            [64, 3, 640, 640],
            [256, 3, 640, 640],
        )
    }
    expected_profiles = [
        {"input": {"min": [1, 3, 640, 640], "opt": [64, 3, 640, 640], "max": [256, 3, 640, 640]}}
    ]
    engine_sha = sha256_file(engine_path) if engine_path.exists() else ""
    cc = gpu_info.get("compute_capability") if gpu_info else ""

    if not args.force and engine_path.exists():
        if metadata_matches(
            existing,
            onnx_sha256=onnx_sha,
            engine_sha256=engine_sha,
            trt_version=trt_version,
            cuda_version=cuda_version,
            gpu_compute_capability=cc,
            precision="FP16",
            profiles=expected_profiles,
        ):
            print(f"Reusing existing {engine_path}")
            return 0
        raise RuntimeError(
            f"Existing engine {engine_path} is stale or incompatible. Use --force to rebuild."
        )

    print(f"Building RetinaFace R50 dynamic engine from {onnx_path}")
    start = time.time()
    info = build_engine(onnx_path, engine_path, fp16=True, input_profiles=input_profiles)[1]
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
        "precision": "FP16",
        "profiles": expected_profiles,
        "tensors": info["tensors"],
    }
    pack_manifest["detector"] = entry
    METADATA_PATH.write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
