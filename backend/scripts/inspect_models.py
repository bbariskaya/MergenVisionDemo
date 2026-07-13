import argparse
import json
import os
import sys
from pathlib import Path

import onnx

DEFAULT_ARTIFACTS_DIR = Path("/app/artifacts")
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

EXPECTED_EMBEDDING_DIM = 512


def inspect_onnx(path: Path) -> dict:
    model = onnx.load(str(path))
    opset = model.opset_import[0].version if model.opset_import else None
    graph = model.graph

    inputs = [
        {
            "name": i.name,
            "shape": [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim],
            "dtype": onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type),
        }
        for i in graph.input
    ]
    outputs = [
        {
            "name": o.name,
            "shape": [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim],
            "dtype": onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type),
        }
        for o in graph.output
    ]
    return {
        "file": str(path.relative_to(ARTIFACTS_DIR)),
        "file_size_bytes": path.stat().st_size,
        "opset_version": opset,
        "producer": model.producer_name or None,
        "domain": model.domain or None,
        "inputs": inputs,
        "outputs": outputs,
        "initializer_count": len(graph.initializer),
    }


def is_detector(meta: dict) -> bool:
    outputs = meta["outputs"]
    if len(outputs) < 3:
        return False
    # SCRFD-style: groups of score(1), bbox(4), landmark(10) across FPN levels.
    counts: dict[int, list[int]] = {}
    for o in outputs:
        shape = o["shape"]
        if len(shape) != 2:
            return False
        count = shape[0] if isinstance(shape[0], int) else 0
        dim2 = shape[1] if isinstance(shape[1], int) else 0
        counts.setdefault(count, []).append(dim2)
    for dims in counts.values():
        if sorted(dims) == [1, 4, 10]:
            return True
    return False


def is_recognizer(meta: dict) -> bool:
    inputs = meta["inputs"]
    outputs = meta["outputs"]
    if not inputs or not outputs:
        return False
    in_shape = inputs[0]["shape"]
    out_shape = outputs[0]["shape"]
    # Typical ArcFace R100: input [N,3,112,112], output [N,512]
    if len(in_shape) != 4:
        return False
    if len(out_shape) != 2:
        return False
    if out_shape[-1] != EXPECTED_EMBEDDING_DIM:
        return False
    return True


def make_symlink(name: str, target: Path) -> None:
    link_path = MODELS_DIR / f"{name}.onnx"
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
    except Exception as exc:
        print(f"Warning: could not remove old {link_path}: {exc}")
    try:
        # Relative symlink so it works whether read via /models or /app/artifacts/models
        rel = os.path.relpath(target, start=link_path.parent)
        os.symlink(rel, link_path)
        print(f"Linked {link_path.name} -> {rel}")
    except Exception as exc:
        print(f"Warning: could not create symlink {link_path}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pack", default="antelopev2")
    args = parser.parse_args()

    pack_dir = MODELS_DIR / args.model_pack
    if not pack_dir.exists():
        print(f"Model pack not found: {pack_dir}", file=sys.stderr)
        return 1

    onnx_files = sorted(pack_dir.rglob("*.onnx"))
    if not onnx_files:
        print("No .onnx files found", file=sys.stderr)
        return 1

    metas = [inspect_onnx(p) for p in onnx_files]
    detector = None
    recognizer = None

    for meta in metas:
        path = ARTIFACTS_DIR / meta["file"]
        if is_recognizer(meta):
            recognizer = meta
            make_symlink("recognizer", path)
        elif is_detector(meta):
            detector = meta
            make_symlink("detector", path)

    metadata = {
        "model_pack": args.model_pack,
        "models": metas,
        "selected": {
            "detector": detector["file"] if detector else None,
            "recognizer": recognizer["file"] if recognizer else None,
        },
    }

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata written to {METADATA_PATH}")
    print(json.dumps(metadata["selected"], indent=2))

    if not detector or not recognizer:
        print("ERROR: Could not identify both detector and recognizer", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
