import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import httpx

DEFAULT_ARTIFACTS_DIR = Path("/app/artifacts")
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
MODELS_DIR = ARTIFACTS_DIR / "models"
MANIFEST_PATH = ARTIFACTS_DIR / "model_manifest.json"

URLS = {
    "antelopev2": "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
}

CHUNK_SIZE = 1024 * 1024


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url} -> {part}")
    with httpx.Client(follow_redirects=True, timeout=300.0) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(part, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
    part.replace(dest)
    print(f"Saved {dest}")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    print(f"Extracting {zip_path} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("Extraction complete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pack", default="antelopev2")
    args = parser.parse_args()

    model_pack = args.model_pack
    if model_pack not in URLS:
        print(f"Unsupported model pack: {model_pack}", file=sys.stderr)
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    pack_manifest = manifest.setdefault(model_pack, {})

    url = URLS[model_pack]
    zip_name = f"{model_pack}.zip"
    zip_path = MODELS_DIR / zip_name
    extract_dir = MODELS_DIR / model_pack

    expected_sha = pack_manifest.get("sha256")

    if zip_path.exists():
        actual_sha = sha256_file(zip_path)
        if expected_sha and actual_sha == expected_sha:
            print(f"{zip_path} already exists with matching SHA-256; skipping download")
        else:
            print(f"{zip_path} exists but SHA mismatch or missing; re-downloading")
            download_file(url, zip_path)
    else:
        download_file(url, zip_path)

    actual_sha = sha256_file(zip_path)
    pack_manifest["sha256"] = actual_sha
    pack_manifest["url"] = url
    pack_manifest["filename"] = zip_name
    save_manifest(manifest)
    print(f"SHA-256: {actual_sha}")

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_zip(zip_path, extract_dir)

    extracted = [str(p.relative_to(MODELS_DIR)) for p in extract_dir.rglob("*") if p.is_file()]
    pack_manifest["extracted_files"] = extracted
    save_manifest(manifest)
    print(f"Extracted {len(extracted)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
