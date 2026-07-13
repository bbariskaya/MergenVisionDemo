import os
import sys
from pathlib import Path

import httpx

DEFAULT_ARTIFACTS_DIR = Path("/app/artifacts")
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
SAMPLES_DIR = ARTIFACTS_DIR / "samples"

SAMPLES = {
    "t1.jpg": "https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/data/images/t1.jpg",
    "mask_blue.jpg": "https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/data/images/mask_blue.jpg",
    "Tom_Hanks_54745.png": "https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/data/images/Tom_Hanks_54745.png",
}


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in SAMPLES.items():
        dest = SAMPLES_DIR / filename
        if dest.exists():
            print(f"{dest} already exists")
            continue
        print(f"Downloading {url} -> {dest}")
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            r = client.get(url)
            r.raise_for_status()
            dest.write_bytes(r.content)
        print(f"Saved {dest} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
