import subprocess
from pathlib import Path
from typing import Iterator

import pytest

from app.ml.pipeline import FacePipeline


SAMPLES_DIR = Path("/app/artifacts/samples")
LFW_DIR = Path("/app/lfw/lfw-deepfunneled/lfw-deepfunneled")


@pytest.fixture(scope="session", autouse=True)
def ensure_samples():
    if not (SAMPLES_DIR / "t1.jpg").exists():
        subprocess.run(
            ["python", "scripts/download_samples.py"],
            cwd="/app",
            check=True,
        )


@pytest.fixture(scope="session")
def pipeline() -> Iterator[FacePipeline]:
    pipe = FacePipeline()
    pipe.warmup()
    yield pipe
    pipe.close()
