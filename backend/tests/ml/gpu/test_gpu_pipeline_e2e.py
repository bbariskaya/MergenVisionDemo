"""GPU/CPU end-to-end face extraction parity on real images."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_RUNNER = Path(__file__).with_name("_run_gpu_pipeline_e2e.py")


@pytest.mark.skipif(
    importlib.util.find_spec("mergenvision_gpu") is None,
    reason="mergenvision-gpu native extension not installed",
)
@pytest.mark.skipif(
    importlib.util.find_spec("nvidia.nvimgcodec") is None,
    reason="nvidia-nvimgcodec-cu12 not installed",
)
@pytest.mark.skipif(
    not Path("/app/lfw/lfw-deepfunneled").exists(),
    reason="LFW dataset not mounted",
)
def test_gpu_pipeline_matches_cpu_on_lfw():
    result = subprocess.run(
        [sys.executable, str(_RUNNER)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 77:
        pytest.skip("E2E runner conditions not met")
    if result.returncode != 0:
        title = "E2E parity runner failed"
        raise AssertionError(
            f"{title}: code={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    metrics = json.loads(result.stdout)
    print(f"GPU/CPU LFW parity metrics: {metrics}")
    assert metrics["count"] >= 1, "No comparable faces found"
    assert metrics["min"] >= 0.990, (
        f"GPU/CPU embedding similarity too low: min={metrics['min']:.4f}"
    )
