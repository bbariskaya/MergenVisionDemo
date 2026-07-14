"""GPU decode+preprocess parity via standalone subprocess.

Pytest-asyncio's event-loop wrapper conflicts with the primary CUDA context
used by nvImageCodec + CV-CUDA in this container, so the real runtime evidence
is produced by a standalone synchronous Python process and asserted here.
"""
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("test_gpu_preprocess_standalone.py")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("nvidia.nvimgcodec") is None,
    reason="nvidia-nvimgcodec-cu12 not installed",
)
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("cvcuda") is None,
    reason="cvcuda-cu12 not installed",
)
def test_preprocess_gpu_matches_cpu_reference():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd="/app",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "finite True" in result.stdout
    # Standalone prints max_abs_err; accept the same bounded tolerance.
    for line in result.stdout.splitlines():
        if line.startswith("max_abs_err"):
            err = float(line.split()[-1])
            assert err < 0.3, f"max_abs_err={err}"
            break
    else:
        raise AssertionError("max_abs_err not found in standalone output")
