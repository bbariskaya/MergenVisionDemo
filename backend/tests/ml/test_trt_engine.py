import threading
from pathlib import Path

import numpy as np
import pytest

from app.core.config import settings
from app.ml.pipeline import FacePipeline
from app.ml.preprocess import load_image, preprocess_detector
from app.ml.trt_engine import TrtEngine


SAMPLES_DIR = Path("/app/artifacts/samples")


def test_trt_engine_input_name_validation(pipeline: FacePipeline) -> None:
    tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
    with pytest.raises(ValueError, match="Input name mismatch"):
        pipeline.detector.infer({"wrong_name": tensor})

    with pytest.raises(ValueError, match="Input name mismatch"):
        pipeline.detector.infer({"input.1": tensor, "extra": tensor})


def test_trt_engine_close_idempotent_and_blocks_infer() -> None:
    engine = TrtEngine(settings.detector_engine_path)
    tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
    engine.infer({"input.1": tensor})
    engine.close()
    engine.close()  # idempotent

    with pytest.raises(RuntimeError, match="closed"):
        engine.infer({"input.1": tensor})


def test_trt_engine_thread_safety(pipeline: FacePipeline) -> None:
    image = load_image(SAMPLES_DIR / "t1.jpg")
    tensor, _ = preprocess_detector(image, settings.detector_input_size)

    result_a: dict = {}
    result_b: dict = {}

    def run_a():
        result_a["out"] = pipeline.detector.infer({"input.1": tensor.copy()})

    def run_b():
        result_b["out"] = pipeline.detector.infer({"input.1": tensor.copy()})

    t1 = threading.Thread(target=run_a)
    t2 = threading.Thread(target=run_b)
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert t1.is_alive() is False
    assert t2.is_alive() is False
    assert result_a["out"] is not None
    assert result_b["out"] is not None
    for name in result_a["out"]:
        np.testing.assert_allclose(
            result_a["out"][name], result_b["out"][name], rtol=1e-5, atol=1e-5
        )
