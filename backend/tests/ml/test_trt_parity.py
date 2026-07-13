from pathlib import Path
from typing import Iterator

import numpy as np
import onnxruntime as ort
import pytest

from app.core.config import settings
from app.ml.alignment import align_face
from app.ml.pipeline import FacePipeline
from app.ml.postprocess import Detection, decode_detections
from app.ml.preprocess import load_image, preprocess_detector, preprocess_recognizer


ONNX_MODELS_DIR = Path("/app/artifacts/models/antelopev2/antelopev2")


@pytest.fixture(scope="session")
def onnx_detector() -> Iterator[ort.InferenceSession]:
    sess = ort.InferenceSession(
        str(ONNX_MODELS_DIR / "scrfd_10g_bnkps.onnx"),
        providers=["CPUExecutionProvider"],
    )
    yield sess


@pytest.fixture(scope="session")
def onnx_recognizer() -> Iterator[ort.InferenceSession]:
    sess = ort.InferenceSession(
        str(ONNX_MODELS_DIR / "glintr100.onnx"),
        providers=["CPUExecutionProvider"],
    )
    yield sess


@pytest.fixture(scope="session")
def pipeline() -> Iterator[FacePipeline]:
    pipe = FacePipeline()
    pipe.warmup()
    yield pipe
    pipe.close()


def _match_onnx_output(
    trt_arr: np.ndarray, onnx_outputs: list[np.ndarray]
) -> np.ndarray:
    for onnx_arr in onnx_outputs:
        if onnx_arr.shape == trt_arr.shape:
            return onnx_arr
    raise AssertionError(f"No ONNX output with shape {trt_arr.shape}")


def test_detector_raw_output_parity(pipeline, onnx_detector):
    image = load_image("/app/artifacts/samples/t1.jpg")
    tensor, _ = preprocess_detector(image, settings.detector_input_size)

    trt_outputs = pipeline.detector.infer({"input.1": tensor})
    onnx_outputs = onnx_detector.run(
        None, {onnx_detector.get_inputs()[0].name: tensor}
    )

    score_diffs = []
    bbox_diffs = []
    landmark_diffs = []
    for name, trt_arr in trt_outputs.items():
        onnx_arr = _match_onnx_output(trt_arr, onnx_outputs)
        diff = np.abs(trt_arr - onnx_arr)
        _, dim2 = trt_arr.shape
        if dim2 == 1:
            score_diffs.append(diff.max())
        elif dim2 == 4:
            bbox_diffs.append(diff.max())
        elif dim2 == 10:
            landmark_diffs.append(diff.max())

    assert max(score_diffs) < 0.001, f"score max diff {max(score_diffs)} too large"
    assert max(bbox_diffs) < 0.1, f"bbox max diff {max(bbox_diffs)} too large"
    assert max(landmark_diffs) < 0.05, f"landmark max diff {max(landmark_diffs)} too large"


def test_detector_postprocess_parity(pipeline, onnx_detector):
    image = load_image("/app/artifacts/samples/t1.jpg")
    tensor, _ = preprocess_detector(image, settings.detector_input_size)

    onnx_outputs = onnx_detector.run(
        None, {onnx_detector.get_inputs()[0].name: tensor}
    )
    onnx_out_dict = {
        o.name: onnx_arr
        for o, onnx_arr in zip(onnx_detector.get_outputs(), onnx_outputs)
    }
    onnx_detections = decode_detections(
        onnx_out_dict,
        input_size=settings.detector_input_size,
        conf_threshold=settings.detector_confidence_threshold,
        nms_threshold=settings.detector_nms_iou,
    )
    trt_detections = pipeline.extract_from_path(
        "/app/artifacts/samples/t1.jpg"
    )

    # Count parity should be close; exact equality is ideal but FP16 may vary.
    assert abs(len(trt_detections) - len(onnx_detections)) <= 1


def test_recognizer_embedding_parity(pipeline, onnx_recognizer):
    image = load_image("/app/artifacts/samples/t1.jpg")
    trt_extractions = pipeline.extract_from_path("/app/artifacts/samples/t1.jpg")
    assert len(trt_extractions) >= 1

    aligned = align_face(image, trt_extractions[0].landmarks)
    tensor = preprocess_recognizer(aligned)

    trt_emb = pipeline.recognizer.infer({"input.1": tensor})["1333"][0]
    onnx_emb = onnx_recognizer.run(
        None, {onnx_recognizer.get_inputs()[0].name: tensor}
    )[0][0]

    trt_norm = trt_emb / np.linalg.norm(trt_emb)
    onnx_norm = onnx_emb / np.linalg.norm(onnx_emb)
    cosine = float(np.dot(trt_norm, onnx_norm))
    print(f"ONNX/TRT recognizer cosine: {cosine:.6f}")
    assert cosine > 0.999, f"recognizer cosine {cosine} below parity gate"


def test_recognizer_real_batch_parity(pipeline):
    image = load_image("/app/artifacts/samples/t1.jpg")
    faces = pipeline.extract_from_path("/app/artifacts/samples/t1.jpg")
    assert len(faces) >= 2

    aligned_a = align_face(image, faces[0].landmarks)
    aligned_b = align_face(image, faces[1].landmarks)

    single_a = pipeline.embed_aligned(aligned_a)
    single_b = pipeline.embed_aligned(aligned_b)
    batch = pipeline.embed_batch([aligned_a, aligned_b])

    np.testing.assert_allclose(single_a, batch[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(single_b, batch[1], rtol=1e-5, atol=1e-5)
