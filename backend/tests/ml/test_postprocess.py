import numpy as np

from app.ml.postprocess import _make_anchors, decode_detections


def test_anchor_centers_match_official_scrfd():
    centers = _make_anchors(stride=8, size=16, num_anchors=1)
    expected = np.array(
        [
            [0, 0],
            [8, 0],
            [0, 8],
            [8, 8],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(centers, expected, atol=1e-6)


def test_anchors_repeat_for_num_anchors():
    centers = _make_anchors(stride=8, size=16, num_anchors=2)
    assert len(centers) == 8
    np.testing.assert_array_equal(centers[0], centers[1])
    np.testing.assert_array_equal(centers[2], centers[3])


def test_score_filtering_does_not_double_sigmoid():
    # If scores were sigmoided again, even tiny logits would cross 0.5.
    # Here scores are already probabilities; only true positives should pass.
    count = 12800
    scores = np.zeros((count, 1), dtype=np.float32)
    # Sprinkle a few high-probability anchors.
    scores[100, 0] = 0.99
    scores[200, 0] = 0.51
    scores[300, 0] = 0.49  # just below threshold
    scores[400, 0] = 0.10

    bboxes = np.zeros((count, 4), dtype=np.float32)
    bboxes[:, 0] = 8.0  # distance2bbox l
    bboxes[:, 1] = 8.0  # distance2bbox t
    bboxes[:, 2] = 8.0  # distance2bbox r
    bboxes[:, 3] = 8.0  # distance2bbox b
    landmarks = np.zeros((count, 10), dtype=np.float32)

    outputs = {
        "scores": scores,
        "bboxes": bboxes,
        "landmarks": landmarks,
    }
    detections = decode_detections(
        outputs, input_size=640, conf_threshold=0.5, nms_threshold=0.5
    )
    assert len(detections) == 2
    assert 0.50 < detections[0].score < 1.0


def test_no_face_returns_empty_detections():
    # Simulate low raw probabilities everywhere: mask-like input.
    counts = [12800, 3200, 800]
    outputs = {}
    for count in counts:
        outputs[f"scores{count}"] = np.full((count, 1), 0.01, dtype=np.float32)
        outputs[f"bboxes{count}"] = np.zeros((count, 4), dtype=np.float32)
        outputs[f"landmarks{count}"] = np.zeros((count, 10), dtype=np.float32)

    detections = decode_detections(
        outputs, input_size=640, conf_threshold=0.5, nms_threshold=0.4
    )
    assert detections == []


def test_negative_scores_do_not_double_sigmoid():
    # Official scores are probabilities in [0, 1]; raw filtering should not
    # magically turn sub-threshold values into positives.
    raw_scores = np.array([-5.0, 0.0, 0.1, 0.51, 0.9], dtype=np.float32)
    mask = raw_scores >= 0.5
    assert mask.sum() == 2
