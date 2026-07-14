from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    landmarks: np.ndarray  # [5, 2]


# SCRFD config used by AntelopeV2 detector: 2 anchors per FPN level.
# Strides are inferred from tensor counts; anchors per level = count / 2.


def _make_anchors(stride: int, size: int, num_anchors: int = 2) -> np.ndarray:
    grid = size // stride
    anchors = []
    for y in range(grid):
        for x in range(grid):
            # Official SCRFD uses top-left grid corners; no +0.5 offset.
            cx = x * stride
            cy = y * stride
            for _ in range(num_anchors):
                anchors.append([cx, cy])
    return np.array(anchors, dtype=np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # Inclusive-coordinate area/intersection to match official InsightFace SCRFD.
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = order[1:][iou <= threshold]
    return keep


def decode_detections(
    outputs: dict[str, np.ndarray],
    input_size: int,
    conf_threshold: float,
    nms_threshold: float,
) -> list[Detection]:
    # Group outputs by semantic type using second-dimension shape, then sort by
    # descending anchor count (which maps to smaller strides first).
    scores_map: dict[int, np.ndarray] = {}
    bboxes_map: dict[int, np.ndarray] = {}
    landmarks_map: dict[int, np.ndarray] = {}
    for name, arr in outputs.items():
        if arr.ndim != 2:
            continue
        count, dim2 = arr.shape
        if dim2 == 1:
            scores_map[count] = arr.reshape(-1)
        elif dim2 == 4:
            bboxes_map[count] = arr
        elif dim2 == 10:
            landmarks_map[count] = arr
    counts = sorted(scores_map.keys(), reverse=True)
    # anchors_per_level = count / num_anchors (2); grid = sqrt(anchors_per_level)
    strides = [
        input_size // int(np.sqrt(count // 2)) for count in counts
    ]

    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_landmarks: list[np.ndarray] = []

    for count, stride in zip(counts, strides):
        score = scores_map[count]
        bbox = bboxes_map[count]
        kps = landmarks_map[count]
        anchors = _make_anchors(stride, input_size)

        cx = anchors[:, 0]
        cy = anchors[:, 1]
        # Decode distance predictions (l,t,r,b) to box corners.
        x1 = cx - bbox[:, 0] * stride
        y1 = cy - bbox[:, 1] * stride
        x2 = cx + bbox[:, 2] * stride
        y2 = cy + bbox[:, 3] * stride
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        landmarks = np.zeros((len(anchors), 5, 2), dtype=np.float32)
        for k in range(5):
            landmarks[:, k, 0] = cx + kps[:, k * 2] * stride
            landmarks[:, k, 1] = cy + kps[:, k * 2 + 1] * stride

        mask = score >= conf_threshold
        if not mask.any():
            continue
        all_boxes.append(boxes[mask])
        all_scores.append(score[mask])
        all_landmarks.append(landmarks[mask])

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    landmarks = np.concatenate(all_landmarks)

    keep = _nms(boxes, scores, nms_threshold)
    detections = [
        Detection(bbox=boxes[i], score=float(scores[i]), landmarks=landmarks[i])
        for i in keep
    ]
    return detections
