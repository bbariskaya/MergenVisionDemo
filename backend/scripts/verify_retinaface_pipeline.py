"""Verify the GPU RetinaFace pipeline against an OpenCV + ONNX Runtime CPU oracle.

Usage inside the API container:
    MODEL_PACK=retinaface_r50 python scripts/verify_retinaface_pipeline.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ml.gpu.face_pipeline import GpuFacePipeline  # noqa: E402
from app.ml.gpu.retinaface_postprocess import (  # noqa: E402
    RetinaFacePostprocess,
    _build_priors,
    _nms_cpu,
)

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/app/artifacts"))
ONNX_PATH = ARTIFACTS / "models" / "retinaface_r50_dynamic.onnx"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_cpu_oracle(image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2.imread failed: {image_path}")
    h, w = img.shape[:2]

    blob = cv2.resize(img, (640, 640)).astype(np.float32)
    blob -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    loc, conf, landms = session.run(None, {"input": blob})

    pp = RetinaFacePostprocess(input_size=640, device_id=0)
    boxes, scores, landmarks = pp._decode_single(
        loc[0].astype(np.float32),
        conf[0].astype(np.float32),
        landms[0].astype(np.float32),
    )
    keep = _nms_cpu(boxes, scores, 0.4, top_k=750)
    boxes = boxes[keep]
    scores = scores[keep]
    landmarks = landmarks[keep]

    mask = scores >= 0.5
    boxes = boxes[mask]
    scores = scores[mask]
    landmarks = landmarks[mask]

    boxes[:, [0, 2]] *= float(w)
    boxes[:, [1, 3]] *= float(h)
    for i in range(5):
        landmarks[:, i * 2] *= float(w)
        landmarks[:, i * 2 + 1] *= float(h)
    return boxes, scores, landmarks


def main() -> int:
    pipe = GpuFacePipeline(device_id=0)
    pipe.warmup()

    root = Path("/app/lfw/lfw-deepfunneled/lfw-deepfunneled")
    images: list[Path] = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            images.extend(sorted(d.glob("*.jpg"))[:1])
        if len(images) >= 20:
            break

    pair_ok = 0
    count_ok = 0
    for img in images:
        gpu_faces = pipe.extract_bytes(img.read_bytes())
        cpu_boxes, cpu_scores, cpu_landms = run_cpu_oracle(img)

        if len(gpu_faces) == cpu_boxes.shape[0]:
            count_ok += 1
        else:
            print(f"COUNT MISMATCH {img.name}: gpu={len(gpu_faces)} cpu={cpu_boxes.shape[0]}")
            continue

        if len(gpu_faces) == 0:
            pair_ok += 1
            continue

        # Match each GPU face to the most overlapping CPU detection.
        ok = True
        for gf in gpu_faces:
            ious = np.array([iou(gf.bbox, cb) for cb in cpu_boxes])
            best = int(np.argmax(ious))
            if ious[best] < 0.7:
                ok = False
                print(f"MISMATCH {img.name}: best IoU for a GPU face = {ious[best]:.3f}")
                break
            ld = np.linalg.norm(
                gf.landmarks.reshape(5, 2) - cpu_landms[best].reshape(5, 2), axis=1
            ).max()
            if ld > 10.0:
                ok = False
                print(
                    f"MISMATCH {img.name}: IoU={ious[best]:.3f} max_landmark_diff={ld:.1f}"
                )
                break
        if ok:
            pair_ok += 1

    print(f"checked {len(images)} images")
    print(f"count matched: {count_ok}/{len(images)}")
    print(f"all faces matched by overlap: {pair_ok}/{len(images)}")
    return 0 if count_ok == len(images) and pair_ok >= len(images) - 1 else 1


if __name__ == "__main__":
    sys.exit(main())
