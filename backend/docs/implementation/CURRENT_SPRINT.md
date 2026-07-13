# WP3 ML Correctness Recovery Sprint

## Objective
Fix correctness, parity, and pathological runtime problems in the TensorRT-based face detection/recognition pipeline before any WP4/API/UI work.

## Acceptance
- Preprocess outputs are `float32`, NCHW, and C-contiguous.
- TensorRT raw input logical layout matches the ONNX reference oracle.
- SCRFD scores are not double-sigmoided.
- Anchor centers and bbox/landmark decode match the official InsightFace SCRFD behavior.
- Alignment is deterministic and uses the canonical five-point ArcFace template without RANSAC.
- `mask_blue.jpg` returns zero faces.
- `t1.jpg` (Friends group image) returns multiple faces.
- A valid same-person pair scores higher than a valid different-person pair.
- Detector and recognizer TensorRT outputs are within FP16 tolerance of the ONNX Runtime reference oracle.
- Recognizer real batch `[A, B]` parity passes.
- Engines are not rebuilt or deserialized per test; pipeline is created once per session.
- All existing backend tests still pass.
- Timing breakdown is recorded; five ML tests must not take minutes.

## Non-goals
- No WP4 file creation.
- No production ONNX/CPU fallback.
- No detector batch capability claims (detector is fixed batch-1).
- No arbitrary threshold lowering to force PASS.
- No skip/xfail on ML correctness tests.

## Reference decisions
- SCRFD inference follows the official InsightFace `scrfd.py` implementation in `deepinsight/insightface`.
- Anchor center is `grid * stride` (no extra 0.5 offset).
- Score outputs are read as raw probabilities; no extra sigmoid.
- Detector preprocess: RGB, NCHW, mean=127.5, std=128.0.
- Recognizer preprocess: RGB, NCHW, mean=127.5, std=127.5 (ArcFace glintr100 contract).
- Alignment uses the standard ArcFace 112x112 template and a deterministic similarity transform.
- ONNX Runtime is used only as a reference oracle in tests, not in production.
