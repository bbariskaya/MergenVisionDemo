# Reference Decision Log

## WP3A — SCRFD score contract
- Source: deepinsight/insightface `python-package/insightface/model_zoo/scrfd.py`
- Decision: Model output scores are already probabilities; do not apply a second sigmoid.
- Rationale: Double sigmoid moved ~16 800 anchors above threshold and caused pathological Python NMS runtime.

## WP3A — SCRFD anchor center
- Source: deepinsight/insightface `python-package/insightface/model_zoo/scrfd.py`
- Decision: Anchor center is `grid * stride`, not `(grid + 0.5) * stride`.
- Rationale: Matches official reference decoder output and reduces landmark drift.

## WP3A — Preprocess normalization
- Detector: mean 127.5, std 128.0 (SCRFD contract).
- Recognizer: mean 127.5, std 127.5 (ArcFace glintr100 contract).
- Rationale: ONNX reference oracle parity requires exact preprocessing.

## WP3A — Alignment
- Source: deepinsight/insightface `python-package/insightface/utils/face_align.py`
- Decision: Deterministic similarity transform via least-squares, no RANSAC.
- Rationale: Five landmarks are sufficient; RANSAC introduces non-determinism.

## WP3B — GPU library choice
- Date: observed during this sprint.
- Options evaluated:
  - `pynvjpeg` (source build failed).
  - `torch`/`torchvision` (very large dependency, not installed).
  - `nvidia-nvimgcodec-cu12` 0.8.0.22 (installs cleanly, device decode, official).
  - `cvcuda-cu12` 0.16.0 (installs cleanly, device resize/normalize/warp, official).
- Decision: Use `nvidia-nvimgcodec-cu12` + `cvcuda-cu12`.
- Caveat: nvImageCodec emits optional warnings about missing nvJPEG2000/nvTIFF extensions; JPEG backend still registers and works.
