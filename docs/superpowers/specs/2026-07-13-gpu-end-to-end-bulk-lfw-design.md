# GPU End-to-End Pipeline + Bulk Enrollment + LFW Benchmark Design

## REFERENCE_CHECK

Task: Complete Gate 4/5 GPU data plane, expose `GpuFacePipeline.extract_bytes()`, build bulk enrollment tool, LFW benchmark harness, and model-swap workflow.
Phase: Phase 1 photo-based recognition / WP2-WP3 residual + full single-GPU data plane.
Allowed scope:
- `backend/app/ml/gpu/*`
- `backend/app/ml/pipeline.py` (add `GpuFacePipeline`, keep `FacePipeline` CPU oracle)
- `backend/native/mergenvision_gpu/*` (L2 wrapper, small CUDA fixes if needed)
- `backend/tests/ml/gpu/*`
- `backend/tools/bulk_enroll.py`, `backend/tools/lfw_benchmark.py`
- `backend/scripts/build_engines.py` (add optional batched-detector profile only as separate file).
Files forbidden to change:
- Existing database schema/migrations.
- Existing FastAPI routes beyond adding any required CLI entrypoints (routes are out of scope).
- Engine files under `/engines/*` except via explicit build script invocation.
- `AGENTS.md`, `CLAUDE.md`, architecture ADRs.
Local docs checked:
- `requirements/phase1requirements.md`
- `backend/docs/implementation/CURRENT_SPRINT.md`
- `AGENTS.md`
Architecture docs checked:
- `backend/app/ml/gpu/*.py` (decoder, preprocess, trt_device_engine, scrfd_postprocess, alignment)
- `backend/native/mergenvision_gpu/src/*`
- `backend/app/ml/pipeline.py`
- `backend/scripts/build_engines.py`
- `artifacts/model_manifest.json`, `artifacts/engine_metadata.json`
Requirements checked:
- Phase 1 requires photo-based identity enrollment and recognition, 10M-person scale roadmap, performance/scalability.
Official docs checked via context7:
- TensorRT Python API direct device binding and execution context shapes (during previous work).
- `cuda.bindings` stream/memory management (during previous work).
Open-source references checked via exa/web:
- `facebookresearch/segment-anything`: model registry, predictor embedding cache, modular encoder/decoder.
- `ultralytics/ultralytics`: AutoBackend multi-format inference, preprocess/infer/postprocess split, export engine, warmup.
- `roboflow/supervision`: `sv.Detections` adapter pattern, model-agnostic downstream tools.
- `PaddlePaddle/Paddle`: `AnalysisPredictor`/Paddle Inference, backend plugins, static graph deploy.
- `AarambhDevHub/multi-cam-face-tracker`: CPU/GPU split baseline (anti-pattern).
Existing local code inspected:
- `backend/app/ml/gpu/decoder.py`, `preprocess.py`, `trt_device_engine.py`, `scrfd_postprocess.py`, `alignment.py`
- `backend/app/ml/pipeline.py` (CPU oracle)
- `backend/native/mergenvision_gpu/src/l2_normalize.cu`, `bindings.cpp`
Old lessons checked:
- `CURRENT_SPRINT.md` Gate 3 completed. Gate 4 alignment synthetic test rewrite pending due to nvimgcodec pytest teardown crash.
Patterns to follow:
- Keep decode/preprocess/inference/postprocess/alignment/L2 all device-resident.
- Reuse `BufferArena` per stage; explicit stream; no implicit D2H.
- Wrap native CUDA ops behind `DeviceTensor` + small Python helpers.
- Use model manifest + build script for engine swaps.
- Verification-first: parity test for each stage, then end-to-end, then benchmark.
Patterns rejected:
- CPU fallback in production path.
- PyTorch/ONNX Runtime in hot path.
- True dynamic-batch detector engine rebuild without explicit approval (current SCRFD ONNX has fixed batch=1).
Architecture decisions that apply:
- Model adapter boundary rule: Detector/Aligner/Recognizer are separate adapters; `FacePipeline` orchestrates ML only.
- Docker/GPU strategy: single `api` service with NVIDIA runtime; engines mounted read-only.
- Data ownership: Postgres owns metadata, Qdrant owns vectors, MinIO owns images.
Docker/GPU strategy that applies:
- Use existing `docker-compose.yml` GPU runtime; no host driver changes.
- Native extension already built in Docker image.
Data ownership rules that apply:
- Embeddings in Qdrant; original image bytes in MinIO; person/photo/sample metadata in Postgres.
Security/PII rules that apply:
- No raw national ID in vector payload or audit log.
Tests/verification planned:
- Synthetic GPU alignment parity test passes in pytest.
- GPU L2 normalize parity vs NumPy.
- `GpuFacePipeline.extract_bytes()` end-to-end parity vs CPU `FacePipeline.extract()` on sample image.
- Bulk enrollment dry-run on small folder.
- LFW verification accuracy + throughput benchmark.
Unverified assumptions:
- Recognizer engine dynamic batch works up to 64 with direct device input.
- `l2_normalize` CUDA kernel parity tolerance acceptable for search thresholds.
- LFW dataset can be obtained manually or via SciKit-Learn fetch_lfw_pairs without violating "no auto-download" test policy.
Approval gates:
- User approval of this design before implementation begins.
Out-of-scope requests detected:
- Full REST API for people/identify (Phase 1 endpoints exist in AGENTS but not implemented yet; left for later).
- RBAC/KMS/multitenancy.
- Phase 2 video pipeline.

## Goal

Deliver the fastest, most accurate single-GPU Phase 1 face enrollment/recognition pipeline we can build with the current stack, then validate it on LFW. All heavy work (JPEG decode, resize, detection, postprocess, alignment, recognition, L2) stays on GPU until the final CPU boundary.

## Current stack lock (do not change)

- `nvidia-nvimgcodec-cu12==0.8.0.22`
- `cvcuda-cu12==0.16.0`
- TensorRT direct device binding
- Custom `mergenvision_gpu` C++/CUDA extension
- No PyTorch/torchvision/CuPy/DALI/Paddle/DeepFace in production hot path.

## Decomposition

This request bundles several subsystems. We split them into ordered milestones so each can be verified before the next.

| # | Milestone | Unlocks | Risk |
|---|---|---|---|
| 1 | Gate 4/5 GPU data plane complete | Everything else | nvimgcodec pytest teardown |
| 2 | `GpuFacePipeline.extract_bytes(bytes) -> list[FaceExtraction]` | Bulk + benchmark | TensorRT shape binding |
| 3 | Bulk enrollment script + Qdrant batch insert | Scale test | IO / DB transaction |
| 4 | LFW benchmark (accuracy + throughput) | Accuracy claim | Dataset availability |
| 5 | Model-swap workflow via `build_engines.py --force` | Experiments | Engine rebuild time |

## Recommended approach

**Approach A — finish the GPU data plane first, then build on top of it.**

- Gate 4: wrap native `l2_normalize` and implement batched recognizer inference with device-resident NCHW chips.
- Gate 5: connect decoder → preprocess → detector → SCRFD postprocess → alignment → recognizer → L2 into `GpuFacePipeline`.
- For throughput: do **not** try to batch images through the detector because the SCRFD ONNX has a fixed `N=1` input. Instead process images sequentially but keep every post-detection step batched (all faces in one image go through alignment/recognizer together; across images we can prefetch the next JPEG while the current runs).
- This keeps all NN/data operations on GPU and avoids risky engine rebuilds.

**Rejected alternatives**

- Approach B: rebuild detector engine with dynamic batch. Rejected because the SCRFD ONNX declares `[1,3,?,?]`; changing the batch dimension may not parse cleanly and would delay everything.
- Approach C: CPU detector with GPU recognizer. Rejected because it violates the "IO dahil her şey GPU'da" goal.

## Design details

### 1. Native L2 wrapper

Add `backend/app/ml/gpu/l2_normalize.py`:

```
l2_normalize_device(input: DeviceTensor [N, D], *, stream) -> DeviceTensor [N, D]
```

- Uses `mergenvision_gpu.l2_normalize`.
- Allocates output via `BufferArena`.
- Reads small status buffer for non-finite / zero-norm errors.

### 2. Gate 4 — GPU recognizer

Add `backend/app/ml/gpu/recognition.py`:

```
class GpuRecognizer:
    def __init__(self, engine_path, device_id=0)
    def infer(self, aligned_nchw: DeviceTensor [N,3,112,112], *, stream) -> DeviceTensor [N,512]
```

- Calls `TrtDeviceEngine.infer_device()` directly on the aligned chip tensor.
- Chunks `N > 64` using the engine profile max batch.
- Applies `l2_normalize_device` to the concatenated result.
- No H2D/D2H inside.

### 3. Gate 5 — `GpuFacePipeline`

Add to `backend/app/ml/pipeline.py`:

```
@dataclass
class FaceExtraction:
    bbox: np.ndarray          # [4] CPU result
    landmarks: np.ndarray     # [5,2] CPU result
    embedding: np.ndarray     # [D] CPU result, already L2 normalized

class GpuFacePipeline:
    def __init__(self, cfg: Settings = settings)
    def warmup(self)
    def extract_bytes(self, image_bytes: bytes) -> list[FaceExtraction]
    def close(self)
```

Data flow:

```
bytes -> JpegGpuDecoder.decode               -> [1,H,W,3] uint8 device
      -> GpuDetectorPreprocessor.preprocess   -> [1,3,640,640] float device
      -> TrtDeviceEngine (detector) infer_device
      -> ScrfdGpuPostprocess.decode            -> boxes/scores/landmarks device
      -> filter keep mask (device), copy kept faces
      -> GpuFaceAligner.align                  -> [N,3,112,112] float device
      -> GpuRecognizer.infer                   -> [N,512] float device
      -> l2_normalize_device                   -> [N,512] float device
      -> D2H of bboxes, landmarks, embeddings  -> CPU boundary
```

All components share the pipeline's CUDA stream where possible; each stage owns its arena.

### 4. Bulk enrollment

New `backend/tools/bulk_enroll.py`:

- Input: folder of images or CSV mapping `person_name -> image_path`.
- For each image:
  - Read bytes from disk (or MinIO).
  - Run `GpuFacePipeline.extract_bytes()`.
  - For each detected face create `person`, `person_photo`, `face_sample` records (UUIDv7) and upload original/crop to MinIO.
  - Accumulate Qdrant points and upsert in batches of 256/512.
- Output: enrollment report (image count, face count, misses, timing).

### 5. LFW benchmark

New `backend/tools/lfw_benchmark.py`:

- Use `sklearn.datasets.fetch_lfw_pairs` (download once, cache under `artifacts/lfw/`).
- Run all 13,233 images through `GpuFacePipeline.extract_bytes()` to build an in-memory embedding cache or a temporary Qdrant collection.
- For the 6,000 pairs compute cosine similarity.
- Sweep threshold to report:
  - Best accuracy
  - TAR@FAR=0.01, 0.001
  - Mean/STD similarity for same vs different
  - Total enrollment time and images/sec
  - Per-image pipeline latency (p50/p95/p99)

### 6. Model swap workflow

- Models are recorded in `artifacts/model_manifest.json`.
- Engines are built via `python -m scripts.build_engines --model-pack <name> --force`.
- `GpuFacePipeline` reads paths from `Settings`; swapping a model means rebuilding engines, updating env/config, and restarting the service. No code change.
- To test different ArcFace/SCRFD variants we add new entries to `model_manifest.json` and re-run the build script.

## Error handling

- Any stage that cannot run on GPU raises explicitly; no silent CPU fallback.
- Empty detection returns `[]` without error.
- Non-finite embeddings are rejected before Qdrant insert.

## Testing / verification

| Milestone | Verification command |
|---|---|
| Gate 4 L2 | `pytest tests/ml/gpu/test_l2_normalize.py -q -s` |
| Gate 4 recognizer | `pytest tests/ml/gpu/test_gpu_recognizer.py -q -s` |
| Gate 5 pipeline | `pytest tests/ml/gpu/test_gpu_pipeline.py -q -s` |
| Bulk dry-run | `python -m tools.bulk_enroll --dry-run --folder /app/artifacts/tiny_face_set` |
| LFW benchmark | `python -m tools.lfw_benchmark --pairs lfw_test` |

## Next step

After user approves this design, invoke `writing-plans` to produce the step-by-step implementation plan and begin execution.
