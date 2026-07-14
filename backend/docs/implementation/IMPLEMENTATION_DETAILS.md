# Implementation Details

## WP2 Runtime / Data Flow

1. `lifespan.py` configures the async SQLAlchemy engine, initializes MinIO bucket, and validates Qdrant collection schema/indexes.
2. Partial startup failures are cleaned up in `try/finally`: if any component fails, the Qdrant client and DB engine/sessionmaker are closed/disposed.
3. `ReadinessService` gathers Postgres (`SELECT 1` + phase-1 table check via DB test), MinIO bucket existence, and Qdrant schema health in parallel with a 5 s timeout per component.
4. MinIO and Qdrant `health_check()` now return booleans reflecting actual state; raw exceptions are sanitized.
5. Qdrant payload validation enforces the exact allowlist (`sampleId`, `photoId`, `personId`, `active`, `modelVersion`), point-id/sample-id equality, valid UUIDs, vector dimensions, finite values, and non-zero vectors. All write calls use `wait=True`.
6. DB `dispose_engine()` nulls both engine and sessionmaker; `configure_engine()` is idempotent and can re-create them.

## WP3A Reference Runtime / Data Flow

1. `FacePipeline` (CPU reference) loads detector and recognizer TensorRT engines once per session.
2. `load_image` decodes JPEG with PIL, resizes/pads with PIL, and returns a NumPy RGB array.
3. `preprocess_detector` produces float32 NCHW C-contiguous tensor with mean 127.5 / std 128.0.
4. `TrtEngine.infer` H2D-copies the input, executes TensorRT asynchronously, synchronizes, and D2H-copies outputs. The engine holds a per-instance lock and validates exact input/output tensor names.
5. `decode_detections` consumes SCRFD score/bbox/landmark outputs on CPU: raw probabilities (no double sigmoid), `grid*stride` anchor centers, distance2bbox/distance2kps, threshold filtering, and NMS.
6. `align_face` maps the five detected landmarks to the canonical 112×112 ArcFace template using a deterministic similarity transform.
7. `extract` detects once, aligns all faces, and runs a single `embed_batch` recognizer inference (batch ≤ 64 engine profile, chunked if larger).
8. `embed_batch` L2-normalizes outputs on the CPU; zero/non-finite embeddings raise explicit errors.
9. ONNX Runtime is used only as an oracle in `tests/ml/test_trt_parity.py`.

## WP3B GPU Runtime (Selected but Not Fully Implemented)

Target production flow intended:

```
JPEG bytes (host)
  → nvImageCodec.Decoder.decode() GPU RGB image
  → CV-CUDA resize/pad/normalize (device)
  → TensorRT detector direct device input
  → detector outputs device memory
  → GPU decode/NMS mask/select
  → GPU five-point batch alignment
  → TensorRT ArcFace batched
  → GPU L2 normalize
  → compact bbox/landmarks/scores/embeddings D2H
```

Implemented evidence:
- `nvidia-nvimgcodec-cu12==0.8.0.22` and `cvcuda-cu12==0.16.0` install on RTX 8000 / CUDA 12.4.
- `tests/ml/test_gpu_decode_smoke.py` proves JPEG decode to a GPU-backed image with a non-zero device pointer (`__cuda_array_interface__`).

Not implemented in this sprint:
- CV-CUDA resize/pad (requires `nvcv` tensor wrapping and layout handling exploration beyond this session).
- GPU SCRFD postprocess/NMS (no NMS primitive in cvcuda; would require PyTorch/CuPy custom CUDA or a managed dependency).
- GPU five-point alignment (requires affine transform + bilinear sampling kernel or torch `grid_sample`).
- Device pointer binding to TensorRT for detector/recognizer inputs/outputs.

For that reason WP3B is reported as `BLOCKED` with decoder evidence, not PASS.

## Frontend UI/UX Sprint

### Stack
- React 19.2.7, React Router 7.5.3 (library declarative mode), TanStack Query 5.84.1, Vite 6.0.0, TypeScript 5.x, Tailwind CSS 3.4.x, Lucide React.
- Vitest + React Testing Library for unit/component tests.
- Playwright for E2E smoke tests against the real backend.

### API Mapping
- `GET /health/live`, `GET /health/ready` → health indicators.
- `POST /api/v1/faces/enroll` → EnrollPage.
- `POST /api/v1/faces/recognize` → IdentifyPage with top-K/threshold controls.
- `GET /api/v1/faces?search&is_active&limit&offset` → FaceSearchPage (added mid-sprint by backend agent).
- `GET|DELETE /api/v1/faces/{face_id}` → FaceDetailPage (detail + history + delete).
- `GET /api/v1/processes/{process_id}` → ProcessDetailPage.

### Security / UX Rules Applied
- Raw `national_id` never rendered; only `nationalIdMasked` shown.
- Internal object keys, embeddings, and stack traces never exposed in UI or console.
- Turkish UI copy; API status enums mapped to Turkish labels.
- Lucide icons only; no emoji structural icons.
- Responsive sidebar (desktop collapse, mobile drawer), breadcrumbs, health indicator.
- Loading/error/empty/no-face/unknown/multi-face states implemented.
- Bounding-box overlays scaled relative to image natural/display dimensions.

### Validation Evidence
- `npm run typecheck` pass.
- `npm run lint` pass.
- `npm run build` pass.
- `npm run test` pass (9 unit/component tests).
- `npx playwright test --project=chromium` pass (8 E2E tests against real backend).
- Nginx config syntax test pass.
- Desktop + mobile viewport screenshots captured.
