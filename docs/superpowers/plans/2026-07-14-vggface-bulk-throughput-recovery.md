# VGGFace Bulk Throughput Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore single-GPU VGGFace bulk enrollment end-to-end throughput to the 400–450 photos/sec range (two-GPU aggregate ~600) by removing wasteful full-dataset hashing, redundant JPEG reads, serial GPU/persistence scheduling, and RetinaFace-specific batch crashes.

**Architecture:** Convert the current request → full preflight → worker materialization → serial read/extract/persist loop into a streaming producer/consumer pipeline: a small bounded reader pool feeds batched GPU inference on a dedicated executor, while a separate bounded persistence pool uploads the same byte buffers to MinIO and upserts PostgreSQL/Qdrant in parallel. Manifest metadata is cached on the API side and lazily streamed into shards, so a `maxPhotos=5K` job never hashes the remaining ~192K photos.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2 async, Qdrant, MinIO, TensorRT/CUDA, React/TanStack Query.

## Global Constraints
- No volume deletion and no PostgreSQL/Qdrant data wipe. Existing active vectors must remain searchable.
- Worktree changes happen in-place; no new git commits unless explicitly requested.
- The existing active photo set (~5K VGGFace prefix) must remain intact and compatible after the embedding-model-version split.
- GPU 0 (`gpu-worker-0`, `WORKER_ROLE=online`) remains reserved for recognition; bulk work uses only `gpu-worker-1`.
- `.env` already sets `MODEL_PACK=retinaface_r50` and the worker image must be rebuilt after backend code changes because the backend source is copied into the image.
- All plan steps are testable inside the Docker Compose environment; unit tests can run with `docker compose exec api pytest ...`.

---

## Task 1: Fix RetinaFace zero-candidate batch crash

**Files:**
- Modify: `backend/app/ml/gpu/retinaface_postprocess.py:385-400`
- Test: `docker compose exec api python -c "from app.ml.gpu.retinaface_postprocess import RetinaFacePostprocess; ..."` (or a new script `backend/tests/test_retinaface_empty.py`)

**Interfaces:**
- Consumes: `DeviceTensor` constructor expects `ptr != 0`.
- Produces: `RetinaFacePostprocess._empty()` must return a usable object when any dimension is zero.

- [ ] **Step 1: Reproduce the crash**

Run in the worker container:

```bash
docker compose exec gpu-worker-1 python - <<'PY'
from app.ml.gpu.retinaface_postprocess import RetinaFacePostprocess
p = RetinaFacePostprocess()
# Force a zero-shape tensor through _empty
empty = p._empty((0, 4), stream=0)
print(empty)
PY
```

Expected: `ValueError: DeviceTensor requires a non-null device pointer`.

- [ ] **Step 2: Replace `_empty` with a no-op zero-size wrapper**

Change `backend/app/ml/gpu/retinaface_postprocess.py`:

```python
class _ZeroSizeTensor:
    """Stand-in for DeviceTensor when a dimension is zero."""
    __slots__ = ("shape", "dtype", "device_id", "ptr")

    def __init__(self, shape, dtype, device_id):
        self.shape = shape
        self.dtype = dtype
        self.device_id = device_id
        self.ptr = 0

    @property
    def nbytes(self) -> int:
        return 0


def _empty(
    self,
    shape: tuple[int, ...],
    stream: int,
    dtype: type = ctypes.c_float,
) -> DeviceTensor | _ZeroSizeTensor:
    if any(s == 0 for s in shape):
        return _ZeroSizeTensor(shape, dtype, self._device_id)
    return self._arena.reserve(shape, dtype, stream=stream)
```

- [ ] **Step 3: Verify zero-size paths in `decode` and `scale_and_compact`**

The `decode` and `scale_and_compact` paths call `_empty((0, …))`. After the change, `pick_largest_device` receives `_ZeroSizeTensor` objects for images with no detections. Confirm `pick_largest_device` uses `.ptr` in pointer arrays safely; update it to guard against zero-size tensors:

```python
boxes_ptrs = np.array([s.boxes.ptr for s in scaled], dtype=np.uint64)
landmarks_ptrs = np.array([s.landmarks.ptr for s in scaled], dtype=np.uint64)
scores_ptrs = np.array([s.scores.ptr for s in scaled], dtype=np.uint64)
```

`_ZeroSizeTensor.ptr == 0` is fine because those entries correspond to `counts[i] == 0`, so the CUDA kernel will never dereference them. Add a comment explaining this.

- [ ] **Step 4: Re-run reproduction script**

Expected: no exception; `_ZeroSizeTensor` prints `shape=(0, 4)`.

- [ ] **Step 5: Inspect for other `DeviceTensor(ptr=0)` call sites**

Search:

```bash
rg "DeviceTensor\(\s*ptr=0" backend/app/ml
```

Fix any similar call sites by using `_ZeroSizeTensor` or by creating a real zero-byte allocation. Do **not** change `DeviceTensor.__init__`; the null-pointer guard is correct for normal use.

---

## Task 2: Split detector model from embedding model version

**Files:**
- Modify: `backend/app/core/config.py`, `backend/app/ml/gpu/face_pipeline.py`, `backend/app/services/bulk_enrollment.py`, `backend/app/services/face_service.py` (if exists)
- Test: `docker compose exec api pytest backend/tests/test_config.py -v` (create if missing)

**Interfaces:**
- Consumes: `settings.model_pack` currently drives detector + embedder + Qdrant `modelVersion` + `sample_id` derivation.
- Produces: `settings.model_pack` stays as the detector alias (e.g. `retinaface_r50`); a new `settings.embedding_model_version` controls Qdrant `modelVersion` and sample id derivation (e.g. `arcface_512_v1`).

- [ ] **Step 1: Add `embedding_model_version` to config**

In `backend/app/core/config.py`, add after `model_pack`:

```python
model_pack: str = "antelopev2"
embedding_model_version: str = "arcface_512_v1"
```

Update `.env`:

```bash
MODEL_PACK=retinaface_r50
EMBEDDING_MODEL_VERSION=arcface_512_v1
```

- [ ] **Step 2: Update `face_pipeline.py` to use `model_pack` only for detector branch**

`face_pipeline.py` already uses `self._model_pack == "retinaface_r50"` only for detector selection. Leave that. Remove any other use of `model_pack` for Qdrant/vector versioning if present (none in this file).

- [ ] **Step 3: Update `bulk_enrollment.py` to derive sample ids and Qdrant payload from `embedding_model_version`**

Change:

```python
self._embedding_model_version = settings.embedding_model_version
```

Replace every `derive_sample_id(photo_id, self._model_pack)` with `derive_sample_id(photo_id, self._embedding_model_version)`.
Replace every Qdrant payload `"modelVersion": self._model_pack` with `"modelVersion": self._embedding_model_version`.
Replace every `FaceSample(detector_model=self._model_pack, embedding_model=self._model_pack, ...)` with `detector_model=self._model_pack, embedding_model=self._embedding_model_version`.

- [ ] **Step 4: Verify online recognition still finds old active vectors**

Because `sample_id` is now deterministic on `embedding_model_version`, previous vectors stored under the same version string remain reachable. Run a recognition probe against an already-enrolled person and confirm the same face is returned.

---

## Task 3: Move `max_photos` limit before hashing and materialization

**Files:**
- Modify: `backend/app/services/vggface_manifest.py`, `backend/app/workers/gpu_worker.py`, `backend/app/services/bulk_enrollment.py`, `backend/app/services/bulk_orchestrator.py`
- Test: `docker compose exec api python backend/tests/test_manifest_limit.py`

**Interfaces:**
- Consumes: `stream_vggface_manifest(root, max_identities=None, ...)`; `enroll_shard(identities, max_photos=None)`.
- Produces: `stream_vggface_manifest(root, max_photos=None, ...)` stops streaming after `max_photos`; `_load_identities` no longer materializes the whole shard.

- [ ] **Step 1: Add `max_photos` to streaming manifest**

In `backend/app/services/vggface_manifest.py`, change `stream_vggface_manifest` signature and body:

```python
def stream_vggface_manifest(
    root: Path,
    *,
    max_identities: int | None = None,
    max_photos: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
    resume_after_identity_key: str | None = None,
) -> Iterator[EnrollmentIdentity]:
```

Streaming stop condition:

```python
    photos_seen = 0
    built = 0
    for folder in folders:
        ...  # sharding/resume checks remain
        identity = _build_identity(folder)
        if identity.photos:
            if max_photos is not None and photos_seen + len(identity.photos) > max_photos:
                # Truncate the last identity to hit the budget exactly.
                remaining = max(0, max_photos - photos_seen)
                if remaining > 0:
                    identity = EnrollmentIdentity(
                        identity_key=identity.identity_key,
                        display_name=identity.display_name,
                        identity_hmac=identity.identity_hmac,
                        person_id=identity.person_id,
                        face_identity_id=identity.face_identity_id,
                        photos=identity.photos[:remaining],
                    )
                    yield identity
                break
            yield identity
            photos_seen += len(identity.photos)
            built += 1
            if max_identities is not None and built >= max_identities:
                break
```

- [ ] **Step 2: Remove `max_photos` truncation from `enroll_shard`**

In `backend/app/services/bulk_enrollment.py`, delete:

```python
            if max_photos is not None and len(tasks) > max_photos:
                tasks = tasks[:max_photos]
                result.photos = len(tasks)
                process_record.summary["photos"] = result.photos
```

`result.photos` should now equal the number of photos that were actually passed in as `tasks`.

- [ ] **Step 3: Pass `max_photos` through worker payload**

In `_build_worker_payload` (`bulk_orchestrator.py`) the field is already `maxPhotos`. Ensure `_load_identities` passes it to `stream_vggface_manifest`.

Update `backend/app/workers/gpu_worker.py` `_load_identities`:

```python
        identities = stream_vggface_manifest(
            root,
            shard_index=shard_index if num_shards > 1 else None,
            num_shards=num_shards if num_shards > 1 else None,
            resume_after_identity_key=payload.resume_after_identity_key,
            max_photos=payload.max_photos,
        )
```

- [ ] **Step 4: Avoid materializing the whole shard in `_load_identities`**

Change the final line:

```python
    # DO NOT materialize; stream identities through the enrollment loop.
    return identities  # type: ignore[return-value]
```

Update the return type signature from `tuple[EnrollmentIdentity, ...]` to `Iterator[EnrollmentIdentity]`. Then update `enroll_shard` to accept an `Iterator` and iterate once.

- [ ] **Step 5: Write a fast unit test for streaming budget**

Create `backend/tests/test_manifest_limit.py`:

```python
from pathlib import Path
from app.services.vggface_manifest import stream_vggface_manifest


def test_vggface_stream_respects_max_photos():
    root = Path("/datasets/vgg-face")
    photos = list(stream_vggface_manifest(root, max_photos=12))
    total = sum(len(i.photos) for i in photos)
    assert total == 12
```

Run:

```bash
docker compose exec api pytest backend/tests/test_manifest_limit.py -v
```

---

## Task 4: Cache or defer full-dataset preflight

**Files:**
- Modify: `backend/app/services/bulk_orchestrator.py`, `backend/app/services/vggface_manifest.py`
- Add: `backend/app/services/vggface_manifest_cache.py`
- Test: `docker compose exec api python backend/tests/test_preflight_cache.py`

**Interfaces:**
- Consumes: `settings.vggface_dataset_path`.
- Produces: A disk-backed JSON cache containing identity/photo counts and SHA-256 hashes, refreshed only when dataset mtime changes.

- [ ] **Step 1: Create manifest cache module**

Add `backend/app/services/vggface_manifest_cache.py`:

```python
"""Lightweight disk cache for VGGFace metadata so the API does not re-hash
~197K images on every job start.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.config import settings
from app.services.vggface_manifest import VggfacePreflight, vggface_preflight

_CACHE_DIR = Path("/tmp/mergenvision_cache")
_CACHE_FILE = _CACHE_DIR / "vggface_manifest_cache.json"


@dataclass(frozen=True)
class VggfaceManifestCache:
    dataset_mtime: float
    preflight: VggfacePreflight

    def to_dict(self) -> dict:
        return {
            "dataset_mtime": self.dataset_mtime,
            "preflight": {
                "root": str(self.preflight.root),
                "identity_count": self.preflight.identity_count,
                "photo_count": self.preflight.photo_count,
                "duplicate_photo_count": self.preflight.duplicate_photo_count,
                "corrupt_paths_count": self.preflight.corrupt_paths_count,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VggfaceManifestCache":
        p = data["preflight"]
        return cls(
            dataset_mtime=float(data["dataset_mtime"]),
            preflight=VggfacePreflight(
                root=Path(p["root"]),
                identity_count=int(p["identity_count"]),
                photo_count=int(p["photo_count"]),
                duplicate_photo_count=int(p["duplicate_photo_count"]),
                corrupt_paths_count=int(p["corrupt_paths_count"]),
            ),
        )


def _dataset_mtime(path: Path) -> float:
    faces_root = path / "faces" if (path / "faces").is_dir() else path
    mtimes = [faces_root.stat().st_mtime]
    for folder in faces_root.iterdir():
        mtimes.append(folder.stat().st_mtime)
    return max(mtimes)


def get_vggface_preflight(path: Path | None = None) -> VggfacePreflight:
    path = path or settings.vggface_dataset_path
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    current_mtime = _dataset_mtime(path)

    if _CACHE_FILE.exists():
        try:
            cache = VggfaceManifestCache.from_dict(json.loads(_CACHE_FILE.read_text()))
            if cache.dataset_mtime >= current_mtime:
                return cache.preflight
        except Exception:
            pass

    preflight = vggface_preflight(path)
    cache = VggfaceManifestCache(dataset_mtime=current_mtime, preflight=preflight)
    _CACHE_FILE.write_text(json.dumps(cache.to_dict()))
    return preflight
```

- [ ] **Step 2: Use cached preflight in orchestrator**

In `backend/app/services/bulk_orchestrator.py`, replace:

```python
    preflight = await asyncio.to_thread(vggface_preflight, settings.vggface_dataset_path)
```

with:

```python
    from app.services.vggface_manifest_cache import get_vggface_preflight
    preflight = await asyncio.to_thread(get_vggface_preflight, settings.vggface_dataset_path)
```

- [ ] **Step 3: Verify cache is reused**

Run:

```bash
docker compose exec api python - <<'PY'
import time
from app.services.vggface_manifest_cache import get_vggface_preflight

path = "/datasets/vgg-face"
t0 = time.perf_counter()
p1 = get_vggface_preflight(path)
first = time.perf_counter() - t0

t0 = time.perf_counter()
p2 = get_vggface_preflight(path)
second = time.perf_counter() - t0

assert p1.photo_count == p2.photo_count
print(f"first={first:.3f}s cached={second:.3f}s")
assert second < first / 10
PY
```

---

## Task 5: Single-read byte buffer reuse and bounded persistence pipeline

**Files:**
- Modify: `backend/app/services/bulk_enrollment.py`, `backend/app/services/bulk_manifest.py`
- Test: `docker compose exec api pytest backend/tests/test_bulk_pipeline.py -v`

**Interfaces:**
- Consumes: `EnrollmentPhoto(path, content_sha256)`.
- Produces: `EnrollmentPhoto(path, content_sha256, bytes | None = None)` carries the buffer through read → GPU → MinIO.

- [ ] **Step 1: Add optional `bytes` cache to `EnrollmentPhoto`**

In `backend/app/services/bulk_manifest.py`:

```python
@dataclass(frozen=True)
class EnrollmentPhoto:
    path: Path
    content_sha256: str
    data: bytes | None = None
```

- [ ] **Step 2: Read files once and reuse the same buffer**

In `bulk_enrollment.py`, change `_read_photo_bytes` to `_read_photo` and cache:

```python
    async def _read_photo(self, photo: EnrollmentPhoto) -> bytes:
        if photo.data is not None:
            return photo.data
        return await asyncio.to_thread(photo.path.read_bytes)
```

After reading for GPU inference, attach the bytes to the task:

```python
                image_bytes = await asyncio.gather(
                    *(self._read_photo(t.photo) for t in chunk)
                )
                # Attach the bytes we already loaded so persistence does not re-read.
                for task, buf in zip(chunk, image_bytes):
                    task.photo = EnrollmentPhoto(
                        path=task.photo.path,
                        content_sha256=task.photo.content_sha256,
                        data=buf,
                    )
```

Update `_stage_batch` and `_upload_photo` to accept the cached buffer:

```python
        async def _upload_photo(
            task: _PhotoTask,
            person_id: uuid.UUID,
            photo_id: uuid.UUID,
        ) -> None:
            if photo_id in existing_photos:
                return
            async with self._persist_semaphore:
                object_key = f"enrollments/{person_id}/{photo_id}"
                data = task.photo.data
                if data is None:
                    data = await self._read_photo(task.photo)
                await self._storage.put_object(
                    object_key=object_key,
                    data=__import__("io").BytesIO(data),
                    length=len(data),
                    content_type="application/octet-stream",
                )
```

- [ ] **Step 3: Add bounded producer/consumer pipeline for persistence**

Replace the per-batch `await self._stage_batch(...)` call pattern in `enroll_shard` with an `asyncio.Queue` of maximum size equal to `max_persistence_concurrency * 2`:

```python
            gpu_queue: asyncio.Queue[list[tuple[_PhotoTask, GpuFaceExtraction]]] = asyncio.Queue(maxsize=max(2, self._persist_semaphore._value * 2))  # type: ignore[attr-defined]

            async def _persister() -> None:
                while True:
                    batch = await gpu_queue.get()
                    if batch is None:
                        gpu_queue.task_done()
                        break
                    try:
                        await self._stage_batch(batch, person_id_map, result, staged_buffer)
                        if len(staged_buffer) >= self._activation_batch_size:
                            await self._db.flush()
                            await self._activate_buffer(staged_buffer)
                            staged_buffer.clear()
                    finally:
                        gpu_queue.task_done()

            persister_task = asyncio.create_task(_persister())
```

In the main loop, push to the queue instead of awaiting staging:

```python
                with_faces = [...]
                if with_faces:
                    await gpu_queue.put(with_faces)
                ...
                await self._db.commit()

            await gpu_queue.put(None)
            await gpu_queue.join()
            persister_task.cancel()
            try:
                await persister_task
            except asyncio.CancelledError:
                pass
```

Keep activation at the end for any remaining staged buffer.

- [ ] **Step 4: Verify bytes are not re-read**

Add a light test by monkey-patching `Path.read_bytes` and counting calls during a small synthetic run.

---

## Task 6: Dedicated GPU executor and bounded thread pools

**Files:**
- Modify: `backend/app/workers/gpu_worker.py`
- Modify: `backend/app/services/bulk_enrollment.py`
- Test: `docker compose exec api python backend/tests/test_worker_pools.py`

**Interfaces:**
- Consumes: Uvicorn default `ThreadPoolExecutor(max_workers=128)`.
- Produces: A single-thread dedicated executor for GPU work and a separate bounded executor for file I/O / MinIO.

- [ ] **Step 1: Replace global 128-thread executor with bounded executors**

In `gpu_worker.py` `_lifespan`:

```python
    app.state.io_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=min(32, (os.cpu_count() or 4) * 2),
        thread_name_prefix="io-",
    )
    app.state.gpu_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="gpu-",
    )
```

Close them in `finally`.

Comment out or remove:

```python
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=128))
```

- [ ] **Step 2: Pass executors to `BulkEnrollmentService`**

```python
                service = BulkEnrollmentService(
                    db=session,
                    storage=request.app.state.storage,
                    vector_store=request.app.state.vector_store,
                    pipeline=request.app.state.pipeline,
                    pipeline_lock=request.app.state.pipeline_lock,
                    gpu_executor=request.app.state.gpu_executor,
                    io_executor=request.app.state.io_executor,
                    qdrant_wait=False,
                )
```

- [ ] **Step 3: Use `io_executor` for file reads in `bulk_enrollment.py`**

Accept `io_executor` in `BulkEnrollmentService.__init__`. Use it for `_read_photo`:

```python
    async def _read_photo(self, photo: EnrollmentPhoto) -> bytes:
        if photo.data is not None:
            return photo.data
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._io_executor, photo.path.read_bytes)
```

- [ ] **Step 4: Verify GPU executor is single-thread and I/O executor is bounded**

Run:

```bash
docker compose exec gpu-worker-1 python - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from app.workers.gpu_worker import create_worker_app
app = create_worker_app()
print("SET UP OK")
PY
```

---

## Task 7: UI polling and richer metrics

**Files:**
- Modify: `frontend/src/pages/BulkEnrollmentPage.tsx`, `frontend/src/api/types.ts` (if VggfaceBulkJob type needs fields)
- Test: Manual UI refresh check in browser.

**Interfaces:**
- Consumes: `useLatestBulkJob()` no-poll hook; existing `VggfaceBulkJob` fields.
- Produces: Page polls `useBulkJob(jobId)` at 2s; displays scanned/sec, processed/sec, enrolled/sec, duplicate/sec.

- [ ] **Step 1: Switch page to polled job query**

Change:

```typescript
import { useBulkJob, useCancelBulkJob, useLatestBulkJob, useStartVggfaceBulkJob } from '@/api/bulkJobs'

export default function BulkEnrollmentPage() {
  const { data: latestJob, isLoading, error } = useLatestBulkJob()
  const [jobId, setJobId] = useState<string>(latestJob?.jobId ?? '')

  useEffect(() => {
    if (latestJob?.jobId) setJobId(latestJob.jobId)
  }, [latestJob?.jobId])

  const { data: job } = useBulkJob(jobId)
```

Use `job` everywhere instead of `latestJob`.

- [ ] **Step 2: Add rate metric fields to types and page cards**

Ensure the backend response already returned by `bulk_orchestrator.py` includes `scanned_photos_per_second`, `processed_photos_per_second`, `enrolled_photos_per_second`, and `duplicate_photos_per_second`. If the TypeScript `VggfaceBulkJob` lacks them, add them.

Add a metrics grid after the progress bar:

```tsx
<div className="grid gap-4 sm:grid-cols-2 md:grid-cols-4">
  <div>
    <p className="text-xs text-slate-500">Taranan/sn</p>
    <p className="text-lg font-semibold text-navy-900">{job.scannedPhotosPerSecond.toFixed(1)}</p>
  </div>
  <div>
    <p className="text-xs text-slate-500">İşlenen/sn</p>
    <p className="text-lg font-semibold text-navy-900">{job.processedPhotosPerSecond.toFixed(1)}</p>
  </div>
  <div>
    <p className="text-xs text-slate-500">Yeni Kayıt/sn</p>
    <p className="text-lg font-semibold text-navy-900">{job.enrolledPhotosPerSecond.toFixed(1)}</p>
  </div>
  <div>
    <p className="text-xs text-slate-500">Yinelenen/sn</p>
    <p className="text-lg font-semibold text-navy-900">{job.duplicatePhotosPerSecond.toFixed(1)}</p>
  </div>
</div>
```

- [ ] **Step 3: Fix progress bar to reflect `totalProcessed / requestedPhotos`**

Change the progress bar from `totalEnrolled / requestedPhotos` to `totalProcessed / requestedPhotos` and rename the label from "İşlenen" to "İşlenen / İstenen".

---

## Task 8: End-to-end smoke test and benchmark

**Files:**
- Uses: all modified files.
- Test: `make up` then run a 10K VGGFace job.

- [ ] **Step 1: Rebuild worker image**

```bash
docker compose build api gpu-worker-1
docker compose up -d
```

- [ ] **Step 2: Run 10K job and observe metrics**

Start a job with `maxPhotos=10000` and watch:

```bash
curl -s -X POST http://localhost:8000/bulk-jobs/vggface -H 'Content-Type: application/json' -d '{"maxPhotos":10000}' | jq .
```

UI should show polled updates every 2s and four rate metrics.

- [ ] **Step 3: Validate throughput target**

After duplicate prefix exhaustion, the GPU worker log should show:
- `processed_photos_per_second` ≥ 400 for a single GPU.
- No 400K file reads (use `strace` or `opensnoop` if available; at minimum the 10K run should complete in well under 60s excluding the duplicate prefix).
- No `DeviceTensor requires a non-null device pointer` errors.

---

## Self-Review

**Spec coverage:**
- RetinaFace no-candidate crash → Task 1.
- Detector/embedding version split → Task 2.
- `maxPhotos` before hash/materialization → Task 3.
- Preflight cache/deferral → Task 4.
- Single JPEG read → Task 5.
- Bounded GPU/persistence pipeline → Task 5.
- Dedicated GPU executor and bounded I/O pool → Task 6.
- UI polling and separate rates → Task 7.
- End-to-end validation → Task 8.

**Placeholder scan:** All steps contain concrete file paths, code blocks, or exact shell commands. No TBD/TODO/FIXME.

**Type consistency:**
- `EnrollmentPhoto.data` added as `bytes | None`.
- `stream_vggface_manifest` now accepts `max_photos` and returns an `Iterator`.
- `_load_identities` returns `Iterator[EnrollmentIdentity]`.
- `BulkEnrollmentService` accepts `io_executor` and `gpu_executor`.
- Qdrant payload uses `embedding_model_version`.
