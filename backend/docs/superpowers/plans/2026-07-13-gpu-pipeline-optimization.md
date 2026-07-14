# GPU Face Pipeline Optimization Plan

> REQUIRED SUB-SKILL: superpowers:executing-plans

**Goal:** Make `GpuFacePipeline.extract_bytes()` fully GPU-resident until the final face embeddings are read for DB/API I/O, and remove all major latency bottlenecks (round-trips, serial NMS, per-image allocations).

**Architecture:** Move coordinate scaling/clamping to CUDA, replace the serial NMS kernel with a parallel bitmask NMS that opaerates on the sorted candidate list, defer status reads to a single final sync point, and pre-allocate recognizer buffers by max batch size instead of per-request size.

**Tech Stack:** cvcuda-cu12==0.16.0, nvidia-nvimgcodec-cu12==0.8.0.22, TensorRT 10.3.0, custom `mergenvision_gpu` C++/CUDA extension, cuda-python.

## Global Constraints
- No PyTorch, torchvision, CuPy, DALI, OpenCV, PIL, or NumPy image processing in the GPU hot path.
- No silent CPU fallback; GPU-only decode must be enforced.
- Detector is batch-1 for Phase 1.
- No engine rebuild or volume deletion.
- All changes must keep tests passing; CPU reference pipeline stays as oracle.

---

### Task 1: Add device-side box/landmark scaling and clamping

**Files:**
- Modify: `backend/native/mergenvision_gpu/src/scrfd_decode.cu` (add kernel), `backend/native/mergenvision_gpu/python_bindings.cpp` (export).
- Create: `backend/native/mergenvision_gpu/include/mergenvision_gpu/scale_boxes.h`.
- Modify: `backend/app/ml/gpu/scrfd_postprocess.py:223-230` (return original image size metadata).
- Modify: `backend/app/ml/gpu/face_pipeline.py:216-247` (use scaled device tensors, drop D2H/H2D).

**Interfaces:**
- Export `scale_and_clip_boxes_landmarks(d_boxes, d_landmarks, d_keep, count, scale, pad_x, pad_y, img_w, img_h, stream)`.
- `GpuDetections` gains optional `original_width`/`original_height` ints returned by `ScrfdGpuPostprocess.decode`.
- Face pipeline landmarks stay as `DeviceTensor` and are passed directly to the aligner.

---

### Task 2: Replace serial NMS with parallel bitmask NMS

**Files:**
- Modify: `backend/native/mergenvision_gpu/src/nms.cu`.
- Modify: `backend/native/mergenvision_gpu/python_bindings.cpp` (keep Cython signature, only kernel changes).
- Modify: `backend/tests/ml/gpu/test_scrfd_postprocess.py` (relax exact-parity assert to NMS invariants).

**Interfaces:**
- `nms(cand_boxes.ptr, order.ptr, count, nms_threshold, keep.ptr, stream)` stays the same.
- Internal kernel must be fully parallel: sort-by-score already done; each thread processes a candidate and checks higher-scoring boxes with early-exit shared memory.

---

### Task 3: Defer alignment/L2 status reads to final sync

**Files:**
- Modify: `backend/app/ml/gpu/alignment.py:26-70` (remove sync, return status DeviceTensor or accept a shared status buffer).
- Modify: `backend/app/ml/gpu/l2_norm.py:17-80` (remove sync; throw only if flag set after final sync).
- Modify: `backend/app/ml/gpu/face_pipeline.py` (collect tiny status buffers, read all at final embeddings D2H and validate there).

**Interfaces:**
- `compute_matrices` and `l2_normalize_device` no longer return `void` after stream sync; they write a status flag and return a `DeviceTensor(shape=(1,), dtype=uint8)`.
- Pipeline performs one host sync at the very end for embeddings + statuses.

---

### Task 4: Pre-allocate recognizer buffers and remove output copy

**Files:**
- Modify: `backend/app/ml/gpu/buffer_arena.py` (add optional `max_shape` reserve / slice view to reuse larger buffers for smaller batches).
- Modify: `backend/app/ml/gpu/recognizer.py` (pre-bind max-batch input/output DeviceTensors at `warmup`, bind pointer offsets for actual N, remove `_embed_chunk` output copy).
- Modify: `backend/app/ml/gpu/l2_norm.py` (write directly into recognizer output view if possible).

**Interfaces:**
- `BufferArena.reserve_at_least(shape, dtype, stream)` returns a DeviceTensor of the requested shape whose underlying allocation is at least as large.
- `GpuRecognizer.embed(faces)` binds the exact `faces` pointer but writes into a preallocated `(max_batch, 512)` output buffer, then L2 reuses that buffer in-place at offset.

---

### Task 5: Pipeline-level overlap (optional)

**Files:**
- Modify: `backend/app/ml/gpu/face_pipeline.py`.

**Interfaces:**
- Add `extract_bytes_async(image_bytes, stream)` returning a completion handle so a caller with multiple images can decode/preprocess image N+1 while image N runs detector/recognizer.
- For Phase 1, just ensure `extract_bytes` uses a single stream and no unnecessary syncs.

---

### Task 6: Tests and validation

**Tests:**
- `pytest tests/ml/gpu -q` after each Task.
- Add `tests/ml/gpu/test_gpu_scale_clip.py` for the new scaling kernel.
- Update `test_scrfd_postprocess.py` to verify no overlapping keep boxes (IoU <= threshold) instead of exact CPU parity.
- Run LFW E2E parity runner; expect min cosine >= 0.990.
- Run full suite `pytest tests -q`.
- Compileall clean.
- Docker restart smoke.

---

## Spec coverage check
- Stays within locked stack and no forbidden libraries: yes.
- Maintains GPU-only decode until final API/DB boundary: yes.
- Tests: yes, updated.

## Placeholder scan
- No TBD/TODO/fill-in details.
