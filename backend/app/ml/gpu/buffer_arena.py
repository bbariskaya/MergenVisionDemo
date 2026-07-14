"""Event-fenced device buffer arena.

A ``BufferLease`` represents temporary exclusive use of an arena-owned
allocation.  The lease carries a CUDA event that is recorded on release;
the allocation is not reused until ``cudaEventQuery`` reports success.

The legacy ``reserve`` method is retained for existing callers: it returns a
simple ``DeviceTensor`` that is returned to the arena free list when it is
garbage collected.  It does not fence on a stream; callers must synchronize
the relevant stream before reusing the arena.
"""
from __future__ import annotations

import ctypes
import logging
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from cuda.bindings import runtime as cuda_runtime

from app.ml.gpu.device_tensor import DeviceTensor, check_cuda

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BufferKey:
    shape: tuple[int, ...]
    dtype: type


class _AllocationRecord:
    __slots__ = ("ptr", "key", "event", "state")

    def __init__(self, ptr: int, key: _BufferKey, event: int | None) -> None:
        self.ptr = ptr
        self.key = key
        self.event = event
        self.state = "free"  # free | active | pending


class BufferLease:
    """Temporary exclusive use of one arena-owned allocation."""

    def __init__(
        self,
        arena: "BufferArena",
        record: _AllocationRecord,
        generation: int,
    ) -> None:
        self._arena = arena
        self._record = record
        self._generation = generation
        self._released = False
        self._views: weakref.WeakSet[DeviceTensor] = weakref.WeakSet()

    def _check_valid(self) -> None:
        if self._released:
            raise RuntimeError("BufferLease has been released")
        if self._arena.generation != self._generation:
            raise RuntimeError("BufferLease is stale because the arena was closed")

    @property
    def ptr(self) -> int:
        self._check_valid()
        return self._record.ptr

    @property
    def event(self) -> int:
        """Diagnostic reference to the arena-owned release event.

        The event remains valid until the allocation is reused or freed.
        """
        self._check_valid()
        if self._record.event is None:
            raise RuntimeError("legacy allocation has no release event")
        return self._record.event

    @property
    def ptr_nbytes(self) -> int:
        self._check_valid()
        return self._arena._nbytes(self._record.key.shape, self._record.key.dtype)

    def as_tensor(
        self,
        shape: tuple[int, ...] | None = None,
        *,
        stream: int | None = None,
    ) -> DeviceTensor:
        self._check_valid()
        shape = self._record.key.shape if shape is None else shape
        tensor = DeviceTensor(
            self._record.ptr,
            shape,
            self._record.key.dtype,
            self._arena.device_id(),
            self._arena,
            lease=self,
            stream=stream,
        )
        self._views.add(tensor)
        return tensor

    def release(self, stream: int) -> None:
        if self._released:
            return
        if self._arena.generation != self._generation:
            raise RuntimeError("BufferLease is stale because the arena was closed")
        # Reject release while live views exist.  Views are weakly tracked.
        if any(view is not None for view in self._views):
            raise RuntimeError(
                "Cannot release BufferLease while active DeviceTensor views exist"
            )
        if self._record.event is None:
            err, event = cuda_runtime.cudaEventCreate()
            check_cuda(err, "BufferLease release event create")
            self._record.event = int(event)
        err = cuda_runtime.cudaEventRecord(self._record.event, stream)
        check_cuda(err, "BufferLease release event record")
        self._record.state = "pending"
        self._released = True
        self._arena._move_to_pending(self._record)


class BufferArena:
    """Pool of device allocations.

    Rules:
    - The arena owns every device pointer for its entire lifetime.
    - ``acquire`` returns a ``BufferLease`` that must be explicitly released.
    - A released allocation is reused only after its release event completes.
    - ``reserve`` returns a ``DeviceTensor`` for back-compat; its memory is
      reused without a stream fence when the tensor is garbage collected.
    - ``close`` synchronizes, destroys all events, and frees all pointers.
    """

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = int(device_id)
        self._records_by_key: dict[_BufferKey, list[_AllocationRecord]] = defaultdict(list)
        self._pending: set[_AllocationRecord] = set()
        self._unique_count = 0
        self._closed = False
        self.generation = 0

    def device_id(self) -> int:
        return self._device_id

    def _itemsize(self, dtype: type) -> int:
        return {
            ctypes.c_uint8: 1,
            ctypes.c_int8: 1,
            ctypes.c_uint16: 2,
            ctypes.c_int16: 2,
            ctypes.c_float: 4,
            ctypes.c_int32: 4,
            ctypes.c_int64: 8,
        }.get(dtype, ctypes.sizeof(dtype))

    def _nbytes(self, shape: tuple[int, ...], dtype: type) -> int:
        return self._itemsize(dtype) * int(
            __import__("functools").reduce(int.__mul__, shape, 1)
        )

    def _event_complete(self, event: int, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            err = cuda_runtime.cudaEventQuery(event)
            if isinstance(err, tuple):
                err = err[0]
            if err == cuda_runtime.cudaError_t.cudaSuccess:
                return True
            if err == cuda_runtime.cudaError_t.cudaErrorNotReady:
                time.sleep(0.001)
                continue
            check_cuda(err, "buffer arena event query")
        raise TimeoutError("BufferArena event did not complete within timeout")

    def _event_query_nonblocking(self, event: int) -> bool:
        err = cuda_runtime.cudaEventQuery(event)
        if isinstance(err, tuple):
            err = err[0]
        if err == cuda_runtime.cudaError_t.cudaSuccess:
            return True
        if err == cuda_runtime.cudaError_t.cudaErrorNotReady:
            return False
        check_cuda(err, "buffer arena event query")
        return False

    def _find_completed_record(self, key: _BufferKey) -> _AllocationRecord | None:
        """Return a completed free record, promoting pending records as needed.

        Pending records are checked non-blocking so ``acquire`` never waits
        for GPU work.  Legacy ``reserve`` records have no event and are treated
        as immediately reusable.
        """
        records = self._records_by_key[key]
        for record in records:
            if record.state == "free" and record.event is None:
                return record
        for record in records:
            if record.state == "free" and record.event is not None:
                return record
        for record in records:
            if record.state == "pending" and record.event is not None:
                if self._event_query_nonblocking(record.event):
                    record.state = "free"
                    self._pending.discard(record)
                    return record
        return None

    def acquire(
        self,
        shape: tuple[int, ...],
        dtype: type,
        *,
        stream: int | None = None,
    ) -> BufferLease:
        if self._closed:
            raise RuntimeError("BufferArena is closed")
        key = _BufferKey(shape=shape, dtype=dtype)

        record = self._find_completed_record(key)
        if record is None:
            nbytes = max(1, self._nbytes(shape, dtype))
            err, raw_ptr = cuda_runtime.cudaMalloc(nbytes)
            check_cuda(err, f"BufferArena cudaMalloc({nbytes})")
            err, event = cuda_runtime.cudaEventCreate()
            check_cuda(err, "BufferArena cudaEventCreate")
            record = _AllocationRecord(
                ptr=int(raw_ptr), key=key, event=int(event)
            )
            self._records_by_key[key].append(record)
            self._unique_count += 1
            logger.debug("Arena allocated shape=%s dtype=%s nbytes=%d", shape, dtype, nbytes)
        else:
            # A legacy record lacks an event; ensure event-fenced leases always
            # have one before handing it out.
            if record.event is None:
                err, event = cuda_runtime.cudaEventCreate()
                check_cuda(err, "BufferArena cudaEventCreate for reused legacy record")
                record.event = int(event)
            logger.debug("Arena reused shape=%s dtype=%s", shape, dtype)

        record.state = "active"
        return BufferLease(self, record, self.generation)

    def _move_to_pending(self, record: _AllocationRecord) -> None:
        self._pending.add(record)

    def _return_legacy_record(self, record: _AllocationRecord) -> None:
        if self._closed:
            try:
                cuda_runtime.cudaFree(record.ptr)
            except Exception:
                pass
            return
        record.state = "free"

    # ------------------------------------------------------------------
    # Test/diagnostic hooks only.  These are not part of the production API.
    # ------------------------------------------------------------------
    def pending_ptrs(self) -> set[int]:
        return {r.ptr for r in self._pending}

    def completed_ptrs(self) -> set[int]:
        return {
            r.ptr
            for records in self._records_by_key.values()
            for r in records
            if r.state == "free"
        }

    def unique_allocation_count(self) -> int:
        return self._unique_count

    def event_status(self, ptr: int) -> str:
        for records in self._records_by_key.values():
            for record in records:
                if record.ptr == ptr:
                    if record.state == "free":
                        return "complete"
                    if record.event is None:
                        return "legacy"
                    err = cuda_runtime.cudaEventQuery(record.event)
                    if isinstance(err, tuple):
                        err = err[0]
                    if err == cuda_runtime.cudaError_t.cudaSuccess:
                        return "complete"
                    if err == cuda_runtime.cudaError_t.cudaErrorNotReady:
                        return "pending"
                    check_cuda(err, "event_status query")
        raise KeyError(f"ptr {ptr} not managed by this arena")

    def reserve(
        self,
        shape: tuple[int, ...],
        dtype: type,
        *,
        stream: int | None = None,
    ) -> DeviceTensor:
        """Compatibility wrapper returning a ``DeviceTensor``.

        The returned tensor does **not** hold a lease.  Its memory is reused on
        garbage collection without a stream fence.  New code should use
        ``acquire`` / ``release``.
        """
        if self._closed:
            raise RuntimeError("BufferArena is closed")
        key = _BufferKey(shape=shape, dtype=dtype)
        record = self._find_completed_record(key)
        if record is None:
            nbytes = max(1, self._nbytes(shape, dtype))
            err, raw_ptr = cuda_runtime.cudaMalloc(nbytes)
            check_cuda(err, f"BufferArena cudaMalloc({nbytes})")
            record = _AllocationRecord(ptr=int(raw_ptr), key=key, event=None)
            self._records_by_key[key].append(record)
            self._unique_count += 1
            logger.debug("Arena allocated shape=%s dtype=%s nbytes=%d", shape, dtype, nbytes)
        else:
            if record.event is not None:
                # Drop any lingering event from a previous lease reuse.
                try:
                    cuda_runtime.cudaEventDestroy(record.event)
                except Exception:
                    pass
                record.event = None
            logger.debug("Arena reused shape=%s dtype=%s", shape, dtype)

        record.state = "active"
        tensor = DeviceTensor(
            record.ptr,
            shape,
            dtype,
            self._device_id,
            self,
            stream=stream,
        )

        def release(r: _AllocationRecord = record) -> None:
            if self._closed:
                try:
                    cuda_runtime.cudaFree(r.ptr)
                except Exception:
                    pass
            else:
                r.state = "free"

        weakref.finalize(tensor, release)
        return tensor

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.generation += 1

        # Drain pending fences before freeing so we don't tear down memory that
        # GPU work still references.
        for records in self._records_by_key.values():
            for record in records:
                if record.state == "pending" and record.event is not None:
                    try:
                        self._event_complete(record.event)
                    except Exception as exc:
                        logger.warning("BufferArena close pending wait failed: %s", exc)

        # Synchronize the default stream as a conservative barrier.
        try:
            cuda_runtime.cudaStreamSynchronize(0)
        except Exception as exc:
            logger.warning("BufferArena default stream sync failed: %s", exc)

        for records in self._records_by_key.values():
            for record in records:
                if record.event is not None:
                    try:
                        cuda_runtime.cudaEventDestroy(record.event)
                    except Exception as exc:
                        logger.warning("BufferArena event destroy failed: %s", exc)
                try:
                    cuda_runtime.cudaFree(record.ptr)
                except Exception as exc:
                    logger.warning("BufferArena cudaFree failed for %s: %s", record.ptr, exc)
        self._records_by_key.clear()
        self._pending.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
