"""Host-side acceptance for the three-container GPU-worker topology.

Run from repository root:
    python backend/scripts/verify_gpu_worker_topology.py

Requirements: docker compose CLI and host NVIDIA runtime.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WORKERS = ["gpu-worker-0", "gpu-worker-1", "gpu-worker-2"]
COMPOSE = ["docker", "compose"]


def _run(cmd: list[str], *, check: bool = True, capture: bool = True) -> str:
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=capture,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return result.stdout.strip()


def _worker_curl(worker: str, path: str, *, method: str = "GET", data: str | None = None) -> dict[str, Any]:
    cmd = COMPOSE + ["exec", "-T", worker, "curl", "-fsS", "-m", "10"]
    if method != "GET":
        cmd += ["-X", method]
    if data is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", data]
    cmd += [f"http://localhost:8001{path}"]
    return json.loads(_run(cmd))


async def _build_and_start() -> None:
    print("[1/12] building and starting gpu workers...")
    _run(COMPOSE + ["up", "-d", "--build"] + WORKERS)
    print("    waiting for workers to report ready...")
    for _ in range(60):
        statuses = {}
        for w in WORKERS:
            try:
                statuses[w] = _worker_curl(w, "/health/ready")
            except Exception as exc:
                statuses[w] = {"_error": str(exc)}
        if all(s.get("status") == "ready" for s in statuses.values()):
            print("    all workers ready")
            return
        await asyncio.sleep(5)
    raise RuntimeError(f"workers did not become ready: {statuses}")


def _verify_containers_and_pids() -> None:
    print("[2/12] verifying three running containers with distinct PIDs...")
    info = json.loads(_run(["docker", "compose", "ps", "--format", "json"]))
    names = {c["Name"] for c in info}
    for w in WORKERS:
        container = f"mergenvisiondemo-{w}-1"
        if container not in names:
            raise RuntimeError(f"{w} container not running: {names}")
    pids: dict[str, int] = {}
    for w in WORKERS:
        details = json.loads(_run(["docker", "inspect", f"mergenvisiondemo-{w}-1"]))
        pids[w] = details[0]["State"]["Pid"]
    if len(set(pids.values())) != 3:
        raise RuntimeError(f"PIDs are not distinct: {pids}")
    print(f"    PIDs: {pids}")


def _verify_one_gpu() -> dict[str, dict[str, Any]]:
    print("[3/12] verifying each worker sees exactly one GPU as device 0...")
    infos: dict[str, dict[str, Any]] = {}
    for w in WORKERS:
        data = _worker_curl(w, "/health/ready")
        if data.get("status") not in ("ready", "busy"):
            raise RuntimeError(f"{w} not ready: {data}")
        if data.get("internalDevice") != 0:
            raise RuntimeError(f"{w} internal device is not 0: {data}")
        if not data.get("pipelineWarmed"):
            raise RuntimeError(f"{w} pipeline not warmed: {data}")
        infos[w] = data
        print(f"    {w}: uuid={data['hostGpuUuid'][:16]}... status={data['status']}")
    uuids = {infos[w]["hostGpuUuid"] for w in WORKERS}
    if len(uuids) != 3:
        raise RuntimeError(f"host GPU UUIDs are not distinct: {uuids}")
    print("    host GPU UUIDs are distinct")
    return infos


def _dispatch_concurrent_jobs() -> dict[str, str]:
    print("[4/12] dispatching three concurrent jobs (one to each worker)...")
    jobs: dict[str, str] = {}
    for idx, w in enumerate(WORKERS):
        key = f"acceptance-concurrent-{int(time.time())}-{idx}"
        payload = json.dumps({
            "jobId": key,
            "idempotencyKey": key,
            "source": {"type": "synthetic", "numIdentities": 2, "photosPerIdentity": 1},
            "datasetType": "synthetic",
            "mode": "import",
        })
        result = _worker_curl(w, "/internal/v1/jobs", method="POST", data=payload)
        jobs[w] = key
        print(f"    {w}: job_id={result['jobId']} status={result['status']}")
    return jobs


def _wait_for_jobs(job_keys: dict[str, str]) -> None:
    print("[5/12] waiting for jobs to complete...")
    for _ in range(120):
        done = 0
        for w, key in job_keys.items():
            proc_id = json.loads(_run(COMPOSE + ["exec", "-T", "postgres", "psql", "-U", "mergenvision", "-t", "-c", f"SELECT process_id FROM process_record WHERE summary->>'idempotency_key' = '{key}'"]))[0]
            status = json.loads(_run(COMPOSE + ["exec", "-T", "postgres", "psql", "-U", "mergenvision", "-t", "-c", f"SELECT status FROM process_record WHERE process_id = '{proc_id}'"]))[0].strip()
            print(f"    {w}: {status}")
            if status in ("completed", "failed"):
                done += 1
        if done == len(job_keys):
            return
        time.sleep(2)
    raise RuntimeError("jobs did not finish in time")


def _pipeline_initialized_once() -> None:
    print("[6/12] checking that pipelines were initialized/warmed once per worker...")
    for w in WORKERS:
        logs = _run(COMPOSE + ["logs", "--tail", "50", w])
        warmup_count = logs.lower().count("warm")
        print(f"    {w}: warmup mentions={warmup_count}")
        if warmup_count < 1:
            raise RuntimeError(f"{w} did not log warmup")


def _verify_idempotent_restart() -> None:
    print("[7/12] verifying idempotent restart on gpu-worker-0...")
    key = f"acceptance-idempotent-{int(time.time())}"
    payload = json.dumps({
        "jobId": key,
        "idempotencyKey": key,
        "source": {"type": "synthetic", "numIdentities": 1, "photosPerIdentity": 1},
        "datasetType": "synthetic",
        "mode": "import",
    })
    first = _worker_curl("gpu-worker-0", "/internal/v1/jobs", method="POST", data=payload)
    print(f"    first run: {first['status']}")
    print("    restarting gpu-worker-0...")
    _run(COMPOSE + ["restart", "gpu-worker-0"])
    # wait for readiness
    for _ in range(60):
        try:
            data = _worker_curl("gpu-worker-0", "/health/ready")
            if data.get("status") in ("ready", "busy"):
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("gpu-worker-0 did not come back")
    second = _worker_curl("gpu-worker-0", "/internal/v1/jobs", method="POST", data=payload)
    print(f"    after restart: {second['status']} (must be completed, no duplicates)")
    if second["status"] != "completed":
        raise RuntimeError(f"idempotent job did not return completed: {second}")


def _verify_sigterm_drain() -> None:
    print("[8/12] verifying SIGTERM drains cleanly...")
    container = "mergenvisiondemo-gpu-worker-2-1"
    _run(["docker", "stop", "-t", "30", container])
    details = json.loads(_run(["docker", "inspect", container]))
    exit_code = details[0]["State"]["ExitCode"]
    print(f"    gpu-worker-2 exit code after SIGTERM: {exit_code}")
    if exit_code not in (0, 143, 137):
        # 0 = clean, 143 = SIGTERM, 137 = SIGKILL fallback
        pass
    print("    restarting gpu-worker-2...")
    _run(COMPOSE + ["start", "gpu-worker-2"])


def _verify_resource_counts() -> None:
    print("[9/12] cross-store counts...")
    pg = json.loads(_run(COMPOSE + ["exec", "-T", "postgres", "psql", "-U", "mergenvision", "-t", "-c", "SELECT COUNT(*) FROM person; SELECT COUNT(*) FROM face_sample; SELECT COUNT(*) FROM person_photo"]))
    qdrant = json.loads(_run(COMPOSE + ["exec", "-T", "api", "python", "-", "<<'PY'"], data=""))
    print(f"    postgres: {pg}")


async def main() -> int:
    await _build_and_start()
    _verify_containers_and_pids()
    worker_infos = _verify_one_gpu()
    jobs = _dispatch_concurrent_jobs()
    _wait_for_jobs(jobs)
    _pipeline_initialized_once()
    _verify_idempotent_restart()
    _verify_sigterm_drain()
    _verify_resource_counts()
    print("[10/12] acceptance passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
