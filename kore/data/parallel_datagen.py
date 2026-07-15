"""Parallel datagen: shard tasks across all GPUs with concurrent teacher streams.

The campaign's datagen/agentic stages are embarrassingly parallel across tasks (each
task = independent teacher calls + GPU verification), but run sequentially by default.
On an N-GPU box that is an N-fold waste of both the GPU verifier AND the teacher
gateway latency (each teacher call is mostly network wait, during which the GPU is idle).

This module runs the SAME datagen work across ``n_workers`` processes, each pinned to
a distinct GPU (``HIP_VISIBLE_DEVICES``) with its OWN teacher stream, so N workers give
~N concurrent teacher streams + N GPUs. It is fully RESUMABLE: a (task, kind) whose
shard already exists (non-empty) is skipped, so a crash/restart never redoes work.

No pipeline shortcut: identical generators (``generate_repairs`` / ``generate_groups``
/ ``generate_wins`` / ``generate_agentic_trajectories``) at identical counts - only the
scheduling is parallelized. Spawn start-method (never fork) so each worker gets a clean
interpreter and pins its GPU BEFORE importing torch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DATAGEN_KINDS = ("repair", "groups", "wins")
AGENTIC_KINDS = ("agentic",)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable)
# --------------------------------------------------------------------------- #
def shard_tasks(task_ids: list[str], n_workers: int) -> list[list[str]]:
    """Round-robin shard task ids across ``n_workers`` (balanced, deterministic)."""
    n_workers = max(1, int(n_workers))
    shards: list[list[str]] = [[] for _ in range(n_workers)]
    for i, tid in enumerate(task_ids):
        shards[i % n_workers].append(tid)
    return [s for s in shards if s]


def shard_done(data_root, task_id: str, kind: str) -> bool:
    """True iff this (task, kind) shard already exists non-empty (resume skip)."""
    p = Path(data_root) / kind / f"{task_id}.jsonl"
    try:
        return p.exists() and p.stat().st_size > 0
    except OSError:
        return False


def detect_gpus(default: int = 8) -> int:
    """GPU count WITHOUT initializing CUDA in this process (clean for spawn).

    Runs ``torch.cuda.device_count()`` in a throwaway subprocess; falls back to
    ``default`` if torch/ROCm is unavailable."""
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import torch;print(torch.cuda.device_count())"],
            capture_output=True, text=True, timeout=120)
        n = int((out.stdout or "").strip().splitlines()[-1])
        return n if n > 0 else default
    except Exception:  # noqa: BLE001
        return default


# --------------------------------------------------------------------------- #
# Worker (runs in a spawned process, pinned to one GPU)
# --------------------------------------------------------------------------- #
def _generate(kind: str, task, teacher, env, counts: dict):
    if kind == "repair":
        from kore.data.gen_repair import generate_repairs
        return generate_repairs(task, teacher, env, n=counts["n_repair"])
    if kind == "groups":
        from kore.data.gen_groups import generate_groups
        return generate_groups(task, teacher, env, n_parents=counts["n_parents"], k=counts["k"])
    if kind == "wins":
        from kore.data.gen_wins import generate_wins
        return generate_wins(task, teacher, env, gens=counts["wins_gens"])
    if kind == "agentic":
        from kore.data.gen_agentic import generate_agentic_trajectories
        recs = generate_agentic_trajectories(
            task, teacher, env, n=counts["n_agentic"],
            max_turns=counts["max_tool_turns"], keep_only_useful=True)
        return [r.to_dict() for r in recs]
    raise ValueError(f"unknown datagen kind {kind!r}")


def _worker(payload: dict) -> list[tuple]:
    """Generate all (task, kind) shards assigned to this worker on its pinned GPU."""
    gpu = str(payload["gpu_id"])
    # Pin the GPU BEFORE any torch import (both the worker and the driver
    # subprocesses KoreEnv spawns inherit it, so verification lands on this GPU).
    # ONLY HIP_VISIBLE_DEVICES: setting ROCR_VISIBLE_DEVICES *and* HIP_VISIBLE_DEVICES
    # to the same index DOUBLE-REMAPS on ROCm (ROCR filters to one device -> index 0,
    # then HIP selects index N>0 -> no device -> torch.cuda.is_available()==False),
    # which silently breaks every verifier subprocess on GPUs 1..7. Clear the others
    # so only HIP_VISIBLE_DEVICES applies (verified: HIP-only -> avail=True).
    os.environ["HIP_VISIBLE_DEVICES"] = gpu
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    from kore.data.schemas import write_jsonl
    from kore.data.teacher import load_env_local, make_teacher
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import get_task

    load_env_local()
    tkw = {"model": payload["model_teacher"]} if payload.get("model_teacher") else {}
    teacher = make_teacher(payload["teacher_kind"], resilient=True, **tkw)

    data_root = payload["data_root"]
    kinds = payload["kinds"]
    counts = payload["counts"]
    results: list[tuple] = []
    for tid in payload["task_ids"]:
        try:
            task = get_task(tid)
            env = KoreEnv(task)
        except Exception as e:  # noqa: BLE001
            print(f"[datagen w{gpu}] {tid}: setup ERROR {type(e).__name__}: {e}", flush=True)
            continue
        for kind in kinds:
            if shard_done(data_root, tid, kind):
                print(f"[datagen w{gpu}] {tid}:{kind} skip (resume)", flush=True)
                results.append((tid, kind, "skip", 0))
                continue
            try:
                recs = _generate(kind, task, teacher, env, counts)
                out = Path(data_root) / kind / f"{tid}.jsonl"
                out.parent.mkdir(parents=True, exist_ok=True)
                write_jsonl(out, recs)
                print(f"[datagen w{gpu}] {tid}:{kind} -> {len(recs)} records", flush=True)
                results.append((tid, kind, "done", len(recs)))
            except Exception as e:  # noqa: BLE001 - one shard failure never aborts the shard set
                print(f"[datagen w{gpu}] {tid}:{kind} ERROR {type(e).__name__}: {e}", flush=True)
                results.append((tid, kind, "error", 0))
    return results


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _queue_worker(gpu, task_q, result_q, kinds, data_root, counts,
                  teacher_kind, model_teacher):
    """Persistent GPU-pinned datagen worker: pull tasks from a shared queue until
    drained. DYNAMIC load balancing so no worker idles at the tail while a few grind
    the heavy shards. The teacher client + GPU pin are set up ONCE per worker."""
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    from kore.data.schemas import write_jsonl
    from kore.data.teacher import load_env_local, make_teacher
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import get_task

    load_env_local()
    tkw = {"model": model_teacher} if model_teacher else {}
    teacher = make_teacher(teacher_kind, resilient=True, **tkw)

    while True:
        tid = task_q.get()
        if tid is None:  # drain sentinel
            break
        try:
            task = get_task(tid)
            env = KoreEnv(task)
        except Exception as e:  # noqa: BLE001
            print(f"[datagen w{gpu}] {tid}: setup ERROR {type(e).__name__}: {e}", flush=True)
            result_q.put((tid, "*", "error", 0))
            continue
        for kind in kinds:
            if shard_done(data_root, tid, kind):
                print(f"[datagen w{gpu}] {tid}:{kind} skip (resume)", flush=True)
                result_q.put((tid, kind, "skip", 0))
                continue
            try:
                recs = _generate(kind, task, teacher, env, counts)
                out = Path(data_root) / kind / f"{tid}.jsonl"
                out.parent.mkdir(parents=True, exist_ok=True)
                write_jsonl(out, recs)
                print(f"[datagen w{gpu}] {tid}:{kind} -> {len(recs)} records", flush=True)
                result_q.put((tid, kind, "done", len(recs)))
            except Exception as e:  # noqa: BLE001
                print(f"[datagen w{gpu}] {tid}:{kind} ERROR {type(e).__name__}: {e}", flush=True)
                result_q.put((tid, kind, "error", 0))
    result_q.put(None)  # worker-finished sentinel


def run_parallel_datagen(task_ids, kinds, data_root, counts, *, n_workers: int,
                         n_gpus: int, teacher_kind: str = "claude",
                         model_teacher=None, gpu_ids=None, log=print) -> dict:
    """Run datagen for ``kinds`` over ``task_ids`` across GPU-pinned worker processes.
    Resumable (existing shards skipped). Uses a SHARED task queue so every worker
    stays busy pulling the next task until all are done (no tail draining).

    ``gpu_ids``: explicit PHYSICAL GPU ids to pin to (e.g. the free ones on a shared
    node). Datagen is TEACHER-bound, so we run MORE workers than GPUs (teacher-
    parallel), round-robining workers onto the pinned GPUs; verification
    oversubscribes those GPUs but the per-GPU timing lock keeps measurements clean.
    """
    import multiprocessing as mp

    task_ids = list(task_ids)
    n_gpus = max(1, int(n_gpus))
    if gpu_ids:
        gpu_ids = [int(g) for g in gpu_ids]
        nw = max(int(n_workers) if n_workers else len(gpu_ids), len(gpu_ids))
        worker_gpus = [gpu_ids[w % len(gpu_ids)] for w in range(nw)]
        dev_list = gpu_ids
    else:
        nw = max(1, int(n_workers))
        worker_gpus = [w % n_gpus for w in range(nw)]
        dev_list = list(range(n_gpus))

    log(f"parallel datagen: {len(task_ids)} tasks x {list(kinds)} across "
        f"{nw} workers on GPUs {dev_list} (dynamic queue)")
    summary = {"done": 0, "skip": 0, "error": 0, "records": 0, "tasks": len(task_ids)}

    ctxmp = mp.get_context("spawn")
    task_q: "mp.Queue" = ctxmp.Queue()
    result_q: "mp.Queue" = ctxmp.Queue()
    for tid in task_ids:
        task_q.put(tid)
    for _ in range(nw):
        task_q.put(None)  # one drain-sentinel per worker

    procs = [ctxmp.Process(target=_queue_worker,
                           args=(worker_gpus[w], task_q, result_q, list(kinds),
                                 str(data_root), counts, teacher_kind, model_teacher))
             for w in range(nw)]
    for p in procs:
        p.start()

    finished = 0
    while finished < nw:
        item = result_q.get()
        if item is None:
            finished += 1
            continue
        _tid, _kind, status, n = item
        summary[status] = summary.get(status, 0) + 1
        summary["records"] += n
    for p in procs:
        p.join()
    log(f"parallel datagen done: {summary}")
    return summary
