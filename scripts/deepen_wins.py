"""Additive, resume-safe GOLD-WINS deepening.

Bring each task's ``wins/<task>.jsonl`` shard up to ``--target`` DISTINCT verified
wins WITHOUT redoing repair/groups and WITHOUT losing or regenerating any existing
win.

Key facts (see kore.data.gen_wins.generate_wins):
  * generate_wins runs ONE evolve trajectory and returns >=1 WinRecord (0 or 1).
    So N distinct wins == N successful, *independent* trajectories - NOT a bigger
    ``gens`` (that only deepens a single trajectory).
  * The teacher samples at temperature 0.7, so independent trajectories diverge;
    we dedup by ``final_source`` so an identical kernel is never stored twice.

Guarantees (the "no wasted effort" contract):
  * Only ``wins/`` shards are ever read/written - repair/groups are never touched.
  * Existing wins are READ and PRESERVED; new wins are APPENDED (atomic tmp+rename,
    so a crash never truncates a shard).
  * A task already at >=target is SKIPPED with ZERO teacher calls (re-runnable).
  * Every distinct win is checkpointed immediately via atomic tmp+rename. A crash
    or burst preemption loses only the currently executing trajectory; a re-run
    resumes from all previously completed trajectories.

Parallel across GPU-pinned spawn workers (HIP pinned BEFORE torch import), one
teacher stream each - the same proven pattern as kore.data.parallel_datagen.

Usage:
  python scripts/deepen_wins.py --data-root data/b05factory \
     --tasks genb_a,genb_b,... --gpu-ids 0,1,2,3,4,5,6,7 --workers 48 \
     --target 3 --gens 8 --teacher claude
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def _src_hash(s) -> str:
    return hashlib.sha1((s or "").encode("utf-8", "ignore")).hexdigest()


def _load_existing(path: Path):
    """Return (existing_dict_records, set_of_final_source_hashes) for a wins shard."""
    from kore.data.schemas import read_jsonl
    if not path.exists() or path.stat().st_size == 0:
        return [], set()
    try:
        recs = read_jsonl(path, typed=False)
    except Exception:
        return [], set()
    distinct = []
    seen = set()
    for record in recs:
        if not isinstance(record, dict):
            continue
        source = str(record.get("final_source", "") or "").strip()
        if not source:
            continue
        key = _src_hash(source)
        if key in seen:
            continue
        seen.add(key)
        distinct.append(record)
    return distinct, seen


def _checkpoint(path: Path, existing: list, added: list) -> None:
    """Atomically persist all wins completed so far for this task."""
    from kore.data.schemas import write_jsonl

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    write_jsonl(tmp, list(existing) + list(added))
    os.replace(tmp, path)


def deepen_one(task_id: str, data_root, target: int, gens: int, teacher, cfg):
    """Additively top up ONE task's wins to `target`. Returns (status, have, added, attempts)."""
    from kore.data.amd_knowledge import ExperienceLedger
    from kore.data.gen_wins import generate_wins
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import get_task

    path = Path(data_root) / "wins" / f"{task_id}.jsonl"
    existing, seen = _load_existing(path)
    have = len(existing)
    if have >= target:
        return ("skip", have, 0, 0)

    task = get_task(task_id)
    env = KoreEnv(task)
    need = target - have
    # Oversample: some trajectories yield no net win / a duplicate. Bound the teacher
    # spend so a stubborn task can never run away.
    max_attempts = max(need * 3, need + 2)
    added = []
    attempts = 0
    # Tier 3: ONE experience ledger shared across every trajectory for this task, so
    # the do-NOT-repeat constraints learned in attempt 1 steer attempts 2..N (the N
    # trajectories no longer re-walk the same dead-ends - the point of deepening).
    ledger = ExperienceLedger()
    while (have + len(added)) < target and attempts < max_attempts:
        attempts += 1
        try:
            ws = generate_wins(task, teacher, env, gens=gens, cfg=cfg, ledger=ledger)
        except Exception as e:  # noqa: BLE001 - one bad trajectory never aborts the task
            print(f"[deepen] {task_id} attempt {attempts}: ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        if not ws:
            continue
        w = ws[0]
        h = _src_hash(getattr(w, "final_source", ""))
        if h in seen:
            continue  # identical kernel -> don't store a duplicate
        seen.add(h)
        added.append(w)
        # Burst jobs can be preempted at any moment. Persist EACH completed
        # trajectory now, not after all 2-9 attempts for the task.
        _checkpoint(path, existing, added)
    status = "done" if (have + len(added)) >= target else "partial"
    return (status, have, len(added), attempts)


def _worker(payload: dict):
    gpu = str(payload["gpu_id"])
    # Pin the GPU BEFORE any torch import (KoreEnv's verifier subprocesses inherit it).
    os.environ["HIP_VISIBLE_DEVICES"] = gpu
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    from kore.config import CONFIG
    from kore.data.teacher import load_env_local, make_teacher

    load_env_local()
    tkw = {"model": payload["model_teacher"]} if payload.get("model_teacher") else {}
    teacher = make_teacher(payload["teacher_kind"], resilient=True, **tkw)

    dr, target, gens = payload["data_root"], payload["target"], payload["gens"]
    task_q, result_q = payload["task_q"], payload["result_q"]
    while True:
        tid = task_q.get()
        if tid is None:
            break
        try:
            st, have, added, att = deepen_one(tid, dr, target, gens, teacher, CONFIG)
            print(f"[deepen w{gpu}] {tid}: {st} have={have} added={added} attempts={att}", flush=True)
            result_q.put((tid, st, added))
        except Exception as e:  # noqa: BLE001
            print(f"[deepen w{gpu}] {tid}: FATAL {type(e).__name__}: {str(e)[:160]}", flush=True)
            result_q.put((tid, "error", 0))
    result_q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tasks", default="", help="comma list; empty => all train tasks")
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--target", type=int, default=3)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--teacher", default="claude")
    ap.add_argument("--model-teacher", default=None)
    a = ap.parse_args()

    import multiprocessing as mp
    import queue
    from kore.tasks.registry import train_tasks

    gpu_ids = [int(x) for x in a.gpu_ids.split(",") if x != ""]
    if a.tasks.strip():
        tasks = [t for t in a.tasks.split(",") if t]
    else:
        tasks = [t.task_id for t in train_tasks()]  # excludes held-out by construction
    n_workers = a.workers or (len(gpu_ids) * 4)

    print(f"[deepen] START tasks={len(tasks)} target={a.target} gens={a.gens} "
          f"gpus={gpu_ids} workers={n_workers} data_root={a.data_root}", flush=True)

    ctx = mp.get_context("spawn")
    task_q, result_q = ctx.Queue(), ctx.Queue()
    for t in tasks:
        task_q.put(t)
    for _ in range(n_workers):
        task_q.put(None)

    procs = []
    for i in range(n_workers):
        payload = dict(gpu_id=gpu_ids[i % len(gpu_ids)], data_root=a.data_root,
                       target=a.target, gens=a.gens, teacher_kind=a.teacher,
                       model_teacher=a.model_teacher, task_q=task_q, result_q=result_q)
        p = ctx.Process(target=_worker, args=(payload,))
        p.start()
        procs.append(p)

    done = total_added = finished = skipped = partials = errors = 0
    worker_failures = 0
    while finished < n_workers:
        try:
            item = result_q.get(timeout=30)
        except queue.Empty:
            failed = [p for p in procs if p.exitcode not in (None, 0)]
            if not failed:
                continue
            worker_failures += len(failed)
            print(
                "[deepen] FATAL worker exit(s): "
                + ", ".join(f"pid={p.pid} rc={p.exitcode}" for p in failed),
                flush=True,
            )
            for p in procs:
                if p.is_alive():
                    p.terminate()
            break
        if item is None:
            finished += 1
            continue
        _tid, st, added = item
        done += 1
        total_added += (added or 0)
        if st == "skip":
            skipped += 1
        elif st == "partial":
            partials += 1
        elif st == "error":
            errors += 1
        if done % 25 == 0:
            print(f"[deepen] progress {done}/{len(tasks)} (+{total_added} wins, {skipped} already-at-target)", flush=True)
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    print(
        f"[deepen] COMPLETE: {done} tasks, +{total_added} new wins, "
        f"{skipped} skipped, {partials} partial, {errors} errors, "
        f"{worker_failures} worker failures",
        flush=True,
    )
    if errors or worker_failures:
        return 2
    if partials:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
