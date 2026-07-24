"""Resume-safe BASE completion: fill missing/empty repair + groups shards.

Companion to ``deepen_wins.py``. deepen_wins owns ``wins/``; this owns ``repair/``
and ``groups/`` and NEVER reads or writes ``wins/`` - so it can run CONCURRENTLY
with a live deepening without any shard collision.

Resume contract (matches run_campaign's datagen semantics):
  * A shard that already exists NON-EMPTY is skipped (zero teacher calls).
  * A missing OR 0-byte shard is (re)generated. Delete a shard to force regen.
  * Every accepted repair/group record is checkpointed immediately with atomic
    tmp+rename. An ``.inprogress`` marker lets a preempted burst job continue
    filling the target rather than treating its partial shard as complete.

Same GPU-pinned spawn-worker pattern as deepen_wins / parallel_datagen (HIP pinned
BEFORE torch import), one teacher stream per worker.

Usage:
  python scripts/complete_base.py --data-root data/b05factory \
     --tasks genb_a,genb_b,... --gpu-ids 0,1,2,3,4,5,6,7 --workers 48 \
     --n-repair 50 --n-parents 20 --k 6 --teacher claude
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path

KINDS = ("repair", "groups")


def _shard_done(data_root, task_id: str, kind: str) -> bool:
    p = Path(data_root) / kind / f"{task_id}.jsonl"
    return p.exists() and p.stat().st_size > 0


def _marker(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".inprogress")


def _record_dict(record):
    if isinstance(record, dict):
        return record
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if is_dataclass(record):
        return asdict(record)
    raise TypeError(f"unsupported record type: {type(record).__name__}")


def _record_key(record) -> str:
    return json.dumps(
        _record_dict(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _load_records(path: Path) -> list:
    from kore.data.schemas import read_jsonl

    if not path.exists() or path.stat().st_size == 0:
        return []
    return read_jsonl(path, typed=False)


def _checkpoint(path: Path, records: list) -> None:
    from kore.data.schemas import write_jsonl

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    write_jsonl(tmp, records)
    os.replace(tmp, path)


def _checkpoint_collector(path: Path, existing: list):
    """Return (callback, state); callback atomically persists each unique record."""
    records = list(existing)
    seen = {_record_key(record) for record in records}
    state = {"records": records, "added": 0}

    def collect(record):
        key = _record_key(record)
        if key in seen:
            return
        seen.add(key)
        records.append(record)
        state["added"] += 1
        _checkpoint(path, records)

    return collect, state


def complete_one(task_id: str, data_root, n_repair: int, n_parents: int, k: int, teacher):
    """(Re)generate any missing/empty repair+groups shard for ONE task.

    Returns (status, {kind: n_records}). Never touches wins/.
    """
    from kore.data.gen_groups import generate_groups
    from kore.data.gen_repair import generate_repairs
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import get_task

    todo = []
    for kind in KINDS:
        path = Path(data_root) / kind / f"{task_id}.jsonl"
        if not _shard_done(data_root, task_id, kind) or _marker(path).exists():
            todo.append(kind)
    if not todo:
        return ("skip", {})

    task = get_task(task_id)
    env = KoreEnv(task)
    counts: dict[str, int] = {}
    for kind in todo:
        out = Path(data_root) / kind / f"{task_id}.jsonl"
        existing = _load_records(out)
        target = n_repair if kind == "repair" else n_parents
        remaining = max(0, target - len(existing))
        if remaining == 0:
            _marker(out).unlink(missing_ok=True)
            counts[kind] = 0
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        _marker(out).write_text(
            json.dumps({"target": target, "existing": len(existing)}) + "\n"
        )
        collect, state = _checkpoint_collector(out, existing)
        if kind == "repair":
            recs = generate_repairs(
                task, teacher, env, n=remaining, seed=len(existing), on_record=collect
            )
        else:
            recs = generate_groups(
                task, teacher, env, n_parents=remaining, k=k,
                seed=len(existing), on_record=collect,
            )
        # Defense-in-depth for alternate/custom generators that return records but
        # do not invoke the optional callback.
        for record in recs:
            collect(record)
        _marker(out).unlink(missing_ok=True)
        counts[kind] = state["added"]
    status = "done" if all(_shard_done(data_root, task_id, k) for k in KINDS) else "partial"
    return (status, counts)


def _worker(payload: dict):
    gpu = str(payload["gpu_id"])
    # Pin the GPU BEFORE any torch import (KoreEnv's verifier subprocesses inherit it).
    os.environ["HIP_VISIBLE_DEVICES"] = gpu
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    from kore.data.teacher import load_env_local, make_teacher

    load_env_local()
    tkw = {"model": payload["model_teacher"]} if payload.get("model_teacher") else {}
    teacher = make_teacher(payload["teacher_kind"], resilient=True, **tkw)

    dr = payload["data_root"]
    nr, npar, k = payload["n_repair"], payload["n_parents"], payload["k"]
    task_q, result_q = payload["task_q"], payload["result_q"]
    while True:
        tid = task_q.get()
        if tid is None:
            break
        try:
            st, counts = complete_one(tid, dr, nr, npar, k, teacher)
            print(f"[base w{gpu}] {tid}: {st} {counts}", flush=True)
            result_q.put((tid, st, counts))
        except Exception as e:  # noqa: BLE001 - one bad task never aborts the pass
            print(f"[base w{gpu}] {tid}: FATAL {type(e).__name__}: {str(e)[:160]}", flush=True)
            result_q.put((tid, "error", {}))
    result_q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tasks", default="", help="comma list; empty => all train tasks")
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--n-repair", type=int, default=50)
    ap.add_argument("--n-parents", type=int, default=20)
    ap.add_argument("--k", type=int, default=6)
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
        tasks = [t.task_id for t in train_tasks()]
    n_workers = a.workers or (len(gpu_ids) * 4)

    print(f"[base] START tasks={len(tasks)} gpus={gpu_ids} workers={n_workers} "
          f"n_repair={a.n_repair} n_parents={a.n_parents} k={a.k} data_root={a.data_root}",
          flush=True)

    ctx = mp.get_context("spawn")
    task_q, result_q = ctx.Queue(), ctx.Queue()
    for t in tasks:
        task_q.put(t)
    for _ in range(n_workers):
        task_q.put(None)

    procs = []
    for i in range(n_workers):
        payload = dict(gpu_id=gpu_ids[i % len(gpu_ids)], data_root=a.data_root,
                       n_repair=a.n_repair, n_parents=a.n_parents, k=a.k,
                       teacher_kind=a.teacher, model_teacher=a.model_teacher,
                       task_q=task_q, result_q=result_q)
        p = ctx.Process(target=_worker, args=(payload,))
        p.start()
        procs.append(p)

    done = finished = skipped = partials = errors = rep_tot = grp_tot = 0
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
                "[base] FATAL worker exit(s): "
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
        _tid, st, counts = item
        done += 1
        if st == "skip":
            skipped += 1
        elif st == "partial":
            partials += 1
        elif st == "error":
            errors += 1
        rep_tot += counts.get("repair", 0)
        grp_tot += counts.get("groups", 0)
        if done % 10 == 0:
            print(f"[base] progress {done}/{len(tasks)} (+{rep_tot} repair, +{grp_tot} groups recs, "
                  f"{skipped} already-done)", flush=True)
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    print(
        f"[base] COMPLETE: {done} tasks, +{rep_tot} repair recs, "
        f"+{grp_tot} groups recs, {skipped} skipped, {partials} partial, "
        f"{errors} errors, {worker_failures} worker failures",
        flush=True,
    )
    if errors or worker_failures:
        return 2
    if partials:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
