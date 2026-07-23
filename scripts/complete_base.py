"""Resume-safe BASE completion: fill missing/empty repair + groups shards.

Companion to ``deepen_wins.py``. deepen_wins owns ``wins/``; this owns ``repair/``
and ``groups/`` and NEVER reads or writes ``wins/`` - so it can run CONCURRENTLY
with a live deepening without any shard collision.

Resume contract (matches run_campaign's datagen semantics):
  * A shard that already exists NON-EMPTY is skipped (zero teacher calls).
  * A missing OR 0-byte shard is (re)generated. Delete a shard to force regen.
  * Atomic tmp+rename write so a crash never truncates a shard.

Same GPU-pinned spawn-worker pattern as deepen_wins / parallel_datagen (HIP pinned
BEFORE torch import), one teacher stream per worker.

Usage:
  python scripts/complete_base.py --data-root data/b05factory \
     --tasks genb_a,genb_b,... --gpu-ids 0,1,2,3,4,5,6,7 --workers 48 \
     --n-repair 50 --n-parents 20 --k 6 --teacher claude
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

KINDS = ("repair", "groups")


def _shard_done(data_root, task_id: str, kind: str) -> bool:
    p = Path(data_root) / kind / f"{task_id}.jsonl"
    return p.exists() and p.stat().st_size > 0


def complete_one(task_id: str, data_root, n_repair: int, n_parents: int, k: int, teacher):
    """(Re)generate any missing/empty repair+groups shard for ONE task.

    Returns (status, {kind: n_records}). Never touches wins/.
    """
    from kore.data.gen_groups import generate_groups
    from kore.data.gen_repair import generate_repairs
    from kore.data.schemas import write_jsonl
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import get_task

    todo = [kind for kind in KINDS if not _shard_done(data_root, task_id, kind)]
    if not todo:
        return ("skip", {})

    task = get_task(task_id)
    env = KoreEnv(task)
    counts: dict[str, int] = {}
    for kind in todo:
        if kind == "repair":
            recs = generate_repairs(task, teacher, env, n=n_repair)
        else:
            recs = generate_groups(task, teacher, env, n_parents=n_parents, k=k)
        out = Path(data_root) / kind / f"{task_id}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".jsonl.tmp")
        write_jsonl(tmp, recs)
        os.replace(tmp, out)
        counts[kind] = len(recs)
    return ("done", counts)


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

    done = finished = skipped = rep_tot = grp_tot = 0
    while finished < n_workers:
        item = result_q.get()
        if item is None:
            finished += 1
            continue
        _tid, st, counts = item
        done += 1
        if st == "skip":
            skipped += 1
        rep_tot += counts.get("repair", 0)
        grp_tot += counts.get("groups", 0)
        if done % 10 == 0:
            print(f"[base] progress {done}/{len(tasks)} (+{rep_tot} repair, +{grp_tot} groups recs, "
                  f"{skipped} already-done)", flush=True)
    for p in procs:
        p.join()
    print(f"[base] COMPLETE: {done} tasks, +{rep_tot} repair recs, +{grp_tot} groups recs, "
          f"{skipped} skipped", flush=True)


if __name__ == "__main__":
    main()
