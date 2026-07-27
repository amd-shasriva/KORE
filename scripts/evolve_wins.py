"""Parallel, resume-safe EVOLUTIONARY win generation with per-task K-concurrency.

Runs :func:`kore.data.evolve.evolve_task` (D-MAB bandit + MAP-Elites islands +
value-prefilter) across many GPU-pinned workers to MANUFACTURE verified,
vendor-beating wins for the hard / under-target tail that one-shot teacher datagen
can't crack. Mirrors ``deepen_wins.py`` (same coordinator/worker/persist shape) so
the SPUR array + supervisor drive it identically; the differences are:

  * generator = ``evolve_task`` (an island search), so ONE claim yields MANY wins,
    not 0/1. The coordinator counts ``n_added`` per run.
  * wins/groups are written to SEPARATE ``<task>.evolve.jsonl`` shards (the build
    stage folds them in via its existing glob + dedup), so the teacher-generated
    ``<task>.jsonl`` is never touched.
  * a task's win count is the COMBINED distinct total across ``<task>.jsonl`` and
    ``<task>.evolve.jsonl`` - evolve tops the combined total up to ``--target``.

Guarantees (same "no wasted effort" contract as deepen):
  * Existing wins (teacher + prior evolve) are READ and PRESERVED; new distinct
    evolve wins are APPENDED (atomic tmp+rename); duplicates (by final_source) are
    dropped, so re-running is free for satisfied tasks.
  * A task already at >=target (combined) with >=1 evolve run is SKIPPED.
  * Every distinct win is checkpointed immediately; a crash / burst preemption
    loses only the currently executing island run.

Usage:
  python scripts/evolve_wins.py --data-root data/b05factory \
     --tasks genb_a,genb_b,... --gpu-ids 0,1,2,3,4,5,6,7 --workers 32 \
     --target 3 --generations 8 --teacher claude
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def _src_hash(s) -> str:
    return hashlib.sha1(str(s or "").strip().encode("utf-8", "ignore")).hexdigest()


def _final_source(record) -> str:
    if isinstance(record, dict):
        value = record.get("final_source", "")
    else:
        value = getattr(record, "final_source", "")
    return str(value or "").strip()


def _rec_dict(record) -> dict:
    return record.to_dict() if hasattr(record, "to_dict") else dict(record)


def _atomic_write_jsonl(path: Path, records: list) -> None:
    from kore.data.schemas import write_jsonl

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write_jsonl(tmp, records)
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        tmp.unlink(missing_ok=True)


def _read_records(path: Path) -> list:
    if not path.exists() or path.stat().st_size == 0:
        return []
    recs = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"invalid JSONL record {path}:{line_no}: expected object")
            recs.append(record)
    return recs


def _wins_shard(data_root: Path, task_id: str) -> Path:
    return Path(data_root) / "wins" / f"{task_id}.jsonl"


def _evolve_wins_shard(data_root: Path, task_id: str) -> Path:
    return Path(data_root) / "wins" / f"{task_id}.evolve.jsonl"


def _evolve_groups_shard(data_root: Path, task_id: str) -> Path:
    return Path(data_root) / "groups" / f"{task_id}.evolve.jsonl"


def _lock(data_root: Path, task_id: str) -> Path:
    return Path(data_root) / ".locks" / "evolve" / f"{task_id}.lock"


def _combined_seen(data_root: Path, task_id: str):
    """(distinct evolve records, set of ALL final_source hashes across both shards)."""
    seen: set[str] = set()
    for rec in _read_records(_wins_shard(data_root, task_id)):
        src = _final_source(rec)
        if src:
            seen.add(_src_hash(src))
    evolve_distinct = []
    ev_seen: set[str] = set()
    for rec in _read_records(_evolve_wins_shard(data_root, task_id)):
        src = _final_source(rec)
        if not src:
            continue
        h = _src_hash(src)
        if h in ev_seen:
            continue
        ev_seen.add(h)
        evolve_distinct.append(rec)
    return evolve_distinct, (seen | ev_seen)


def combined_have(data_root: Path, task_id: str) -> int:
    _distinct, seen = _combined_seen(data_root, task_id)
    return len(seen)


def _persist_evolve(data_root: Path, task_id: str, wins: list, groups: list, target: int) -> int:
    """Append new distinct evolve wins + groups under a short per-task lock.

    Returns the number of NEWLY added distinct wins. The lock covers only the
    read -> dedup -> atomic-append critical section, never the island run, so K
    workers can evolve the same task concurrently without racing.
    """
    lock_path = _lock(data_root, task_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        evolve_distinct, seen = _combined_seen(data_root, task_id)
        have = len(seen)
        new_wins = list(evolve_distinct)
        for w in wins:
            if have + added >= target:
                break
            src = _final_source(w)
            if not src:
                continue
            h = _src_hash(src)
            if h in seen:
                continue
            seen.add(h)
            new_wins.append(_rec_dict(w))
            added += 1
        if added:
            _atomic_write_jsonl(_evolve_wins_shard(data_root, task_id), new_wins)
        # Groups: merge distinct ranked-group records (dedup by candidate-set hash).
        if groups:
            gpath = _evolve_groups_shard(data_root, task_id)
            existing = _read_records(gpath)
            gseen = {_src_hash(json.dumps(g, sort_keys=True)) for g in existing}
            merged = list(existing)
            changed = False
            for g in groups:
                gd = _rec_dict(g)
                gh = _src_hash(json.dumps(gd, sort_keys=True))
                if gh in gseen:
                    continue
                gseen.add(gh)
                merged.append(gd)
                changed = True
            if changed:
                _atomic_write_jsonl(gpath, merged)
    return added


class EvolveCoordinator:
    """Per-task island-run scheduler shared across spawn workers.

    A task is SATISFIED when it has >=target combined wins AND has had >=1 evolve
    run (so even an at-target frontier task gets one push to add higher-quality,
    vendor-beating wins). Hands run-tickets to the most-under-target incomplete
    task, bounding runs-in-flight (``inflight_cap``) and total runs
    (``max_runs``) per task so one impossible task cannot absorb the whole pool.
    """

    def __init__(self, manager, tasks_have: dict, target: int, max_runs: int, inflight_cap: int):
        self.target = int(target)
        self.max_runs = max(1, int(max_runs))
        self.inflight_cap = max(1, int(inflight_cap))
        self._order = list(tasks_have)
        self.lock = manager.Lock()
        self.have = manager.dict({t: int(h) for t, h in tasks_have.items()})
        self.runs = manager.dict({t: 0 for t in tasks_have})
        self.inflight = manager.dict({t: 0 for t in tasks_have})

    def _satisfied(self, t) -> bool:
        return self.have[t] >= self.target and self.runs[t] >= 1

    def claim(self):
        """Returns (task_id, 'GO') | (None, 'WAIT') | (None, 'DONE')."""
        with self.lock:
            any_incomplete = False
            any_inflight = False
            best = None
            best_need = -1
            for t in self._order:
                if self._satisfied(t):
                    continue
                any_incomplete = True
                if self.inflight[t] > 0:
                    any_inflight = True
                if self.runs[t] >= self.max_runs:
                    continue  # budget exhausted; only awaiting in-flight results
                if self.inflight[t] >= self.inflight_cap:
                    continue
                need = self.target - self.have[t]
                if need > best_need:
                    best_need = need
                    best = t
            if best is not None:
                self.runs[best] = self.runs[best] + 1
                self.inflight[best] = self.inflight[best] + 1
                return (best, "GO")
            if not any_incomplete:
                return (None, "DONE")
            if any_inflight:
                return (None, "WAIT")
            return (None, "DONE")

    def record(self, task_id: str, added: int):
        with self.lock:
            self.inflight[task_id] = max(0, self.inflight[task_id] - 1)
            if added:
                self.have[task_id] = self.have[task_id] + int(added)

    def snapshot_have(self) -> dict:
        with self.lock:
            return dict(self.have)


_MAX_IDLE_POLLS = 240  # ~2 min of empty WAITs before a worker gives up (safety)


def _worker(payload: dict):
    gpu = str(payload["gpu_id"])
    os.environ["HIP_VISIBLE_DEVICES"] = gpu
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    from kore.config import CONFIG  # noqa: F401  (import parity w/ deepen; env side effects)
    from kore.data.evolve import EvolveConfig, evolve_task
    from kore.data.teacher import load_env_local, make_teacher
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import get_task

    load_env_local()
    tkw = {"model": payload["model_teacher"]} if payload.get("model_teacher") else {}
    teacher = make_teacher(payload["teacher_kind"], resilient=True, **tkw)

    coord = payload["coord"]
    data_root = Path(payload["data_root"])
    target = payload["target"]
    gens = payload["gens"]
    result_q = payload["result_q"]
    seed_base = int(payload.get("seed_base", 0))

    envs: dict = {}
    added_total = 0
    idle = 0
    run_ix = 0
    while True:
        tid, status = coord.claim()
        if status == "DONE":
            break
        if status == "WAIT":
            idle += 1
            if idle > _MAX_IDLE_POLLS:
                break
            time.sleep(0.5)
            continue
        idle = 0
        added = 0
        run_ix += 1
        try:
            task = get_task(tid)
            env = envs.get(tid)
            if env is None:
                env = envs[tid] = KoreEnv(task)
            # Vary seed per (worker, run) so concurrent islands on one task diverge.
            cfg = EvolveConfig(seed=seed_base + run_ix * 101 + (int(gpu) if gpu.isdigit() else 0))
            result = evolve_task(task, teacher, env, generations=gens, cfg=cfg)
            wins = list(getattr(result, "wins", []) or [])
            groups = list(getattr(result, "groups", []) or [])
            if wins or groups:
                added = _persist_evolve(data_root, tid, wins, groups, target)
                added_total += added
            best = None
            try:
                best = (result.stats or {}).get("best_speedup")
            except Exception:
                pass
            print(f"[evolve w{gpu}] {tid}: +{added} wins (raw={len(wins)}) "
                  f"have={coord.have.get(tid)} best_speedup={best}", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad run never aborts the pool
            print(f"[evolve w{gpu}] {tid}: ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
        finally:
            coord.record(tid, added)
    result_q.put(("worker_done", gpu, added_total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tasks", default="", help="comma list; empty => all train tasks under target")
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--target", type=int, default=3)
    ap.add_argument("--generations", type=int,
                    default=int(os.environ.get("KORE_EVOLVE_GENERATIONS", "8")))
    ap.add_argument("--teacher", default="claude")
    ap.add_argument("--model-teacher", default=None)
    ap.add_argument("--max-runs", type=int,
                    default=int(os.environ.get("KORE_EVOLVE_MAX_RUNS", "4")),
                    help="max island runs per task before giving up")
    ap.add_argument("--inflight", type=int,
                    default=int(os.environ.get("KORE_EVOLVE_INFLIGHT", "2")),
                    help="max concurrent island runs per task")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import multiprocessing as mp
    import queue
    from kore.tasks.registry import train_tasks

    gpu_ids = [int(x) for x in a.gpu_ids.split(",") if x != ""]
    if not gpu_ids:
        ap.error("--gpu-ids must contain at least one GPU")
    if a.target < 1:
        ap.error("--target must be positive")
    if a.generations < 1:
        ap.error("--generations must be positive")

    if a.tasks.strip():
        tasks = list(dict.fromkeys(t for t in a.tasks.split(",") if t))
    else:
        tasks = [t.task_id for t in train_tasks()]

    # Defense-in-depth: never evolve held-out/eval tasks (prevents eval leakage).
    from kore.tasks.registry import is_heldout as _is_heldout, get_task as _get_task, task_ids as _task_ids
    _known = set(_task_ids())
    kept = []
    for t in tasks:
        try:
            if t in _known and _is_heldout(_get_task(t)):
                print(f"[evolve] SKIP held-out task {t}", flush=True)
                continue
        except Exception:
            pass
        kept.append(t)
    tasks = kept
    if not tasks:
        print("[evolve] COMPLETE: no train tasks after held-out filter", flush=True)
        return 0

    # Resume-safe: seed combined win counts from disk; satisfied tasks cost zero.
    root = Path(a.data_root)
    tasks_have = {t: combined_have(root, t) for t in tasks}
    n_workers = a.workers or (len(gpu_ids) * 4)
    if n_workers < 1:
        ap.error("--workers must be positive")

    print(f"[evolve] START tasks={len(tasks)} target={a.target} gens={a.generations} "
          f"gpus={gpu_ids} workers={n_workers} max_runs={a.max_runs} inflight={a.inflight} "
          f"data_root={a.data_root}", flush=True)

    ctx = mp.get_context("spawn")
    manager = mp.Manager()
    coord = EvolveCoordinator(manager, tasks_have, a.target, a.max_runs, a.inflight)
    result_q = manager.Queue()

    procs = []
    for i in range(n_workers):
        payload = dict(gpu_id=gpu_ids[i % len(gpu_ids)], data_root=a.data_root,
                       target=a.target, gens=a.generations, teacher_kind=a.teacher,
                       model_teacher=a.model_teacher, coord=coord, result_q=result_q,
                       seed_base=a.seed + i * 1009)
        p = ctx.Process(target=_worker, args=(payload,))
        p.start()
        procs.append(p)

    finished = 0
    total_added = 0
    worker_failures = 0
    last_report = time.monotonic()
    while finished < n_workers:
        try:
            item = result_q.get(timeout=30)
        except queue.Empty:
            failed = [p for p in procs if p.exitcode not in (None, 0)]
            if failed:
                worker_failures += len(failed)
                print("[evolve] FATAL worker exit(s): "
                      + ", ".join(f"pid={p.pid} rc={p.exitcode}" for p in failed), flush=True)
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                break
            if time.monotonic() - last_report > 60:
                have = coord.snapshot_have()
                done_n = sum(1 for t in tasks if have.get(t, 0) >= a.target)
                print(f"[evolve] progress {done_n}/{len(tasks)} tasks at target "
                      f"(+{total_added} new evolve wins)", flush=True)
                last_report = time.monotonic()
            continue
        if item and item[0] == "worker_done":
            finished += 1
            total_added += (item[2] or 0)

    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    have = coord.snapshot_have()
    done_n = sum(1 for t in tasks if have.get(t, 0) >= a.target)
    partials = [t for t in tasks if 0 < have.get(t, 0) < a.target]
    missing = [t for t in tasks if have.get(t, 0) == 0]
    print(f"[evolve] COMPLETE: {done_n}/{len(tasks)} tasks at target, +{total_added} new "
          f"evolve wins, {len(partials)} partial, {len(missing)} still-empty, "
          f"{worker_failures} worker failures", flush=True)

    if worker_failures:
        return 2
    if partials or missing:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
