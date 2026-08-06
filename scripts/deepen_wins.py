"""Additive, resume-safe GOLD-WINS deepening with per-task K-concurrency.

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
    or burst preemption loses only the currently executing trajectory.

Concurrency model (per-task K-concurrency)
------------------------------------------
Work is scheduled per-TRAJECTORY, not per-task. Many GPU-pinned spawn workers
cooperate on the SAME task concurrently: each runs an independent evolve
trajectory (the 18-90s teacher call is I/O-bound, so trajectories overlap and the
teacher - the real bottleneck - stays saturated). The expensive trajectory runs
LOCK-FREE; only the fast win-persist (read -> dedup -> atomic append) takes a
short per-task file lock, so concurrent workers on one shard never race.

This removes the old ``min(workers, len(tasks))`` ceiling + whole-task exclusive
lock: with few remaining tasks the full worker pool still saturates the teacher
instead of idling one-worker-per-task. Per-task teacher spend stays bounded by the
same oversample budget as the original sequential loop
(``max(need*3, need+2)`` total attempts per task); ``KORE_DEEP_OVERSAMPLE``
(default 3) caps how many trajectories may be in flight for one task at once.

Usage:
  python scripts/deepen_wins.py --data-root data/b05factory \
     --tasks genb_a,genb_b,... --gpu-ids 0,1,2,3,4,5,6,7 --workers 48 \
     --target 3 --gens 8 --teacher claude
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pathlib
from pathlib import Path
import tempfile
import time


def _src_hash(s) -> str:
    source = str(s or "").strip()
    return hashlib.sha1(source.encode("utf-8", "ignore")).hexdigest()


def _final_source(record) -> str:
    """final_source of a WinRecord OR a plain dict record."""
    if isinstance(record, dict):
        value = record.get("final_source", "")
    else:
        value = getattr(record, "final_source", "")
    return str(value or "").strip()


def _atomic_write_jsonl(path: Path, records: list) -> None:
    from kore.data.schemas import write_jsonl

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
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


def _load_existing(path: Path):
    """Return (distinct_dict_records, set_of_final_source_hashes) for a wins shard."""
    if not path.exists() or path.stat().st_size == 0:
        return [], set()
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
                raise RuntimeError(
                    f"invalid JSONL record {path}:{line_no}: expected object"
                )
            recs.append(record)
    distinct = []
    seen = set()
    for record in recs:
        source = _final_source(record)
        if not source:
            continue
        key = _src_hash(source)
        if key in seen:
            continue
        seen.add(key)
        distinct.append(record)
    return distinct, seen


def _wins_shard(data_root: Path, task_id: str) -> Path:
    return Path(data_root) / "wins" / f"{task_id}.jsonl"


def _wins_lock(data_root: Path, task_id: str) -> Path:
    return Path(data_root) / ".locks" / "deepen" / f"{task_id}.lock"


def _persist_win(path: Path, lock_path: Path, record, target: int):
    """Append ONE candidate win under a short exclusive per-task file lock.

    The lock is held ONLY for the read -> dedup -> atomic-append critical section
    (milliseconds), never for the trajectory itself, so K workers can generate for
    the same task concurrently and still never lose or duplicate a win.

    Returns (status, have_after) where status is 'added' | 'dup' | 'full' | 'empty'.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing, seen = _load_existing(path)
        have = len(existing)
        if have >= target:
            return ("full", have)
        source = _final_source(record)
        if not source:
            return ("empty", have)
        if _src_hash(source) in seen:
            return ("dup", have)
        _atomic_write_jsonl(path, list(existing) + [record])
        return ("added", have + 1)
        # flock released when lock_file closes


def deepen_one(task_id: str, data_root, target: int, gens: int, teacher, cfg):
    """Sequential single-task additive top-up (compatibility API).

    Runs independent evolve trajectories for ONE task until it reaches ``target``
    distinct wins or exhausts the oversample budget, checkpointing EACH distinct
    win immediately via the collision-safe :func:`_persist_win` (so a crash or
    burst preemption between trajectories never loses a completed win). The
    parallel campaign path uses :class:`TaskCoordinator` + :func:`_worker`; this
    helper is kept for single-task callers and the preemption/resume contract.

    Returns ``(status, have_before, added, attempts)``.
    """
    from kore.data.amd_knowledge import ExperienceLedger
    from kore.data.gen_wins import generate_wins
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import get_task

    root = Path(data_root)
    path = _wins_shard(root, task_id)
    lock = _wins_lock(root, task_id)
    existing, _seen = _load_existing(path)
    have = len(existing)
    if have >= target:
        return ("skip", have, 0, 0)

    task = get_task(task_id)
    env = KoreEnv(task)
    need = target - have
    max_attempts = max(need * 3, need + 2)
    ledger = ExperienceLedger()
    added = 0
    attempts = 0
    while (have + added) < target and attempts < max_attempts:
        attempts += 1
        try:
            ws = generate_wins(task, teacher, env, gens=gens, cfg=cfg, ledger=ledger)
        except Exception as e:  # noqa: BLE001 - one bad trajectory never aborts the task
            print(f"[deepen] {task_id} attempt {attempts}: ERROR "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        if not ws:
            continue
        st, _new_have = _persist_win(path, lock, ws[0], target)
        if st == "added":
            added += 1
    status = "done" if (have + added) >= target else "partial"
    return (status, have, added, attempts)


class TaskCoordinator:
    """Manager-backed per-task trajectory scheduler shared across spawn workers.

    Bounds, per task, the TOTAL trajectories launched to the same oversample
    budget the old sequential loop used (``max(need*3, need+2)``) and the number
    IN FLIGHT at once to ``need*oversample``. Hands out trajectory tickets to
    whichever incomplete task currently has the most uncovered need, so a small
    set of remaining tasks still keeps the whole worker pool busy.
    """

    def __init__(self, manager, tasks_have: dict, target: int, oversample: int):
        self.target = int(target)
        self.oversample = max(1, int(oversample))
        self._order = list(tasks_have)
        self.lock = manager.Lock()
        self.have = manager.dict({t: int(h) for t, h in tasks_have.items()})
        self.inflight = manager.dict({t: 0 for t in tasks_have})
        self.attempts = manager.dict({t: 0 for t in tasks_have})
        self.max_attempts = manager.dict({
            t: (max((self.target - int(h)) * 3, (self.target - int(h)) + 2)
                if int(h) < self.target else 0)
            for t, h in tasks_have.items()
        })

    def claim(self):
        """Reserve a trajectory ticket.

        Returns (task_id, 'GO') | (None, 'WAIT') | (None, 'DONE').
        WAIT means no open slot right now but an in-flight trajectory may fail and
        reopen one. DONE means every task is complete or has exhausted its budget.
        """
        with self.lock:
            any_incomplete = False
            any_inflight = False
            best = None
            best_slack = 0
            for t in self._order:
                have = self.have[t]
                if have >= self.target:
                    continue
                any_incomplete = True
                infl = self.inflight[t]
                if infl > 0:
                    any_inflight = True
                att = self.attempts[t]
                maxa = self.max_attempts[t]
                if att >= maxa:
                    continue  # budget exhausted; only awaiting in-flight results
                need = self.target - have
                cap = max(1, min(need * self.oversample, maxa - att))
                if infl >= cap:
                    continue
                slack = need * self.oversample - infl
                if slack > best_slack:
                    best_slack = slack
                    best = t
            if best is not None:
                self.attempts[best] = self.attempts[best] + 1
                self.inflight[best] = self.inflight[best] + 1
                return (best, "GO")
            if not any_incomplete:
                return (None, "DONE")
            if any_inflight:
                return (None, "WAIT")
            return (None, "DONE")

    def record(self, task_id: str, added: bool):
        with self.lock:
            self.inflight[task_id] = max(0, self.inflight[task_id] - 1)
            if added:
                self.have[task_id] = self.have[task_id] + 1

    def snapshot_have(self) -> dict:
        with self.lock:
            return dict(self.have)


_MAX_IDLE_POLLS = 240  # ~2 min of empty WAITs before a worker gives up (safety)


def _worker(payload: dict):
    gpu = str(payload["gpu_id"])
    # Pin the GPU BEFORE any torch import (KoreEnv's verifier subprocesses inherit it).
    os.environ["HIP_VISIBLE_DEVICES"] = gpu
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    from kore.config import CONFIG
    from kore.data.amd_knowledge import ExperienceLedger
    from kore.data.gen_wins import generate_wins
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

    envs: dict = {}
    ledgers: dict = {}
    added_total = 0
    idle = 0
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
        added = False
        try:
            task = get_task(tid)
            env = envs.get(tid)
            if env is None:
                env = envs[tid] = KoreEnv(task)
            ledger = ledgers.setdefault(tid, ExperienceLedger())
            ws = generate_wins(task, teacher, env, gens=gens, cfg=CONFIG, ledger=ledger)
            if ws:
                st, _have = _persist_win(
                    _wins_shard(data_root, tid), _wins_lock(data_root, tid), ws[0], target
                )
                added = (st == "added")
                if added:
                    added_total += 1
            print(f"[deepen w{gpu}] {tid}: {'+1' if added else 'no-net'} "
                  f"have={coord.have.get(tid)}", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad trajectory never aborts the pool
            print(f"[deepen w{gpu}] {tid}: ERROR {type(e).__name__}: {str(e)[:140]}", flush=True)
        finally:
            coord.record(tid, added)
    result_q.put(("worker_done", gpu, added_total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tasks", default="", help="comma list; empty => all train tasks")
    ap.add_argument("--tasks-file", default="",
                    help="file of task ids, comma or newline separated. Preferred "
                         "over --tasks: a 4,524-id shard is 180 KB as one argument "
                         "and exec fails with 'Argument list too long' before the "
                         "process starts, which reads as a job that ran and "
                         "produced nothing")
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--target", type=int, default=3)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--teacher", default="claude")
    ap.add_argument("--model-teacher", default=None)
    ap.add_argument("--oversample", type=int,
                    default=int(os.environ.get("KORE_DEEP_OVERSAMPLE", "3")),
                    help="max concurrent trajectories per task = need * oversample")
    a = ap.parse_args()

    # A shard list passed as one argument hits ARG_MAX. pool-Triton's 4,524 ids are
    # 180 KB and exec failed with "Argument list too long", so the worker held a node
    # for its whole allocation having never started, and the stream looked merely
    # slow. Reading the same ids from a file has no such limit.
    if getattr(a, "tasks_file", ""):
        _raw = pathlib.Path(a.tasks_file).read_text()
        a.tasks = ",".join(t.strip() for t in _raw.replace("\n", ",").split(",")
                           if t.strip())

    import multiprocessing as mp
    import queue
    from kore.tasks.registry import train_tasks

    gpu_ids = [int(x) for x in a.gpu_ids.split(",") if x != ""]
    if not gpu_ids:
        ap.error("--gpu-ids must contain at least one GPU")
    if a.target < 1:
        ap.error("--target must be positive")
    if a.gens < 1:
        ap.error("--gens must be positive")
    if a.tasks.strip():
        tasks = list(dict.fromkeys(t for t in a.tasks.split(",") if t))
    else:
        tasks = [t.task_id for t in train_tasks()]  # excludes held-out by construction
    if not tasks:
        print("[deepen] COMPLETE: no tasks", flush=True)
        return 0
    # Defense-in-depth: never generate for held-out/eval tasks even if an explicit
    # --tasks list (e.g. from the partition) includes one (prevents eval leakage).
    from kore.tasks.registry import is_heldout as _is_heldout, get_task as _get_task, task_ids as _task_ids
    _known_ids = set(_task_ids())
    _kept = []
    for _t in tasks:
        try:
            if _t in _known_ids and _is_heldout(_get_task(_t)):
                print(f"[deepen] SKIP held-out task {_t}", flush=True)
                continue
        except Exception:
            pass
        _kept.append(_t)
    tasks = _kept
    if not tasks:
        print("[deepen] COMPLETE: no train tasks after held-out filter", flush=True)
        return 0

    # Seed persisted win counts from disk (resume-safe): tasks already at target
    # are never scheduled, so a re-run costs zero teacher calls for them.
    tasks_have = {}
    for t in tasks:
        existing, _ = _load_existing(_wins_shard(Path(a.data_root), t))
        tasks_have[t] = len(existing)
    remaining = [t for t, h in tasks_have.items() if h < a.target]
    already = len(tasks) - len(remaining)
    if not remaining:
        print(f"[deepen] COMPLETE: all {len(tasks)} tasks already at target", flush=True)
        return 0

    # NO min(., len(tasks)) ceiling: per-task K-concurrency keeps every worker busy
    # even when few tasks remain (the teacher, not the task count, is the limiter).
    n_workers = a.workers or (len(gpu_ids) * 4)
    if n_workers < 1:
        ap.error("--workers must be non-negative")

    print(f"[deepen] START tasks={len(tasks)} remaining={len(remaining)} "
          f"already_at_target={already} target={a.target} gens={a.gens} "
          f"gpus={gpu_ids} workers={n_workers} oversample={a.oversample} "
          f"data_root={a.data_root}", flush=True)

    ctx = mp.get_context("spawn")
    manager = mp.Manager()
    coord = TaskCoordinator(manager, tasks_have, a.target, a.oversample)
    result_q = manager.Queue()

    procs = []
    for i in range(n_workers):
        payload = dict(gpu_id=gpu_ids[i % len(gpu_ids)], data_root=a.data_root,
                       target=a.target, gens=a.gens, teacher_kind=a.teacher,
                       model_teacher=a.model_teacher, coord=coord, result_q=result_q)
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
                print("[deepen] FATAL worker exit(s): "
                      + ", ".join(f"pid={p.pid} rc={p.exitcode}" for p in failed), flush=True)
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                break
            if time.monotonic() - last_report > 60:
                have = coord.snapshot_have()
                done_n = sum(1 for t in remaining if have.get(t, 0) >= a.target)
                print(f"[deepen] progress {done_n}/{len(remaining)} remaining tasks at target "
                      f"(+{total_added} new wins)", flush=True)
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
    done_n = sum(1 for t in remaining if have.get(t, 0) >= a.target)
    partials = [t for t in remaining if 0 < have.get(t, 0) < a.target]
    missing = [t for t in remaining if have.get(t, 0) == 0]
    print(f"[deepen] COMPLETE: {done_n}/{len(remaining)} remaining tasks reached target, "
          f"+{total_added} new wins, {len(partials)} partial, {len(missing)} still-empty, "
          f"{worker_failures} worker failures", flush=True)

    if worker_failures:
        return 2
    if partials or missing:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
