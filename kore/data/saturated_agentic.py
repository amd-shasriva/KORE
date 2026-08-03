"""Node-saturating agentic trajectory generation.

``generate_agentic_trajectories`` parallelizes the episodes of ONE task. That is
the right unit for the generator but the wrong unit for a node: episodes-per-task
is ~5-10 (task diversity is the binding constraint, not depth), so a per-task pool
drains to a handful of live episodes at every task boundary and the node idles.
This module owns the other axis - a single pool over ``(task, episode)`` work
items that keeps every worker busy across task boundaries - and delegates each
work item back to the generator at ``n=1`` so episode semantics, record shape and
category labelling stay in exactly one place.

Three properties the shape of this job forces:

* **Checkpoint per episode.** Nodes are preempted (SPUR requeues aggressively) and
  an episode costs minutes, so a shard that is only durable at the end throws away
  hours. Every finished episode is appended, locked and fsynced, and a resume reads
  the shard back to skip work items it already holds.
* **Breadth before depth.** Work items are ordered by episode index first, so a
  node killed halfway has one episode for every task rather than every episode for
  half the tasks. Diversity is what the training mixture is short of.
* **Fail loudly on disk.** ``/home`` is a shared volume with other people's jobs on
  it, and trajectories are large. The runner refuses to start, and stops mid-run,
  rather than driving the volume to full.

The GPU-bound leg (verification) and the network-bound leg (the teacher) overlap
inside each worker, which is why the useful worker count is well above the eight
physical GPUs and has to be measured rather than derived - see
``scripts/pilot_agentic_throughput.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from kore.obs import get_logger

log = get_logger("data.saturated_agentic")

# Poll interval for the shared work queue; also bounds how quickly a worker
# notices a stop request or the run deadline.
_QUEUE_POLL_SECONDS = 0.5


class DiskBudgetExceeded(RuntimeError):
    """Free space on the output volume fell below the configured floor."""


# --------------------------------------------------------------------------- #
# Work items
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorkItem:
    """One episode of one task. ``key`` is the resume/dedup identity."""

    task_id: str
    episode: int

    @property
    def key(self) -> str:
        return f"{self.task_id}#{self.episode}"


def plan_work(task_ids: Sequence[str], episodes_per_task: int) -> list[WorkItem]:
    """Work items ordered breadth-first: episode 0 of every task, then episode 1...

    A preempted node keeps whatever prefix it finished. Depth-first ordering would
    make that prefix a few tasks explored deeply, which is the opposite of what the
    mixture needs - the registry's ~1.3K tasks are the scarce axis, not the number
    of samples drawn from any one of them.
    """
    if episodes_per_task < 1:
        raise ValueError("episodes_per_task must be >= 1")
    seen: list[str] = list(dict.fromkeys(task_ids))
    return [
        WorkItem(task_id, episode)
        for episode in range(episodes_per_task)
        for task_id in seen
    ]


# --------------------------------------------------------------------------- #
# Timing instrumentation
# --------------------------------------------------------------------------- #
class Meter:
    """Thread-safe wall-clock accumulator for the two legs of an episode.

    Attribution matters for sizing: if the teacher dominates, more workers help
    until the gateway's rate limit; if verification dominates, the GPUs are the
    ceiling and more workers only add contention. Measuring both is the only way
    to tell those apart from a single episodes/hour number.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.teacher_seconds = 0.0
        self.teacher_calls = 0
        self.env_seconds = 0.0
        self.env_calls = 0

    def add_teacher(self, seconds: float) -> None:
        with self._lock:
            self.teacher_seconds += seconds
            self.teacher_calls += 1

    def add_env(self, seconds: float) -> None:
        with self._lock:
            self.env_seconds += seconds
            self.env_calls += 1

    def merge(self, other: "Meter") -> None:
        with self._lock:
            self.teacher_seconds += other.teacher_seconds
            self.teacher_calls += other.teacher_calls
            self.env_seconds += other.env_seconds
            self.env_calls += other.env_calls

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "teacher_seconds": round(self.teacher_seconds, 3),
                "teacher_calls": self.teacher_calls,
                "env_seconds": round(self.env_seconds, 3),
                "env_calls": self.env_calls,
            }


class TimedTeacher:
    """Thin per-episode timing proxy over a shared teacher client."""

    def __init__(self, inner: Any, meter: Meter):
        self._inner = inner
        self._meter = meter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def generate(self, messages, **kwargs) -> str:
        start = time.monotonic()
        try:
            return self._inner.generate(messages, **kwargs)
        finally:
            self._meter.add_teacher(time.monotonic() - start)


class TimedEnv:
    """Timing proxy over a KoreEnv covering every GPU-bound entry point."""

    def __init__(self, inner: Any, meter: Meter):
        self._inner = inner
        self._meter = meter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def step(self, *args, **kwargs):
        start = time.monotonic()
        try:
            return self._inner.step(*args, **kwargs)
        finally:
            self._meter.add_env(time.monotonic() - start)

    def collect_counters(self, *args, **kwargs):
        start = time.monotonic()
        try:
            return self._inner.collect_counters(*args, **kwargs)
        finally:
            self._meter.add_env(time.monotonic() - start)


# --------------------------------------------------------------------------- #
# Durable, resumable shard I/O
# --------------------------------------------------------------------------- #
def _flock_append(path: Path, payloads: Iterable[str]) -> int:
    """Append newline-terminated payloads under an exclusive lock, then fsync.

    Workers on one node share a shard file, and the volume is NFS, so appends take
    the same advisory-lock discipline the rest of the data lane uses. fsync on every
    append is affordable here because an episode costs minutes: the durability is
    free relative to the work it protects.
    """
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(p if p.endswith("\n") else p + "\n" for p in payloads)
    if not blob:
        return 0
    data = blob.encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            written = 0
            view = memoryview(data)
            while written < len(data):
                written += os.write(fd, view[written:])
            os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return len(data)


def completed_keys(path: Path) -> set[str]:
    """Work-item keys already durable in a shard.

    Tolerates a torn tail: a node killed mid-append can leave a partial final line,
    and refusing to resume over it would discard every completed episode in the
    shard. An unparseable line simply does not count as done, so the work item is
    regenerated.
    """
    keys: set[str] = set()
    if not Path(path).exists():
        return keys
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("_work_key")
            if isinstance(key, str) and key:
                keys.add(key)
    return keys


def free_bytes(path: Path) -> int:
    """Free bytes on the volume that will hold ``path`` (walks up to a real dir)."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(str(probe)).free


class DiskGuard:
    """Refuse to run, and stop mid-run, when free space nears a floor.

    Filling the shared volume would take down other users' jobs, so the failure
    mode has to be a loud stop rather than a slow squeeze. Rechecked on a timer
    instead of on every write because ``statvfs`` on NFS is not free and the volume
    moves on the scale of other people's jobs, not ours.
    """

    def __init__(self, path: Path, min_free_bytes: int, recheck_seconds: float = 30.0):
        self.path = Path(path)
        self.min_free_bytes = int(min_free_bytes)
        self.recheck_seconds = float(recheck_seconds)
        self._lock = threading.Lock()
        self._last_check = 0.0
        self._last_free = -1

    def check(self, *, force: bool = False) -> int:
        if self.min_free_bytes <= 0:
            return -1
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last_check) < self.recheck_seconds:
                return self._last_free
            available = free_bytes(self.path)
            self._last_check = now
            self._last_free = available
        if available < self.min_free_bytes:
            raise DiskBudgetExceeded(
                f"{self.path}: {available / 1e9:.1f} GB free is below the "
                f"{self.min_free_bytes / 1e9:.1f} GB floor; refusing to write more "
                "trajectories"
            )
        return available


# --------------------------------------------------------------------------- #
# Node runner
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeOutcome:
    """Per-episode telemetry. One row per attempted work item."""

    key: str
    task_id: str
    episode: int
    worker: int
    gpu: Optional[int]
    wall_seconds: float
    meter: dict
    turns: int = 0
    category: str = "error"
    success: bool = False
    best_reward: Optional[float] = None
    best_speedup: Optional[float] = None
    record_bytes: int = 0
    kept: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        row = {
            "key": self.key,
            "task_id": self.task_id,
            "episode": self.episode,
            "worker": self.worker,
            "gpu": self.gpu,
            "wall_seconds": round(self.wall_seconds, 3),
            "turns": self.turns,
            "category": self.category,
            "success": self.success,
            "best_reward": self.best_reward,
            "best_speedup": self.best_speedup,
            "record_bytes": self.record_bytes,
            "kept": self.kept,
        }
        row.update(self.meter)
        if self.error:
            row["error"] = self.error
        return row


@dataclass
class NodeRunResult:
    attempted: int = 0
    kept: int = 0
    errors: int = 0
    skipped_resume: int = 0
    bytes_written: int = 0
    wall_seconds: float = 0.0
    by_category: dict = field(default_factory=dict)
    outcomes: list = field(default_factory=list)
    stopped_reason: str = "complete"

    def summary(self) -> dict:
        eph = (self.kept / self.wall_seconds * 3600.0) if self.wall_seconds > 0 else 0.0
        attempted_per_hour = (
            self.attempted / self.wall_seconds * 3600.0 if self.wall_seconds > 0 else 0.0
        )
        return {
            "attempted": self.attempted,
            "kept": self.kept,
            "errors": self.errors,
            "skipped_resume": self.skipped_resume,
            "bytes_written": self.bytes_written,
            "wall_seconds": round(self.wall_seconds, 2),
            "kept_per_hour": round(eph, 1),
            "attempted_per_hour": round(attempted_per_hour, 1),
            "by_category": dict(self.by_category),
            "stopped_reason": self.stopped_reason,
        }


def _best_speedup(record: Any) -> Optional[float]:
    """Largest measured speedup in the episode's bench results.

    Read from the tool trace rather than the reward, because the reward folds in
    correctness tiers and phase curricula; the filter downstream needs the raw
    measured number to judge gain and to catch implausible values.
    """
    trace = getattr(record, "tool_trace", None) or []
    best: Optional[float] = None
    for call in trace:
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        if not result.get("correct"):
            continue
        speedup = result.get("speedup")
        if isinstance(speedup, (int, float)) and (best is None or speedup > best):
            best = float(speedup)
    return best


def _default_env_factory(task: Any, gpu: Optional[int]):
    from kore.env.kore_env import KoreEnv

    return KoreEnv(task, gpu=None if gpu is None else str(gpu))


def run_node_shard(
    *,
    task_ids: Sequence[str],
    episodes_per_task: int,
    workers: int,
    out_path: Any,
    gpu_ids: Sequence[int] = (0,),
    telemetry_path: Any = None,
    max_turns: int = 8,
    teacher: Any = None,
    task_loader: Optional[Callable[[str], Any]] = None,
    env_factory: Optional[Callable[[Any, Optional[int]], Any]] = None,
    generator: Optional[Callable[..., list]] = None,
    keep_only_useful: bool = False,
    min_free_bytes: int = 0,
    disk_recheck_seconds: float = 30.0,
    deadline: Optional[float] = None,
    stop_event: Optional[threading.Event] = None,
    resume: bool = True,
    shard_meta: Optional[dict] = None,
    progress_every: int = 10,
    log_fn: Callable[[str], None] = print,
) -> NodeRunResult:
    """Run ``episodes_per_task`` episodes of every task across ``workers`` threads.

    One shared pool over all ``(task, episode)`` pairs, one private env per episode
    (the generator's concurrency contract), one shared teacher client (its HTTP
    connection pool is the thing worth sharing). Returns per-episode telemetry
    alongside the totals so a caller can build a worker-count curve without
    re-running anything.
    """
    from queue import Empty, Queue

    if workers < 1:
        raise ValueError("workers must be >= 1")
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty")

    out_path = Path(out_path)
    telemetry_path = Path(telemetry_path) if telemetry_path else None
    task_loader = task_loader or _import_get_task()
    env_factory = env_factory or _default_env_factory
    generator = generator or _import_generator()

    guard = DiskGuard(out_path, min_free_bytes, recheck_seconds=disk_recheck_seconds)
    guard.check(force=True)

    items = plan_work(task_ids, episodes_per_task)
    result = NodeRunResult()
    if resume:
        done = completed_keys(out_path)
        if done:
            before = len(items)
            items = [item for item in items if item.key not in done]
            result.skipped_resume = before - len(items)
            log_fn(
                f"[saturate] resume: {result.skipped_resume} of {before} work items "
                f"already durable in {out_path.name}"
            )
    if not items:
        log_fn("[saturate] nothing to do")
        return result

    queue: "Queue[Optional[WorkItem]]" = Queue()
    for item in items:
        queue.put(item)

    stop_event = stop_event or threading.Event()
    counters_lock = threading.Lock()
    task_cache: dict[str, Any] = {}
    task_cache_lock = threading.Lock()
    started = time.monotonic()

    def _task(task_id: str):
        # Task construction reads the registry and the task's seed file; cache it so
        # a node running many episodes per task pays that once, not per episode.
        with task_cache_lock:
            cached = task_cache.get(task_id)
        if cached is None:
            cached = task_loader(task_id)
            with task_cache_lock:
                task_cache[task_id] = cached
        return cached

    def _publish(outcome: EpisodeOutcome, payloads: list[str]) -> None:
        written = _flock_append(out_path, payloads) if payloads else 0
        outcome.record_bytes = written
        with counters_lock:
            result.attempted += 1
            result.bytes_written += written
            result.by_category[outcome.category] = (
                result.by_category.get(outcome.category, 0) + 1
            )
            if outcome.kept:
                result.kept += 1
            if outcome.error:
                result.errors += 1
            result.outcomes.append(outcome)
            n_done = result.attempted
        if telemetry_path is not None:
            _flock_append(telemetry_path, [json.dumps(outcome.to_dict())])
        if progress_every and n_done % progress_every == 0:
            elapsed = time.monotonic() - started
            rate = n_done / elapsed * 3600.0 if elapsed > 0 else 0.0
            log_fn(
                f"[saturate] {n_done}/{len(items)} episodes  "
                f"{rate:.0f} ep/h  kept={result.kept}  "
                f"cats={result.by_category}"
            )

    def _worker(worker_id: int) -> None:
        gpu = gpu_ids[worker_id % len(gpu_ids)]
        while not stop_event.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                _set_stop(stop_event, result, counters_lock, "deadline")
                return
            try:
                item = queue.get(timeout=_QUEUE_POLL_SECONDS)
            except Empty:
                return
            if item is None:
                return
            episode_start = time.monotonic()
            meter = Meter()
            outcome = EpisodeOutcome(
                key=item.key, task_id=item.task_id, episode=item.episode,
                worker=worker_id, gpu=gpu, wall_seconds=0.0, meter=meter.snapshot(),
            )
            try:
                guard.check()
                task = _task(item.task_id)
                timed_teacher = TimedTeacher(teacher, meter)

                def _make_env(_task=task, _gpu=gpu, _meter=meter):
                    return TimedEnv(env_factory(_task, _gpu), _meter)

                # n=1 keeps every episode-level decision (category labelling, record
                # construction, keep_only_useful) inside the one generator the
                # contract tests cover; this module supplies only the scheduling.
                records = generator(
                    task, timed_teacher, None, n=1, max_turns=max_turns,
                    keep_only_useful=keep_only_useful, env_factory=_make_env,
                )
                outcome.wall_seconds = time.monotonic() - episode_start
                outcome.meter = meter.snapshot()
                payloads: list[str] = []
                for record in records:
                    provenance = dict(getattr(record, "provenance", {}) or {})
                    outcome.category = str(provenance.get("category") or "unknown")
                    outcome.turns = int(provenance.get("turns_used") or 0)
                    outcome.success = bool(getattr(record, "success", False))
                    outcome.best_reward = getattr(record, "best_reward", None)
                    outcome.best_speedup = _best_speedup(record)
                    row = record.to_dict()
                    row["_work_key"] = item.key
                    row["_best_speedup"] = outcome.best_speedup
                    row["_episode_seconds"] = round(outcome.wall_seconds, 3)
                    if shard_meta:
                        row["_shard"] = dict(shard_meta)
                    payloads.append(json.dumps(row))
                    outcome.kept = True
                if not records:
                    # keep_only_useful dropped it, or the teacher produced nothing
                    # usable. Still a completed work item: record a resume marker so
                    # a restart does not pay for it again.
                    outcome.category = "dropped"
                    payloads.append(json.dumps({
                        "_work_key": item.key, "_dropped": True,
                        "task_id": item.task_id,
                    }))
                _publish(outcome, payloads)
            except DiskBudgetExceeded as exc:
                outcome.error = str(exc)
                outcome.wall_seconds = time.monotonic() - episode_start
                outcome.meter = meter.snapshot()
                with counters_lock:
                    result.errors += 1
                    result.outcomes.append(outcome)
                log_fn(f"[saturate] FATAL {exc}")
                _set_stop(stop_event, result, counters_lock, "disk_floor")
                return
            except Exception as exc:  # noqa: BLE001 - one bad task must not end the shard
                outcome.error = f"{type(exc).__name__}: {exc}"
                outcome.wall_seconds = time.monotonic() - episode_start
                outcome.meter = meter.snapshot()
                log.warn("agentic_episode_failed", task=item.task_id,
                         episode=item.episode, error=outcome.error[:200])
                _publish(outcome, [])
            finally:
                queue.task_done()

    threads = [
        threading.Thread(target=_worker, args=(i,), name=f"saturate-{i}", daemon=True)
        for i in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    result.wall_seconds = time.monotonic() - started
    log_fn(f"[saturate] done: {json.dumps(result.summary())}")
    return result


def _set_stop(stop_event, result, lock, reason: str) -> None:
    with lock:
        if result.stopped_reason == "complete":
            result.stopped_reason = reason
    stop_event.set()


def _import_get_task():
    from kore.tasks.registry import get_task

    return get_task


def _import_generator():
    from kore.data.gen_agentic import generate_agentic_trajectories

    return generate_agentic_trajectories


__all__ = [
    "DiskBudgetExceeded",
    "DiskGuard",
    "EpisodeOutcome",
    "Meter",
    "NodeRunResult",
    "TimedEnv",
    "TimedTeacher",
    "WorkItem",
    "completed_keys",
    "free_bytes",
    "plan_work",
    "run_node_shard",
]
