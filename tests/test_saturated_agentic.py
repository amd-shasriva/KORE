"""Contract for the node-saturating agentic runner.

The properties tested here are the ones whose failure is silent and expensive:
a resume that regenerates work it already paid for, a preemption that loses the
prefix it finished, a disk floor that is checked but not enforced, and a worker
pool that reports concurrency it never achieved. Each of those looks like a
working run right up until the point where it has wasted a day of eight nodes.

CPU-only: the generator, the env and the teacher are all injected.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from kore.data import saturated_agentic as sat


class _Record:
    def __init__(self, task_id: str, category: str = "success", speedup=2.0):
        self.task_id = task_id
        self.success = category != "attempt"
        self.best_reward = 1.0
        self.provenance = {"category": category, "turns_used": 3}
        self.tool_trace = [
            {"name": "bench", "result": {"correct": True, "speedup": speedup}},
            {"name": "bench", "result": {"correct": True, "speedup": speedup / 2}},
        ]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "type": "agentic",
            "messages": [{"role": "user", "content": "x"}],
            "provenance": dict(self.provenance),
        }


class _Recorder:
    """Injected generator that records concurrency and env identity."""

    def __init__(self, sleep: float = 0.02, category: str = "success", fail_on=()):
        self.sleep = sleep
        self.category = category
        self.fail_on = set(fail_on)
        self.lock = threading.Lock()
        self.live = 0
        self.peak = 0
        self.envs: list = []
        self.calls: list = []

    def __call__(self, task, teacher, env, *, n, max_turns, keep_only_useful,
                 env_factory):
        assert env is None, "the node pool supplies a private env per episode"
        assert n == 1, "the node pool owns concurrency; the generator runs one episode"
        made = env_factory()
        with self.lock:
            self.envs.append(made)
            self.calls.append(task.task_id)
            self.live += 1
            self.peak = max(self.peak, self.live)
        try:
            # Exercise both instrumented legs so the meter has something to report.
            teacher.generate([{"role": "user", "content": "hi"}])
            made.step("src")
            time.sleep(self.sleep)
            if task.task_id in self.fail_on:
                raise RuntimeError("boom")
            return [_Record(task.task_id, self.category)]
        finally:
            with self.lock:
                self.live -= 1


class _Task:
    def __init__(self, task_id: str):
        self.task_id = task_id


class _Env:
    def __init__(self, task, gpu):
        self.task = task
        self.gpu = gpu

    def step(self, src, **kwargs):
        time.sleep(0.001)
        return object()


class _Teacher:
    def generate(self, messages, **kwargs) -> str:
        time.sleep(0.001)
        return "ok"


def _run(tmp_path, *, tasks, episodes=1, workers=4, generator=None, **kwargs):
    generator = generator or _Recorder()
    result = sat.run_node_shard(
        task_ids=tasks,
        episodes_per_task=episodes,
        workers=workers,
        gpu_ids=[0, 1],
        out_path=tmp_path / "shard.jsonl",
        telemetry_path=tmp_path / "shard.telemetry.jsonl",
        teacher=_Teacher(),
        task_loader=_Task,
        env_factory=_Env,
        generator=generator,
        log_fn=lambda *_: None,
        **kwargs,
    )
    return result, generator


# --------------------------------------------------------------------------- #
# Work planning
# --------------------------------------------------------------------------- #
def test_work_is_planned_breadth_first_across_tasks():
    items = sat.plan_work(["a", "b", "c"], 3)
    assert [item.key for item in items[:3]] == ["a#0", "b#0", "c#0"]
    # A node killed after three items must hold one episode of every task, not
    # three episodes of one; task diversity is the scarce axis.
    assert len({item.task_id for item in items[:3]}) == 3
    assert len(items) == 9


def test_plan_work_deduplicates_task_ids():
    assert len(sat.plan_work(["a", "a", "b"], 2)) == 4


def test_plan_work_rejects_zero_episodes():
    with pytest.raises(ValueError):
        sat.plan_work(["a"], 0)


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
def test_workers_overlap_across_task_boundaries(tmp_path):
    # Four tasks with one episode each: a per-task pool would never exceed one
    # live episode. The whole reason this module exists is that it does.
    result, generator = _run(tmp_path, tasks=["a", "b", "c", "d"], workers=4)
    assert result.attempted == 4
    assert generator.peak > 1, f"no overlap observed (peak={generator.peak})"


def test_every_episode_gets_its_own_env(tmp_path):
    _run(tmp_path, tasks=["a", "b", "c", "d"], episodes=2, workers=4)
    result, generator = _run(
        tmp_path / "second", tasks=["a", "b", "c", "d"], episodes=2, workers=4)
    assert len(generator.envs) == 8
    assert len({id(env) for env in generator.envs}) == 8


def test_workers_are_pinned_round_robin_across_gpus(tmp_path):
    result, _ = _run(tmp_path, tasks=[f"t{i}" for i in range(8)], workers=4)
    assert {outcome.gpu for outcome in result.outcomes} == {0, 1}


# --------------------------------------------------------------------------- #
# Durability and resume
# --------------------------------------------------------------------------- #
def test_finished_episodes_are_durable_and_resume_skips_them(tmp_path):
    result, _ = _run(tmp_path, tasks=["a", "b", "c"], workers=2)
    assert result.kept == 3
    shard = tmp_path / "shard.jsonl"
    keys = sat.completed_keys(shard)
    assert keys == {"a#0", "b#0", "c#0"}

    again, generator = _run(tmp_path, tasks=["a", "b", "c"], workers=2)
    assert again.skipped_resume == 3
    assert again.attempted == 0
    assert generator.calls == []


def test_resume_tolerates_a_torn_tail(tmp_path):
    _run(tmp_path, tasks=["a", "b"], workers=2)
    shard = tmp_path / "shard.jsonl"
    # A node killed mid-append leaves a partial line. Refusing to parse the file
    # would discard both completed episodes and pay for them again.
    with shard.open("a") as handle:
        handle.write('{"_work_key": "c#0", "part')
    assert sat.completed_keys(shard) == {"a#0", "b#0"}


def test_records_carry_the_resume_key_and_measured_speedup(tmp_path):
    _run(tmp_path, tasks=["a"], workers=1)
    rows = [
        json.loads(line)
        for line in (tmp_path / "shard.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[0]["_work_key"] == "a#0"
    # Best measured speedup across the episode's correct benches, not the last one.
    assert rows[0]["_best_speedup"] == 2.0
    assert rows[0]["_episode_seconds"] >= 0


def test_dropped_episodes_still_checkpoint_so_resume_does_not_repay(tmp_path):
    result, _ = _run(
        tmp_path, tasks=["a", "b"], workers=2, generator=lambda *a, **k: [])
    assert result.kept == 0
    assert result.by_category == {"dropped": 2}
    assert sat.completed_keys(tmp_path / "shard.jsonl") == {"a#0", "b#0"}


# --------------------------------------------------------------------------- #
# Failure containment
# --------------------------------------------------------------------------- #
def test_one_failing_task_does_not_end_the_shard(tmp_path):
    generator = _Recorder(fail_on=["b"])
    result, _ = _run(tmp_path, tasks=["a", "b", "c"], workers=2, generator=generator)
    assert result.errors == 1
    assert result.kept == 2
    assert result.stopped_reason == "complete"


def test_telemetry_row_per_attempted_episode(tmp_path):
    _run(tmp_path, tasks=["a", "b"], workers=2)
    rows = [
        json.loads(line)
        for line in (tmp_path / "shard.telemetry.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    for row in rows:
        assert row["teacher_calls"] == 1
        assert row["env_calls"] == 1
        assert row["teacher_seconds"] >= 0.0
        assert row["turns"] == 3
        assert row["category"] == "success"


# --------------------------------------------------------------------------- #
# Disk guard
# --------------------------------------------------------------------------- #
def test_disk_guard_refuses_to_start_below_the_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(sat, "free_bytes", lambda _p: 1)
    with pytest.raises(sat.DiskBudgetExceeded):
        _run(tmp_path, tasks=["a"], workers=1, min_free_bytes=10**9)


def test_disk_guard_stops_a_running_shard(tmp_path, monkeypatch):
    # Enough headroom to start, none once the run is under way: the guard must
    # stop the shard rather than quietly keep filling a shared volume.
    values = iter([10**12])

    def _free(_path):
        try:
            return next(values)
        except StopIteration:
            return 1

    monkeypatch.setattr(sat, "free_bytes", _free)
    result, _ = _run(
        tmp_path, tasks=[f"t{i}" for i in range(8)], workers=2,
        min_free_bytes=10**9, disk_recheck_seconds=0.0,
        generator=_Recorder(sleep=0.0),
    )
    assert result.stopped_reason == "disk_floor"
    assert result.attempted < 8


def test_disk_guard_is_disabled_by_a_zero_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(sat, "free_bytes", lambda _p: 0)
    result, _ = _run(tmp_path, tasks=["a"], workers=1, min_free_bytes=0)
    assert result.kept == 1


# --------------------------------------------------------------------------- #
# Deadline
# --------------------------------------------------------------------------- #
def test_deadline_stops_the_pool_without_losing_finished_work(tmp_path):
    result, _ = _run(
        tmp_path, tasks=[f"t{i}" for i in range(40)], workers=2,
        generator=_Recorder(sleep=0.05),
        deadline=time.monotonic() + 0.25,
    )
    assert result.stopped_reason == "deadline"
    assert 0 < result.attempted < 40
    assert len(sat.completed_keys(tmp_path / "shard.jsonl")) == result.kept


# --------------------------------------------------------------------------- #
# Instrumentation
# --------------------------------------------------------------------------- #
def test_meter_attributes_both_legs_separately():
    meter = sat.Meter()
    teacher = sat.TimedTeacher(_Teacher(), meter)
    env = sat.TimedEnv(_Env(_Task("a"), 0), meter)
    teacher.generate([])
    env.step("src")
    snapshot = meter.snapshot()
    assert snapshot["teacher_calls"] == 1
    assert snapshot["env_calls"] == 1
    assert snapshot["teacher_seconds"] > 0
    assert snapshot["env_seconds"] > 0


def test_timed_proxies_delegate_unknown_attributes():
    meter = sat.Meter()
    env = sat.TimedEnv(_Env(_Task("abc"), 3), meter)
    assert env.gpu == 3
    assert env.task.task_id == "abc"


def test_outage_decision_uses_the_failure_count_this_call_observed(monkeypatch):
    """A concurrent success must not talk a failing call out of the hard stop.

    One teacher client is now shared by a node's worth of episode threads. The
    outage detector reads the consecutive-failure count to decide whether to stop
    the campaign; if that read happens after the increment rather than with it,
    any other thread's success resets the counter in between and the stop never
    fires. The campaign then writes empty data for hours, which is the exact
    silent degradation ResilientTeacher exists to prevent.
    """
    from kore.data import teacher as teacher_module

    class _Failing:
        def generate(self, messages, **kwargs):
            raise RuntimeError("gateway down")

    teacher = teacher_module.ResilientTeacher(_Failing(), max_consecutive_failures=1)
    interleaved = threading.Event()

    class _InterleavingLog:
        """Runs a concurrent success at the moment the failure path logs."""

        def __getattr__(self, name):
            return getattr(teacher_module.log, name)

        def error(self, *args, **kwargs):
            if interleaved.is_set():
                return
            interleaved.set()
            other = threading.Thread(target=teacher._succeed_once)
            other.start()
            other.join()

    teacher._succeed_once = lambda: setattr(teacher, "_consec", 0)
    monkeypatch.setattr(teacher_module, "log", _InterleavingLog())

    with pytest.raises(RuntimeError, match="teacher unavailable"):
        teacher.generate([])
    assert interleaved.is_set(), "the interleaving under test never happened"
