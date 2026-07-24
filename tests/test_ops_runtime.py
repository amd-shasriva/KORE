from __future__ import annotations

import os
from pathlib import Path
import signal

import pytest

from kore.ops.runtime import (
    ArtifactStatus,
    IncrementalLogReader,
    ProcessIdentity,
    ProcessSnapshot,
    RunPhase,
    SecureRuntime,
    SecurityError,
    SupervisorStateMachine,
    task_set_identity,
    terminate_owned,
)


class FakeProcessBackend:
    """In-memory process table; no real process is created or signalled."""

    def __init__(self, snapshots: list[ProcessSnapshot], *, term_stops: bool = False):
        self.snapshots = {snapshot.pid: snapshot for snapshot in snapshots}
        self.term_stops = term_stops
        self.signals: list[tuple[str, int, int]] = []
        self.now = 0.0

    def snapshot(self, pid: int) -> ProcessSnapshot | None:
        return self.snapshots.get(pid)

    def group_members(self, pgid: int) -> list[ProcessSnapshot]:
        return sorted(
            (item for item in self.snapshots.values() if item.pgid == pgid),
            key=lambda item: item.pid,
        )

    def signal_pid(self, pid: int, sig: int) -> None:
        self.signals.append(("pid", pid, sig))
        if sig == signal.SIGKILL or self.term_stops:
            self.snapshots.pop(pid, None)

    def signal_group(self, pgid: int, sig: int) -> None:
        self.signals.append(("group", pgid, sig))
        if sig == signal.SIGKILL or self.term_stops:
            self.snapshots = {
                pid: item
                for pid, item in self.snapshots.items()
                if item.pgid != pgid
            }

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _snapshot(
    pid: int,
    *,
    start: int = 100,
    pgid: int = 10,
    run_id: str | None = "run-1",
    uid: int | None = None,
    cgroup: str = "0::/user.slice/kore",
) -> ProcessSnapshot:
    return ProcessSnapshot(
        pid=pid,
        uid=os.getuid() if uid is None else uid,
        start_ticks=start,
        pgid=pgid,
        cgroup=cgroup,
        run_id=run_id,
    )


def _identity() -> ProcessIdentity:
    return ProcessIdentity(
        run_id="run-1",
        pid=10,
        uid=os.getuid(),
        start_ticks=100,
        pgid=10,
        cgroup="0::/user.slice/kore",
        owns_process_group=True,
        marker_required=True,
    )


def test_termination_escalates_only_the_owned_fake_group():
    backend = FakeProcessBackend([_snapshot(10), _snapshot(11)])

    result = terminate_owned(
        _identity(),
        term_timeout=0.2,
        kill_timeout=0.2,
        poll_interval=0.1,
        backend=backend,
    )

    assert result.stopped
    assert result.term_sent and result.kill_sent
    assert backend.signals == [
        ("group", 10, signal.SIGTERM),
        ("group", 10, signal.SIGKILL),
    ]
    assert backend.snapshots == {}


def test_termination_refuses_pid_reuse_without_signalling():
    backend = FakeProcessBackend([_snapshot(10, start=101)])

    result = terminate_owned(_identity(), backend=backend)

    assert result.refused
    assert result.reason == "pid identity mismatch"
    assert backend.signals == []


def test_termination_refuses_foreign_group_member():
    backend = FakeProcessBackend(
        [_snapshot(10), _snapshot(11, run_id="some-other-run")]
    )

    result = terminate_owned(_identity(), backend=backend)

    assert result.refused
    assert "foreign run marker" in result.reason
    assert backend.signals == []


def test_termination_cleans_owned_children_after_leader_exits():
    backend = FakeProcessBackend([_snapshot(11)])

    result = terminate_owned(
        _identity(), term_timeout=0, kill_timeout=0.2, backend=backend
    )

    assert result.stopped
    assert backend.signals == [
        ("group", 10, signal.SIGTERM),
        ("group", 10, signal.SIGKILL),
    ]


def test_supervisor_state_machine_requires_zero_exit_and_verifier():
    machine = SupervisorStateMachine(max_attempts=2)
    assert machine.begin_attempt() == 1
    machine.process_exited(0)
    assert not machine.verification_finished(
        ArtifactStatus.failure("artifact incomplete")
    )
    assert machine.phase is RunPhase.WAITING

    assert machine.begin_attempt() == 2
    machine.process_exited(9)
    assert not machine.verification_finished(ArtifactStatus.success())
    assert machine.phase is RunPhase.GAVE_UP
    assert "rc=9" in machine.reason


def test_supervisor_state_machine_accepts_strict_completion():
    machine = SupervisorStateMachine(max_attempts=1)
    machine.begin_attempt()
    machine.process_exited(0)

    assert machine.verification_finished(ArtifactStatus.success(path="artifact"))
    assert machine.phase is RunPhase.SUCCEEDED


def test_private_runtime_sentinel_and_immutable_task_set(tmp_path: Path):
    runtime = SecureRuntime(tmp_path / "runtime")
    assert (runtime.path.stat().st_mode & 0o777) == 0o700

    identity = runtime.store_task_set("runs/run-1/tasks.json", ["task-b", "task-a"])
    assert identity == task_set_identity(["task-a", "task-b"])
    assert runtime.store_task_set(
        "runs/run-1/tasks.json", ["task-b", "task-a"]
    ) == identity
    with pytest.raises(SecurityError, match="immutable task-set"):
        runtime.store_task_set("runs/run-1/tasks.json", ["task-a", "task-c"])

    sentinel = runtime.write_sentinel("paused", {"run_id": "run-1"})
    assert (sentinel.stat().st_mode & 0o777) == 0o600
    assert runtime.consume_sentinel("paused") == {"run_id": "run-1"}
    assert runtime.peek_sentinel("paused") is None


def test_private_runtime_refuses_symlink_sentinel(tmp_path: Path):
    runtime = SecureRuntime(tmp_path / "runtime")
    target = tmp_path / "target"
    target.write_text("{}")
    sentinel = runtime.state_path("sentinels/paused.json")
    sentinel.symlink_to(target)

    with pytest.raises(SecurityError, match="symlink sentinel"):
        runtime.consume_sentinel("paused")


def test_incremental_reader_handles_partial_append_and_truncation(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text("first")
    reader = IncrementalLogReader(log)

    assert reader.read_lines() == []
    with log.open("a") as handle:
        handle.write(" line\nsecond\n")
    assert reader.read_lines() == ["first line", "second"]
    assert reader.read_lines() == []

    log.write_text("rotated\n")
    assert reader.read_lines() == ["rotated"]
