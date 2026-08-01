from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess

import pytest

from scripts._kf_verify import PARTITION_TASK_PREFIXES, split_prefixes
from scripts import spur_supervise_datagen as sup
from scripts.spur_supervise_datagen import (
    _json_line,
    classify_queue_reply,
    factory_jobs_active,
    progress_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

HEADER = "JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)\n"
ACTIVE_QUEUE = HEADER + "123 amd-spur kore-factory user R 00:10 1 node\n"

# Captured live from SPUR login node 013: the control plane answered `squeue`
# with an anyhow cause chain instead of a job table. The supervisor must treat
# this as "I could not read the scheduler", never as "no jobs are queued".
SPUR_ERROR_BLOB = (
    "Error: failed to connect to spurctld\n"
    "\n"
    "Caused by:\n"
    "    0: transport error\n"
    "    3: Connection refused (os error 111)\n"
)


def _supervisor(tmp_path: Path, **overrides) -> sup.Supervisor:
    args = argparse.Namespace(
        repo=str(tmp_path),
        python="python",
        data_root=str(tmp_path / "data"),
        target=3,
        shards=4,
        wave_nodes=4,
        poll_seconds=0,
        submission_grace_seconds=0,
        queue_timeout_seconds=5,
        queue_attempts=3,
        empty_polls_to_finish=3,
        visibility_timeout_seconds=0,
        max_scheduler_failures=3,
        max_stalled_waves=2,
        max_waves=2,
        verify_prefix=PARTITION_TASK_PREFIXES,
        user="tester",
        log=str(tmp_path / "supervisor.log"),
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return sup.Supervisor(args)


def _replies(monkeypatch, supervisor, outputs, *, returncode=0):
    """Feed ``outputs`` to successive ``run`` calls; returns the call log."""
    seen: list[str] = []

    def fake_run(command, *, check=True, timeout=None):
        stdout = outputs[len(seen)]
        seen.append(stdout)
        return subprocess.CompletedProcess(command, returncode, stdout=stdout)

    monkeypatch.setattr(supervisor, "run", fake_run)
    monkeypatch.setattr(sup.time, "sleep", lambda *_: None)
    return seen


# --------------------------------------------------------------------------- #
# queue classification: an unreadable scheduler is not an empty queue
# --------------------------------------------------------------------------- #
def test_factory_job_detection_uses_job_name_column():
    unrelated = HEADER + "124 amd-spur bash user R 00:10 1 node\n"

    assert not factory_jobs_active(HEADER)
    assert factory_jobs_active(ACTIVE_QUEUE)
    assert not factory_jobs_active(unrelated)


def test_spurctld_error_blob_is_unreadable_not_empty():
    assert classify_queue_reply(SPUR_ERROR_BLOB) == sup.QUEUE_UNREADABLE
    # It is also not "active" -- which is exactly why a bare boolean was unsafe.
    assert factory_jobs_active(SPUR_ERROR_BLOB) is False


@pytest.mark.parametrize(
    "reply",
    (
        "Error: failed to connect to spurctld\n",
        "slurm_load_jobs error: Unable to contact slurm controller\n",
        "squeue: error: Socket timed out on send/recv operation\n",
        "no leader elected yet\n",
        "Caused by:\n    0: transport error\n",
    ),
)
def test_error_shaped_replies_are_unreadable(reply):
    assert classify_queue_reply(reply) == sup.QUEUE_UNREADABLE


@pytest.mark.parametrize(
    "reply",
    (
        "",
        "   \n\n",
        HEADER,
        HEADER + "\n",
        "No jobs found\n",
    ),
)
def test_empty_shaped_replies_are_never_treated_as_errors(reply):
    # SPUR's output for a genuinely empty queue is unconfirmed, so every plausible
    # shape of "nothing is queued" must classify as empty. Calling one of these an
    # error would stall the campaign at the end of every wave.
    assert classify_queue_reply(reply) == sup.QUEUE_EMPTY


def test_active_queue_classifies_as_active():
    assert classify_queue_reply(ACTIVE_QUEUE) == sup.QUEUE_ACTIVE
    assert classify_queue_reply(ACTIVE_QUEUE + HEADER) == sup.QUEUE_ACTIVE


# --------------------------------------------------------------------------- #
# queue(): an error blob at exit code 0 is a transient failure, not a result
# --------------------------------------------------------------------------- #
def test_queue_retries_error_blob_delivered_at_exit_zero(tmp_path, monkeypatch):
    supervisor = _supervisor(tmp_path, queue_attempts=3)
    seen = _replies(monkeypatch, supervisor, [SPUR_ERROR_BLOB] * 3)

    with pytest.raises(RuntimeError, match="squeue failed"):
        supervisor.queue()

    assert len(seen) == 3  # bounded retry, not a silent "empty queue"


def test_queue_returns_the_table_once_the_controller_recovers(tmp_path, monkeypatch):
    supervisor = _supervisor(tmp_path, queue_attempts=3)
    seen = _replies(monkeypatch, supervisor, [SPUR_ERROR_BLOB, ACTIVE_QUEUE])

    assert supervisor.queue_state() == sup.QUEUE_ACTIVE
    assert len(seen) == 2


def test_startup_guard_never_submits_while_the_scheduler_is_unreadable(
    tmp_path, monkeypatch
):
    supervisor = _supervisor(tmp_path, queue_attempts=2)
    _replies(monkeypatch, supervisor, [SPUR_ERROR_BLOB] * 2)
    submitted: list[str] = []
    monkeypatch.setattr(
        supervisor, "submit_wave", lambda: submitted.append("wave") or "1"
    )
    monkeypatch.setattr(supervisor, "verify", lambda: pytest.fail("verify ran"))

    with pytest.raises(RuntimeError, match="squeue failed"):
        supervisor.supervise()

    # The old bool guard read this reply as "no factory jobs" and stacked a
    # second 64-node wave on top of the live one.
    assert submitted == []


# --------------------------------------------------------------------------- #
# wait_for_wave(): completion needs consecutive empty polls
# --------------------------------------------------------------------------- #
def test_wave_completion_requires_consecutive_empty_polls(tmp_path, monkeypatch):
    supervisor = _supervisor(tmp_path, empty_polls_to_finish=3)
    seen = _replies(
        monkeypatch,
        supervisor,
        [
            ACTIVE_QUEUE,
            HEADER,        # one empty poll must NOT end the wait
            ACTIVE_QUEUE,  # ... and the wave is indeed still running
            HEADER,
            HEADER,
            HEADER,        # third consecutive empty -> drained
            ACTIVE_QUEUE,  # never reached
        ],
    )

    supervisor.wait_for_wave()

    assert len(seen) == 6


def test_unreadable_reply_resets_the_empty_poll_debounce(tmp_path, monkeypatch):
    supervisor = _supervisor(
        tmp_path, empty_polls_to_finish=3, queue_attempts=1, max_scheduler_failures=3
    )
    seen = _replies(
        monkeypatch,
        supervisor,
        [
            ACTIVE_QUEUE,
            HEADER,
            HEADER,           # debounce at 2/3
            SPUR_ERROR_BLOB,  # unreadable -> tells us nothing; drop the debounce
            HEADER,
            HEADER,
            HEADER,           # a fresh run of three empties
        ],
    )

    supervisor.wait_for_wave()

    assert len(seen) == 7


def test_wait_gives_up_on_a_persistently_unreadable_scheduler(tmp_path, monkeypatch):
    supervisor = _supervisor(
        tmp_path, queue_attempts=1, max_scheduler_failures=2
    )
    _replies(monkeypatch, supervisor, [SPUR_ERROR_BLOB] * 2)

    with pytest.raises(RuntimeError, match="squeue failed"):
        supervisor.wait_for_wave()


# --------------------------------------------------------------------------- #
# verify(): the supervisor must check every family the partitioner shards
# --------------------------------------------------------------------------- #
_COMPLETE_SUMMARY = {
    "tasks": 2,
    "fully_complete": 2,
    "wins_hist": {"3": 2},
    "missing_repair": 0,
    "missing_groups": 0,
    "remaining_undone": 0,
}


def test_verify_requests_every_partitioned_family(tmp_path, monkeypatch):
    supervisor = _supervisor(tmp_path)
    captured: list[list[str]] = []

    def fake_run(command, *, check=True, timeout=None):
        captured.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(_COMPLETE_SUMMARY)
        )

    monkeypatch.setattr(supervisor, "run", fake_run)
    supervisor.verify()

    command = captured[0]
    assert "--prefix" in command, command
    prefixes = command[command.index("--prefix") + 1].split(",")
    # Must match the partitioner's selection exactly. Empty means every train
    # task; without passing it explicitly the verifier falls back to its own
    # genb_ default and the completion test / stall score would only ever see
    # breadth tasks (1009 of 1289).
    assert prefixes == PARTITION_TASK_PREFIXES.split(",")

    from kore.tasks.registry import train_tasks

    task_ids = {task.task_id for task in train_tasks()}
    covered = {t for t in task_ids if any(t.startswith(p) for p in prefixes)}
    assert covered == task_ids


def test_supervisor_prefixes_match_the_partitioner_default():
    """The verified family set is the partitioner's, read from its own source.

    ``spur_partition.py`` keeps the list inline in ``add_argument`` with no
    importable constant, so this reads that default out of the AST rather than
    restating it. If the partitioner starts sharding a ninth family, this fails
    instead of letting the supervisor quietly under-verify.
    """
    source = (REPO_ROOT / "scripts" / "spur_partition.py").read_text()
    defaults = [
        keyword.value.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and getattr(node.args[0], "value", None) == "--prefix"
        for keyword in node.keywords
        if keyword.arg == "default" and isinstance(keyword.value, ast.Constant)
    ]

    assert defaults == [PARTITION_TASK_PREFIXES]


def test_partitioner_selection_covers_every_train_task():
    """The partitioned scope must equal the registry's train set, exactly.

    A prefix allowlist on top of the taxonomy split can only lose work, and did:
    ``genb_`` alone reached 1009 of 1289 train tasks, and an eight-family list
    reached 1278, silently excluding eleven hand-authored vendor-lane tasks.
    Asserting equality (not a superset of breadth) is what makes a future
    narrowing fail here instead of in a campaign that reports COMPLETE early.
    """
    from kore.tasks.registry import train_tasks

    task_ids = {task.task_id for task in train_tasks()}
    prefixes = split_prefixes(PARTITION_TASK_PREFIXES)
    selected = {tid for tid in task_ids if any(tid.startswith(p) for p in prefixes)}

    missing = sorted(task_ids - selected)
    assert not missing, f"partitioner would never shard {len(missing)} train tasks: {missing[:10]}"
    assert selected == task_ids

    breadth = {tid for tid in task_ids if tid.startswith("genb_")}
    assert selected > breadth


def test_kf_verify_selects_every_partitioned_family(tmp_path):
    """End-to-end: the CLI honors a comma-separated prefix list."""
    result = subprocess.run(
        [
            "python",
            "scripts/_kf_verify.py",
            str(tmp_path),
            "1",
            "--prefix",
            PARTITION_TASK_PREFIXES,
            "--json",
            "--cleanup-out",
            str(tmp_path / "cleanup.txt"),
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    summary = _json_line(result.stdout)

    breadth = subprocess.run(
        [
            "python",
            "scripts/_kf_verify.py",
            str(tmp_path),
            "1",
            "--json",
            "--cleanup-out",
            str(tmp_path / "breadth.txt"),
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert breadth.returncode == 0, breadth.stderr
    assert summary["tasks"] > _json_line(breadth.stdout)["tasks"]


# --------------------------------------------------------------------------- #
# scoring helpers
# --------------------------------------------------------------------------- #
def test_progress_score_counts_partial_wins_and_base_stages():
    summary = {
        "tasks": 4,
        "wins_hist": {"0": 1, "1": 1, "2": 1, "3": 1},
        "missing_repair": 1,
        "missing_groups": 2,
    }

    assert progress_score(summary) == (0 + 1 + 2 + 3) + (8 - 1 - 2)


def test_json_line_uses_last_json_object():
    output = 'noise\n{"first": 1}\nmore noise\n{"last": 2}\n'

    assert _json_line(output) == {"last": 2}
