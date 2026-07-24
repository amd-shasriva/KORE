"""CPU tests for the parallel datagen sharding + resume logic (no GPU/teacher)."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time

import pytest

from kore.data.schemas import RepairRecord, read_jsonl
from kore.data.generation_identity import (
    DATA_LANE_VERSION,
    GENERATION_IDENTITY_VERSION,
    identity_digest,
)
from kore.data.parallel_datagen import (
    AGENTIC_KINDS,
    DATAGEN_KINDS,
    DatagenRunError,
    CompletionGate,
    CompletionStatus,
    ShardIncompleteError,
    ShardContractConflict,
    _collect_worker_results,
    build_generator_contract,
    claim_shard,
    run_parallel_datagen,
    shard_done,
    shard_receipt_path,
    shard_tasks,
    write_completed_shard,
)


_COUNTS = {
    "n_repair": 2,
    "n_parents": 1,
    "k": 2,
    "wins_gens": 4,
    "n_agentic": 1,
    "max_tool_turns": 3,
}


def _identity(kind: str, marker: str = "identity-v1") -> dict:
    value = {
        "identity_version": GENERATION_IDENTITY_VERSION,
        "data_lane_version": DATA_LANE_VERSION,
        "kind": kind,
        "task": {"task_id": "t0", "digest": marker},
        "code": {"digest": marker, "files": {"generator.py": marker}},
        "evaluation": {"digest": marker},
        "teacher": {
            "backend": "stub",
            "model": "stub-model",
            "immutable_revision": marker,
        },
        "hardware": {"architecture": "test", "hardware_id": marker},
        "software": {"python": "test"},
        "seeds": {"generator_seed": 0},
        "behavioral_environment": {},
    }
    value["digest"] = identity_digest(value)
    return value


def _contract(kind: str, identity_marker: str = "identity-v1",
              **count_overrides) -> dict:
    counts = {**_COUNTS, **count_overrides}
    return build_generator_contract(
        kind,
        counts,
        teacher_kind="stub",
        model_teacher="stub-model",
        generation_identity=_identity(kind, identity_marker),
    )


def _repair(task_id: str = "t0", parent_hash: str = "parent") -> RepairRecord:
    return RepairRecord(
        task_id=task_id,
        failure_class="compile_fail",
        parent_hash=parent_hash,
        error_text="compile failed",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "FULL_KERNEL:\ndef k(): pass"},
        ],
    )


def _concurrent_claim_writer(root: str, marker: str, start, result_q) -> None:
    try:
        contract = _contract("repair")
        start.wait(5)
        with claim_shard(
            root, "t0", "repair", contract=contract,
            gate=CompletionGate.TERMINAL_NONEMPTY,
        ) as claimed:
            if claimed:
                time.sleep(0.1)
                write_completed_shard(
                    root,
                    "t0",
                    "repair",
                    [_repair(parent_hash=marker)],
                    contract=contract,
                )
        result_q.put(("ok", claimed))
    except BaseException as exc:  # pragma: no cover - asserted via child result
        result_q.put(("error", repr(exc)))
        raise


def _exit_nonzero() -> None:
    os._exit(17)


def _send_events(result_q, events) -> None:
    for event in events:
        result_q.put(event)


def _summary() -> dict:
    return {"done": 0, "skip": 0, "error": 0, "records": 0, "tasks": 1}


def test_shard_balanced_and_complete():
    ids = [f"t{i}" for i in range(17)]
    shards = shard_tasks(ids, 8)
    assert len(shards) == 8
    # every task assigned exactly once
    flat = [t for s in shards for t in s]
    assert sorted(flat) == sorted(ids)
    # balanced within 1
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1


def test_shard_fewer_tasks_than_workers():
    shards = shard_tasks(["a", "b", "c"], 8)
    assert len(shards) == 3  # empty shards dropped
    assert sorted(t for s in shards for t in s) == ["a", "b", "c"]


def test_shard_single_worker():
    shards = shard_tasks(["a", "b", "c"], 1)
    assert shards == [["a", "b", "c"]]


def test_shard_done_never_trusts_nonempty_bytes(tmp_path):
    (tmp_path / "repair").mkdir()
    done = tmp_path / "repair" / "t0.jsonl"
    done.write_text('{"type":"repair"}\n')
    assert shard_done(
        tmp_path, "t0", "repair", gate="quota_only") is False
    empty = tmp_path / "repair" / "t1.jsonl"
    empty.write_text("")
    assert shard_done(
        tmp_path, "t1", "repair", gate="quota_only") is False
    assert shard_done(
        tmp_path, "t2", "repair", gate="quota_only") is False


def test_valid_receipt_enables_resume_and_binds_contract(tmp_path):
    contract = _contract("repair")
    with claim_shard(
        tmp_path, "t0", "repair", contract=contract, gate="quota_only"
    ) as claimed:
        assert claimed is True
        assert write_completed_shard(
            tmp_path,
            "t0",
            "repair",
            [_repair(parent_hash="one"), _repair(parent_hash="two")],
            contract=contract,
        ) == 2

    path = tmp_path / "repair" / "t0.jsonl"
    receipt = shard_receipt_path(path)
    assert receipt.exists()
    assert shard_done(
        tmp_path, "t0", "repair", contract=contract, gate="quota_only") is True
    with claim_shard(
        tmp_path, "t0", "repair", contract=contract, gate="quota_only"
    ) as claimed:
        assert claimed is False

    different_contract = _contract("repair", n_repair=3)
    assert shard_done(
        tmp_path, "t0", "repair", contract=different_contract,
        gate="quota_only") is False


def test_valid_work_is_not_replaced_by_different_contract(tmp_path):
    contract = _contract("repair")
    with claim_shard(
        tmp_path, "t0", "repair", contract=contract,
        gate="terminal_nonempty"):
        write_completed_shard(
            tmp_path, "t0", "repair", [_repair()], contract=contract)
    path = tmp_path / "repair" / "t0.jsonl"
    receipt = shard_receipt_path(path)
    before = (path.read_bytes(), receipt.read_bytes())

    different_contract = _contract("repair", n_repair=9)
    with pytest.raises(ShardContractConflict, match="different generator contract"):
        with claim_shard(
            tmp_path, "t0", "repair", contract=different_contract,
            gate="terminal_nonempty",
        ):
            pytest.fail("conflicting shard must not be claimed")

    assert (path.read_bytes(), receipt.read_bytes()) == before


def test_truncation_and_receipt_tampering_invalidate_completion(tmp_path):
    contract = _contract("repair")
    with claim_shard(
        tmp_path, "t0", "repair", contract=contract, gate="quota_only"):
        write_completed_shard(
            tmp_path,
            "t0",
            "repair",
            [_repair(parent_hash="one"), _repair(parent_hash="two")],
            contract=contract,
        )
    path = tmp_path / "repair" / "t0.jsonl"
    original = path.read_bytes()
    path.write_bytes(original[:-5])
    assert shard_done(
        tmp_path, "t0", "repair", contract=contract, gate="quota_only") is False

    # Restore data, then corrupt the independently checked count.
    path.write_bytes(original)
    receipt_path = shard_receipt_path(path)
    receipt = json.loads(receipt_path.read_text())
    receipt["record_count"] += 1
    receipt_path.write_text(json.dumps(receipt) + "\n")
    assert shard_done(
        tmp_path, "t0", "repair", contract=contract, gate="quota_only") is False


def test_wrong_task_id_cannot_be_published(tmp_path):
    contract = _contract("repair")
    with claim_shard(
        tmp_path, "t0", "repair", contract=contract, gate="quota_only"):
        with pytest.raises(Exception, match="task_id"):
            write_completed_shard(
                tmp_path,
                "t0",
                "repair",
                [_repair(task_id="wrong")],
                contract=contract,
            )
    assert shard_done(
        tmp_path, "t0", "repair", contract=contract, gate="quota_only") is False


def test_empty_receipt_is_auditable_but_never_generically_complete(tmp_path):
    contract = _contract("wins")
    with claim_shard(
        tmp_path, "t0", "wins", contract=contract, gate="quota_only"):
        assert write_completed_shard(
            tmp_path, "t0", "wins", [], contract=contract) == 0

    path = tmp_path / "wins" / "t0.jsonl"
    assert path.read_bytes() == b""
    receipt = json.loads(shard_receipt_path(path).read_text())
    assert receipt["completion_status"] == "empty_failure"
    assert receipt["accepted_count"] == 0
    assert receipt["requested_count"] == 1
    assert shard_done(
        tmp_path, "t0", "wins", contract=contract, gate="quota_only") is False
    assert shard_done(
        tmp_path, "t0", "wins", contract=contract,
        gate="audit_any_status") is True


def test_under_quota_and_partial_statuses_require_explicit_gates(tmp_path):
    contract = _contract("repair")
    with claim_shard(
        tmp_path, "t0", "repair", contract=contract, gate="quota_only"
    ):
        write_completed_shard(
            tmp_path, "t0", "repair", [_repair()], contract=contract)
    receipt = json.loads(
        shard_receipt_path(tmp_path / "repair" / "t0.jsonl").read_text())
    assert receipt["completion_status"] == "bounded_attempts_exhausted"
    assert receipt["accepted_count"] == 1
    assert receipt["requested_count"] == 2
    assert not shard_done(
        tmp_path, "t0", "repair", contract=contract, gate="quota_only")
    assert shard_done(
        tmp_path, "t0", "repair", contract=contract,
        gate="terminal_nonempty")
    with pytest.raises(ShardIncompleteError, match="preserves 1 accepted"):
        with claim_shard(
            tmp_path, "t0", "repair", contract=contract, gate="quota_only"
        ):
            pass

def test_explicit_partial_checkpoint_gate(tmp_path):
    identity = _identity("repair", "partial")
    identity["task"]["task_id"] = "t1"
    identity["digest"] = identity_digest({
        key: value for key, value in identity.items() if key != "digest"})
    contract = build_generator_contract(
        "repair",
        _COUNTS,
        teacher_kind="stub",
        model_teacher="stub-model",
        generation_identity=identity,
    )
    with claim_shard(
        tmp_path, "t1", "repair", contract=contract, gate="quota_only"
    ):
        write_completed_shard(
            tmp_path,
            "t1",
            "repair",
            [_repair(task_id="t1")],
            contract=contract,
            completion_status=CompletionStatus.PARTIAL_CHECKPOINT,
        )
    assert not shard_done(
        tmp_path, "t1", "repair", contract=contract,
        gate="terminal_nonempty")
    assert shard_done(
        tmp_path, "t1", "repair", contract=contract,
        gate="include_partial_checkpoint")


def test_changed_generation_identity_conflicts_even_when_counts_match(tmp_path):
    contract = _contract("repair", identity_marker="identity-a")
    with claim_shard(
        tmp_path, "t0", "repair", contract=contract,
        gate="terminal_nonempty",
    ):
        write_completed_shard(
            tmp_path, "t0", "repair", [_repair()], contract=contract)
    changed = _contract("repair", identity_marker="identity-b")
    assert not shard_done(
        tmp_path, "t0", "repair", contract=changed,
        gate="terminal_nonempty")
    with pytest.raises(ShardContractConflict, match="different generator contract"):
        with claim_shard(
            tmp_path, "t0", "repair", contract=changed,
            gate="terminal_nonempty",
        ):
            pass


def test_pending_receipt_recovers_valid_post_replace_work(tmp_path):
    contract = _contract("repair")
    with claim_shard(
        tmp_path, "t0", "repair", contract=contract,
        gate="terminal_nonempty"):
        write_completed_shard(
            tmp_path, "t0", "repair", [_repair()], contract=contract)
    path = tmp_path / "repair" / "t0.jsonl"
    final = shard_receipt_path(path)
    pending = path.with_name(f"{path.name}.complete.pending.json")
    os.replace(final, pending)

    assert shard_done(
        tmp_path, "t0", "repair", contract=contract,
        gate="terminal_nonempty") is True
    assert final.exists()
    assert not pending.exists()


def test_recovery_replaces_stale_prior_receipt_after_data_publish(
    tmp_path, monkeypatch
):
    import kore.data.parallel_datagen as parallel

    contract = _contract("repair")
    with claim_shard(
        tmp_path, "t0", "repair", contract=contract, gate="quota_only"
    ):
        write_completed_shard(
            tmp_path,
            "t0",
            "repair",
            [_repair(parent_hash="old-1"), _repair(parent_hash="old-2")],
            contract=contract,
        )
    path = tmp_path / "repair" / "t0.jsonl"
    old_receipt = shard_receipt_path(path).read_bytes()
    real_replace = parallel._durable_replace
    failed = False

    def fail_final_receipt_once(source, target):
        nonlocal failed
        if (
            not failed
            and str(source).endswith(".complete.pending.json")
            and str(target).endswith(".complete.json")
        ):
            failed = True
            raise OSError("crash before replacing prior receipt")
        return real_replace(source, target)

    monkeypatch.setattr(parallel, "_durable_replace", fail_final_receipt_once)
    with pytest.raises(OSError, match="prior receipt"):
        write_completed_shard(
            tmp_path,
            "t0",
            "repair",
            [_repair(parent_hash="new-1"), _repair(parent_hash="new-2")],
            contract=contract,
        )
    assert shard_receipt_path(path).read_bytes() == old_receipt
    monkeypatch.setattr(parallel, "_durable_replace", real_replace)

    assert shard_done(
        tmp_path, "t0", "repair", contract=contract, gate="quota_only")
    records = read_jsonl(path, mode="production_strict")
    assert {record.parent_hash for record in records} == {"new-1", "new-2"}


def test_overlapping_processes_get_exactly_one_shard_owner(tmp_path):
    ctx = mp.get_context("fork")
    start = ctx.Event()
    result_q = ctx.Queue()
    processes = [
        ctx.Process(
            target=_concurrent_claim_writer,
            args=(str(tmp_path), marker, start, result_q),
        )
        for marker in ("left", "right")
    ]
    for process in processes:
        process.start()
    start.set()
    results = [result_q.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert sorted(results) == [("ok", False), ("ok", True)]
    assert shard_done(
        tmp_path, "t0", "repair", contract=_contract("repair"),
        gate="terminal_nonempty") is True
    records = read_jsonl(
        tmp_path / "repair" / "t0.jsonl", mode="production_strict")
    assert len(records) == 1
    assert records[0].parent_hash in {"left", "right"}


def test_spawn_orchestrator_resumes_receipted_work_and_deduplicates_tasks(
    tmp_path, monkeypatch
):
    from kore.tasks.registry import get_task

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    task = get_task("rmsnorm_aiter")
    contract = build_generator_contract(
        "repair",
        _COUNTS,
        teacher_kind="stub",
        model_teacher="stub-model",
        model_teacher_revision="stub-revision",
        task=task,
    )
    with claim_shard(
        tmp_path, task.task_id, "repair", contract=contract, gate="quota_only"):
        write_completed_shard(
            tmp_path,
            task.task_id,
            "repair",
            [
                _repair(task.task_id, "one"),
                _repair(task.task_id, "two"),
            ],
            contract=contract,
        )

    summary = run_parallel_datagen(
        [task.task_id, task.task_id],
        ["repair"],
        tmp_path,
        _COUNTS,
        n_workers=2,
        n_gpus=1,
        teacher_kind="stub",
        model_teacher="stub-model",
        model_teacher_revision="stub-revision",
        completion_gate="quota_only",
        log=lambda _message: None,
    )

    assert summary == {
        "done": 0,
        "skip": 1,
        "error": 0,
        "records": 0,
        "tasks": 1,
    }


def test_spawn_orchestrator_propagates_contract_conflict_nonzero(
    tmp_path, monkeypatch
):
    from kore.tasks.registry import get_task

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    task = get_task("rmsnorm_aiter")
    original_contract = build_generator_contract(
        "repair",
        _COUNTS,
        teacher_kind="stub",
        model_teacher="stub-model",
        model_teacher_revision="stub-revision",
        task=task,
    )
    with claim_shard(
        tmp_path, task.task_id, "repair", contract=original_contract,
        gate="quota_only",
    ):
        write_completed_shard(
            tmp_path,
            task.task_id,
            "repair",
            [_repair(task.task_id, "one"), _repair(task.task_id, "two")],
            contract=original_contract,
        )

    changed_counts = {**_COUNTS, "n_repair": 9}
    with pytest.raises(DatagenRunError, match="different generator contract"):
        run_parallel_datagen(
            [task.task_id],
            ["repair"],
            tmp_path,
            changed_counts,
            n_workers=1,
            n_gpus=1,
            teacher_kind="stub",
            model_teacher="stub-model",
            model_teacher_revision="stub-revision",
            completion_gate="quota_only",
            log=lambda _message: None,
        )

    assert shard_done(
        tmp_path, task.task_id, "repair", contract=original_contract,
        gate="quota_only") is True


def test_worker_death_terminates_collection_without_hanging():
    ctx = mp.get_context("fork")
    result_q = ctx.Queue()
    process = ctx.Process(target=_exit_nonzero)
    process.start()
    started = time.monotonic()
    with pytest.raises(DatagenRunError, match="exit code 17"):
        _collect_worker_results(
            [process],
            result_q,
            {("t0", "repair")},
            _summary(),
            poll_seconds=0.01,
            exit_grace_seconds=0.05,
        )
    process.join(timeout=2)
    assert time.monotonic() - started < 2


def test_lost_finish_sentinel_terminates_collection_without_hanging():
    ctx = mp.get_context("fork")
    result_q = ctx.Queue()
    event = {
        "event": "result",
        "worker_id": 0,
        "task_id": "t0",
        "kind": "repair",
        "status": "done",
        "records": 1,
    }
    process = ctx.Process(target=_send_events, args=(result_q, [event]))
    process.start()
    with pytest.raises(DatagenRunError, match="without a finish sentinel"):
        _collect_worker_results(
            [process],
            result_q,
            {("t0", "repair")},
            _summary(),
            poll_seconds=0.01,
            exit_grace_seconds=0.05,
        )
    process.join(timeout=2)


def test_partial_results_and_explicit_worker_errors_propagate():
    ctx = mp.get_context("fork")

    partial_q = ctx.Queue()
    partial = ctx.Process(
        target=_send_events,
        args=(partial_q, [{"event": "finished", "worker_id": 0}]),
    )
    partial.start()
    with pytest.raises(DatagenRunError, match="partial results"):
        _collect_worker_results(
            [partial],
            partial_q,
            {("t0", "repair")},
            _summary(),
            poll_seconds=0.01,
            exit_grace_seconds=0.05,
        )
    partial.join(timeout=2)

    error_q = ctx.Queue()
    error_event = {
        "event": "result",
        "worker_id": 0,
        "task_id": "t0",
        "kind": "repair",
        "status": "error",
        "records": 0,
        "error": "RuntimeError: boom",
    }
    errored = ctx.Process(target=_send_events, args=(error_q, [error_event]))
    errored.start()
    summary = _summary()
    with pytest.raises(DatagenRunError, match="boom"):
        _collect_worker_results(
            [errored],
            error_q,
            {("t0", "repair")},
            summary,
            poll_seconds=0.01,
            exit_grace_seconds=0.05,
        )
    errored.join(timeout=2)
    assert summary["error"] == 1


def test_kind_sets():
    assert DATAGEN_KINDS == ("repair", "groups", "wins")
    assert AGENTIC_KINDS == ("agentic",)
