"""Focused CPU tests for strict schemas and durable generic JSONL writes."""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
from pathlib import Path

import pytest

import kore.data.schemas as schemas
from kore.agent.schema import AgenticTrajectoryRecord
from kore.data.schemas import (
    JsonlValidationError,
    RECORD_SCHEMA_VERSION,
    RecordValidationError,
    RepairRecord,
    RankedGroupRecord,
    WinRecord,
    read_jsonl,
    read_jsonl_legacy,
    record_from_dict,
    stamp_legacy_record_unknown,
    stamp_production_record,
    stamp_source_only_record,
    validate_record_dict,
    write_jsonl,
)


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "optimize"},
        {"role": "assistant", "content": "FULL_KERNEL:\n```python\ndef k(): pass\n```"},
    ]


def _repair(task_id: str = "task") -> RepairRecord:
    return RepairRecord(
        task_id=task_id,
        failure_class="compile_fail",
        parent_hash="abc123",
        error_text="compile failed",
        messages=_messages(),
        child_snr_db=40.0,
    )


def _group(task_id: str = "task") -> RankedGroupRecord:
    return RankedGroupRecord(
        task_id=task_id,
        parent_id="parent",
        candidates=[
            {"source": "def fast(): pass", "rank": 0, "wall_us": 1.0, "snr_db": 40.0},
            {"source": "def slow(): pass", "rank": 1, "wall_us": 2.0, "snr_db": 39.0},
        ],
        preferences=[[0, 1]],
    )


def _win(task_id: str = "task") -> WinRecord:
    return WinRecord(
        task_id=task_id,
        trajectory=_messages(),
        initial_wall_us=2.0,
        final_wall_us=1.0,
        speedup=2.0,
        final_source="def k(): pass",
        snr_db=40.0,
    )


def _agentic(task_id: str = "task") -> AgenticTrajectoryRecord:
    return AgenticTrajectoryRecord(
        task_id=task_id,
        messages=_messages()
        + [{"role": "tool", "content": '{"ok":true,"tool":"keep"}'}],
        tool_trace=[{"turn": 0, "name": "keep", "arguments": {}, "result": {"ok": True}}],
        best_kernel="def k(): pass",
        best_reward=1.0,
        turns_to_best=0,
        success=True,
        reflections=[],
        phase_trace=[],
        provenance={"category": "success"},
    )


def _competing_jsonl_writer(path: str, writer: str, start) -> None:
    start.wait(5)
    write_jsonl(path, [{"writer": writer, "index": index} for index in range(200)])


def test_typed_roundtrip_all_four_record_types(tmp_path):
    records = [_repair(), _group(), _win(), _agentic()]
    path = tmp_path / "records.jsonl"

    write_jsonl(path, records)

    loaded = read_jsonl(path, mode="generic_training_row")
    assert loaded == records
    assert [type(record) for record in loaded] == [
        RepairRecord,
        RankedGroupRecord,
        WinRecord,
        AgenticTrajectoryRecord,
    ]
    raw = read_jsonl(path, typed=False, mode="generic_training_row")
    assert {record["schema_version"] for record in raw} == {RECORD_SCHEMA_VERSION}


def test_strict_validation_rejects_non_dict_unknown_and_wrong_version():
    with pytest.raises(RecordValidationError, match="must be an object"):
        validate_record_dict(["not", "an", "object"])

    unknown = _repair().to_dict()
    unknown["type"] = "mystery"
    with pytest.raises(RecordValidationError, match="unknown record type"):
        validate_record_dict(unknown)

    unversioned = _repair().to_dict()
    unversioned.pop("schema_version")
    with pytest.raises(RecordValidationError, match="schema_version"):
        record_from_dict(unversioned)

    float_version = _repair().to_dict()
    float_version["schema_version"] = 1.0
    with pytest.raises(RecordValidationError, match="schema_version"):
        validate_record_dict(float_version)


@pytest.mark.parametrize("bad_number", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_rejected_recursively_and_on_write(tmp_path, bad_number):
    record = _repair().to_dict()
    record["nested_provenance"] = {"value": [bad_number]}
    with pytest.raises(RecordValidationError, match="NaN and infinity"):
        validate_record_dict(record)
    with pytest.raises(RecordValidationError, match="NaN and infinity"):
        write_jsonl(tmp_path / "bad.jsonl", [record])


@pytest.mark.parametrize(
    "message",
    [
        "not-an-object",
        {"role": "assistant"},
        {"role": "unknown", "content": "x"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": ["not", "text"]},
    ],
)
def test_malformed_messages_rejected(message):
    record = _repair().to_dict()
    record["messages"] = [message]
    with pytest.raises(RecordValidationError, match="record.messages"):
        validate_record_dict(record)


def test_invalid_ranks_and_preferences_rejected():
    duplicate_rank = _group().to_dict()
    duplicate_rank["candidates"][1]["rank"] = 0
    with pytest.raises(RecordValidationError, match="ranks must be unique"):
        validate_record_dict(duplicate_rank)

    out_of_range = _group().to_dict()
    out_of_range["preferences"] = [[0, 2]]
    with pytest.raises(RecordValidationError, match="out of range"):
        validate_record_dict(out_of_range)

    reversed_pair = _group().to_dict()
    reversed_pair["preferences"] = [[1, 0]]
    with pytest.raises(RecordValidationError, match="better"):
        validate_record_dict(reversed_pair)


def test_empty_win_source_and_wrong_task_binding_rejected():
    empty_source = _win().to_dict()
    empty_source["final_source"] = "  "
    with pytest.raises(RecordValidationError, match="final_source"):
        validate_record_dict(empty_source)

    with pytest.raises(RecordValidationError, match="expected 'other'"):
        validate_record_dict(_repair().to_dict(), expected_task_id="other")


def test_strict_reader_fails_closed_and_legacy_reader_is_explicit(tmp_path):
    legacy = _repair().to_dict()
    legacy.pop("schema_version")
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(legacy) + "\n{broken\n", encoding="utf-8")

    with pytest.raises(JsonlValidationError, match="schema_version"):
        read_jsonl(path, mode="production_strict")

    recovered = read_jsonl_legacy(path)
    assert recovered == [_repair()]


def test_atomic_write_preserves_old_target_when_replace_crashes(tmp_path, monkeypatch):
    path = tmp_path / "records.jsonl"
    old_bytes = b'{"old":true}\n'
    path.write_bytes(old_bytes)

    def fail_replace(_source, _target):
        raise OSError("simulated crash before replace")

    monkeypatch.setattr(schemas.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated crash"):
        write_jsonl(path, [{"new": True}])

    assert path.read_bytes() == old_bytes
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_atomic_write_fsyncs_file_and_directory_with_local_unique_temp(
    tmp_path, monkeypatch
):
    path = tmp_path / "records.jsonl"
    real_fsync = schemas.os.fsync
    real_replace = schemas.os.replace
    fsync_calls: list[int] = []
    replacements: list[tuple[Path, Path]] = []

    def tracking_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    def tracking_replace(source, target):
        replacements.append((Path(source), Path(target)))
        return real_replace(source, target)

    monkeypatch.setattr(schemas.os, "fsync", tracking_fsync)
    monkeypatch.setattr(schemas.os, "replace", tracking_replace)
    write_jsonl(path, [{"ok": True}])

    assert len(fsync_calls) >= 2
    assert len(replacements) == 1
    source, target = replacements[0]
    assert source.parent == target.parent == tmp_path
    assert source != target
    assert source.name.startswith(f".{path.name}.")


def test_concurrent_generic_writers_never_publish_interleaved_jsonl(tmp_path):
    path = tmp_path / "concurrent.jsonl"
    ctx = mp.get_context("fork")
    start = ctx.Event()
    processes = [
        ctx.Process(target=_competing_jsonl_writer, args=(str(path), writer, start))
        for writer in ("left", "right")
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    records = read_jsonl(
        path, typed=False, mode="generic_training_row")
    assert len(records) == 200
    assert {record["writer"] for record in records} in ({"left"}, {"right"})
    assert {record["index"] for record in records} == set(range(200))
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_record_validation_does_not_mutate_input():
    record = _group().to_dict()
    before = copy.deepcopy(record)
    validate_record_dict(record)
    assert record == before


def test_reader_mode_is_mandatory_and_production_is_stric(tmp_path):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [_repair()])
    with pytest.raises(TypeError, match="mode"):
        read_jsonl(path)
    assert read_jsonl(path, mode="generic_training_row") == [_repair()]
    with pytest.raises(JsonlValidationError, match="production lane"):
        read_jsonl(path, mode="production_strict")


def test_policy_training_requires_transcript_and_contract_provenance(tmp_path):
    empty = _win()
    empty.trajectory = []
    stamped = stamp_production_record(
        empty, provenance_id="contract", evaluation_id="evaluation")
    with pytest.raises(RecordValidationError, match="must not be empty"):
        validate_record_dict(stamped, production=True)

    source_only = stamp_source_only_record(
        empty,
        provenance_id="champion-source",
        evaluation_id="external-eval-1",
        source_status="verified_external",
    )
    validate_record_dict(source_only, production=True)


def test_legacy_migration_marks_semantics_unknown_and_stays_quarantined():
    legacy = _group().to_dict()
    legacy.pop("schema_version")
    migrated = stamp_legacy_record_unknown(legacy)
    assert migrated["candidate_outcome_schema"] == {
        "name": "candidate_outcome_legacy_v1",
        "version": 1,
        "semantic_validity": "unknown",
    }
    with pytest.raises(RecordValidationError, match="production lane"):
        validate_record_dict(migrated, production=True)
