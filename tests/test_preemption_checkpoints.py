from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kore.data.schemas import (
    RankedGroupRecord,
    RepairRecord,
    WinRecord,
    read_jsonl,
)
from scripts.complete_base import _shard_done, complete_one
from scripts.deepen_wins import _load_existing, deepen_one


class _FakeEnv:
    def __init__(self, task):
        self.task = task


def _repair(parent_hash: str) -> RepairRecord:
    return RepairRecord(
        task_id="task",
        failure_class="compile_fail",
        parent_hash=parent_hash,
        error_text="broken",
        messages=[],
    )


def _group(parent_id: str) -> RankedGroupRecord:
    return RankedGroupRecord(
        task_id="task",
        parent_id=parent_id,
        candidates=[],
        preferences=[],
    )


def _win(source: str) -> WinRecord:
    return WinRecord(
        task_id="task",
        trajectory=[],
        initial_wall_us=2.0,
        final_wall_us=1.0,
        speedup=2.0,
        final_source=source,
    )


def test_deepen_checkpoints_each_win_before_preemption(tmp_path, monkeypatch):
    calls = 0

    def generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [_win("kernel-one")]
        raise KeyboardInterrupt

    monkeypatch.setattr("kore.tasks.registry.get_task", lambda _: SimpleNamespace())
    monkeypatch.setattr("kore.env.kore_env.KoreEnv", _FakeEnv)
    monkeypatch.setattr("kore.data.gen_wins.generate_wins", generate)

    with pytest.raises(KeyboardInterrupt):
        deepen_one("task", tmp_path, 3, 8, object(), SimpleNamespace())

    records = read_jsonl(tmp_path / "wins" / "task.jsonl", typed=False)
    assert [record["final_source"] for record in records] == ["kernel-one"]


def test_deepen_existing_count_matches_distinct_partition_count(tmp_path):
    path = tmp_path / "wins.jsonl"
    records = [
        _win("same").to_dict(),
        _win("same").to_dict(),
        _win("").to_dict(),
        _win("different").to_dict(),
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    existing, seen = _load_existing(path)

    assert [record["final_source"] for record in existing] == ["same", "different"]
    assert len(seen) == 2


def test_deepen_fails_closed_on_malformed_existing_shard(tmp_path):
    path = tmp_path / "wins.jsonl"
    path.write_text("{broken\n")

    with pytest.raises(Exception, match="Expecting property name"):
        _load_existing(path)


def test_base_rejects_malformed_nonempty_shard(tmp_path):
    path = tmp_path / "repair" / "task.jsonl"
    path.parent.mkdir()
    path.write_text("{broken\n")

    with pytest.raises(Exception, match="Expecting property name"):
        _shard_done(tmp_path, "task", "repair")


def test_base_resumes_from_each_checkpointed_record(tmp_path, monkeypatch):
    monkeypatch.setattr("kore.tasks.registry.get_task", lambda _: SimpleNamespace())
    monkeypatch.setattr("kore.env.kore_env.KoreEnv", _FakeEnv)

    def interrupted_repairs(*args, on_record, **kwargs):
        on_record(_repair("one"))
        raise KeyboardInterrupt

    monkeypatch.setattr("kore.data.gen_repair.generate_repairs", interrupted_repairs)
    monkeypatch.setattr("kore.data.gen_groups.generate_groups", lambda *a, **k: [])

    with pytest.raises(KeyboardInterrupt):
        complete_one("task", tmp_path, 2, 1, 2, object())

    repair_path = tmp_path / "repair" / "task.jsonl"
    assert len(read_jsonl(repair_path, typed=False)) == 1
    assert (tmp_path / "repair" / "task.jsonl.inprogress").exists()

    def resumed_repairs(*args, n, on_record, **kwargs):
        assert n == 1
        record = _repair("two")
        on_record(record)
        return [record]

    def resumed_groups(*args, n_parents, on_record, **kwargs):
        assert n_parents == 1
        record = _group("parent")
        on_record(record)
        return [record]

    monkeypatch.setattr("kore.data.gen_repair.generate_repairs", resumed_repairs)
    monkeypatch.setattr("kore.data.gen_groups.generate_groups", resumed_groups)

    status, counts = complete_one("task", tmp_path, 2, 1, 2, object())

    assert status == "done"
    assert counts == {"repair": 1, "groups": 1}
    assert len(read_jsonl(repair_path, typed=False)) == 2
    assert len(read_jsonl(tmp_path / "groups" / "task.jsonl", typed=False)) == 1
    assert not (tmp_path / "repair" / "task.jsonl.inprogress").exists()
    assert not (tmp_path / "groups" / "task.jsonl.inprogress").exists()
