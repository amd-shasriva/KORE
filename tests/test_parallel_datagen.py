"""CPU tests for the parallel datagen sharding + resume logic (no GPU/teacher)."""

from __future__ import annotations

from kore.data.parallel_datagen import (
    AGENTIC_KINDS,
    DATAGEN_KINDS,
    shard_done,
    shard_tasks,
)


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


def test_shard_done_resume(tmp_path):
    (tmp_path / "repair").mkdir()
    done = tmp_path / "repair" / "t0.jsonl"
    done.write_text('{"type":"repair"}\n')
    assert shard_done(tmp_path, "t0", "repair") is True      # non-empty -> done
    empty = tmp_path / "repair" / "t1.jsonl"
    empty.write_text("")
    assert shard_done(tmp_path, "t1", "repair") is False     # empty -> redo
    assert shard_done(tmp_path, "t2", "repair") is False     # missing -> redo


def test_kind_sets():
    assert DATAGEN_KINDS == ("repair", "groups", "wins")
    assert AGENTIC_KINDS == ("agentic",)
