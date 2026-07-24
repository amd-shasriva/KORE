from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from scripts.spur_partition import (
    WorkItem,
    balanced_partition,
    distinct_wins,
    jsonl_record_count,
    shard_present,
    work_item,
)


def _item(task_id: str, cost: int) -> WorkItem:
    return WorkItem(
        task_id=task_id,
        cost=cost,
        needs_deepen=True,
        needs_base=False,
        wins=0,
        missing_repair=False,
        missing_groups=False,
    )


def test_distinct_wins_deduplicates_kernel_source(tmp_path):
    path = tmp_path / "wins.jsonl"
    records = [
        {"task_id": "x", "final_source": "same"},
        {"task_id": "x", "final_source": " same ", "speedup": 2.0},
        {"task_id": "x", "final_source": "different"},
        {"task_id": "x", "final_source": ""},
        {"task_id": "x", "type": "win"},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert distinct_wins(path) == 2


def test_jsonl_validation_fails_closed_on_non_object_record(tmp_path):
    path = tmp_path / "repair.jsonl"
    path.write_text("[]\n")

    with pytest.raises(RuntimeError, match="expected object"):
        jsonl_record_count(path)


def test_shard_present_requires_valid_record_and_no_marker(tmp_path):
    path = tmp_path / "repair" / "task.jsonl"
    path.parent.mkdir()
    path.write_text("\n")
    assert not shard_present(tmp_path, "repair", "task")

    path.write_text("{}\n")
    assert shard_present(tmp_path, "repair", "task")

    path.with_suffix(".jsonl.inprogress").write_text("{}\n")
    assert not shard_present(tmp_path, "repair", "task")


def test_balanced_partition_is_disjoint_complete_and_cost_balanced():
    items = [_item("a", 9), _item("b", 8), _item("c", 7), _item("d", 6)]

    shards = balanced_partition(items, 2)

    assigned = [item.task_id for shard in shards for item in shard]
    costs = [sum(item.cost for item in shard) for shard in shards]
    assert sorted(assigned) == ["a", "b", "c", "d"]
    assert len(assigned) == len(set(assigned))
    assert costs == [15, 15]


def test_work_item_models_deepen_and_base_gaps(tmp_path):
    for kind in ("wins", "repair", "groups"):
        (tmp_path / kind).mkdir()
    (tmp_path / "wins" / "task.jsonl").write_text(
        json.dumps({"final_source": "one"}) + "\n"
    )
    (tmp_path / "groups" / "task.jsonl").write_text("{}\n")

    item = work_item(tmp_path, "task", target=3)

    assert item is not None
    assert item.wins == 1
    assert item.needs_deepen
    assert item.needs_base
    assert item.missing_repair
    assert not item.missing_groups
    # need=2 -> six deepen attempts, plus repair weight nine.
    assert item.cost == 15


def test_runs_dir_can_be_isolated_per_spur_process(tmp_path):
    env = os.environ.copy()
    env["KORE_RUNS_DIR"] = str(tmp_path / "replay")

    out = subprocess.check_output(
        [sys.executable, "-c", "from kore.config import RUNS_DIR; print(RUNS_DIR)"],
        env=env,
        text=True,
    ).strip()

    assert out == str(tmp_path / "replay")


def test_inprogress_base_shard_remains_schedulable(tmp_path):
    for kind in ("wins", "repair", "groups"):
        (tmp_path / kind).mkdir()
    (tmp_path / "wins" / "task.jsonl").write_text(
        "".join(json.dumps({"final_source": f"k{i}"}) + "\n" for i in range(3))
    )
    repair = tmp_path / "repair" / "task.jsonl"
    repair.write_text("{}\n")
    repair.with_suffix(".jsonl.inprogress").write_text("{}\n")
    (tmp_path / "groups" / "task.jsonl").write_text("{}\n")

    item = work_item(tmp_path, "task", target=3)

    assert item is not None
    assert not item.needs_deepen
    assert item.needs_base
    assert item.missing_repair
