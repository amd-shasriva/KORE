from __future__ import annotations

import json

import pytest

from scripts._kf_verify import verify


def _write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_verify_requires_distinct_wins_and_completed_base_shards(tmp_path):
    task = "genb_task"
    _write(
        tmp_path / "wins" / f"{task}.jsonl",
        [
            {"final_source": "same"},
            {"final_source": " same "},
            {"final_source": "different"},
        ],
    )
    _write(tmp_path / "repair" / f"{task}.jsonl", [{"type": "repair"}])
    _write(tmp_path / "groups" / f"{task}.jsonl", [{"type": "ranked_group"}])

    summary, undone = verify(tmp_path, [task], target=2)

    assert summary["fully_complete"] == 1
    assert summary["wins_hist"] == {2: 1}
    assert undone == []


def test_verify_treats_inprogress_shard_as_incomplete(tmp_path):
    task = "genb_task"
    _write(tmp_path / "wins" / f"{task}.jsonl", [{"final_source": "one"}])
    repair = tmp_path / "repair" / f"{task}.jsonl"
    _write(repair, [{"type": "repair"}])
    repair.with_suffix(".jsonl.inprogress").write_text("{}\n")
    _write(tmp_path / "groups" / f"{task}.jsonl", [{"type": "ranked_group"}])

    summary, undone = verify(tmp_path, [task], target=1)

    assert summary["missing_repair"] == 1
    assert undone == [task]


def test_verify_fails_closed_on_malformed_base_jsonl(tmp_path):
    path = tmp_path / "repair" / "genb_task.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{broken\n")

    with pytest.raises(RuntimeError, match="invalid JSONL"):
        verify(tmp_path, ["genb_task"], target=1)
