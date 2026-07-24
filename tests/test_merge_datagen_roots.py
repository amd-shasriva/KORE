from __future__ import annotations

import json

import pytest

from scripts.merge_datagen_roots import _read_records, merge_records, merge_roots


def _write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_win_merge_is_destination_first_and_source_canonical(tmp_path):
    destination = [{"final_source": "same", "origin": "destination"}]
    source = [
        {"final_source": " same ", "origin": "source-duplicate"},
        {"final_source": "new", "origin": "source-new"},
    ]

    merged, added = merge_records("wins", destination, source)

    assert added == 1
    assert merged == [destination[0], source[1]]


def test_merge_roots_apply_is_atomic_and_idempotent(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    task = "genb_task"
    _write(source / "repair" / f"{task}.jsonl", [{"id": 1}, {"id": 2}])
    _write(destination / "repair" / f"{task}.jsonl", [{"id": 1}])

    dry_run = merge_roots(source, destination, kinds=("repair",), apply=False)
    first = merge_roots(source, destination, kinds=("repair",), apply=True)
    second = merge_roots(source, destination, kinds=("repair",), apply=True)

    assert dry_run["records_added"] == 1
    assert first["records_added"] == 1
    assert second["records_added"] == 0
    assert _read_records(destination / "repair" / f"{task}.jsonl") == [
        {"id": 1},
        {"id": 2},
    ]
    assert not list((destination / "repair").glob("*.tmp"))


def test_merge_fails_closed_on_malformed_source(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    path = source / "groups" / "genb_task.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{broken\n")

    with pytest.raises(RuntimeError, match="invalid JSONL"):
        merge_roots(source, destination, kinds=("groups",), apply=False)
