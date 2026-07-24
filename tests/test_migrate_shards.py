"""Non-destructive legacy shard inventory and quarantine migration tests."""

from __future__ import annotations

import json

import pytest

from kore.data.migrate_shards import inventory_roots
from kore.data.schemas import RepairRecord, read_jsonl


def _legacy_repair() -> dict:
    record = RepairRecord(
        task_id="task",
        failure_class="compile_fail",
        parent_hash="parent",
        error_text="legacy label; validity not re-derived",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "FULL_KERNEL:\ndef k(): pass"},
        ],
    ).to_dict()
    record.pop("schema_version")
    return record


def test_dry_run_inventories_without_mutation_and_marks_unknown(tmp_path):
    root = tmp_path / "full14b"
    path = root / "repair" / "task.jsonl"
    path.parent.mkdir(parents=True)
    original = (json.dumps(_legacy_repair()) + "\n").encode()
    path.write_bytes(original)

    report = inventory_roots([root])

    assert report.dry_run is True
    assert report.summary()["record_shards"] == 1
    assert report.summary()["records"] == 1
    assert report.summary()["proposed_records"] == 1
    assert report.summary()["production_valid_paths"] == 0
    assert path.read_bytes() == original
    assert list(root.rglob("*.bak")) == []


def test_apply_requires_separate_root_preserves_source_and_exact_backup(tmp_path):
    root = tmp_path / "full14b"
    path = root / "repair" / "task.jsonl"
    path.parent.mkdir(parents=True)
    original = (json.dumps(_legacy_repair()) + "\n").encode()
    path.write_bytes(original)
    output = tmp_path / "quarantine"

    report = inventory_roots([root], output_root=output, apply=True)

    item = report.paths[0]
    assert path.read_bytes() == original
    assert item.backup_path and item.output_path
    assert open(item.backup_path, "rb").read() == original
    migrated = read_jsonl(
        item.output_path, typed=False, mode="generic_training_row")
    assert migrated[0]["semantic_schema"]["semantic_validity"] == "unknown"
    assert migrated[0]["data_lane_version"] == "kore-legacy-quarantine-v1"
    with pytest.raises(Exception, match="production lane"):
        read_jsonl(item.output_path, mode="production_strict")
    assert not list(output.rglob("*.complete.json"))


def test_malformed_lines_report_path_and_line_and_block_apply(tmp_path):
    root = tmp_path / "full14b"
    path = root / "wins" / "task.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{broken\n")
    output = tmp_path / "quarantine"

    report = inventory_roots([root], output_root=output, apply=True)

    issue = report.paths[0].issues[0]
    assert issue.path == str(path)
    assert issue.line == 1
    assert issue.stage == "parse"
    assert report.paths[0].output_path is None
    assert report.paths[0].backup_path is not None


def test_generic_jsonl_is_inventoried_and_copied_byte_exact(tmp_path):
    root = tmp_path / "full14b"
    path = root / "events.jsonl"
    path.parent.mkdir(parents=True)
    original = b'{"event":"start","value":1}\n'
    path.write_bytes(original)
    output = tmp_path / "quarantine"

    report = inventory_roots([root], output_root=output, apply=True)

    assert report.paths[0].lane == "generic"
    assert open(report.paths[0].output_path, "rb").read() == original
    assert path.read_bytes() == original
