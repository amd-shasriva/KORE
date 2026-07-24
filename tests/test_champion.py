"""CPU tests for the champion re-evaluation anti-hack gate (verdict core + loader)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kore.data.schemas import WinRecord, stamp_source_only_record, write_jsonl
from kore.eval.champion import (
    Champion,
    ChampionReport,
    champion_verdict,
    held_out_shapes,
    load_champions,
    load_shape_manifests,
)
from kore.tasks.augment import (
    FrozenShapeSplit,
    boundary_regime,
    freeze_shape_split,
    shape_key,
    validate_frozen_split,
)
from kore.tasks.base import Shape
from kore.tasks.registry import get_task


def test_certified_when_survives():
    v = champion_verdict("t", claimed_speedup=1.5, measured_speedup=1.45,
                         correct=True, hack_free=True, high_variance=False)
    assert v.certified and not v.collapsed


def test_collapse_detected():
    # claimed 3.13x, re-measures 1.49x -> collapse (the classic hack signature).
    v = champion_verdict("t", claimed_speedup=3.13, measured_speedup=1.49,
                         correct=True, hack_free=True, high_variance=False,
                         collapse_ratio=0.7)
    assert v.collapsed and not v.certified
    assert "collapsed" in v.reason


def test_incorrect_rejected():
    v = champion_verdict("t", 2.0, 2.0, correct=False, hack_free=True,
                         high_variance=False)
    assert not v.certified and "incorrect" in v.reason


def test_hack_flagged_rejected():
    v = champion_verdict("t", 2.0, 2.0, correct=True, hack_free=False,
                         high_variance=False)
    assert not v.certified and "hack" in v.reason


def test_not_faster_rejected():
    v = champion_verdict("t", None, 0.8, correct=True, hack_free=True,
                         high_variance=False, min_speedup=1.0)
    assert not v.certified and "faster" in v.reason


def test_high_variance_rejected():
    v = champion_verdict("t", 2.0, 2.0, correct=True, hack_free=True,
                         high_variance=True)
    assert not v.certified and "variance" in v.reason


def test_no_claim_still_certifiable():
    # a champion with no claimed speedup is certified purely on the re-measurement.
    v = champion_verdict("t", None, 1.3, correct=True, hack_free=True,
                         high_variance=False)
    assert v.certified and not v.collapsed


def test_load_champions_keeps_best_per_task(tmp_path):
    p = tmp_path / "wins.jsonl"
    records = [
        WinRecord(task_id="a", trajectory=[], initial_wall_us=None,
                  final_wall_us=None, speedup=1.2, final_source="src_a_slow"),
        WinRecord(task_id="a", trajectory=[], initial_wall_us=None,
                  final_wall_us=None, speedup=1.9, final_source="src_a_fast"),
        WinRecord(task_id="b", trajectory=[], initial_wall_us=None,
                  final_wall_us=None, speedup=2.5, final_source="src_b"),
    ]
    write_jsonl(p, [
        stamp_source_only_record(
            record,
            provenance_id="champion-test",
            evaluation_id=f"champion-test:{index}",
            source_status="verified_external",
        )
        for index, record in enumerate(records)
    ])
    champs = {c.task_id: c for c in load_champions(str(p))}
    assert set(champs) == {"a", "b"}
    assert champs["a"].claimed_speedup == 1.9 and champs["a"].source == "src_a_fast"
    assert champs["b"].claimed_speedup == 2.5


def test_report_summary_counts():
    verdicts = [
        champion_verdict("a", 1.5, 1.5, correct=True, hack_free=True, high_variance=False),
        champion_verdict("b", 3.0, 1.0, correct=True, hack_free=True, high_variance=False),
    ]
    rep = ChampionReport(n_champions=2,
                         n_certified=sum(v.certified for v in verdicts),
                         n_collapsed=sum(v.collapsed for v in verdicts),
                         verdicts=verdicts)
    assert rep.n_certified == 1 and rep.n_collapsed == 1
    assert "CERTIFIED" in rep.summary() and "COLLAPSED" in rep.summary()


def test_hidden_shapes_are_built_after_and_disjoint_from_frozen_train_lane():
    task = get_task("softmax_bf16")
    split = freeze_shape_split(task)
    hidden = held_out_shapes(task, max_shapes=8, frozen_split=split)
    assert len(hidden) == 8
    assert {shape_key(shape) for shape in hidden}.isdisjoint(split.train_keys)
    assert all(shape.dims["N"] in {s.dims["N"] for s in task.shapes} for shape in hidden)
    assert len({boundary_regime(shape) for shape in hidden}) >= 3


def test_hidden_evaluation_requires_training_time_manifest():
    task = get_task("softmax_bf16")
    with pytest.raises(ValueError, match="training-time frozen"):
        held_out_shapes(task, max_shapes=8)
    split = freeze_shape_split(task)
    assert held_out_shapes(task, max_shapes=0, frozen_split=split) == []


def test_hidden_generation_rejects_task_changes_after_freeze():
    task = SimpleNamespace(
        task_id="snapshot",
        operation="softmax",
        raw={},
        shapes=[Shape("primary", {"M": 4096, "N": 4096})],
    )
    split = freeze_shape_split(task)
    task.shapes[0].dims["M"] = 8192
    with pytest.raises(ValueError, match="task digest changed"):
        held_out_shapes(task, max_shapes=1, frozen_split=split)


def test_manifest_round_trip_and_loader(tmp_path):
    task = get_task("softmax_bf16")
    split = freeze_shape_split(
        task, seed=9, created_at="2000-01-01T00:00:00+00:00")
    path = tmp_path / "softmax_bf16.json"
    split.write(path)
    loaded = FrozenShapeSplit.read(path)
    assert loaded.to_dict() == split.to_dict()
    assert loaded.schema_version == 1
    assert loaded.seed == 9
    assert loaded.created_at == "2000-01-01T00:00:00+00:00"
    assert loaded.code_identity and loaded.policy_digest and loaded.task_file_digest
    assert load_shape_manifests(str(tmp_path))["softmax_bf16"] == loaded
    assert held_out_shapes(task, frozen_split=path) == list(loaded.hidden_shapes)


def test_manifest_rejects_content_and_code_tampering(tmp_path):
    task = get_task("softmax_bf16")
    split = freeze_shape_split(task)
    value = split.to_dict()
    value["hidden_shapes"][0]["dims"]["M"] += 2
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="content hash"):
        FrozenShapeSplit.read(path)
    with pytest.raises(ValueError, match="code identity"):
        validate_frozen_split(task, split, code_identity="different-commit")
    incomplete = replace(split, train_shapes=split.train_shapes[:-1], content_hash="")
    incomplete = replace(incomplete, content_hash=incomplete.computed_hash())
    with pytest.raises(ValueError, match="train/augmentation universe"):
        validate_frozen_split(task, incomplete)


def test_manifest_rejects_task_file_change(tmp_path):
    task_file = tmp_path / "task.yaml"
    task_file.write_text("task_id: file_change\n")
    task = SimpleNamespace(
        task_id="file_change",
        operation="softmax",
        dtype="bf16",
        backend="triton",
        gpu_target="gfx950",
        dir=tmp_path,
        raw={"task_id": "file_change"},
        task_file_digest=hashlib.sha256(task_file.read_bytes()).hexdigest(),
        shapes=[Shape("primary", {"M": 4096, "N": 4096})],
    )
    split = freeze_shape_split(task)
    task_file.write_text("task_id: file_change\nchanged: true\n")
    with pytest.raises(ValueError, match="task file digest changed"):
        held_out_shapes(task, frozen_split=split)
