"""CPU-only tests for the co-evolution DistillationSink (RFT/expert-iteration).

No GPU / torch required: the sink is fed plain win dicts (with a stand-in
descriptor) and we assert the on-disk JSONL round-trips as valid ``WinRecord``s.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kore.data.prompts import extract_kernel
from kore.data.schemas import WinRecord, read_jsonl
from kore.policy.coevolve_distill import DistillationSink


@dataclass(frozen=True)
class _Desc:
    """Minimal TaskDescriptor stand-in (duck-typed: task_id/op/shape_regime)."""

    op: str = "rmsnorm"
    dtype: str = "bf16"
    shape_regime: str = "primary"

    @property
    def task_id(self) -> str:
        return f"genv_{self.op}_{self.dtype}"


def _win(op="rmsnorm", src="def k():\n    return 1", speedup=1.5,
         verified=True, **extra):
    d = {"descriptor": _Desc(op=op), "kernel_src": src, "speedup": speedup,
         "reward": speedup, "verified": verified}
    d.update(extra)
    return d


# --------------------------------------------------------------------------- #
# basic write / mapping
# --------------------------------------------------------------------------- #
def test_qualifying_win_is_written_and_roundtrips(tmp_path):
    path = tmp_path / "wins.jsonl"
    sink = DistillationSink(path)
    n = sink.record([_win(speedup=1.7)])
    assert n == 1
    assert path.exists()

    recs = read_jsonl(path)
    assert len(recs) == 1
    rec = recs[0]
    assert isinstance(rec, WinRecord)          # typed dispatch on read
    assert rec.type == "win"
    assert rec.task_id == "genv_rmsnorm_bf16"
    assert rec.speedup == 1.7
    assert rec.operation == "rmsnorm"
    assert rec.final_source == "def k():\n    return 1"
    assert rec.shape == "primary"
    assert rec.gpu == "gfx950"   # default arch retargeted to the KORE hardware (CDNA4)


def test_trajectory_is_valid_and_extracts_final_kernel(tmp_path):
    path = tmp_path / "wins.jsonl"
    src = "def add(a, b):\n    return a + b"
    DistillationSink(path).record([_win(src=src)])
    rec = read_jsonl(path)[0]

    assert len(rec.trajectory) == 2
    roles = [m["role"] for m in rec.trajectory]
    assert roles == ["user", "assistant"]
    for m in rec.trajectory:
        assert isinstance(m["content"], str) and m["content"]
    # the assistant turn uses the repo FULL_KERNEL convention and re-extracts.
    assert extract_kernel(rec.trajectory[-1]["content"]) == src


def test_callable_alias_matches_distillfn_signature(tmp_path):
    path = tmp_path / "wins.jsonl"
    sink = DistillationSink(path)
    # __call__ == record, so it drops straight into coevolve as distill_fn.
    n = sink([_win(speedup=1.2)])
    assert n == 1
    assert len(read_jsonl(path)) == 1


# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #
def test_subthreshold_speedup_is_filtered(tmp_path):
    path = tmp_path / "wins.jsonl"
    sink = DistillationSink(path, min_speedup=1.0)
    n = sink.record([_win(speedup=0.9, src="slow"), _win(speedup=1.0, src="tie"),
                     _win(speedup=2.0, src="fast")])
    assert n == 2                              # 0.9x dropped; 1.0x and 2.0x kept
    speeds = sorted(r.speedup for r in read_jsonl(path))
    assert speeds == [1.0, 2.0]


def test_custom_min_speedup(tmp_path):
    path = tmp_path / "wins.jsonl"
    sink = DistillationSink(path, min_speedup=1.5)
    n = sink.record([_win(speedup=1.2, src="a"), _win(speedup=1.6, src="b")])
    assert n == 1
    assert read_jsonl(path)[0].speedup == 1.6


def test_unverified_win_is_filtered_when_required(tmp_path):
    path = tmp_path / "wins.jsonl"
    sink = DistillationSink(path, require_verified=True)
    n = sink.record([_win(verified=False, src="a"), _win(verified=True, src="b")])
    assert n == 1
    assert read_jsonl(path)[0].final_source == "b"


def test_unverified_kept_when_not_required(tmp_path):
    path = tmp_path / "wins.jsonl"
    sink = DistillationSink(path, require_verified=False)
    n = sink.record([_win(verified=False, src="a"), _win(verified=False, src="b")])
    assert n == 2


def test_malformed_wins_are_skipped(tmp_path):
    path = tmp_path / "wins.jsonl"
    sink = DistillationSink(path)
    bad = [
        "not a dict",
        {"kernel_src": "x", "speedup": 1.5},            # no descriptor/task_id
        {"descriptor": _Desc(), "speedup": 1.5},         # no source
        {"descriptor": _Desc(), "kernel_src": "", "speedup": 1.5, "verified": True},  # empty src
        {"descriptor": _Desc(), "kernel_src": "y", "speedup": "fast", "verified": True},  # bad speedup
        _win(speedup=1.9, src="good"),                   # the only valid one
    ]
    n = sink.record(bad)
    assert n == 1
    assert read_jsonl(path)[0].final_source == "good"


def test_empty_batch_returns_zero_and_writes_nothing(tmp_path):
    path = tmp_path / "wins.jsonl"
    sink = DistillationSink(path)
    assert sink.record([]) == 0
    assert not path.exists()


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #
def test_dedup_keeps_best_speedup_within_batch(tmp_path):
    path = tmp_path / "wins.jsonl"
    sink = DistillationSink(path)
    src = "def k():\n    return 42"
    n = sink.record([_win(src=src, speedup=1.2), _win(src=src, speedup=1.9),
                     _win(src=src, speedup=1.5)])
    assert n == 1                              # same (task, source) collapses
    recs = read_jsonl(path)
    assert len(recs) == 1
    assert recs[0].speedup == 1.9              # best retained


def test_reemit_same_kernel_does_not_bloat(tmp_path):
    path = tmp_path / "wins.jsonl"
    sink = DistillationSink(path)
    assert sink.record([_win(src="same", speedup=1.3)]) == 1
    # re-emitting the identical kernel adds no NEW record.
    assert sink.record([_win(src="same", speedup=1.3)]) == 0
    assert len(read_jsonl(path)) == 1


def test_improved_speedup_updates_in_place_not_counted_new(tmp_path):
    path = tmp_path / "wins.jsonl"
    sink = DistillationSink(path)
    assert sink.record([_win(src="same", speedup=1.3)]) == 1
    assert sink.record([_win(src="same", speedup=2.5)]) == 0   # update, not new
    recs = read_jsonl(path)
    assert len(recs) == 1
    assert recs[0].speedup == 2.5              # best speedup kept


def test_same_source_different_task_are_distinct(tmp_path):
    path = tmp_path / "wins.jsonl"
    sink = DistillationSink(path)
    n = sink.record([_win(op="rmsnorm", src="shared", speedup=1.2),
                     _win(op="softmax", src="shared", speedup=1.2)])
    assert n == 2
    assert len({r.task_id for r in read_jsonl(path)}) == 2


# --------------------------------------------------------------------------- #
# persistence across re-instantiation
# --------------------------------------------------------------------------- #
def test_reinstantiation_dedups_against_existing_file(tmp_path):
    path = tmp_path / "wins.jsonl"
    DistillationSink(path).record([_win(src="a", speedup=1.2),
                                   _win(src="b", speedup=1.4)])

    sink2 = DistillationSink(path)             # loads prior contents
    # "a" already present (no new); "c" is new.
    n = sink2.record([_win(src="a", speedup=1.2), _win(src="c", speedup=1.6)])
    assert n == 1
    recs = read_jsonl(path)
    assert len(recs) == 3
    assert {r.final_source for r in recs} == {"a", "b", "c"}


def test_reinstantiation_improves_existing_best(tmp_path):
    path = tmp_path / "wins.jsonl"
    DistillationSink(path).record([_win(src="a", speedup=1.2)])

    sink2 = DistillationSink(path)
    assert sink2.record([_win(src="a", speedup=3.0)]) == 0   # improves, not new
    recs = read_jsonl(path)
    assert len(recs) == 1
    assert recs[0].speedup == 3.0


def test_parent_dirs_created(tmp_path):
    path = tmp_path / "nested" / "deeper" / "wins.jsonl"
    sink = DistillationSink(path)
    sink.record([_win(speedup=1.5)])
    assert path.exists()


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def test_stats_reports_count_tasks_and_speedups(tmp_path):
    path = tmp_path / "wins.jsonl"
    sink = DistillationSink(path)
    sink.record([
        _win(op="rmsnorm", src="a", speedup=1.0),
        _win(op="rmsnorm", src="b", speedup=2.0),
        _win(op="softmax", src="c", speedup=3.0),
    ])
    st = sink.stats()
    assert st["count"] == 3
    assert st["unique_tasks"] == 2
    assert st["mean_speedup"] == pytest.approx(2.0)
    assert st["median_speedup"] == pytest.approx(2.0)


def test_stats_empty(tmp_path):
    path = tmp_path / "wins.jsonl"
    st = DistillationSink(path).stats()
    assert st == {"count": 0, "unique_tasks": 0,
                  "mean_speedup": None, "median_speedup": None}
