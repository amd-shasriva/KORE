"""CPU tests for the GRPO co-evolution distillation hook (_build_distill_sink /
_distill_group). No torch/GPU needed — grpo's module-level imports are CPU-safe."""

from __future__ import annotations

from types import SimpleNamespace

from kore.data.schemas import WinRecord, read_jsonl
from kore.policy.grpo import _build_distill_sink, _distill_group


def _cfg(path, min_speedup=1.0):
    return SimpleNamespace(coevolve_distill_path=str(path),
                           coevolve_distill_min_speedup=min_speedup)


def test_build_distill_sink_none_without_path():
    assert _build_distill_sink(SimpleNamespace(coevolve_distill_path=None)) is None


def test_distill_group_writes_win(tmp_path):
    p = tmp_path / "wins.jsonl"
    cfg = _cfg(p)
    sink = _build_distill_sink(cfg)
    assert sink is not None
    _distill_group(sink, "gen_relu_bf16", 1.5, "def relu(x):\n    return x", cfg)
    recs = read_jsonl(p)
    assert len(recs) == 1
    assert isinstance(recs[0], WinRecord)
    assert recs[0].task_id == "gen_relu_bf16"
    assert recs[0].speedup == 1.5
    assert "relu" in recs[0].final_source


def test_distill_group_filters_slow_and_empty(tmp_path):
    p = tmp_path / "wins.jsonl"
    cfg = _cfg(p, min_speedup=1.0)
    sink = _build_distill_sink(cfg)
    _distill_group(sink, "gen_relu_bf16", 0.5, "def relu(x): return x", cfg)  # too slow
    _distill_group(sink, "gen_relu_bf16", None, "def relu(x): return x", cfg)  # no speedup
    _distill_group(sink, "gen_relu_bf16", 2.0, None, cfg)                       # no source
    assert read_jsonl(p) == []  # nothing written


def test_distill_group_dedup_keeps_best(tmp_path):
    p = tmp_path / "wins.jsonl"
    cfg = _cfg(p)
    sink = _build_distill_sink(cfg)
    src = "def relu(x):\n    return x"
    _distill_group(sink, "gen_relu_bf16", 1.2, src, cfg)
    _distill_group(sink, "gen_relu_bf16", 1.9, src, cfg)  # same kernel, faster
    recs = read_jsonl(p)
    assert len(recs) == 1 and recs[0].speedup == 1.9
