"""Pillar 6 — curation & balancing."""

from __future__ import annotations

from kore.data.curate import (
    balance_by_family,
    curriculum_order,
    difficulty_score,
    filter_trivial_wins,
    quality_score,
    row_family,
    curate,
)


def _win(op, sp, content="x"):
    return {"messages": [{"role": "assistant", "content": content}],
            "_source": "kernel_repair_opt",
            "_provenance": {"kind": "win", "operation": op, "speedup": sp,
                            "snr_db": 99, "verified": True}}


def _repair(op):
    return {"messages": [{"role": "assistant", "content": "y"}],
            "_source": "kernel_repair_opt",
            "_provenance": {"kind": "repair", "operation": op, "verified": True, "snr_db": 99}}


def _chat():
    return {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
            "_source": "general_chat"}


def test_quality_score_orders_by_speedup_and_neutral_retention():
    assert quality_score(_win("gemm", 3.0)) > quality_score(_win("gemm", 1.2))
    assert quality_score(_chat()) == 1.0  # neutral, never ranked out


def test_filter_trivial_wins_keeps_repairs_and_retention():
    kept, st = filter_trivial_wins([_win("gemm", 1.05), _win("gemm", 2.0), _repair("gemm"), _chat()], 1.1)
    assert st["n_dropped_trivial_wins"] == 1
    assert len(kept) == 3


def test_balance_by_family_caps_and_keeps_best():
    rows = [_win("gemm", 1.2), _win("gemm", 3.0), _win("gemm", 2.0), _win("rmsnorm", 1.5), _chat()]
    bal, sb = balance_by_family(rows, cap_per_family=2)
    gemms = [r for r in bal if row_family(r) == "gemm"]
    assert len(gemms) == 2 and max(r["_provenance"]["speedup"] for r in gemms) == 3.0
    assert any(r.get("_source") == "general_chat" for r in bal)  # retention exempt
    assert sb["capped"] == 1


def test_difficulty_and_curriculum_order():
    hard = _win("gemm", 1.0, content="z" * 16000)
    easy = _win("gemm", 4.0, content="z")
    assert difficulty_score(hard) > difficulty_score(easy)
    order = curriculum_order([hard, easy])
    assert order[0] is easy and order[-1] is hard


def test_curate_orchestrator():
    rows = [_win("gemm", 1.02), _win("gemm", 3.0), _repair("gemm"), _chat()]
    out, stats = curate(rows, min_win_speedup=1.1, family_cap_frac=0.5)
    assert stats["dropped_trivial_wins"] == 1
    assert stats["n_out"] <= stats["n_in"]
