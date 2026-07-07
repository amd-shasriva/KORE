"""CPU-only tests for the zero-shot cross-family generalization harness (Phase 6).

Verifies holdout-split integrity (no train/eval leakage), family mapping matches
the task registry, and that the offline evaluator scores ONLY held-out families.
"""

from __future__ import annotations

import pytest

from kore.eval import generalization as G


def test_every_registry_task_has_a_family():
    from kore.tasks.registry import all_tasks
    registered = G.all_registered_tasks()
    for t in all_tasks():
        assert t.task_id in registered, f"{t.task_id} missing from FAMILIES"
        assert G.family_of(t.task_id) is not None


def test_families_are_disjoint():
    seen: set[str] = set()
    for members in G.FAMILIES.values():
        assert not (members & seen), f"task in >1 family: {members & seen}"
        seen |= members


def test_holdout_split_partitions_without_leakage():
    split = G.make_holdout_split(["attention", "moe"])
    tset, hset = set(split.train_tasks), set(split.heldout_tasks)
    assert tset & hset == set()                      # no task overlap
    assert hset == G.FAMILIES["attention"] | G.FAMILIES["moe"]
    # every held-out family truly absent from train
    train_fams = {G.family_of(t) for t in split.train_tasks}
    assert "attention" not in train_fams and "moe" not in train_fams


def test_assert_no_leakage_raises_on_tamper():
    split = G.make_holdout_split(["attention"])
    split.train_tasks.append(next(iter(G.FAMILIES["attention"])))  # inject leak
    with pytest.raises(AssertionError):
        G.assert_no_leakage(split)


def test_unknown_family_rejected():
    with pytest.raises(ValueError):
        G.make_holdout_split(["not_a_family"])


def test_unregistered_task_rejected():
    with pytest.raises(ValueError):
        G.make_holdout_split(["norm"], task_ids=["definitely_not_a_task"])


def test_evaluate_scores_only_heldout_families():
    split = G.make_holdout_split(["norm"])
    measures = [
        # held-out (norm) — should be scored
        {"task_id": "rmsnorm_aiter", "correct": True, "snr_db": 40.0,
         "cand_ms": 1.0, "t_min_ms": 0.5, "eta": 0.5, "speedup": 0.7,
         "stall_frac": 0.2, "occupancy": 0.8},
        {"task_id": "layernorm_bf16", "correct": True, "snr_db": 40.0,
         "cand_ms": 2.0, "t_min_ms": 0.5, "eta": 0.25, "speedup": 0.9,
         "stall_frac": 0.3, "occupancy": 0.7},
        # train family (gemm) — must be ignored
        {"task_id": "gemm_bf16", "correct": True, "snr_db": 40.0,
         "cand_ms": 1.0, "t_min_ms": 0.5, "eta": 0.5, "speedup": 0.5,
         "stall_frac": 0.1, "occupancy": 0.9},
        # incorrect held-out kernel — must be skipped
        {"task_id": "fused_add_rmsnorm_bf16", "correct": False, "snr_db": 5.0,
         "cand_ms": 1.0, "t_min_ms": 0.5},
    ]
    res = G.evaluate_generalization(split, measures)
    assert set(res["per_family"].keys()) == {"norm"}
    assert res["per_family"]["norm"]["n_kernels"] == 2
    assert set(res["scored_tasks"]) == {"rmsnorm_aiter", "layernorm_bf16"}
    assert res["per_family"]["norm"]["median_eta"] is not None
    assert res["per_family"]["norm"]["median_residual_reward"] is not None


def test_evaluate_pmc_fallback_counts():
    split = G.make_holdout_split(["positional"])
    measures = [
        {"task_id": "rope_bf16", "correct": True, "snr_db": 40.0,
         "cand_ms": 1.0, "t_min_ms": 0.4, "eta": 0.4},  # no counters -> eta fallback
    ]
    res = G.evaluate_generalization(split, measures)
    assert res["per_family"]["positional"]["pmc_kernels"] == 0
    assert res["per_family"]["positional"]["n_kernels"] == 1
