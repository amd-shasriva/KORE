"""CPU-only tests for the zero-shot cross-family generalization harness (Phase 6).

Verifies the operator-family classifier (registry-independent, order-sensitive),
holdout-split integrity (no train/eval leakage), full-registry coverage, and that
the offline evaluator scores ONLY held-out families.
"""

from __future__ import annotations

import pytest

from kore.eval import generalization as G


# ---------------- classifier ---------------- #
def test_classifier_covers_families_and_ordering():
    # order-sensitive edge cases
    assert G.classify("gen_gemm_silu_fp16") == "gemm"      # gemm-epilogue fusion, not activation
    assert G.classify("gemm_bias_gelu") == "gemm"
    assert G.classify("gen_maximum_fp32") == "activation"  # NOT reduction (bare 'max')
    assert G.classify("gen_row_mean_fp16") == "reduction"  # row_ reduction
    assert G.classify("softmax_bf16") == "reduction"
    assert G.classify("topk_softmax_bf16") == "moe"        # MoE routing, not reduction
    assert G.classify("fused_add_rmsnorm_bf16") == "norm"
    assert G.classify("layernorm_bf16") == "norm"
    assert G.classify("genv_rope_fp16") == "positional"
    assert G.classify("quant_fp8_pertoken") == "quant"
    assert G.classify("flash_attn_prefill_bf16") == "attention"
    assert G.classify("gen_silu_mul_fp16") == "activation"


def test_every_registry_task_classifies_into_a_known_family():
    from kore.tasks.registry import all_tasks
    fams = set(G.FAMILIES)
    for t in all_tasks():
        fam = G.family_of(t.task_id)
        assert fam in fams, f"{t.task_id} -> {fam} not a known family"


def test_family_of_empty_is_none():
    assert G.family_of("") is None


# ---------------- holdout split ---------------- #
def test_holdout_split_partitions_registry_without_leakage():
    split = G.make_holdout_split(["attention", "moe"])
    tset, hset = set(split.train_tasks), set(split.heldout_tasks)
    assert tset & hset == set()                       # no task overlap
    assert hset, "expected some attention/moe tasks held out"
    for t in split.heldout_tasks:
        assert G.family_of(t) in {"attention", "moe"}
    train_fams = {G.family_of(t) for t in split.train_tasks}
    assert "attention" not in train_fams and "moe" not in train_fams


def test_holdout_split_over_explicit_task_universe():
    ids = ["gemm_bf16", "rmsnorm_aiter", "silu_mul_bf16", "rope_bf16", "softmax_bf16"]
    split = G.make_holdout_split(["norm"], task_ids=ids)
    assert split.heldout_tasks == ["rmsnorm_aiter"]
    assert set(split.train_tasks) == {"gemm_bf16", "silu_mul_bf16", "rope_bf16", "softmax_bf16"}


def test_assert_no_leakage_raises_on_tamper():
    split = G.make_holdout_split(["attention"], task_ids=["flash_attn_decode_bf16", "gemm_bf16"])
    split.train_tasks.append("flash_attn_prefill_bf16")  # inject a held-out-family task
    with pytest.raises(AssertionError):
        G.assert_no_leakage(split)


def test_unknown_family_rejected():
    with pytest.raises(ValueError):
        G.make_holdout_split(["not_a_family"])


# ---------------- offline evaluation ---------------- #
def test_evaluate_scores_only_heldout_families():
    ids = ["rmsnorm_aiter", "layernorm_bf16", "gemm_bf16", "fused_add_rmsnorm_bf16"]
    split = G.make_holdout_split(["norm"], task_ids=ids)
    measures = [
        {"task_id": "rmsnorm_aiter", "correct": True, "snr_db": 40.0,
         "cand_ms": 1.0, "t_min_ms": 0.5, "eta": 0.5, "speedup": 0.7,
         "stall_frac": 0.2, "occupancy": 0.8},
        {"task_id": "layernorm_bf16", "correct": True, "snr_db": 40.0,
         "cand_ms": 2.0, "t_min_ms": 0.5, "eta": 0.25, "speedup": 0.9,
         "stall_frac": 0.3, "occupancy": 0.7},
        {"task_id": "gemm_bf16", "correct": True, "snr_db": 40.0,          # train family -> ignored
         "cand_ms": 1.0, "t_min_ms": 0.5, "eta": 0.5, "speedup": 0.5,
         "stall_frac": 0.1, "occupancy": 0.9},
        {"task_id": "fused_add_rmsnorm_bf16", "correct": False, "snr_db": 5.0,  # incorrect -> skipped
         "cand_ms": 1.0, "t_min_ms": 0.5},
    ]
    res = G.evaluate_generalization(split, measures)
    assert set(res["per_family"].keys()) == {"norm"}
    assert res["per_family"]["norm"]["n_kernels"] == 2
    assert set(res["scored_tasks"]) == {"rmsnorm_aiter", "layernorm_bf16"}
    assert res["per_family"]["norm"]["median_eta"] is not None
    assert res["per_family"]["norm"]["median_residual_reward"] is not None


def test_evaluate_pmc_fallback_counts():
    split = G.make_holdout_split(["positional"], task_ids=["rope_bf16", "gemm_bf16"])
    measures = [
        {"task_id": "rope_bf16", "correct": True, "snr_db": 40.0,
         "cand_ms": 1.0, "t_min_ms": 0.4, "eta": 0.4},  # no counters -> eta fallback
    ]
    res = G.evaluate_generalization(split, measures)
    assert res["per_family"]["positional"]["pmc_kernels"] == 0
    assert res["per_family"]["positional"]["n_kernels"] == 1
