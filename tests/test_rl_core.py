"""CPU-only tests for the RL core: Kevin credit, StarPO-S, KL, value prefilter,
ToolRL compositing, and model soup. No torch/transformers imported at top level."""

from __future__ import annotations

import math

from kore.policy import grpo


# --------------------------------------------------------------------------- #
# Kevin multi-turn credit
# --------------------------------------------------------------------------- #
def test_kevin_trajectory_score_best_correct():
    # turns: r=[-1(fail), 0.5(correct), 0.3(correct)], best correct = 0.5
    assert grpo.kevin_trajectory_score([-1.0, 0.5, 0.3], [False, True, True]) == 0.5
    # no correct turn -> 0
    assert grpo.kevin_trajectory_score([-1.0, 0.0], [False, False]) == 0.0


def test_kevin_turn_returns_gates_on_correctness():
    # incorrect turns contribute 0 immediate reward but still get look-ahead credit
    r = grpo.kevin_turn_returns([9.0, 2.0, 3.0], [False, True, True], gamma=0.4)
    # gated = [0, 2, 3]; returns: R2=3, R1=2+0.4*3=3.2, R0=0+0.4*3.2=1.28
    assert abs(r[2] - 3.0) < 1e-9
    assert abs(r[1] - 3.2) < 1e-9
    assert abs(r[0] - 1.28) < 1e-9


# --------------------------------------------------------------------------- #
# StarPO-S variance filtering
# --------------------------------------------------------------------------- #
def test_starpo_keep_group_drops_collapsed():
    assert grpo.starpo_keep_group([1.0, 1.0, 1.0]) is False
    assert grpo.starpo_keep_group([1.0, 0.0, 0.5]) is True


def test_starpo_select_high_variance():
    groups = [
        [1.0, 1.0, 1.0],   # collapsed -> dropped
        [0.0, 1.0, 0.5],   # mid variance
        [-2.0, 2.0, 0.0],  # high variance
    ]
    keep = grpo.starpo_select_high_variance(groups, keep_frac=0.5, min_std=1e-3)
    assert 0 not in keep          # collapsed dropped
    assert 2 in keep              # highest variance kept
    assert len(keep) == 1         # keep_frac 0.5 of 2 live groups -> 1


def test_starpo_all_collapsed_returns_empty():
    assert grpo.starpo_select_high_variance([[1.0, 1.0], [2.0, 2.0]]) == []


# --------------------------------------------------------------------------- #
# KL k3 estimator
# --------------------------------------------------------------------------- #
def test_kl_k3_zero_when_equal_and_positive_otherwise():
    assert abs(grpo.kl_k3(-1.0, -1.0)) < 1e-12
    assert grpo.kl_k3(-1.0, -2.0) > 0.0  # any divergence -> positive
    assert grpo.kl_k3(-2.0, -1.0) > 0.0


# --------------------------------------------------------------------------- #
# value-model bench prefilter
# --------------------------------------------------------------------------- #
def test_value_prefilter_selects_topk_by_score():
    cands = ["a", "b", "c", "d"]
    scores = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.2}
    idx = grpo.value_prefilter(cands, lambda c: scores[c], k=2)
    assert idx == [1, 2]  # b(0.9), c(0.5), returned sorted by index
    # k >= n returns everything
    assert grpo.value_prefilter(cands, lambda c: scores[c], k=10) == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# ToolRL compositing
# --------------------------------------------------------------------------- #
def test_composite_agentic_reward():
    assert abs(grpo.composite_agentic_reward(1.0, tool_reward=0.5, tool_weight=0.2) - 1.1) < 1e-9
    assert grpo.composite_agentic_reward(1.0) == 1.0  # no tool term -> kernel reward


# --------------------------------------------------------------------------- #
# model soup (pure tensor math, torch guarded)
# --------------------------------------------------------------------------- #
def test_interpolate_state_dicts():
    import torch

    base = {"w": torch.ones(3), "i": torch.tensor([1, 2, 3])}
    kore = {"w": torch.ones(3) * 3, "i": torch.tensor([4, 5, 6])}
    out = grpo_soup_interp(base, kore, 0.5)
    assert torch.allclose(out["w"], torch.ones(3) * 2.0)  # (1-.5)*1 + .5*3 = 2
    assert torch.equal(out["i"], kore["i"])  # non-float taken from kore


def grpo_soup_interp(base, kore, alpha):
    from kore.policy.soup import interpolate_state_dicts

    return interpolate_state_dicts(base, kore, alpha)


def test_soup_sweep_respects_retention_gate():
    from kore.policy.soup import soup_sweep

    import torch

    base = {"w": torch.zeros(2)}
    kore = {"w": torch.ones(2)}
    base_scores = {"mmlu": 0.60, "fastp": 0.10}

    # higher alpha -> better kernel but worse general (regresses past epsilon)
    def eval_fn(sd):
        a = float(sd["w"][0].item())  # equals alpha
        return {"fastp": 0.10 + a, "mmlu": 0.60 - 0.2 * a}

    res = soup_sweep(base, kore, [0.0, 0.5, 1.0], eval_fn, kernel_key="fastp",
                     general_keys=["mmlu"], base_scores=base_scores, epsilon=0.005)
    # alpha=0 keeps mmlu; alpha>=0.5 regresses mmlu by >0.005 -> only 0.0 passes
    assert res["best_alpha"] == 0.0
    assert res["gate_satisfied"] is True
