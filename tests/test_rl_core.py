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


def test_build_kevin_samples_flattens_mxn_with_index_map():
    # 2 trajectories x 2 turns -> 4 flat per-turn samples with (traj, turn) map.
    traj_rewards = [[9.0, 2.0], [1.0, 5.0]]
    traj_correct = [[False, True], [True, True]]
    returns, index = grpo.build_kevin_samples(traj_rewards, traj_correct, gamma=0.4)
    assert index == [(0, 0), (0, 1), (1, 0), (1, 1)]
    # traj0 gated=[0,2]: R0=0+0.4*2=0.8, R1=2 ; traj1 gated=[1,5]: R0=1+0.4*5=3.0, R1=5
    assert [round(x, 6) for x in returns] == [0.8, 2.0, 3.0, 5.0]
    # equivalent to concatenating per-trajectory kevin_turn_returns.
    expect = (grpo.kevin_turn_returns(traj_rewards[0], traj_correct[0], 0.4)
              + grpo.kevin_turn_returns(traj_rewards[1], traj_correct[1], 0.4))
    assert [round(x, 6) for x in returns] == [round(x, 6) for x in expect]


def test_token_mean_logprob_length_debias():
    # token-mean divides the summed sequence log-prob by its token count.
    assert abs(grpo.token_mean_logprob(-6.0, 3) - (-2.0)) < 1e-12
    # guards against zero-length (div-by-zero) -> treated as 1 token.
    assert abs(grpo.token_mean_logprob(-4.0, 0) - (-4.0)) < 1e-12


def test_mask_cot_turns_drops_thinking_keeps_kernel():
    turns = [{
        "response": (
            "<think>secret reasoning</think>\n"
            "ANALYSIS:\nverbose chain of thought here\n\n"
            "PROPOSED_CHANGE:\nbump BLOCK_M to 128\n\n"
            "FULL_KERNEL:\n```python\n@triton.jit\ndef k():\n    pass\n```"
        ),
        "feedback": "RESULT: CORRECT",
    }]
    masked = grpo.mask_cot_turns(turns)
    m = masked[0]
    # thinking + analysis are dropped; the durable artifact is preserved.
    assert m["analysis"] == ""
    assert "response" not in m
    assert "BLOCK_M" in m["proposed_change"]
    assert "@triton.jit" in m["kernel"]
    assert "secret reasoning" not in m.get("kernel", "")
    # feedback (verifier signal) is retained; original input is untouched.
    assert m["feedback"] == "RESULT: CORRECT"
    assert "response" in turns[0]


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
