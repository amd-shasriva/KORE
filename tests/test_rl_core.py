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


def test_build_kevin_samples_drops_infra_turns():
    # turn 1 hit an infrastructure error (timeout/OOM) -> dropped from the batch,
    # NOT trained as reward-0. Kept turns keep their (gamma-look-ahead) returns.
    traj_rewards = [[9.0, 2.0, 3.0]]
    traj_correct = [[False, True, True]]
    traj_infra = [[False, True, False]]
    returns, index = grpo.build_kevin_samples(traj_rewards, traj_correct, gamma=0.4,
                                              traj_infra=traj_infra)
    # gated=[0,2,3]: R0=0+0.4*3.2=1.28, R1=3.2 (dropped), R2=3.0
    assert index == [(0, 0), (0, 2)]
    assert [round(x, 6) for x in returns] == [1.28, 3.0]
    # no infra arg -> fully backwards compatible (all turns kept).
    r2, i2 = grpo.build_kevin_samples(traj_rewards, traj_correct, gamma=0.4)
    assert i2 == [(0, 0), (0, 1), (0, 2)]
    assert [round(x, 6) for x in r2] == [1.28, 3.2, 3.0]


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


def test_dynamic_sampling_refill_oversamples_past_degenerate():
    # DAPO dynamic sampling (item 2): a stream where attempts 1 and 3 are
    # degenerate (collapsed) groups must be REFILLED, not shrink the batch.
    stream = [
        [1.0, 0.0],   # attempt 0: signal -> keep
        [2.0, 2.0],   # attempt 1: collapsed -> refill
        [0.0, 1.0],   # attempt 2: signal -> keep
        [3.0, 3.0],   # attempt 3: collapsed -> refill
        [ -1.0, 1.0], # attempt 4: signal -> keep (target reached)
        [5.0, 6.0],   # attempt 5: never rolled
    ]
    kept, attempts = grpo.dynamic_sampling_refill(
        lambda i: stream[i], target_groups=3, min_std=1e-3, max_attempts=10)
    assert kept == [[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]  # exactly 3 non-degenerate
    assert attempts == 5                                    # 2 collapsed refilled
    # bounded: if the stream never yields enough signal, stop at max_attempts.
    kept2, attempts2 = grpo.dynamic_sampling_refill(
        lambda i: [1.0, 1.0], target_groups=3, min_std=1e-3, max_attempts=4)
    assert kept2 == [] and attempts2 == 4
    # dynamic=False keeps every rolled group (legacy fixed batch).
    kept3, attempts3 = grpo.dynamic_sampling_refill(
        lambda i: [1.0, 1.0], target_groups=2, min_std=1e-3, max_attempts=10, dynamic=False)
    assert len(kept3) == 2 and attempts3 == 2


def test_episode_turn_rewards_prefers_contract_per_turn_trace():
    # item 5 / contract (a): when the episode exposes per-turn rewards+correct,
    # they drive the SAME per-turn Kevin credit as the serial path.
    class Ep:
        turn_rewards = [0.0, 1.5, 2.0]
        turn_correct = [False, True, True]
        best_reward = 2.0
        success = True

    tr, tc = grpo._episode_turn_rewards(Ep())
    assert tr == [0.0, 1.5, 2.0] and tc == [False, True, True]
    # feeding them through build_kevin_samples == the serial per-turn credit.
    returns, index = grpo.build_kevin_samples([tr], [tc], gamma=0.4)
    assert index == [(0, 0), (0, 1), (0, 2)]
    # gated=[0,1.5,2]: R2=2, R1=1.5+0.4*2=2.3, R0=0+0.4*2.3=0.92
    assert [round(x, 6) for x in returns] == [0.92, 2.3, 2.0]

    # fallback: no per-turn trace -> single terminal (best_reward, success) sample.
    class EpBest:
        best_reward = 1.2
        success = True

    tr2, tc2 = grpo._episode_turn_rewards(EpBest())
    assert tr2 == [1.2] and tc2 == [True]

    class EpNone:
        best_reward = None
        success = False

    assert grpo._episode_turn_rewards(EpNone()) == ([0.0], [False])


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


def test_soup_sweep_order_independent_with_snapshot():
    """Fix 3: cloned endpoints keep the sweep order-independent even when the
    eval_fn materializes each alpha with an in-place ``load_state_dict``."""
    import torch

    from kore.policy.soup import soup_sweep

    class FakeModel:
        """Mimics nn.Module: state_dict() ALIASES params; load_state_dict is in-place."""

        def __init__(self, sd):
            self._sd = sd

        def state_dict(self):
            return self._sd

        def load_state_dict(self, sd):
            for k, v in sd.items():
                self._sd[k].copy_(v)

    base_model = FakeModel({"w": torch.zeros(2)})
    kore_model = FakeModel({"w": torch.ones(2)})
    # THE FIX: snapshot immutable clones of the sweep endpoints.
    base_sd = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
    kore_sd = {k: v.detach().clone() for k, v in kore_model.state_dict().items()}
    scratch = kore_model  # materialization writes here only

    base_scores = {"mmlu": 0.60}

    def eval_fn(sd):
        scratch.load_state_dict(sd)  # in-place mutate scratch (the original bug channel)
        a = float(sd["w"][0].item())
        return {"fastp": 0.10 + a, "mmlu": 0.60 - 0.2 * a}

    r1 = soup_sweep(base_sd, kore_sd, [0.0, 0.5, 1.0], eval_fn, kernel_key="fastp",
                    general_keys=["mmlu"], base_scores=base_scores, epsilon=0.005)
    # the cloned endpoints are NEVER mutated by the scratch writes.
    assert torch.equal(kore_sd["w"], torch.ones(2))
    assert torch.equal(base_sd["w"], torch.zeros(2))
    # reversing the alpha order yields the SAME best_alpha (order-independent).
    r2 = soup_sweep(base_sd, kore_sd, [1.0, 0.5, 0.0], eval_fn, kernel_key="fastp",
                    general_keys=["mmlu"], base_scores=base_scores, epsilon=0.005)
    assert r1["best_alpha"] == r2["best_alpha"] == 0.0
    # every alpha is interpolated from the pristine endpoints -> matching kernel scores.
    k_by_alpha_1 = {r["alpha"]: round(r["kernel"], 6) for r in r1["sweep"]}
    k_by_alpha_2 = {r["alpha"]: round(r["kernel"], 6) for r in r2["sweep"]}
    assert k_by_alpha_1 == k_by_alpha_2


# --------------------------------------------------------------------------- #
# micro-batched GRPO backward (Fix 1): grad-equivalence to a single-backward mean
# --------------------------------------------------------------------------- #
def test_microbatch_grad_matches_global_token_mean_clip_higher_loss():
    """Micro-batched per-sample (1/total_tokens)-scaled backward accumulates the
    SAME gradient as ONE backward on the full GLOBAL TOKEN-MEAN clip-higher loss.

    Math legitimately changed (items 1 + 3): the objective is now the DAPO
    clip-higher importance-ratio surrogate ``-min(r*A, clip(r,1-lo,1+hi)*A)`` with
    ``r = exp(logp - old_logp)`` (turn-level geometric-mean ratio), aggregated as a
    global token-mean ``sum(n_tok*term)/sum(n_tok)`` — NOT the old ratio-free
    sample-mean ``-adv*logp``. Sample tuple is now
    ``(ret, gen_inputs, ref_logp, old_logp, n_tokens, sc_weight)``.
    """
    import torch

    from kore.policy.grpo import _accumulate_grpo_grads, clip_higher_ratio, group_advantages

    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(4))
    coeffs = {
        "a": torch.tensor([0.5, -0.2, 0.1, 0.3]),
        "b": torch.tensor([-0.4, 0.7, 0.2, -0.1]),
        "c": torch.tensor([0.3, 0.3, -0.5, 0.2]),
        "d": torch.tensor([0.1, -0.6, 0.4, 0.9]),
        "e": torch.tensor([-0.7, 0.2, 0.5, -0.3]),
    }

    def logp_fn(gen_inputs):
        # gen_inputs: list of keys; recomputed token-mean log-prob = sum of w.coeff.
        total = None
        for key in gen_inputs:
            lp = (w * coeffs[key]).sum()
            total = lp if total is None else total + lp
        return total

    coef, lo, hi = 0.05, 0.2, 0.28
    # samples = [ret, gen_inputs, ref_logp, old_logp, n_tokens, sc_weight].
    # old_logp is set away from lp so the importance ratio is genuinely != 1 (some
    # samples land in the clipped region), and n_tokens varies (global token-mean).
    kept_groups = [
        [[1.0, ["a"], torch.tensor(0.2), torch.tensor(-0.3), 5, None],
         [0.0, ["b"], None, torch.tensor(0.1), 3, None],
         [2.0, ["c", "d"], torch.tensor(-0.1), torch.tensor(0.4), 8, None]],
        [[-1.0, ["e"], None, torch.tensor(-0.2), 2, None],
         [0.5, ["a", "b"], None, torch.tensor(0.0), 6, None]],
    ]

    def full_global_token_mean_loss():
        terms, total_tokens = [], 0
        for samples in kept_groups:
            advs = group_advantages([s[0] for s in samples])  # AVSPO tau=0 == plain GRPO
            for adv, s in zip(advs, samples):
                lp = logp_fn(s[1])
                old_lp = s[3]
                ratio = torch.exp(lp - old_lp)
                pg = -clip_higher_ratio(ratio, adv, lo, hi)
                term = pg
                if s[2] is not None:
                    d = s[2] - lp
                    term = term + coef * (torch.exp(d) - d - 1.0)
                n_tok = s[4]
                terms.append(term * n_tok)
                total_tokens += n_tok
        return sum(terms) / total_tokens

    # reference: build the whole global token-mean loss, single backward.
    w.grad = None
    loss = full_global_token_mean_loss()
    loss.backward()
    grad_ref = w.grad.clone()

    # micro-batched: per-sample (1/total_tokens)-scaled backward, grads accumulated.
    w.grad = None
    loss_val, n_terms = _accumulate_grpo_grads(
        kept_groups, logp_fn, ref_anchor_coef=coef,
        clip_ratio_low=lo, clip_ratio_high=hi, variance_floor=0.0)
    grad_mb = w.grad.clone()

    assert n_terms == 5
    assert torch.allclose(grad_ref, grad_mb, atol=1e-6)
    assert abs(loss_val - float(loss.detach())) < 1e-6


def test_clip_higher_ratio_tensor_gradient_clips_upper():
    """The tensor path stays differentiable and ZEROES the gradient in the
    upper-clip region for a positive advantage (DAPO clip-higher), while flowing
    the plain PG gradient near ratio==1."""
    import torch

    from kore.policy.grpo import clip_higher_ratio

    # ratio well above 1+hi with A>0 -> surrogate clipped -> zero gradient.
    old = torch.tensor(0.0)
    lp = torch.nn.Parameter(torch.tensor(1.0))          # ratio = exp(1) ~ 2.718 > 1.28
    ratio = torch.exp(lp - old)
    surr = clip_higher_ratio(ratio, 1.0, 0.2, 0.28)     # advantage +1
    (-surr).backward()
    assert abs(float(lp.grad)) < 1e-7                   # clipped -> no gradient

    # ratio ~ 1 -> surrogate == advantage*ratio -> gradient == -A (plain PG).
    lp2 = torch.nn.Parameter(torch.tensor(0.0))
    ratio2 = torch.exp(lp2 - old)                        # == 1.0
    surr2 = clip_higher_ratio(ratio2, 0.5, 0.2, 0.28)
    (-surr2).backward()
    assert abs(float(lp2.grad) - (-0.5)) < 1e-6          # d(-r*A)/dlp = -A*r = -0.5


def test_prefilter_bench_indices_selects_topk(monkeypatch):
    """Value-model bench prefilter wiring (item 6): only the top-k candidates by
    the reranker are benched. The rerank order is honored and the fallback
    (reranker unavailable) degrades to the natural order."""
    from kore.policy import grpo

    cands = ["k0", "k1", "k2", "k3", "k4"]
    # reranker ranks index 3 best, then 1, then 4, ... (best-first).
    monkeypatch.setattr(grpo, "_value_rank_order", lambda codes, task: [3, 1, 4, 0, 2])
    idx = grpo._prefilter_bench_indices(cands, task=None, k=2)
    assert idx == [1, 3]                    # top-2 by rank (3,1), returned sorted by index
    # k >= n benches everything.
    assert grpo._prefilter_bench_indices(cands, task=None, k=10) == [0, 1, 2, 3, 4]
    # reranker unavailable -> natural order fallback (bench the first k).
    monkeypatch.setattr(grpo, "_value_rank_order", lambda codes, task: None)
    assert grpo._prefilter_bench_indices(cands, task=None, k=2) == [0, 1]


# --------------------------------------------------------------------------- #
# backend routing — KORE runs ONE native in-process GRPO loop on AMD (no verl)
# --------------------------------------------------------------------------- #
def test_inprocess_backend_alias_is_the_fallback():
    assert grpo._train_grpo_inprocess is grpo._train_grpo_fallback


def test_train_grpo_routes_to_native_inprocess(monkeypatch):
    # Any backend value routes to the single native in-process loop — no verl,
    # no server, no config. Self-contained on AMD.
    for backend in ("inprocess", "fallback", "auto", "anything"):
        seen = {}

        def inproc(cfg, tasks=None, _s=seen):
            _s["ok"] = True
            return "runs/native"

        monkeypatch.setattr(grpo, "_train_grpo_inprocess", inproc)
        out = grpo.train_grpo(object(), tasks=["t"], backend=backend)
        assert out == "runs/native" and seen["ok"] is True


def test_no_verl_symbols_remain():
    # verl fully removed — the pipeline runs on pure AMD architecture.
    for sym in ("_train_grpo_verl", "build_verl_grpo_config", "_verl_available",
                "kore_verl_reward", "_verl_hydra_overrides"):
        assert not hasattr(grpo, sym), f"{sym} should be removed"
