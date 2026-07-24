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
    import pytest
    from kore.policy.soup import SoupPromotionError, soup_sweep

    import torch

    base = {"w": torch.zeros(2)}
    kore = {"w": torch.ones(2)}
    base_scores = {"mmlu": 0.60, "fastp": 0.10}

    # higher alpha -> better kernel but worse general (regresses past epsilon)
    def eval_fn(sd):
        a = float(sd["w"][0].item())  # equals alpha
        return {"fastp": 0.10 + a, "mmlu": 0.60 - 0.2 * a}

    with pytest.raises(SoupPromotionError) as exc:
        soup_sweep(base, kore, [0.0, 0.5, 1.0], eval_fn, kernel_key="fastp",
                   general_keys=["mmlu"], base_scores=base_scores, epsilon=0.005)
    # alpha=0 is safety-only; it can never be silently promoted as the "best" soup.
    assert exc.value.sweep[0]["alpha"] == 0.0
    assert exc.value.sweep[0]["passed"] is True
    assert all(not row["passed"] for row in exc.value.sweep if row["alpha"] > 0)


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
        return {"fastp": 0.10 + a, "mmlu": 0.60 - 0.004 * a}

    r1 = soup_sweep(base_sd, kore_sd, [0.0, 0.5, 1.0], eval_fn, kernel_key="fastp",
                    general_keys=["mmlu"], base_scores=base_scores, epsilon=0.005)
    # the cloned endpoints are NEVER mutated by the scratch writes.
    assert torch.equal(kore_sd["w"], torch.ones(2))
    assert torch.equal(base_sd["w"], torch.zeros(2))
    # reversing the alpha order yields the SAME best_alpha (order-independent).
    r2 = soup_sweep(base_sd, kore_sd, [1.0, 0.5, 0.0], eval_fn, kernel_key="fastp",
                    general_keys=["mmlu"], base_scores=base_scores, epsilon=0.005)
    assert r1["best_alpha"] == r2["best_alpha"] == 1.0
    # every alpha is interpolated from the pristine endpoints -> matching kernel scores.
    k_by_alpha_1 = {r["alpha"]: round(r["kernel"], 6) for r in r1["sweep"]}
    k_by_alpha_2 = {r["alpha"]: round(r["kernel"], 6) for r in r2["sweep"]}
    assert k_by_alpha_1 == k_by_alpha_2


def test_soup_injects_alpha_zero_and_requires_compatible_state_dicts():
    import pytest
    import torch

    from kore.policy.soup import SoupError, soup_sweep, validate_state_dict_compatibility

    base = {"w": torch.zeros(2), "counter": torch.tensor([1], dtype=torch.int64)}
    kore = {"w": torch.ones(2), "counter": torch.tensor([9], dtype=torch.int64)}

    def evaluate(sd):
        alpha = float(sd["w"][0])
        return {"fastp": 0.1 + alpha, "mmlu": 0.6}

    result = soup_sweep(
        base, kore, [0.5], evaluate, kernel_key="fastp",
        general_keys=["mmlu"], base_scores={"mmlu": 0.6},
    )
    assert [row["alpha"] for row in result["sweep"]] == [0.0, 0.5]
    assert result["best_alpha"] == 0.5

    with pytest.raises(SoupError, match="key mismatch"):
        validate_state_dict_compatibility(base, {"w": torch.ones(2)})
    with pytest.raises(SoupError, match="shape mismatch"):
        validate_state_dict_compatibility(base, {
            "w": torch.ones(3), "counter": torch.tensor([9], dtype=torch.int64),
        })


def test_soup_alpha_zero_is_literal_base_and_nonfinite_metrics_abort():
    import pytest
    import torch

    from kore.policy.soup import (
        SoupPromotionError,
        interpolate_state_dicts,
        soup_sweep,
    )

    base = {"w": torch.zeros(1), "counter": torch.tensor([1], dtype=torch.int64)}
    kore = {"w": torch.ones(1), "counter": torch.tensor([9], dtype=torch.int64)}
    safety = interpolate_state_dicts(base, kore, 0.0)
    assert torch.equal(safety["w"], base["w"])
    assert torch.equal(safety["counter"], base["counter"])

    def evaluate(sd):
        alpha = float(sd["w"][0])
        return {"fastp": float("nan") if alpha > 0 else 0.1, "mmlu": 0.6}

    with pytest.raises(SoupPromotionError, match="non-finite"):
        soup_sweep(
            base, kore, [0.5], evaluate, kernel_key="fastp",
            general_keys=["mmlu"], base_scores={"mmlu": 0.6},
        )


def test_soup_rejects_model_architecture_mismatch_before_weights():
    import pytest

    from kore.policy.soup import SoupError, _validate_model_architecture

    class Config:
        def __init__(self, hidden_size):
            self.hidden_size = hidden_size

        def to_dict(self):
            return {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": self.hidden_size,
                "num_hidden_layers": 2,
            }

    class Model:
        def __init__(self, hidden_size):
            self.config = Config(hidden_size)

    with pytest.raises(SoupError, match="architecture mismatch"):
        _validate_model_architecture(Model(4), Model(8))


# --------------------------------------------------------------------------- #
# micro-batched GRPO backward (Fix 1): grad-equivalence to a single-backward mean
# --------------------------------------------------------------------------- #
def test_microbatch_grad_matches_global_token_mean_clip_higher_loss():
    """Micro-batched per-sample (1/total_tokens)-scaled backward accumulates the
    SAME gradient as ONE backward on the full GLOBAL TOKEN-MEAN clip-higher loss.

    Math legitimately changed (items 1 + 3): the objective is now the DAPO
    clip-higher importance-ratio surrogate ``-min(r*A, clip(r,1-lo,1+hi)*A)`` with
    ``r = exp(logp - old_logp)`` (turn-level geometric-mean ratio), aggregated as a
    global token-mean ``sum(n_tok*term)/sum(n_tok)`` - NOT the old ratio-free
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
# backend routing - KORE runs ONE native in-process GRPO loop on AMD (no verl)
# --------------------------------------------------------------------------- #
def test_inprocess_backend_alias_is_the_fallback():
    assert grpo._train_grpo_inprocess is grpo._train_grpo_fallback


def test_train_grpo_routes_to_native_inprocess(monkeypatch):
    # Any backend value routes to the single native in-process loop - no verl,
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
    # verl fully removed - the pipeline runs on pure AMD architecture.
    for sym in ("_train_grpo_verl", "build_verl_grpo_config", "_verl_available",
                "kore_verl_reward", "_verl_hydra_overrides"):
        assert not hasattr(grpo, sym), f"{sym} should be removed"


# --------------------------------------------------------------------------- #
# Fix 1: value-model activation logs TRAINED vs heuristic ranker
# --------------------------------------------------------------------------- #
def test_activate_value_ranker_trained_vs_heuristic(monkeypatch, capsys):
    import kore.value.rerank as rr

    # A set path that loads -> the TRAINED model is installed & announced.
    sentinel = object()
    monkeypatch.setattr(rr, "load_default_model", lambda p: sentinel)
    from kore.policy.configs import GRPOConfig

    cfg = GRPOConfig(value_prefilter=True, value_model_path="runs/value/vm.pkl")
    out = grpo._activate_value_ranker(cfg)
    assert out is sentinel
    assert "TRAINED value model loaded" in capsys.readouterr().out

    # Unset path -> heuristic cold-start fallback (logged clearly, not silent).
    cfg2 = GRPOConfig(value_prefilter=True, value_model_path=None)
    assert grpo._activate_value_ranker(cfg2) is None
    assert "heuristic cold-start fallback" in capsys.readouterr().out

    # A set path that FAILS to load -> heuristic fallback (still logged).
    monkeypatch.setattr(rr, "load_default_model", lambda p: None)
    cfg3 = GRPOConfig(value_prefilter=True, value_model_path="missing.pkl")
    assert grpo._activate_value_ranker(cfg3) is None
    assert "heuristic cold-start fallback" in capsys.readouterr().out

    # prefilter OFF -> no-op (never touches the ranker).
    assert grpo._activate_value_ranker(GRPOConfig(value_prefilter=False)) is None


# --------------------------------------------------------------------------- #
# Fix 3: LR warmup+scheduler and max_prompt_length truncation are wired
# --------------------------------------------------------------------------- #
def test_lr_scheduler_warmup_then_decay():
    import torch

    from kore.policy.configs import GRPOConfig

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    cfg = GRPOConfig(total_steps=10, warmup_ratio=0.2, lr_scheduler_type="cosine")
    sched = grpo._build_lr_scheduler(opt, cfg)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(9):
        opt.step()  # mirror the loop's optimizer-then-scheduler order
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    # warmup = round(0.2*10) = 2 steps: lr ramps UP over the first steps...
    assert lrs[0] < lrs[1] < lrs[2]
    # ...then cosine-decays toward 0 by the end.
    assert lrs[-1] < lrs[2]
    assert lrs[-1] < 0.05

    # "constant" keeps a flat LR after warmup.
    opt2 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    csched = grpo._build_lr_scheduler(opt2, GRPOConfig(total_steps=5, warmup_ratio=0.0,
                                                       lr_scheduler_type="constant"))
    flat = []
    for _ in range(5):
        flat.append(opt2.param_groups[0]["lr"])
        opt2.step()
        csched.step()
    assert all(abs(x - 1.0) < 1e-9 for x in flat)


def test_truncate_prompt_ids_left_truncates():
    import torch

    from kore.policy.configs import GRPOConfig

    ids = torch.arange(20).unsqueeze(0)  # [1, 20]
    trunc = grpo._truncate_prompt_ids(ids, GRPOConfig(max_prompt_length=8))
    assert trunc.shape[1] == 8
    # keeps the MOST RECENT (rightmost) tokens incl. the generation prompt.
    assert trunc[0, -1].item() == 19
    # no truncation when already within budget.
    short = torch.arange(4).unsqueeze(0)
    assert grpo._truncate_prompt_ids(short, GRPOConfig(max_prompt_length=8)).shape[1] == 4


# --------------------------------------------------------------------------- #
# Fix 5: end-to-end - actually RUN _train_grpo_inprocess for 1+ steps on a TINY
# CPU model + a fake KoreEnv, driving the REAL loop (dynamic sampling, StarPO-S,
# AVSPO, KL anchor, micro-batched clip-higher backward, save). GPU-only pieces
# (the 14B forward) are replaced by a tiny nn.Module; everything else is real.
# --------------------------------------------------------------------------- #
from types import SimpleNamespace  # noqa: E402


def _build_tiny_stack(decode_text, vocab=32):
    """A tiny HF-like (model, tokenizer, env, task) quadruple that runs on CPU."""
    import os

    import torch

    from kore.reward.reward import Observation

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab = vocab
            self.emb = torch.nn.Embedding(vocab, 8)
            self.head = torch.nn.Linear(8, vocab)
            self.device = torch.device("cpu")
            self._init = self.head.weight.detach().clone()
            # Mirror a real HF model surface the trainer touches.
            self.config = SimpleNamespace(use_cache=True)

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.head(self.emb(input_ids)))

        def generate(self, input_ids, max_new_tokens=4, **kw):
            n = max(1, min(int(max_new_tokens), 3))
            gen = torch.randint(0, self.vocab, (1, n))
            return SimpleNamespace(sequences=torch.cat([input_ids, gen], dim=1))

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            pass

        def enable_input_require_grads(self):
            pass

        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            open(os.path.join(path, "model.marker"), "w").close()

        def param_changed(self):
            return not torch.allclose(self.head.weight.detach(), self._init)

    class TinyTok:
        def _encode(self, text):
            return [(ord(c) % 29) + 1 for c in (text or "x")[:12]] or [1]

        def apply_chat_template(self, messages, add_generation_prompt=True,
                                return_tensors="pt", **kw):
            text = " ".join(str(m.get("content", "")) for m in messages)
            return torch.tensor([self._encode(text)])

        def __call__(self, text, return_tensors="pt", **kw):
            return SimpleNamespace(input_ids=torch.tensor([self._encode(text)]))

        def decode(self, seq, skip_special_tokens=True):
            return decode_text

        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            open(os.path.join(path, "tok.marker"), "w").close()

    def _correct():
        return Observation(compiled=True, dtype="bf16", validation_passed=True,
                           snr_by_shape={"primary": 100.0}, snr_db=100.0,
                           wall_by_shape={"primary": 1.0},
                           baseline_by_shape={"primary": 2.0},
                           wall_ms=1.0, baseline_ms=2.0)

    def _incorrect():
        return Observation(compiled=True, dtype="bf16", validation_passed=False,
                           snr_by_shape={"primary": 3.0}, snr_db=3.0,
                           error_text="worst SNR 3.0 < gate")

    class FakeEnv:
        """First step()/trajectory is correct+fast, the rest incorrect -> the
        group carries real per-trajectory reward variance (StarPO-S keeps it, and
        SC-GRPO sees a partial-solve group)."""

        def __init__(self, task, n_correct=1):
            self.task = task
            self.n = 0
            self.n_correct = n_correct

        def step(self, source, full_validation=True, multi_shape=True):
            i = self.n
            self.n += 1
            return _correct() if i < self.n_correct else _incorrect()

    class FakeTask:
        task_id = "fake_gemm_bf16"
        operation = "gemm"
        dtype = "bf16"
        gpu_target = "gfx942"
        backend = "triton"
        comparison_baseline = "aiter"
        seed_source = "def seed():\n    return 0"
        shapes = []
        snr_threshold = None

    return TinyLM, TinyTok(), FakeEnv, FakeTask


def _install_stubs(monkeypatch, TinyLM, tok, FakeEnv, FakeTask):
    """Redirect the loop's heavy deps to the tiny CPU stack; return created models."""
    import transformers

    import kore.env.kore_env as ke
    import kore.tasks.registry as reg

    created = []

    def load_model(model_id, **kw):
        m = TinyLM()
        created.append(m)
        return m

    monkeypatch.setattr(transformers, "AutoModelForCausalLM",
                        SimpleNamespace(from_pretrained=load_model))
    monkeypatch.setattr(transformers, "AutoTokenizer",
                        SimpleNamespace(from_pretrained=lambda mid, **kw: tok))
    monkeypatch.setattr(ke, "KoreEnv", FakeEnv)
    monkeypatch.setattr(reg, "get_task", lambda tid: FakeTask())
    monkeypatch.setattr(reg, "task_ids", lambda: ["fake_gemm_bf16"])
    return created


_SERIAL_DECODE = (
    "ANALYSIS:\ntile too small\n\nPROPOSED_CHANGE:\nbump BLOCK_M\n\n"
    "FULL_KERNEL:\n```python\ndef k():\n    return 0\n```"
)
_AGENTIC_DECODE = (
    '<tool_call>\n{"name": "test", "arguments": {"kernel_src": '
    '"def k():\\n    return 0"}}\n</tool_call>'
)


def test_e2e_train_grpo_serial_steps_and_exercises_scgrpo_prefilter_curriculum(
        monkeypatch, tmp_path):
    """Drive the REAL ``_train_grpo_inprocess`` for 2 steps on a tiny CPU model +
    fake KoreEnv with value_prefilter + SC-GRPO + GTPO + the correctness phase all
    ON, asserting it completes, takes a real gradient step, writes a periodic
    checkpoint, and exercises the prefilter / SC-GRPO / curriculum code paths."""
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from kore.policy.configs import GRPOConfig

    TinyLM, tok, FakeEnv, FakeTask = _build_tiny_stack(_SERIAL_DECODE)
    created = _install_stubs(monkeypatch, TinyLM, tok, FakeEnv, FakeTask)

    # Spies that COUNT calls but call through to the real code paths.
    calls = {"prefilter": 0, "scgrpo": 0, "phase": 0}
    real_pf, real_sc, real_ph = (grpo._prefilter_bench_indices, grpo._scgrpo_weight,
                                 grpo.apply_reward_phase)
    monkeypatch.setattr(grpo, "_prefilter_bench_indices",
                        lambda *a, **k: calls.__setitem__("prefilter", calls["prefilter"] + 1)
                        or real_pf(*a, **k))
    monkeypatch.setattr(grpo, "_scgrpo_weight",
                        lambda *a, **k: calls.__setitem__("scgrpo", calls["scgrpo"] + 1)
                        or real_sc(*a, **k))
    monkeypatch.setattr(grpo, "apply_reward_phase",
                        lambda *a, **k: calls.__setitem__("phase", calls["phase"] + 1)
                        or real_ph(*a, **k))

    cfg = GRPOConfig(
        model_id="tiny", output_dir=str(tmp_path),
        num_trajectories=2, num_turns=1, tasks_per_step=1, total_steps=2,
        use_lora=False, gradient_checkpointing=False, bf16=False,
        learning_rate=0.1, warmup_ratio=0.0, lr_scheduler_type="constant",
        max_response_length=4, max_prompt_length=16, temperature=0.9, top_p=0.95,
        value_prefilter=True, num_candidates_per_turn=2, value_prefilter_k=1,
        sc_grpo=True, gtpo_codesim=True, reward_phase="correctness",
        ref_anchor_coef=1e-3, starpo_s=True, dynamic_sampling=True,
        save_steps=1, logging_steps=1, agentic=False,
    )

    out = grpo._train_grpo_inprocess(cfg, tasks=["fake_gemm_bf16"])

    assert out == str(tmp_path)
    assert (tmp_path / "model.marker").exists()      # final full-FT save happened
    assert (tmp_path / "checkpoint-1").is_dir()      # periodic save (save_steps=1)
    assert created and created[0].param_changed()    # a real gradient step landed
    assert calls["prefilter"] > 0, "value prefilter path not exercised"
    assert calls["scgrpo"] > 0, "SC-GRPO KL-weighting path not exercised"
    assert calls["phase"] > 0, "correctness->latency curriculum mask not exercised"


def test_e2e_train_grpo_agentic_steps(monkeypatch, tmp_path):
    """Drive the REAL loop in AGENTIC mode: the tiny model emits a Hermes tool
    call, the real AgentHarness runs it against the fake env, and the loop applies
    per-turn Kevin credit + takes a gradient step. Confirms the agentic rollout
    path (not the serial one) is exercised end-to-end."""
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from kore.policy.configs import GRPOConfig

    TinyLM, tok, FakeEnv, FakeTask = _build_tiny_stack(_AGENTIC_DECODE)
    created = _install_stubs(monkeypatch, TinyLM, tok, FakeEnv, FakeTask)

    seen = {"agentic": 0, "serial": 0}
    real_ag, real_se = grpo._rollout_agentic, grpo._rollout
    monkeypatch.setattr(grpo, "_rollout_agentic",
                        lambda *a, **k: seen.__setitem__("agentic", seen["agentic"] + 1)
                        or real_ag(*a, **k))
    monkeypatch.setattr(grpo, "_rollout",
                        lambda *a, **k: seen.__setitem__("serial", seen["serial"] + 1)
                        or real_se(*a, **k))

    cfg = GRPOConfig(
        model_id="tiny", output_dir=str(tmp_path),
        num_trajectories=2, num_turns=1, tasks_per_step=1, total_steps=1,
        use_lora=False, gradient_checkpointing=False, bf16=False,
        learning_rate=0.1, warmup_ratio=0.0, lr_scheduler_type="constant",
        max_response_length=4, max_prompt_length=16, temperature=0.9, top_p=1.0,
        agentic=True, max_tool_turns=1, ref_anchor_coef=1e-3,
        starpo_s=True, dynamic_sampling=True, save_steps=0, logging_steps=1,
    )

    out = grpo._train_grpo_inprocess(cfg, tasks=["fake_gemm_bf16"])

    assert out == str(tmp_path)
    assert (tmp_path / "model.marker").exists()
    assert created and created[0].param_changed()    # a real gradient step landed
    assert seen["agentic"] > 0, "agentic rollout path not exercised"
    assert seen["serial"] == 0, "serial rollout must not run in agentic mode"
