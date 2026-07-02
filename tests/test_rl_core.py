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
def test_microbatch_grad_matches_accumulated_mean_loss():
    """Per-sample (1/n)-scaled backward accumulates the SAME gradient as one
    backward on the full sample-mean loss, while only one graph is ever alive."""
    import torch

    from kore.policy.grpo import _accumulate_grpo_grads, group_advantages

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
        # gen_inputs: list of keys; recomputed log-prob = sum of w . coeff terms.
        total = None
        for key in gen_inputs:
            lp = (w * coeffs[key]).sum()
            total = lp if total is None else total + lp
        return total

    coef = 0.05
    # samples = (return, gen_inputs, ref_logp_or_None); a KL anchor on two samples.
    kept_groups = [
        [(1.0, ["a"], torch.tensor(0.2)), (0.0, ["b"], None), (2.0, ["c", "d"], torch.tensor(-0.1))],
        [(-1.0, ["e"], None), (0.5, ["a", "b"], None)],
    ]

    def full_mean_loss():
        terms = []
        for samples in kept_groups:
            advs = group_advantages([s[0] for s in samples])
            for adv, s in zip(advs, samples):
                lp = logp_fn(s[1])
                term = -adv * lp
                if s[2] is not None:
                    d = s[2] - lp
                    term = term + coef * (torch.exp(d) - d - 1.0)
                terms.append(term)
        return sum(terms) / len(terms)

    # reference: build the whole loss, single backward.
    w.grad = None
    loss = full_mean_loss()
    loss.backward()
    grad_ref = w.grad.clone()

    # micro-batched: per-sample scaled backward, grads accumulated.
    w.grad = None
    loss_val, n_terms = _accumulate_grpo_grads(
        kept_groups, logp_fn, ref_anchor_coef=coef,
        sc_grpo_allfail=False, sc_grpo_alpha=0.1)
    grad_mb = w.grad.clone()

    assert n_terms == 5
    assert torch.allclose(grad_ref, grad_mb, atol=1e-6)
    assert abs(loss_val - float(loss.detach())) < 1e-6


# --------------------------------------------------------------------------- #
# backend routing (verl vs in-process) — no verl / torch import required
# --------------------------------------------------------------------------- #
def test_inprocess_backend_alias_is_the_fallback():
    # first-class in-process name aliases the tested fallback loop (behavior kept).
    assert grpo._train_grpo_inprocess is grpo._train_grpo_fallback


def test_train_grpo_rejects_unknown_backend():
    import pytest

    with pytest.raises(ValueError):
        grpo.train_grpo(object(), backend="bogus")


def _route_recorder(calls):
    def inproc(cfg, tasks=None):
        calls["route"] = "inprocess"
        return "runs/inprocess"

    def verl(cfg, tasks=None):
        calls["route"] = "verl"
        return "runs/verl"

    return inproc, verl


def test_auto_backend_routes_to_inprocess_when_verl_missing(monkeypatch):
    # backend="auto" must NEVER crash on a missing verl — it uses the in-process
    # backend and logs. We stub verl-availability False and capture the route.
    calls = {}
    inproc, verl = _route_recorder(calls)
    monkeypatch.setattr(grpo, "_verl_available", lambda: False)
    monkeypatch.setattr(grpo, "_train_grpo_inprocess", inproc)
    monkeypatch.setattr(grpo, "_train_grpo_verl", verl)
    out = grpo.train_grpo(object(), tasks=["t"], backend="auto")
    assert out == "runs/inprocess"
    assert calls["route"] == "inprocess"


def test_auto_backend_routes_to_verl_when_available(monkeypatch):
    calls = {}
    inproc, verl = _route_recorder(calls)
    monkeypatch.setattr(grpo, "_verl_available", lambda: True)
    monkeypatch.setattr(grpo, "_train_grpo_verl", verl)
    monkeypatch.setattr(grpo, "_train_grpo_inprocess", inproc)
    out = grpo.train_grpo(object(), tasks=["t"], backend="auto")
    assert out == "runs/verl"
    assert calls["route"] == "verl"


def test_verl_backend_raises_actionable_error_when_missing(monkeypatch):
    import pytest

    monkeypatch.setattr(grpo, "_verl_available", lambda: False)
    with pytest.raises(RuntimeError) as ei:
        grpo.train_grpo(object(), backend="verl")
    msg = str(ei.value)
    # the error must be actionable: name the launcher + the doc + the install path.
    assert "scripts/launch_verl.sh" in msg
    assert "docs/rl_server.md" in msg
    assert "import verl" in msg


def test_fallback_backend_calls_inprocess(monkeypatch):
    seen = {}

    def inproc(cfg, tasks=None):
        seen["ok"] = True
        return "runs/fb"

    monkeypatch.setattr(grpo, "_train_grpo_inprocess", inproc)
    assert grpo.train_grpo(object(), backend="fallback") == "runs/fb"
    assert seen["ok"] is True


# --------------------------------------------------------------------------- #
# verl config builder (PURE — no verl import) maps GRPOConfig -> verl config
# --------------------------------------------------------------------------- #
def test_build_verl_grpo_config_maps_the_kevin_recipe():
    from kore.policy.configs import GRPOConfig

    cfg = GRPOConfig(model_id="Qwen/Qwen3-32B", output_dir="runs/grpo32",
                     num_trajectories=16, num_turns=4, tasks_per_step=8,
                     tensor_parallel_size=4, gamma=0.4, clip_ratio_low=0.2,
                     clip_ratio_high=0.28, ref_anchor_coef=1e-3, temperature=1.0,
                     top_p=1.0, use_lora=False)
    v = grpo.build_verl_grpo_config(cfg, tasks=["rmsnorm_aiter", "gemm_bf16"])

    # GRPO group-normalized advantages + discounted multi-turn credit.
    assert v["algorithm"]["adv_estimator"] == "grpo"
    assert v["algorithm"]["gamma"] == 0.4

    ar = v["actor_rollout_ref"]
    # SGLang rollout, group size = num_trajectories, TP, multi-turn refinement.
    assert ar["rollout"]["name"] == "sglang"
    assert ar["rollout"]["n"] == 16
    assert ar["rollout"]["tensor_model_parallel_size"] == 4
    assert ar["rollout"]["multi_turn"]["enable"] is True
    assert ar["rollout"]["multi_turn"]["max_assistant_turns"] == 4
    assert ar["rollout"]["temperature"] == 1.0

    # retention KL anchor (k3 low-var) + Clip-Higher + DAPO token-mean.
    assert ar["actor"]["use_kl_loss"] is True
    assert ar["actor"]["kl_loss_coef"] == 1e-3
    assert ar["actor"]["kl_loss_type"] == "low_var_kl"
    assert ar["actor"]["clip_ratio_low"] == 0.2
    assert ar["actor"]["clip_ratio_high"] == 0.28
    assert ar["actor"]["loss_agg_mode"] == "token-mean"

    # full-FT -> lora_rank 0; verified reward fn wired as the custom reward.
    assert ar["model"]["lora_rank"] == 0
    assert v["custom_reward_function"]["name"] == "kore_verl_reward"
    assert v["custom_reward_function"]["path"].endswith("grpo.py")
    assert v["reward_model"]["enable"] is False

    # trainer wiring from the config.
    assert v["trainer"]["default_local_dir"] == "runs/grpo32"
    assert v["trainer"]["n_gpus_per_node"] == 4
    assert v["data"]["kore_tasks"] == ["rmsnorm_aiter", "gemm_bf16"]


def test_build_verl_grpo_config_lora_and_no_kl():
    from kore.policy.configs import GRPOConfig

    cfg = GRPOConfig(use_lora=True, ref_anchor_coef=0.0)
    v = grpo.build_verl_grpo_config(cfg, tasks=["t"])
    ar = v["actor_rollout_ref"]
    # LoRA -> non-zero rank/alpha + explicit target modules.
    assert ar["model"]["lora_rank"] == cfg.lora.r
    assert ar["model"]["lora_alpha"] == cfg.lora.lora_alpha
    assert ar["model"]["target_modules"] == list(cfg.lora.target_modules)
    # ref_anchor_coef == 0 -> KL loss disabled.
    assert ar["actor"]["use_kl_loss"] is False


def test_verl_hydra_overrides_flatten_and_render():
    ov = grpo._verl_hydra_overrides({
        "a": {"b": 1, "c": True},
        "d": None,
        "e": [1, 2, 3],
    })
    assert "a.b=1" in ov
    assert "a.c=true" in ov
    assert "d=null" in ov
    assert "e=[1,2,3]" in ov
