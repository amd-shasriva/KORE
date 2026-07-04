"""CPU-only tests for the KORE policy module.

No torch / vllm / transformers / trl / verl are imported. We exercise the pure
math (grpo, anticollapse), the format/parse helpers, and confirm every policy
submodule imports without the heavy training stack.
"""

from __future__ import annotations

import math

from kore.policy import anticollapse as ac
from kore.policy import format as fmt
from kore.policy import grpo


# --------------------------------------------------------------------------- #
# grpo pure math
# --------------------------------------------------------------------------- #
def test_group_advantages_mean_zero_and_normalized():
    advs = grpo.group_advantages([1.0, 2.0, 3.0])
    # Mean of the normalized advantages is ~0.
    assert abs(sum(advs) / len(advs)) < 1e-6
    # Correct normalization: (r - mean) / (pop_std + eps).
    mean = 2.0
    std = math.sqrt(((1 - mean) ** 2 + 0 + (3 - mean) ** 2) / 3)
    expected = [(r - mean) / (std + 1e-6) for r in (1.0, 2.0, 3.0)]
    for a, e in zip(advs, expected):
        assert abs(a - e) < 1e-9


def test_group_advantages_collapse_all_equal():
    advs = grpo.group_advantages([5.0, 5.0, 5.0, 5.0])
    # std == 0 -> every advantage collapses to ~0 (the degenerate case).
    assert all(abs(a) < 1e-3 for a in advs)


def test_discounted_returns_gamma_04():
    scores = [1.0, 2.0, 3.0]
    got = grpo.discounted_returns(scores, gamma=0.4)
    # R_2 = 3
    # R_1 = 2 + 0.4*3 = 3.2
    # R_0 = 1 + 0.4*2 + 0.16*3 = 1 + 0.8 + 0.48 = 2.28
    expected = [2.28, 3.2, 3.0]
    for g, e in zip(got, expected):
        assert abs(g - e) < 1e-9


def test_clip_higher_ratio_asymmetric():
    # Positive advantage, ratio above the higher clip -> clipped at 1 + hi.
    v = grpo.clip_higher_ratio(2.0, 1.0, lo=0.2, hi=0.28)
    assert abs(v - 1.28) < 1e-9
    # Ratio ~ 1 -> surrogate ~ advantage.
    assert abs(grpo.clip_higher_ratio(1.0, 0.5) - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# format / parse helpers
# --------------------------------------------------------------------------- #
def test_parse_response_extracts_kernel_block():
    text = (
        "ANALYSIS:\nThe tile is too small, increasing occupancy pressure.\n\n"
        "PROPOSED_CHANGE:\nBump BLOCK_M to 128.\n\n"
        "FULL_KERNEL:\n```python\n"
        "import triton\n@triton.jit\ndef k():\n    pass\n"
        "```\n"
    )
    parsed = fmt.parse_response(text)
    assert "tile is too small" in parsed["analysis"]
    assert "BLOCK_M" in parsed["proposed_change"]
    assert "@triton.jit" in parsed["kernel"]
    assert "ANALYSIS" not in parsed["kernel"]


def test_summarize_cot_truncates():
    long_text = "x" * 5000
    out = fmt.summarize_cot(long_text, max_chars=200)
    assert len(out) <= 200
    # Short text is returned untouched.
    assert fmt.summarize_cot("short", max_chars=200) == "short"


def test_build_transcript_shape():
    turns = [{"response": "ANALYSIS:\na\nFULL_KERNEL:\n```python\nx=1\n```", "feedback": "RESULT: CORRECT"}]
    msgs = fmt.build_transcript("optimize this", turns)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == "optimize this"
    assert msgs[2]["role"] == "assistant"
    assert msgs[3]["role"] == "user" and "CORRECT" in msgs[3]["content"]


def test_build_turn_feedback_from_observation():
    from kore.reward.reward import Observation

    obs = Observation(
        compiled=True, snr_db=90.0, wall_ms=0.5, baseline_ms=1.0,
        wall_by_shape={"s": 0.5}, baseline_by_shape={"s": 1.0},
        snr_by_shape={"s": 90.0}, validation_passed=True,
    )
    fb = fmt.build_turn_feedback(obs)
    assert "CORRECT" in fb and "2.0" in fb  # 2x speedup

    bad = Observation(compiled=False, snr_db=None, wall_ms=None, error_text="boom")
    fb2 = fmt.build_turn_feedback(bad)
    assert "FAILED" in fb2 and "boom" in fb2


# --------------------------------------------------------------------------- #
# anticollapse pure math
# --------------------------------------------------------------------------- #
def test_sample_reward_tokens_fraction():
    G, p = 16, 0.5
    toks = ac.sample_reward_tokens(G, p, seed=1)
    assert len(toks) == G
    frac = toks.count(ac.HIGH_REWARD_TOKEN) / G
    assert abs(frac - p) < 0.1
    assert set(toks) <= {ac.HIGH_REWARD_TOKEN, ac.LOW_REWARD_TOKEN}


def test_prepend_reward_token():
    out = ac.prepend_reward_token("do the thing", ac.HIGH_REWARD_TOKEN)
    assert out.startswith(ac.HIGH_REWARD_TOKEN)
    assert "do the thing" in out
    try:
        ac.prepend_reward_token("x", "<|bogus|>")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_variance_floor_true_when_modes_differ():
    # Two reward-token modes with different means -> variance meets the floor.
    tokens = [ac.HIGH_REWARD_TOKEN, ac.HIGH_REWARD_TOKEN, ac.LOW_REWARD_TOKEN, ac.LOW_REWARD_TOKEN]
    rewards = [1.0, 1.0, 0.0, 0.0]  # rewards track their conditioned mode
    means = {ac.HIGH_REWARD_TOKEN: 1.0, ac.LOW_REWARD_TOKEN: 0.0}
    assert ac.variance_floor(rewards, tokens, means) is True


def test_variance_floor_collapse_below_floor():
    # Modes claim a gap, but the realized rewards collapsed -> below the floor.
    tokens = [ac.HIGH_REWARD_TOKEN, ac.HIGH_REWARD_TOKEN, ac.LOW_REWARD_TOKEN, ac.LOW_REWARD_TOKEN]
    rewards = [0.5, 0.5, 0.5, 0.5]
    means = {ac.HIGH_REWARD_TOKEN: 1.0, ac.LOW_REWARD_TOKEN: 0.0}
    assert ac.variance_floor(rewards, tokens, means) is False


def test_avspo_advantages_variance_floor():
    # AVSPO (item 4a): a near-degenerate group (tiny spread) would collapse under
    # plain GRPO; injecting k virtual samples at +/- tau raises the denominator so
    # advantages stay finite and BOUNDED below the un-floored (huge) values.
    returns = [0.0, 0.0, 0.0, 1e-4]  # std ~ 4e-5, far below tau
    tau = 0.5
    plain = ac.avspo_advantages(returns, tau=0.0)            # tau=0 disables -> pure GRPO
    floored = ac.avspo_advantages(returns, tau=tau, k=2)
    # floor injection strictly SHRINKS the advantage magnitude (bigger denom).
    assert max(abs(a) for a in floored) < max(abs(a) for a in plain)
    # denominator is the augmented std: aug_std = sqrt((n*var + k*tau^2)/(n+k)).
    import math
    n = len(returns); mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / n
    aug_std = math.sqrt((n * var + 2 * tau ** 2) / (n + 2))
    assert aug_std >= math.sqrt(2 * tau ** 2 / (n + 2)) > 0.0   # guaranteed floor
    for a, r in zip(floored, returns):
        assert abs(a - (r - mean) / (aug_std + 1e-6)) < 1e-9
    # exactly len(returns) advantages: virtual samples get NO PG term.
    assert len(floored) == len(returns)
    # a group whose std already exceeds tau is untouched (== plain GRPO).
    hi = [-2.0, 2.0, 0.0]
    assert ac.avspo_advantages(hi, tau=0.5) == ac.avspo_advantages(hi, tau=0.0)


def test_scgrpo_weight_from_kl_bounded():
    # SC-GRPO (item 4b): higher mean per-token KL -> larger PG weight, clipped.
    assert ac.scgrpo_weight_from_kl([]) == 1.0                     # no tokens -> neutral
    assert abs(ac.scgrpo_weight_from_kl([0.0, 0.0]) - 1.0) < 1e-9  # zero KL -> neutral
    w = ac.scgrpo_weight_from_kl([0.5, 0.5], scale=1.0)
    assert abs(w - 1.5) < 1e-9                                     # 1 + 1.0*0.5
    assert ac.scgrpo_weight_from_kl([100.0], w_max=2.0) == 2.0     # clipped to ceiling
    assert ac.scgrpo_weight_from_kl([-100.0], w_min=0.5) == 0.5    # clipped to floor


def test_gtpo_codesim_shaping():
    # GTPO all-fail shaping (item 4c): partial reward = scale * max shingle-cosine
    # similarity to the nearest reference kernel, in [0, scale].
    ref = "def k(): a = tl.load(x); b = a + 1; tl.store(y, b)"
    codes = [
        "def k(): a = tl.load(x); b = a + 1; tl.store(y, b)",  # identical -> ~scale
        "totally unrelated python text here nothing alike",     # dissimilar -> ~0
    ]
    out = ac.gtpo_codesim_shaping(codes, [ref], scale=0.3)
    assert 0.29 < out[0] <= 0.3001
    assert out[1] < out[0]
    assert all(0.0 <= v <= 0.3 + 1e-9 for v in out)
    # no references -> all zeros (group stays collapsed, dropped by StarPO-S).
    assert ac.gtpo_codesim_shaping(codes, [], scale=0.3) == [0.0, 0.0]


def test_code_shingle_cosine_symmetric_and_bounded():
    a = "x = tl.load(p); y = x * 2"
    assert abs(ac.code_shingle_cosine(a, a) - 1.0) < 1e-9
    assert ac.code_shingle_cosine(a, "") == 0.0
    s = ac.code_shingle_cosine(a, "x = tl.load(p); y = x * 3")
    assert 0.0 < s < 1.0
    assert abs(ac.code_shingle_cosine(a, "z=1") - ac.code_shingle_cosine("z=1", a)) < 1e-9


# --------------------------------------------------------------------------- #
# import safety (no heavy deps at import time)
# --------------------------------------------------------------------------- #
def test_all_policy_modules_import_without_heavy_deps():
    import importlib

    for mod in ("format", "grpo", "anticollapse", "sft", "dpo", "serve", "configs"):
        importlib.import_module(f"kore.policy.{mod}")


def test_configs_defaults():
    from kore.policy.configs import GRPOConfig, SFTConfig, DPOConfig

    sft = SFTConfig()
    assert sft.model_id == "Qwen/Qwen3-14B"
    assert abs(sft.learning_rate - 1e-5) < 1e-12
    assert sft.max_seq_length == 16384

    dpo = DPOConfig()
    assert abs(dpo.beta - 0.1) < 1e-12

    grpo_cfg = GRPOConfig()
    assert grpo_cfg.model_id == "Qwen/Qwen3-32B"
    assert grpo_cfg.num_trajectories == 16 and grpo_cfg.num_turns == 4
    assert abs(grpo_cfg.gamma - 0.4) < 1e-12
    assert abs(grpo_cfg.correctness_weight - 0.3) < 1e-12
    # The only anchor is ref_anchor_coef (Kevin's "KL=0" == a tiny retention
    # anchor); the dead/misleading flags were removed (see test_grpo_config_flags).
    assert abs(grpo_cfg.ref_anchor_coef - 1e-3) < 1e-12


def test_grpo_config_flags_wired_or_removed():
    """Audit fix 3+4: dead/misleading GRPOConfig flags are removed; the wired
    ones survive; the FSDP/distributed fields mirror SFTConfig."""
    from kore.policy.configs import GRPOConfig, fsdp_enabled

    cfg = GRPOConfig()
    # REMOVED (dead/misleading for the native single-process loop).
    for dead in ("kl_coef", "rollout_backend", "tensor_parallel_size",
                 "gradient_accumulation_steps", "per_device_train_batch_size"):
        assert not hasattr(cfg, dead), f"{dead} should be removed from GRPOConfig"
    # WIRED (kept because the loop now honors them).
    assert cfg.bf16 is True
    assert cfg.lr_scheduler_type == "constant" and cfg.warmup_ratio == 0.0
    assert cfg.max_prompt_length == 16384
    assert cfg.save_steps == 50 and cfg.logging_steps == 1

    # Fix 4: FSDP/distributed fields (from DistributedMixin) mirror SFTConfig.
    assert cfg.distributed is False
    assert cfg.fsdp == "full_shard auto_wrap"
    assert cfg.fsdp_transformer_layer_cls is None
    # distributed full-FT GRPO takes the FSDP path; LoRA / single-process do not.
    assert fsdp_enabled(GRPOConfig(use_lora=False, distributed=True)) is True
    assert fsdp_enabled(GRPOConfig(use_lora=True, distributed=True)) is False
    assert fsdp_enabled(GRPOConfig(use_lora=False, distributed=False)) is False
