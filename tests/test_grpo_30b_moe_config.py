"""The 30B MoE TRLOO recipe must not request a capability it cannot deliver.

``tests/test_grpo_recipe_honesty.py`` holds this contract for the 14B recipe. The
30B config adds three new ways to be dishonest, so each gets a test:

* an advantage estimator that is named but not the one that runs;
* an AVSPO variance floor requested alongside TRLOO, which never calls it;
* a coverage reward armed without evidence that the trace collector works on this
  hardware, which would state a reward that silently contributes 0.0 every turn.

It also pins the MoE FSDP wrap class, because getting that wrong does not crash:
``TRANSFORMER_BASED_WRAP`` with a class name matching nothing in the module tree
wraps NOTHING, so FSDP degrades to one flat parameter for the whole model and
either OOMs or loses every overlap the sharding was meant to buy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kore.policy.capabilities import (
    DECLARED,
    audit_requested_capabilities,
    format_capability_audit,
)
from kore.policy.configs import GRPOConfig, detect_transformer_layer_cls
from kore.policy.grpo import grpo_config_from_dict

REPO = Path(__file__).resolve().parents[1]
SHIPPED = "configs/grpo_coder30b_a3b_trloo.json"
SFT_CONFIG = "configs/sft_coder30b_a3b.json"


def _raw() -> dict:
    return json.loads((REPO / SHIPPED).read_text())


def _config() -> GRPOConfig:
    return grpo_config_from_dict(_raw())


# --------------------------------------------------------------------------- #
# 1. it parses, validates, and requests nothing provably inert
# --------------------------------------------------------------------------- #
def test_the_config_parses_and_validates():
    config = _config()
    assert config.validate() is None
    assert config.output_dir


def test_it_has_no_config_provable_inert_capability():
    """``ARTIFACT`` findings are excluded: the SFT checkpoint is cluster-only.

    A ``DECLARED`` finding is provable from the config text alone, so one here is
    a defect in the config rather than a missing file.
    """
    findings = [f for f in audit_requested_capabilities(_config(), root=REPO)
                if f.scope == DECLARED]
    assert not findings, format_capability_audit(findings)


#: Keys a launch JSON carries for the launcher / model-identity split rather than
#: for the stage dataclass, mirroring ``tests/test_docs_contract.py``.
LAUNCHER_OWNED = frozenset({
    "fsdp", "fsdp_version", "fsdp_transformer_layer_cls", "fsdp_cpu_offload",
    "zero_stage", "synced_gpus", "cpu_offload", "model_revision",
})


def test_every_comment_key_names_a_real_field():
    """``_comment_<field>`` is this repo's in-config documentation convention.

    A justification that names no knob is a justification for nothing, and reads
    as though the lever is still governed. A grouped rationale is anchored on the
    decision's PRIMARY field so this property still holds.
    """
    raw = _raw()
    known = set(GRPOConfig.__dataclass_fields__) | LAUNCHER_OWNED
    comments = [k for k in raw if k.startswith("_comment_")]
    assert comments, "the recipe carries no in-config justifications"
    for key in comments:
        assert key[len("_comment_"):] in known, f"{key} names no real knob"
        assert isinstance(raw[key], str) and raw[key].strip()


def test_no_unknown_keys_reach_the_dataclass():
    """A typo'd key must not be accepted and then ignored."""
    fields = set(GRPOConfig.__dataclass_fields__)
    unknown = sorted(
        k for k in _raw()
        if not k.startswith("_") and k not in fields and k not in LAUNCHER_OWNED)
    assert not unknown, f"{SHIPPED} has keys that are not GRPOConfig fields: {unknown}"


# --------------------------------------------------------------------------- #
# 2. TRLOO is actually selected, and the floor it bypasses is off
# --------------------------------------------------------------------------- #
def test_the_recipe_selects_trloo():
    """The reason this config exists; a silent "grpo" would make it pointless."""
    assert _config().advantage_estimator == "trloo"


def test_trloo_and_the_avspo_floor_are_not_both_requested():
    """TRLOO never calls avspo_advantages, so a nonzero floor is an inert request.

    It is also wrong on the merits: the floor injects virtual samples into the
    NORMALISATION statistics, and every self-inclusive statistic is what biases
    the estimator TRLOO exists to fix.
    """
    config = _config()
    if config.advantage_estimator == "trloo":
        assert config.variance_floor == 0.0
    findings = audit_requested_capabilities(
        GRPOConfig(advantage_estimator="trloo", variance_floor=0.1,
                   ref_anchor_coef=0.0))
    assert any(f.feature == "avspo" and f.scope == DECLARED for f in findings), (
        format_capability_audit(findings))


def test_the_paper_defaults_are_what_the_recipe_states():
    """Dr. Kernel's reported defaults: ROLLOUT_N=16, MAX_TURN=3, LR=1e-6."""
    config = _config()
    assert config.num_trajectories == 16
    assert config.learning_rate == pytest.approx(1e-6)
    # The agentic path's horizon is max_tool_turns, so num_turns must agree with
    # it or the config states a horizon the run does not use.
    assert config.max_tool_turns == 3
    if config.agentic:
        assert config.num_turns == config.max_tool_turns


def test_rejection_sampling_uses_the_geometric_aggregate():
    config = _config()
    assert config.rejection_sampling
    assert config.rejection_aggregate == "geometric"


# --------------------------------------------------------------------------- #
# 3. the coverage reward is not armed without evidence it works
# --------------------------------------------------------------------------- #
def test_the_coverage_reward_is_armed_only_with_a_collector_receipt():
    """``collect_kernel_trace`` fails safe, so an unvalidated weight pays nothing.

    rocprofv3's kernel-trace export layout is not confirmed on this ROCm build,
    which is precisely why arming the weight requires a receipt from a run that
    observed a non-None coverage -- the same contract ``physics_shaping`` has.
    """
    config = _config()
    if config.profiling_reward_weight > 0.0:
        assert config.profiling_reward_evidence_path
        # Shaping must never be able to outrank correctness.
        assert config.profiling_reward_weight < config.correctness_weight
    else:
        assert config.profiling_reward_weight == 0.0


def test_prs_does_not_require_a_profile_that_is_never_collected():
    """Requiring a profile with no trace collection rejects every candidate."""
    config = _config()
    if config.prs_require_profile:
        assert config.profiling_reward_weight > 0.0
    findings = audit_requested_capabilities(
        GRPOConfig(prs_require_profile=True, profiling_reward_weight=0.0,
                   ref_anchor_coef=0.0))
    assert any(f.feature == "rejection_sampling" and f.scope == DECLARED
               for f in findings), format_capability_audit(findings)


def test_physics_shaping_stays_unarmed_on_the_existing_evidence():
    """``docs/P0_RESULTS.md`` authorises no family, so the weight must be 0.0."""
    config = _config()
    assert config.physics_shaping_weight == 0.0
    assert not config.physics_live_counters


# --------------------------------------------------------------------------- #
# 4. MoE FSDP wrapping
# --------------------------------------------------------------------------- #
def test_the_moe_decoder_layer_is_named_explicitly():
    """Wrapping the wrong class is silent, not a crash."""
    raw = _raw()
    assert raw["fsdp_transformer_layer_cls"] == "Qwen3MoeDecoderLayer"


def test_the_backbone_agrees_with_the_sft_stage():
    """RL must train the model SFT produced, not a different one."""
    sft = json.loads((REPO / SFT_CONFIG).read_text())
    raw = _raw()
    assert raw["model_id"] == sft["model_id"]
    assert raw["model_revision"] == sft["model_revision"]
    assert raw["fsdp_transformer_layer_cls"] == sft["fsdp_transformer_layer_cls"]


def test_detection_would_reach_the_same_class_from_the_model_id():
    """Belt and braces must agree, or one of them is wrong."""
    assert detect_transformer_layer_cls(
        _raw()["model_id"]) == "Qwen3MoeDecoderLayer"


def test_the_explicit_plugin_overrides_the_accelerate_yaml(monkeypatch):
    """``configs/accelerate_fsdp_grpo.yaml`` still names the DENSE Qwen3 layer.

    ``accelerate launch --config_file`` exports it as ``FSDP_TRANSFORMER_CLS_TO_WRAP``,
    so this proves the explicitly-constructed plugin wins. If it ever stopped
    winning, a 30B MoE launch would wrap the wrong class and OOM with no
    explanation.
    """
    pytest.importorskip("accelerate")
    from kore.policy.grpo import build_fsdp_plugin

    monkeypatch.setenv("FSDP_TRANSFORMER_CLS_TO_WRAP", "Qwen3DecoderLayer")
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    plugin = build_fsdp_plugin(_config())
    assert list(plugin.transformer_cls_names_to_wrap) == ["Qwen3MoeDecoderLayer"]


def test_the_accelerate_yaml_says_which_value_is_authoritative():
    """The YAML's literal is overridden, so it must not read as the source of truth.

    A reader who trusts it would conclude a 30B MoE launch wraps
    ``Qwen3DecoderLayer``, which would be a real bug if the plugin path ever
    changed. The comment has to name what actually decides.
    """
    lines = (REPO / "configs" / "accelerate_fsdp_grpo.yaml").read_text().splitlines()
    at = next(i for i, l in enumerate(lines)
              if "fsdp_transformer_layer_cls_to_wrap" in l)
    window = "\n".join(lines[max(0, at - 8):at + 1])
    assert "build_fsdp_plugin" in window, window
    assert "Qwen3MoeDecoderLayer" in window, (
        "the comment must name the class the MoE backbone actually needs")


# --------------------------------------------------------------------------- #
# 5. disk, which a 30B run can genuinely exhaust
# --------------------------------------------------------------------------- #
def test_the_checkpoint_budget_matches_the_30b_arithmetic():
    """Adam keeps ~16 bytes/param, so a 30.5B checkpoint is ~488GB.

    A rotation transiently holds two, peaking near 976GB on a SHARED volume, so
    save_total_limit above 1 does not fit. The SFT config makes the same argument
    for the same model; the two must not disagree.
    """
    config = _config()
    sft = json.loads((REPO / SFT_CONFIG).read_text())
    assert config.save_total_limit == 1
    assert sft["save_total_limit"] == 1
    assert config.save_steps > 0


def test_the_kl_anchor_is_explicit():
    """Null silently anchors to model_id, which the launcher rewrites."""
    config = _config()
    assert config.ref_checkpoint
    if config.ref_anchor_coef > 0.0:
        assert config.ref_checkpoint != config.output_dir
