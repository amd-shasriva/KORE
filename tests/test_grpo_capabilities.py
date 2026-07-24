"""CPU-only rejection and liveness tests for strict GRPO capabilities."""

from __future__ import annotations

import json

import pytest

from kore.policy import grpo
from kore.policy.budget import BudgetLedgerV1
from kore.policy.capabilities import (
    FeatureConfigurationError,
    FeatureManifest,
    FeatureLivenessError,
    FeatureRuntime,
    apply_runtime_env,
    emit_feature_manifest,
    initialize_grpo_foundations,
    stale_feature_env,
    validate_grpo_startup,
)
from kore.policy.configs import GRPOConfig


TRAIN_TASKS = ["gemm_bf16", "rmsnorm_aiter"]


def _strict(**overrides) -> GRPOConfig:
    base = dict(
        production_profile="test_strict_v1",
        strict_feature_validation=True,
        use_lora=False,
        num_candidates_per_turn=1,
        value_prefilter_k=1,
    )
    base.update(overrides)
    return GRPOConfig(**base)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"agentic": True, "value_prefilter": True}, "agentic value_prefilter"),
        ({"starpo_s": False, "dynamic_sampling": True}, "requires parent starpo_s"),
        ({"agentic": False, "agentic_transform_tools": True}, "requires parent agentic"),
        ({"use_search": False, "search_bnb": True}, "requires parent use_search"),
        ({"use_search": False, "search_value_prior": True}, "requires parent use_search"),
        ({"transform_discover": True}, "requires use_search"),
        ({"coevolve": False, "coevolve_mint": True}, "requires parent coevolve"),
        (
            {"coevolve_mint": False, "coevolve_evolve_grammar": True},
            "requires parent coevolve_mint",
        ),
        (
            {"coevolve": False, "coevolve_regret_vs_opus": True},
            "requires parent coevolve",
        ),
        (
            {"coevolve_opus_scores_path": "scores.json"},
            "requires coevolve_regret_vs_opus",
        ),
        ({"value_model_path": "value.pkl"}, "no enabled value-model consumer"),
        (
            {"curriculum_mode": "legacy", "curriculum_state_path": "state.json"},
            "requires registered_stratified",
        ),
        (
            {"curriculum_mode": "legacy", "resume_state_required": True},
            "requires registered_stratified",
        ),
        (
            {"physics_live_counters": True, "physics_shaping_weight": 0.0},
            "requires physics shaping",
        ),
        (
            {"adversarial_coevolve": True, "coevolve": False},
            "requires parent coevolve",
        ),
        (
            {"distributed": True, "agentic": True, "rc_grpo": True},
            "distributed-agentic RC-GRPO",
        ),
        (
            {"distributed": True, "agentic": True, "sc_grpo": True},
            "distributed-agentic SC-GRPO",
        ),
    ],
)
def test_disabled_parent_and_inert_path_rejection(overrides, message):
    with pytest.raises(FeatureConfigurationError, match=message):
        validate_grpo_startup(_strict(**overrides), TRAIN_TASKS)


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_search": True},
        {"value_prefilter": True},
        {"coevolve": True, "coevolve_mint": True},
        {"use_search": True, "search_bnb": True},
        {"coevolve": True, "coevolve_regret_vs_opus": True},
        {"coevolve_distill_path": "wins.jsonl"},
    ],
)
def test_six_audited_research_levers_have_no_strict_consumer(overrides):
    with pytest.raises(
        FeatureConfigurationError,
        match="consumer|artifact|requested_features",
    ):
        validate_grpo_startup(_strict(**overrides), TRAIN_TASKS)


def test_requested_features_exactly_equal_effective_features():
    manifest = validate_grpo_startup(_strict(), TRAIN_TASKS)
    assert manifest.requested_features == manifest.effective_features
    assert {"starpo_s", "dynamic_sampling", "avspo"} <= set(
        manifest.effective_features
    )
    assert manifest.task_set_digest


@pytest.mark.parametrize("tasks", [None, [], ["unknown_task"], ["mla_decode_bf16"]])
def test_empty_unknown_and_heldout_tasks_rejected(tasks):
    with pytest.raises(FeatureConfigurationError):
        validate_grpo_startup(_strict(), tasks)


def test_explicit_heldout_overlap_rejected():
    with pytest.raises(FeatureConfigurationError, match="overlaps held-out"):
        validate_grpo_startup(
            _strict(), TRAIN_TASKS, explicit_heldout_ids=["rmsnorm_aiter"]
        )


def test_unsupported_lora_rejected_on_validation_but_constructible():
    cfg = _strict(use_lora=True)
    assert cfg.use_lora is True  # migration/inspection remains backward-compatible
    with pytest.raises(FeatureConfigurationError, match="LoRA is unsupported"):
        cfg.validate()
    with pytest.raises(FeatureConfigurationError, match="LoRA is unsupported"):
        grpo.train_grpo(cfg, tasks=TRAIN_TASKS)


@pytest.mark.parametrize(
    "overrides",
    [
        {"budget_limits": {"generated_tokens": -1}},
        {"budget_limits": {"verifier_gpu_seconds": float("nan")}},
        {"budget_limits": {"not_a_counter": 1}},
        {"target_groups": 4, "max_sampling_attempts": 3},
        {"search_budget": 0},
    ],
)
def test_invalid_budgets_rejected(overrides):
    with pytest.raises(FeatureConfigurationError):
        _strict(**overrides).validate()


def test_stale_feature_environment_is_cleared_without_touching_unrelated_keys():
    cfg = _strict()
    env = {
        "KORE_USE_SEARCH": "1",
        "KORE_SEARCH_BNB": "1",
        "KORE_PHYSICS_LIVE_COUNTERS": "1",
        "KORE_PHYSICS_SHAPING": "1",
        "KORE_MINTER_EVOLVE_GRAMMAR": "1",
        "KORE_PROFILE_REWARD_WEIGHT": "0.4",
        "KORE_REWARD_MODE": "residual",
        "KORE_TRAIN_ARCHS": "gfx950",
        "PATH": "/bin",
    }
    assert stale_feature_env(cfg, env)
    changes = apply_runtime_env(cfg, env)
    assert changes
    for key in (
        "KORE_USE_SEARCH",
        "KORE_SEARCH_BNB",
        "KORE_PHYSICS_LIVE_COUNTERS",
        "KORE_PHYSICS_SHAPING",
        "KORE_MINTER_EVOLVE_GRAMMAR",
        "KORE_PROFILE_REWARD_WEIGHT",
    ):
        assert key not in env
    assert env["KORE_REWARD_MODE"] == "speedup"
    assert env["KORE_REWARD_PHASE"] == "all"
    assert env["KORE_TRAIN_ARCHS"] == "gfx950"
    assert stale_feature_env(cfg, env) == {}


def test_manifest_is_stable_and_atomic(tmp_path):
    first = validate_grpo_startup(_strict(), TRAIN_TASKS)
    second = validate_grpo_startup(_strict(), list(TRAIN_TASKS))
    assert first == second
    assert first.digest == second.digest
    assert FeatureManifest.from_dict(first.to_dict()) == first
    path = emit_feature_manifest(first, tmp_path)
    before = path.read_bytes()
    emit_feature_manifest(second, tmp_path)
    assert path.read_bytes() == before


def test_first_rollout_and_update_liveness_canaries():
    runtime = initialize_grpo_foundations(_strict(), TRAIN_TASKS)
    runtime.invoked("dynamic_sampling")
    runtime.assert_phase("rollout")
    with pytest.raises(FeatureLivenessError, match="avspo|starpo_s"):
        runtime.assert_phase("update")
    runtime.invoked("starpo_s")
    runtime.invoked("avspo")
    runtime.assert_phase("update")
    assert runtime.ledger.feature_count("dynamic_sampling") == 1


def test_disabled_feature_invocation_fails_closed():
    manifest = validate_grpo_startup(_strict(), TRAIN_TASKS)
    runtime = FeatureRuntime(manifest, BudgetLedgerV1())
    with pytest.raises(FeatureLivenessError, match="disabled feature"):
        runtime.invoked("use_search")


def test_manifest_changes_when_feature_config_changes():
    first = validate_grpo_startup(_strict(seed=1), TRAIN_TASKS)
    second = validate_grpo_startup(_strict(seed=2), TRAIN_TASKS)
    assert first.config_digest != second.config_digest
    assert first.digest != second.digest


def test_train_entry_emits_manifest_and_zero_ledger_before_backend(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    cfg = _strict(
        output_dir=str(tmp_path), curriculum_mode="registered_stratified"
    )
    monkeypatch.setattr(
        grpo, "_train_grpo_inprocess", lambda config, tasks: "stub-output"
    )
    assert grpo.train_grpo(cfg, tasks=TRAIN_TASKS) == "stub-output"
    manifest = json.loads((tmp_path / "feature_manifest.json").read_text())
    ledger = json.loads((tmp_path / "budget_ledger.json").read_text())
    curriculum = json.loads((tmp_path / "curriculum_state.json").read_text())
    assert manifest["requested_features"] == manifest["effective_features"]
    assert manifest["task_set_digest"]
    assert ledger["generated_tokens"] == 0
    assert ledger["optimizer_tokens"] == 0
    assert curriculum["draw_index"] == 0
    assert curriculum["task_set_digest"] == manifest["task_set_digest"]
