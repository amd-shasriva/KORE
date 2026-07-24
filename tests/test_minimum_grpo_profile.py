"""Contract tests for configs/grpo_32b_min_trustworthy.json."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from kore.policy import grpo
from kore.policy.capabilities import (
    AUDITED_RESEARCH_FEATURES,
    FeatureConfigurationError,
    apply_runtime_env,
    validate_grpo_startup,
)
from kore.policy.configs import GRPOConfig


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs" / "grpo_32b_min_trustworthy.json"
TRAIN_TASKS = ["gemm_bf16", "rmsnorm_aiter"]


def _raw():
    return json.loads(PROFILE_PATH.read_text())


def _config():
    return grpo.grpo_config_from_dict(_raw())


def test_profile_is_strict_minimum_without_claiming_a_launch():
    raw = _raw()
    cfg = _config()
    assert cfg.production_profile == "minimum_32b_v1"
    assert cfg.model_id == "Qwen/Qwen3-32B"
    assert cfg.strict_feature_validation is True
    assert cfg.curriculum_mode == "registered_stratified"
    assert cfg.use_lora is False
    assert cfg.budget_limits == {}  # caps require measured launch preflight
    assert "fit" not in raw and "launch_approved" not in raw


def test_all_audited_and_adjacent_research_levers_are_disabled():
    cfg = _config()
    assert cfg.use_search is False
    assert cfg.value_prefilter is False
    assert cfg.coevolve_mint is False
    assert cfg.search_bnb is False
    assert cfg.coevolve_regret_vs_opus is False
    assert cfg.coevolve_distill_path is None
    assert cfg.coevolve is False
    assert cfg.rc_grpo is False
    assert cfg.sc_grpo is False
    assert cfg.gtpo_codesim is False
    assert cfg.search_value_prior is False
    assert cfg.transform_discover is False
    assert cfg.agentic_transform_tools is False
    assert cfg.credit_incorrect_turns is False
    assert cfg.physics_shaping_weight == 0.0
    assert cfg.physics_live_counters is False
    assert set(AUDITED_RESEARCH_FEATURES) == {
        "use_search",
        "value_prefilter",
        "coevolve_mint",
        "search_bnb",
        "coevolve_regret_vs_opus",
        "distillation",
    }


def test_only_provisional_antocollapse_features_and_agentic_are_effective():
    cfg = _config()
    manifest = validate_grpo_startup(cfg, TRAIN_TASKS)
    assert set(manifest.effective_features) == {
        "agentic",
        "starpo_s",
        "dynamic_sampling",
        "avspo",
    }
    assert set(manifest.provisional_features) == {
        "starpo_s",
        "dynamic_sampling",
        "avspo",
    }
    expected = {
        feature: (phase, minimum)
        for feature, phase, minimum in manifest.expected_liveness
    }
    assert expected["dynamic_sampling"] == ("rollout", 1)
    assert expected["starpo_s"] == ("update", 1)
    assert expected["avspo"] == ("update", 1)


@pytest.mark.parametrize(
    "change",
    [
        {"use_search": True},
        {"value_prefilter": True},
        {"coevolve": True},
        {"rc_grpo": True},
        {"sc_grpo": True},
        {"gtpo_codesim": True},
        {"credit_incorrect_turns": True},
        {"physics_shaping_weight": 0.1},
        {"physics_live_counters": True, "reward_mode": "residual"},
        {"agentic_transform_tools": True},
    ],
)
def test_minimum_profile_rejects_research_lever_drift(change):
    cfg = replace(_config(), **change)
    with pytest.raises(FeatureConfigurationError):
        cfg.validate()


def test_disabled_optional_modules_are_not_called(monkeypatch):
    cfg = _config()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled feature hook was invoked")

    monkeypatch.setattr(grpo, "_activate_value_ranker", forbidden)
    monkeypatch.setattr(grpo, "_build_distill_sink", forbidden)
    monkeypatch.setattr(grpo, "_maybe_search_then_distill", forbidden)
    assert (
        grpo._initialize_optional_features(cfg, build_distill_sink=True) is None
    )
    assert grpo._run_optional_search([], cfg, None, 0) is None


def test_disabled_feature_packages_are_not_imported_by_startup_helpers(monkeypatch):
    cfg = _config()
    names = (
        "kore.search.propose",
        "kore.value.rerank",
        "kore.openended.controller",
        "kore.policy.coevolve_distill",
    )
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    grpo._initialize_optional_features(cfg, build_distill_sink=True)
    grpo._run_optional_search([], cfg, None, 0)
    assert all(name not in sys.modules for name in names)


def test_disabled_physics_module_is_not_invoked_by_agent_tools(monkeypatch):
    from kore.agent.tools import ToolExecutor
    from kore.reward.reward import Observation
    import kore.reward.whitebox as whitebox

    class Task:
        task_id = "fake"
        operation = "gemm"
        dtype = "bf16"
        gpu_target = "gfx950"

    class Env:
        def step(self, *_args, **_kwargs):
            return Observation(
                compiled=True,
                validation_passed=True,
                snr_db=50.0,
                snr_by_shape={"primary": 50.0},
                wall_ms=1.0,
                baseline_ms=2.0,
                wall_by_shape={"primary": 1.0},
                baseline_by_shape={"primary": 2.0},
            )

    monkeypatch.setenv("KORE_PHYSICS_SHAPING", "1")  # stale parent value
    apply_runtime_env(_config())
    monkeypatch.setattr(
        whitebox,
        "phi_potential",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled physics module invoked")
        ),
    )
    result = ToolExecutor(Env(), Task()).dispatch(
        {"name": "bench", "arguments": {"kernel_src": "def kernel(): pass"}}
    )
    assert result["ok"] is True


def test_json_parser_rejects_lora_launch_request():
    raw = _raw()
    raw["use_lora"] = True
    with pytest.raises(FeatureConfigurationError, match="LoRA"):
        grpo.grpo_config_from_dict(raw)


def test_legacy_constructor_and_parser_remain_compatible():
    cfg = GRPOConfig()
    assert cfg.strict_feature_validation is False
    parsed = grpo.grpo_config_from_dict(
        {
            "model_id": "legacy",
            "use_lora": False,
            "lora": {"r": 8},
            "tasks": ["ignored_by_parser"],
        }
    )
    assert parsed.model_id == "legacy"
    assert not hasattr(parsed, "lora")
