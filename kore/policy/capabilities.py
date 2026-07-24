"""Fail-closed GRPO feature resolution, manifests, and liveness.

The legacy GRPO configuration remains constructible for inspection and old
checkpoints, but a strict profile must prove that every requested feature has a
real consumer in the selected runtime.  Silent fallback is a configuration
error: ``requested_features`` and ``effective_features`` must be identical.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from kore.policy.budget import BudgetLedgerV1, BudgetLimitsV1


FEATURE_MANIFEST_VERSION = "FeatureManifestV1"

AUDITED_RESEARCH_FEATURES = (
    "use_search",
    "value_prefilter",
    "coevolve_mint",
    "search_bnb",
    "coevolve_regret_vs_opus",
    "distillation",
)

_KNOWN_FEATURES = (
    "agentic",
    "agentic_transform_tools",
    "adversarial_coevolve",
    "avspo",
    "coevolve",
    "coevolve_evolve_grammar",
    "coevolve_mint",
    "coevolve_regret_vs_opus",
    "credit_incorrect_turns",
    "distillation",
    "dynamic_sampling",
    "gtpo_codesim",
    "physics_live_counters",
    "physics_shaping",
    "rc_grpo",
    "roofline_gate",
    "sc_grpo",
    "search_bnb",
    "search_value_prior",
    "starpo_s",
    "transform_discover",
    "use_search",
    "value_prefilter",
)

# These flags were audited as unsound or without a trustworthy production
# consumer.  Artifact schemas described by the redesign do not exist yet, so a
# strict run cannot make them effective merely by supplying an arbitrary path.
_UNSUPPORTED_STRICT_FEATURES = frozenset(AUDITED_RESEARCH_FEATURES)

_EXPECTED_PHASE = {
    "agentic": "rollout",
    "agentic_transform_tools": "rollout",
    "adversarial_coevolve": "rollout",
    "coevolve": "rollout",
    "coevolve_evolve_grammar": "rollout",
    "coevolve_mint": "rollout",
    "coevolve_regret_vs_opus": "rollout",
    "credit_incorrect_turns": "rollout",
    "distillation": "rollout",
    "dynamic_sampling": "rollout",
    "physics_live_counters": "rollout",
    "physics_shaping": "rollout",
    "rc_grpo": "rollout",
    "roofline_gate": "rollout",
    "transform_discover": "rollout",
    "use_search": "rollout",
    "value_prefilter": "rollout",
    "avspo": "update",
    "gtpo_codesim": "update",
    "sc_grpo": "update",
    "search_bnb": "update",
    "search_value_prior": "update",
    "starpo_s": "update",
}
_PHASE_ORDER = {"rollout": 0, "update": 1}

_MANAGED_FEATURE_ENV = frozenset(
    {
        "KORE_ADVERSARIAL_COEVOLVE",
        "KORE_COEVOLVE_DISTILL",
        "KORE_COEVOLVE_MINT",
        "KORE_COEVOLVE_REGRET_VS_OPUS",
        "KORE_GTPO_CODESIM",
        "KORE_MINTER_EVOLVE_GRAMMAR",
        "KORE_PHYSICS_LIVE_COUNTERS",
        "KORE_PHYSICS_SHAPING",
        "KORE_PROFILE_REWARD_WEIGHT",
        "KORE_RC_GRPO",
        "KORE_REWARD_MODE",
        "KORE_REWARD_PHASE",
        "KORE_ROOFLINE_GATE",
        "KORE_ROOFLINE_TOL",
        "KORE_SC_GRPO",
        "KORE_SEARCH_BNB",
        "KORE_TRANSFORM_DISCOVER",
        "KORE_USE_SEARCH",
        "KORE_VALUE_PREFILTER",
    }
)


class FeatureConfigurationError(ValueError):
    """Raised when a requested feature cannot be made effective."""


class FeatureLivenessError(RuntimeError):
    """Raised when an enabled feature misses its first-phase canary."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_json(payload: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        _jsonable(payload),
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    )


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _config_payload(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return {
            field.name: _jsonable(getattr(config, field.name))
            for field in fields(config)
            if not field.name.startswith("_")
        }
    return {
        key: _jsonable(value)
        for key, value in vars(config).items()
        if not key.startswith("_") and not callable(value)
    }


def _requested_features(config: Any) -> set[str]:
    requested: set[str] = set()
    boolean_flags = {
        "agentic": "agentic",
        "agentic_transform_tools": "agentic_transform_tools",
        "adversarial_coevolve": "adversarial_coevolve",
        "coevolve": "coevolve",
        "coevolve_evolve_grammar": "coevolve_evolve_grammar",
        "coevolve_mint": "coevolve_mint",
        "coevolve_regret_vs_opus": "coevolve_regret_vs_opus",
        "credit_incorrect_turns": "credit_incorrect_turns",
        "dynamic_sampling": "dynamic_sampling",
        "gtpo_codesim": "gtpo_codesim",
        "physics_live_counters": "physics_live_counters",
        "rc_grpo": "rc_grpo",
        "roofline_gate": "roofline_gate",
        "sc_grpo": "sc_grpo",
        "search_bnb": "search_bnb",
        "search_value_prior": "search_value_prior",
        "starpo_s": "starpo_s",
        "transform_discover": "transform_discover",
        "use_search": "use_search",
        "value_prefilter": "value_prefilter",
    }
    for attr, feature in boolean_flags.items():
        if bool(getattr(config, attr, False)):
            requested.add(feature)
    if float(getattr(config, "variance_floor", 0.0) or 0.0) > 0.0:
        requested.add("avspo")
    if float(getattr(config, "physics_shaping_weight", 0.0) or 0.0) > 0.0:
        requested.add("physics_shaping")
    if getattr(config, "coevolve_distill_path", None):
        requested.add("distillation")
    return requested


def _positive_int(config: Any, name: str, errors: list[str], *, optional: bool = False) -> None:
    value = getattr(config, name, None)
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{name} must be a positive integer")


def _finite_range(
    config: Any,
    name: str,
    errors: list[str],
    *,
    minimum: float,
    maximum: Optional[float] = None,
    minimum_inclusive: bool = True,
) -> None:
    value = getattr(config, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{name} must be numeric")
        return
    value = float(value)
    if not math.isfinite(value):
        errors.append(f"{name} must be finite")
        return
    too_low = value < minimum if minimum_inclusive else value <= minimum
    if too_low or (maximum is not None and value > maximum):
        right = f", {maximum}]" if maximum is not None else ", infinity)"
        left = "[" if minimum_inclusive else "("
        errors.append(f"{name} must be in {left}{minimum}{right}")


def validate_grpo_config(config: Any, *, strict: Optional[bool] = None) -> None:
    """Validate scalar budgets and, for strict profiles, feature topology."""

    strict = (
        bool(getattr(config, "strict_feature_validation", False))
        if strict is None
        else bool(strict)
    )
    errors: list[str] = []

    if bool(getattr(config, "use_lora", False)):
        errors.append("GRPO LoRA is unsupported; use_lora must be false")

    for name in (
        "num_trajectories",
        "num_turns",
        "tasks_per_step",
        "ppo_epochs",
        "total_steps",
        "max_prompt_length",
        "max_response_length",
        "search_budget",
        "search_every",
        "search_k_expand",
    ):
        _positive_int(config, name, errors)
    for name in ("target_groups", "max_sampling_attempts", "search_max_depth"):
        _positive_int(config, name, errors, optional=True)
    if bool(getattr(config, "agentic", False)):
        _positive_int(config, "max_tool_turns", errors)

    _finite_range(config, "temperature", errors, minimum=0.0, minimum_inclusive=False)
    _finite_range(config, "top_p", errors, minimum=0.0, maximum=1.0, minimum_inclusive=False)
    _finite_range(config, "starpo_min_std", errors, minimum=0.0)
    _finite_range(
        config,
        "starpo_keep_frac",
        errors,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    _finite_range(config, "variance_floor", errors, minimum=0.0)
    _finite_range(config, "physics_shaping_weight", errors, minimum=0.0)
    _finite_range(config, "ref_anchor_coef", errors, minimum=0.0)
    _finite_range(config, "learning_rate", errors, minimum=0.0, minimum_inclusive=False)
    _finite_range(config, "max_grad_norm", errors, minimum=0.0, minimum_inclusive=False)
    _finite_range(config, "adv_eps", errors, minimum=0.0, minimum_inclusive=False)
    _finite_range(config, "gamma", errors, minimum=0.0, maximum=1.0)

    try:
        BudgetLimitsV1.from_mapping(getattr(config, "budget_limits", None))
    except ValueError as exc:
        errors.append(str(exc))

    target = getattr(config, "target_groups", None) or getattr(
        config, "tasks_per_step", 0
    )
    attempts = getattr(config, "max_sampling_attempts", None)
    if attempts is not None and isinstance(target, int) and attempts < target:
        errors.append("max_sampling_attempts cannot be smaller than target_groups")

    if strict:
        requested = _requested_features(config)
        profile = getattr(config, "production_profile", None)
        if profile is not None and (
            not isinstance(profile, str) or not profile.strip()
        ):
            errors.append("production_profile must be a non-empty string or null")
        curriculum_mode = getattr(config, "curriculum_mode", "legacy")
        if curriculum_mode not in {"legacy", "registered_stratified"}:
            errors.append(
                "curriculum_mode must be 'legacy' or 'registered_stratified'"
            )
        if getattr(config, "curriculum_state_path", None) and (
            curriculum_mode != "registered_stratified"
        ):
            errors.append(
                "curriculum_state_path requires registered_stratified curriculum"
            )
        if bool(getattr(config, "resume_state_required", False)) and (
            curriculum_mode != "registered_stratified"
        ):
            errors.append(
                "resume_state_required requires registered_stratified curriculum"
            )
        if bool(getattr(config, "agentic", False)) and bool(
            getattr(config, "value_prefilter", False)
        ):
            errors.append("agentic value_prefilter has no on-policy consumer")
        if bool(getattr(config, "dynamic_sampling", False)) and not bool(
            getattr(config, "starpo_s", False)
        ):
            errors.append("dynamic_sampling requires parent starpo_s")
        if bool(getattr(config, "agentic_transform_tools", False)) and not bool(
            getattr(config, "agentic", False)
        ):
            errors.append("agentic_transform_tools requires parent agentic")
        if bool(getattr(config, "search_bnb", False)) and not bool(
            getattr(config, "use_search", False)
        ):
            errors.append("search_bnb requires parent use_search")
        if bool(getattr(config, "search_value_prior", False)) and not bool(
            getattr(config, "use_search", False)
        ):
            errors.append("search_value_prior requires parent use_search")
        if bool(getattr(config, "transform_discover", False)) and not (
            bool(getattr(config, "use_search", False))
            or bool(getattr(config, "agentic_transform_tools", False))
        ):
            errors.append(
                "transform_discover requires use_search or agentic_transform_tools"
            )
        if bool(getattr(config, "coevolve_mint", False)) and not bool(
            getattr(config, "coevolve", False)
        ):
            errors.append("coevolve_mint requires parent coevolve")
        if bool(getattr(config, "coevolve_evolve_grammar", False)) and not bool(
            getattr(config, "coevolve_mint", False)
        ):
            errors.append("coevolve_evolve_grammar requires parent coevolve_mint")
        if bool(getattr(config, "coevolve_regret_vs_opus", False)) and not bool(
            getattr(config, "coevolve", False)
        ):
            errors.append("coevolve_regret_vs_opus requires parent coevolve")
        if getattr(config, "coevolve_opus_scores_path", None) and not bool(
            getattr(config, "coevolve_regret_vs_opus", False)
        ):
            errors.append(
                "coevolve_opus_scores_path requires coevolve_regret_vs_opus"
            )
        if getattr(config, "value_model_path", None) and not (
            bool(getattr(config, "value_prefilter", False))
            or bool(getattr(config, "search_value_prior", False))
        ):
            errors.append(
                "value_model_path has no enabled value-model consumer"
            )
        if bool(getattr(config, "search_value_prior", False)):
            value_path = getattr(config, "value_model_path", None)
            if not value_path or not Path(value_path).is_file():
                errors.append(
                    "search_value_prior requires an existing value-model artifact"
                )
        if bool(getattr(config, "coevolve_regret_vs_opus", False)):
            opus_path = getattr(config, "coevolve_opus_scores_path", None)
            if not opus_path or not Path(opus_path).is_file():
                errors.append(
                    "Opus regret requires an existing accepted score artifact"
                )
        if bool(getattr(config, "physics_live_counters", False)) and not (
            float(getattr(config, "physics_shaping_weight", 0.0) or 0.0) > 0.0
            or getattr(config, "reward_mode", "speedup") == "residual"
        ):
            errors.append(
                "physics_live_counters requires physics shaping or residual reward"
            )
        if bool(getattr(config, "adversarial_coevolve", False)) and not bool(
            getattr(config, "coevolve", False)
        ):
            errors.append("adversarial_coevolve requires parent coevolve")
        if (
            bool(getattr(config, "distributed", False))
            and bool(getattr(config, "agentic", False))
            and bool(getattr(config, "rc_grpo", False))
        ):
            errors.append("distributed-agentic RC-GRPO has no active consumer")
        if (
            bool(getattr(config, "distributed", False))
            and bool(getattr(config, "agentic", False))
            and bool(getattr(config, "sc_grpo", False))
        ):
            errors.append("distributed-agentic SC-GRPO has no active consumer")
        if not bool(getattr(config, "value_prefilter", False)):
            if int(getattr(config, "num_candidates_per_turn", 1)) != 1:
                errors.append(
                    "num_candidates_per_turn must be 1 when value_prefilter is disabled"
                )
            if int(getattr(config, "value_prefilter_k", 1)) != 1:
                errors.append(
                    "value_prefilter_k must be 1 when value_prefilter is disabled"
                )

        provisional = list(getattr(config, "provisional_features", ()) or ())
        if len(provisional) != len(set(provisional)):
            errors.append("provisional_features must not contain duplicates")
        unknown_provisional = sorted(set(provisional) - set(_KNOWN_FEATURES))
        if unknown_provisional:
            errors.append(
                "unknown provisional feature(s): " + ", ".join(unknown_provisional)
            )
        missing_provisional = sorted(set(provisional) - requested)
        if missing_provisional:
            errors.append(
                "provisional features are not enabled: "
                + ", ".join(missing_provisional)
            )
        canaries = getattr(config, "required_feature_canaries", {}) or {}
        if not isinstance(canaries, Mapping):
            errors.append("required_feature_canaries must be a mapping")
        else:
            for feature, count in canaries.items():
                if feature not in _KNOWN_FEATURES:
                    errors.append(f"unknown feature canary: {feature}")
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    errors.append(
                        f"feature canary {feature!r} must be a positive integer"
                    )
            if bool(getattr(config, "require_canary_counters", False)):
                missing_canaries = sorted(set(provisional) - set(canaries))
                if missing_canaries:
                    errors.append(
                        "provisional features missing canary counters: "
                        + ", ".join(missing_canaries)
                    )

        if profile == "minimum_32b_v1":
            if curriculum_mode != "registered_stratified":
                errors.append(
                    "minimum_32b_v1 requires curriculum_mode=registered_stratified"
                )
            must_be_disabled = {
                "coevolve",
                "rc_grpo",
                "sc_grpo",
                "gtpo_codesim",
                "value_prefilter",
                "use_search",
                "search_bnb",
                "search_value_prior",
                "transform_discover",
                "coevolve_mint",
                "coevolve_evolve_grammar",
                "coevolve_regret_vs_opus",
                "distillation",
                "agentic_transform_tools",
                "adversarial_coevolve",
                "credit_incorrect_turns",
                "physics_shaping",
                "physics_live_counters",
            }
            enabled_forbidden = sorted(requested & must_be_disabled)
            if enabled_forbidden:
                errors.append(
                    "minimum_32b_v1 forbids: " + ", ".join(enabled_forbidden)
                )
            required_provisional = {"starpo_s", "dynamic_sampling", "avspo"}
            if not required_provisional <= requested:
                errors.append(
                    "minimum_32b_v1 requires provisional StarPO-S, dynamic "
                    "sampling, and AVSPO"
                )
            if set(provisional) != required_provisional:
                errors.append(
                    "minimum_32b_v1 provisional_features must be exactly "
                    "avspo,dynamic_sampling,starpo_s"
                )
            if not bool(getattr(config, "require_canary_counters", False)):
                errors.append("minimum_32b_v1 requires feature canary counters")

    if errors:
        raise FeatureConfigurationError(
            "invalid GRPO configuration:\n- " + "\n- ".join(dict.fromkeys(errors))
        )


def _effective_features(config: Any, requested: set[str]) -> set[str]:
    effective: set[str] = set(requested) - _UNSUPPORTED_STRICT_FEATURES
    distributed = bool(getattr(config, "distributed", False))
    agentic = bool(getattr(config, "agentic", False))
    if agentic:
        effective.discard("rc_grpo")
    if distributed:
        effective.discard("sc_grpo")
    if "starpo_s" not in effective:
        effective.discard("dynamic_sampling")
    if "agentic" not in effective:
        effective.discard("agentic_transform_tools")
    if "coevolve" not in effective:
        effective.difference_update(
            {
                "coevolve_mint",
                "coevolve_evolve_grammar",
                "coevolve_regret_vs_opus",
                "adversarial_coevolve",
            }
        )
    if "coevolve_mint" not in effective:
        effective.discard("coevolve_evolve_grammar")
    if "use_search" not in effective:
        effective.difference_update({"search_bnb", "search_value_prior"})
    if not (
        "use_search" in effective or "agentic_transform_tools" in effective
    ):
        effective.discard("transform_discover")
    return effective


@dataclass(frozen=True)
class FeatureManifest:
    schema_version: str
    production_profile: Optional[str]
    requested_features: tuple[str, ...]
    effective_features: tuple[str, ...]
    disabled_features: tuple[str, ...]
    provisional_features: tuple[str, ...]
    expected_liveness: tuple[tuple[str, str, int], ...]
    config_digest: str
    task_set_digest: str
    runtime: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "production_profile": self.production_profile,
            "requested_features": list(self.requested_features),
            "effective_features": list(self.effective_features),
            "disabled_features": list(self.disabled_features),
            "provisional_features": list(self.provisional_features),
            "expected_liveness": [
                {"feature": feature, "phase": phase, "minimum_invocations": minimum}
                for feature, phase, minimum in self.expected_liveness
            ],
            "config_digest": self.config_digest,
            "task_set_digest": self.task_set_digest,
            "runtime": dict(self.runtime),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FeatureManifest":
        if not isinstance(raw, Mapping):
            raise FeatureConfigurationError("feature manifest must be a mapping")
        expected_keys = {
            "schema_version",
            "production_profile",
            "requested_features",
            "effective_features",
            "disabled_features",
            "provisional_features",
            "expected_liveness",
            "config_digest",
            "task_set_digest",
            "runtime",
        }
        if set(raw) != expected_keys:
            missing = sorted(expected_keys - set(raw))
            unknown = sorted(set(raw) - expected_keys)
            raise FeatureConfigurationError(
                f"malformed feature manifest (missing={missing}, unknown={unknown})"
            )
        if raw["schema_version"] != FEATURE_MANIFEST_VERSION:
            raise FeatureConfigurationError(
                f"unsupported feature manifest schema: {raw['schema_version']!r}"
            )
        feature_lists: dict[str, tuple[str, ...]] = {}
        for key in (
            "requested_features",
            "effective_features",
            "disabled_features",
            "provisional_features",
        ):
            value = raw[key]
            if (
                not isinstance(value, list)
                or any(not isinstance(item, str) for item in value)
                or len(value) != len(set(value))
                or tuple(value) != tuple(sorted(value))
                or not set(value) <= set(_KNOWN_FEATURES)
            ):
                raise FeatureConfigurationError(
                    f"feature manifest {key} is not canonical"
                )
            feature_lists[key] = tuple(value)
        if not isinstance(raw["runtime"], Mapping):
            raise FeatureConfigurationError("feature manifest runtime must be a mapping")
        liveness: list[tuple[str, str, int]] = []
        for item in raw["expected_liveness"]:
            if not isinstance(item, Mapping) or set(item) != {
                "feature",
                "phase",
                "minimum_invocations",
            }:
                raise FeatureConfigurationError("malformed feature liveness entry")
            feature = item["feature"]
            phase = item["phase"]
            minimum = item["minimum_invocations"]
            if (
                feature not in _KNOWN_FEATURES
                or phase not in _PHASE_ORDER
                or isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or minimum <= 0
            ):
                raise FeatureConfigurationError("invalid feature liveness entry")
            liveness.append((feature, phase, minimum))
        manifest = cls(
            schema_version=FEATURE_MANIFEST_VERSION,
            production_profile=raw["production_profile"],
            requested_features=feature_lists["requested_features"],
            effective_features=feature_lists["effective_features"],
            disabled_features=feature_lists["disabled_features"],
            provisional_features=feature_lists["provisional_features"],
            expected_liveness=tuple(liveness),
            config_digest=str(raw["config_digest"]),
            task_set_digest=str(raw["task_set_digest"]),
            runtime=tuple(sorted(dict(raw["runtime"]).items())),
        )
        if manifest.requested_features != manifest.effective_features:
            raise FeatureConfigurationError(
                "feature manifest requested/effective mismatch"
            )
        if set(manifest.disabled_features) != (
            set(_KNOWN_FEATURES) - set(manifest.requested_features)
        ):
            raise FeatureConfigurationError(
                "feature manifest disabled set is not the requested complement"
            )
        if not set(manifest.provisional_features) <= set(
            manifest.effective_features
        ):
            raise FeatureConfigurationError(
                "feature manifest has disabled provisional features"
            )
        if tuple(sorted(manifest.expected_liveness)) != manifest.expected_liveness:
            raise FeatureConfigurationError("feature manifest is not canonical")
        if {item[0] for item in manifest.expected_liveness} != set(
            manifest.effective_features
        ):
            raise FeatureConfigurationError(
                "feature manifest liveness does not cover every effective feature"
            )
        return manifest

    @property
    def digest(self) -> str:
        return _sha256(self.to_dict())

    def write_json(self, path: str | os.PathLike[str]) -> Path:
        return _atomic_json_write(path, self.to_dict())


def resolve_grpo_features(
    config: Any,
    *,
    task_set_digest: str,
    runtime: Optional[Mapping[str, Any]] = None,
) -> FeatureManifest:
    """Resolve a strict manifest or raise on any inert requested feature."""

    validate_grpo_config(config, strict=True)
    requested = _requested_features(config)
    effective = _effective_features(config, requested)
    errors: list[str] = []
    unsupported = sorted(requested & _UNSUPPORTED_STRICT_FEATURES)
    if unsupported:
        errors.append(
            "no trustworthy consumer/artifact contract for: "
            + ", ".join(unsupported)
        )
    if requested != effective:
        errors.append(
            "requested_features != effective_features "
            f"(requested={sorted(requested)}, effective={sorted(effective)})"
        )
    profile = getattr(config, "production_profile", None)
    if profile == "minimum_32b_v1" and bool(getattr(config, "coevolve", False)):
        errors.append("minimum_32b_v1 requires registered tasks; coevolve must be false")
    if errors:
        raise FeatureConfigurationError(
            "GRPO capability resolution failed:\n- " + "\n- ".join(errors)
        )

    canaries = dict(getattr(config, "required_feature_canaries", {}) or {})
    expected = tuple(
        sorted(
            (
                feature,
                _EXPECTED_PHASE.get(feature, "update"),
                int(canaries.get(feature, 1)),
            )
            for feature in effective
        )
    )
    runtime_payload = {
        **dict(runtime or {}),
        "agentic": bool(getattr(config, "agentic", False)),
        "distributed": bool(getattr(config, "distributed", False)),
    }
    return FeatureManifest(
        schema_version=FEATURE_MANIFEST_VERSION,
        production_profile=profile,
        requested_features=tuple(sorted(requested)),
        effective_features=tuple(sorted(effective)),
        disabled_features=tuple(sorted(set(_KNOWN_FEATURES) - requested)),
        provisional_features=tuple(
            sorted(getattr(config, "provisional_features", ()) or ())
        ),
        expected_liveness=expected,
        config_digest=_sha256(_config_payload(config)),
        task_set_digest=task_set_digest,
        runtime=tuple(sorted(runtime_payload.items())),
    )


def validate_grpo_startup(
    config: Any,
    tasks: Optional[Sequence[str]],
    *,
    runtime: Optional[Mapping[str, Any]] = None,
    explicit_heldout_ids: Sequence[str] = (),
) -> FeatureManifest:
    """Validate strict startup, including registered/held-out task integrity."""

    if tasks is None or not list(tasks):
        raise FeatureConfigurationError(
            "strict GRPO startup requires a non-empty explicit task list"
        )
    try:
        from kore.policy.curriculum import registered_task_set_digest

        digest = registered_task_set_digest(
            list(tasks), explicit_heldout_ids=explicit_heldout_ids
        )
    except ValueError as exc:
        raise FeatureConfigurationError(str(exc)) from exc
    return resolve_grpo_features(config, task_set_digest=digest, runtime=runtime)


class FeatureRuntime:
    """Per-process liveness tracker backed by ``BudgetLedgerV1`` counters."""

    def __init__(self, manifest: FeatureManifest, ledger: BudgetLedgerV1) -> None:
        if manifest.requested_features != manifest.effective_features:
            raise FeatureConfigurationError(
                "cannot initialize liveness for an inexact feature manifest"
            )
        self.manifest = manifest
        self.ledger = ledger
        self._asserted_phases: set[str] = set()

    def invoked(self, feature: str, count: int = 1) -> None:
        if feature not in self.manifest.effective_features:
            raise FeatureLivenessError(
                f"disabled feature was invoked: {feature}"
            )
        self.ledger.record_feature(feature, count)

    def assert_phase(self, phase: str) -> None:
        if phase not in _PHASE_ORDER:
            raise FeatureLivenessError(f"unknown liveness phase: {phase}")
        if phase in self._asserted_phases:
            return
        missing: list[str] = []
        for feature, expected_phase, minimum in self.manifest.expected_liveness:
            if _PHASE_ORDER[expected_phase] <= _PHASE_ORDER[phase]:
                actual = self.ledger.feature_count(feature)
                if actual < minimum:
                    missing.append(f"{feature}={actual}<{minimum}")
        if missing:
            raise FeatureLivenessError(
                f"enabled feature liveness failed at first {phase}: "
                + ", ".join(missing)
            )
        self._asserted_phases.add(phase)


def initialize_grpo_foundations(
    config: Any,
    tasks: Sequence[str],
    *,
    runtime: Optional[Mapping[str, Any]] = None,
) -> FeatureRuntime:
    manifest = validate_grpo_startup(config, tasks, runtime=runtime)
    ledger = BudgetLedgerV1(
        limits=BudgetLimitsV1.from_mapping(getattr(config, "budget_limits", None))
    )
    return FeatureRuntime(manifest, ledger)


def build_runtime_env(
    config: Any, environ: Optional[Mapping[str, str]] = None
) -> dict[str, str]:
    """Return an environment with all stale feature bridges removed."""

    rendered = dict(os.environ if environ is None else environ)
    for key in _MANAGED_FEATURE_ENV:
        rendered.pop(key, None)
    rendered["KORE_REWARD_MODE"] = str(
        getattr(config, "reward_mode", "speedup")
    )
    rendered["KORE_REWARD_PHASE"] = str(getattr(config, "reward_phase", "all"))
    if bool(getattr(config, "roofline_gate", False)):
        rendered["KORE_ROOFLINE_GATE"] = "1"
        rendered["KORE_ROOFLINE_TOL"] = str(getattr(config, "roofline_tol", 0.25))
    if bool(getattr(config, "physics_live_counters", False)):
        rendered["KORE_PHYSICS_LIVE_COUNTERS"] = "1"
    if float(getattr(config, "physics_shaping_weight", 0.0) or 0.0) > 0.0:
        rendered["KORE_PHYSICS_SHAPING"] = "1"
    if bool(getattr(config, "coevolve_evolve_grammar", False)):
        rendered["KORE_MINTER_EVOLVE_GRAMMAR"] = "1"
    if bool(getattr(config, "transform_discover", False)):
        rendered["KORE_TRANSFORM_DISCOVER"] = "1"
    if bool(getattr(config, "adversarial_coevolve", False)):
        rendered["KORE_ADVERSARIAL_COEVOLVE"] = "1"
    return rendered


def stale_feature_env(
    config: Any, environ: Optional[Mapping[str, str]] = None
) -> dict[str, tuple[Optional[str], Optional[str]]]:
    current = dict(os.environ if environ is None else environ)
    desired = build_runtime_env(config, current)
    stale: dict[str, tuple[Optional[str], Optional[str]]] = {}
    for key in _MANAGED_FEATURE_ENV:
        before, after = current.get(key), desired.get(key)
        if before != after:
            stale[key] = (before, after)
    return stale


def apply_runtime_env(
    config: Any, environ: Optional[MutableMapping[str, str]] = None
) -> dict[str, tuple[Optional[str], Optional[str]]]:
    """Clear stale feature flags, install the resolved bridges, and report changes."""

    target = os.environ if environ is None else environ
    changes = stale_feature_env(config, target)
    rendered = build_runtime_env(config, target)
    for key in _MANAGED_FEATURE_ENV:
        target.pop(key, None)
    for key in _MANAGED_FEATURE_ENV:
        if key in rendered:
            target[key] = rendered[key]
    # Fail closed if a mutable mapping refused an update.
    residual = stale_feature_env(config, target)
    if residual:
        raise FeatureConfigurationError(
            "stale KORE feature environment survived clearing: "
            + ", ".join(sorted(residual))
        )
    return changes


def _atomic_json_write(
    path: str | os.PathLike[str], payload: Mapping[str, Any]
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = _canonical_json(payload, pretty=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return target


def emit_feature_manifest(
    manifest: FeatureManifest, output_dir: str | os.PathLike[str]
) -> Path:
    return manifest.write_json(Path(output_dir) / "feature_manifest.json")
