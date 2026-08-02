"""The shipped GRPO recipe must not request a capability it cannot deliver.

Every optional lever in ``kore.policy.grpo`` fails SAFE: a missing value model
degrades to a heuristic ranker, a missing shaping fingerprint zeroes the shaping
weight, an unreadable co-evolution archive returns ``{}``, an unloadable KL
reference trains with no anchor. That is the right runtime behavior and the
wrong audit behavior -- it means a config can request four capabilities that
never happen, produce a completely healthy-looking 2000-step run, and train a
different recipe than the one written down. ``strict_feature_validation`` would
catch it, but the strict profile bans six of this recipe's defining research
levers outright, so the frontier config cannot adopt it.

These tests are the replacement contract:

* ``kore.policy.capabilities.audit_requested_capabilities`` finds each class of
  inert request, and the shipped config produces ZERO config-provable findings;
* a config that requests an unavailable feature is either refused (strict, or
  ``KORE_GRPO_INERT_FEATURES=error``) or loudly reported (everything else);
* the four specific defects this file was written for stay fixed: no value-model
  consumer without a resolvable model, no nonzero ``physics_shaping_weight``
  without both evidence fields, co-evolution paths rooted where the archives
  actually are, and an explicit ``ref_checkpoint``.

CPU-only and filesystem-light: the corpora and checkpoints live on the cluster,
so nothing here asserts that an artifact EXISTS -- only that the config points
where the archives are and agrees with the launcher that will run it.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
import subprocess
import sys

import pytest

from kore.policy.capabilities import (
    ARTIFACT,
    DECLARED,
    FeatureConfigurationError,
    INERT_FEATURE_POLICY_ENV,
    InertFeatureError,
    audit_requested_capabilities,
    format_capability_audit,
    log_capability_audit,
    validate_grpo_startup,
)
from kore.policy.configs import GRPOConfig
from kore.policy.grpo import grpo_config_from_dict

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHIPPED = "configs/grpo_14b_full.json"
LAUNCHER = "scripts/spur_grpo_1node.sbatch"

#: The corpus this 14B lineage was built from. Measured on the cluster at
#: commit aeada9b with ``kore.openended.opus_baseline.build_opus_scores``:
#: data/b05factory -> 1157 Opus-scored tasks (1460 wins + 1903 group files);
#: data/full14b -> 51 (51 wins records, 132 group files), i.e. 4.4% as much.
#: Neither corpus is checked in, so this pins the ROOT, never its contents.
EXPECTED_COEVOLUTION_ROOT = "data/b05factory"
STALE_COEVOLUTION_ROOT = "data/full14b"

#: midtrain -> sft -> dpo -> grpo. ``ref_checkpoint``'s contract in
#: ``kore/policy/configs.py`` is "the post-SFT multi-capability checkpoint",
#: which is the SFT output, not the DPO output GRPO trains from.
POST_SFT_CHECKPOINT = "runs/sft_14b_frontier"


def _raw() -> dict:
    return json.loads((REPO / SHIPPED).read_text())


def _shipped() -> GRPOConfig:
    return grpo_config_from_dict(_raw())


def _launcher() -> str:
    return (REPO / LAUNCHER).read_text()


def _launcher_default(marker: str) -> str:
    return _launcher().split(marker)[1].split("}")[0]


# --------------------------------------------------------------------------- #
# 1. the shipped config parses and requests nothing it cannot deliver
# --------------------------------------------------------------------------- #
def test_shipped_config_parses_and_validates():
    config = _shipped()
    assert config.validate() is None
    assert config.output_dir


def test_shipped_config_has_no_config_provable_inert_capability():
    """The whole point. A DECLARED finding is provable from the config text.

    ``ARTIFACT`` findings are deliberately excluded: the corpora and the SFT
    checkpoint live on the cluster, so they are absent on a dev box and in CI,
    and reporting them is the runtime audit's job rather than a CI failure.
    """
    findings = audit_requested_capabilities(_shipped(), root=REPO)
    declared = [f for f in findings if f.scope == DECLARED]
    assert not declared, format_capability_audit(declared)


def test_every_comment_key_names_a_real_field_and_is_stripped():
    """``_comment_<field>`` is this repo's in-config documentation convention.

    A comment naming a field that no longer exists is how a justification
    outlives the thing it justified.
    """
    raw = _raw()
    fields = set(GRPOConfig.__dataclass_fields__)
    comments = [key for key in raw if key.startswith("_comment_")]
    assert comments, "the shipped recipe carries no in-config justifications"
    for key in comments:
        assert key[len("_comment_"):] in fields, f"{key} names no GRPOConfig field"
        assert isinstance(raw[key], str) and raw[key].strip()
    # All four stage loaders strip underscore keys before the strict parse.
    assert not any(
        key.startswith("_") for key in vars(grpo_config_from_dict(raw))
        if key in fields
    )


# --------------------------------------------------------------------------- #
# 2. defect 1 - the co-evolution archive root
# --------------------------------------------------------------------------- #
def test_coevolution_paths_are_rooted_where_the_archives_actually_are():
    config = _shipped()
    distill = config.coevolve_distill_path
    scores = config.coevolve_opus_scores_path
    assert distill and scores

    # _build_opus_scores mines dirname(coevolve_distill_path), so the DIRECTORY
    # is the load-bearing part of this value, not the filename.
    root = str(PurePosixPath(distill).parent)
    assert root == EXPECTED_COEVOLUTION_ROOT, (
        f"co-evolution archive root is {root!r}; the regret-vs-Opus curriculum "
        f"mines that directory and only {EXPECTED_COEVOLUTION_ROOT} holds this "
        "lineage's corpus"
    )
    assert root != STALE_COEVOLUTION_ROOT
    # A non-empty opus_scores.json cache is authoritative and skips the scan, so
    # a cache under a different root would silently override a correct archive.
    assert str(PurePosixPath(scores).parent) == root


def test_regret_curriculum_has_its_parents_enabled():
    config = _shipped()
    if config.coevolve_regret_vs_opus:
        assert config.coevolve, "regret-vs-Opus is inert without co-evolution"
        assert config.coevolve_distill_path, (
            "with no distill path _build_opus_scores falls back to a hardcoded "
            "data/full14b root"
        )


def test_audit_catches_a_stale_coevolution_root(tmp_path):
    """The original defect, reconstructed: a root with no wins/ or groups/."""
    stale = GRPOConfig(
        coevolve=True,
        coevolve_regret_vs_opus=True,
        coevolve_distill_path="data/nothing_here/coevolve_wins.jsonl",
    )
    findings = audit_requested_capabilities(stale, root=tmp_path)
    assert any(
        f.feature == "coevolve_regret_vs_opus" and f.scope == ARTIFACT
        for f in findings
    ), format_capability_audit(findings)


def test_audit_catches_a_score_cache_from_another_root(tmp_path):
    split = GRPOConfig(
        coevolve=True,
        coevolve_regret_vs_opus=True,
        coevolve_distill_path="data/b05factory/coevolve_wins.jsonl",
        coevolve_opus_scores_path="data/full14b/opus_scores.json",
    )
    findings = audit_requested_capabilities(split, root=tmp_path)
    assert any(
        f.scope == DECLARED and "authoritative" in f.reason for f in findings
    ), format_capability_audit(findings)


# --------------------------------------------------------------------------- #
# 3. defect 2 - no value-model consumer without a resolvable value model
# --------------------------------------------------------------------------- #
def test_shipped_config_requests_no_value_model_consumer_without_a_model():
    config = _shipped()
    if config.value_prefilter or config.search_value_prior:
        assert config.value_model_path, (
            "value_prefilter / search_value_prior request the TRAINED value "
            "model; load_default_model(None) returns None, so with no path "
            "both silently fall back to a heuristic"
        )
    else:
        assert config.value_model_path is None


def test_disabled_prefilter_pins_the_single_candidate_bench_economy():
    """With no prefilter, every generated candidate is benched.

    Leaving the 8/4 defaults in place while the prefilter is off states a
    verifier budget half the size of the one the run would actually spend.
    """
    config = _shipped()
    if not config.value_prefilter:
        assert config.num_candidates_per_turn == 1
        assert config.value_prefilter_k == 1


def test_agentic_recipes_cannot_request_the_bench_prefilter():
    """Only the serial ``_rollout`` calls ``_prefilter_bench_indices``."""
    config = _shipped()
    if config.agentic:
        assert not config.value_prefilter
    findings = audit_requested_capabilities(
        GRPOConfig(agentic=True, value_prefilter=True, value_model_path=None)
    )
    assert any(
        f.feature == "value_prefilter" and f.scope == DECLARED for f in findings
    ), format_capability_audit(findings)


@pytest.mark.parametrize(
    "overrides",
    [
        {"value_prefilter": True},
        {"use_search": True, "search_value_prior": True},
    ],
)
def test_audit_catches_a_value_consumer_with_no_artifact(overrides, tmp_path):
    config = GRPOConfig(value_model_path=None, **overrides)
    findings = audit_requested_capabilities(config, root=tmp_path)
    assert any("value_model_path" in f.requested_by for f in findings), (
        format_capability_audit(findings)
    )


def test_audit_catches_an_unreadable_value_model(tmp_path):
    config = GRPOConfig(
        use_search=True,
        search_value_prior=True,
        value_model_path="runs/value/value_model.pkl",
    )
    findings = audit_requested_capabilities(config, root=tmp_path)
    assert any(
        f.feature == "search_value_prior" and "not a readable file" in f.reason
        for f in findings
    ), format_capability_audit(findings)


# --------------------------------------------------------------------------- #
# 4. defect 3 - no nonzero shaping weight without BOTH evidence fields
# --------------------------------------------------------------------------- #
def test_shipped_config_arms_physics_shaping_only_with_full_evidence():
    """``_physics_shaping_weight()`` needs the path AND the fingerprint.

    ``docs/P0_RESULTS.md`` records three studies returning ``INTEGRITY_ONLY``
    with no authorized families, so on this evidence the weight must be 0.0 --
    the alternative is manufacturing the document the gate exists to demand.
    """
    config = _shipped()
    if config.physics_shaping_weight > 0.0:
        assert config.physics_shaping_evidence_path
        assert config.physics_shaping_evidence_fingerprint
    else:
        assert config.physics_shaping_weight == 0.0


def test_shipped_config_does_not_collect_counters_it_cannot_use():
    """Counter collection is nested INSIDE the shaping gate, not beside it."""
    config = _shipped()
    armed = bool(
        config.physics_shaping_weight > 0.0
        and config.physics_shaping_evidence_path
        and config.physics_shaping_evidence_fingerprint
    )
    if config.physics_live_counters:
        assert armed or config.reward_mode == "residual"


def test_p0_results_still_authorizes_no_shaping_family():
    """If P0 ever passes, this fails and the shaping decision gets re-made."""
    p0 = (REPO / "docs" / "P0_RESULTS.md").read_text()
    assert "INTEGRITY_ONLY" in p0
    assert "authorized families: none" in p0


@pytest.mark.parametrize(
    "overrides",
    [
        {"physics_shaping_weight": 0.15},
        {"physics_shaping_weight": 0.15,
         "physics_shaping_evidence_path": "data/evidence.json"},
        {"physics_shaping_weight": 0.15,
         "physics_shaping_evidence_fingerprint": "sha256:beef"},
    ],
)
def test_audit_catches_half_armed_physics_shaping(overrides, tmp_path):
    findings = audit_requested_capabilities(GRPOConfig(**overrides), root=tmp_path)
    assert any(
        f.feature == "physics_shaping" and f.scope == DECLARED for f in findings
    ), format_capability_audit(findings)


def test_audit_catches_counters_requested_without_shaping(tmp_path):
    config = GRPOConfig(physics_live_counters=True, physics_shaping_weight=0.0)
    findings = audit_requested_capabilities(config, root=tmp_path)
    assert any(f.feature == "physics_live_counters" for f in findings)


# --------------------------------------------------------------------------- #
# 5. defect 4 - an explicit KL retention anchor
# --------------------------------------------------------------------------- #
def test_ref_checkpoint_is_explicit_and_names_the_post_sft_checkpoint():
    """Null falls back to ``model_id``, which the launcher sets to the DPO output.

    The retention anchor then measures drift from a preference-tuned model
    instead of from the broad post-SFT policy it is meant to preserve.
    """
    config = _shipped()
    assert config.ref_anchor_coef > 0.0, (
        "a zero anchor coefficient would make ref_checkpoint decorative"
    )
    assert config.ref_checkpoint, (
        "ref_checkpoint must be explicit; null silently anchors to model_id"
    )
    assert config.ref_checkpoint == POST_SFT_CHECKPOINT
    # Whatever GRPO trains FROM, the anchor must not be that same checkpoint.
    grpo_from = _launcher_default('FROM_STAGE="${2:-')
    assert config.ref_checkpoint != grpo_from, (
        f"the KL anchor {config.ref_checkpoint!r} is the checkpoint GRPO trains "
        "from, so the retention term measures drift from the run's own start"
    )
    assert config.ref_checkpoint != config.output_dir


def test_audit_catches_an_unresolvable_kl_anchor(tmp_path):
    (tmp_path / "runs").mkdir()
    config = GRPOConfig(ref_anchor_coef=1e-3, ref_checkpoint="runs/never_trained")
    findings = audit_requested_capabilities(config, root=tmp_path)
    assert any(f.feature == "ref_anchor" for f in findings), (
        format_capability_audit(findings)
    )


def test_audit_does_not_mistake_a_hub_repo_id_for_a_missing_directory(tmp_path):
    """``Qwen/Qwen3-14B`` and ``runs/x`` have the same shape; only one is local."""
    config = GRPOConfig(ref_anchor_coef=1e-3, ref_checkpoint=None,
                        model_id="Qwen/Qwen3-14B")
    findings = audit_requested_capabilities(config, root=tmp_path)
    assert not any(f.feature == "ref_anchor" for f in findings)


# --------------------------------------------------------------------------- #
# 6. beyond the four - topology levers with no consumer in THIS runtime
# --------------------------------------------------------------------------- #
def test_shipped_config_enables_no_lever_its_own_topology_disables():
    config = _shipped()
    if config.agentic:
        assert not config.rc_grpo, (
            "reward-control tokens reach only the serial _rollout"
        )
    if config.distributed:
        assert not config.sc_grpo, (
            "the SC-GRPO block exists only in the single-process loop"
        )
    if config.dynamic_sampling:
        assert config.starpo_s
    if config.agentic_transform_tools:
        assert config.agentic
    if config.search_bnb:
        assert config.use_search
    if config.transform_discover:
        assert config.use_search or config.agentic_transform_tools


@pytest.mark.parametrize(
    ("overrides", "feature"),
    [
        ({"agentic": True, "rc_grpo": True}, "rc_grpo"),
        ({"distributed": True, "sc_grpo": True}, "sc_grpo"),
        ({"starpo_s": False, "dynamic_sampling": True}, "dynamic_sampling"),
        ({"agentic": False, "agentic_transform_tools": True},
         "agentic_transform_tools"),
        ({"use_search": False, "search_bnb": True}, "search_bnb"),
        ({"use_search": False, "transform_discover": True}, "transform_discover"),
        ({"coevolve": False, "coevolve_mint": True}, "coevolve_mint"),
        ({"coevolve_mint": False, "coevolve_evolve_grammar": True},
         "coevolve_evolve_grammar"),
        ({"coevolve": False, "adversarial_coevolve": True}, "adversarial_coevolve"),
    ],
)
def test_audit_catches_each_disabled_parent(overrides, feature, tmp_path):
    findings = audit_requested_capabilities(GRPOConfig(**overrides), root=tmp_path)
    assert any(
        f.feature == feature and f.scope == DECLARED for f in findings
    ), format_capability_audit(findings)


# --------------------------------------------------------------------------- #
# 7. an inert request is refused or loudly reported - never silent
# --------------------------------------------------------------------------- #
def _inert() -> GRPOConfig:
    """A config requesting a capability nothing can deliver."""
    return GRPOConfig(physics_shaping_weight=0.15)


def test_inert_request_is_loudly_reported(capsys, tmp_path):
    findings = log_capability_audit(_inert(), root=tmp_path, environ={})
    assert findings
    captured = capsys.readouterr()
    assert "INERT" in captured.err
    assert "physics_shaping" in captured.err
    # The remedy has to be in the message; a warning nobody can act on is noise.
    assert "FIX:" in captured.err


def test_inert_request_is_refused_when_the_policy_says_error(tmp_path):
    with pytest.raises(InertFeatureError, match="physics_shaping"):
        log_capability_audit(
            _inert(), root=tmp_path,
            environ={INERT_FEATURE_POLICY_ENV: "error"},
        )


def test_inert_request_is_refused_by_the_strict_profile():
    """The other lane: a strict config raises rather than reporting."""
    strict = GRPOConfig(
        production_profile="test_strict_v1",
        strict_feature_validation=True,
        num_candidates_per_turn=1,
        value_prefilter_k=1,
        physics_live_counters=True,
        physics_shaping_weight=0.0,
    )
    with pytest.raises(FeatureConfigurationError, match="physics_live_counters"):
        validate_grpo_startup(strict, ["gemm_bf16", "rmsnorm_aiter"])


def test_clean_config_reports_clean_and_never_raises(capsys, tmp_path):
    findings = log_capability_audit(
        GRPOConfig(ref_anchor_coef=0.0), root=tmp_path,
        environ={INERT_FEATURE_POLICY_ENV: "error"},
    )
    assert findings == ()
    assert "OK" in capsys.readouterr().err


def test_audit_document_is_written_next_to_the_feature_manifest(tmp_path):
    log_capability_audit(_inert(), root=tmp_path, output_dir=tmp_path, environ={})
    document = json.loads((tmp_path / "capability_audit.json").read_text())
    assert document["schema_version"] == "CapabilityAuditV1"
    assert [entry["feature"] for entry in document["inert"]] == ["physics_shaping"]
    assert "physics_shaping" in document["requested_features"]


def test_audit_is_deterministic_and_total():
    """It runs on every launch, so it must never raise and never reorder."""
    config = _shipped()
    first = audit_requested_capabilities(config, root=REPO)
    second = audit_requested_capabilities(config, root=REPO)
    assert first == second
    assert list(first) == sorted(first, key=lambda f: f.sort_key)
    # A config full of nonsense paths must still produce findings, not an error.
    assert audit_requested_capabilities(
        GRPOConfig(value_model_path="\0/bad", coevolve_distill_path="\0/bad"),
        root=REPO,
    ) is not None


def test_train_grpo_runs_the_audit_before_the_backend(monkeypatch, tmp_path):
    """Wiring test: the audit must fire on the real entry point, not just here."""
    from kore.policy import grpo

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv(INERT_FEATURE_POLICY_ENV, "error")
    monkeypatch.setattr(grpo, "_train_grpo_inprocess", lambda config, tasks: "out")
    config = GRPOConfig(output_dir=str(tmp_path), physics_shaping_weight=0.15)
    with pytest.raises(InertFeatureError):
        grpo.train_grpo(config, tasks=["gemm_bf16"])
    assert (tmp_path / "capability_audit.json").is_file()

    monkeypatch.setenv(INERT_FEATURE_POLICY_ENV, "warn")
    assert grpo.train_grpo(config, tasks=["gemm_bf16"]) == "out"


# --------------------------------------------------------------------------- #
# 8. the launcher and the config must agree
# --------------------------------------------------------------------------- #
def test_launcher_data_root_default_matches_the_config_archive_root():
    """The rewrite is an override facility now, not a correction.

    If these drift apart again, a default launch silently re-roots the
    curriculum -- the exact failure this file exists to prevent.
    """
    config = _shipped()
    default = _launcher_default('DATA_ROOT="${4:-')
    assert default == EXPECTED_COEVOLUTION_ROOT
    assert default == str(PurePosixPath(config.coevolve_distill_path).parent)
    assert default == str(PurePosixPath(config.coevolve_opus_scores_path).parent)


def test_launcher_is_fail_closed_on_inert_capabilities():
    source = _launcher()
    assert f'export {INERT_FEATURE_POLICY_ENV}="${{{INERT_FEATURE_POLICY_ENV}:-error}}"' \
        in source, "the production launcher must fail closed on an inert feature"


def test_launcher_trains_from_dpo_and_anchors_to_sft():
    """The two checkpoints have different jobs and must not be the same one."""
    config = _shipped()
    from_stage = _launcher_default('FROM_STAGE="${2:-')
    out_dir = _launcher_default('OUT_DIR="${3:-')
    assert from_stage == "runs/dpo_14b_frontier"
    assert out_dir == "runs/grpo_14b_frontier"
    assert config.ref_checkpoint == POST_SFT_CHECKPOINT
    assert len({from_stage, out_dir, config.ref_checkpoint}) == 3


def test_resolver_can_still_retarget_the_archive_root(tmp_path):
    """The rewrite is now a no-op on the default, so prove it still works.

    ``tests/test_spur_stage_launchers.py`` asserts the resolved values, which
    the corrected config already satisfies; this drives the mechanism with a
    root the config does NOT name.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import spur_resolve_launch_config as resolver

    checkpoint = tmp_path / "runs" / "dpo_out"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 8)

    resolved, changes = resolver.resolve(
        "grpo", _raw(), from_stage=str(checkpoint),
        output_dir="runs/grpo_out", data_root="data/b05factory_synced",
        repo_root=tmp_path)

    assert resolved["coevolve_distill_path"] == \
        "data/b05factory_synced/coevolve_wins.jsonl"
    assert resolved["coevolve_opus_scores_path"] == \
        "data/b05factory_synced/opus_scores.json"
    assert sum("coevolve" in change for change in changes) == 2

    # And the default launch changes neither co-evolution key.
    _, unchanged = resolver.resolve(
        "grpo", _raw(), from_stage=str(checkpoint),
        output_dir="runs/grpo_out", data_root=EXPECTED_COEVOLUTION_ROOT,
        repo_root=tmp_path)
    assert not any("coevolve" in change for change in unchanged)


@pytest.mark.shell
def test_launcher_is_syntactically_valid_bash():
    result = subprocess.run(["bash", "-n", str(REPO / LAUNCHER)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":  # pragma: no cover - convenience for ad-hoc runs
    sys.exit(pytest.main([__file__, "-v"]))
