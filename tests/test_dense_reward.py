"""GRPO profile shaping requires fingerprinted family evidence."""

from __future__ import annotations

import json
import types

from kore.analysis.roofline import make_physical_model
from kore.policy import grpo
from kore.reward.reward import Observation
from kore.reward.shaping import _document_fingerprint


MODEL = make_physical_model("mi350x")
BUSY = {"MemUnitStalled": 10.0, "MfmaUtil": 80.0}
STALLING = {"MemUnitStalled": 90.0, "MfmaUtil": 10.0}


class StubEnv:
    def __init__(self, counters):
        self.counters = counters
        self.calls = []

    def collect_counters(self, source, shape=None):
        self.calls.append(source)
        return self.counters


def _task():
    shape = types.SimpleNamespace(
        name="primary", dims={"M": 512, "N": 512, "K": 512})
    return types.SimpleNamespace(
        task_id="gemm_stub",
        operation="gemm",
        dtype="bf16",
        shape=lambda name: shape if name == "primary" else None,
        shapes=[shape],
    )


def _obs(wall=0.001, correct=True):
    return Observation(
        compiled=True,
        validation_passed=correct,
        snr_by_shape={"primary": 99.0},
        wall_ms=wall,
        wall_by_shape={"primary": wall},
        dtype="bf16",
    )


def _evidence_file(tmp_path, family="gemm"):
    family_data = {
        "family": family,
        "report_fingerprint": "sha256:report",
        "model_fingerprint": MODEL.fingerprint,
        "n_points": 100,
        "n_task_clusters": 8,
        "normalized_cv_r2": 0.8,
        "baseline_cv_r2": 0.1,
        "ci95": [0.5, 0.9],
        "adjusted_p": 0.01,
        "coefficients": [0.5, 0.25, 0.05],
        "verdict": "PASS",
    }
    document = {"shaping_evidence": {"families": {family: family_data}}}
    fingerprint = _document_fingerprint(document)
    document["evidence_fingerprint"] = fingerprint
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(document))
    return str(path), fingerprint


def _cfg(weight, evidence=None, phase="latency"):
    path, fingerprint = evidence or (None, None)
    return types.SimpleNamespace(
        profile_reward_weight=weight,
        reward_phase=phase,
        physics_sku="mi350x",
        physics_calibration_path=None,
        physics_model_fingerprint=MODEL.fingerprint,
        physics_shaping_evidence_path=path,
        physics_shaping_evidence_fingerprint=fingerprint,
        physics_shaping_weight=weight,
    )


def test_dense_weight_is_zero_without_evidence():
    config = _cfg(0.15)
    assert grpo._dense_profile_weight(config) == 0.0
    assert grpo._physics_shaping_weight(config) == 0.0


def test_dense_bonus_no_evidence_does_not_profile():
    env = StubEnv(BUSY)
    dense, feedback = grpo._dense_profile_bonus(
        env, _task(), "source", _obs(), _cfg(0.15))
    assert dense == 0.0 and feedback == ""
    assert env.calls == []


def test_passing_family_evidence_enables_bounded_bonus(tmp_path):
    evidence = _evidence_file(tmp_path)
    config = _cfg(0.15, evidence)
    env = StubEnv(BUSY)
    dense, feedback = grpo._dense_profile_bonus(
        env, _task(), "source", _obs(), config)
    assert 0.0 < dense <= 0.15
    assert env.calls == ["source"]
    assert MODEL.fingerprint in feedback
    assert "bottleneck=compute-bound" in feedback


def test_evidence_enabled_bonus_tracks_diagnostic_quality(tmp_path):
    config = _cfg(0.15, _evidence_file(tmp_path))
    good, _ = grpo._dense_profile_bonus(
        StubEnv(BUSY), _task(), "s", _obs(0.001), config)
    bad, _ = grpo._dense_profile_bonus(
        StubEnv(STALLING), _task(), "s", _obs(0.01), config)
    assert good > bad >= 0.0


def test_wrong_family_evidence_disables_before_collection(tmp_path):
    config = _cfg(0.15, _evidence_file(tmp_path, family="norm"))
    env = StubEnv(BUSY)
    assert grpo._dense_profile_bonus(
        env, _task(), "s", _obs(), config) == (0.0, "")
    assert env.calls == []


def test_wrong_model_fingerprint_disables(tmp_path):
    path, fingerprint = _evidence_file(tmp_path)
    document = json.loads(open(path).read())
    document["shaping_evidence"]["families"]["gemm"][
        "model_fingerprint"] = "sha256:other"
    document.pop("evidence_fingerprint")
    fingerprint = _document_fingerprint(document)
    document["evidence_fingerprint"] = fingerprint
    open(path, "w").write(json.dumps(document))
    config = _cfg(0.15, (path, fingerprint))
    env = StubEnv(BUSY)
    assert grpo._dense_profile_bonus(
        env, _task(), "s", _obs(), config) == (0.0, "")
    assert env.calls == []


def test_incorrect_and_correctness_phase_short_circuit(tmp_path):
    config = _cfg(0.15, _evidence_file(tmp_path))
    env = StubEnv(BUSY)
    assert grpo._dense_profile_bonus(
        env, _task(), "s", _obs(correct=False), config) == (0.0, "")
    assert env.calls == []
    correctness = _cfg(0.15, _evidence_file(tmp_path), phase="correctness")
    assert grpo._dense_profile_bonus(
        env, _task(), "s", _obs(), correctness) == (0.0, "")
    assert env.calls == []


def test_make_rollout_env_only_disables_internal_profile_with_evidence(
    tmp_path, monkeypatch
):
    import kore.env.kore_env as env_module

    seen = {}

    class FakeEnv:
        def __init__(self, task, config=None, gpu=None):
            seen["config"] = config

    monkeypatch.setattr(env_module, "KoreEnv", FakeEnv)
    grpo._make_rollout_env(
        "T", _cfg(0.15, _evidence_file(tmp_path)), serial=True)
    assert seen["config"] is not None
    assert seen["config"].profile_reward_weight == 0.0


def test_turn_phi_is_disabled_without_evidence():
    assert grpo._turn_phi(_task(), _obs(), config=_cfg(0.15)) is None
