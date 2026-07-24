"""White-box evidence and finite-potential tests."""

from __future__ import annotations

import math

import pytest

from kore.analysis.roofline import estimate_work, make_physical_model
from kore.reward import shaping
from kore.reward import whitebox
from kore.reward.reward import Observation
from kore.reward.shaping import FamilyShapingEvidence


MODEL = make_physical_model("mi350x")


class Shape:
    name = "primary"
    dims = {"M": 4096, "N": 4096}


class Task:
    task_id = "rmsnorm_x"
    operation = "rmsnorm"
    dtype = "bf16"
    shapes = [Shape()]

    def shape(self, name):
        return Shape() if name == "primary" else None


def _obs():
    return Observation(
        compiled=True,
        validation_passed=True,
        snr_db=40.0,
        wall_ms=1.0,
        wall_by_shape={"primary": 1.0},
        dtype="bf16",
    )


def _evidence(**changes):
    values = dict(
        family="norm",
        report_fingerprint="sha256:report",
        model_fingerprint=MODEL.fingerprint,
        n_points=100,
        n_task_clusters=8,
        normalized_cv_r2=0.8,
        baseline_cv_r2=0.1,
        ci95=(0.5, 0.9),
        adjusted_p=0.01,
        coefficients=(0.5, 0.25, 0.05),
    )
    values.update(changes)
    return FamilyShapingEvidence(**values)


def test_raw_wait_counter_is_unavailable_not_fabricated():
    assert whitebox.stall_frac_from_counters(
        {"SQ_WAIT_INST_ANY": 10, "SQ_INSTS_VALU": 90}) is None
    assert whitebox.stall_frac_from_counters(
        {"MemUnitStalled": 25.0}) == pytest.approx(0.25)
    assert whitebox.stall_frac_from_counters(
        {"MemUnitStalled": 250.0}) is None


def test_resource_occupancy_requires_explicit_model():
    counters = {"vgpr_count": 128, "lds_bytes": 32768, "num_warps": 4}
    assert whitebox.occupancy_from_counters(counters) is None
    assert whitebox.occupancy_from_counters(
        counters, MODEL) == pytest.approx(0.5)


def test_no_evidence_means_no_potential_even_with_counters():
    counters = {"MemUnitStalled": 20.0, "OccupancyPercent": 70.0}
    assert whitebox.phi_potential(
        Task(), _obs(), counters, model=MODEL) is None


def test_potential_requires_both_valid_features():
    evidence = _evidence()
    assert whitebox.phi_potential(
        Task(), _obs(), {"MemUnitStalled": 20.0},
        model=MODEL, evidence=evidence) is None
    assert whitebox.phi_potential(
        Task(), _obs(), {"OccupancyPercent": 70.0},
        model=MODEL, evidence=evidence) is None


def test_matching_passing_evidence_yields_bounded_potential():
    counters = {"MemUnitStalled": 20.0, "OccupancyPercent": 70.0}
    value = whitebox.phi_potential(
        Task(), _obs(), counters, model=MODEL, evidence=_evidence())
    # predicted gap .5*.2 + .25*.3 + .05 = .225
    assert value == pytest.approx(0.775)
    assert 0.0 <= value <= 1.0


def test_mismatched_model_or_family_disables_potential():
    counters = {"MemUnitStalled": 20.0, "OccupancyPercent": 70.0}
    assert whitebox.phi_potential(
        Task(), _obs(), counters, model=MODEL,
        evidence=_evidence(model_fingerprint="sha256:other")) is None
    assert whitebox.phi_potential(
        Task(), _obs(), counters, model=MODEL,
        evidence=_evidence(family="gemm")) is None


def test_failed_held_out_evidence_disables_potential():
    weak = _evidence(ci95=(-0.1, 0.9))
    assert weak.passes() is False
    assert whitebox.phi_potential(
        Task(), _obs(),
        {"MemUnitStalled": 20.0, "OccupancyPercent": 70.0},
        model=MODEL, evidence=weak) is None


def test_structural_score_is_diagnostic_with_explicit_units():
    work = estimate_work("rmsnorm", Shape.dims, "bf16")
    good = whitebox.whitebox_structural_score(
        {"MemUnitStalled": 10.0},
        work=work,
        model=MODEL,
        measured_ms=0.1,
    )
    bad = whitebox.whitebox_structural_score(
        {"MemUnitStalled": 90.0},
        work=work,
        model=MODEL,
        measured_ms=1.0,
    )
    assert good is not None and bad is not None and good > bad


def test_shaping_terms_require_finite_bounded_potentials():
    with pytest.raises(ValueError):
        shaping.shaping_terms([0.1, math.nan], gamma=0.4)
    with pytest.raises(ValueError):
        shaping.shaping_terms([0.1, 1.1], gamma=0.4)
    with pytest.raises(ValueError):
        shaping.shaping_terms([0.1], gamma=1.1)
    with pytest.raises(ValueError):
        shaping.shaped_turn_rewards([1.0], [0.1], gamma=0.4, weight=math.inf)


def test_pbs_discounted_sum_telescopes_when_defined():
    phis = [0.2, 0.5, 0.8]
    gamma = 0.4
    assert shaping.discounted_shaping_sum(phis, gamma) == pytest.approx(-0.2)
    rewards = [0.1, 0.2, 1.0]
    shaped = shaping.shaped_turn_rewards(rewards, phis, gamma, weight=0.3)
    original = sum(gamma ** i * value for i, value in enumerate(rewards))
    changed = sum(gamma ** i * value for i, value in enumerate(shaped))
    assert changed - original == pytest.approx(-0.3 * phis[0])


def test_none_potential_is_zero_boundary():
    assert shaping.shaping_terms([None, 0.5, None], 0.9) == [0.0, 0.0, 0.0]
