"""Counter-to-signal compatibility tests."""

from __future__ import annotations

import pytest

from kore.analysis.roofline import make_physical_model
from kore.reward.physics import PhysicsSignal, residual_descent_frac
from kore.reward.reward import Observation
from kore.reward.shaping import FamilyShapingEvidence
from kore.reward import whitebox


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


OBS = Observation(
    compiled=True,
    validation_passed=True,
    snr_db=40.0,
    wall_ms=1.0,
    wall_by_shape={"primary": 1.0},
    dtype="bf16",
)


def _evidence():
    return FamilyShapingEvidence(
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


def test_signal_attaches_only_complete_validated_features():
    signal = whitebox.physics_signal_from_counters(
        Task(),
        OBS,
        {"MemUnitStalled": 25.0, "OccupancyPercent": 80.0},
        model=MODEL,
    )
    assert isinstance(signal, PhysicsSignal)
    assert signal.stall_frac == pytest.approx(0.25)
    assert signal.occupancy == pytest.approx(0.8)
    assert signal.model_fingerprint == MODEL.fingerprint

    incomplete = whitebox.physics_signal_from_counters(
        Task(), OBS, {"MemUnitStalled": 25.0}, model=MODEL)
    assert incomplete.stall_frac is None
    assert incomplete.occupancy is None


def test_whitebox_attainment_distinguishes_diagnostic_and_evidence_paths():
    counters = {"MemUnitStalled": 25.0, "OccupancyPercent": 80.0}
    eta, used = whitebox.whitebox_attainment(
        Task(), OBS, counters, model=MODEL)
    assert eta is not None and used is False
    shaped, used = whitebox.whitebox_attainment(
        Task(), OBS, counters, model=MODEL, evidence=_evidence())
    assert shaped is not None and used is True
    assert 0.0 <= shaped <= 1.0


def test_phi_is_unavailable_without_evidence():
    counters = {"MemUnitStalled": 25.0, "OccupancyPercent": 80.0}
    assert whitebox.phi_potential(
        Task(), OBS, counters, model=MODEL) is None
    assert whitebox.phi_potential(
        Task(), OBS, counters, model=MODEL, evidence=_evidence()) is not None


def test_counter_ranges_are_rejected_not_clamped():
    assert whitebox.stall_frac_from_counters(
        {"MemUnitStalled": 101.0}) is None
    assert whitebox.occupancy_from_counters(
        {"OccupancyPercent": -1.0}, MODEL) is None


def test_resource_occupancy_is_model_specific():
    counters = {"vgpr_count": 128, "lds_bytes": 32768, "num_warps": 4}
    cdna4 = whitebox.occupancy_from_counters(counters, MODEL)
    cdna3 = whitebox.occupancy_from_counters(
        counters, make_physical_model("mi300x"))
    assert cdna4 == pytest.approx(0.5)
    assert cdna3 == pytest.approx(0.25)


def test_unmodelable_operation_has_no_signal():
    class Unknown(Task):
        operation = "flash_attn_decode"

    assert whitebox.physics_signal_from_counters(
        Unknown(),
        OBS,
        {"MemUnitStalled": 25.0, "OccupancyPercent": 80.0},
        model=MODEL,
    ) is None
