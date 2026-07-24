"""Dimension and evidence gates for profile diagnostics."""

from __future__ import annotations

import dataclasses
import math

import pytest

from kore.analysis.roofline import (
    estimate_work,
    make_physical_model,
    mfma_flops,
)
from kore.config import CONFIG
from kore.reward import profile_reward as profile
from kore.reward.reward import Observation, compute_reward


MODEL = make_physical_model("mi350x")
WORK = estimate_work("elementwise", 1_000_000, "bf16")


def test_raw_qcycles_are_not_divided_by_instructions():
    raw = {"SQ_WAIT_INST_ANY": 100, "SQ_INSTS_VALU": 100}
    assert profile.stall_fraction(raw) is None
    assert profile.issue_efficiency(raw) is None
    assert profile.issued_instructions(raw) == 100


def test_derived_percentage_is_validated():
    assert profile.stall_fraction({"MemUnitStalled": 25.0}) == pytest.approx(0.25)
    assert profile.issue_efficiency({"MemUnitStalled": 25.0}) == pytest.approx(0.75)
    assert profile.stall_fraction({"MemUnitStalled": 250.0}) is None
    assert profile.stall_fraction({"MemUnitStalled": math.nan}) is None


def test_mfma_mops_never_enter_instruction_total():
    counters = {
        "SQ_INSTS_VALU": 100,
        "SQ_INSTS_VMEM": 20,
        "SQ_INSTS_VALU_MFMA_MOPS_BF16": 1000,
    }
    assert profile.issued_instructions(counters) == 120
    assert mfma_flops(counters) == 1000 * 512 * 2


def test_profile_score_uses_same_unit_components():
    ref = {
        "MemUnitStalled": 40.0,
        "SQ_INSTS_VMEM": 200,
    }
    better = {
        "MemUnitStalled": 20.0,
        "SQ_INSTS_VMEM": 100,
    }
    worse = {
        "MemUnitStalled": 80.0,
        "SQ_INSTS_VMEM": 400,
    }
    assert profile.profile_efficiency_score(better, ref) == pytest.approx(1.0)
    assert 0.0 <= profile.profile_efficiency_score(worse, ref) < 1.0


def test_profile_score_requires_usable_counter_units():
    assert profile.profile_efficiency_score(
        {"SQ_WAIT_INST_ANY": 1}, {"SQ_WAIT_INST_ANY": 2}) is None


def test_roofline_score_requires_explicit_model():
    t_min = WORK.bytes / MODEL.hbm_bytes_per_s * 1e3
    assert profile.roofline_dense_score(
        {}, work=WORK, measured_ms=t_min) is None
    near = profile.roofline_dense_score(
        {}, work=WORK, model=MODEL, measured_ms=t_min * 1.1)
    far = profile.roofline_dense_score(
        {}, work=WORK, model=MODEL, measured_ms=t_min * 10)
    assert near > far
    assert 0.0 <= far < near <= 1.0


def test_roofline_score_blends_valid_derived_counter():
    t_min = WORK.bytes / MODEL.hbm_bytes_per_s * 1e3
    good = profile.roofline_dense_score(
        {"MemUnitStalled": 10.0},
        work=WORK,
        model=MODEL,
        measured_ms=t_min * 1.1,
    )
    bad = profile.roofline_dense_score(
        {"MemUnitStalled": 90.0},
        work=WORK,
        model=MODEL,
        measured_ms=t_min * 10,
    )
    assert good > bad


def _obs(profile_value, *, evidence=False):
    return Observation(
        compiled=True,
        validation_passed=True,
        snr_by_shape={"s": 99.0},
        wall_by_shape={"s": 1.25},
        baseline_by_shape={"s": 1.0},
        profile_efficiency=profile_value,
        profile_evidence_passed=evidence,
        profile_evidence_fingerprint=("sha256:evidence" if evidence else None),
    )


def test_profile_reward_weight_is_not_sufficient_without_evidence():
    cfg = dataclasses.replace(CONFIG, profile_reward_weight=0.15)
    with_profile = compute_reward(
        _obs(1.0, evidence=False), "x=1", dtype="bf16", cfg=cfg)
    without = compute_reward(
        _obs(None, evidence=False), "x=1", dtype="bf16", cfg=cfg)
    assert with_profile.reward == without.reward
    assert not any(flag.startswith("profile+") for flag in with_profile.flags)


def test_profile_reward_applies_only_with_explicit_evidence_gate():
    cfg = dataclasses.replace(CONFIG, profile_reward_weight=0.15)
    high = compute_reward(
        _obs(1.0, evidence=True), "x=1", dtype="bf16", cfg=cfg)
    low = compute_reward(
        _obs(0.0, evidence=True), "x=1", dtype="bf16", cfg=cfg)
    assert high.reward - low.reward == pytest.approx(0.15)
    assert any(flag.startswith("profile+") for flag in high.flags)


def test_nonfinite_profile_value_never_enters_reward():
    cfg = dataclasses.replace(CONFIG, profile_reward_weight=0.15)
    result = compute_reward(
        _obs(math.nan, evidence=True), "x=1", dtype="bf16", cfg=cfg)
    assert math.isfinite(result.reward)
    assert not any(flag.startswith("profile+") for flag in result.flags)


def test_long_format_parser_preserves_units_without_inventing_efficiency(tmp_path):
    from kore.verifier.parsers.rocprofv3 import parse_rocprofv3_csv

    csv = tmp_path / "counter_collection.csv"
    csv.write_text(
        "Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value\n"
        "1,k,SQ_INSTS_VALU,1000\n"
        "1,k,SQ_WAIT_INST_ANY,500\n"
    )
    (kernel,) = parse_rocprofv3_csv(csv)
    assert kernel.counters["SQ_WAIT_INST_ANY"] == 500
    assert profile.issue_efficiency(kernel.counters) is None
