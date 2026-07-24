"""CPU-only tests for leakage-controlled P0 validation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from kore.analysis import p0_sol as P
from kore.analysis.residual_transfer import run as residual_run


def _measure(
    *,
    task: str,
    label: str,
    cand: float,
    t_min: float,
    stall: float | None = None,
    occ: float | None = None,
    speedup: float | None = None,
    family: str | None = None,
) -> P.KernelMeasure:
    return P.KernelMeasure(
        task_id=task,
        label=label,
        correct=True,
        snr_db=40.0,
        cand_ms=cand,
        vendor_ms=(speedup * cand if speedup is not None else None),
        t_min_ms=t_min,
        eta=t_min / cand,
        speedup=speedup,
        residual_ms=cand - t_min,
        stall_frac=stall,
        occupancy=occ,
        family=family,
    )


def test_spearman_basic():
    assert P.spearman([1, 2, 3], [2, 4, 8]) == pytest.approx(1.0)
    assert P.spearman([1, 2, 3], [8, 4, 2]) == pytest.approx(-1.0)


def test_decompose_rejects_malformed_percentages():
    assert P._decompose(
        {"MemUnitStalled": 40.0, "OccupancyPercent": 75.0}) == (0.4, 0.75)
    assert P._decompose(
        {"MemUnitStalled": 140.0, "OccupancyPercent": -1.0}) == (None, None)


def test_check_a_shared_denominator_control_defeats_tautology():
    measures = []
    for task_index in range(6):
        for point in range(6):
            cand = 1.0 + task_index + point / 10
            # Both eta and speedup are exactly proportional to 1/T_candidate.
            measures.append(_measure(
                task=f"task{task_index}",
                label=str(point),
                cand=cand,
                t_min=0.5,
                speedup=2.0 / cand,
            ))
    result = P.check_a_rigorous(
        measures, permutations=100, bootstrap=100, seed=7)
    assert result["rho"] > 0.99
    assert result["tcand_only_rho"] > 0.99
    assert result["increment_over_tcand"] == pytest.approx(0.0)
    assert result["verdict"] == "FAIL"


def test_check_b_recovers_true_normalized_held_task_signal():
    rng = random.Random(4)
    measures = []
    for task_index in range(8):
        for point in range(12):
            stall = 0.05 + 0.8 * rng.random()
            occ_deficit = 0.05 + 0.8 * rng.random()
            gap = 0.10 + 0.45 * stall + 0.25 * occ_deficit
            cand = 0.5 + 2.0 * rng.random()
            t_min = cand * (1.0 - gap)
            measures.append(_measure(
                task=f"norm_task_{task_index}",
                label=str(point),
                cand=cand,
                t_min=t_min,
                stall=stall,
                occ=1.0 - occ_deficit,
                family="norm",
            ))
    result = P.check_b(
        measures, permutations=100, bootstrap=100, seed=11)
    primary = result["normalized_primary"]
    assert primary["task_cluster_cv_r2"] > 0.95
    assert primary["increment_over_tcand"] > 0.5
    assert primary["ci95_task_bootstrap"][0] > 0.9
    assert result["_eligible"] is True


def test_stored_report_negative_controls_invalidate_raw_headline():
    source = json.loads((P.REPO_ROOT / "data" / "p0_study_final.json").read_text())
    report = P.reanalyze_report(
        source, permutations=100, bootstrap=100, seed=20260723)
    a, b = report["checks"]["a"], report["checks"]["b"]
    raw = b["raw_in_sample"]
    primary = b["normalized_primary"]
    assert raw["named_r2"] > 0.97
    assert raw["tcand_only_r2"] > raw["named_r2"]
    assert raw["denominator_preserving_null"]["null_median"] > raw["named_r2"]
    assert primary["task_cluster_cv_r2"] < 0.0
    assert primary["increment_over_tcand"] < 0.0
    assert a["increment_over_tcand"] < 0.0
    assert report["decision"] == "INTEGRITY_ONLY"
    assert report["shaping_evidence"]["families"] == {}
    assert report["model_fingerprint_status"] == "legacy-unfingerprinted"


def test_residual_transfer_is_exact_canonical_wrapper(tmp_path):
    source_path = P.REPO_ROOT / "data" / "p0_study_final.json"
    source = json.loads(source_path.read_text())
    canonical = P.reanalyze_report(
        source, permutations=50, bootstrap=50, seed=20260723)
    wrapped = residual_run(
        source_path, permutations=50, bootstrap=50, seed=20260723)
    assert wrapped["canonical_check"] == canonical["checks"]["b"]
    assert wrapped["analysis_fingerprint"] == canonical["analysis_fingerprint"]


def test_collection_order_is_not_eta_sorted():
    # In collection order the dominant residual rises; sorting by eta would
    # reverse these records and fabricate improvement.
    trajectory = []
    for index, dominant in enumerate([0.1, 0.2, 0.3, 0.4]):
        trajectory.append(_measure(
            task="t",
            label=str(index),
            cand=1.0,
            t_min=0.2 + index * 0.1,
            stall=dominant,
            occ=1.0,
        ))
    result = P.check_c({"t@shape": trajectory}, bootstrap=0)
    assert result["frac"] == 0.0


def test_bh_adjustment_is_monotone_and_conservative():
    adjusted = P._bh_adjust([("a", 0.01), ("b", 0.03), ("c", 0.20)])
    assert adjusted["a"] >= 0.01
    assert adjusted["b"] >= 0.03
    assert adjusted["c"] >= 0.20
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_decision_is_conservative():
    passed = {"verdict": "PASS"}
    failed = {"verdict": "FAIL"}
    skipped = {"verdict": "SKIP"}
    assert P.decide(passed, passed, passed, False) == "GO"
    assert P.decide(passed, passed, failed, False) == "EVIDENCE_PARTIAL"
    assert P.decide(passed, failed, failed, False) == "INTEGRITY_ONLY"
    assert P.decide(skipped, failed, failed, False) == "INSUFFICIENT_DATA"
    assert P.decide(skipped, skipped, skipped, True) == "DRY_RUN"


def test_report_fingerprint_is_deterministic():
    source = json.loads((P.REPO_ROOT / "data" / "p0_study_final.json").read_text())
    first = P.reanalyze_report(source, permutations=20, bootstrap=20, seed=9)
    second = P.reanalyze_report(source, permutations=20, bootstrap=20, seed=9)
    assert first["analysis_fingerprint"] == second["analysis_fingerprint"]
    assert first["evidence_fingerprint"] == second["evidence_fingerprint"]
