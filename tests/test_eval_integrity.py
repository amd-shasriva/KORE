"""Integrity tests for the ways an eval/physics claim could be inflated.

Each test below pins one audited hole shut:

1. ``KORE_PEAK_*`` was a documented but dead reproduce path (a silent no-op), so
   calibrated peaks must instead round-trip through a ``kore.runtime-calibration.v1``
   document, apply, and fail closed on an unfingerprinted or mismatched pin.
2. gfx950 spans two boards with an 8.7% bf16 peak difference, so the SKU must be
   observed/explicit and a contradiction must raise instead of silently selecting
   the neighbour.
3. A saturated arm must not produce a zero-width confidence interval, and a
   zero-variance bootstrap must not manufacture a superiority p-value.
4. ``fast_p@k`` must score the integrity-GATED speedup, never the raw one.
5. The KernelBench-AMD claim track must apply a real threshold.
6. ``leakage_check`` must fail closed when it cannot check family leakage.
"""

from __future__ import annotations

import json
import math

import pytest

from kore.analysis import roofline as canonical
from kore.analysis import rooflines as legacy
from kore.eval import bakeoff
from kore.eval import kernelbench_amd as kb
from kore.eval.frontier_protocol import (
    BOOTSTRAP_DEGENERATE,
    PairedDatum,
    clopper_pearson_interval,
    hierarchical_paired_bootstrap,
    wilson_interval,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _calibration(sku: str = "mi350x", *, hbm: float = 4.598902e12,
                 bf16: float = 1.272550e15) -> dict:
    """A measured-peak calibration document for ``sku`` (datasheet elsewhere)."""
    peaks = dict(canonical.hardware_spec(sku).compute_flops_per_s)
    peaks["bf16"] = bf16
    return legacy.calibration_document(
        sku,
        hbm_bytes_per_s=hbm,
        compute_flops_per_s=peaks,
        calibration_id=f"{sku}-test-stream-matmul",
        runtime={"rocm": "6.4.0", "torch": "2.7.0", "host": "test"},
    )


def _paired(candidate, comparator, *, runs: int = 2, tasks: int = 4) -> list[PairedDatum]:
    return [
        PairedDatum(
            run_id=f"run-{run}",
            family_id=f"family-{task // 2}",
            task_id=f"task-{task}",
            candidate=float(candidate(task)),
            comparator=float(comparator(task)),
        )
        for run in range(runs)
        for task in range(tasks)
    ]


# =========================================================================== #
# 1. Calibration documents round-trip and actually apply.
# =========================================================================== #
def test_calibration_document_round_trips_and_applies(tmp_path):
    document = _calibration()
    assert document["schema"] == legacy.CALIBRATION_SCHEMA
    # Every key make_physical_model requires is present, so it is applicable.
    assert {"architecture", "sku", "calibration_id", "runtime",
            "hbm_bytes_per_s", "compute_flops_per_s"} <= set(document)

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document))

    peaks = legacy.resolve_peaks("gfx950", sku="mi350x", calibration=path)
    assert peaks["hbm_bytes_per_s"] == pytest.approx(4.598902e12)
    assert peaks["bf16_flops_per_s"] == pytest.approx(1.272550e15)
    # ...and the measured peaks genuinely differ from the datasheet they replace.
    datasheet = legacy.resolve_peaks("gfx950", sku="mi350x")
    assert datasheet["hbm_bytes_per_s"] == pytest.approx(8.0e12)
    assert datasheet["bf16_flops_per_s"] == pytest.approx(2.30e15)
    assert peaks["model_fingerprint"] != datasheet["model_fingerprint"]


def test_calibration_is_fingerprint_pinned_and_fails_closed(tmp_path):
    document = _calibration()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document))

    # The recorded fingerprint reproduces...
    model = legacy.resolve_model(
        sku="mi350x", calibration=path,
        expected_fingerprint=document["model_fingerprint"],
    )
    assert model.fingerprint == document["model_fingerprint"]

    # ...a mismatched pin refuses to apply rather than applying silently.
    with pytest.raises(canonical.ModelError):
        legacy.resolve_model(sku="mi350x", calibration=path,
                             expected_fingerprint="sha256:" + "0" * 64)

    # An unfingerprint-able document (no runtime identity) is rejected outright.
    anonymous = dict(document)
    anonymous.pop("runtime")
    with pytest.raises(canonical.ModelError):
        legacy.resolve_model(sku="mi350x", calibration=anonymous)

    # A calibration measured on another board cannot be attached to this one.
    with pytest.raises(canonical.ModelError):
        legacy.resolve_model(sku="mi355x", calibration=document)


def test_calibrate_peaks_emits_the_supported_apply_contract(tmp_path):
    from kore.analysis import calibrate_peaks

    path = tmp_path / "calibration.json"
    document = _calibration()
    path.write_text(json.dumps(document))

    info = calibrate_peaks.verify(str(path))
    assert info["model_fingerprint"] == document["model_fingerprint"]
    assert info["hbm_bytes_per_s"] == pytest.approx(4.598902e12)

    exports = calibrate_peaks._exports(path, info["model_fingerprint"])
    joined = "\n".join(exports)
    # The apply contract points at the SUPPORTED, fingerprinted env vars; the dead
    # KORE_PEAK_* globals must not reappear as the output contract.
    assert legacy.CALIBRATION_ENV_VAR in joined
    assert legacy.FINGERPRINT_ENV_VAR in joined
    assert not any(name in joined for name in legacy.LEGACY_PEAK_ENV_VARS)

    with pytest.raises(canonical.ModelError):
        calibrate_peaks.verify(str(path), expected_fingerprint="sha256:" + "1" * 64)


def test_legacy_peak_env_is_loud_instead_of_a_silent_noop(monkeypatch):
    monkeypatch.setenv("KORE_PEAK_HBM_BW", "4.599e12")
    monkeypatch.setenv("KORE_PEAK_BF16", "1.273e15")
    with pytest.warns(RuntimeWarning, match="has NO effect"):
        peaks = legacy.resolve_peaks("gfx950", sku="mi350x")
    # Still ignored (they are unfingerprinted), but no longer ignored SILENTLY.
    assert peaks["hbm_bytes_per_s"] == pytest.approx(8.0e12)
    assert set(legacy.legacy_peak_env_overrides()) == {"KORE_PEAK_HBM_BW", "KORE_PEAK_BF16"}


# =========================================================================== #
# 2. SKU identity is explicit/observed, and a contradiction is loud.
# =========================================================================== #
def test_gfx950_sku_is_resolved_explicitly_not_guessed():
    # Explicit wins and is labelled as such.
    assert legacy.resolve_sku("gfx950", "mi355x") == ("mi355x", "explicit")
    assert legacy.resolve_sku("gfx950", "mi350x") == ("mi350x", "explicit")
    # An arch/SKU contradiction raises instead of quietly re-picking.
    with pytest.raises(canonical.ModelError):
        legacy.resolve_sku("gfx942", "mi355x")
    with pytest.raises(canonical.ModelError):
        legacy.resolve_sku("gfx950", "mi300x")
    with pytest.raises(canonical.ModelError):
        legacy.resolve_sku("gfx950", "mi400x")


def test_resolved_sku_matches_the_observed_device_or_the_documented_default():
    sku, source = legacy.resolve_sku("gfx950")
    assert source in {"runtime-probe", "arch-fallback"}
    observed = legacy.observed_sku("gfx950")
    if observed is not None:
        # On real hardware the peaks come from the board that is actually present.
        assert sku == observed
        assert legacy.verify_runtime_sku(sku)["status"] == "verified"
        other = "mi355x" if observed == "mi350x" else "mi350x"
        with pytest.raises(canonical.ModelError):
            legacy.verify_runtime_sku(other)
    else:
        assert sku == legacy.DEFAULT_SKU


def test_gfx950_peak_difference_is_material_to_eta():
    """The two gfx950 boards move eta by ~8%, which is why the SKU must be right."""
    mi350x = canonical.hardware_spec("mi350x")
    mi355x = canonical.hardware_spec("mi355x")
    assert mi350x.architecture == mi355x.architecture == "gfx950"
    ratio = mi355x.compute_flops_per_s["bf16"] / mi350x.compute_flops_per_s["bf16"]
    assert ratio == pytest.approx(2.50 / 2.30, rel=1e-9)

    work = canonical.estimate_work("gemm", {"M": 8192, "N": 8192, "K": 8192}, "bf16")
    on_350 = canonical.evaluate_roofline(work, canonical.make_physical_model("mi350x"))
    on_355 = canonical.evaluate_roofline(work, canonical.make_physical_model("mi355x"))
    assert on_350.bound == on_355.bound == "compute"
    # eta = T_min / T_measured, so the wrong SKU rescales every eta by this factor.
    measured_ms = 10.0
    eta_350 = on_350.t_min_ms / measured_ms
    eta_355 = on_355.t_min_ms / measured_ms
    assert eta_355 / eta_350 == pytest.approx(1.0 / ratio, rel=1e-9)
    assert abs(eta_355 / eta_350 - 1.0) > 0.07


def test_config_rejects_a_physics_sku_that_contradicts_the_target():
    from kore.config import CONFIG, KoreConfig

    spec = canonical.hardware_spec(CONFIG.physics_sku)
    assert spec.architecture == CONFIG.gpu_target
    with pytest.raises(ValueError, match="not a known SKU"):
        KoreConfig(physics_sku="mi999x")
    with pytest.raises(ValueError, match="contradicts"):
        KoreConfig(physics_sku="mi300x", gpu_target="gfx950")
    # gfx942 is a legitimate target with its own SKU, so the check is about
    # agreement rather than about pinning gfx950.
    assert KoreConfig(physics_sku="mi300x", gpu_target="gfx942").physics_sku == "mi300x"
    # A fingerprint pin with no calibration path is legal (it pins the datasheet
    # model); a WRONG pin still fails closed where the model is built.
    KoreConfig(physics_model_fingerprint="sha256:" + "2" * 64,
               physics_calibration_path=None)
    with pytest.raises(canonical.ModelError, match="fingerprint mismatch"):
        canonical.make_physical_model(
            "mi350x", expected_fingerprint="sha256:" + "2" * 64)


# =========================================================================== #
# 3. Saturated arms and degenerate bootstraps.
# =========================================================================== #
def test_saturated_arm_does_not_produce_a_zero_width_interval():
    # Candidate wins every preregistered cell; comparator wins half.
    data = _paired(lambda _task: 1.0, lambda task: float(task % 2 == 0))
    result = hierarchical_paired_bootstrap(
        data, n_boot=500, ci_level=0.95, seed=99, noninferiority_margin=0.1
    )
    assert result.candidate_estimate == 1.0
    assert result.candidate_saturated is True

    # The raw percentile bootstrap DOES collapse - that is the bug being fixed.
    assert result.candidate_ci_bootstrap == (1.0, 1.0)
    # The reported interval must not.
    low, high = result.candidate_ci
    assert high == 1.0
    assert low < 1.0
    assert high - low > 0.0
    # It is the exact interval the cell count supports: (alpha/2) ** (1/n).
    assert low == pytest.approx(0.025 ** (1.0 / result.n))
    # The certified lower bound still separates from the point estimate.
    assert low < result.candidate_estimate
    assert result.interval_method.startswith("union(")


def test_zero_variance_bootstrap_cannot_manufacture_significance():
    # Candidate wins every cell, comparator none: every resample is identical.
    data = _paired(lambda _task: 1.0, lambda _task: 0.0)
    result = hierarchical_paired_bootstrap(
        data, n_boot=1_000, ci_level=0.95, seed=3, superiority_margin=0.0
    )
    assert result.delta_se == 0.0
    assert result.degenerate is True
    assert BOOTSTRAP_DEGENERATE  # named integrity signal exists

    # A degenerate bootstrap cannot be the source of the interval or the p-value.
    assert result.delta_ci_bootstrap == (1.0, 1.0)
    assert result.delta_ci[0] < 1.0
    assert result.p_value_method == "exact-one-sided-sign-test"
    # The exact one-sided paired test on n cells, not the 1/(n_boot+1) floor.
    assert result.p_superiority == pytest.approx(0.5 ** result.n)
    assert result.p_superiority > 1.0 / (result.n_boot + 1.0)


def test_degenerate_bootstrap_scales_with_evidence():
    small = hierarchical_paired_bootstrap(
        _paired(lambda _t: 1.0, lambda _t: 0.0, runs=1, tasks=3),
        n_boot=200, ci_level=0.95, seed=5,
    )
    large = hierarchical_paired_bootstrap(
        _paired(lambda _t: 1.0, lambda _t: 0.0, runs=5, tasks=40),
        n_boot=200, ci_level=0.95, seed=5,
    )
    # Same point estimate, but only the well-powered sweep certifies a positive delta.
    assert small.delta_estimate == large.delta_estimate == 1.0
    assert small.delta_ci[0] < 0.0 < large.delta_ci[0]
    assert large.p_superiority < small.p_superiority


def test_tied_arms_do_not_pass_a_superiority_gate_on_zero_variance():
    data = _paired(lambda task: float(task % 2 == 0), lambda task: float(task % 2 == 0))
    result = hierarchical_paired_bootstrap(
        data, n_boot=500, ci_level=0.95, seed=7, superiority_margin=0.0
    )
    assert result.degenerate is True
    assert result.delta_estimate == 0.0
    assert result.delta_ci[0] < 0.0  # cannot clear a margin of 0
    assert result.p_superiority == 1.0


def test_exact_and_score_intervals_match_published_values():
    # Textbook Clopper-Pearson: 4/8 successes at 95% is [0.1570, 0.8430].
    low, high = clopper_pearson_interval(4, 8, 0.95)
    assert low == pytest.approx(0.1570, abs=5e-4)
    assert high == pytest.approx(0.8430, abs=5e-4)
    # Both boundaries stay inside [0, 1] and stay non-degenerate.
    assert clopper_pearson_interval(0, 8, 0.95)[0] == 0.0
    assert clopper_pearson_interval(0, 8, 0.95)[1] == pytest.approx(1 - 0.025 ** (1 / 8))
    assert clopper_pearson_interval(8, 8, 0.95)[1] == 1.0
    # Wilson is also boundary-safe and, at saturation, is the narrower of the two.
    assert 0.0 < wilson_interval(8, 8, 0.95)[0] < 1.0
    assert wilson_interval(8, 8, 0.95)[0] > clopper_pearson_interval(8, 8, 0.95)[0]
    # The interval tightens monotonically as evidence accumulates.
    assert (clopper_pearson_interval(80, 80, 0.95)[0]
            > clopper_pearson_interval(8, 8, 0.95)[0])


def test_reported_interval_is_never_narrower_than_the_bootstrap():
    data = _paired(
        lambda task: float(task != 3), lambda task: float(task % 2 == 0), runs=3, tasks=6
    )
    result = hierarchical_paired_bootstrap(data, n_boot=800, ci_level=0.95, seed=11)
    for reported, raw in (
        (result.candidate_ci, result.candidate_ci_bootstrap),
        (result.comparator_ci, result.comparator_ci_bootstrap),
        (result.delta_ci, result.delta_ci_bootstrap),
    ):
        assert reported[0] <= raw[0]
        assert reported[1] >= raw[1]


# =========================================================================== #
# 4. fast_p@k scores the integrity-gated speedup.
# =========================================================================== #
def _eval_result(samples: list[dict]) -> dict:
    return {"per_task": [{"task_id": "t1", "trajectory": samples}], "n": 1}


def test_fast_p_at_k_scores_the_gated_speedup_not_the_raw_one():
    # A timing glitch: raw 40x, capped to 10x by the excessive_speedup gate.
    glitch = {"turn": 0, "correct": True, "speedup": 40.0, "speedup_gated": 10.0,
              "flags": ["excessive_speedup"]}
    # A noisy bench: raw 3x, damped to <=1x by the high_variance gate.
    noisy = {"turn": 1, "correct": True, "speedup": 3.0, "speedup_gated": 1.0,
             "flags": ["high_variance"]}
    out = bakeoff.best_of_n_pass_at_k(_eval_result([glitch, noisy]), ks=[1],
                                      ps=[1.0, 2.0, 20.0])
    assert out["speedup_basis"] == "speedup_gated"
    # Only the capped 10x clears 2x; the damped 1.0x clears nothing (1.0 > 1.0 false).
    assert out["fast_p_at_k"]["k=1,p=1"] == pytest.approx(0.5)
    assert out["fast_p_at_k"]["k=1,p=2"] == pytest.approx(0.5)
    # Scoring the RAW 40x would have credited p=20; the gated value must not.
    assert out["fast_p_at_k"]["k=1,p=20"] == 0.0
    # Correctness is untouched by the timing gate.
    assert out["pass_at_k"][1] == 1.0


def test_fast_p_at_k_refuses_an_ungated_trajectory():
    ungated = {"turn": 0, "correct": True, "speedup": 40.0}
    with pytest.raises(ValueError, match="speedup_gated"):
        bakeoff.best_of_n_pass_at_k(_eval_result([ungated]), ks=[1], ps=[1.0])


def test_gated_and_raw_diverge_end_to_end_through_evaluate_policy():
    from kore.reward.reward import Observation

    def policy(task, feedback=None):
        return "def kernel(*args):\n    return compute(*args)\n"

    # A single glitched measurement: 50x worst-shape ratio, flagged excessive.
    glitch = Observation(
        compiled=True, snr_db=90.0, wall_ms=0.02, baseline_ms=1.0,
        wall_by_shape={"s": 0.02}, baseline_by_shape={"s": 1.0},
        snr_by_shape={"s": 90.0}, validation_passed=True, dtype="bf16",
    )
    result = bakeoff.evaluate_policy(policy, ["t1"], budget=1, mode="parallel",
                                     dry_run={"t1": [glitch]})
    sample = result["per_task"][0]["trajectory"][0]
    assert sample["speedup"] > sample["speedup_gated"]  # the divergence is real
    out = bakeoff.best_of_n_pass_at_k(result, ks=[1], ps=[float(sample["speedup_gated"])])
    # Scored at exactly the gated value, the sample is NOT a win; on the raw value
    # it would have been.
    assert out["fast_p_at_k"][f"k=1,p={float(sample['speedup_gated']):g}"] == 0.0


# =========================================================================== #
# 5. The KernelBench-AMD claim track applies a real threshold.
# =========================================================================== #
def _kb_report(fast_1: float, *, n: int = 50, correct_rate: float = 0.9) -> dict:
    return {"n": n, "correct_rate": correct_rate, "fast_1": fast_1,
            "fast_p": {1.0: fast_1, 1.5: fast_1 / 2.0, 2.0: 0.0}}


def test_kernelbench_zero_fast_1_is_not_a_pass():
    gate = kb.kernelbench_claim_gate(_kb_report(0.0), source="full")
    assert gate["passed"] is False
    assert any("fast_1" in reason for reason in gate["reasons"])
    # The bar is recorded in the artifact, not hidden in the caller.
    assert gate["thresholds"]["min_fast_1"] == pytest.approx(kb.DEFAULT_MIN_FAST_1)


def test_kernelbench_gate_thresholds_are_enforced_and_configurable():
    assert kb.kernelbench_claim_gate(_kb_report(0.35), source="full")["passed"] is True
    # Just below the default bar fails; an explicit lower bar passes.
    assert kb.kernelbench_claim_gate(_kb_report(0.19), source="full")["passed"] is False
    assert kb.kernelbench_claim_gate(
        _kb_report(0.19), source="full", min_fast_1=0.10)["passed"] is True
    # A high fast_1 cannot rescue a split that mostly fails to compile.
    assert kb.kernelbench_claim_gate(
        _kb_report(0.35, correct_rate=0.2), source="full")["passed"] is False
    # A too-small split is not a claim, whatever the score.
    assert kb.kernelbench_claim_gate(_kb_report(1.0, n=4), source="full")["passed"] is False
    # Non-finite metrics fail closed.
    broken = _kb_report(0.5)
    broken["fast_p"][1.5] = float("nan")
    assert kb.kernelbench_claim_gate(broken, source="full")["passed"] is False


def test_kernelbench_bundled_smoke_specs_can_never_pass_the_claim_gate():
    assert kb.CLAIMABLE_SOURCES == ("full",)
    gate = kb.kernelbench_claim_gate(_kb_report(1.0), source="bundled-smoke")
    assert gate["passed"] is False
    assert any("not claimable" in reason for reason in gate["reasons"])


def test_run_kernelbench_amd_embeds_its_own_gate():
    from kore.reward.reward import Observation

    def policy(task, feedback=None):
        return "def kernel(*args):\n    return compute(*args)\n"

    def observation(speedup: float) -> Observation:
        return Observation(
            compiled=True, snr_db=90.0, wall_ms=1.0 / speedup, baseline_ms=1.0,
            wall_by_shape={"s": 1.0 / speedup}, baseline_by_shape={"s": 1.0},
            snr_by_shape={"s": 90.0}, validation_passed=True,
        )

    specs = kb.bundled_specs()
    tasks = kb.specs_to_tasks(specs)
    out = kb.run_kernelbench_amd(
        policy, specs, budget=1,
        dry_run={t.task_id: [observation(2.0)] for t in tasks},
        source="bundled-smoke",
    )
    # A perfect score on the 4 bundled fixtures is still not a claim.
    assert out["report"]["fast_1"] == 1.0
    assert out["gate"]["passed"] is False
    assert out["report"]["gate"] is out["gate"]
    assert "claim gate: FAIL" in kb.format_kernelbench_report(out["report"])


# =========================================================================== #
# 6. leakage_check fails closed when it cannot check.
# =========================================================================== #
def _protocol(**kwargs) -> kb.HeldoutProtocol:
    base = {"heldout_families": ["gemm"], "heldout_tasks": ["a", "b"],
            "train_tasks": ["c"], "by_family": {"gemm": ["a", "b"]}}
    base.update(kwargs)
    return kb.HeldoutProtocol(**base)


def test_leakage_check_without_tasks_fails_closed_and_warns():
    protocol = _protocol()
    with pytest.warns(RuntimeWarning, match="without task objects"):
        report = kb.leakage_check(protocol)
    # No task overlap, yet ok is False: family leakage was NOT checked.
    assert report["task_overlap"] == []
    assert report["family_check"] == "unverifiable"
    assert report["ok"] is False


def test_assert_no_leakage_requires_tasks():
    with pytest.raises(AssertionError, match="requires the task objects"):
        kb.assert_no_leakage(_protocol())


def test_assert_no_leakage_catches_a_family_leak_with_tasks():
    from kore.tasks import registry as reg

    all_tasks = reg.all_tasks()
    proto = kb.propose_heldout_protocol(all_tasks)
    kb.assert_no_leakage(proto, all_tasks)  # the honest split passes

    # Move one held-out task into train: the family now straddles the boundary,
    # which is exactly the leak the task-less path used to wave through.
    leaked = proto.heldout_tasks[0]
    tampered = kb.HeldoutProtocol(
        heldout_families=list(proto.heldout_families),
        heldout_tasks=[t for t in proto.heldout_tasks if t != leaked],
        train_tasks=sorted(proto.train_tasks + [leaked]),
        by_family={
            family: [t for t in ids if t != leaked]
            for family, ids in proto.by_family.items()
        },
    )
    report = kb.leakage_check(tampered, all_tasks)
    assert report["ok"] is False
    assert report["family_overlap"]
    with pytest.raises(AssertionError, match="FAMILY leakage"):
        kb.assert_no_leakage(tampered, all_tasks)


def test_leakage_check_flags_unknown_tasks_and_declared_map_drift():
    from kore.tasks import registry as reg

    all_tasks = reg.all_tasks()
    proto = kb.propose_heldout_protocol(all_tasks)
    ghost = kb.HeldoutProtocol(
        heldout_families=list(proto.heldout_families),
        heldout_tasks=sorted(proto.heldout_tasks + ["not_a_real_task"]),
        train_tasks=list(proto.train_tasks),
        by_family=dict(proto.by_family),
    )
    report = kb.leakage_check(ghost, all_tasks)
    assert report["ok"] is False
    assert "not_a_real_task" in report["unresolved_tasks"]
    assert "not_a_real_task" in report["declared_family_mismatch"]


# =========================================================================== #
# 7. The publication figures read the current keys and the primary statistics.
# =========================================================================== #
def _p0_report() -> dict:
    """A tiny report in the CURRENT p0_sol schema."""
    measures = []
    for index in range(6):
        # Candidate time RISES along the collection order while eta FALLS, so an
        # eta sort would reorder every trajectory (which is what check (c) forbids).
        candidate = 1.0 + 0.1 * index
        t_min = 0.5 - 0.05 * index
        measures.append({
            "task_id": f"task-{index % 3}",
            "label": "seed@primary",
            "shape_id": "primary",
            "correct": True,
            "cand_ms": candidate,
            "vendor_ms": 0.9 * candidate,
            "t_min_ms": t_min,
            "eta": t_min / candidate,
            "speedup": 0.9,
            "residual_ms": candidate - t_min,
            "stall_frac": 0.2 + 0.01 * index,
            "occupancy": 0.5 + 0.01 * index,
            "baseline_type": "aiter_vendor",
        })
    return {
        "arch": "gfx950",
        "model": {"sku": "MI350X", "architecture": "gfx950"},
        "measures": measures,
        "rooflines": [{"task_id": f"task-{i}", "bound": "memory"} for i in range(3)],
        "checks": {
            "a": {"rho": 0.53, "n": 114, "verdict": "FAIL",
                  "rho_ci95_task_bootstrap": [0.13, 0.80],
                  "tcand_only_rho": 0.73, "increment_over_tcand": -0.20,
                  "increment_ci95_task_bootstrap": [-0.45, 0.02]},
            "b": {"r2": 0.9783, "n": 132, "verdict": "FAIL",
                  "raw_in_sample": {
                      "named_r2": 0.9783, "tcand_only_r2": 0.9971,
                      "denominator_preserving_null": {"null_median": 0.9813,
                                                      "p_value": 0.751}},
                  "normalized_primary": {
                      "task_cluster_cv_r2": -0.4582,
                      "ci95_task_bootstrap": [-1.2528, 0.6218],
                      "tcand_only_cv_r2": -0.1217,
                      "intercept_only_cv_r2": -0.2402,
                      "coefficients": [-9.575, 0.381, 0.582]}},
            "c": {"frac": 0.5, "in_valley_pairs": 38, "tasks": 42, "verdict": "FAIL",
                  "ci95_task_bootstrap": [0.3548, 0.6364]},
        },
    }


def test_plots_read_the_current_check_keys_for_their_annotations():
    from kore.analysis import plots

    report = _p0_report()
    checks = report["checks"]
    # None of the three checks carries the legacy `ci95` key any more, so a figure
    # that reads it annotates nothing.
    assert "ci95" not in checks["a"] and "ci95" not in checks["b"]
    assert plots._interval(checks["a"].get("ci95")) == "  95%CI unavailable"
    # The current keys resolve.
    assert "95%CI[0.130,0.800]" in plots._interval(
        checks["a"]["rho_ci95_task_bootstrap"])
    assert "95%CI[-1.253,0.622]" in plots._interval(
        checks["b"]["normalized_primary"]["ci95_task_bootstrap"])
    assert "95%CI[0.355,0.636]" in plots._interval(checks["c"]["ci95_task_bootstrap"])


def test_plots_render_the_preregistered_primaries(tmp_path):
    from kore.analysis import plots

    report = _p0_report()
    plots.fig_eta_vs_speedup(report, tmp_path)
    plots.fig_residual_fit(report, tmp_path)
    plots.fig_monotone_valley(report, tmp_path)
    plots.fig_roofline_eta(report, tmp_path)
    plots.fig_correct_but_slow(report, tmp_path)
    for name in ("fig1_roofline_eta", "fig2_eta_vs_speedup", "fig3_residual_fit",
                 "fig4_monotone_valley", "fig5_correct_but_slow"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0

    # check (b): the plotted target is the normalized gap the primary uses, and the
    # point cloud is the same admissible sample the statistic scored.
    rows = plots._counter_rows(report)
    assert len(rows) == len(report["measures"])
    for row in rows:
        assert row["gap"] == pytest.approx(row["residual_ms"] / row["cand_ms"])
        assert 0.0 <= row["gap"] <= 1.0

    # check (c): trajectories keep the preregistered collection order.
    trajectories = plots._trajectories(report)
    for values in trajectories.values():
        etas = [m["eta"] for m in values]
        assert etas != sorted(etas)  # NOT re-sorted by the outcome under test
        assert [m["cand_ms"] for m in values] == sorted(m["cand_ms"] for m in values)


def test_plots_skip_super_sol_measures_like_the_statistic_does():
    from kore.analysis import plots

    report = _p0_report()
    report["measures"].append({
        "task_id": "task-9", "label": "seed@primary", "correct": True,
        "cand_ms": 1.0, "t_min_ms": 2.0, "residual_ms": -1.0,
        "eta": 2.0, "speedup": 1.0, "stall_frac": 0.2, "occupancy": 0.5,
    })
    rows = plots._counter_rows(report)
    assert all(row["task_id"] != "task-9" for row in rows)
    assert all(math.isfinite(row["gap"]) for row in rows)
