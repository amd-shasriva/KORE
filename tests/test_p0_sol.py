"""CPU-only tests for the P0 roofline/SOL analysis (no GPU, no torch)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kore.analysis import p0_sol as P
from kore.analysis import rooflines as R


# ---------------- roofline model ---------------- #
def test_flops_bytes_gemm():
    flops, by = R.flops_bytes("gemm", {"M": 4096, "N": 4096, "K": 4096}, "bf16")
    assert flops == 2 * 4096 ** 3
    assert by == 3 * 4096 ** 2 * 2  # A+B+C, bf16=2B


def test_flops_bytes_gemm_fp8():
    flops, by = R.flops_bytes("gemm_fp8", {"M": 4096, "N": 4096, "K": 4096}, "fp8_e4m3fnuz")
    assert flops == 2 * 4096 ** 3
    assert by == (4096 ** 2 + 4096 ** 2) * 1 + 4096 ** 2 * 2  # fp8 in, bf16 out


def test_flops_bytes_unmodeled_returns_none_without_dims():
    # ops with no usable dims are unmodelable; ops WITH dims now get a generic
    # memory-bound elementwise lower bound (see test_rooflines).
    assert R.flops_bytes("totally_unknown_op", {}, "bf16") is None


def test_roofline_gemm_compute_bound_gfx950():
    peaks = R.resolve_peaks("gfx950")
    rf = R.roofline("gemm_bf16", "gemm", "bf16", "M=4096,N=4096,K=4096",
                    {"M": 4096, "N": 4096, "K": 4096}, peaks, "gfx950")
    assert rf.bound == "compute"
    assert abs(rf.arithmetic_intensity - 1365.3) < 1.0
    # 2*4096^3 / 2.5e15 s -> ~0.055 ms
    assert 0.04 < rf.t_min_ms < 0.07


def test_roofline_rmsnorm_memory_bound():
    peaks = R.resolve_peaks("gfx950")
    rf = R.roofline("rmsnorm", "rmsnorm", "bf16", "M=4096,N=4096",
                    {"M": 4096, "N": 4096}, peaks, "gfx950")
    assert rf.bound == "memory"


def test_resolve_peaks_env_override(monkeypatch):
    monkeypatch.setenv("KORE_PEAK_HBM_BW", "1.0e13")
    monkeypatch.setenv("KORE_PEAK_BF16", "3.3e15")
    p = R.resolve_peaks("gfx950")
    assert p["hbm_bytes_per_s"] == 1.0e13
    assert p["bf16_flops_per_s"] == 3.3e15


def test_default_arch_is_gfx950():
    assert R.DEFAULT_ARCH == "gfx950"


# ---------------- stats ---------------- #
def test_spearman_monotonic():
    rho = P.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert rho is not None and rho > 0.99


def test_spearman_anti_monotonic():
    rho = P.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert rho is not None and rho < -0.99


def test_ols_r2_perfect_linear():
    X = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]
    y = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    r2 = P.ols_r2(X, y)
    assert r2 is None or r2 > 0.99  # None only if numpy missing


# ---------------- decomposition ---------------- #
def test_decompose_counters():
    stall, occ = P._decompose({"MemUnitStalled": 40.0, "OccupancyPercent": 75.0})
    assert abs(stall - 0.40) < 1e-6   # MemUnitStalled/100
    assert abs(occ - 0.75) < 1e-6     # OccupancyPercent/100


def test_decompose_empty():
    assert P._decompose({}) == (None, None)


# ---------------- checks + decision ---------------- #
def _mk(eta=None, speedup=None, stall=None, occ=None, resid=None, correct=True, cand=1.0):
    return P.KernelMeasure(task_id="t", label="l", correct=correct, snr_db=40.0,
                           cand_ms=cand, vendor_ms=None, t_min_ms=0.5, eta=eta,
                           speedup=speedup, residual_ms=resid, stall_frac=stall, occupancy=occ)


def test_check_a_pass():
    ms = [_mk(eta=e, speedup=s) for e, s in
          [(0.1, 0.5), (0.2, 0.7), (0.3, 0.9), (0.4, 1.1), (0.5, 1.4)]]
    res = P.check_a(ms)
    assert res["verdict"] == "PASS" and res["rho"] > 0.9


def test_check_a_skip_too_few():
    assert P.check_a([_mk(eta=0.1, speedup=0.5)])["verdict"] == "SKIP"


def test_check_b_skip_without_counters():
    assert P.check_b([_mk(eta=0.1, speedup=0.5)])["verdict"] == "SKIP"


def test_decide_branches():
    P_ = {"verdict": "PASS"}
    W_ = {"verdict": "WEAK"}
    S_ = {"verdict": "SKIP"}
    assert P.decide(P_, P_, P_, False) == "GO"
    assert P.decide(P_, P_, W_, False) == "PARTIAL"
    assert P.decide(P_, W_, W_, False) == "FALLBACK"
    assert P.decide(W_, W_, W_, False) == "PIVOT?"
    assert P.decide(S_, S_, S_, True) == "DRY_RUN"


# ---------------- dry-run end-to-end (CPU, uses registry + replay caches) ------ #
def test_dry_run_smoke():
    peaks = R.resolve_peaks("gfx950")
    rep = P.run(["gemm_bf16", "rmsnorm_aiter"], "gfx950", peaks, warmup=1, iters=1,
                max_kernels=4, device="0", dry_run=True, do_pmc=False,
                replay_dir=P.REPO_ROOT / "runs")
    assert rep["decision"] == "DRY_RUN"
    assert len(rep["rooflines"]) == 2
    # rendering must not crash
    assert "P0 roofline" in P.render(rep)
