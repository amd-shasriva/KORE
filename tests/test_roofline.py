"""CPU-only tests for the MI300X roofline model + counter-grounded bottleneck.

Pins: the MI300X hardware constants, the roofline bound selection (a large-K GEMM
is compute-bound, elementwise is memory-bound), the op FLOP/byte formulas, the
attained-fraction math (100% exactly on the roofline), and the upgraded
bottleneck classifier (L2 hit-rate / HBM bytes / occupancy).
"""

from __future__ import annotations

import pytest

from kore.analysis import roofline as R
from kore.verifier import pmc


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
def test_mi300x_constants_match_datasheet():
    # gfx942 / CDNA3 bundle (previous gen) - always available in the arch table.
    mi = R.MI300X
    assert mi["hbm_bw_bytes_per_s"] == pytest.approx(5.325e12, rel=1e-3)
    assert mi["peak_flops_bf16"] == pytest.approx(1.3074e15, rel=1e-3)
    assert mi["peak_flops_fp8"] == pytest.approx(2.6149e15, rel=1e-3)
    assert mi["peak_flops_fp32"] == pytest.approx(1.634e14, rel=1e-3)
    assert mi["num_cus"] == 304
    # MI300X (CDNA3) carries its OWN occupancy constants (64 KiB LDS, granularity 16)
    # regardless of the ACTIVE arch -- NOT the gfx950 default (audit R2 pmc).
    assert mi["lds_bytes_per_cu"] == 65536
    assert mi["vgpr_per_simd"] == 512


def test_mi350x_constants_match_datasheet():
    # gfx950 / CDNA4 - the KORE target hardware (default ACTIVE arch).
    mi = R.MI350X
    assert mi["arch"] == "gfx950"
    assert mi["hbm_bw_bytes_per_s"] == pytest.approx(8.0e12, rel=1e-3)   # HBM3E 8 TB/s
    assert mi["peak_flops_bf16"] == pytest.approx(2.30e15, rel=1e-3)
    assert mi["peak_flops_fp8"] == pytest.approx(4.60e15, rel=1e-3)      # OCP-FP8
    assert mi["peak_flops_fp4"] == pytest.approx(9.20e15, rel=1e-3)      # MXFP4
    assert mi["peak_flops_fp6"] == pytest.approx(9.20e15, rel=1e-3)      # MXFP6
    assert mi["peak_flops_fp64"] == pytest.approx(7.21e13, rel=1e-3)     # CDNA4 halves FP64
    assert mi["num_cus"] == 256
    # CDNA4 occupancy limits: 160 KiB LDS/CU (2.5x CDNA3), 512 VGPR/SIMD (unchanged).
    assert mi["lds_bytes_per_cu"] == 163840
    assert mi["vgpr_per_simd"] == 512
    # MI355X (liquid-cooled) variant runs hotter/faster, same CDNA4 occupancy limits.
    assert R.MI355X["peak_flops_bf16"] == pytest.approx(2.50e15, rel=1e-3)
    assert R.MI355X["peak_flops_fp8"] == pytest.approx(5.00e15, rel=1e-3)
    assert R.MI355X["lds_bytes_per_cu"] == 163840


def test_active_arch_is_cdna4_and_consistent():
    # The module-level constants must mirror the ACTIVE arch's table entry, and
    # the KORE node default is CDNA4 (gfx950).
    a = R._ARCH_PEAKS[R.ACTIVE_ARCH]
    assert R.HBM_BW_BYTES_PER_S == a["hbm_bw_bytes_per_s"]
    assert R.PEAK_FLOPS_BF16 == a["peak_flops_bf16"]
    assert R.PEAK_FLOPS_FP8 == a["peak_flops_fp8"]
    assert R.ACTIVE["arch"] == "gfx950"


def test_detect_arch_env_override(monkeypatch):
    monkeypatch.setenv("KORE_ROOFLINE_ARCH", "gfx942")
    assert R._detect_arch() == "gfx942"
    monkeypatch.setenv("KORE_ROOFLINE_ARCH", "mi355x")
    assert R._detect_arch() == "mi355x"
    monkeypatch.setenv("KORE_ROOFLINE_ARCH", "mi350x")
    assert R._detect_arch() == "gfx950"


def test_dtype_bytes_and_peak_selection():
    assert R.dtype_bytes("bf16") == 2 and R.dtype_bytes("fp16") == 2
    assert R.dtype_bytes("fp8_e4m3fnuz") == 1 and R.dtype_bytes("fp32") == 4
    assert R.dtype_bytes("fp4_e2m1") == 0.5 and R.dtype_bytes("mxfp6") == 0.75
    assert R.peak_flops("bf16") == R.PEAK_FLOPS_BF16
    assert R.peak_flops("fp8_e4m3") == R.PEAK_FLOPS_FP8
    assert R.peak_flops("mxfp4") == R.PEAK_FLOPS_FP4
    assert R.peak_flops("fp6_e3m2") == R.PEAK_FLOPS_FP6
    assert R.peak_flops("fp32") == R.PEAK_FLOPS_FP32


# --------------------------------------------------------------------------- #
# roofline bound selection
# --------------------------------------------------------------------------- #
def test_ridge_point():
    r = R.roofline(1.0, 1.0, "bf16")
    assert r["ridge_point"] == pytest.approx(R.PEAK_FLOPS_BF16 / R.HBM_BW_BYTES_PER_S)
    # gfx950/MI350X default: 2.3 PFLOP / 8 TB/s = 287.5 FLOP/byte (MI300X was 245.5).
    a = R._ARCH_PEAKS[R.ACTIVE_ARCH]
    assert r["ridge_point"] == pytest.approx(
        a["peak_flops_bf16"] / a["hbm_bw_bytes_per_s"], rel=1e-2)


def test_large_gemm_is_compute_bound():
    M = N = K = 4096
    flops = 2.0 * M * N * K
    by = (M * K + K * N + M * N) * 2  # bf16
    r = R.roofline(flops, by, "bf16")
    assert r["bound"] == "compute"
    assert r["arithmetic_intensity"] == pytest.approx(flops / by)
    # compute-bound -> attainable peak is the dtype compute ceiling
    assert r["peak_attainable_flops"] == pytest.approx(R.PEAK_FLOPS_BF16)
    assert r["t_min_ms"] == pytest.approx(r["t_compute_ms"])


def test_small_k_gemm_is_memory_bound():
    # same M,N but K tiny -> low arithmetic intensity -> memory-bound
    M = N = 4096
    K = 8
    flops = 2.0 * M * N * K
    by = (M * K + K * N + M * N) * 2
    r = R.roofline(flops, by, "bf16")
    assert r["bound"] == "memory"
    assert r["peak_attainable_flops"] < R.PEAK_FLOPS_BF16


def test_elementwise_is_memory_bound():
    numel = 1_000_000
    flops = float(numel)
    by = 2.0 * numel * 2  # read + write, bf16
    r = R.roofline(flops, by, "bf16")
    assert r["bound"] == "memory"
    assert r["arithmetic_intensity"] == pytest.approx(0.25)
    # memory-bound -> attainable peak = AI * bandwidth (well below compute peak)
    assert r["peak_attainable_flops"] == pytest.approx(0.25 * R.HBM_BW_BYTES_PER_S)
    assert r["t_min_ms"] == pytest.approx(r["t_mem_ms"])


# --------------------------------------------------------------------------- #
# op_flop_bytes formulas
# --------------------------------------------------------------------------- #
def test_op_flop_bytes_gemm_exact():
    flops, by = R.op_flop_bytes("gemm", {"M": 512, "N": 1024, "K": 2048}, "bf16")
    assert flops == 2.0 * 512 * 1024 * 2048
    assert by == (512 * 2048 + 2048 * 1024 + 512 * 1024) * 2


def test_op_flop_bytes_gemm_needs_mnk():
    assert R.op_flop_bytes("gemm", {"M": 512, "N": 1024}, "bf16") is None


def test_op_flop_bytes_batched_gemm():
    flops, by = R.op_flop_bytes("batched_gemm", {"B": 8, "M": 128, "N": 256, "K": 512}, "bf16")
    assert flops == 2.0 * 8 * 128 * 256 * 512
    assert by == 8 * (128 * 512 + 512 * 256 + 128 * 256) * 2


def test_op_flop_bytes_elementwise():
    # int shape == numel; unary default n_tensors=2 (read + write)
    flops, by = R.op_flop_bytes("elementwise", 1000, "bf16")
    assert flops == 1000 and by == 2 * 1000 * 2
    # binary op streams 3 tensors
    flops, by = R.op_flop_bytes("pointwise", {"M": 128, "N": 256}, "bf16", n_tensors=3)
    assert flops == 128 * 256 and by == 3 * 128 * 256 * 2


def test_op_flop_bytes_reduction():
    flops, by = R.op_flop_bytes("row_sum", {"M": 64, "N": 512}, "bf16")
    assert flops == 64 * 512
    assert by == (64 * 512 + 64) * 2   # read all + write one per row


def test_op_flop_bytes_norms_and_softmax():
    assert R.op_flop_bytes("rmsnorm", {"M": 4096, "N": 8192}, "bf16") == (
        4.0 * 4096 * 8192, float((2 * 4096 * 8192 + 8192) * 2))
    assert R.op_flop_bytes("layernorm", {"M": 4096, "N": 8192}, "bf16") == (
        6.0 * 4096 * 8192, float((2 * 4096 * 8192 + 2 * 8192) * 2))
    assert R.op_flop_bytes("softmax", {"M": 4096, "N": 8192}, "bf16") == (
        5.0 * 4096 * 8192, float((2 * 4096 * 8192) * 2))


def test_op_flop_bytes_unmodelable_returns_none():
    assert R.op_flop_bytes("mystery_op", {}, "bf16") is None
    # but anything with usable dims falls back to a memory-bound EW estimate
    flops, by = R.op_flop_bytes("weird_fused_thing", {"X": 10, "Y": 20}, "bf16")
    assert flops == 200 and by == 2 * 200 * 2


# --------------------------------------------------------------------------- #
# attained fraction
# --------------------------------------------------------------------------- #
def test_attained_fraction_is_100pct_on_the_roofline():
    # compute-bound op: at exactly t_min the achieved FLOP/s == the compute peak
    flops = 2.0 * 4096 ** 3
    by = (4096 * 4096 * 3) * 2
    r = R.roofline(flops, by, "bf16")
    assert R.attained_fraction(r["t_min_ms"], flops, by, "bf16") == pytest.approx(100.0)
    # half speed -> half the roofline
    assert R.attained_fraction(2 * r["t_min_ms"], flops, by, "bf16") == pytest.approx(50.0)


def test_attained_fraction_memory_bound_and_super_roofline():
    numel = 1_000_000
    flops = float(numel)
    by = 2.0 * numel * 2
    r = R.roofline(flops, by, "bf16")
    assert R.attained_fraction(r["t_min_ms"], flops, by, "bf16") == pytest.approx(100.0)
    # faster than the HBM lower bound (cache reuse) -> >100%
    assert R.attained_fraction(0.5 * r["t_min_ms"], flops, by, "bf16") > 100.0


def test_attained_fraction_guards_bad_input():
    assert R.attained_fraction(0.0, 100.0, 100.0, "bf16") == 0.0
    assert R.attained_fraction(1.0, 0.0, 100.0, "bf16") == 0.0


def test_attained_metrics_bandwidth_fraction():
    numel = 1_000_000
    flops = float(numel)
    by = 2.0 * numel * 2
    r = R.roofline(flops, by, "bf16")
    m = R.attained_metrics(r["t_min_ms"], flops, by, "bf16")
    # memory-bound op running at t_min saturates HBM bandwidth
    assert m["pct_of_peak_bw"] == pytest.approx(100.0)
    assert m["pct_of_roofline"] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# occupancy re-export (single source of truth in pmc)
# --------------------------------------------------------------------------- #
def test_roofline_reexports_pmc_helpers():
    assert R.est_occupancy is pmc.est_occupancy
    assert R.l2_hit_rate is pmc.l2_hit_rate
    assert R.hbm_bytes is pmc.hbm_bytes


def test_mfma_flops():
    # MOPS counter is "ops in units of 512"; each FMA = 2 FLOPs
    assert R.mfma_flops({"SQ_INSTS_VALU_MFMA_MOPS_BF16": 10}) == pytest.approx(512 * 2 * 10)
    assert R.mfma_flops({"SQ_INSTS_VALU_MFMA_MOPS_BF16": 0}) == 0.0  # present but zero
    # only issue-count MFMA (no MOPS) -> cannot derive FLOPs
    assert R.mfma_flops({"SQ_INSTS_VALU_MFMA_F16": 5}) is None


# --------------------------------------------------------------------------- #
# bottleneck classification (the upgraded heuristic)
# --------------------------------------------------------------------------- #
def test_bottleneck_no_matrix_cores():
    label, ev = R.bottleneck_from_counters({"SQ_INSTS_VALU": 5000, "SQ_INSTS_VMEM": 200})
    assert label == "no-matrix-cores"
    assert "tl.dot" in ev


def test_bottleneck_memory_bound_from_low_l2_hit_rate():
    c = {"TCC_HIT_sum": 300, "TCC_MISS_sum": 700, "SQ_INSTS_VMEM": 1000,
         "TCC_EA0_RDREQ_sum": 100000, "TCC_EA0_RDREQ_32B_sum": 0}
    label, ev = R.bottleneck_from_counters(c)
    assert label == "memory-bound"
    assert "L2 hit-rate 30%" in ev and "MB HBM" in ev


def test_bottleneck_compute_bound():
    c = {"SQ_INSTS_VALU_MFMA_MOPS_BF16": 9000, "SQ_INSTS_VMEM": 100,
         "SQ_WAIT_INST_ANY": 50, "TCC_HIT_sum": 950, "TCC_MISS_sum": 50}
    label, ev = R.bottleneck_from_counters(c)
    assert label == "compute-bound"
    assert "roofline" in ev


def test_bottleneck_lds_bound():
    c = {"SQ_INSTS_VALU_MFMA_MOPS_BF16": 100, "SQ_INSTS_VMEM": 500,
         "SQ_WAIT_INST_LDS": 600, "SQ_WAIT_INST_VMEM": 100, "SQ_WAIT_INST_ANY": 1000}
    label, ev = R.bottleneck_from_counters(c)
    assert label == "lds-bound"
    assert "SQ_WAIT_INST_LDS" in ev


def test_bottleneck_occupancy_bound_uses_registers():
    # MFMA present (not no-matrix-cores), heavy stalls, and low occupancy from
    # high VGPR pressure -> occupancy-bound (the register-pressure upgrade).
    c = {"SQ_INSTS_VALU_MFMA_MOPS_BF16": 1000, "SQ_INSTS_VMEM": 500,
         "SQ_WAIT_INST_ANY": 5000}
    label, ev = R.bottleneck_from_counters(c, vgpr=200, num_warps=8)
    assert label == "occupancy-bound"
    assert "occupancy" in ev and "vgpr" in ev.lower()


def test_bottleneck_unknown_when_empty():
    assert R.bottleneck_from_counters({})[0] == "unknown"


def test_canonicalize_label_maps_to_grounded_reasoning_vocab():
    assert R.canonicalize_label("l2-bound") == "memory-bound"
    assert R.canonicalize_label("occupancy-bound") == "compute-bound"
    assert R.canonicalize_label("memory-bound") == "memory-bound"
    assert R.canonicalize_label("no-matrix-cores") == "no-matrix-cores"
    # every label we emit has grounding terms (so verify_reasoning_grounding works)
    for lbl in ("memory-bound", "l2-bound", "lds-bound", "no-matrix-cores",
                "occupancy-bound", "compute-bound"):
        assert R.BOTTLENECK_GROUNDING_TERMS[lbl]
