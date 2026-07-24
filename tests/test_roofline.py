"""Unit/property tests for the authoritative physical model."""

from __future__ import annotations

import math
import random

import pytest

from kore.analysis import roofline as R


def test_explicit_mi350x_model_is_stable_and_fingerprinted():
    model = R.make_physical_model("mi350x")
    assert model.architecture == "gfx950"
    assert model.sku == "MI350X"
    assert model.hbm_bytes_per_s == pytest.approx(8.0e12)
    assert model.peak_flops_per_s("bf16") == pytest.approx(2.30e15)
    assert model.fingerprint.startswith("sha256:")
    assert model == R.make_physical_model("mi350x")
    assert not hasattr(R, "ACTIVE_ARCH")
    assert not hasattr(R, "HBM_BW_BYTES_PER_S")


def test_sku_is_not_inferred_from_ambiguous_architecture():
    with pytest.raises(R.ModelError):
        R.hardware_spec("gfx950")
    assert R.hardware_spec("mi350x").sku != R.hardware_spec("mi355x").sku


def test_calibration_requires_identity_runtime_and_units():
    incomplete = {
        "architecture": "gfx950",
        "hbm_bytes_per_s": 4.5e12,
        "compute_flops_per_s": {"bf16": 1.2e15},
    }
    with pytest.raises(R.ModelError, match="fingerprint-safe"):
        R.make_physical_model("mi350x", incomplete)

    calibration = {
        "architecture": "gfx950",
        "sku": "MI350X",
        "calibration_id": "node-a-2026-07-23",
        "source": "runtime-measured",
        "runtime": {"rocm": "7.2.3", "device_id": "abc"},
        "hbm_bytes_per_s": 4.5e12,
        "compute_flops_per_s": {"bf16": 1.2e15, "fp16": 1.2e15},
    }
    model = R.make_physical_model("mi350x", calibration)
    assert model.peak_flops_per_s("fp8") is None
    assert model.runtime["rocm"] == "7.2.3"
    with pytest.raises(R.ModelError, match="fingerprint mismatch"):
        R.make_physical_model(
            "mi350x", calibration, expected_fingerprint="sha256:wrong")


def test_integrity_model_uses_upper_not_achievable_peaks():
    calibration = {
        "architecture": "gfx950",
        "sku": "MI350X",
        "calibration_id": "slow-cal",
        "runtime": {"rocm": "x"},
        "hbm_bytes_per_s": 4.0e12,
        "compute_flops_per_s": {"bf16": 1.0e15},
    }
    empirical = R.make_physical_model("mi350x", calibration)
    integrity = empirical.for_integrity()
    assert integrity.integrity_upper_bound
    assert integrity.hbm_bytes_per_s == 8.0e12
    assert integrity.peak_flops_per_s("bf16") == 2.30e15
    assert integrity.fingerprint != empirical.fingerprint


def test_dtype_support_is_explicit():
    assert R.dtype_bytes("bf16") == 2.0
    assert R.dtype_bytes("mxfp4") == 0.5
    assert R.dtype_bytes("mystery16") is None
    assert R.make_physical_model("mi300x").peak_flops_per_s("mxfp4") is None
    assert R.peak_flops("unknown", R.make_physical_model("mi350x")) is None


def test_work_units_validate_nonfinite_and_nonpositive():
    with pytest.raises(R.ModelError):
        R.WorkEstimate("x", "bf16", math.nan, 1.0, "exact")
    with pytest.raises(R.ModelError):
        R.WorkEstimate("x", "bf16", 1.0, 0.0, "exact")
    with pytest.raises(R.ModelError):
        R.WorkEstimate("x", "unknown", 1.0, 1.0, "exact")


def test_supported_work_formulas():
    gemm = R.estimate_work(
        "gemm", {"M": 512, "N": 1024, "K": 2048}, "bf16")
    assert gemm.flops == 2.0 * 512 * 1024 * 2048
    assert gemm.bytes == (512 * 2048 + 2048 * 1024 + 512 * 1024) * 2

    fp8 = R.estimate_work(
        "gemm_fp8", {"M": 256, "N": 256, "K": 256}, "fp8_e4m3")
    assert fp8.bytes == 2 * 256 * 256 + 2 * 256 * 256
    assert "bf16 output" in fp8.assumptions

    norm = R.estimate_work("rmsnorm", {"M": 64, "N": 512}, "bf16")
    assert norm.flops == 4.0 * 64 * 512
    assert norm.bytes == (2 * 64 * 512 + 512) * 2


@pytest.mark.parametrize(
    "operation,dtype",
    [
        ("mystery_fusion", "bf16"),
        ("flash_attn_decode", "bf16"),
        ("fused_moe", "bf16"),
        ("topk_softmax", "bf16"),
        ("rmsnorm_backward", "bf16"),
        ("gemm_mxfp4", "mxfp4"),
        ("gemm_int8", "int8"),
    ],
)
def test_unsupported_work_is_unavailable(operation, dtype):
    assert R.estimate_work(
        operation, {"M": 128, "N": 128, "K": 128}, dtype) is None


def test_no_generic_elementwise_fabrication():
    assert R.op_flop_bytes(
        "weird_fused_thing", {"X": 10, "Y": 20}, "bf16") is None
    assert R.op_flop_bytes("elementwise", 200, "bf16") == (200.0, 800.0)


def test_roofline_and_attainment_use_explicit_model():
    model = R.make_physical_model("mi350x")
    work = R.estimate_work(
        "gemm", {"M": 4096, "N": 4096, "K": 4096}, "bf16")
    result = R.evaluate_roofline(work, model)
    assert result is not None and result.bound == "compute"
    assert result.model_fingerprint == model.fingerprint
    assert R.attainment(result.t_min_ms, result) == pytest.approx(1.0)
    assert R.attainment(2 * result.t_min_ms, result) == pytest.approx(0.5)
    assert R.attainment(math.nan, result) is None
    assert R.attained_fraction(
        result.t_min_ms, work.flops, work.bytes, work.dtype, model
    ) == pytest.approx(100.0)
    assert R.attained_fraction(
        0.0, work.flops, work.bytes, work.dtype, model) is None


def test_roofline_property_tmin_is_max_of_unit_terms():
    rng = random.Random(19)
    model = R.make_physical_model("mi350x")
    for _ in range(100):
        work = R.WorkEstimate(
            "property",
            "bf16",
            10 ** rng.uniform(3, 15),
            10 ** rng.uniform(3, 10),
            "exact",
        )
        result = R.evaluate_roofline(work, model)
        assert result.t_min_ms == pytest.approx(
            max(result.t_compute_ms, result.t_memory_ms))
        assert result.attainable_flops_per_s <= result.peak_flops_per_s
        assert result.attainable_flops_per_s > 0.0


def test_counter_units_prevent_mops_and_instruction_mixing():
    counters = {
        "SQ_INSTS_VALU": 100,
        "SQ_INSTS_VMEM": 20,
        "SQ_INSTS_VALU_MFMA_F16": 7,
        "SQ_INSTS_VALU_MFMA_MOPS_BF16": 10,
        "SQ_WAIT_INST_ANY": 999,
    }
    assert R.counter_unit("SQ_WAIT_INST_ANY") == R.CounterUnit.QCYCLES
    assert R.counter_unit("SQ_INSTS_VALU") == R.CounterUnit.INSTRUCTIONS
    assert R.counter_unit(
        "SQ_INSTS_VALU_MFMA_MOPS_BF16") == R.CounterUnit.MOPS_512_FMA
    assert R.issued_instructions(counters) == 120  # MFMA already in VALU
    assert R.mfma_instruction_count(counters) == 7
    assert R.mfma_flops(counters) == 10 * 512 * 2


def test_exact_hbm_bytes_requires_transaction_split():
    assert R.hbm_bytes({"TCC_EA0_RDREQ_sum": 10}) is None
    counters = {
        "TCC_EA0_RDREQ_sum": 10,
        "TCC_EA0_RDREQ_32B_sum": 4,
        "TCC_EA0_WRREQ_sum": 5,
        "TCC_EA0_WRREQ_64B_sum": 2,
    }
    assert R.hbm_bytes(counters) == 4 * 32 + 6 * 64 + 2 * 64 + 3 * 32


def test_bottleneck_requires_same_unit_utilization_evidence():
    label, _ = R.bottleneck_from_counters(
        {"SQ_INSTS_VALU_MFMA_MOPS_BF16": 1000})
    assert label == "unknown"
    label, evidence = R.bottleneck_from_counters({"MfmaUtil": 80.0})
    assert label == "compute-bound" and "80%" in evidence
    label, _ = R.bottleneck_from_counters(
        {"SQ_INSTS_VALU": 1000, "SQ_INSTS_VALU_MFMA_MOPS_BF16": 0})
    assert label == "no-matrix-cores"


def test_occupancy_is_sku_explicit():
    cdna4 = R.est_occupancy(
        vgpr=128, lds=32768, num_warps=4,
        model=R.make_physical_model("mi350x"))
    cdna3 = R.est_occupancy(
        vgpr=128, lds=32768, num_warps=4,
        model=R.make_physical_model("mi300x"))
    assert cdna4.occupancy == pytest.approx(0.5)
    assert cdna3.occupancy == pytest.approx(0.25)
