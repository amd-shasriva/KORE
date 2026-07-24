"""Compatibility-facade equivalence tests."""

from __future__ import annotations

import pytest

from kore.analysis import roofline as canonical
from kore.analysis import rooflines as legacy


@pytest.mark.parametrize(
    "operation,dims,dtype",
    [
        ("gemm", {"M": 512, "N": 1024, "K": 2048}, "bf16"),
        ("gemm_fp8", {"M": 256, "N": 256, "K": 256}, "fp8_e4m3"),
        ("rmsnorm", {"M": 4096, "N": 8192}, "bf16"),
        ("fused_add_rmsnorm", {"M": 4096, "N": 8192}, "bf16"),
        ("layernorm", {"M": 4096, "N": 8192}, "bf16"),
        ("silu_and_mul", {"M": 4096, "N": 8192}, "bf16"),
        ("gelu_tanh", {"M": 4096, "N": 8192}, "bf16"),
        ("softmax", {"M": 4096, "N": 8192}, "bf16"),
        ("rope", {"S": 2048, "B": 4, "H": 32, "D": 128}, "bf16"),
        ("quant_fp8_pertoken", {"M": 4096, "N": 8192}, "fp8_e4m3"),
    ],
)
def test_legacy_flops_bytes_is_exact_canonical_adapter(operation, dims, dtype):
    work = canonical.estimate_work(operation, dims, dtype)
    assert work is not None
    assert legacy.flops_bytes(operation, dims, dtype) == (work.flops, work.bytes)


@pytest.mark.parametrize(
    "operation",
    ["unknown", "flash_attn_decode", "fused_moe", "topk_softmax"],
)
def test_both_modules_return_unavailable(operation):
    dims = {"M": 128, "N": 128, "K": 128}
    assert canonical.estimate_work(operation, dims, "bf16") is None
    assert legacy.flops_bytes(operation, dims, "bf16") is None


def test_legacy_roofline_answer_matches_canonical():
    model = canonical.make_physical_model("mi350x")
    dims = {"M": 4096, "N": 4096, "K": 4096}
    work = canonical.estimate_work("gemm", dims, "bf16")
    expected = canonical.evaluate_roofline(work, model)
    peaks = legacy.resolve_peaks("gfx950", sku="mi350x")
    actual = legacy.roofline(
        "gemm_bf16", "gemm", "bf16", legacy.shape_to_str(dims), dims,
        peaks, "gfx950")
    assert actual is not None
    assert actual.t_min_ms == pytest.approx(expected.t_min_ms)
    assert actual.t_compute_ms == pytest.approx(expected.t_compute_ms)
    assert actual.t_mem_ms == pytest.approx(expected.t_memory_ms)
    assert actual.model_fingerprint == expected.model_fingerprint
    assert actual.work_model == work.model_kind


def test_resolve_peaks_is_explicit_and_ignores_legacy_env(monkeypatch):
    monkeypatch.setenv("KORE_PEAK_HBM_BW", "1")
    peaks = legacy.resolve_peaks("gfx950", sku="mi350x")
    assert peaks["hbm_bytes_per_s"] == 8.0e12
    assert peaks["model_fingerprint"] == canonical.make_physical_model(
        "mi350x").fingerprint


def test_calibrated_wrapper_and_canonical_match():
    calibration = {
        "architecture": "gfx950",
        "sku": "MI350X",
        "calibration_id": "test",
        "runtime": {"rocm": "test"},
        "hbm_bytes_per_s": 4.0e12,
        "compute_flops_per_s": {"bf16": 1.0e15},
    }
    model = legacy.resolve_model(sku="mi350x", calibration=calibration)
    peaks = legacy.resolve_peaks(
        "gfx950", sku="mi350x", calibration=calibration)
    dims = {"M": 1024, "N": 1024, "K": 1024}
    actual = legacy.roofline(
        "x", "gemm", "bf16", legacy.shape_to_str(dims), dims,
        peaks, "gfx950")
    expected = canonical.evaluate_roofline(
        canonical.estimate_work("gemm", dims, "bf16"), model)
    assert actual.t_min_ms == pytest.approx(expected.t_min_ms)
    assert actual.model_fingerprint == expected.model_fingerprint
