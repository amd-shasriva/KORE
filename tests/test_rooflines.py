"""CPU-only per-operator roofline flops/bytes formula tests (Phase 7).

Asserts the EXACT mandatory-FLOP and minimal-HBM-byte formulas for the modeled
operators (GEMM + norms + elementwise + rope/quant), the roofline bound
selection, and the KORE_PEAK_* override path. These are the load-bearing numbers
behind eta and the residual decomposition, so they are pinned exactly.
"""

from __future__ import annotations

from kore.analysis import rooflines as R


def test_gemm_exact():
    flops, by = R.flops_bytes("gemm", {"M": 512, "N": 1024, "K": 2048}, "bf16")
    assert flops == 2.0 * 512 * 1024 * 2048
    assert by == (512 * 2048 + 2048 * 1024 + 512 * 1024) * 2


def test_gemm_fp8_bytes_mixed():
    flops, by = R.flops_bytes("gemm_fp8", {"M": 256, "N": 256, "K": 256}, "fp8_e4m3fnuz")
    assert flops == 2.0 * 256 ** 3
    # fp8 inputs (1B) + bf16 output (2B)
    assert by == (256 * 256 + 256 * 256) * 1 + 256 * 256 * 2


def test_rmsnorm_exact():
    flops, by = R.flops_bytes("rmsnorm", {"M": 4096, "N": 8192}, "bf16")
    assert flops == 4.0 * 4096 * 8192
    assert by == (2 * 4096 * 8192 + 8192) * 2


def test_fused_add_rmsnorm_exact():
    flops, by = R.flops_bytes("fused_add_rmsnorm", {"M": 4096, "N": 8192}, "bf16")
    assert flops == 5.0 * 4096 * 8192
    assert by == (4 * 4096 * 8192 + 8192) * 2


def test_layernorm_exact():
    flops, by = R.flops_bytes("layernorm", {"M": 4096, "N": 8192}, "bf16")
    assert flops == 6.0 * 4096 * 8192
    assert by == (2 * 4096 * 8192 + 2 * 8192) * 2


def test_silu_mul_exact():
    flops, by = R.flops_bytes("silu_and_mul", {"M": 4096, "N": 8192}, "bf16")
    assert flops == 4.0 * 4096 * 8192
    assert by == (3 * 4096 * 8192) * 2   # in=2N, out=N -> 3*M*N elems


def test_gelu_exact():
    flops, by = R.flops_bytes("gelu_tanh", {"M": 4096, "N": 8192}, "bf16")
    assert flops == 8.0 * 4096 * 8192
    assert by == (2 * 4096 * 8192) * 2


def test_softmax_exact():
    flops, by = R.flops_bytes("softmax", {"M": 4096, "N": 8192}, "bf16")
    assert flops == 5.0 * 4096 * 8192
    assert by == (2 * 4096 * 8192) * 2


def test_rope_exact():
    dims = {"S": 2048, "B": 4, "H": 32, "D": 128}
    flops, by = R.flops_bytes("rope", dims, "bf16")
    n = 2048 * 4 * 32 * 128
    assert flops == 6.0 * n
    assert by == 2 * n * 2 + 2048 * 128 * 2


def test_quant_fp8_pertoken_exact():
    flops, by = R.flops_bytes("quant_fp8_pertoken", {"M": 4096, "N": 8192}, "fp8_e4m3fnuz")
    assert flops == 2.0 * 4096 * 8192
    assert by == 4096 * 8192 * 2 + 4096 * 8192 * 1 + 4096 * 4


def test_unmodelable_op_no_dims_returns_none():
    assert R.flops_bytes("no_such_op", {}, "bf16") is None


def test_batched_gemm_exact():
    flops, by = R.flops_bytes("batched_gemm", {"B": 8, "M": 128, "N": 256, "K": 512}, "bf16")
    assert flops == 2.0 * 8 * 128 * 256 * 512
    assert by == 8 * (128 * 512 + 512 * 256 + 128 * 256) * 2


def test_generic_elementwise_memory_bound_lower_bound():
    # any op with usable dims but no explicit model -> memory-bound elementwise
    # LOWER bound: flops = size, bytes = 2*size (read one operand + write output),
    # a true traffic lower bound so eta = T_min/T_measured stays in (0, 1].
    flops, by = R.flops_bytes("some_pointwise", {"M": 128, "N": 256}, "bf16")
    assert flops == 128 * 256
    assert by == 2 * 128 * 256 * 2   # 2 * size * bf16(2B)


def test_roofline_bound_is_max_of_compute_mem():
    peaks = R.resolve_peaks("gfx950")
    rf = R.roofline("gemm_bf16", "gemm", "bf16", "M=4096,N=4096,K=4096",
                    {"M": 4096, "N": 4096, "K": 4096}, peaks, "gfx950")
    assert rf.t_min_ms == max(rf.t_compute_ms, rf.t_mem_ms)
    assert rf.bound == ("compute" if rf.t_compute_ms >= rf.t_mem_ms else "memory")


def test_rmsnorm_is_memory_bound():
    peaks = R.resolve_peaks("gfx950")
    rf = R.roofline("rmsnorm", "rmsnorm", "bf16", "M=4096,N=4096",
                    {"M": 4096, "N": 4096}, peaks, "gfx950")
    assert rf.bound == "memory"


def test_peak_override_env(monkeypatch):
    monkeypatch.setenv("KORE_PEAK_HBM_BW", "9.9e12")
    monkeypatch.setenv("KORE_PEAK_BF16", "3.0e15")
    monkeypatch.setenv("KORE_PEAK_FP8", "6.0e15")
    p = R.resolve_peaks("gfx950")
    assert p["hbm_bytes_per_s"] == 9.9e12
    assert p["bf16_flops_per_s"] == p["fp16_flops_per_s"] == 3.0e15
    assert p["fp8_flops_per_s"] == 6.0e15


def test_default_arch_gfx950():
    assert R.DEFAULT_ARCH == "gfx950"
