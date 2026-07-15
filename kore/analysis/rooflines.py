"""Roofline / Speed-of-Light (SOL) model for KORE operators.

The physical lower bound on a kernel's runtime is set by the operator (its
mandatory FLOPs and HBM byte traffic), NOT by how the kernel is written:

    T_min = max( W_flops / P_peak ,  Q_bytes / B_peak )

A kernel can only *approach* T_min. The SOL-attainment ratio

    eta = T_min / T_measured   in (0, 1]

is therefore an absolute, arch-normalized measure of how close a kernel got to
the physics limit. Because T_min uses each arch's own peaks, eta is directly
comparable across gfx942 (MI325X / CDNA3) and gfx950 (MI350-class / CDNA4) --
which is exactly what the cross-architecture-transfer thesis needs.

Peaks are curated approximate vendor specs (dense matrix, no structured
sparsity). They set the *absolute scale* of eta; refine with a STREAM triad or
datasheet. Override any of them at runtime, no code edit:

    KORE_PEAK_BF16   dense bf16/fp16 matrix peak, FLOP/s
    KORE_PEAK_FP8    dense fp8 matrix peak, FLOP/s
    KORE_PEAK_HBM_BW HBM bandwidth, byte/s

This node is gfx950, so gfx950 is the default arch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------- #
# Curated peaks. Units: FLOP/s and byte/s. Dense matrix, no 2x sparsity.
# gfx950 = MI350X/MI355X (CDNA4); gfx942 = MI325X (CDNA3, kept for cross-arch).
# --------------------------------------------------------------------------- #
PEAKS: dict[str, dict[str, float]] = {
    "gfx950": {  # AMD Instinct MI350X, CDNA4 -- THIS NODE (rocminfo: MI350X @ 2.2 GHz)
        # Datasheet dense matrix peaks (no 2x structured sparsity). MI355X (liquid,
        # 2.4 GHz) is ~9% higher: bf16 2.5, fp8 5.0. Kept to the reported board.
        "hbm_bytes_per_s": 8.0e12,    # 8.0 TB/s HBM3E (288 GB)
        "bf16_flops_per_s": 2.3e15,   # 2.3 PFLOP/s dense bf16/fp16 matrix
        "fp16_flops_per_s": 2.3e15,
        "fp8_flops_per_s": 4.6e15,    # 4.6 PFLOP/s OCP-FP8 matrix
        "int8_flops_per_s": 4.6e15,   # 4.6 POPS INT8 matrix
        "fp6_flops_per_s": 9.2e15,    # 9.2 PFLOP/s MXFP6 matrix (CDNA4 headline)
        "fp4_flops_per_s": 9.2e15,    # 9.2 PFLOP/s MXFP4 matrix (CDNA4 headline)
        "fp32_flops_per_s": 1.442e14, # 144.2 TFLOP/s FP32 matrix
    },
    "gfx942": {  # MI325X, CDNA3
        "hbm_bytes_per_s": 6.0e12,    # ~6.0 TB/s HBM3E
        "bf16_flops_per_s": 1.3e15,   # ~1.3 PFLOP/s dense bf16/fp16
        "fp16_flops_per_s": 1.3e15,
        "fp8_flops_per_s": 2.6e15,
        "int8_flops_per_s": 2.6e15,
        "fp6_flops_per_s": 2.6e15,    # no native sub-8-bit matrix on CDNA3 -> = fp8
        "fp4_flops_per_s": 2.6e15,
        "fp32_flops_per_s": 1.6e14,
    },
}

DEFAULT_ARCH = "gfx950"


def detect_arch(default: str = DEFAULT_ARCH) -> str:
    """Best-effort GPU arch from rocminfo/rocm-smi; falls back to ``default``."""
    for cmd in (["rocminfo"], ["rocm-smi", "--showproductname"]):
        exe = shutil.which(cmd[0])
        if not exe:
            continue
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        except Exception:  # noqa: BLE001
            continue
        for tok in ("gfx950", "gfx942", "gfx90a", "gfx1100"):
            if tok in out:
                return tok
    return default


def resolve_peaks(arch: Optional[str] = None) -> dict[str, float]:
    """Peak table for ``arch`` with KORE_PEAK_* env overrides applied."""
    arch = arch or DEFAULT_ARCH
    p = dict(PEAKS.get(arch, PEAKS[DEFAULT_ARCH]))
    bf16 = _env_float("KORE_PEAK_BF16")
    fp8 = _env_float("KORE_PEAK_FP8")
    hbm = _env_float("KORE_PEAK_HBM_BW")
    if bf16:
        p["bf16_flops_per_s"] = p["fp16_flops_per_s"] = bf16
    if fp8:
        p["fp8_flops_per_s"] = fp8
    if hbm:
        p["hbm_bytes_per_s"] = hbm
    return p


def _env_float(name: str) -> Optional[float]:
    v = os.environ.get(name)
    try:
        return float(v) if v else None
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# dtype byte sizes and peak selection
# --------------------------------------------------------------------------- #
def dtype_bytes(dtype: str) -> float:
    d = (dtype or "").lower()
    # sub-8-bit first (mxfp4/mxfp6 must not be caught by the fp8 test)
    if "fp4" in d or "mxfp4" in d or "e2m1" in d:
        return 0.5   # packed 4-bit
    if "fp6" in d or "mxfp6" in d or "e2m3" in d or "e3m2" in d:
        return 0.75  # packed 6-bit
    if "fp8" in d or "float8" in d or "int8" in d:
        return 1
    if "bf16" in d or "fp16" in d or "float16" in d or "bfloat16" in d:
        return 2
    if "fp32" in d or "float32" in d:
        return 4
    return 2


def peak_flops(peaks: dict[str, float], dtype: str) -> float:
    d = (dtype or "").lower()
    # sub-8-bit MXFP4/MXFP6 are CDNA4's headline matrix path (9.2 PF); check before fp8.
    if "fp4" in d or "mxfp4" in d or "e2m1" in d:
        return peaks.get("fp4_flops_per_s", peaks["fp8_flops_per_s"])
    if "fp6" in d or "mxfp6" in d or "e2m3" in d or "e3m2" in d:
        return peaks.get("fp6_flops_per_s", peaks["fp8_flops_per_s"])
    if "fp8" in d or "float8" in d:
        return peaks["fp8_flops_per_s"]
    if "int8" in d:
        return peaks.get("int8_flops_per_s", peaks["fp8_flops_per_s"])
    if "fp32" in d or "float32" in d:
        return peaks["fp32_flops_per_s"]
    return peaks["bf16_flops_per_s"]


# --------------------------------------------------------------------------- #
# Mandatory FLOPs + HBM bytes per operator family.
# bytes = minimal HBM traffic (read inputs once + write outputs once); cache
# reuse is deliberately IGNORED so this is a true LOWER bound on traffic (hence
# an UPPER bound on eta). Returns (flops, bytes) or None if unmodeled.
#
# The GEMM / norm / elementwise models are exact. Attention and MoE are
# first-order approximations (documented); tighten if check (a) passes on
# GEMM/norm but fails on those.
# --------------------------------------------------------------------------- #
def flops_bytes(operation: str, dims: dict[str, int], dtype: str) -> Optional[tuple[float, float]]:
    op = (operation or "").lower()
    e = dtype_bytes(dtype)
    is_fp8 = "fp8" in (dtype or "").lower()

    def g(*keys, default=None):
        for k in keys:
            if k in dims:
                return dims[k]
        return default

    try:
        # ---- batched / grouped GEMM (exact) -------------------------------
        # Must precede the dense-GEMM branch ("batched_gemm" contains "gemm").
        if "batched_gemm" in op or "grouped_gemm" in op or op == "bmm":
            B = g("B", "batch", "num_batches", default=1)
            M, N, K = dims["M"], dims["N"], dims["K"]
            flops = 2.0 * B * M * N * K
            if is_fp8:
                by = B * ((M * K + K * N) * 1 + M * N * 2)
            else:
                by = B * (M * K + K * N + M * N) * e
            return flops, float(by)

        # ---- GEMM / matmul (exact) -----------------------------------------
        if "gemm" in op or "matmul" in op:
            M, N, K = dims["M"], dims["N"], dims["K"]
            flops = 2.0 * M * N * K
            if is_fp8:
                by = (M * K + K * N) * 1 + M * N * 2   # fp8 inputs, bf16 output
            else:
                by = (M * K + K * N + M * N) * e
            return flops, float(by)

        # ---- fused add + rmsnorm (2 reads, 2 writes) (exact-ish) -----------
        if "fused_add" in op or ("rmsnorm" in op and "fused" in op):
            M, N = dims["M"], dims["N"]
            return 5.0 * M * N, float((4 * M * N + N) * e)

        # ---- rmsnorm (memory-bound; exact-ish) -----------------------------
        if "rmsnorm" in op:
            M, N = dims["M"], dims["N"]
            return 4.0 * M * N, float((2 * M * N + N) * e)

        # ---- layernorm -----------------------------------------------------
        if "layernorm" in op or "layer_norm" in op:
            M, N = dims["M"], dims["N"]
            return 6.0 * M * N, float((2 * M * N + 2 * N) * e)

        # ---- gated SiLU-mul (input width 2N -> output N) -------------------
        if "silu" in op and "moe" not in op:
            M, N = dims["M"], dims["N"]
            return 4.0 * M * N, float((3 * M * N) * e)

        # ---- GELU / activation ---------------------------------------------
        if "gelu" in op or "relu" in op:
            M, N = dims["M"], dims["N"]
            return 8.0 * M * N, float((2 * M * N) * e)

        # ---- topk-softmax MoE router ---------------------------------------
        if "topk" in op or "router" in op:
            M, E = dims["M"], dims["E"]
            topk = g("topk", "k", default=max(1, E // 4))
            return 5.0 * M * E, float(M * E * e + M * topk * (e + 4))

        # ---- dense softmax -------------------------------------------------
        if "softmax" in op and "moe" not in op:
            M, N = dims["M"], dims["N"]
            return 5.0 * M * N, float((2 * M * N) * e)

        # ---- RoPE ----------------------------------------------------------
        if "rope" in op:
            S, B, H, D = dims["S"], dims["B"], dims["H"], dims["D"]
            n = S * B * H * D
            return 6.0 * n, float(2 * n * e + S * D * e)

        # ---- per-token fp8 quant -------------------------------------------
        if "quant" in op:
            M, N = dims["M"], dims["N"]
            return 2.0 * M * N, float(M * N * 2 + M * N * 1 + M * 4)

        # ---- fused MoE MLP (grouped GEMM: gate+up+down) (approx) -----------
        if "moe" in op:
            M = dims["M"]
            E = g("E", "n_experts", default=8)
            topk = g("topk", "k", default=2)
            D = g("D", "hidden", "d_model")
            I = g("I", "inter", "d_ff")
            if D is None or I is None:
                return None
            flops = M * topk * 6.0 * D * I            # gate+up+down GEMMs
            by = (E * 3 * D * I + 2 * M * D) * e       # all expert weights + in/out act
            return flops, float(by)

        # ---- attention (prefill vs decode) (approx) -----------------------
        if "attn" in op or "attention" in op:
            B, H, D = dims["B"], dims["H"], dims["D"]
            KV = g("KV", "kv_heads", default=H)
            if "decode" in op:
                Skv = g("Skv", "S_kv", "S", "seqlen")
                if Skv is None:
                    return None
                flops = 4.0 * B * H * Skv * D           # QK^T + PV, seq_q=1
                by = (2 * B * KV * Skv * D + 2 * B * H * D) * e  # stream K,V + Q,O
                return flops, float(by)
            S = g("S", "seqlen")
            if S is None:
                return None
            flops = 4.0 * B * H * S * S * D * 0.5        # causal prefill (~half)
            by = (2 * B * H * S * D + 2 * B * KV * S * D) * e
            return flops, float(by)

        # ---- generic elementwise (memory-bound LOWER bound) ---------------
        # Any remaining op with usable integer dims (the gen_*/genv_* activation
        # + pointwise zoo: relu, add, mul, exp, sqrt, sigmoid, tanh, mish, ...)
        # is modeled as a memory-bound elementwise kernel. Mandatory HBM traffic
        # is at LEAST read one operand + write one output = 2*size bytes -- a true
        # LOWER bound on traffic (never an over-estimate), so T_min is a valid
        # lower bound and eta = T_min/T_measured stays in (0, 1]. ~1 flop/elem.
        size = 1
        seen_dim = False
        for v in dims.values():
            if isinstance(v, int) and v > 0:
                size *= v
                seen_dim = True
        if seen_dim:
            return float(size), float(2 * size * e)
    except (KeyError, TypeError):
        return None
    return None


# --------------------------------------------------------------------------- #
# Roofline result
# --------------------------------------------------------------------------- #
@dataclass
class Roofline:
    task_id: str
    operation: str
    dtype: str
    shape: str
    arch: str
    flops: float
    bytes: float
    arithmetic_intensity: float   # FLOP/byte
    t_compute_ms: float
    t_mem_ms: float
    t_min_ms: float               # max(compute, mem) -- physical lower bound
    bound: str                    # "compute" | "memory"


def roofline(task_id: str, operation: str, dtype: str, shape_str: str,
             dims: dict[str, int], peaks: dict[str, float], arch: str) -> Optional[Roofline]:
    fb = flops_bytes(operation, dims, dtype)
    if fb is None:
        return None
    flops, by = fb
    if by <= 0:
        return None
    pf = peak_flops(peaks, dtype)
    pb = peaks["hbm_bytes_per_s"]
    t_c = flops / pf
    t_m = by / pb
    t_min = max(t_c, t_m)
    return Roofline(
        task_id=task_id, operation=operation, dtype=dtype, shape=shape_str, arch=arch,
        flops=flops, bytes=by, arithmetic_intensity=flops / by,
        t_compute_ms=t_c * 1e3, t_mem_ms=t_m * 1e3, t_min_ms=t_min * 1e3,
        bound="compute" if t_c >= t_m else "memory",
    )


def shape_to_str(dims: dict[str, int]) -> str:
    return ",".join(f"{k}={v}" for k, v in dims.items())
