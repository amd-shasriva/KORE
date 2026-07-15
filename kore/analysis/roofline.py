"""Arch-aware (gfx950/CDNA4 default) roofline model + counter-grounded analysis.

Auto-selects the running GPU's peaks: **gfx950 / CDNA4 (AMD Instinct MI350X /
MI355X)** is the KORE target hardware and the default; gfx942 / CDNA3 (MI300X)
is retained for the previous-gen node. Override with ``KORE_ROOFLINE_ARCH``.

Pure, importable, CPU-only (no torch / no GPU at import) so it can run inside
datagen, the reward path, and unit tests. It answers three questions KORE's
grounded-reasoning (Pillar 4) and gold-win minting need:

  1. Given an op's FLOPs and mandatory HBM bytes, is it compute- or memory-bound,
     and what is the peak FLOP/s it could *possibly* attain?           :func:`roofline`
  2. Given a measured runtime, what fraction of that roofline peak did the kernel
     actually reach?                                              :func:`attained_fraction`
  3. Given rocprofv3 counters (+ optional VGPR/LDS/num_warps), what is the
     dominant hardware bottleneck, with quantitative evidence?  :func:`bottleneck_from_counters`

plus :func:`op_flop_bytes` so callers can turn an (op_family, shape, dtype) into
(FLOPs, bytes) and hence arithmetic intensity.

Relationship to :mod:`kore.analysis.rooflines` (note the plural): that module is
the operator-level Speed-of-Light (eta = T_min / T_measured) model used by the P0
falsification harness. THIS module is counter/measurement-oriented; the two are
intentionally decoupled so neither's edits break the other.

Hardware constants (DENSE matrix-core peaks - no 2x structured sparsity - and
datasheet HBM peak), per arch (see ``_ARCH_PEAKS``):
  * gfx950 / CDNA4 (MI350X, DEFAULT): HBM3E 8 TB/s, FP16/BF16 2.3 PFLOP/s,
    OCP-FP8/MXFP8/INT8 4.6, MXFP6/MXFP4 9.2, FP32 144.2 TFLOP/s, FP64 72.1;
    256 CUs @ 2.2 GHz. AMD Instinct MI350X product spec (CDNA4).
    https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html
  * gfx950 / CDNA4 (MI355X): HBM3E 8 TB/s, FP16/BF16 2.5, FP8 5.0, MXFP6/4 10.1,
    FP32 157.3, FP64 78.6; 256 CUs @ 2.4 GHz.
  * gfx942 / CDNA3 (MI300X): HBM3 5.325 TB/s, FP16/BF16 1307.4 TFLOP/s, FP8/INT8
    2614.9, TF32 653.7, FP32/FP64 163.4; 304 CUs @ 2.1 GHz. MI300-05A data sheet.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Union

from kore.verifier.pmc import (
    MAX_WAVES_PER_SIMD,
    _occupancy_constants,
    est_occupancy,
    hbm_bytes,
    l2_hit_rate,
)

# --------------------------------------------------------------------------- #
# Per-architecture hardware constants. FLOP/s are the DENSE matrix-core peaks (no
# 2x structured sparsity - the right ceiling for tl.dot/MFMA kernels); byte/s is
# the datasheet HBM peak. The ACTIVE set is auto-selected from the running GPU
# (see _detect_arch); override with KORE_ROOFLINE_ARCH=gfx950|gfx942|mi355x.
# --------------------------------------------------------------------------- #
_ARCH_PEAKS: dict[str, dict] = {
    # gfx950 / CDNA4 - AMD Instinct MI350X (air-cooled, 2.2 GHz).  DEFAULT hardware.
    "gfx950": {
        "name": "MI350X", "arch": "gfx950",
        "hbm_bw_bytes_per_s": 8.0e12,                              # 8 TB/s HBM3E
        "peak_flops_bf16": 2.30e15, "peak_flops_fp16": 2.30e15,   # 2.3 PFLOP matrix
        "peak_flops_fp8": 4.60e15, "peak_flops_int8": 4.60e15,    # OCP-FP8 / INT8
        "peak_flops_fp6": 9.20e15, "peak_flops_fp4": 9.20e15,     # MXFP6 / MXFP4
        "peak_flops_tf32": 1.153e15, "peak_flops_fp32": 1.442e14, # FP32 144.2 TFLOP
        "peak_flops_fp64": 7.21e13,                               # CDNA4 halves FP64
        "num_cus": 256, "peak_clock_hz": 2.2e9,
        "infinity_cache_bytes": 256 * 1024 * 1024,
    },
    # gfx950 / CDNA4 - AMD Instinct MI355X (direct-liquid-cooled, 2.4 GHz).
    "mi355x": {
        "name": "MI355X", "arch": "gfx950",
        "hbm_bw_bytes_per_s": 8.0e12,
        "peak_flops_bf16": 2.50e15, "peak_flops_fp16": 2.50e15,
        "peak_flops_fp8": 5.00e15, "peak_flops_int8": 5.00e15,
        "peak_flops_fp6": 10.10e15, "peak_flops_fp4": 10.10e15,
        "peak_flops_tf32": 1.2583e15, "peak_flops_fp32": 1.573e14,
        "peak_flops_fp64": 7.86e13,
        "num_cus": 256, "peak_clock_hz": 2.4e9,
        "infinity_cache_bytes": 256 * 1024 * 1024,
    },
    # gfx942 / CDNA3 - AMD Instinct MI300X (previous gen).
    "gfx942": {
        "name": "MI300X", "arch": "gfx942",
        "hbm_bw_bytes_per_s": 5.325e12,
        "peak_flops_bf16": 1.3074e15, "peak_flops_fp16": 1.3074e15,
        "peak_flops_fp8": 2.6149e15, "peak_flops_int8": 2.6149e15,
        "peak_flops_fp6": 2.6149e15, "peak_flops_fp4": 2.6149e15,  # no native fp4/6
        "peak_flops_tf32": 6.537e14, "peak_flops_fp32": 1.634e14,
        "peak_flops_fp64": 1.634e14,
        "num_cus": 304, "peak_clock_hz": 2.1e9,
        "infinity_cache_bytes": 256 * 1024 * 1024,
    },
}


def _detect_arch() -> str:
    """Select the roofline arch: ``KORE_ROOFLINE_ARCH`` override, else the running
    GPU's gfx target (via torch, no import cost if torch absent), else gfx950 (the
    KORE target hardware / CDNA4)."""
    env = os.environ.get("KORE_ROOFLINE_ARCH", "").strip().lower()
    if env in _ARCH_PEAKS:
        return env
    if env in ("mi350x", "mi350", "cdna4"):
        return "gfx950"
    if env in ("mi355", "mi355x"):
        return "mi355x"
    if env in ("mi300x", "mi300", "mi325x", "cdna3"):
        return "gfx942"
    try:  # pragma: no cover - hardware dependent
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            gcn = (getattr(torch.cuda.get_device_properties(0),
                           "gcnArchName", "") or "").lower()
            if "mi355" in name:
                return "mi355x"
            if "gfx950" in gcn or "gfx950" in name or "mi350" in name:
                return "gfx950"
            if "gfx942" in gcn or "mi300" in name or "mi325" in name:
                return "gfx942"
    except Exception:
        pass
    return "gfx950"  # KORE target hardware (CDNA4)


ACTIVE_ARCH: str = _detect_arch()
_ACTIVE: dict = _ARCH_PEAKS[ACTIVE_ARCH]

# Module-level constants reflect the ACTIVE arch (all downstream code uses these).
HBM_BW_BYTES_PER_S: float = _ACTIVE["hbm_bw_bytes_per_s"]
PEAK_FLOPS_BF16: float = _ACTIVE["peak_flops_bf16"]
PEAK_FLOPS_FP16: float = _ACTIVE["peak_flops_fp16"]
PEAK_FLOPS_FP8: float = _ACTIVE["peak_flops_fp8"]
PEAK_FLOPS_INT8: float = _ACTIVE["peak_flops_int8"]
PEAK_FLOPS_FP6: float = _ACTIVE["peak_flops_fp6"]
PEAK_FLOPS_FP4: float = _ACTIVE["peak_flops_fp4"]
PEAK_FLOPS_TF32: float = _ACTIVE["peak_flops_tf32"]
PEAK_FLOPS_FP32: float = _ACTIVE["peak_flops_fp32"]
PEAK_FLOPS_FP64: float = _ACTIVE["peak_flops_fp64"]

NUM_CUS: int = int(_ACTIVE["num_cus"])
PEAK_CLOCK_HZ: float = _ACTIVE["peak_clock_hz"]
INFINITY_CACHE_BYTES: int = int(_ACTIVE["infinity_cache_bytes"])


def _bundle(peaks: dict) -> dict:
    """Attach the ARCH-CORRECT pmc occupancy constants to a per-arch peak dict, so
    each board reports its OWN limits (MI300X = 64 KiB LDS / VGPR-granularity 16;
    MI350X/MI355X = 160 KiB LDS / granularity 8) instead of the ACTIVE arch's -- the
    old code stamped every board with the ACTIVE arch's constants (audit R2 pmc)."""
    occ = _occupancy_constants(peaks.get("arch", "gfx950"))
    return {**peaks, "lds_bytes_per_cu": occ["lds_bytes_per_cu"],
            "vgpr_per_simd": occ["vgpr_per_simd"],
            "max_waves_per_simd": occ["max_waves_per_simd"]}


# Per-board bundles + the ACTIVE one. ``MI300X`` kept as a back-compat name.
MI300X: dict = _bundle(_ARCH_PEAKS["gfx942"])
MI350X: dict = _bundle(_ARCH_PEAKS["gfx950"])
MI355X: dict = _bundle(_ARCH_PEAKS["mi355x"])
ACTIVE: dict = _bundle(_ACTIVE)


# --------------------------------------------------------------------------- #
# dtype helpers
# --------------------------------------------------------------------------- #
def dtype_bytes(dtype: str) -> float:
    """Element size in bytes for a dtype string (fp4=0.5, fp6=0.75, fp8/int8=1,
    fp16/bf16=2, fp32=4, ...). Sub-byte types are the packed storage size (the MX
    block scale overhead is negligible for the HBM-traffic lower bound)."""
    d = (dtype or "").lower()
    if "fp4" in d or "e2m1" in d:
        return 0.5   # packed 4-bit
    if "fp6" in d or "e2m3" in d or "e3m2" in d:
        return 0.75  # packed 6-bit
    if "fp8" in d or "float8" in d or "int8" in d or "e4m3" in d or "e5m2" in d:
        return 1
    if "bf16" in d or "fp16" in d or "float16" in d or "bfloat16" in d or "half" in d:
        return 2
    if "tf32" in d:
        return 4
    if "fp64" in d or "float64" in d or "double" in d:
        return 8
    if "fp32" in d or "float32" in d or "float" in d:
        return 4
    return 2  # default to bf16-sized (KORE's dominant training dtype)


def peak_flops(dtype: str) -> float:
    """Dense matrix-core peak FLOP/s on the ACTIVE arch for ``dtype``.

    fp4/fp6 (MXFP4/MXFP6) and OCP-FP8 are first-class on gfx950/CDNA4; on gfx942
    they fall back to the fp8 peak (no native sub-8-bit matrix path).
    """
    d = (dtype or "").lower()
    # sub-8-bit first (so mxfp4/mxfp6 don't get caught by an "fp8" test)
    if "fp4" in d or "e2m1" in d:
        return PEAK_FLOPS_FP4
    if "fp6" in d or "e2m3" in d or "e3m2" in d:
        return PEAK_FLOPS_FP6
    if "fp8" in d or "float8" in d or "e4m3" in d or "e5m2" in d:
        return PEAK_FLOPS_FP8
    if "int8" in d:
        return PEAK_FLOPS_INT8
    if "tf32" in d:
        return PEAK_FLOPS_TF32
    if "fp64" in d or "float64" in d or "double" in d:
        return PEAK_FLOPS_FP64
    if "fp32" in d or "float32" in d or ("float" in d and "16" not in d):
        return PEAK_FLOPS_FP32
    return PEAK_FLOPS_BF16  # bf16/fp16/half and default


# --------------------------------------------------------------------------- #
# The roofline
# --------------------------------------------------------------------------- #
def roofline(flops: float, bytes: float, dtype: str = "bf16") -> dict:
    """Roofline classification for an op that does ``flops`` FLOPs moving ``bytes``.

    Returns a dict with:
      * ``arithmetic_intensity``  - FLOP/byte (op intensity; dtype-independent).
      * ``bound``                 - ``"compute"`` if AI >= ridge point else ``"memory"``.
      * ``peak_attainable_flops`` - ``min(peak_flops, AI * peak_bw)`` FLOP/s, the
                                     highest FLOP/s this op could reach on the ACTIVE
                                     arch (gfx950 by default; see ``_detect_arch``).
      * ``ridge_point``           - peak_flops / peak_bw (FLOP/byte); ops above it
                                     are compute-bound.
      * ``peak_flops`` / ``peak_bandwidth_bytes_per_s`` - the dtype's ceilings used.
      * ``t_compute_ms`` / ``t_mem_ms`` / ``t_min_ms`` - the compute/memory time
                                     lower bounds and their max (the SOL runtime).
    """
    pf = peak_flops(dtype)
    pb = HBM_BW_BYTES_PER_S
    flops = max(0.0, float(flops))
    bytes = max(0.0, float(bytes))

    ai = (flops / bytes) if bytes > 0 else math.inf
    ridge = pf / pb
    # attainable FLOP/s = min(compute ceiling, memory-bandwidth ceiling at this AI)
    peak_attainable = pf if bytes <= 0 else min(pf, ai * pb)
    t_compute = flops / pf if pf > 0 else 0.0
    t_mem = bytes / pb if pb > 0 else 0.0
    bound = "compute" if ai >= ridge else "memory"

    return {
        "arithmetic_intensity": ai,
        "bound": bound,
        "peak_attainable_flops": peak_attainable,
        "ridge_point": ridge,
        "peak_flops": pf,
        "peak_bandwidth_bytes_per_s": pb,
        "flops": flops,
        "bytes": bytes,
        "t_compute_ms": t_compute * 1e3,
        "t_mem_ms": t_mem * 1e3,
        "t_min_ms": max(t_compute, t_mem) * 1e3,
    }


def attained_fraction(measured_ms: float, flops: float, bytes: float,
                      dtype: str = "bf16") -> float:
    """Percent of the ACTIVE-arch roofline peak a kernel achieved at ``measured_ms``.

    ``100 * (flops / measured_s) / peak_attainable_flops``. 100% == on the roofline.
    Values >100% are legitimate and signal cache reuse (true HBM traffic below the
    modeled ``bytes``, i.e. the op ran faster than its HBM lower bound). Returns 0.0
    for non-positive / unmodelable inputs (fail-safe for datagen).
    """
    if measured_ms is None or measured_ms <= 0 or flops is None or flops <= 0:
        return 0.0
    roof = roofline(flops, bytes, dtype)
    peak = roof["peak_attainable_flops"]
    if peak <= 0:
        return 0.0
    achieved = float(flops) / (float(measured_ms) / 1e3)
    return 100.0 * achieved / peak


def attained_metrics(measured_ms: float, flops: float, bytes: float,
                     dtype: str = "bf16") -> dict:
    """Richer companion to :func:`attained_fraction`: achieved FLOP/s + bandwidth
    and their fractions of the respective ceilings (all 0.0 for bad input)."""
    if measured_ms is None or measured_ms <= 0:
        return {"pct_of_roofline": 0.0, "achieved_flops": 0.0, "achieved_bw": 0.0,
                "pct_of_peak_flops": 0.0, "pct_of_peak_bw": 0.0}
    t_s = float(measured_ms) / 1e3
    roof = roofline(flops, bytes, dtype)
    achieved_flops = (float(flops) / t_s) if flops and flops > 0 else 0.0
    achieved_bw = (float(bytes) / t_s) if bytes and bytes > 0 else 0.0
    return {
        "pct_of_roofline": attained_fraction(measured_ms, flops, bytes, dtype),
        "achieved_flops": achieved_flops,
        "achieved_bw": achieved_bw,
        "pct_of_peak_flops": 100.0 * achieved_flops / roof["peak_flops"],
        "pct_of_peak_bw": 100.0 * achieved_bw / roof["peak_bandwidth_bytes_per_s"],
    }


# --------------------------------------------------------------------------- #
# op_family -> (FLOPs, mandatory HBM bytes)
# bytes is a LOWER bound on traffic (read inputs once + write outputs once; cache
# reuse ignored) so the derived roofline is a valid physical lower bound.
# --------------------------------------------------------------------------- #
ShapeLike = Union[dict, tuple, list, int]


def _mnk(shape: ShapeLike) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Best-effort (M, N, K) from a shape (dict with M/N/K, positional seq, or int)."""
    if isinstance(shape, dict):
        def gi(*keys):
            for k in keys:
                v = shape.get(k)
                if isinstance(v, int) and v > 0:
                    return v
            return None
        return gi("M", "m", "rows"), gi("N", "n", "cols"), gi("K", "k")
    if isinstance(shape, (tuple, list)):
        vals = [int(v) for v in shape if isinstance(v, int) and v > 0]
        if len(vals) >= 3:
            return vals[0], vals[1], vals[2]
        if len(vals) == 2:
            return vals[0], vals[1], None
        if len(vals) == 1:
            return 1, vals[0], None
        return None, None, None
    if isinstance(shape, int) and shape > 0:
        return 1, shape, None
    return None, None, None


def _numel(shape: ShapeLike) -> Optional[int]:
    """Total element count implied by a shape."""
    if isinstance(shape, int):
        return shape if shape > 0 else None
    if isinstance(shape, (tuple, list)):
        n = 1
        seen = False
        for v in shape:
            if isinstance(v, int) and v > 0:
                n *= v
                seen = True
        return n if seen else None
    if isinstance(shape, dict):
        n = 1
        seen = False
        for v in shape.values():
            if isinstance(v, int) and v > 0:
                n *= v
                seen = True
        return n if seen else None
    return None


def op_flop_bytes(op_family: str, shape: ShapeLike, dtype: str = "bf16",
                  n_tensors: int = 2, flops_per_elem: float = 1.0
                  ) -> Optional[tuple[float, float]]:
    """(FLOPs, mandatory HBM bytes) for a common op family, or None if unmodelable.

    Families (substring-matched, so ``"gemm_bias_relu"`` -> gemm, ``"rms_norm"`` ->
    norm):
      * elementwise / pointwise / activation:
            flops = flops_per_elem * numel ; bytes = n_tensors * numel * dsize
        ``n_tensors`` = #tensors streamed (default 2 = 1 read + 1 write; a binary
        op like add is 3).
      * gemm / matmul (needs M,N,K):
            flops = 2*M*N*K ; bytes = (M*K + K*N + M*N) * dsize
        (batched: multiply both by the leading batch dim B if present.)
      * reduction / reduce (rows M x cols N -> M):
            flops = M*N ; bytes = (M*N + M) * dsize
      * norm / rmsnorm / layernorm / softmax (rows M x cols N):
            per-family flops; bytes ~ read input + write output.
    """
    op = (op_family or "").lower()
    e = dtype_bytes(dtype)

    # ---- GEMM (batched first: "batched_gemm" contains "gemm") --------------
    if "gemm" in op or "matmul" in op or op in ("mm", "bmm"):
        M, N, K = _mnk(shape)
        if not (M and N and K):
            return None
        B = 1
        if isinstance(shape, dict):
            for bk in ("B", "batch", "num_batches"):
                if isinstance(shape.get(bk), int) and shape[bk] > 0:
                    B = shape[bk]
                    break
        flops = 2.0 * B * M * N * K
        by = float(B) * (M * K + K * N + M * N) * e
        return flops, by

    # ---- normalization families (rows M x cols N) --------------------------
    if "rmsnorm" in op or "rms_norm" in op:
        M, N, _ = _mnk(shape)
        if not (M and N):
            return None
        # fused add+rmsnorm streams 2 in + 2 out
        if "fused" in op or "add" in op:
            return 5.0 * M * N, float((4 * M * N + N) * e)
        return 4.0 * M * N, float((2 * M * N + N) * e)
    if "layernorm" in op or "layer_norm" in op:
        M, N, _ = _mnk(shape)
        if not (M and N):
            return None
        return 6.0 * M * N, float((2 * M * N + 2 * N) * e)
    if "softmax" in op:
        M, N, _ = _mnk(shape)
        if not (M and N):
            return None
        return 5.0 * M * N, float((2 * M * N) * e)
    if op in ("norm",) or op.endswith("_norm") or op.startswith("norm"):
        M, N, _ = _mnk(shape)
        if not (M and N):
            return None
        return 5.0 * M * N, float((2 * M * N) * e)

    # ---- reduction (rows M x cols N -> M) ----------------------------------
    if "reduc" in op or op in ("reduce", "sum", "mean", "row_sum", "row_mean",
                               "row_max", "row_min", "row_l2", "row_rms"):
        M, N, _ = _mnk(shape)
        if not N:
            return None
        M = M or 1
        return float(M * N), float((M * N + M) * e)

    # ---- elementwise / pointwise / activation (the default modelable path) -
    if op in ("elementwise", "pointwise", "ew") or "elementwise" in op \
            or "pointwise" in op or "activation" in op:
        numel = _numel(shape)
        if not numel:
            return None
        return float(flops_per_elem) * numel, float(n_tensors) * numel * e

    # ---- generic fallback: anything with usable dims -> memory-bound EW -----
    numel = _numel(shape)
    if numel:
        return float(flops_per_elem) * numel, float(n_tensors) * numel * e
    return None


# --------------------------------------------------------------------------- #
# Counter-grounded bottleneck classification (upgrades the crude wait/MFMA and
# wait-fraction heuristics with L2 hit-rate, HBM bytes, and occupancy).
# --------------------------------------------------------------------------- #
# Concepts a GROUNDED reasoning should mention for each label (superset of
# kore.data.grounded_reasoning._GROUNDING_TERMS; the two new labels have terms
# so verify_reasoning_grounding still works if the parent adopts these).
BOTTLENECK_GROUNDING_TERMS: dict[str, tuple[str, ...]] = {
    "memory-bound": ("memory", "vmem", "bandwidth", "coalesc", "global load",
                     "hbm", "load", "l2", "cache", "traffic"),
    "l2-bound": ("l2", "cache", "reuse", "hit rate", "hit-rate", "tile", "blocking"),
    "lds-bound": ("lds", "shared memory", "bank conflict", "smem"),
    "no-matrix-cores": ("mfma", "tl.dot", "matrix core", "matrix-core", "matrix unit"),
    "occupancy-bound": ("occupancy", "waves", "wavefront", "vgpr", "register",
                        "spill", "num_warps", "lds"),
    "compute-bound": ("compute", "mfma", "occupancy", "valu", "unroll", "pipeline"),
    "unknown": (),
}

# Map the richer labels onto kore.data.grounded_reasoning's 4-label vocabulary so
# the parent can swap this in without expanding its _GROUNDING_TERMS if it prefers.
_CANONICAL = {
    "memory-bound": "memory-bound",
    "l2-bound": "memory-bound",
    "lds-bound": "lds-bound",
    "no-matrix-cores": "no-matrix-cores",
    "occupancy-bound": "compute-bound",  # grounded_reasoning compute terms incl. "occupancy"
    "compute-bound": "compute-bound",
    "unknown": "unknown",
}


def canonicalize_label(label: str) -> str:
    """Map a :func:`bottleneck_from_counters` label onto grounded_reasoning's set."""
    return _CANONICAL.get(label, label)


def _sum_counter(counters: dict, *names: str) -> float:
    """Sum the first present of each name (0.0 if absent) - matches grounded_reasoning."""
    total = 0.0
    for n in names:
        v = counters.get(n)
        if isinstance(v, (int, float)):
            total += float(v)
    return total


def _mfma_ops(counters: dict) -> float:
    return sum(float(v) for k, v in counters.items()
               if "MFMA" in str(k).upper() and isinstance(v, (int, float)))


def mfma_flops(counters: dict) -> Optional[float]:
    """FLOPs performed by the matrix cores, from the ``*_MFMA_MOPS_*`` counters.

    Those count matrix-FMA ops "in units of 512"; each FMA is 2 FLOPs, so
    ``flops = 512 * 2 * sum(MOPS counters)``. None if no MOPS counter is present.
    """
    mops = sum(float(v) for k, v in counters.items()
               if "MFMA_MOPS" in str(k).upper() and isinstance(v, (int, float)))
    if mops <= 0 and not any("MFMA_MOPS" in str(k).upper() for k in counters):
        return None
    return 512.0 * 2.0 * mops


def bottleneck_from_counters(counters: dict, vgpr: Optional[int] = None,
                             lds: Optional[int] = None,
                             num_warps: Optional[int] = None) -> tuple[str, str]:
    """Dominant hardware bottleneck (label, evidence) from rocprofv3 counters.

    Upgrades the crude wait/MFMA-ratio heuristic with three real signals:
      * **L2 hit-rate** (``TCC_HIT``/``TCC_MISS``) and **HBM bytes** (EA req
        counts) to separate bandwidth-bound from cache/reuse-bound;
      * **occupancy** (from ``vgpr``/``lds``/``num_warps`` via the CDNA3 formula)
        to catch register/LDS-pressure limits that manifest as latency stalls.

    Priority: no-matrix-cores > occupancy-bound (low + stalling) > lds-bound >
    memory-bound (low L2 reuse / VMEM stalls) > compute-bound. Returns
    ``("unknown", ...)`` when there is nothing to go on. Labels are a superset of
    grounded_reasoning's; use :func:`canonicalize_label` to fold to its 4-set.
    """
    have_occ_inputs = vgpr is not None or lds is not None
    if not counters and not have_occ_inputs:
        return "unknown", "no counters collected"

    counters = counters or {}
    mfma = _mfma_ops(counters)
    valu = _sum_counter(counters, "SQ_INSTS_VALU")
    vmem = _sum_counter(counters, "SQ_INSTS_VMEM")
    lds_wait = _sum_counter(counters, "SQ_WAIT_INST_LDS")
    vmem_wait = _sum_counter(counters, "SQ_WAIT_INST_VMEM")  # may be absent on gfx942
    any_wait = _sum_counter(counters, "SQ_WAIT_INST_ANY") or (lds_wait + vmem_wait)
    vmem_active = _sum_counter(counters, "SQ_ACTIVE_INST_VMEM")

    hit = l2_hit_rate(counters)
    hbm = hbm_bytes(counters)
    occ = est_occupancy(vgpr, lds, num_warps) if have_occ_inputs else None

    def _hbm_mb() -> str:
        return f"{hbm / 1e6:.1f} MB HBM" if hbm is not None else "HBM n/a"

    # 1) matrix cores idle -----------------------------------------------------
    if valu > 0 and mfma == 0.0 and (vmem > 0 or valu > 0):
        return ("no-matrix-cores",
                f"MFMA ops=0 while SQ_INSTS_VALU={valu:.0f} - matrix cores idle; use tl.dot")

    # 2) occupancy-limited (low AND unable to hide latency) --------------------
    if occ is not None and occ.occupancy <= 0.25:
        stalling = (any_wait > 0 and (valu + vmem + mfma) > 0
                    and any_wait >= 0.30 * (valu + vmem + mfma + any_wait))
        very_low = occ.waves_per_simd <= 1.0
        if stalling or very_low:
            lim = occ.limiter
            reg = (f"VGPR={vgpr}" if vgpr is not None else "")
            ldss = (f"LDS={lds}B/wg" if lds is not None else "")
            detail = ", ".join(x for x in (reg, ldss) if x)
            return ("occupancy-bound",
                    f"occupancy {occ.occupancy*100:.0f}% "
                    f"({occ.waves_per_simd:.1f}/{MAX_WAVES_PER_SIMD} waves per SIMD, "
                    f"{lim}-limited{': ' + detail if detail else ''}) - too few waves to "
                    f"hide latency; cut register/LDS pressure to raise occupancy")

    # 3) LDS-bound -------------------------------------------------------------
    if any_wait > 0:
        lds_frac = lds_wait / any_wait
        vmem_frac = vmem_wait / any_wait
        if lds_frac >= 0.30 and lds_frac >= vmem_frac:
            return ("lds-bound",
                    f"SQ_WAIT_INST_LDS {lds_frac:.0%} of SQ_WAIT_INST_ANY "
                    f"({lds_wait:.0f}/{any_wait:.0f}) - LDS bank conflicts / pressure")

    # 4) memory-bound: low L2 reuse (bandwidth) or VMEM stalls -----------------
    if hit is not None and hit < 0.50 and (hbm is None or hbm > 0):
        return ("memory-bound",
                f"L2 hit-rate {hit*100:.0f}% (TCC_HIT/(TCC_HIT+TCC_MISS)); {_hbm_mb()} "
                f"traffic - HBM-bandwidth-bound; improve coalescing / reuse")
    if any_wait > 0 and (vmem_wait / any_wait) >= 0.50:
        return ("memory-bound",
                f"SQ_WAIT_INST_VMEM {vmem_wait / any_wait:.0%} of SQ_WAIT_INST_ANY "
                f"({vmem_wait:.0f}/{any_wait:.0f}); {_hbm_mb()} - stalled on global loads")

    # 5) high L2 reuse but still memory-heavy -> cache/reuse regime ------------
    if hit is not None and hit >= 0.90 and vmem > 0 and mfma < vmem:
        return ("l2-bound",
                f"L2 hit-rate {hit*100:.0f}% with VMEM-heavy issue "
                f"(SQ_INSTS_VMEM={vmem:.0f} > MFMA={mfma:.0f}); {_hbm_mb()} - working set "
                f"fits L2, bound by cache/VMEM throughput not HBM")

    # 6) compute-bound ---------------------------------------------------------
    if mfma > 0 and mfma >= vmem:
        hitnote = f", L2 hit-rate {hit*100:.0f}%" if hit is not None else ""
        return ("compute-bound",
                f"MFMA-heavy (MFMA ops={mfma:.0f} >= SQ_INSTS_VMEM={vmem:.0f}{hitnote}) "
                f"- near the compute roofline")

    # 7) weak fallbacks --------------------------------------------------------
    if hit is not None:
        return ("memory-bound",
                f"L2 hit-rate {hit*100:.0f}%; {_hbm_mb()} - memory-dominated")
    if vmem > 0 or vmem_active > 0:
        return ("memory-bound",
                f"VMEM-heavy (SQ_INSTS_VMEM={vmem:.0f}, MFMA={mfma:.0f})")
    if mfma > 0:
        return ("compute-bound", f"MFMA ops={mfma:.0f} with little memory traffic")
    if occ is not None:
        return ("occupancy-bound",
                f"occupancy {occ.occupancy*100:.0f}% "
                f"({occ.waves_per_simd:.1f}/{MAX_WAVES_PER_SIMD} waves per SIMD)")
    return "unknown", "counters inconclusive"


__all__ = [
    "MI300X",
    "HBM_BW_BYTES_PER_S",
    "PEAK_FLOPS_BF16",
    "PEAK_FLOPS_FP16",
    "PEAK_FLOPS_FP8",
    "PEAK_FLOPS_FP32",
    "dtype_bytes",
    "peak_flops",
    "roofline",
    "attained_fraction",
    "attained_metrics",
    "op_flop_bytes",
    "bottleneck_from_counters",
    "canonicalize_label",
    "BOTTLENECK_GROUNDING_TERMS",
    "mfma_flops",
    # re-exported counter helpers (single source of truth in kore.verifier.pmc)
    "l2_hit_rate",
    "hbm_bytes",
    "est_occupancy",
]
