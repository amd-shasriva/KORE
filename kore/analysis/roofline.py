"""MI300X (gfx942 / CDNA3) roofline model + counter-grounded bottleneck analysis.

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
falsification harness, and its ``gfx942`` peaks are tuned for the MI325X node it
runs on. THIS module is anchored to the **MI300X** datasheet (the KORE target
device) and is counter/measurement-oriented; the two are intentionally decoupled
so neither's edits break the other.

Hardware constants (well-sourced; see inline citations):
  * HBM3 bandwidth 5.325 TB/s, FP16/BF16 matrix 1307.4 TFLOP/s, FP8/INT8 2614.9,
    TF32 653.7, FP32 163.4, FP64 163.4 (matrix) / 81.7 (vector): AMD Instinct
    MI300X data sheet (MI300-05A) + ROCm MI300 microarchitecture peak table
    https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html
  * 304 CUs @ up to 2.1 GHz, 256 MB Infinity Cache, 64 KB LDS/CU, 512 VGPR/SIMD:
    same data sheet + ROCm workload-optimization guide (occupancy section).
"""

from __future__ import annotations

import math
from typing import Optional, Union

from kore.verifier.pmc import (
    LDS_BYTES_PER_CU,
    MAX_WAVES_PER_SIMD,
    VGPR_PER_SIMD,
    est_occupancy,
    hbm_bytes,
    l2_hit_rate,
)

# --------------------------------------------------------------------------- #
# MI300X / gfx942 / CDNA3 hardware constants.
# FLOP/s are the DENSE matrix-core peaks (no 2x structured sparsity), which is
# the right ceiling for tl.dot/MFMA kernels. byte/s is the datasheet HBM3 peak.
# --------------------------------------------------------------------------- #
HBM_BW_BYTES_PER_S: float = 5.325e12       # 5.325 TB/s (8192-bit bus * 5.2 Gbps / 8)
PEAK_FLOPS_BF16: float = 1.3074e15         # 1307.4 TFLOP/s dense BF16 matrix
PEAK_FLOPS_FP16: float = 1.3074e15         # 1307.4 TFLOP/s dense FP16 matrix
PEAK_FLOPS_FP8: float = 2.6149e15          # 2614.9 TFLOP/s dense FP8 matrix
PEAK_FLOPS_INT8: float = 2.6149e15         # 2614.9 TOP/s INT8 matrix
PEAK_FLOPS_TF32: float = 6.537e14          # 653.7 TFLOP/s TF32
PEAK_FLOPS_FP32: float = 1.634e14          # 163.4 TFLOP/s FP32 (vector or matrix)
PEAK_FLOPS_FP64: float = 1.634e14          # 163.4 TFLOP/s FP64 matrix (vector: 81.7)

NUM_CUS: int = 304
PEAK_CLOCK_HZ: float = 2.1e9
INFINITY_CACHE_BYTES: int = 256 * 1024 * 1024  # 256 MB L2/Infinity Cache

# Convenience bundle of every MI300X constant (also surfaces the occupancy
# constants that live canonically in kore.verifier.pmc).
MI300X: dict[str, float] = {
    "arch": "gfx942",
    "hbm_bw_bytes_per_s": HBM_BW_BYTES_PER_S,
    "peak_flops_bf16": PEAK_FLOPS_BF16,
    "peak_flops_fp16": PEAK_FLOPS_FP16,
    "peak_flops_fp8": PEAK_FLOPS_FP8,
    "peak_flops_int8": PEAK_FLOPS_INT8,
    "peak_flops_tf32": PEAK_FLOPS_TF32,
    "peak_flops_fp32": PEAK_FLOPS_FP32,
    "peak_flops_fp64": PEAK_FLOPS_FP64,
    "num_cus": NUM_CUS,
    "peak_clock_hz": PEAK_CLOCK_HZ,
    "lds_bytes_per_cu": LDS_BYTES_PER_CU,
    "vgpr_per_simd": VGPR_PER_SIMD,
    "max_waves_per_simd": MAX_WAVES_PER_SIMD,
    "infinity_cache_bytes": INFINITY_CACHE_BYTES,
}


# --------------------------------------------------------------------------- #
# dtype helpers
# --------------------------------------------------------------------------- #
def dtype_bytes(dtype: str) -> int:
    """Element size in bytes for a dtype string (fp8=1, fp16/bf16=2, fp32=4, ...)."""
    d = (dtype or "").lower()
    if "fp8" in d or "float8" in d or "int8" in d or "e4m3" in d or "e5m2" in d:
        return 1
    if "fp4" in d or "mxfp4" in d:
        return 1  # packed 4-bit; approximate as 1B
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
    """Dense matrix-core peak FLOP/s on MI300X for ``dtype``."""
    d = (dtype or "").lower()
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
      * ``arithmetic_intensity``  — FLOP/byte (op intensity; dtype-independent).
      * ``bound``                 — ``"compute"`` if AI >= ridge point else ``"memory"``.
      * ``peak_attainable_flops`` — ``min(peak_flops, AI * peak_bw)`` FLOP/s, the
                                     highest FLOP/s this op could reach on MI300X.
      * ``ridge_point``           — peak_flops / peak_bw (FLOP/byte); ops above it
                                     are compute-bound.
      * ``peak_flops`` / ``peak_bandwidth_bytes_per_s`` — the dtype's ceilings used.
      * ``t_compute_ms`` / ``t_mem_ms`` / ``t_min_ms`` — the compute/memory time
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
    """Percent of the MI300X roofline peak a kernel achieved at ``measured_ms``.

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
    """Sum the first present of each name (0.0 if absent) — matches grounded_reasoning."""
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
                f"MFMA ops=0 while SQ_INSTS_VALU={valu:.0f} — matrix cores idle; use tl.dot")

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
                    f"{lim}-limited{': ' + detail if detail else ''}) — too few waves to "
                    f"hide latency; cut register/LDS pressure to raise occupancy")

    # 3) LDS-bound -------------------------------------------------------------
    if any_wait > 0:
        lds_frac = lds_wait / any_wait
        vmem_frac = vmem_wait / any_wait
        if lds_frac >= 0.30 and lds_frac >= vmem_frac:
            return ("lds-bound",
                    f"SQ_WAIT_INST_LDS {lds_frac:.0%} of SQ_WAIT_INST_ANY "
                    f"({lds_wait:.0f}/{any_wait:.0f}) — LDS bank conflicts / pressure")

    # 4) memory-bound: low L2 reuse (bandwidth) or VMEM stalls -----------------
    if hit is not None and hit < 0.50 and (hbm is None or hbm > 0):
        return ("memory-bound",
                f"L2 hit-rate {hit*100:.0f}% (TCC_HIT/(TCC_HIT+TCC_MISS)); {_hbm_mb()} "
                f"traffic — HBM-bandwidth-bound; improve coalescing / reuse")
    if any_wait > 0 and (vmem_wait / any_wait) >= 0.50:
        return ("memory-bound",
                f"SQ_WAIT_INST_VMEM {vmem_wait / any_wait:.0%} of SQ_WAIT_INST_ANY "
                f"({vmem_wait:.0f}/{any_wait:.0f}); {_hbm_mb()} — stalled on global loads")

    # 5) high L2 reuse but still memory-heavy -> cache/reuse regime ------------
    if hit is not None and hit >= 0.90 and vmem > 0 and mfma < vmem:
        return ("l2-bound",
                f"L2 hit-rate {hit*100:.0f}% with VMEM-heavy issue "
                f"(SQ_INSTS_VMEM={vmem:.0f} > MFMA={mfma:.0f}); {_hbm_mb()} — working set "
                f"fits L2, bound by cache/VMEM throughput not HBM")

    # 6) compute-bound ---------------------------------------------------------
    if mfma > 0 and mfma >= vmem:
        hitnote = f", L2 hit-rate {hit*100:.0f}%" if hit is not None else ""
        return ("compute-bound",
                f"MFMA-heavy (MFMA ops={mfma:.0f} >= SQ_INSTS_VMEM={vmem:.0f}{hitnote}) "
                f"— near the compute roofline")

    # 7) weak fallbacks --------------------------------------------------------
    if hit is not None:
        return ("memory-bound",
                f"L2 hit-rate {hit*100:.0f}%; {_hbm_mb()} — memory-dominated")
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
