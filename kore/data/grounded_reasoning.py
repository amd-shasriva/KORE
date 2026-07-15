"""Profiler-counter-grounded optimization reasoning (Pillar 4).

The gold-win reasoning is a templated string and QA is unverified teacher NL -
which teaches the model to *say* optimization words, not to reason from evidence.
World-class kernel-optimization CoT follows PROFILE -> DIAGNOSE -> TRANSFORM ->
MEASURE, grounded in REAL rocprofv3 hardware counters (MFMA util, VMEM traffic,
LDS/VMEM stall waits). This module turns the counters KORE already collects
(``kore.verifier.pmc.COUNTER_SETS`` via ``KoreEnv._collect_profile``) into:

  * :func:`diagnose_bottleneck` - a counter-driven bottleneck classification
    (memory-bound / LDS-bound / no-matrix-cores / compute-bound) with the specific
    evidence, so the teacher (and the demo) reason from measured signal.
  * :func:`counter_grounded_prompt` - a teacher prompt that injects the real
    counters + the diagnosis and requires every claim to cite a counter.
  * :func:`verify_reasoning_grounding` - a check that a produced reasoning actually
    references the measured bottleneck (reject fabricated/ungrounded CoT).

The pure diagnosis/prompt/verify core is CPU-testable; :func:`collect_counters`
is a thin, fail-safe GPU wrapper over the existing profiler path.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Bottleneck label -> the human concepts a GROUNDED reasoning should reference
# (used by verify_reasoning_grounding). Keys mirror diagnose_bottleneck outputs.
_GROUNDING_TERMS: dict[str, tuple[str, ...]] = {
    "memory-bound": ("memory", "vmem", "bandwidth", "coalesc", "global load",
                     "hbm", "load"),
    "lds-bound": ("lds", "shared memory", "bank conflict", "smem"),
    "no-matrix-cores": ("mfma", "tl.dot", "matrix core", "matrix-core", "matrix unit"),
    "compute-bound": ("compute", "mfma", "occupancy", "valu", "unroll", "pipeline"),
}


def _get(counters: dict, *names: str) -> float:
    for n in names:
        v = counters.get(n)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def diagnose_bottleneck(counters: dict) -> tuple[str, str]:
    """Classify the kernel bottleneck from rocprofv3 counters. Returns (label, evidence).

    Heuristics (gfx942), in priority order:
      * no-matrix-cores - MFMA issue count is ~0 while VALU is nonzero (the kernel
        hand-rolls scalar FMAs instead of using tl.dot -> the matrix cores idle);
      * lds-bound - LDS stall waits dominate total waits (bank conflicts / pressure);
      * memory-bound - VMEM stall waits dominate (stalled on global loads);
      * compute-bound - MFMA-heavy with few memory stalls (near the compute roofline).
    Falls back to ("unknown", ...) when counters are missing.
    """
    if not counters:
        return "unknown", "no counters collected"
    mfma = _get(counters, "SQ_INSTS_VALU_MFMA_BF16") + \
        _get(counters, "SQ_INSTS_VALU_MFMA_F16") + _get(counters, "SQ_INSTS_VALU_MFMA_F32")
    valu = _get(counters, "SQ_INSTS_VALU")
    vmem = _get(counters, "SQ_INSTS_VMEM")
    lds_wait = _get(counters, "SQ_WAIT_INST_LDS")
    vmem_wait = _get(counters, "SQ_WAIT_INST_VMEM")
    any_wait = _get(counters, "SQ_WAIT_INST_ANY") or (lds_wait + vmem_wait)

    if valu > 0 and mfma == 0.0 and vmem > 0:
        return ("no-matrix-cores",
                f"MFMA issues=0 while VALU={valu:.0f} - matrix cores idle; use tl.dot")
    if any_wait > 0:
        lds_frac = lds_wait / any_wait
        vmem_frac = vmem_wait / any_wait
        if lds_frac >= 0.30 and lds_frac >= vmem_frac:
            return ("lds-bound",
                    f"LDS stall waits {lds_frac:.0%} of total ({lds_wait:.0f}/{any_wait:.0f})")
        if vmem_frac >= 0.50:
            return ("memory-bound",
                    f"VMEM stall waits {vmem_frac:.0%} of total ({vmem_wait:.0f}/{any_wait:.0f})")
    if mfma > 0 and mfma >= vmem:
        return ("compute-bound",
                f"MFMA issues={mfma:.0f} >= VMEM={vmem:.0f}; near the compute roofline")
    if vmem > 0:
        return ("memory-bound", f"VMEM-heavy (VMEM={vmem:.0f}, MFMA={mfma:.0f})")
    return "unknown", "counters inconclusive"


def _roofline():
    """Import the roofline module if present (built by the analysis workstream).

    Returns the module or None so grounded reasoning works before/without it and
    gets RICHER (L2 hit-rate, HBM bytes, occupancy, %-of-peak) once it lands."""
    try:
        from kore.analysis import roofline as _rf
        return _rf
    except Exception:  # noqa: BLE001 - optional, forward-compatible
        return None


def diagnose_bottleneck_rich(counters: dict, *, vgpr=None, lds=None, num_warps=None,
                             flops=None, bytes=None, measured_ms=None, dtype=None):
    """Bottleneck diagnosis that prefers the roofline module's counter model (real
    L2 hit-rate / HBM bytes / occupancy) and falls back to :func:`diagnose_bottleneck`.

    Returns ``(label, evidence)`` where evidence cites measured signal."""
    # collect_counters stores resource fields inside the counters dict; use them for
    # occupancy if the caller didn't pass them explicitly.
    if isinstance(counters, dict):
        vgpr = vgpr if vgpr is not None else counters.get("vgpr_count")
        lds = lds if lds is not None else counters.get("lds_bytes")
        num_warps = num_warps if num_warps is not None else counters.get("num_warps")
    rf = _roofline()
    if rf is not None:
        try:
            fn = getattr(rf, "bottleneck_from_counters", None)
            if fn is not None:
                label, evidence = fn(counters, vgpr=vgpr, lds=lds, num_warps=num_warps)
                # augment with roofline attainment when we can compute it
                if flops and bytes and measured_ms:
                    try:
                        frac = rf.attained_fraction(measured_ms, flops, bytes, dtype)
                        evidence = f"{evidence}; ~{frac:.0%} of {label.split('-')[0]} roofline"
                    except Exception:  # noqa: BLE001
                        pass
                return label, evidence
        except Exception:  # noqa: BLE001 - never fatal; fall back
            pass
    return diagnose_bottleneck(counters)


def _delta_note(parent: dict, best: dict) -> str:
    """Human-readable counter DELTAS parent->best (only for counters present in both)."""
    if not (isinstance(parent, dict) and isinstance(best, dict)):
        return ""
    rf = _roofline()
    parts: list[str] = []
    if rf is not None:
        for name, fn in (("L2 hit-rate", getattr(rf, "l2_hit_rate", None)),
                         ("HBM bytes", getattr(rf, "hbm_bytes", None))):
            if fn is None:
                continue
            try:
                pv, bv = fn(parent), fn(best)
                if pv and bv:
                    unit = "%" if "rate" in name else ""
                    scale = 100.0 if unit == "%" else 1.0
                    parts.append(f"{name} {pv*scale:.0f}{unit}->{bv*scale:.0f}{unit}")
            except Exception:  # noqa: BLE001
                pass
    # generic stall-wait delta from the always-present issue/wait counters
    pv, bv = _get(parent, "SQ_WAIT_INST_VMEM"), _get(best, "SQ_WAIT_INST_VMEM")
    if pv and bv:
        parts.append(f"VMEM stall-waits {pv:.0f}->{bv:.0f}")
    return "; ".join(parts)


def build_grounded_analysis(op: str, *, parent_counters: Optional[dict],
                            best_counters: Optional[dict] = None,
                            parent_wall_us: Optional[float] = None,
                            best_wall_us: Optional[float] = None,
                            snr_db: Optional[float] = None,
                            speedup: Optional[float] = None,
                            flops: Optional[float] = None, bytes: Optional[float] = None,
                            dtype: Optional[str] = None) -> Optional[str]:
    """Frontier PROFILE->DIAGNOSE->TRANSFORM->MEASURE analysis grounded in REAL counters.

    Uses the PARENT's counters to diagnose the bottleneck the optimization targets
    (fixing the old bug that narrated the winner's counters as the parent's), the
    BEST kernel's counters + measured speedup for the MEASURE step, and the roofline
    module (when present) for %-of-peak attainment + L2/HBM deltas. Returns ``None``
    when no parent counters are available (caller falls back to the measurement note)."""
    if not isinstance(parent_counters, dict) or not parent_counters:
        return None
    p_ms = (parent_wall_us / 1000.0) if parent_wall_us else None
    b_ms = (best_wall_us / 1000.0) if best_wall_us else None
    p_label, p_ev = diagnose_bottleneck_rich(
        parent_counters, flops=flops, bytes=bytes, measured_ms=p_ms, dtype=dtype)
    profile = (f"PROFILE: the parent kernel"
               + (f" ({parent_wall_us:.1f}us)" if parent_wall_us else "")
               + f" profiled as {p_label} ({p_ev}).")
    diagnose = (f"DIAGNOSE: the dominant cost is the {p_label.replace('-', ' ')} regime above, "
                f"so the highest-impact change targets exactly that (not micro-tuning).")
    transform = f"TRANSFORM: {_transform_hint(p_label)}."
    measure_bits = []
    if speedup:
        measure_bits.append(f"{speedup:.2f}x faster")
    if best_wall_us:
        measure_bits.append(f"{best_wall_us:.1f}us")
    if snr_db is not None:
        measure_bits.append(f"SNR {snr_db:.0f} dB")
    delta = _delta_note(parent_counters, best_counters or {})
    measure = ("MEASURE: the optimized kernel is " + ", ".join(measure_bits)
               + (f" ({delta})" if delta else "") + ".") if measure_bits else ""
    return " ".join(x for x in (profile, diagnose, transform, measure) if x)


_TRANSFORM_HINTS = {
    "memory-bound": ("coalesce global loads into 128-bit (global_load_dwordx4) and reuse "
                     "operands in LDS to cut redundant HBM traffic"),
    "lds-bound": ("remove LDS bank conflicts (XOR-swizzle / pad) and widen LDS access to "
                  "ds_read_b128 to cut shared-memory stalls"),
    "no-matrix-cores": ("route the inner product through tl.dot so the MFMA matrix cores do "
                        "the multiply-accumulate instead of scalar VALU FMAs"),
    "compute-bound": ("raise MFMA utilization/occupancy (matrix_instr_nonkdim=16, tune "
                      "num_warps/num_stages, waves_per_eu) to approach the compute roofline"),
}


def _transform_hint(label: str) -> str:
    return _TRANSFORM_HINTS.get(label,
                                "address the measured bottleneck with the minimal structural change")


def _fmt_counters(counters: dict) -> str:
    return "\n".join(f"  {k} = {counters[k]}" for k in sorted(counters)) or "  (none)"


def counter_grounded_prompt(op: str, counters: dict, transform: Optional[str] = None) -> str:
    """Teacher prompt that requires reasoning GROUNDED in the measured counters."""
    label, evidence = diagnose_bottleneck(counters)
    ask_transform = (f"The change applied was: {transform}. Explain why it addresses "
                     f"the measured bottleneck.\n" if transform else
                     "Name the single highest-impact change and why the counters justify it.\n")
    return (
        f"You are optimizing a `{op}` Triton kernel on AMD gfx942 (CDNA3).\n"
        f"MEASURED rocprofv3 hardware counters for the current kernel:\n"
        f"{_fmt_counters(counters)}\n\n"
        f"Counter-based diagnosis: {label} ({evidence}).\n\n"
        f"{ask_transform}"
        "Write the ANALYSIS as a PROFILE -> DIAGNOSE -> FIX chain. Ground EVERY claim "
        "in a specific counter above (cite the counter name and value); do NOT assert "
        "a bottleneck the counters do not support."
    )


def verify_reasoning_grounding(reasoning: str, counters: dict) -> dict:
    """Check that ``reasoning`` references the measured bottleneck. Returns a report.

    ``grounded`` is True iff the reasoning mentions at least one concept associated
    with the counter-diagnosed bottleneck (a cheap guard against fabricated CoT that
    ignores the profile). Also reports whether any raw counter name is cited.
    """
    label, evidence = diagnose_bottleneck(counters)
    text = (reasoning or "").lower()
    terms = _GROUNDING_TERMS.get(label, ())
    mentions_bottleneck = any(t in text for t in terms)
    cites_counter = bool(re.search(r"sq_[a-z_0-9]+", text)) or any(
        str(k).lower() in text for k in counters)
    return {
        "bottleneck": label,
        "evidence": evidence,
        "mentions_bottleneck": mentions_bottleneck,
        "cites_counter": cites_counter,
        "grounded": bool(mentions_bottleneck),
    }


def collect_counters(env: Any, source: str, shape: Any = None) -> Optional[dict]:
    """Best-effort rocprofv3 counter collection for a kernel via the KoreEnv path.

    Returns ``{counter: value}`` or None if profiling is unavailable/failed. Fully
    fail-safe (never raises) - grounded reasoning degrades to the templated path when
    counters can't be collected (e.g. profiler off, CPU box).
    """
    try:
        fn = getattr(env, "collect_counters", None) or getattr(env, "_counters_for", None)
        if fn is None:
            return None
        out = fn(source) if shape is None else fn(source, shape)
        return out if isinstance(out, dict) and out else None
    except Exception:  # noqa: BLE001 - profiling is advisory; never fatal
        return None


__all__ = [
    "diagnose_bottleneck",
    "diagnose_bottleneck_rich",
    "build_grounded_analysis",
    "counter_grounded_prompt",
    "verify_reasoning_grounding",
    "collect_counters",
]
