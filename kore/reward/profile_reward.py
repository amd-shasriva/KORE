"""Hardware-counter-grounded dense reward (KORE's flagship novelty).

Prior kernel-generation RL (Kevin-32B, AutoTriton, KernelBench-style agents)
rewards only two things: correctness and *wall-clock* speedup. That signal is
SPARSE and, worse, FLAT across the "correct-but-slow" regime the policy stalls
in — a 0.7x kernel and a 0.95x kernel look almost the same to the advantage
estimator, so the model learns "be correct" and stops optimizing.

KORE adds a DENSE reward grounded in AMD hardware performance counters
(rocprofv3 PMC). The idea: wall-clock speedup is the *effect*; the *causes* are
measurable — pipeline stalls (SQ_WAIT_*), issued-instruction efficiency, and
memory traffic (SQ_INSTS_VMEM / TCP-TCC). A kernel that moves toward the
hardware roofline (fewer stalls per issued instruction, less memory traffic than
the vendor baseline) is genuinely closer to being fast, even before it crosses
1x. Rewarding roofline *attainment relative to the tuned baseline* gives gradient
exactly where the sparse speedup reward is flat.

Anti-hacking (this is a NEW reward surface, so it is designed defensively):
  * All components are RELATIVE to the reference (AITER/hipBLASLt) and bounded to
    [0, 1]; absolute counter magnitudes (which scale with problem size and are
    trivially inflatable) never enter the reward.
  * The term is only ever applied on the CORRECT tier, so a kernel cannot lower
    its stall/traffic counters by "doing less" — it must still produce the right
    answer on every shape (and pass the determinism re-check).
  * The caller keeps ``profile_reward_weight`` strictly below the fast_p bonuses,
    so actually beating the baseline (wall-clock) always dominates a merely
    counter-efficient kernel — the profiler reward SHAPES, it never leads.

All functions here are pure and CPU-testable; GPU collection lives in
``kore.verifier.pmc`` and is wired in by ``KoreEnv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _get(counters: dict, *names: str) -> int:
    """Sum the first matching counter for each requested name (0 if absent)."""
    total = 0
    for n in names:
        for k, v in counters.items():
            if k.upper() == n.upper():
                total += int(v)
                break
    return total


def _mfma(counters: dict) -> int:
    return sum(int(v) for k, v in counters.items() if "MFMA" in k.upper())


def issued_instructions(counters: dict) -> int:
    """Total issued instructions (VALU + SALU + VMEM + MFMA), a work proxy."""
    valu = _get(counters, "SQ_INSTS_VALU")
    salu = _get(counters, "SQ_INSTS_SALU")
    vmem = _get(counters, "SQ_INSTS_VMEM")
    return valu + salu + vmem + _mfma(counters)


def stall_fraction(counters: dict) -> Optional[float]:
    """Fraction of cycles the wavefronts spent WAITING vs issuing work.

    ``SQ_WAIT_INST_ANY / (issued + SQ_WAIT_INST_ANY)`` in [0, 1]; lower is better
    (a well-scheduled kernel keeps the ALUs fed). None when counters are missing.
    """
    wait = _get(counters, "SQ_WAIT_INST_ANY")
    issued = issued_instructions(counters)
    denom = issued + wait
    if denom <= 0:
        return None
    return wait / denom


def issue_efficiency(counters: dict) -> Optional[float]:
    """1 - stall_fraction: fraction of activity spent issuing real work."""
    sf = stall_fraction(counters)
    return None if sf is None else (1.0 - sf)


@dataclass
class ProfileMetrics:
    """Human-readable derived metrics (observability; not the reward itself)."""
    cand_stall_fraction: Optional[float]
    ref_stall_fraction: Optional[float]
    cand_issue_efficiency: Optional[float]
    ref_issue_efficiency: Optional[float]
    cand_vmem: int
    ref_vmem: int
    efficiency_score: float


def profile_efficiency_score(cand: dict, ref: dict) -> Optional[float]:
    """Roofline-attainment reward in [0, 1] for a CORRECT candidate vs the baseline.

    Two hardware-grounded, baseline-relative, bounded components:

      stall_component   = issue_efficiency(cand) / issue_efficiency(ref), clamp[0,1]
                          -> 1.0 when the candidate keeps the ALUs as busy as the
                             vendor baseline (or busier); < 1 when it stalls more.
      traffic_component = vmem(ref) / vmem(cand), clamp[0,1]
                          -> 1.0 when the candidate moves no more memory than the
                             baseline; < 1 when it is more bandwidth-hungry.

    Score = 0.5*stall_component + 0.5*traffic_component. Returns None when neither
    component can be computed (no usable counters), so the caller can no-op.
    """
    comps: list[float] = []

    ce, re_ = issue_efficiency(cand), issue_efficiency(ref)
    if ce is not None and re_ is not None and re_ > 0:
        comps.append(_clamp01(ce / re_))

    cv, rv = _get(cand, "SQ_INSTS_VMEM"), _get(ref, "SQ_INSTS_VMEM")
    if cv > 0 and rv > 0:
        comps.append(_clamp01(rv / cv))

    if not comps:
        return None
    return sum(comps) / len(comps)


def profile_metrics(cand: dict, ref: dict) -> ProfileMetrics:
    """Full derived-metrics record for logging/diagnosis."""
    score = profile_efficiency_score(cand, ref)
    return ProfileMetrics(
        cand_stall_fraction=stall_fraction(cand),
        ref_stall_fraction=stall_fraction(ref),
        cand_issue_efficiency=issue_efficiency(cand),
        ref_issue_efficiency=issue_efficiency(ref),
        cand_vmem=_get(cand, "SQ_INSTS_VMEM"),
        ref_vmem=_get(ref, "SQ_INSTS_VMEM"),
        efficiency_score=(0.0 if score is None else score),
    )
