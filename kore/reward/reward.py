"""Lexicographic, anti-hackable reward for KORE.

Priority order (a strictly better outcome in an earlier tier always dominates):
    1. compiles
    2. passes 5-stage validation + SNR gate on ALL shapes (correctness)
    3. speedup vs the *production* baseline (AITER/hipBLASLt), scored on the
       WORST shape so a candidate cannot win by over-fitting one easy shape.

Speed is shaped with log relative speedup and only counts once correctness is
achieved, so the policy can never trade correctness for speed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from kore.config import CONFIG


@dataclass
class Observation:
    compiled: bool
    snr_db: Optional[float] = None
    wall_ms: Optional[float] = None
    baseline_ms: Optional[float] = None
    wall_by_shape: dict[str, float] = field(default_factory=dict)
    baseline_by_shape: dict[str, float] = field(default_factory=dict)
    snr_by_shape: dict[str, float] = field(default_factory=dict)
    validation_passed: bool = False
    error_text: Optional[str] = None
    registers: Optional[int] = None
    occupancy: Optional[float] = None
    dtype: str = "fp32"
    cv_pct: Optional[float] = None
    flagged_hack: bool = False
    hack_reason: Optional[str] = None


# Patterns that indicate the "kernel" is cheating rather than computing.
_HACK_PATTERNS = [
    (r"\bimport\s+aiter\b", "imports aiter (production baseline) instead of computing"),
    (r"\baiter\.", "calls aiter op instead of computing"),
    (r"\bimport\s+rocblas\b|\bhipblaslt\b|\brocblas\b", "calls a vendor BLAS instead of computing"),
    (r"torch\.(matmul|mm|bmm|nn\.functional|softmax|rms_norm|layer_norm)\s*\(", "delegates to torch instead of a kernel"),
    (r"F\.(scaled_dot_product_attention|linear|softmax|rms_norm|layer_norm)\s*\(", "delegates to torch.nn.functional"),
]
_SILENT_FALLBACK = re.compile(r"except\s*[\w. ,()]*:\s*(?:\n\s*)*(?:return|pass|out\s*=)", re.MULTILINE)


def scan_for_hacks(source: str) -> Optional[str]:
    """Return a reason string if the source looks like a reward hack, else None.

    Comments and docstrings are stripped first so legitimate references (e.g. a
    comment "matches aiter's layout") do not trip the scanner.
    """
    code = _strip_comments_and_docstrings(source)
    for pat, reason in _HACK_PATTERNS:
        if re.search(pat, code):
            return reason
    if _SILENT_FALLBACK.search(code):
        return "silent except-fallback that can mask incorrect results"
    return None


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove triple-quoted strings and ``#`` comments while preserving code
    spacing (so patterns like ``torch.matmul`` stay intact for scanning)."""
    src = re.sub(r'"""[\s\S]*?"""', " ", src)
    src = re.sub(r"'''[\s\S]*?'''", " ", src)
    src = re.sub(r"#.*", "", src)
    return src


def _worst_speedup(obs: Observation) -> Optional[float]:
    """Speedup on the worst shape: min over shapes of baseline/candidate."""
    if obs.baseline_by_shape and obs.wall_by_shape:
        ratios = []
        for k, cand in obs.wall_by_shape.items():
            base = obs.baseline_by_shape.get(k)
            if base and cand and cand > 0:
                ratios.append(base / cand)
        if ratios:
            return min(ratios)
    if obs.baseline_ms and obs.wall_ms and obs.wall_ms > 0:
        return obs.baseline_ms / obs.wall_ms
    return None


@dataclass
class RewardResult:
    reward: float
    correct: bool
    speedup: Optional[float]
    tier: str
    flags: list[str] = field(default_factory=list)
    detail: str = ""


def _all_shapes_pass(obs: Observation, dtype: str, cfg) -> bool:
    thr = cfg.snr_threshold_for(dtype)
    if obs.snr_by_shape:
        return all(v is not None and v >= thr for v in obs.snr_by_shape.values())
    return obs.snr_db is not None and obs.snr_db >= thr


def compute_reward(obs: Observation, source: str = "", dtype: str = "fp32",
                   mode: str = "eval", cfg=CONFIG) -> RewardResult:
    """Lexicographic, anti-hackable reward. Returns a :class:`RewardResult`.

    ``mode`` = "train" | "eval": eval never awards positive reward to a flagged
    hack; train hard-penalizes it (same here, kept explicit for clarity).
    """
    flags: list[str] = []

    # Tier 0: anti-hack scan (a hack that "passes" must never be rewarded).
    hack = obs.hack_reason or (scan_for_hacks(source) if source else None)
    if hack:
        flags.append("hack")
        return RewardResult(cfg.reward_compile_fail, False, None, "hack", flags, str(hack))

    # Tier 1: compile
    if not obs.compiled:
        flags.append("compile_fail")
        return RewardResult(cfg.reward_compile_fail, False, None, "compile_fail", flags,
                            obs.error_text or "did not compile")

    # Tier 2: correctness (validation + SNR gate on all shapes)
    correct = obs.validation_passed and _all_shapes_pass(obs, dtype, cfg)
    if not correct:
        flags.append("incorrect")
        return RewardResult(cfg.reward_incorrect, False, None, "incorrect", flags,
                            obs.error_text or "failed correctness/SNR")

    # Tier 3: speed (only once correct)
    base = cfg.correctness_weight
    su = _worst_speedup(obs)
    if su is None:
        return RewardResult(base, True, None, "correct_no_bench", flags, "correct; no timing")

    if su >= cfg.excessive_speedup_flag:
        flags.append("excessive_speedup")  # likely measurement error; cap shaping
        su_shaped = cfg.excessive_speedup_flag
    else:
        su_shaped = su
    reward = base + math.log(max(su_shaped, 1e-3))
    return RewardResult(reward, True, su, "correct_timed", flags,
                        f"worst-shape speedup {su:.3f}x vs baseline")
