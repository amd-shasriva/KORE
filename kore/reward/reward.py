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
from kore.obs import get_logger

_LOG = get_logger("reward")


def _log_decision(rr: "RewardResult") -> None:
    """Emit the reward *decision* as a structured event (JSONL always).

    Per-candidate detail rides at DEBUG so it never spams INFO; a flagged hack
    is surfaced at INFO (event) + WARN (reason) so cheating is impossible to
    miss. This is additive — it never touches the value being returned. NB: we
    deliberately do NOT log inside ``scan_for_hacks`` (the hot regex path);
    only the final decision is recorded here.
    """
    level = "INFO" if rr.tier == "hack" else "DEBUG"
    _LOG._emit(level, "reward", {
        "tier": rr.tier,
        "reward": round(rr.reward, 4),
        "correct": rr.correct,
        "speedup": (round(rr.speedup, 4) if rr.speedup is not None else None),
        "flags": list(rr.flags),
        "detail": rr.detail,
    }, kind="event")
    if rr.tier == "hack":
        _LOG.warn("reward-hack flagged", reason=rr.detail, flags=list(rr.flags))


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
    infra_error: bool = False   # timeout/OOM/segfault/import — NOT a kernel signal


# Patterns that indicate the "kernel" is cheating rather than computing.
_HACK_PATTERNS = [
    (r"\b(?:import|from)\s+aiter\b", "imports aiter (production baseline) instead of computing"),
    (r"\baiter\.", "calls aiter op instead of computing"),
    (r"\bimport\s+rocblas\b|\bhipblaslt\b|\brocblas\b|\bmiopen\b|\brocsolver\b|\bhipblas\b",
     "calls a vendor library instead of computing"),
    (r"torch\.(matmul|mm|bmm|addmm|einsum|softmax|rms_norm|layer_norm|scaled_dot_product_attention)\s*\(",
     "delegates to a torch op instead of a kernel"),
    (r"torch\.nn\.functional\.\w+\s*\(", "delegates to torch.nn.functional"),
    (r"\bF\.(scaled_dot_product_attention|linear|softmax|rms_norm|layer_norm|gelu|silu|conv\w*)\s*\(",
     "delegates to torch.nn.functional"),
    # aliased delegation via a handle: `import torch as t; t.matmul(...)`, `fn.gelu(...)`,
    # `x.softmax(...)`. Excludes the Triton language namespaces (`tl.`/`triton.`),
    # which legitimately provide tl.sigmoid/tl.exp/etc.
    (r"\b(?!tl\.)(?!triton\.)[A-Za-z_]\w*\.(?:matmul|addmm|baddbmm|einsum|"
     r"scaled_dot_product_attention|softmax|log_softmax|gelu|silu|elu|sigmoid|"
     r"layer_norm|rms_norm|group_norm|batch_norm|linear|conv\w*)\s*\(",
     "delegates an op via a handle instead of computing"),
    (r"\b(?!tl\.)(?!triton\.)[A-Za-z_]\w*\.(?:bmm|mm)\s*\(", "delegates a matmul via an aliased handle"),
    (r"\bfrom\s+torch(?:\.\w+)*\s+import\b", "imports torch ops directly (delegation channel)"),
    # aliasing the torch import (`import torch as t`, `import torch.nn.functional as fn`).
    (r"\bimport\s+torch(?:\.\w+)*\s+as\s+\w+", "aliases the torch import (delegation channel)"),
    # binding a torch op for later delegation (`m = torch.matmul; m(a,b)`).
    (r"=\s*torch\.(?:matmul|mm|bmm|addmm|baddbmm|einsum|softmax|log_softmax|gelu|silu|"
     r"layer_norm|rms_norm|scaled_dot_product_attention|linear|conv\w*)\b",
     "binds a torch op for later delegation"),
    # dynamic attribute lookup on a numeric lib to dodge literal-name scans.
    (r"getattr\s*\(\s*(?:torch|np|numpy|F|nn|aiter|__builtins__|builtins)\b",
     "dynamic getattr on a numeric library (delegation/escape channel)"),
    # post-verdict forgery channels: code that runs AFTER the driver prints its
    # verdict (atexit/signal/__del__/excepthook) can beat the last-match parse.
    (r"\batexit\b|signal\.signal|\bfaulthandler\b|sys\.excepthook|def\s+__del__\b",
     "registers a shutdown/exit/signal hook (post-verdict forgery channel)"),
    (r"\.(flash_attn\w*|fused_moe|paged_attention)\s*\(", "calls a fused vendor kernel instead of computing"),
    # copy-reference: returning the oracle's output passes the SNR gate, so it
    # MUST be rejected statically (runtime correctness can never catch it).
    (r"\b(?:import\s+reference|from\s+reference\s+import)\b", "imports the reference oracle"),
    (r"\bfrom\s+[\w.]*\breference\b\s+import\b", "imports the reference oracle (dotted path)"),
    (r"\b(?:reference|ref_program|ref_impl|matmul_ref|\w+_oracle|oracle)\s*\(",
     "calls the reference oracle instead of computing the result"),
    # accessing the KORE package (to import the task's oracle) from a kernel.
    (r"\b(?:import\s+kore\b|from\s+kore\b|kore\.tasks)", "imports the KORE package to reach the oracle"),
    # dynamic import / code exec — an escape hatch to reach vendor libs / the oracle.
    (r"\bimportlib\b|__import__\s*\(|\bexec\s*\(|\beval\s*\(", "uses dynamic import/exec to escape"),
    (r"\bctypes\b|\bcffi\b|\bCDLL\b|dlopen|LoadLibrary", "loads a native lib via ctypes/cffi"),
    # forging the verifier verdict on stdout.
    (r"(?:SNR|allclose|median_ms)\s*:", "prints a forged verifier verdict line"),
    # process/thread/file escape (fork-bomb, background verdict-overwrite, fs escape).
    (r"\bsubprocess\b|\bmultiprocessing\b|\bthreading\b|os\.system|os\.popen|os\.fork",
     "spawns processes/threads (isolation escape)"),
    (r"open\s*\([^)]*['\"][waxr]?[wax]\+?['\"]", "opens a file for writing (filesystem escape)"),
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


def _all_shapes_pass(obs: Observation, dtype: str, cfg, snr_threshold: Optional[float] = None) -> bool:
    thr = snr_threshold if snr_threshold is not None else cfg.snr_threshold_for(dtype)
    if obs.snr_by_shape:
        return all(v is not None and v >= thr for v in obs.snr_by_shape.values())
    return obs.snr_db is not None and obs.snr_db >= thr


def compute_reward(obs: Observation, source: str = "", dtype: str = "fp32",
                   mode: str = "eval", cfg=CONFIG,
                   snr_threshold: Optional[float] = None) -> RewardResult:
    """Lexicographic, anti-hackable reward. Returns a :class:`RewardResult`.

    Tier order (a strictly better outcome in an earlier tier ALWAYS dominates):
        hack/compile_fail (<0) < incorrect (0) < correct-but-slow < correct-fast.
    Correctness is scored with the Kevin reward ``correctness_weight + speedup``
    (linear, capped) — NOT log — so a correct kernel is *never* punished below an
    incorrect one, even when slower than the production baseline.

    ``snr_threshold`` overrides the dtype default (honors per-task task.yaml).
    """
    flags: list[str] = []

    # Tier -1: infrastructure error (timeout/OOM/segfault/import) — not the
    # kernel's fault; caller must NOT cache it and should resample.
    if obs.infra_error:
        flags.append("infra")
        rr = RewardResult(cfg.reward_incorrect, False, None, "infra", flags,
                          obs.error_text or "infrastructure error")
        _log_decision(rr)
        return rr

    # Tier 0: anti-hack scan (a hack that "passes" must never be rewarded).
    hack = obs.hack_reason or (scan_for_hacks(source) if source else None)
    if hack:
        flags.append("hack")
        rr = RewardResult(cfg.reward_compile_fail, False, None, "hack", flags, str(hack))
        _log_decision(rr)
        return rr

    # Tier 1: compile
    if not obs.compiled:
        flags.append("compile_fail")
        rr = RewardResult(cfg.reward_compile_fail, False, None, "compile_fail", flags,
                          obs.error_text or "did not compile")
        _log_decision(rr)
        return rr

    # Tier 2: correctness (validation + SNR gate on all shapes)
    correct = obs.validation_passed and _all_shapes_pass(obs, dtype, cfg, snr_threshold)
    if not correct:
        flags.append("incorrect")
        rr = RewardResult(cfg.reward_incorrect, False, None, "incorrect", flags,
                          obs.error_text or "failed correctness/SNR")
        _log_decision(rr)
        return rr

    # Tier 3: speed (only once correct). Kevin reward: base + linear speedup,
    # capped to bound measurement-error outliers. base>0 guarantees every correct
    # kernel (even a slow one, speedup>0) strictly beats the incorrect tier (0).
    base = cfg.correctness_weight
    su = _worst_speedup(obs)
    if su is None:
        rr = RewardResult(base, True, None, "correct_no_bench", flags, "correct; no timing")
        _log_decision(rr)
        return rr

    su_scored = su
    if su >= cfg.excessive_speedup_flag:
        flags.append("excessive_speedup")  # likely measurement error; cap contribution
        su_scored = cfg.excessive_speedup_flag
    if obs.cv_pct is not None and obs.cv_pct > cfg.cv_threshold_pct:
        flags.append("high_variance")  # noisy timing; keep correctness credit, damp speed
        su_scored = min(su_scored, 1.0)
    reward = base + max(su_scored, 0.0)
    rr = RewardResult(reward, True, su, "correct_timed", flags,
                      f"worst-shape speedup {su:.3f}x vs baseline")
    _log_decision(rr)
    return rr
