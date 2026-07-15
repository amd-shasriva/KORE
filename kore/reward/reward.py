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
    miss. This is additive - it never touches the value being returned. NB: we
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
    dtype: str = "fp32"
    cv_pct: Optional[float] = None
    flagged_hack: bool = False
    hack_reason: Optional[str] = None
    infra_error: bool = False   # timeout/OOM/segfault/import - NOT a kernel signal
    # P5: baseline-relative hardware-counter efficiency in [0,1] (rocprofv3), or
    # None when profiling is off/unavailable. Consumed as a bounded dense bonus.
    profile_efficiency: Optional[float] = None


# Patterns that indicate the "kernel" is cheating rather than computing.
_HACK_PATTERNS = [
    (r"\b(?:import|from)\s+aiter\b", "imports aiter (production baseline) instead of computing"),
    (r"\baiter\.", "calls aiter op instead of computing"),
    (r"\bimport\s+rocblas\b|\bhipblaslt\b|\brocblas\b|\bmiopen\b|\brocsolver\b|\bhipblas\b",
     "calls a vendor library instead of computing"),
    (r"torch\.(matmul|mm|bmm|addmm|baddbmm|einsum|softmax|rms_norm|layer_norm|scaled_dot_product_attention)\s*\(",
     "delegates to a torch op instead of a kernel"),
    # extended matmul family (all contract to a vendor GEMM/BLAS call).
    (r"torch\.(tensordot|chain_matmul|mv|inner|vdot|dot|kron)\s*\(",
     "delegates to a torch matmul-family op instead of computing"),
    (r"torch\.linalg\.(multi_dot|matmul)\s*\(", "delegates to torch.linalg matmul"),
    (r"\b(?!tl\.)(?!triton\.)[A-Za-z_]\w*\.(?:tensordot|chain_matmul|mv|inner|vdot|kron)\s*\(",
     "delegates a matmul-family op via a handle"),
    # augmented matmul assignment `c @= b` (the non-augmented form is caught below).
    (r"[\w\)\]][^\S\n]*@=", "uses the @= matmul-assign operator (delegates to the vendor GEMM)"),
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
    # importing the sibling driver/reference modules (which re-export ref_fn/baseline_fn
    # /the vendor kernel) is a copy-reference delegation hack: the candidate returns the
    # oracle's own output and passes correctness with SNR~=inf (audit R2 reverify).
    (r"\b(?:import|from)\s+(?:driver|reference)\b",
     "imports the sibling driver/reference module to reach the oracle/baseline"),
    (r"\b(?:ref_fn|baseline_fn|matmul_ref)\s*\(",
     "calls the reference oracle / vendor baseline function instead of computing"),
    # dynamic import / code exec - an escape hatch to reach vendor libs / the oracle.
    (r"\bimportlib\b|__import__\s*\(|\bexec\s*\(|\beval\s*\(", "uses dynamic import/exec to escape"),
    (r"\bctypes\b|\bcffi\b|\bCDLL\b|dlopen|LoadLibrary", "loads a native lib via ctypes/cffi"),
    # forging the verifier verdict on stdout (incl. the bench timing line).
    (r"(?:SNR|allclose|median_ms|wall_ms)\s*:", "prints a forged verifier verdict line"),
    # MODE-SNIFFING: the driver runs the SAME kernel for correctness (--impl ...)
    # and timing (--bench-mode); a kernel that inspects argv / the bench flags can
    # compute correctly when checked and skip work when timed (fake speedup). A
    # kernel has no legitimate reason to read the driver's CLI.
    (r"\bsys\.argv\b|\bargparse\b|\bgetopt\b",
     "reads the driver CLI (mode-sniffing: cheat the bench-vs-correctness split)"),
    (r"['\"]--?(?:bench[-_]?mode|impl|warmup|iters|reference|candidate)['\"]",
     "references the driver's benchmark flags (mode-sniffing channel)"),
    # TIMING MANIPULATION: sleeping / stalling only warmup, or busy-loop skew.
    (r"\btime\.sleep\s*\(|\basyncio\.sleep\s*\(", "calls sleep (benchmark timing manipulation)"),
    # tampering with GPU synchronization so the timed region under-measures.
    (r"set_sync_debug_mode|cudaProfilerStart|hipDeviceSetLimit",
     "tampers with GPU sync/profiling state (timing manipulation)"),
    # process/thread/file escape (fork-bomb, background verdict-overwrite, fs escape).
    (r"\bsubprocess\b|\bmultiprocessing\b|\bthreading\b|os\.system|os\.popen|os\.fork",
     "spawns processes/threads (isolation escape)"),
    (r"open\s*\([^)]*['\"][waxr]?[wax]\+?['\"]", "opens a file for writing (filesystem escape)"),
    # filesystem escape beyond open(): pathlib write, chmod (defeat 0o444 staging),
    # process spawn.
    (r"\.write_text\s*\(|\.write_bytes\s*\(", "writes a file via pathlib (filesystem escape)"),
    (r"\bos\.(chmod|replace|rename|remove|unlink|spawn\w*|posix_spawn)\b",
     "mutates the filesystem / spawns a process (isolation escape)"),
    # matmul OPERATOR delegation: `return a @ b` lowers to aten::matmul -> hipBLASLt
    # (pure vendor delegation). `@decorator` lines start with @ (no operand before),
    # so requiring an operand char before @ excludes decorators.
    # NB: horizontal-whitespace only ([^\S\n]) so a decorator stack (`)\n@triton.jit`
    # / `tl\n@triton.jit`) is NOT matched - only an operand `@` operand on ONE line.
    (r"[\w\)\]][^\S\n]*@[^\S\n]*[\w\(]", "uses the @ matmul operator (delegates to the vendor GEMM)"),
    # module-table access to reach torch/vendor/oracle while dodging import scans.
    (r"\bsys\.modules\b", "reaches libraries via sys.modules (delegation/escape channel)"),
    # reading the environment: a mode-sniff / escape channel a pure kernel never needs.
    (r"\bos\.environ\b|\bos\.getenv\b|\bgetenv\s*\(", "reads the environment (mode-sniff/escape channel)"),
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


def _shape_ratios(obs: Observation) -> list[float]:
    """Per-shape speedup ratios base_ms/cand_ms (a gain; higher is better)."""
    out: list[float] = []
    if obs.baseline_by_shape and obs.wall_by_shape:
        for k, cand in obs.wall_by_shape.items():
            base = obs.baseline_by_shape.get(k)
            if base and cand and cand > 0:
                out.append(base / cand)
    return out


def _worst_speedup(obs: Observation) -> Optional[float]:
    """Speedup on the worst shape: min over shapes of baseline/candidate.

    This is the diagnostic + eval metric (always worst-shape) and the CVaR_{a->0}
    endpoint of :func:`_aggregate_speedup`."""
    ratios = _shape_ratios(obs)
    if ratios:
        return min(ratios)
    if obs.baseline_ms and obs.wall_ms and obs.wall_ms > 0:
        return obs.baseline_ms / obs.wall_ms
    return None


def _aggregate_speedup(obs: Observation, cfg) -> Optional[float]:
    """Distributionally-robust speed aggregation over the per-shape speedup sweep.

    KORE's contribution is a *distributionally-robust* speed objective against the
    PRODUCTION vendor baseline: rather than the average-case speedup, it optimizes
    the worst shapes, so the policy must be fast on the hardest shape a practitioner
    hits - not just on average. This exposes the whole CVaR_alpha family (worst =
    CVaR_{a->0}, mean = CVaR_1) at a single point; all downstream shaping (log term,
    fast_p bonuses, significance) then applies to the chosen aggregate.

      "worst" : min over shapes (current behavior; the robust objective / default).
      "cvar"  : geometric mean of the worst ceil(alpha*N) shapes (CVaR_alpha).
      "mean"  : geometric-mean speedup over all shapes (average-case ablation arm).

    Geometric mean (mean-of-logs) is used for cvar/mean so the family is linear in
    ln(ratio) - consistent with the log-speedup shaping - and scale-correct for
    ratios. Degrades to the single-shape / scalar case identically to _worst_speedup,
    so the default ("worst") is byte-identical to the previous reward.
    """
    ratios = _shape_ratios(obs)
    if not ratios:
        if obs.baseline_ms and obs.wall_ms and obs.wall_ms > 0:
            return obs.baseline_ms / obs.wall_ms
        return None
    mode = (getattr(cfg, "speed_aggregation", "worst") or "worst").lower()
    n = len(ratios)
    if mode == "worst" or n == 1:
        return min(ratios)                       # CVaR_{alpha->0}
    if mode == "mean":
        k = n
    else:  # "cvar"
        alpha = float(getattr(cfg, "cvar_alpha", 0.5) or 0.5)
        k = max(1, min(n, math.ceil(alpha * n)))
    worst_logs = sorted(math.log(r) for r in ratios)[:k]  # k worst (smallest) ratios
    return math.exp(sum(worst_logs) / k)


def _worst_snr(obs: Observation) -> Optional[float]:
    """Worst-shape SNR (min over shapes), falling back to the primary ``snr_db``.

    Mirrors the correctness gate, which is also scored on the WORST shape, so the
    sub-threshold credit reflects the same "hardest shape" the gate cares about.
    """
    if obs.snr_by_shape:
        vals = [v for v in obs.snr_by_shape.values() if v is not None]
        if vals:
            return min(vals)
    return obs.snr_db


def _subthreshold_credit(obs: Observation, dtype: str, cfg,
                         snr_threshold: Optional[float]) -> float:
    """P1: bounded, continuous credit for a compiled-but-INCORRECT kernel.

    Returns ``eps_shape * clamp(worst_snr / snr_threshold, 0, 1)`` - a dense
    signal proportional to progress toward the correctness gate, so early RL
    isn't stuck on a flat-zero reward. The value lies in ``[0, eps_shape]`` and,
    because a kernel in the incorrect tier has worst-shape SNR *below* the gate,
    it is in practice strictly ``< eps_shape < correctness_weight`` - it can
    never reach, let alone cross, the correct tier. Returns 0 when shaping is
    off, when there is no SNR signal, or (by construction of the caller) for a
    flagged hack / compile-fail / infra error.
    """
    if not getattr(cfg, "subthreshold_shaping", False):
        return 0.0
    eps = float(getattr(cfg, "eps_shape", 0.0) or 0.0)
    if eps <= 0.0:
        return 0.0
    thr = snr_threshold if snr_threshold is not None else cfg.snr_threshold_for(dtype)
    snr = _worst_snr(obs)
    if snr is None or thr is None or thr <= 0.0:
        return 0.0
    progress = snr / thr
    progress = 0.0 if progress < 0.0 else (1.0 if progress > 1.0 else progress)
    return eps * progress


def _format_component(response: Optional[str], cfg) -> float:
    """P2: bounded format-compliance term for the incorrect/correct tiers.

    ``response`` is the RAW policy output (the FULL_KERNEL contract), NOT the
    already-extracted kernel. Returns ``+format_weight`` when the response parses
    to a non-empty kernel (valid contract), ``-format_weight`` when it is
    malformed, and 0 when no response is supplied (the default - preserves the
    exact legacy reward for every current caller). The magnitude is kept far
    below every inter-tier gap, so this term can never flip tier ordering.
    """
    if response is None:
        return 0.0
    w = float(getattr(cfg, "format_weight", 0.0) or 0.0)
    if w <= 0.0:
        return 0.0
    # Lazy import to avoid any import cycle and to keep the hot path dependency-free.
    from kore.policy.format import parse_response
    kernel = (parse_response(response).get("kernel") or "").strip()
    return w if kernel else -w


@dataclass
class RewardResult:
    reward: float
    correct: bool
    speedup: Optional[float]
    tier: str
    flags: list[str] = field(default_factory=list)
    detail: str = ""


def _speedup_term(su_scored: float, su_raw: float, obs: Observation, cfg,
                  flags: list[str]) -> float:
    """P4 speed reward: log-shaped speedup + significance-gated fast_p bonuses.

    ``su_scored`` is the (excessive-capped / high-variance-damped) speedup used for
    the continuous term; ``su_raw`` is the measured speedup used for the discrete
    threshold checks. Returns a NON-NEGATIVE speed contribution, so a correct
    kernel always scores >= ``correctness_weight`` (lexicographic dominance holds).

    Continuous term (breaks the linear plateau, steeper at the 1x crossover):
        speedup_log=True  ->  w*su           (su <= 1, linear, non-negative)
                              w*(1 + ln(su))  (su >  1, emphasized)
        speedup_log=False ->  w*max(su, 0)    (legacy linear)
    Discrete term (the strong ">1x" signal): cumulative ``fast_p_bonus`` for each
    threshold met, awarded ONLY when the speedup is statistically trustworthy
    (cv <= cv_threshold_pct) and not an excessive-speedup measurement outlier.
    """
    w = float(getattr(cfg, "speedup_weight", 1.0) or 0.0)
    if getattr(cfg, "speedup_log", False) and su_scored > 1.0:
        term = w * (1.0 + math.log(su_scored))
    else:
        term = w * max(su_scored, 0.0)

    bonuses = getattr(cfg, "fast_p_bonus", ()) or ()
    if bonuses:
        sig_only = bool(getattr(cfg, "fast_p_significant_only", True))
        trustworthy = (obs.cv_pct is None) or (obs.cv_pct <= cfg.cv_threshold_pct)
        excessive = "excessive_speedup" in flags
        # Require the speedup to clear the threshold by the measurement noise floor
        # (not just tie it): a kernel that merely PARITIES the baseline (1.00x) - or
        # beats it only within combined timing noise - must not farm the crossover
        # bonus. margin = 1 + noise_floor_pct/100 (e.g. 1.0x threshold -> need 1.02x).
        margin = 1.0 + float(getattr(cfg, "noise_floor_pct", 0.0) or 0.0) / 100.0
        if (not sig_only) or (trustworthy and not excessive):
            for thr, bonus in bonuses:
                if su_raw >= thr * margin:
                    term += float(bonus)
                    flags.append(f"fast_p>={thr}")
    return term


def _all_shapes_pass(obs: Observation, dtype: str, cfg, snr_threshold: Optional[float] = None) -> bool:
    thr = snr_threshold if snr_threshold is not None else cfg.snr_threshold_for(dtype)
    if obs.snr_by_shape:
        return all(v is not None and v >= thr for v in obs.snr_by_shape.values())
    return obs.snr_db is not None and obs.snr_db >= thr


def compute_reward(obs: Observation, source: str = "", dtype: str = "fp32",
                   mode: str = "eval", cfg=CONFIG,
                   snr_threshold: Optional[float] = None,
                   phase: Optional[str] = None,
                   response: Optional[str] = None) -> RewardResult:
    """Lexicographic, anti-hackable reward. Returns a :class:`RewardResult`.

    Tier order (a strictly better outcome in an earlier tier ALWAYS dominates):
        hack < compile_fail < incorrect (shaped) < correct-but-slow < correct-fast.
    Correctness is scored with the Kevin reward ``correctness_weight + speedup``
    (linear, capped) - NOT log - so a correct kernel is *never* punished below an
    incorrect one, even when slower than the production baseline.

    Shaping upgrades (all bounded so lexicographic dominance holds absolutely):
      * P1 sub-threshold shaping - a compiled-but-incorrect kernel gets a small
        continuous credit in ``[0, eps_shape]`` toward the correctness gate,
        never enough to reach the correct tier. Never applied to a hack/compile
        failure/infra error.
      * P2 format term - pass the raw ``response`` (FULL_KERNEL contract) to add
        a tiny ``±format_weight`` bonus/penalty on the incorrect/correct tiers.
      * P3 curriculum ``phase`` - ``"correctness"`` zeroes the speed term (every
        correct kernel scores ``correctness_weight``); ``"full"``/``"latency"``
        (default) use ``correctness_weight + speedup``. Falls back to
        ``cfg.reward_phase`` when ``phase`` is None.

    ``snr_threshold`` overrides the dtype default (honors per-task task.yaml).
    """
    flags: list[str] = []
    phase = (phase or getattr(cfg, "reward_phase", "full") or "full").lower()

    # Tier -1: infrastructure error (timeout/OOM/segfault/import) - not the
    # kernel's fault; caller must NOT cache it and should resample.
    if obs.infra_error:
        flags.append("infra")
        rr = RewardResult(cfg.reward_incorrect, False, None, "infra", flags,
                          obs.error_text or "infrastructure error")
        _log_decision(rr)
        return rr

    # Tier 0: anti-hack scan (a hack that "passes" must never be rewarded).
    # Punished STRICTLY harder than a compile failure (reward_hack < reward_compile_fail)
    # and never eligible for any shaping/format credit: cheating is the unique floor.
    hack = obs.hack_reason or (scan_for_hacks(source) if source else None)
    if hack:
        flags.append("hack")
        rr = RewardResult(cfg.reward_hack, False, None, "hack", flags, str(hack))
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
        # P1 sub-threshold shaping + P2 format term. Bounded so that
        #   max shaped-incorrect = eps_shape + format_weight < correctness_weight,
        # i.e. a shaped-incorrect kernel can never reach the correct tier.
        credit = _subthreshold_credit(obs, dtype, cfg, snr_threshold)
        fmt = _format_component(response, cfg)
        if credit > 0.0:
            flags.append("shaped")
        reward = cfg.reward_incorrect + credit + fmt
        detail = obs.error_text or "failed correctness/SNR"
        if credit > 0.0 or fmt:
            detail += f" (shaped +{credit:.4f}, format {fmt:+.4f})"
        rr = RewardResult(reward, False, None, "incorrect", flags, detail)
        _log_decision(rr)
        return rr

    # Tier 3: speed (only once correct). Kevin reward: base + linear speedup,
    # capped to bound measurement-error outliers. base>0 guarantees every correct
    # kernel (even a slow one, speedup>0) strictly beats the incorrect tier.
    # A correct kernel always parses to a kernel, so its format term is +format_weight
    # (never a penalty) - correct-fast vs correct-slow stays a pure speed ordering.
    base = cfg.correctness_weight
    fmt = _format_component(response, cfg)
    su = _aggregate_speedup(obs, cfg)  # distributionally-robust (default: worst-shape)
    if su is None:
        rr = RewardResult(base + fmt, True, None, "correct_no_bench", flags,
                          "correct; no timing")
        _log_decision(rr)
        return rr

    su_scored = su
    if su >= cfg.excessive_speedup_flag:
        flags.append("excessive_speedup")  # likely measurement error; cap contribution
        su_scored = cfg.excessive_speedup_flag
    if obs.cv_pct is not None and obs.cv_pct > cfg.cv_threshold_pct:
        flags.append("high_variance")  # noisy timing; keep correctness credit, damp speed
        su_scored = min(su_scored, 1.0)
    # P3 curriculum: the "correctness" phase zeroes the speed term so every
    # correct kernel scores exactly correctness_weight (+format); "full"/"latency"
    # keep the full correctness_weight + speedup.
    if phase == "correctness":
        flags.append("phase:correctness")
        speed_term = 0.0
    else:
        speed_term = _speedup_term(su_scored, su, obs, cfg, flags)
        # P5: bounded, baseline-relative hardware-counter dense bonus (flagship
        # novelty). Only on the correct tier; strictly below the fast_p bonuses so
        # real wall-clock wins always dominate. Inert when weight==0 / no profile.
        pw = float(getattr(cfg, "profile_reward_weight", 0.0) or 0.0)
        if pw > 0.0 and obs.profile_efficiency is not None:
            prof_term = pw * max(0.0, min(1.0, obs.profile_efficiency))
            speed_term += prof_term
            flags.append(f"profile+{prof_term:.3f}")
    reward = base + speed_term + fmt
    rr = RewardResult(reward, True, su, "correct_timed", flags,
                      f"worst-shape speedup {su:.3f}x vs baseline"
                      + (" [correctness phase: speed zeroed]" if phase == "correctness" else ""))
    _log_decision(rr)
    return rr
