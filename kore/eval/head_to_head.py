"""KORE-vs-Opus HEAD-TO-HEAD with PAIRED significance on per-task deltas.

The campaign's default eval reports KORE vs the frozen *seed* kernel. That answers
"did training improve on the starting point?" - NOT the flagship question: "does
the trained specialist beat the *frontier generalist* (Claude Opus 4.x)?" This
module answers the latter, rigorously, on the SAME held-out tasks.

Both sides are scored through the IDENTICAL path:

  * KORE is a ``PolicyFn`` (usually :func:`kore.eval.policies.model_policy` over the
    trained checkpoint);
  * Opus is a ``PolicyFn`` too, built by :func:`kore.eval.opus_policy.opus_policy`
    (``model_policy`` with the Opus/Claude teacher injected as the token source) -
    so it shares KORE's prompt contract and response parser;
  * both run through the SAME ``env_factory`` (:class:`~kore.env.kore_env.KoreEnv`:
    verified correctness oracle + cold-cache timing), the SAME matched ``budget``,
    and the SAME timing-INTEGRITY gate (``evaluate_policy`` uses
    ``_integrity_gated_speedup`` so neither side can farm the headline metric with a
    glitch or a noisy bench).

Because both sides are scored on the SAME tasks, the comparison is PAIRED. We take
the per-task KORE-minus-Opus speedup DELTA and run the existing paired battery
(:mod:`kore.eval.paired_stats`): a paired BOOTSTRAP confidence interval + two-sided
bootstrap p, the exact SIGN test, and the WILCOXON signed-rank test. Paired tests
cancel the huge task-to-task difficulty variance, so they are far more powerful than
comparing two independent fast_p numbers. We report TWO effect sizes, both via
``paired_stats`` (nothing reinvented):

  * a MEAN per-task speedup delta over ALL held-out tasks (a non-competing side
    contributes speedup 0, exactly like fast_p's uncorrected denominator), with a
    bootstrap 95% CI + sign + Wilcoxon p-values;
  * a geometric-mean speedup RATIO ("KORE is X times faster than Opus") over the
    tasks BOTH sides solved correctly (the multiplicatively-correct effect size for
    speedups), with an exponentiated bootstrap CI.

Plus the per-side fast_p curves (reused from :func:`kore.eval.bakeoff.evaluate_policy`
/ :mod:`kore.eval.fastp`) and their per-threshold delta, a win/loss/tie tally, a
JSON (+ markdown) report, and a one-line verdict.

Graceful degradation (no API key): if no Opus ``PolicyFn`` is supplied and one
cannot be provisioned (no ``anthropic`` SDK, missing ``AMD_LLM_API_KEY``, gateway
down - or a sustained outage mid-run), the Opus side is SKIPPED with a loud warning
and the KORE-only fast_p is still returned (``opus_skipped=True``). This never
crashes the caller because the gateway is down.

Import-safe / offline: nothing heavy is imported at module load; torch/vLLM are
reached only via the injected KORE ``generate`` and the Anthropic SDK only via the
lazily-built teacher. The whole comparison (deltas, fast_p, paired stats, report) is
CPU-unit-tested with a mock Opus policy + fabricated ``Observation`` measurements
(no GPU, no network - see ``tests/test_opus_head_to_head.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Sequence

from kore.config import CONFIG, KoreConfig
from kore.eval.bakeoff import evaluate_policy
from kore.eval.fastp import DEFAULT_PS
from kore.eval.opus_policy import (
    DEFAULT_OPUS_KIND,
    _loud_warn,
    try_opus_policy,
)
from kore.eval.paired_stats import (
    format_paired_report,
    paired_comparison,
    paired_speedup_comparison,
)
from kore.eval.policies import PolicyFn
from kore.obs import get_logger

_LOG = get_logger("eval.head_to_head")


# --------------------------------------------------------------------------- #
# PURE per-task comparison logic (CPU-unit-tested; no GPU, no network).
# --------------------------------------------------------------------------- #
def _competing_speedup(rec: Optional[dict]) -> Optional[float]:
    """The integrity-gated best speedup a side brings to the comparison, or None.

    A side only competes with a CORRECT kernel; ``best_speedup`` is already the
    timing-INTEGRITY-gated value ``evaluate_policy`` ranks fast_p on (excessive
    ratios capped, high-variance benches damped to <=1x), so comparing the two
    sides' values is apples-to-apples and cannot be farmed by a glitch/noise.
    """
    if not rec or not rec.get("correct"):
        return None
    su = rec.get("best_speedup")
    return float(su) if su is not None else None


def _competing_value(rec: Optional[dict]) -> float:
    """Numeric competing speedup for the paired DELTA: the gated speedup, else 0.0.

    A non-competing (incorrect / unattempted) side contributes 0.0, mirroring
    fast_p's uncorrected denominator where such a task counts against the side.
    """
    su = _competing_speedup(rec)
    return float(su) if su is not None else 0.0


def _winner_for_task(kore_rec: Optional[dict], opus_rec: Optional[dict],
                     *, margin: float = 1.0) -> str:
    """Decide one matched task: ``"kore"`` | ``"opus"`` | ``"tie"`` | ``"both_incorrect"``.

    KORE wins iff its correct kernel is faster than Opus's correct kernel by at
    least the multiplicative ``margin`` (``margin=1.0`` = strictly faster). A
    correct side always beats a non-competing one; neither correct => both_incorrect
    (still counted in the denominator, penalizing both).
    """
    ks = _competing_speedup(kore_rec)
    os_ = _competing_speedup(opus_rec)
    if ks is None and os_ is None:
        return "both_incorrect"
    if os_ is None:
        return "kore"
    if ks is None:
        return "opus"
    m = float(margin) if margin and margin > 0 else 1.0
    if ks > os_ * m:
        return "kore"
    if os_ > ks * m:
        return "opus"
    return "tie"


def _fastp_at(res: dict, p: float) -> float:
    """Extract fast_p at threshold ``p`` from an ``evaluate_policy`` result."""
    fp = (res or {}).get("fast_p", {}) or {}
    if float(p) in fp:
        return float(fp[float(p)])
    if p in fp:
        return float(fp[p])
    return 0.0


def compare_per_task(kore_per_task: Sequence[dict], opus_per_task: Sequence[dict],
                     *, margin: float = 1.0) -> dict:
    """Match KORE and Opus per-task records and build the paired-delta inputs.

    Both inputs are ``evaluate_policy(...)["per_task"]`` lists (each carries
    ``task_id`` / ``correct`` / ``best_speedup``). Tasks are matched by ``task_id``;
    the denominator ``n`` is the KORE split size (uncorrected, like fast_p). PURE:
    no GPU, no network, no torch. Returns per-task rows, the KORE-minus-Opus speedup
    ``deltas`` over ALL tasks, the both-correct positive-speedup arrays for the
    geomean-ratio effect size, and a win/loss/tie tally.
    """
    opus_by_id = {r.get("task_id"): r for r in opus_per_task}
    rows: list[dict] = []
    deltas: list[float] = []
    kore_both: list[float] = []
    opus_both: list[float] = []
    tally = {"kore": 0, "opus": 0, "tie": 0, "both_incorrect": 0}

    for kr in kore_per_task:
        tid = kr.get("task_id")
        orr = opus_by_id.get(tid)
        ks = _competing_speedup(kr)
        os_ = _competing_speedup(orr)
        kv = _competing_value(kr)
        ov = _competing_value(orr)
        delta = kv - ov
        winner = _winner_for_task(kr, orr, margin=margin)
        tally[winner] += 1
        deltas.append(delta)
        # Both correct + strictly positive => eligible for the geomean speedup ratio.
        if ks is not None and os_ is not None and ks > 0.0 and os_ > 0.0:
            kore_both.append(ks)
            opus_both.append(os_)
        rows.append({
            "task_id": tid,
            "winner": winner,
            "kore_correct": bool(kr.get("correct")),
            "opus_correct": bool(orr.get("correct")) if orr else False,
            "kore_speedup": ks,
            "opus_speedup": os_,
            "delta": delta,
        })

    return {
        "n": len(kore_per_task),
        "margin": float(margin),
        "per_task": rows,
        "deltas": deltas,
        "kore_both_correct": kore_both,
        "opus_both_correct": opus_both,
        "n_both_correct": len(kore_both),
        "winners": tally,
    }


def _fastp_block(kore_res: dict, opus_res: dict, ps: Sequence[float]) -> dict:
    """Per-side fast_p at each threshold plus the KORE-minus-Opus per-threshold delta."""
    kore_fp = {float(p): _fastp_at(kore_res, p) for p in ps}
    opus_fp = {float(p): _fastp_at(opus_res, p) for p in ps}
    delta_fp = {float(p): kore_fp[float(p)] - opus_fp[float(p)] for p in ps}
    return {"kore": kore_fp, "opus": opus_fp, "delta": delta_fp}


def _verdict(paired_delta: dict, ratio: Optional[dict], winners: dict) -> str:
    """One-line human verdict from the paired-delta direction/significance + tally."""
    direction = paired_delta.get("direction", "tie")
    eff = paired_delta.get("effect_size", 0.0)
    p = paired_delta.get("p_value", 1.0)
    sig = "significant" if paired_delta.get("significant") else "not significant"
    side = {"kore_better": "KORE WINS", "baseline_better": "Opus WINS"}.get(direction, "TIE")
    ratio_txt = ""
    if ratio:
        ratio_txt = (f"; geomean {ratio.get('effect_size', 0.0):.3f}x on "
                     f"{ratio.get('n', 0)} both-correct tasks")
    return (f"{side} the head-to-head (mean per-task speedup delta {eff:+.3f}, "
            f"Wilcoxon p={p:.4g}, {sig}{ratio_txt}; "
            f"wins K/O/T/-={winners.get('kore', 0)}/{winners.get('opus', 0)}/"
            f"{winners.get('tie', 0)}/{winners.get('both_incorrect', 0)}).")


# --------------------------------------------------------------------------- #
# The head-to-head driver.
# --------------------------------------------------------------------------- #
def head_to_head_vs_opus(
    kore_policy: PolicyFn,
    tasks: Sequence,
    env_factory: Optional[Callable[[object], object]] = None,
    budget: int = 5,
    *,
    opus_policy: Optional[PolicyFn] = None,
    teacher=None,
    opus_kind: str = DEFAULT_OPUS_KIND,
    opus_model: Optional[str] = None,
    multi_turn: bool = True,
    mode: str = "serial",
    margin: float = 1.0,
    ps: Sequence[float] = DEFAULT_PS,
    cfg: KoreConfig = CONFIG,
    system_prompt: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    n_boot: int = 10000,
    ci_level: float = 0.95,
    seed: int = 0,
    out: Optional[object] = None,
    kore_dry_run: Optional[object] = None,
    opus_dry_run: Optional[object] = None,
    log=None,
) -> dict:
    """Run KORE and Opus on the SAME tasks and report paired significance + fast_p.

    ``kore_policy`` is any ``PolicyFn`` (e.g. ``model_policy(kore_ckpt)``). The Opus
    side is, in order: the explicit ``opus_policy`` if given; else built from
    ``teacher`` if given; else provisioned from ``opus_kind`` / ``opus_model`` via
    :func:`kore.eval.opus_policy.try_opus_policy`. Both sides are scored with
    :func:`kore.eval.bakeoff.evaluate_policy` under the SAME ``env_factory``,
    ``budget``, ``mode``, ``ps``, and ``cfg``.

    Returns a structured dict (see ``per_task`` / ``deltas`` / ``fast_p`` /
    ``paired_delta`` / ``paired_speedup_ratio`` / ``winners`` / ``verdict``) and, when
    ``out`` is given, writes ``<out>.json`` (+ ``<out>.md``) beside it. The paired
    battery is the existing :func:`kore.eval.paired_stats.paired_comparison` (mean
    delta + bootstrap CI + sign + Wilcoxon) on the KORE-minus-Opus per-task speedup
    deltas, plus :func:`kore.eval.paired_stats.paired_speedup_comparison` (geomean
    ratio) on the both-correct subset.

    Graceful degradation: if the Opus side is unavailable (no API key / SDK, gateway
    down, or a teacher outage mid-eval), the result has ``opus_skipped=True`` and a
    ``skip_reason``, the KORE-only fast_p is still present, and NOTHING crashes.

    CPU tests pass ``kore_dry_run`` / ``opus_dry_run`` (precomputed ``Observation``
    maps or callables, per :func:`evaluate_policy`) and a mock ``opus_policy`` so the
    whole path runs with no GPU and no network.
    """
    tasks = list(tasks)
    task_ids = [getattr(t, "task_id", t if isinstance(t, str) else str(t)) for t in tasks]

    out_dict: dict = {
        "n": len(tasks),
        "budget": budget,
        "mode": mode,
        "margin": float(margin),
        "seed": seed,
        "ps": [float(p) for p in ps],
        "tasks": task_ids,
        "opus_skipped": False,
        "skip_reason": None,
        "kore": None,
        "opus": None,
        "per_task": [],
        "deltas": [],
        "fast_p": None,
        "paired_delta": None,
        "paired_speedup_ratio": None,
        "winners": None,
        "n_both_correct": 0,
        "verdict": None,
    }

    # KORE side (always scored; its numbers survive an Opus skip).
    kore_res = evaluate_policy(kore_policy, tasks, env_factory=env_factory,
                               budget=budget, mode=mode, dry_run=kore_dry_run,
                               ps=ps, cfg=cfg)
    out_dict["kore"] = kore_res

    # Resolve the Opus side (explicit policy > teacher > provisioned), degrading
    # gracefully to a KORE-only report if it cannot be built.
    opus_pol = opus_policy
    if opus_pol is None:
        opus_pol, reason = try_opus_policy(
            teacher=teacher, kind=opus_kind, model=opus_model, multi_turn=multi_turn,
            system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature,
            log=log,
        )
        if opus_pol is None:
            out_dict["opus_skipped"] = True
            out_dict["skip_reason"] = reason or "Opus policy unavailable"
            _loud_warn(log, "head_to_head_vs_opus: Opus side SKIPPED; reporting "
                            "KORE-only fast_p (paired head-to-head not computed)",
                       reason=(reason or "")[:200])
            if out is not None:
                write_report(out_dict, out)
            return out_dict

    # Opus side (graceful): a teacher outage mid-eval (e.g. ResilientTeacher's
    # sustained-outage hard-stop) must SKIP the comparison, never crash the eval.
    try:
        opus_res = evaluate_policy(opus_pol, tasks, env_factory=env_factory,
                                   budget=budget, mode=mode, dry_run=opus_dry_run,
                                   ps=ps, cfg=cfg)
    except Exception as e:  # noqa: BLE001 - teacher/gateway failure is non-fatal here
        out_dict["opus_skipped"] = True
        out_dict["skip_reason"] = f"teacher failed during eval: {type(e).__name__}: {str(e)[:200]}"
        _loud_warn(log, "head_to_head_vs_opus: Opus side SKIPPED mid-eval "
                        "(teacher/gateway failure); reporting KORE-only fast_p",
                   exc_type=type(e).__name__, exc=str(e)[:200])
        if out is not None:
            write_report(out_dict, out)
        return out_dict

    out_dict["opus"] = opus_res

    # Paired per-task comparison + the existing paired-stats battery on the deltas.
    cmp = compare_per_task(kore_res.get("per_task", []), opus_res.get("per_task", []),
                           margin=margin)
    deltas = cmp["deltas"]
    paired_delta = paired_comparison(deltas=deltas, n_boot=n_boot, ci_level=ci_level,
                                     seed=seed).to_dict()

    # Geometric-mean speedup RATIO on the tasks BOTH sides solved (>=2 pairs needed
    # for a meaningful bootstrap; strictly-positive already guaranteed by the filter).
    ratio_dict: Optional[dict] = None
    if cmp["n_both_correct"] >= 2:
        ratio_dict = paired_speedup_comparison(
            cmp["kore_both_correct"], cmp["opus_both_correct"],
            n_boot=n_boot, ci_level=ci_level, seed=seed,
        ).to_dict()

    out_dict["per_task"] = cmp["per_task"]
    out_dict["deltas"] = deltas
    out_dict["fast_p"] = _fastp_block(kore_res, opus_res, ps)
    out_dict["paired_delta"] = paired_delta
    out_dict["paired_speedup_ratio"] = ratio_dict
    out_dict["winners"] = cmp["winners"]
    out_dict["n_both_correct"] = cmp["n_both_correct"]
    out_dict["verdict"] = _verdict(paired_delta, ratio_dict, cmp["winners"])

    if out is not None:
        write_report(out_dict, out)
    return out_dict


# --------------------------------------------------------------------------- #
# Reporting (PURE markdown; ASCII only, no em/en dashes) + JSON persistence.
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def format_head_to_head_report(res: dict) -> str:
    """Human-readable ASCII markdown for a :func:`head_to_head_vs_opus` result."""
    lines: list[str] = []
    lines.append("# KORE vs Opus head-to-head (paired, held-out kernels)")
    lines.append("")
    lines.append(f"- **split size (n)**: {res.get('n', '?')}")
    lines.append(f"- **budget** (max benches/task): {res.get('budget', '?')}")
    lines.append(f"- **mode**: {res.get('mode', '?')}")
    lines.append(f"- **win margin**: {res.get('margin', 1.0)}")
    lines.append("")

    kore = res.get("kore") or {}
    if res.get("opus_skipped"):
        lines.append(f"> **Opus side SKIPPED**: {res.get('skip_reason')}")
        lines.append("> Reporting KORE-only fast_p (the paired head-to-head was not computed).")
        lines.append("")
        lines.append("| p | KORE fast_p |")
        lines.append("| --- | --- |")
        kore_fp = kore.get("fast_p", {}) or {}
        for p in sorted(float(k) for k in kore_fp.keys()):
            lines.append(f"| {p:g} | {_pct(kore_fp.get(p, kore_fp.get(float(p), 0.0)))} |")
        lines.append("")
        return "\n".join(lines)

    fp = res.get("fast_p", {}) or {}
    kore_fp = fp.get("kore", {}) or {}
    opus_fp = fp.get("opus", {}) or {}
    delta_fp = fp.get("delta", {}) or {}
    lines.append("## fast_p (correct AND faster than the production baseline by >p)")
    lines.append("")
    lines.append("| p | KORE | Opus | KORE - Opus |")
    lines.append("| --- | --- | --- | --- |")
    for p in sorted(kore_fp.keys()):
        lines.append(f"| {float(p):g} | {_pct(kore_fp.get(p, 0.0))} "
                     f"| {_pct(opus_fp.get(p, 0.0))} | {_pct(delta_fp.get(p, 0.0))} |")
    lines.append("")

    pd = res.get("paired_delta") or {}
    ci = pd.get("ci", [0.0, 0.0])
    lines.append("## Paired significance (KORE - Opus per-task speedup delta)")
    lines.append("")
    lines.append(f"- **paired tasks (n)**: {pd.get('n', 0)}")
    lines.append(f"- **mean per-task delta**: {pd.get('effect_size', 0.0):+.4f}")
    lines.append(f"- **{int(pd.get('ci_level', 0.95) * 100)}% bootstrap CI**: "
                 f"[{ci[0]:+.4f}, {ci[1]:+.4f}] (null = 0)")
    lines.append(f"- **p-values**: bootstrap {pd.get('p_bootstrap', 1.0):.4g}, "
                 f"sign {pd.get('p_sign', 1.0):.4g}, Wilcoxon {pd.get('p_wilcoxon', 1.0):.4g}")
    lines.append(f"- **direction**: {pd.get('direction', 'tie')} "
                 f"(significant at alpha=0.05: {pd.get('significant', False)})")
    lines.append("")

    ratio = res.get("paired_speedup_ratio")
    if ratio:
        rci = ratio.get("ci", [1.0, 1.0])
        lines.append("## Geometric-mean speedup ratio (KORE / Opus, both-correct tasks)")
        lines.append("")
        lines.append(f"- **both-correct tasks (n)**: {ratio.get('n', 0)}")
        lines.append(f"- **KORE is {ratio.get('effect_size', 1.0):.4f}x "
                     "the speed of Opus** (geomean ratio)")
        lines.append(f"- **{int(ratio.get('ci_level', 0.95) * 100)}% CI**: "
                     f"[{rci[0]:.4f}x, {rci[1]:.4f}x] (null = 1.0)")
        lines.append(f"- **p-value (Wilcoxon)**: {ratio.get('p_wilcoxon', 1.0):.4g} "
                     f"(significant: {ratio.get('significant', False)})")
        lines.append("")

    w = res.get("winners") or {}
    lines.append("## Win / loss / tie (per-task, at the win margin)")
    lines.append("")
    lines.append(f"- **KORE wins**: {w.get('kore', 0)}")
    lines.append(f"- **Opus wins**: {w.get('opus', 0)}")
    lines.append(f"- **ties**: {w.get('tie', 0)}")
    lines.append(f"- **both incorrect**: {w.get('both_incorrect', 0)}")
    lines.append("")
    if res.get("verdict"):
        lines.append(f"**Verdict**: {res['verdict']}")
        lines.append("")
    return "\n".join(lines)


def _json_default(o):
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def write_report(res: dict, out) -> dict:
    """Persist a head-to-head ``res`` as ``<out>.json`` (+ ``<out>.md``).

    ``out`` may carry an extension (stripped) or be a stem. Returns the two paths.
    The JSON is the machine-readable head-to-head record; the markdown is
    :func:`format_head_to_head_report`.
    """
    stem = Path(out).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = stem.with_suffix(".json")
    md_path = stem.with_suffix(".md")
    json_path.write_text(json.dumps(res, indent=2, default=_json_default))
    md_path.write_text(format_head_to_head_report(res))
    return {"json": str(json_path), "md": str(md_path)}


__all__ = [
    "compare_per_task",
    "head_to_head_vs_opus",
    "format_head_to_head_report",
    "write_report",
]
