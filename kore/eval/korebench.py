"""KORE-Bench: a standardized, production-baseline, distribution-swept, timing-
hardened kernel-generation benchmark report.

Assembles the pieces KORE already has into ONE reproducible artifact:
  * matched-budget evaluation (kore.eval.bakeoff) of a policy over the task suite,
  * the headline DISTRIBUTIONALLY-ROBUST metric - WORST-SHAPE win-rate vs the
    PRODUCTION vendor baseline (AITER/hipBLASLt / framework path), i.e. the
    fraction of operators whose *hardest* shape still beats the vendor library,
  * fast_p curve + geometric-mean speedup vs that baseline,
  * per-operator-family breakdown (generalization view),
  * the data-scale summary (kore.tasks.audit) and the timing-integrity coverage
    guarantee (kore.reward.timing_integrity).

This is the reusable benchmark deliverable; ``run_korebench`` accepts a live
``env_factory`` (GPU) or a ``dry_run`` (precomputed Observations) so it is fully
CPU-testable.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from kore.config import CONFIG, KoreConfig
from kore.eval.bakeoff import evaluate_policy
from kore.eval.fastp import DEFAULT_PS


def _win(t: dict, margin: float) -> bool:
    return bool(t.get("correct")) and (t.get("best_speedup") or 0.0) > margin


def run_korebench(
    policy_fn: Callable,
    tasks: Sequence,
    *,
    env_factory: Optional[Callable] = None,
    dry_run: Optional[object] = None,
    budget: int = 5,
    ps: Sequence[float] = DEFAULT_PS,
    cfg: KoreConfig = CONFIG,
    win_margin: float = 1.0,
) -> dict:
    """Run the KORE-Bench protocol and return the standardized report dict.

    ``win_margin`` is the speedup a candidate must EXCEED vs the production baseline
    to count as a win (1.0 == strictly beat the vendor library). The headline
    number is the WORST-SHAPE win-rate (best_speedup is the worst-shape speedup
    under the default distributionally-robust aggregation).
    """
    res = evaluate_policy(policy_fn, tasks, env_factory=env_factory, budget=budget,
                          dry_run=dry_run, ps=ps, cfg=cfg)
    per = res["per_task"]
    n = len(per)
    n_wins = sum(1 for t in per if _win(t, win_margin))

    # per operator-family breakdown (generalization view)
    from kore.tasks.registry import get_task, operator_family
    fam: dict[str, list] = {}
    for t in per:
        try:
            f = operator_family(get_task(t["task_id"]))
        except Exception:  # noqa: BLE001 - unknown/ad-hoc task id
            f = "other"
        fam.setdefault(f, []).append(t)
    per_family = {
        f: {"n": len(ts), "win_rate": sum(1 for t in ts if _win(t, win_margin)) / len(ts)}
        for f, ts in fam.items()
    }

    from kore.reward import timing_integrity as ti
    return {
        "n_tasks": n,
        "budget": budget,
        "win_margin": win_margin,
        "worst_shape_win_rate_vs_baseline": (n_wins / n) if n else 0.0,
        "num_correct": res["num_correct"],
        "correct_rate": (res["num_correct"] / n) if n else 0.0,
        "fast_p": res["fast_p"],
        "geometric_mean_speedup": res["geometric_mean_speedup"],
        "per_family": per_family,
        "timing_integrity_complete": ti.uncovered() == [],
        "speed_aggregation": getattr(cfg, "speed_aggregation", "worst"),
        "per_task": per,
    }


def data_scale_summary() -> dict:
    """The benchmark's own data-scale descriptor (operators/families/shapes)."""
    from kore.tasks.audit import audit
    rep = audit()
    return {
        "operators": rep.n_operators, "train": rep.n_train, "heldout": rep.n_heldout,
        "families": len(rep.families), "base_shapes": rep.total_base_shapes,
        "dtypes": rep.dtypes, "heldout_families": rep.heldout_families,
    }


def format_report(report: dict) -> str:
    lines = [
        "KORE-Bench report",
        f"  operators evaluated:   {report['n_tasks']}  (budget {report['budget']}/task)",
        f"  speed aggregation:     {report['speed_aggregation']} (distributionally-robust)",
        f"  correct rate:          {report['correct_rate']:.3f}",
        f"  WORST-SHAPE win-rate vs production baseline (>{report['win_margin']}x): "
        f"{report['worst_shape_win_rate_vs_baseline']:.3f}",
        f"  geomean speedup:       {report['geometric_mean_speedup']:.3f}",
        f"  fast_p:                {report['fast_p']}",
        f"  timing-integrity:      {'COMPLETE' if report['timing_integrity_complete'] else 'INCOMPLETE'}",
        "  per-family win-rate:",
    ]
    for f in sorted(report["per_family"]):
        d = report["per_family"][f]
        lines.append(f"    {f:14s} n={d['n']:<4d} win_rate={d['win_rate']:.3f}")
    return "\n".join(lines)
