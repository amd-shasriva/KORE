"""KEEP/REVERT decision rules for kernel-agents iterations.

This module is the single source of truth for the policy that decides
whether an Iteration is committed (KEEP) or rolled back (REVERT). It
is intentionally separate from tracker/schema.py so policy changes do
not force data-schema migrations, and so the rules are unit-testable
without instantiating an Experiment.

Design constraints (from review round 1):

  B3: orchestrator/agent.py:273 is a one-shot SDK query. There is no
      state machine and no DEFER state. propose_decision MUST return
      ("KEEP", reason) or ("REVERT", reason) — never DEFER.

  M1: noise_floor_pct = 3.0 (not 1.0); the bench-on-bench variance
      envelope on the AMD vllm-private nightly is 3-5%.

  M2: min_variance_runs = 3; sigma cannot be computed from 2 points.

  M3: require_same_hw_version (hard) + baseline_age_minutes_max (soft)
      replaces the conflated "require_same_node_baseline" of v1.

  M4: workload_type-conditional P99 rule (throughput / latency_sla /
      mixed).

  M5: decision_override_reason escape hatch — reviewer can KEEP a
      below-floor structural simplification (TBO iter6 class).

Prerequisite handling (replaces DEFER): if a required input is missing
the rule returns REVERT with revert_reason starting "PREREQ: ...". The
orchestrator prompt instructs the orchestrator to detect that string
and re-attempt the iteration after gathering the prerequisite (re-bench,
re-baseline, ask reviewer) WITHIN THE SAME SESSION before exit.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kore.verifier.tracker import Experiment, Iteration


def propose_decision(
    experiment: "Experiment",
    iteration: "Iteration",
    now_minutes: float | None = None,
) -> tuple[str, str]:
    """Return ("KEEP" | "REVERT", reason) for the given iteration.

    Args:
        experiment: holds the policy fields (noise_floor_pct,
            min_variance_runs, require_same_hw_version, etc.)
        iteration:  the candidate change with its metrics, sigma,
            reviewer_verdict, decision_override_reason
        now_minutes: optional override for the "now" timestamp in
            minutes (epoch / 60). Defaults to time.time()/60.
            Tests inject a fixed value.

    Never returns DEFER. Missing-prerequisite cases return
    ("REVERT", "PREREQ: <what's missing>") — the orchestrator prompt
    is responsible for catching that prefix and re-attempting the
    iteration after the prereq is satisfied.
    """
    if now_minutes is None:
        now_minutes = time.time() / 60.0

    # ─── Override path (M5) ───
    # Reviewer in ab_decision mode may issue a decision_override_reason
    # to KEEP a sub-floor structural simplification (TBO iter6 class).
    # Override still requires reviewer APPROVE — REQUEST_CHANGES kills
    # the override.
    if iteration.decision_override_reason and iteration.reviewer_verdict == "APPROVE":
        return ("KEEP", f"override: {iteration.decision_override_reason}")

    # ─── Kernel iterations: original semantics (no change) ───
    if iteration.change_type == "kernel":
        # The kernel mandate hasn't changed: SNR pass + wall_ms improved
        # over best.
        if iteration.snr_db is None:
            return ("REVERT", "PREREQ: SNR not measured")
        if iteration.snr_db < 30.0:
            return ("REVERT", f"SNR {iteration.snr_db:.1f} dB < 30 dB")
        if iteration.wall_ms is None:
            return ("REVERT", "PREREQ: wall_ms not measured")
        # Compare to best kernel iteration so far (excluding this one).
        prior_best = _prior_best_wall_ms(experiment, iteration)
        if prior_best is None:
            # First iteration — KEEP unconditionally; baseline establishment.
            return ("KEEP", f"first kernel iteration, wall_ms={iteration.wall_ms:.3f}")
        if iteration.wall_ms >= prior_best:
            return ("REVERT",
                    f"wall_ms {iteration.wall_ms:.3f} ms not better than "
                    f"best {prior_best:.3f} ms")
        return ("KEEP",
                f"wall_ms {iteration.wall_ms:.3f} < best {prior_best:.3f}")

    # ─── Framework_patch / config iterations: TBO-style rules ───

    # 1. Reviewer verdict required (M5 also depends on this).
    if not iteration.reviewer_verdict:
        return ("REVERT", "PREREQ: reviewer not consulted")
    if iteration.reviewer_verdict == "REQUEST_CHANGES":
        return ("REVERT", "reviewer REQUEST_CHANGES")
    if iteration.reviewer_verdict == "NEEDS_DISCUSSION":
        return ("REVERT", "PREREQ: reviewer NEEDS_DISCUSSION — operator review")
    if iteration.reviewer_verdict != "APPROVE":
        return ("REVERT", f"reviewer verdict not recognized: {iteration.reviewer_verdict}")

    # 2. Hardware version invariant (M3 hard gate).
    if experiment.require_same_hw_version:
        cfg_hw = iteration.config.get("hw_version", "")
        if not cfg_hw:
            return ("REVERT", "PREREQ: iteration.config['hw_version'] not recorded")
        if experiment.hw_version_snapshot and cfg_hw != experiment.hw_version_snapshot:
            return ("REVERT",
                    f"hw_version {cfg_hw} != experiment snapshot "
                    f"{experiment.hw_version_snapshot}")

    # 3. Baseline age (M3 soft gate, replaces "same node").
    baseline_age = _baseline_age_minutes(experiment, iteration, now_minutes)
    if baseline_age is None:
        return ("REVERT", "PREREQ: no baseline timestamp on this experiment")
    if baseline_age > experiment.baseline_age_minutes_max:
        return ("REVERT",
                f"baseline is {baseline_age:.1f} min old "
                f"(max {experiment.baseline_age_minutes_max} min); "
                f"PREREQ: re-baseline on same hw")

    # 4. Variance gate (M2).
    if iteration.metric_runs < experiment.min_variance_runs:
        return ("REVERT",
                f"PREREQ: {iteration.metric_runs} runs, need "
                f"{experiment.min_variance_runs}")
    primary = iteration.primary_metric(experiment.primary_metric_name)
    if primary is None:
        return ("REVERT",
                f"PREREQ: primary metric '{experiment.primary_metric_name}' not measured")
    sigma_pct = iteration.metric_sigma_pct.get(experiment.primary_metric_name, math.inf)
    if sigma_pct > experiment.target_sigma_pct and iteration.metric_runs < experiment.max_variance_runs:
        return ("REVERT",
                f"PREREQ: sigma/mu {sigma_pct:.2f}% > target {experiment.target_sigma_pct}% "
                f"and runs {iteration.metric_runs} < cap {experiment.max_variance_runs}")

    # 5. Variance-aware noise floor (M1).
    #    Gain must clear max(2*sigma, noise_floor_pct).
    prior_primary = _prior_best_primary(experiment, iteration)
    if prior_primary is None:
        # No prior measurement to compare against — this iteration IS the
        # baseline-bench row. KEEP and record.
        return ("KEEP",
                f"baseline bench: {experiment.primary_metric_name}={primary}")
    gain_pct = _gain_pct(prior_primary, primary, experiment.primary_metric_polarity)
    floor = max(2.0 * sigma_pct, experiment.noise_floor_pct)
    if gain_pct < floor:
        return ("REVERT",
                f"gain {gain_pct:+.2f}% < noise floor "
                f"max(2*sigma={2*sigma_pct:.2f}%, {experiment.noise_floor_pct}%)"
                f" = {floor:.2f}%")

    # 6. Workload-conditional P99 rule (M4).
    if experiment.workload_type == "latency_sla":
        p50 = iteration.metrics.get("p50_itl_ms")
        p99 = iteration.metrics.get("p99_itl_ms")
        if p50 and p99 and p99 > 5.0 * p50:
            return ("REVERT",
                    f"latency_sla: p99_itl={p99:.2f} > 5*p50={5*p50:.2f}")
    # throughput / mixed: P99 > 5x P50 is INFO only; reviewer flagged it
    # in the verdict text, propose_decision does not block.

    return ("KEEP",
            f"gain {gain_pct:+.2f}% >= floor {floor:.2f}%, "
            f"sigma {sigma_pct:.2f}%, runs {iteration.metric_runs}, "
            f"reviewer APPROVE")


# ─── helpers ───

def _prior_best_wall_ms(experiment: "Experiment",
                        current: "Iteration") -> float | None:
    """Best wall_ms among kernel iterations before `current`."""
    walls = [it.wall_ms for it in experiment.iterations
             if it.iteration_id < current.iteration_id
             and it.change_type == "kernel"
             and it.snr_db is not None and it.snr_db >= 30.0
             and it.wall_ms is not None
             and it.decision == "KEEP"]
    return min(walls) if walls else None


def _prior_best_primary(experiment: "Experiment",
                        current: "Iteration") -> float | None:
    """Best primary metric among prior framework_patch/config KEEP iterations.

    If no prior KEEP exists, fall back to iteration 1's measured primary
    metric (the experiment-level baseline-bench row, regardless of decision).
    """
    name = experiment.primary_metric_name
    polarity = experiment.primary_metric_polarity
    prior_kept = [it.primary_metric(name) for it in experiment.iterations
                  if it.iteration_id < current.iteration_id
                  and it.change_type in ("framework_patch", "config")
                  and it.decision == "KEEP"
                  and it.primary_metric(name) is not None]
    if prior_kept:
        return max(prior_kept) if polarity == "higher_better" else min(prior_kept)
    # Fallback: first recorded primary metric.
    for it in experiment.iterations:
        if it.iteration_id < current.iteration_id and it.primary_metric(name) is not None:
            return it.primary_metric(name)
    return None


def _baseline_age_minutes(experiment: "Experiment",
                          current: "Iteration",
                          now_minutes: float) -> float | None:
    """Minutes since the most recent recorded primary-metric measurement
    before the current iteration."""
    name = experiment.primary_metric_name
    for it in reversed(experiment.iterations):
        if it.iteration_id < current.iteration_id and it.primary_metric(name) is not None:
            try:
                ts = datetime.fromisoformat(it.timestamp).timestamp() / 60.0
            except (TypeError, ValueError):
                return None
            return now_minutes - ts
    # Experiment's own created_at fallback (covers iteration 1 bench).
    try:
        ts = datetime.fromisoformat(experiment.created_at).timestamp() / 60.0
        return now_minutes - ts
    except (TypeError, ValueError):
        return None


def _gain_pct(baseline: float, new: float,
              polarity: str) -> float:
    """Compute percent gain with sign for KEEP/REVERT comparison."""
    if baseline == 0:
        return 0.0
    if polarity == "higher_better":
        return (new - baseline) / baseline * 100.0
    return (baseline - new) / baseline * 100.0
