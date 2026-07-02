"""Data schemas for experiment tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class Iteration:
    """A single build-test-bench-profile cycle.

    Iterations come in three flavors, distinguished by change_type:
      - "kernel"          : the original mandate; snr_db / wall_ms / vgpr fields populated
      - "framework_patch" : Python-source diff in vllm/ or sglang/; metrics dict populated, snr_db None
      - "config"          : env var / CLI flag tweak; metrics dict populated, snr_db None
    """

    iteration_id: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Configuration that was tested
    config: dict = field(default_factory=dict)

    # NEW: change classification (extends instead of replaces)
    change_type: Literal["kernel", "framework_patch", "config"] = "kernel"
    target_file: str | None = None            # framework_patch / kernel: edited file
    patch_path: str | None = None             # framework_patch: $KA_SESSION_PATCHES_HOST/<iter>_<desc>.diff
    backup_path: str | None = None            # framework_patch: <basename>.bak.<iter-id>.<sha256[:8]>

    # Correctness (kernel only; framework_patch leaves snr_db=None — see Phase F)
    snr_db: float | None = None
    allclose: bool | None = None
    max_diff: float | None = None

    # Performance — kernel mandate (unchanged)
    wall_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None

    # NEW: rich metrics (framework_patch and config modes carry these instead
    # of the single wall_ms scalar). Schema is workload-defined but TBO's
    # canonical key set is:
    #   "output_tps", "tpot_p50_ms", "ttft_p50_ms",
    #   "p90_itl_ms", "p99_itl_ms", "correctness_pct"
    metrics: dict[str, float] = field(default_factory=dict)
    metric_runs: int = 0                      # how many variance runs produced `metrics`
    metric_sigma_pct: dict[str, float] = field(default_factory=dict)  # per-key sigma/mu

    # PMC analysis (kernel only)
    pmc: dict = field(default_factory=dict)
    wait_mfma_ratio: float | None = None
    pmc_diagnosis: str = ""

    # Register info (kernel only)
    vgpr: int | None = None
    agpr: int | None = None
    spill_bytes: int = 0

    # NEW: reviewer integration
    reviewer_verdict: str = ""                # APPROVE / REQUEST_CHANGES / NEEDS_DISCUSSION / ""
    decision_override_reason: str | None = None  # M5 — honored by propose_decision

    # Decision made after this iteration
    decision: str = ""                        # "KEEP" / "REVERT" / ""
    revert_reason: str = ""                   # populated by propose_decision when decision=="REVERT"
    notes: str = ""

    def to_dict(self) -> dict:
        # Keep existing semantics: skip falsy/empty fields for compactness, but
        # preserve the new ones explicitly when populated.
        return {k: v for k, v in self.__dict__.items()
                if v is not None and v != "" and v != {} and v != []}

    @classmethod
    def from_dict(cls, d: dict) -> Iteration:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def primary_metric(self, name: str = "output_tps") -> float | None:
        """Return the primary metric for KEEP/REVERT decisions.

        - kernel rows: wall_ms (LOWER is better; caller inverts for gain)
        - framework_patch / config rows: metrics[name] (HIGHER is better
          for tps; caller knows the polarity).
        """
        if self.change_type == "kernel":
            return self.wall_ms
        return self.metrics.get(name)

    def summary_row(self) -> str:
        """One-line summary for experiment log table."""
        if self.change_type == "kernel":
            snr = f"{self.snr_db:.1f}" if self.snr_db is not None else "?"
            wall = f"{self.wall_ms:.3f}" if self.wall_ms is not None else "?"
            ratio = f"{self.wait_mfma_ratio:.1f}" if self.wait_mfma_ratio is not None else "?"
            vgpr_s = str(self.vgpr) if self.vgpr is not None else "?"
            return (
                f"| {self.iteration_id:4d} | {self.change_type:>14s} | "
                f"{snr:>8s} | {wall:>9s} | "
                f"{ratio:>8s} | {vgpr_s:>5s} | {self.decision} |"
            )
        tps = self.metrics.get("output_tps")
        tps_s = f"{tps:.1f}" if tps is not None else "?"
        sigma = self.metric_sigma_pct.get("output_tps", 0.0)
        sigma_s = f"{sigma:.2f}%" if sigma else "?"
        return (
            f"| {self.iteration_id:4d} | {self.change_type:>14s} | "
            f"   n/a  |     n/a   | sigma={sigma_s:>8s} | tps={tps_s:>7s} | {self.decision} |"
        )


@dataclass
class Experiment:
    """A complete development experiment spanning multiple iterations.

    Iterations may be kernel iterations (the original mandate — SNR-gated),
    framework_patch iterations (perf-only, no SNR), or config iterations
    (perf-only, no SNR). The KEEP/REVERT decision is delegated to
    orchestrator/decisions.py propose_decision(), which reads the policy
    fields below.
    """

    experiment_id: str
    task_id: str = ""
    backend: str = ""          # ck, flydsl, triton, aiter, framework
    fellow: str = ""           # which fellow agent ran this
    description: str = ""
    target_wall_ms: float | None = None
    baseline_wall_ms: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    iterations: list[Iteration] = field(default_factory=list)

    # NEW: orchestrator decision policy (read by propose_decision; persisted
    # so a resumed experiment uses the same policy as the original run)
    noise_floor_pct: float = 3.0              # M1 — was 1.0, now 3.0 to clear bench-variance envelope
    min_variance_runs: int = 3                # M2 — was 2; sigma not computable from 2 points
    max_variance_runs: int = 5                # M2 — cap
    target_sigma_pct: float = 2.0             # M2 — stop adding runs once sigma/mu <= target
    require_same_hw_version: bool = True      # M3 — replaces "same node"
    baseline_age_minutes_max: int = 60        # M3 — replaces "same node" (thermal drift envelope)
    workload_type: Literal["throughput", "latency_sla", "mixed"] = "throughput"  # M4

    # NEW: revert ledger (parallels TBO session_state.changes_reverted)
    changes_reverted: list[str] = field(default_factory=list)

    # NEW: which metric is the KEEP/REVERT primary
    primary_metric_name: str = "output_tps"
    primary_metric_polarity: Literal["higher_better", "lower_better"] = "higher_better"

    # NEW: same-hw-version snapshot (set on first bench; future iterations
    # are required to match unless an override is documented)
    hw_version_snapshot: str = ""             # e.g. "gfx950"
    node_snapshot: str = ""                   # e.g. SLURM node name — informational

    # NEW: total LLM token spend for the whole run, summed from the
    # claude-agent-sdk ResultMessage stream (see tracker/usage.py). Canonical
    # keys: input_tokens / output_tokens / cache_creation_input_tokens /
    # cache_read_input_tokens / total_cost_usd / calls. Empty until the loop
    # finishes (or when no agent ran), so an external caller can read the run's
    # token cost straight off the experiment record.
    llm_usage: dict = field(default_factory=dict)

    def add_iteration(self, **kwargs) -> Iteration:
        """Add a new iteration with auto-incrementing ID."""
        iter_id = len(self.iterations) + 1
        iteration = Iteration(iteration_id=iter_id, **kwargs)
        self.iterations.append(iteration)
        return iteration

    def best_iteration(self) -> Iteration | None:
        """Return the iteration with lowest wall_ms that passed SNR gate.

        Kernel-only iterations are eligible. Framework_patch and config
        iterations are tracked separately via best_framework_patch() and
        do not interfere with the kernel-best logic (addresses A2 — they
        were silently excluded by the old SNR gate; that exclusion is now
        explicit and there's a sibling helper for the non-SNR case).
        """
        passing = [
            it for it in self.iterations
            if it.change_type == "kernel"
            and it.snr_db is not None and it.snr_db >= 30.0
            and it.wall_ms is not None
        ]
        return min(passing, key=lambda it: it.wall_ms) if passing else None

    def best_framework_patch(self) -> Iteration | None:
        """Return the framework_patch/config iteration with best primary metric."""
        candidates = [
            it for it in self.iterations
            if it.change_type in ("framework_patch", "config")
            and it.decision == "KEEP"
            and it.primary_metric(self.primary_metric_name) is not None
        ]
        if not candidates:
            return None
        reverse = (self.primary_metric_polarity == "higher_better")
        return sorted(
            candidates,
            key=lambda it: it.primary_metric(self.primary_metric_name),
            reverse=reverse,
        )[0]

    def is_plateaued(self, n: int = 3, threshold: float = 0.02) -> bool:
        """Check if last n passing kernel iterations improved less than threshold."""
        passing = [
            it.wall_ms for it in self.iterations
            if it.change_type == "kernel"
            and it.snr_db is not None and it.snr_db >= 30.0
            and it.wall_ms is not None
        ]
        if len(passing) < n:
            return False
        recent = passing[-n:]
        return (max(recent) - min(recent)) / min(recent) < threshold

    def is_gate_met(self) -> bool:
        """Check if target wall_ms has been met by any passing kernel iteration."""
        if self.target_wall_ms is None:
            return False
        best = self.best_iteration()
        return best is not None and best.wall_ms <= self.target_wall_ms

    def effective_baseline_ms(self) -> float | None:
        """Kernel-baseline anchor for speedup reporting (unchanged semantics)."""
        if self.baseline_wall_ms is not None:
            return self.baseline_wall_ms
        for it in self.iterations:
            if it.change_type == "kernel" and it.wall_ms is not None:
                return it.wall_ms
        return None

    def speedup_vs_baseline(self) -> float | None:
        """Speedup of best kernel iteration vs baseline (unchanged)."""
        best = self.best_iteration()
        baseline = self.effective_baseline_ms()
        if best is None or baseline is None:
            return None
        return baseline / best.wall_ms

    def cumulative_gain_pct(self) -> float:
        """TBO-style cumulative gain on the primary metric across KEPT
        framework_patch + config iterations.

        Computed as the ratio of best primary metric to baseline primary
        metric (drawn from iteration 1's metrics dict if present; otherwise
        from the first KEPT row). Returns 0.0 when no baseline is recorded.
        """
        kept = [it for it in self.iterations
                if it.change_type in ("framework_patch", "config")
                and it.decision == "KEEP"
                and it.primary_metric(self.primary_metric_name) is not None]
        if not kept:
            return 0.0
        # Baseline: first iteration that recorded the primary metric, KEEP or not.
        recorded = [it for it in self.iterations
                    if it.primary_metric(self.primary_metric_name) is not None]
        if not recorded:
            return 0.0
        base = recorded[0].primary_metric(self.primary_metric_name)
        best = self.best_framework_patch()
        if best is None or base in (None, 0):
            return 0.0
        cur = best.primary_metric(self.primary_metric_name)
        if self.primary_metric_polarity == "higher_better":
            return (cur - base) / base * 100.0
        return (base - cur) / base * 100.0

    def consecutive_reverts(self) -> int:
        """How many of the most-recent iterations were REVERTs in a row.

        Used by the orchestrator to bail out of a session that's only
        producing reverts (cross-session signal).
        """
        n = 0
        for it in reversed(self.iterations):
            if it.decision == "REVERT":
                n += 1
            elif it.decision == "KEEP":
                break
            # ignore "" (incomplete) rows
        return n

    def propose_decision(self, iteration: Iteration,
                         now_minutes: float | None = None) -> tuple[str, str]:
        """KEEP/REVERT for `iteration`. Delegates to orchestrator/decisions.py.

        Kept here as a thin wrapper so existing tracker callers don't need
        to import the new module. The real rules live in decisions.py
        (Change 4) so they can be unit-tested without instantiating an
        Experiment.
        """
        from kore.verifier.decisions import propose_decision as _pd
        return _pd(self, iteration, now_minutes=now_minutes)

    def summary_table(self) -> str:
        """Markdown table of all iterations."""
        header = ("| Iter |    change_type |   SNR dB |  wall_ms |  variance | "
                  "  primary metric | Decision |")
        sep    =  "|------|----------------|----------|----------|-----------|-----------------|----------|"
        rows = [it.summary_row() for it in self.iterations]
        lines = [header, sep] + rows

        # Summaries
        best_k = self.best_iteration()
        if best_k:
            lines.append(f"\nBest kernel iter: {best_k.iteration_id} @ {best_k.wall_ms:.3f} ms")
        if self.speedup_vs_baseline():
            lines.append(f"Speedup vs baseline: {self.speedup_vs_baseline():.3f}x")
        if self.is_gate_met():
            lines.append(f"Gate ({self.target_wall_ms} ms): MET")
        elif self.target_wall_ms:
            lines.append(f"Gate ({self.target_wall_ms} ms): NOT MET")
        if self.is_plateaued():
            lines.append("Status: PLATEAUED (last 3 kernel iters <2% improvement)")

        cum = self.cumulative_gain_pct()
        if cum:
            lines.append(f"Cumulative framework_patch+config gain: {cum:+.2f}% "
                         f"on {self.primary_metric_name}")
        if self.changes_reverted:
            lines.append(f"Reverted: {', '.join(self.changes_reverted)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "iterations"}
        d["iterations"] = [it.to_dict() for it in self.iterations]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Experiment:
        iterations = [Iteration.from_dict(it) for it in d.pop("iterations", [])]
        exp = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        exp.iterations = iterations
        return exp
