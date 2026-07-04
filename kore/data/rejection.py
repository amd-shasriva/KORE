"""Stratified rejection-sampling / RFT selection of >1x winning kernels.

The policy stalls in the "correct-but-slow" band. A direct, high-leverage fix
(ReST / RFT / rejection-sampling fine-tuning, a la Kevin-32B's iterative SFT and
DeepMind's ReST) is to BOOTSTRAP the policy on its own best outputs: keep only
trajectories that are correct AND actually beat the vendor baseline (speedup >=
tau, default 1.0), then fine-tune on them. This concentrates probability mass on
the >1x region the sparse RL reward struggles to reach.

Naive rejection sampling has two well-documented failure modes, both guarded here:
  1. Entropy collapse — a few easy tasks dominate the kept set, so the model
     overfits them and generalization drops. We STRATIFY: round-robin across
     tasks with a per-task fraction cap, maximizing task entropy of the kept set.
  2. Near-duplicate memorization — the same winning kernel appears many times.
     We DEDUP by normalized source (comments/whitespace stripped), keeping the
     fastest instance.

Everything is pure and deterministic (seeded); GPU-measured ``speedup``/``snr_db``
already live on the ``WinRecord``s produced by datagen/evolve, so this is a
data-only transform that feeds the SFT ``extra_records`` bucket.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from kore.data.schemas import WinRecord


def _norm_source(src: str) -> str:
    """Normalize a kernel source for dedup: drop comments + collapse whitespace."""
    src = re.sub(r'"""[\s\S]*?"""', " ", src or "")
    src = re.sub(r"'''[\s\S]*?'''", " ", src)
    src = re.sub(r"#.*", "", src)
    return re.sub(r"\s+", " ", src).strip()


def win_speedup(rec: Any) -> Optional[float]:
    return rec.speedup if isinstance(rec, WinRecord) else None


def passes_win_filter(rec: Any, tau: float = 1.0,
                      min_snr: Optional[float] = None) -> bool:
    """A record is a keepable win iff it is a WinRecord that (a) beats the
    baseline by >= tau, (b) meets the SNR gate if one is given, and (c) carries
    a non-empty final source + trajectory to train on."""
    if not isinstance(rec, WinRecord):
        return False
    if rec.speedup is None or rec.speedup < tau:
        return False
    if min_snr is not None and (rec.snr_db is None or rec.snr_db < min_snr):
        return False
    return bool(rec.final_source and rec.trajectory)


def task_entropy(per_task: dict[str, int]) -> float:
    """Normalized Shannon entropy in [0,1] of the kept-set task distribution.
    1.0 = perfectly uniform across tasks (max diversity), 0 = single task."""
    total = sum(per_task.values())
    if total <= 0 or len(per_task) <= 1:
        return 0.0 if total <= 0 else 1.0 if len(per_task) == 1 else 0.0
    h = -sum((n / total) * math.log(n / total) for n in per_task.values() if n > 0)
    return h / math.log(len(per_task))


@dataclass
class RFTReport:
    n_in: int
    n_pass_filter: int
    n_after_dedup: int
    n_kept: int
    per_task: dict[str, int]
    task_entropy: float
    tau: float


def stratified_rft_select(
    records: list[Any], *, tau: float = 1.0, min_snr: Optional[float] = None,
    max_total: Optional[int] = None, per_task_frac_cap: float = 0.34,
    dedup: bool = True, seed: int = 0,
) -> tuple[list[WinRecord], RFTReport]:
    """Select a stratified, deduped set of >=tau wins for RFT.

    Round-robin across tasks (fastest-first within each task) so no single task
    exceeds ``per_task_frac_cap`` of the kept set until every other task is
    exhausted — this maximizes task entropy and prevents easy-task collapse.
    """
    wins = [r for r in records if passes_win_filter(r, tau, min_snr)]
    n_pass = len(wins)

    if dedup:
        best: dict[str, WinRecord] = {}
        for r in wins:
            key = _norm_source(r.final_source)
            cur = best.get(key)
            if cur is None or (r.speedup or 0) > (cur.speedup or 0):
                best[key] = r
        wins = list(best.values())
    n_dedup = len(wins)

    by_task: dict[str, list[WinRecord]] = defaultdict(list)
    for r in wins:
        by_task[r.task_id].append(r)
    # deterministic: fastest-first within task, tasks ordered by id (seed rotates)
    for tid in by_task:
        by_task[tid].sort(key=lambda r: (-(r.speedup or 0.0), _norm_source(r.final_source)))
    task_ids = sorted(by_task)
    if task_ids:
        rot = seed % len(task_ids)
        task_ids = task_ids[rot:] + task_ids[:rot]

    limit = n_dedup if max_total is None else min(max_total, n_dedup)
    cap = max(1, int(per_task_frac_cap * limit)) if limit else 0

    kept: list[WinRecord] = []
    taken = defaultdict(int)
    idx = defaultdict(int)
    # first pass respects the per-task cap; if we still have room afterwards we
    # relax the cap (rare: few tasks) so we never under-fill available budget.
    for enforce_cap in (True, False):
        progressed = True
        while len(kept) < limit and progressed:
            progressed = False
            for tid in task_ids:
                if len(kept) >= limit:
                    break
                if enforce_cap and taken[tid] >= cap:
                    continue
                i = idx[tid]
                if i < len(by_task[tid]):
                    kept.append(by_task[tid][i])
                    idx[tid] = i + 1
                    taken[tid] += 1
                    progressed = True
        if len(kept) >= limit:
            break

    per_task = dict(taken)
    report = RFTReport(
        n_in=len(records), n_pass_filter=n_pass, n_after_dedup=n_dedup,
        n_kept=len(kept), per_task=per_task,
        task_entropy=round(task_entropy(per_task), 4), tau=tau,
    )
    return kept, report
