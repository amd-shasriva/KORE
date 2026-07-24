"""Champion re-evaluation gate: prove the best kernels aren't reward-hacked.

The dominant credibility risk for a kernel-RL result is PERFORMANCE reward hacking:
a speedup that evaporates once the loopholes are excluded (the well-known
3.13x -> 1.49x collapse). KORE already defends every hack class in-loop
(:mod:`kore.reward.timing_integrity`), but a publishable claim needs a single,
consolidated, maximum-scrutiny RE-EVALUATION of the CHAMPION kernels at the end of
a run - the "robust-kbench" bar.

This gate re-benchmarks each champion (the best kernel discovered per task) under
strictly harder conditions than training and only certifies the ones that survive:

  * HELD-OUT shapes - semantics-preserving, non-power-of-two "odd" variants from
    a lane frozen away from every training augmentation (catches shape overfit).
  * VERIFIED correctness - the enumerated adversarial gate + determinism re-check
    (KORE_VERIFIED_CORRECTNESS=1) + more reseeded trials.
  * HONEST baseline - the compiler-fused bar (KORE_COMPILE_BASELINE=1) so a
    "speedup" over unfused eager can't survive.
  * COLD cache + post-timing correctness re-verification (already in the driver).
  * NO replay cache (every re-eval is fresh) and the static hack scan.
  * COLLAPSE detection - if the re-measured speedup falls below ``collapse_ratio``
    of the training-claimed speedup, the champion is flagged as collapsed (hacked
    or overfit) and NOT certified.

The verdict logic (:func:`champion_verdict`) is pure and CPU-unit-tested; the
runner (:func:`reeval_champion` / :func:`run_champion_reeval`) drives the real
:class:`~kore.env.kore_env.KoreEnv` on hardware. ``load_champions`` reads the
WinRecord JSONL emitted by the co-evolution distillation sink.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from kore.obs import get_logger

log = get_logger("eval.champion")


# --------------------------------------------------------------------------- #
# Pure verdict logic (CPU-unit-tested)
# --------------------------------------------------------------------------- #
@dataclass
class ChampionVerdict:
    task_id: str
    certified: bool
    reason: str
    claimed_speedup: Optional[float]
    measured_speedup: Optional[float]
    correct: bool
    hack_free: bool
    collapsed: bool
    high_variance: bool
    worst_snr_db: Optional[float] = None
    n_heldout_shapes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def champion_verdict(
    task_id: str,
    claimed_speedup: Optional[float],
    measured_speedup: Optional[float],
    *,
    correct: bool,
    hack_free: bool,
    high_variance: bool,
    worst_snr_db: Optional[float] = None,
    n_heldout_shapes: int = 0,
    min_speedup: float = 1.0,
    collapse_ratio: float = 0.7,
) -> ChampionVerdict:
    """Certify a champion iff it survives maximum scrutiny (pure decision).

    A champion is CERTIFIED only when it is: correct on the held-out shapes,
    static-hack-free, stably-timed (not high-variance), genuinely faster than the
    honest (compiler-fused / vendor) baseline (``measured > min_speedup``), and its
    re-measured speedup did NOT collapse below ``collapse_ratio * claimed``.
    """
    collapsed = bool(
        claimed_speedup is not None and measured_speedup is not None
        and claimed_speedup > 0 and measured_speedup < collapse_ratio * claimed_speedup
    )

    def _mk(certified: bool, reason: str) -> ChampionVerdict:
        return ChampionVerdict(
            task_id=task_id, certified=certified, reason=reason,
            claimed_speedup=claimed_speedup, measured_speedup=measured_speedup,
            correct=correct, hack_free=hack_free, collapsed=collapsed,
            high_variance=high_variance, worst_snr_db=worst_snr_db,
            n_heldout_shapes=n_heldout_shapes)

    if not hack_free:
        return _mk(False, "static hack scan flagged the kernel")
    if not correct:
        return _mk(False, "incorrect on held-out shapes (failed verified gate)")
    if measured_speedup is None:
        return _mk(False, "no speedup measured on held-out shapes")
    if high_variance:
        return _mk(False, "timing too high-variance to certify")
    if measured_speedup <= min_speedup:
        return _mk(False, f"not faster than the honest baseline "
                          f"({measured_speedup:.3f}x <= {min_speedup:.3f}x)")
    if collapsed:
        return _mk(False, f"speedup collapsed under scrutiny "
                          f"({measured_speedup:.3f}x < {collapse_ratio:g} x "
                          f"claimed {claimed_speedup:.3f}x)")
    return _mk(True, "certified: survives held-out + verified + honest-baseline scrutiny")


# --------------------------------------------------------------------------- #
# Held-out shapes
# --------------------------------------------------------------------------- #
def held_out_shapes(task, max_shapes: int = 8, *, frozen_split=None):
    """Consume hidden shapes from the training-time frozen split artifact."""
    from kore.tasks.augment import FrozenShapeSplit, generate_hidden_shapes

    if frozen_split is None:
        raise ValueError("a training-time frozen shape manifest is required")
    split = FrozenShapeSplit.read(frozen_split) if isinstance(
        frozen_split, (str, os.PathLike)) else frozen_split
    return generate_hidden_shapes(task, split, max_shapes=max_shapes)


# --------------------------------------------------------------------------- #
# Scrutiny config + env
# --------------------------------------------------------------------------- #
_SCRUTINY_ENV = {
    "KORE_VERIFIED_CORRECTNESS": "1",   # enumerated adversarial gate + no lucky-pass
    "KORE_COMPILE_BASELINE": "1",       # honest compiler-fused baseline
    "KORE_BENCH_COLD": "1",             # cold-cache (L2-flushed) timing
    "KORE_CORRECTNESS_TRIALS": "10",    # more reseeded correctness trials
}


def _scrutiny_config():
    """A copy of CONFIG with the determinism re-check forced on."""
    import copy

    from kore.config import CONFIG
    cfg = copy.copy(CONFIG)
    try:
        cfg.verifier_determinism_check = True
    except Exception:  # noqa: BLE001 - frozen config: env-var defenses still apply
        pass
    return cfg


# --------------------------------------------------------------------------- #
# GPU runner
# --------------------------------------------------------------------------- #
@dataclass
class Champion:
    task_id: str
    source: str
    claimed_speedup: Optional[float] = None


def reeval_champion(champ: Champion, *, max_shapes: int = 8,
                    min_speedup: float = 1.0, collapse_ratio: float = 0.7,
                    config=None, shape_manifest=None) -> ChampionVerdict:
    """Re-evaluate ONE champion under maximum scrutiny on held-out shapes."""
    from kore.env.kore_env import KoreEnv
    from kore.reward.reward import compute_reward
    from kore.tasks.registry import get_task

    prev_env = {k: os.environ.get(k) for k in _SCRUTINY_ENV}
    os.environ.update(_SCRUTINY_ENV)
    try:
        task = get_task(champ.task_id)
        cfg = config or _scrutiny_config()
        if shape_manifest is None:
            v = champion_verdict(champ.task_id, champ.claimed_speedup, None,
                                 correct=False, hack_free=True, high_variance=False)
            v.reason = "training-time frozen shape manifest is required"
            return v
        shapes = held_out_shapes(
            task, max_shapes=max_shapes, frozen_split=shape_manifest)
        if not shapes:
            v = champion_verdict(champ.task_id, champ.claimed_speedup, None,
                                 correct=False, hack_free=True, high_variance=False)
            v.reason = "task has no augmentable held-out shapes"
            return v
        env = KoreEnv(task, config=cfg, use_replay=False)
        obs = env.evaluate(task, champ.source, shapes=shapes, do_bench=True)
        rr = compute_reward(obs, champ.source, dtype=task.dtype,
                            snr_threshold=getattr(task, "snr_threshold", None))
        hack_free = not bool(getattr(obs, "flagged_hack", False))
        high_variance = "high_variance" in (getattr(rr, "flags", []) or [])
        snrs = list((getattr(obs, "snr_by_shape", {}) or {}).values())
        worst_snr = min(snrs) if snrs else None
        v = champion_verdict(
            champ.task_id, champ.claimed_speedup, rr.speedup,
            correct=bool(rr.correct), hack_free=hack_free, high_variance=high_variance,
            worst_snr_db=worst_snr, n_heldout_shapes=len(shapes),
            min_speedup=min_speedup, collapse_ratio=collapse_ratio)
        log.event("champion_reeval", task=champ.task_id, certified=v.certified,
                  reason=v.reason, claimed=champ.claimed_speedup,
                  measured=rr.speedup, collapsed=v.collapsed, n_shapes=len(shapes))
        return v
    finally:
        for k, prev in prev_env.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


@dataclass
class ChampionReport:
    n_champions: int
    n_certified: int
    n_collapsed: int
    verdicts: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "KORE champion re-evaluation (maximum-scrutiny anti-hack gate)",
            f"  champions:  {self.n_champions}",
            f"  certified:  {self.n_certified}",
            f"  collapsed:  {self.n_collapsed}",
        ]
        for v in self.verdicts:
            tag = "CERTIFIED" if v.certified else ("COLLAPSED" if v.collapsed else "REJECTED")
            claimed = f"{v.claimed_speedup:.3f}x" if v.claimed_speedup else "?"
            meas = f"{v.measured_speedup:.3f}x" if v.measured_speedup else "-"
            lines.append(f"    [{tag:9s}] {v.task_id:28s} claimed={claimed:>8} "
                         f"measured={meas:>8}  {v.reason}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"n_champions": self.n_champions, "n_certified": self.n_certified,
                "n_collapsed": self.n_collapsed,
                "verdicts": [v.to_dict() for v in self.verdicts]}


def run_champion_reeval(champions: list[Champion], *, max_shapes: int = 8,
                        min_speedup: float = 1.0, collapse_ratio: float = 0.7,
                        out_path: Optional[str] = None, config=None,
                        shape_manifests: Optional[Mapping[str, object]] = None
                        ) -> ChampionReport:
    """Re-evaluate all champions and write a JSON certification report."""
    verdicts: list[ChampionVerdict] = []
    for champ in champions:
        try:
            v = reeval_champion(champ, max_shapes=max_shapes, min_speedup=min_speedup,
                               collapse_ratio=collapse_ratio, config=config,
                               shape_manifest=(
                                   shape_manifests.get(champ.task_id)
                                   if shape_manifests else None))
        except Exception as e:  # noqa: BLE001 - one bad champion can't abort the gate
            v = champion_verdict(champ.task_id, champ.claimed_speedup, None,
                                 correct=False, hack_free=True, high_variance=False)
            v.reason = f"re-eval error: {type(e).__name__}: {e}"
            log.warn("champion re-eval error", task=champ.task_id, error=repr(e))
        verdicts.append(v)
    report = ChampionReport(
        n_champions=len(champions),
        n_certified=sum(1 for v in verdicts if v.certified),
        n_collapsed=sum(1 for v in verdicts if v.collapsed),
        verdicts=verdicts)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report.to_dict(), indent=2))
    log.metric("champion_reeval_done", n_champions=report.n_champions,
               n_certified=report.n_certified, n_collapsed=report.n_collapsed)
    return report


def load_shape_manifests(path: str) -> dict[str, object]:
    """Load one frozen split JSON per task from a lineage artifact directory."""
    from kore.tasks.augment import FrozenShapeSplit

    manifests: dict[str, object] = {}
    for manifest_path in sorted(Path(path).glob("*.json")):
        manifest = FrozenShapeSplit.read(manifest_path)
        if manifest.task_id in manifests:
            raise ValueError(f"duplicate shape manifest for {manifest.task_id!r}")
        manifests[manifest.task_id] = manifest
    return manifests


# --------------------------------------------------------------------------- #
# Load champions from the distillation-sink WinRecord JSONL
# --------------------------------------------------------------------------- #
def load_champions(path: str, *, min_speedup: float = 1.0) -> list[Champion]:
    """Load champions (best kernel per task) from a WinRecord JSONL.

    Keeps the HIGHEST-claimed-speedup record per task_id, so each task is
    re-evaluated once on its best discovered kernel."""
    from kore.data.schemas import read_jsonl

    best: dict[str, Champion] = {}
    for rec in read_jsonl(path):
        src = getattr(rec, "final_source", None) if not isinstance(rec, dict) \
            else rec.get("final_source")
        tid = getattr(rec, "task_id", None) if not isinstance(rec, dict) \
            else rec.get("task_id")
        su = getattr(rec, "speedup", None) if not isinstance(rec, dict) \
            else rec.get("speedup")
        if not src or not tid:
            continue
        su = float(su) if su is not None else None
        cur = best.get(tid)
        if cur is None or (su or 0.0) > (cur.claimed_speedup or 0.0):
            best[tid] = Champion(task_id=tid, source=src, claimed_speedup=su)
    return list(best.values())


def main(argv=None) -> int:  # pragma: no cover - CLI
    import argparse

    p = argparse.ArgumentParser(description="KORE champion re-evaluation anti-hack gate")
    p.add_argument("champions", help="WinRecord JSONL (distillation-sink output)")
    p.add_argument("--out", default="runs/champion_reeval.json")
    p.add_argument("--max-shapes", type=int, default=8)
    p.add_argument("--min-speedup", type=float, default=1.0)
    p.add_argument("--collapse-ratio", type=float, default=0.7)
    p.add_argument("--shape-manifests",
                   help="directory of training-time frozen shape manifests")
    a = p.parse_args(argv)
    champs = load_champions(a.champions)
    manifests = load_shape_manifests(a.shape_manifests) if a.shape_manifests else {}
    report = run_champion_reeval(
        champs, max_shapes=a.max_shapes, min_speedup=a.min_speedup,
        collapse_ratio=a.collapse_ratio, out_path=a.out,
        shape_manifests=manifests)
    print(report.summary())
    print(f"\n[champion] report -> {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
