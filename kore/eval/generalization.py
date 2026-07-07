"""Zero-shot cross-family generalization harness (KORE P0, Phase 6).

The paradigm claim is that the residual space (SOL attainment + named
stall/occupancy residual) is *operator-independent*, so a policy trained to
descend the residual on some operator families should transfer to families it
never saw. This harness makes that claim falsifiable WITHOUT any training:

  1. Partition the operator zoo into disjoint FAMILIES (norm, activation, gemm,
     attention, moe, reduction, positional, quant).
  2. Hold out ENTIRE families (e.g. train excludes attention + moe) and assert
     there is no task- or family-level leakage between the train and held-out
     splits.
  3. Evaluate eta and the physics residual-descent reward on the HELD-OUT
     families only, aggregated per family, from an offline measurement JSON
     (as produced by ``kore.analysis.p0_sol``). The same call runs later against
     a trained checkpoint's measured kernels -- it is a pure offline eval and
     NEVER launches training.

This is deliberately measurement-consuming: it does not itself run the GPU. Feed
it the p0 study JSON (seed/zero-shot kernels now, a checkpoint's kernels later)
and it reports transfer to unseen families.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Optional

from kore.reward.physics import (
    DEFAULT_PHYSICS_WEIGHT,
    PhysicsSignal,
    compute_residual_reward,
    observation_from_measure,
    physics_from_measure,
)

# --------------------------------------------------------------------------- #
# Operator families: the single source of truth for the holdout split. Every
# KORE task id maps to exactly one family. Keep in sync with tasks/registry.py.
# --------------------------------------------------------------------------- #
FAMILIES: dict[str, set[str]] = {
    "norm": {"rmsnorm_aiter", "layernorm_bf16", "fused_add_rmsnorm_bf16"},
    "activation": {"silu_mul_bf16", "gelu_tanh_bf16"},
    "gemm": {"gemm_bf16", "gemm_fp8_a8w8"},
    "attention": {"flash_attn_decode_bf16", "flash_attn_prefill_bf16",
                  "paged_attn_decode_bf16"},
    "moe": {"fused_moe_silu_bf16", "topk_softmax_bf16"},
    "reduction": {"softmax_bf16"},
    "positional": {"rope_bf16"},
    "quant": {"quant_fp8_pertoken"},
}


def family_of(task_id: str) -> Optional[str]:
    """Family for a task id, or None if it is not registered in any family."""
    for fam, members in FAMILIES.items():
        if task_id in members:
            return fam
    return None


def all_registered_tasks() -> set[str]:
    out: set[str] = set()
    for members in FAMILIES.values():
        out |= members
    return out


@dataclass
class HoldoutSplit:
    """A leakage-checked train / held-out partition BY FAMILY."""

    heldout_families: list[str]
    train_families: list[str]
    heldout_tasks: list[str]
    train_tasks: list[str]

    def as_dict(self) -> dict:
        return {
            "heldout_families": sorted(self.heldout_families),
            "train_families": sorted(self.train_families),
            "heldout_tasks": sorted(self.heldout_tasks),
            "train_tasks": sorted(self.train_tasks),
        }


def make_holdout_split(heldout_families: list[str],
                       task_ids: Optional[list[str]] = None) -> HoldoutSplit:
    """Build a family-level holdout split.

    ``heldout_families`` are excluded from training; every other family is a
    training family. ``task_ids`` restricts the universe (defaults to all
    registered tasks) so the split reflects only tasks actually present.
    Raises ``ValueError`` for an unknown family or an unregistered task.
    """
    universe = set(task_ids) if task_ids is not None else all_registered_tasks()
    unknown = [t for t in universe if family_of(t) is None]
    if unknown:
        raise ValueError(f"tasks not in any family: {sorted(unknown)}")
    bad_fam = [f for f in heldout_families if f not in FAMILIES]
    if bad_fam:
        raise ValueError(f"unknown families: {bad_fam} (known: {sorted(FAMILIES)})")
    held_fams = set(heldout_families)
    train_fams = [f for f in FAMILIES if f not in held_fams]
    heldout_tasks = sorted(t for t in universe if family_of(t) in held_fams)
    train_tasks = sorted(t for t in universe if family_of(t) not in held_fams)
    split = HoldoutSplit(
        heldout_families=list(held_fams),
        train_families=train_fams,
        heldout_tasks=heldout_tasks,
        train_tasks=train_tasks,
    )
    assert_no_leakage(split)
    return split


def assert_no_leakage(split: HoldoutSplit) -> None:
    """Fail loudly on ANY train/held-out overlap (task- or family-level)."""
    tset, hset = set(split.train_tasks), set(split.heldout_tasks)
    inter = tset & hset
    if inter:
        raise AssertionError(f"train/held-out TASK leakage: {sorted(inter)}")
    tfam = {family_of(t) for t in split.train_tasks}
    hfam = {family_of(t) for t in split.heldout_tasks}
    fam_inter = tfam & hfam
    if fam_inter:
        raise AssertionError(f"train/held-out FAMILY leakage: {sorted(fam_inter)}")
    declared_held = set(split.heldout_families)
    if hfam - declared_held:
        raise AssertionError(
            f"held-out tasks span undeclared families: {sorted(hfam - declared_held)}")


# --------------------------------------------------------------------------- #
# Offline evaluation over a measurement JSON (kore.analysis.p0_sol output).
# --------------------------------------------------------------------------- #
class _Rec:
    """Duck-typed KernelMeasure view over a p0_sol JSON record."""

    def __init__(self, d: dict):
        self.task_id = d.get("task_id")
        self.correct = bool(d.get("correct"))
        self.snr_db = d.get("snr_db")
        self.cand_ms = d.get("cand_ms")
        self.vendor_ms = d.get("vendor_ms")
        self.t_min_ms = d.get("t_min_ms", float("nan"))
        self.eta = d.get("eta")
        self.speedup = d.get("speedup")
        self.stall_frac = d.get("stall_frac")
        self.occupancy = d.get("occupancy")
        self.baseline_type = d.get("baseline_type")


def evaluate_generalization(split: HoldoutSplit, measures: list[dict],
                            physics_weight: float = DEFAULT_PHYSICS_WEIGHT,
                            dtype: str = "bf16") -> dict:
    """Compute per-family transfer metrics on the HELD-OUT families only.

    ``measures`` is the ``measures`` list from a p0_sol JSON report (each item a
    dict). Returns per-family aggregates (median eta / residual reward / speedup,
    counts) restricted to held-out families, plus a leakage-free assertion of the
    task set actually scored.
    """
    assert_no_leakage(split)
    held = set(split.heldout_families)
    per_family: dict[str, dict] = {}
    scored_tasks: set[str] = set()
    for d in measures:
        rec = _Rec(d)
        if not rec.task_id or not rec.correct:
            continue
        fam = family_of(rec.task_id)
        if fam not in held:
            continue
        if rec.eta is None and not (rec.cand_ms and rec.t_min_ms == rec.t_min_ms):
            continue
        obs = observation_from_measure(rec, dtype=dtype)
        sig = physics_from_measure(rec)
        rr = compute_residual_reward(obs, sig, source="", dtype=dtype,
                                     physics_weight=physics_weight)
        eta = rec.eta if rec.eta is not None else (
            rec.t_min_ms / rec.cand_ms if rec.cand_ms else None)
        agg = per_family.setdefault(fam, {"etas": [], "rewards": [], "speedups": [],
                                          "tasks": set(), "pmc": 0, "n": 0})
        if eta is not None:
            agg["etas"].append(eta)
        agg["rewards"].append(rr.reward)
        if rec.speedup is not None:
            agg["speedups"].append(rec.speedup)
        if "no_pmc" not in rr.flags:
            agg["pmc"] += 1
        agg["tasks"].add(rec.task_id)
        agg["n"] += 1
        scored_tasks.add(rec.task_id)

    families_out = {}
    for fam, agg in sorted(per_family.items()):
        families_out[fam] = {
            "n_tasks": len(agg["tasks"]),
            "n_kernels": agg["n"],
            "pmc_kernels": agg["pmc"],
            "median_eta": (median(agg["etas"]) if agg["etas"] else None),
            "median_residual_reward": (median(agg["rewards"]) if agg["rewards"] else None),
            "median_speedup_vs_baseline": (median(agg["speedups"]) if agg["speedups"] else None),
            "tasks": sorted(agg["tasks"]),
        }
    return {
        "split": split.as_dict(),
        "physics_weight": physics_weight,
        "heldout_families_evaluated": sorted(per_family.keys()),
        "scored_tasks": sorted(scored_tasks),
        "per_family": families_out,
        "note": ("offline zero-shot eval; feed a trained checkpoint's measured "
                 "kernels to measure transfer. No training is performed here."),
    }


def load_measures(path: Path) -> list[dict]:
    """Load the ``measures`` list from a p0_sol JSON report (or a bare list)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        return data.get("measures", []) or []
    return data or []


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Zero-shot cross-family generalization eval (no training)")
    ap.add_argument("--heldout", required=True,
                    help="comma-separated held-out families, e.g. attention,moe")
    ap.add_argument("--measures", required=True, help="p0_sol JSON report path")
    ap.add_argument("--physics-weight", type=float, default=DEFAULT_PHYSICS_WEIGHT)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args(argv)

    measures = load_measures(Path(args.measures))
    present = sorted({m.get("task_id") for m in measures if m.get("task_id")})
    present = [t for t in present if family_of(t) is not None]
    heldout = [f.strip() for f in args.heldout.split(",") if f.strip()]
    split = make_holdout_split(heldout, task_ids=present or None)
    result = evaluate_generalization(split, measures, physics_weight=args.physics_weight)

    print(f"# zero-shot cross-family transfer  (held-out: {', '.join(sorted(split.heldout_families))})")
    print(f"# train families: {', '.join(sorted(split.train_families))}")
    print(f"{'family':12s} {'ntask':>5s} {'nkern':>5s} {'eta':>7s} {'resid_rwd':>9s} {'speedup':>8s}")
    print("-" * 52)
    for fam, r in result["per_family"].items():
        eta = f"{r['median_eta']*100:.1f}%" if r["median_eta"] is not None else "-"
        rwd = f"{r['median_residual_reward']:.3f}" if r["median_residual_reward"] is not None else "-"
        sp = f"{r['median_speedup_vs_baseline']:.3f}x" if r["median_speedup_vs_baseline"] is not None else "-"
        print(f"{fam:12s} {r['n_tasks']:5d} {r['n_kernels']:5d} {eta:>7s} {rwd:>9s} {sp:>8s}")
    if not result["per_family"]:
        print("(no held-out kernels found in measures)")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\n[generalization] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
