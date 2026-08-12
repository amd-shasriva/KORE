#!/usr/bin/env python
"""Stage 3 of the v5 build: recover the generation signal SFT never read.

Two large verified artifacts reach training as nothing at all.

**Ranked groups.** ``build_sft`` consumes repair and win records and silently
drops :class:`RankedGroupRecord`, so 71,095 groups holding hundreds of thousands
of measured candidate kernels are used to teach *ranking* through DPO and never
once to teach *generation*. Since we are not training a preference model, that is
currently worth nothing. Each group's rank-0 candidate is the robustly-best
correct kernel for its shape; framing a slower correct sibling as the parent
turns the group into an ordinary optimization demonstration. The gate is a real
gain over the SIBLING rather than over the vendor baseline -- the vendor gate is
what held real wins down to 5,497, and beating AMD's own library is a far higher
bar than demonstrating a correct improvement.

**Agentic trajectories.** 108,822 multi-turn episodes under ``b05factory`` are
the only records carrying per-turn correctness and speedup, and the gather stage
never opened them because it reads only ``repair``, ``wins`` and ``groups``.
Half the arena is "here is a working kernel, make it faster under execution
feedback", which is exactly an agentic episode and exactly what a win record --
already flattened to a single system/user/assistant turn, with no provenance --
cannot represent.

A trajectory trained whole teaches imitation of a search; what the RL stage will
sample is a local improver. So each becomes up to N-1 examples keeping only
correctness-preserving, high-gain revisions, and an episode that yielded no steps
(a first-turn win has no parent to improve on) is emitted whole, truncated at the
winning turn. Never both, because a step row's messages are a prefix of the full
one and content hashing does not catch a prefix.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

REPO = Path("/home/shasriva/Kore-RL/KORE")
sys.path.insert(0, str(REPO))

#: A group must demonstrate a real gain over its own sibling. 1.05 is the floor
#: at which "faster" is a claim about the kernel rather than about the clock.
MIN_SPEEDUP = 1.05
#: Candidates below this are not clearly correct. Group candidates sit at 76-999 dB.
SNR_GATE = 40.0
#: No single task may dominate -- the corpus is long-tailed and a handful of
#: heavily-mined ops would otherwise supply most of the slice. But this bounds
#: DISTINCT kernels, not repeats: each ranked group has its own candidate set, so
#: its rank-0 is a different kernel even when the task repeats. At 12 it was
#: rejecting 23,040 groups, which is coverage thrown away rather than redundancy
#: removed. Exact duplicates are caught later by content hash, so the cap only
#: has to stop domination, and 40 does that while keeping the tail.
PER_TASK_CAP = 40
#: Trajectory directories, all under data/b05factory.
AGENTIC_DIRS = ("agentic_mt", "agentic_v2")


def as_dict(rec) -> dict:
    if isinstance(rec, dict):
        return rec
    if is_dataclass(rec):
        try:
            return asdict(rec)
        except Exception:  # noqa: BLE001 - a non-serialisable field
            pass
    return {k: getattr(rec, k) for k in dir(rec)
            if not k.startswith("_") and not callable(getattr(rec, k, None))}


def mint_gold(groups, policy: str, per_task_cap: int, arena) -> tuple[list, dict]:
    from kore.data.gold_wins import mint_gold_win
    from kore.data.v5_emit import cheats
    from kore.data.v5_policy import admits, credible_speedup, dialect

    per_task: collections.Counter = collections.Counter()
    reasons: collections.Counter = collections.Counter()
    gold = []
    for g in groups:
        gd = as_dict(g)
        tid = str(gd.get("task_id") or "?")
        if per_task[tid] >= per_task_cap:
            reasons["per_task_cap"] += 1
            continue
        # Stage 1 filtered under `strict` and without a benchmark screen; re-asking
        # here is what applies the arena index and lets `audited` readmit a group
        # the classifier rejected only for an unknown operation.
        ok, why = admits(gd, policy, arena)
        if not ok:
            reasons[f"blocked::{why.split(':')[0]}"] += 1
            continue
        try:
            w = mint_gold_win(gd, None, SNR_GATE, MIN_SPEEDUP)
        except Exception as exc:  # noqa: BLE001 - one bad group must not abort
            reasons[f"error::{type(exc).__name__}"] += 1
            continue
        if w is None:
            reasons["no_qualifying_pair"] += 1
            continue
        if not credible_speedup(w.speedup):
            reasons["implausible_speedup"] += 1
            continue
        why = cheats(getattr(w, "final_source", "") or "")
        if why:
            reasons[f"cheat::{why.split(':')[0]}"] += 1
            continue
        gold.append(w)
        per_task[tid] += 1

    d = collections.Counter(dialect(w.task_id) for w in gold)
    sp = sorted(w.speedup for w in gold if w.speedup)
    stats = {
        "gold_wins": len(gold), "tasks": len(per_task),
        "by_dialect": dict(d), "reasons": dict(reasons.most_common(10)),
        "speedup_median": round(sp[len(sp) // 2], 3) if sp else None,
        "speedup_p90": round(sp[int(len(sp) * 0.9)], 3) if sp else None,
        "speedup_max": round(sp[-1], 3) if sp else None,
    }
    return gold, stats


def decompose_agentic(policy: str, per_task_cap: int, arena) -> tuple[list, dict]:
    """Stream the trajectory directories; they total 3.9 GB."""
    from kore.data.step_centric import extract_full_trajectory, extract_steps
    from kore.data.v5_emit import (assistant_turns, cheats, flatten_history,
                                   system_prompt)
    from kore.data.v5_policy import admits, credible_speedup, dialect

    rows: list[dict] = []
    per_task: collections.Counter = collections.Counter()
    stats: collections.Counter = collections.Counter()

    for sub in AGENTIC_DIRS:
        d = REPO / "data" / "b05factory" / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jsonl")):
            with p.open(errors="ignore") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001 - torn line after a kill
                        stats["unparseable"] += 1
                        continue
                    stats["rows"] += 1
                    tid = str(rec.get("task_id") or "")
                    ok, why = admits(rec, policy, arena)
                    if not ok:
                        stats[f"blocked::{why.split(':')[0]}"] += 1
                        continue
                    if per_task[tid] >= per_task_cap:
                        stats["per_task_cap"] += 1
                        continue

                    steps = extract_steps(rec)
                    emitted = 0
                    for s in steps:
                        if not credible_speedup(s.speedup_after):
                            stats["implausible_step"] += 1
                            continue
                        row = s.to_row()
                        # The trainer puts full loss on EVERY assistant turn and
                        # offers no per-turn opt-out, so a step row's earlier
                        # assistant turns -- which are by construction the
                        # revisions that were rejected -- would be trained as
                        # targets. Collapse the history into the user turn so the
                        # model still sees it but is only asked to produce the
                        # revision worth imitating.
                        before = assistant_turns(row["messages"])
                        row["messages"] = flatten_history(row["messages"])
                        if before > 1:
                            stats["flattened"] += 1
                        target = (row["messages"][-1].get("content") or "")
                        why = cheats(target)
                        if why:
                            stats[f"cheat::{why.split(':')[0]}"] += 1
                            continue
                        row["_dialect"] = dialect(tid)
                        row["_provenance"] = {"kind": "step_centric"}
                        rows.append(row)
                        stats[f"kind_{s.kind}"] += 1
                        emitted += 1
                    if emitted:
                        stats["with_steps"] += 1
                        per_task[tid] += emitted
                        continue
                    # No usable revision: keep the whole episode only if it won.
                    full = extract_full_trajectory(rec)
                    if full is None:
                        stats["no_usable_row"] += 1
                        continue
                    if not credible_speedup(full.best_speedup):
                        stats["implausible_full"] += 1
                        continue
                    row = full.to_row()
                    before = assistant_turns(row["messages"])
                    row["messages"] = flatten_history(row["messages"])
                    if before > 1:
                        stats["flattened"] += 1
                    target = (row["messages"][-1].get("content") or "")
                    why = cheats(target)
                    if why:
                        stats[f"cheat::{why.split(':')[0]}"] += 1
                        continue
                    row["_dialect"] = dialect(tid)
                    row["_provenance"] = {"kind": "full_trajectory"}
                    rows.append(row)
                    stats["residual_full"] += 1
                    per_task[tid] += 1

    dial = collections.Counter(r.get("_dialect") for r in rows)
    stats["tasks"] = len(per_task)
    return rows, {**{k: v for k, v in stats.items()}, "by_dialect": dict(dial)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=str(REPO / "runs/v5_build/stage1.pkl"))
    ap.add_argument("--out-dir", default=str(REPO / "runs/v5_build"))
    ap.add_argument("--heldout-policy", choices=("strict", "audited"), default="strict")
    ap.add_argument("--per-task-cap", type=int, default=PER_TASK_CAP)
    ap.add_argument("--skip-agentic", action="store_true")
    ap.add_argument("--arena-index", default=str(REPO / "data/arena_contamination.json"))
    ap.add_argument("--allow-unscreened", action="store_true",
                    help="build without benchmark screening (not for training data)")
    args = ap.parse_args()

    from kore.data.arena_index import ArenaIndex
    arena = None
    if Path(args.arena_index).is_file():
        arena = ArenaIndex.load(args.arena_index)
        print(f"arena screen: {arena}")
    elif not args.allow_unscreened:
        print(f"ERROR: no arena index at {args.arena_index}; refusing to build "
              f"unscreened training data.", file=sys.stderr)
        return 2

    with open(args.stage1, "rb") as fh:
        stage1 = pickle.load(fh)
    groups = stage1["groups"]
    print(f"policy={args.heldout_policy}  groups={len(groups):,}\n", flush=True)

    gold, gstats = mint_gold(groups, args.heldout_policy, args.per_task_cap, arena)
    print(f"=== gold wins: {gstats['gold_wins']:,} across {gstats['tasks']:,} tasks ===")
    print(f"    dialect  {gstats['by_dialect']}")
    print(f"    speedup  median={gstats['speedup_median']}  p90={gstats['speedup_p90']}  "
          f"max={gstats['speedup_max']}")
    for k, v in list(gstats["reasons"].items())[:6]:
        print(f"    {k:<32} {v:,}")

    step_rows, sstats = ([], {})
    if not args.skip_agentic:
        print("\n=== streaming agentic trajectories (3.9 GB) ===", flush=True)
        step_rows, sstats = decompose_agentic(args.heldout_policy, args.per_task_cap, arena)
        print(f"=== step-centric: {len(step_rows):,} rows across "
              f"{sstats.get('tasks', 0):,} tasks ===")
        print(f"    dialect  {sstats.get('by_dialect')}")
        for k in ("rows", "with_steps", "kind_fix", "kind_speedup", "residual_full",
                  "no_usable_row", "per_task_cap", "implausible_step"):
            if sstats.get(k):
                print(f"    {k:<24} {sstats[k]:,}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.heldout_policy
    with (out_dir / f"stage3_{tag}.pkl").open("wb") as fh:
        pickle.dump({"gold_wins": gold, "step_rows": step_rows},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / f"stage3_{tag}.meta.json").write_text(
        json.dumps({"policy": tag, "gold": gstats, "step": sstats}, indent=2, default=str))
    print(f"\nwrote {out_dir / f'stage3_{tag}.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
