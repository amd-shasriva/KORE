#!/usr/bin/env python
"""Probe the agentic multi-turn trajectories: how many are worth training on.

These 108,822 episodes are the only multi-turn data the pipeline ever produced
and the only records that carry per-turn correctness and speedup. Half the arena
is "here is a working kernel, make it faster under execution feedback", which is
precisely the shape of an agentic episode and precisely what a flattened
single-turn win record cannot represent.

Streams rather than loads: the three directories are 3.9 GB.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path("/home/shasriva/Kore-RL/KORE")
sys.path.insert(0, str(REPO))


def dialect(task_id: str) -> str:
    t = task_id or ""
    if t.endswith(("__hip", "__hipf")) or t.startswith("hip_"):
        return "HIP"
    if t.endswith("__flydsl"):
        return "FlyDSL"
    return "Triton"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=["agentic", "agentic_mt", "agentic_v2"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from kore.data.step_centric import extract_full_trajectory, extract_steps
    from kore.tasks.registry import is_heldout_record

    stats: collections.Counter = collections.Counter()
    per_dir: dict[str, collections.Counter] = {}
    step_gain: list[float] = []
    tasks: set[str] = set()
    step_tasks: set[str] = set()
    dial = collections.Counter()

    for sub in args.dirs:
        d = REPO / "data" / "b05factory" / sub
        if not d.is_dir():
            continue
        c: collections.Counter = collections.Counter()
        for p in sorted(d.glob("*.jsonl")):
            with p.open(errors="ignore") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001 - torn line after a kill
                        c["unparseable"] += 1
                        continue
                    c["rows"] += 1
                    if args.limit and c["rows"] > args.limit:
                        break
                    tid = str(rec.get("task_id") or "")
                    tasks.add(tid)
                    try:
                        if is_heldout_record(rec):
                            c["heldout"] += 1
                            continue
                    except Exception:  # noqa: BLE001
                        pass
                    if rec.get("success"):
                        c["success"] += 1
                    prov = rec.get("provenance") or {}
                    if prov.get("turn_correct"):
                        c["has_turn_correct"] += 1
                    if prov.get("turn_speedups"):
                        c["has_turn_speedups"] += 1
                    if any(bool(x) for x in (prov.get("turn_correct") or [])):
                        c["reached_correct"] += 1

                    steps = extract_steps(rec)
                    if steps:
                        c["with_steps"] += 1
                        c["steps"] += len(steps)
                        step_tasks.add(tid)
                        dial[dialect(tid)] += len(steps)
                        for s in steps:
                            c[f"kind_{s.kind}"] += 1
                            step_gain.append(s.gain)
                    else:
                        full = extract_full_trajectory(rec)
                        if full is not None:
                            c["residual_full"] += 1
                            dial[dialect(tid)] += 1
            if args.limit and c["rows"] > args.limit:
                break
        per_dir[sub] = c
        stats.update(c)

    print("=== agentic trajectory probe ===")
    hdr = ("rows", "heldout", "success", "reached_correct", "with_steps", "steps",
           "kind_fix", "kind_speedup", "residual_full")
    print(f"{'dir':<12}" + "".join(f"{h:>16}" for h in hdr))
    for sub, c in per_dir.items():
        print(f"{sub:<12}" + "".join(f"{c.get(h,0):>16,}" for h in hdr))
    print(f"{'TOTAL':<12}" + "".join(f"{stats.get(h,0):>16,}" for h in hdr))

    usable = stats.get("steps", 0) + stats.get("residual_full", 0)
    print(f"\nusable step-centric rows : {usable:,}")
    print(f"distinct tasks (all)     : {len(tasks):,}")
    print(f"distinct tasks (w/ steps): {len(step_tasks):,}")
    print(f"dialect of usable rows   : HIP={dial['HIP']:,} Triton={dial['Triton']:,} "
          f"FlyDSL={dial['FlyDSL']:,}")
    if step_gain:
        step_gain.sort()
        n = len(step_gain)
        print(f"step gain  median={step_gain[n//2]:.3f}  p90={step_gain[int(n*.9)]:.3f}  "
              f"max={step_gain[-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
