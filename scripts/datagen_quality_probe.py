#!/usr/bin/env python3
"""Live yield check on an in-flight agentic datagen run.

The number that matters while a wave is running is not how many trajectories
exist but how many of them will survive filtering, because the previous wave
produced ~11.2k episodes of which roughly 70% never reached a correct kernel and
therefore contributed nothing to the mixture. Reading that early is the
difference between catching a bad wave in ten minutes and finding out after a
day of GPU time.

Counts a trajectory as useful on the same basis the step-centric extractor does:
it reached correctness at least once. Telemetry shards are excluded explicitly --
they match shard_*.jsonl too, and counting them once inflated a wave's size by
2.34x.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/b05factory/agentic_v2")
    args = ap.parse_args()

    files = [f for f in glob.glob(os.path.join(args.dir, "shard_*.jsonl"))
             if "telemetry" not in f]
    total = 0
    useful = 0
    correct_turns = 0
    all_turns = 0
    keys: set[str] = set()
    for path in files:
        with open(path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001 - a partial last line is normal
                    continue
                total += 1
                keys |= set(row.keys())
                turns = row.get("turns") or []
                if isinstance(turns, list):
                    all_turns += len(turns)
                    hits = sum(1 for t in turns
                               if isinstance(t, dict) and t.get("correct"))
                    correct_turns += hits
                    if hits:
                        useful += 1
                        continue
                if (row.get("final_correct") or row.get("success")
                        or row.get("any_correct")):
                    useful += 1

    print(f"shards        : {len(files)}")
    print(f"trajectories  : {total}")
    if total:
        print(f"reached correct: {useful} ({100.0 * useful / total:.1f}%)")
    if all_turns:
        print(f"turns         : {all_turns}  correct: {correct_turns} "
              f"({100.0 * correct_turns / all_turns:.1f}%)")
    print(f"row keys      : {sorted(keys)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
