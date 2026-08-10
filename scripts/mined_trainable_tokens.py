#!/usr/bin/env python3
"""What the mined win-groups would add to a mixture, in trainable tokens.

A mined record is not a training row. It is a win group: one task, a handful of
ranked candidates, and the counters the gate measured. Only the kernel source
inside those candidates reaches the model, and the record around it -- shapes,
wall times, hardware counters, the parent's counters -- is several times larger
than the code. Sizing the next mixture from file bytes therefore overstates it
badly, which is how the mined roots looked like 1.5B tokens against a v4
mixture that is 540M in total.

Two figures, because the build has latitude between them. The floor counts the
best candidate only, as a single-turn example. The ceiling counts every
candidate, which is what a step-centric build emits when it turns an episode
into one row per revision -- the shape v4 used for its 34,806 AMD-native rows.
The truth is in between and depends on the builder, so both are printed rather
than one being passed off as the answer.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

#: Prompt the row carries in addition to the code. The v4 kernel rows average
#: well above the code alone because each one restates the task, the reference
#: and the arch; 2k chars is the observed floor for that preamble.
PROMPT_CHARS = 2000


def dialect_of(task_id: str, root: str) -> str:
    if task_id.endswith("__hip") or task_id.endswith("__hipf"):
        return "HIP"
    if task_id.endswith("__flydsl"):
        return "FlyDSL"
    if task_id.startswith("hip_"):
        return "HIP"
    low = root.lower()
    if "flydsl" in low:
        return "FlyDSL"
    if "hip" in low:
        return "HIP"
    return "Triton"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--roots", nargs="*", default=[
        "v5frontier_twins", "v5hippool", "v5hip", "v5pool_flydsl",
        "v5frontier", "v5pooltriton", "v5pool", "hipwave1", "hip_pool",
    ])
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    args = ap.parse_args()

    floor: dict[str, int] = defaultdict(int)
    ceil: dict[str, int] = defaultdict(int)
    recs: dict[str, int] = defaultdict(int)
    cands: dict[str, int] = defaultdict(int)

    for root in args.roots:
        base = Path(args.data_dir) / root
        if not base.is_dir():
            continue
        for path in base.glob("**/*.jsonl"):
            s = str(path)
            if any(k in s for k in ("seed_attempts", "repair_attempts",
                                    "events", "telemetry", "manifest")):
                continue
            with path.open("r", errors="ignore") as fh:
                for line in fh:
                    if len(line) < 2:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    tid = str(row.get("task_id") or "")
                    d = dialect_of(tid, root)
                    cs = row.get("candidates")
                    if not isinstance(cs, list) or not cs:
                        continue
                    srcs = [c.get("source") for c in cs
                            if isinstance(c, dict) and isinstance(c.get("source"), str)]
                    if not srcs:
                        continue
                    recs[d] += 1
                    cands[d] += len(srcs)
                    floor[d] += max(len(x) for x in srcs) + PROMPT_CHARS
                    ceil[d] += sum(len(x) for x in srcs) + PROMPT_CHARS * len(srcs)

    cpt = args.chars_per_token
    print(f"\n{'dialect':<10}{'groups':>9}{'candidates':>12}"
          f"{'floor tokens':>15}{'ceiling tokens':>16}")
    print("-" * 62)
    for d in ("Triton", "HIP", "FlyDSL"):
        print(f"{d:<10}{recs[d]:>9,}{cands[d]:>12,}"
              f"{floor[d]/cpt:>15,.0f}{ceil[d]/cpt:>16,.0f}")
    print(f"{'TOTAL':<10}{sum(recs.values()):>9,}{sum(cands.values()):>12,}"
          f"{sum(floor.values())/cpt:>15,.0f}{sum(ceil.values())/cpt:>16,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
