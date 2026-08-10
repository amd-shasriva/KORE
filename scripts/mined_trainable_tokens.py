#!/usr/bin/env python3
"""What the mined corpus would add to a mixture, in trainable tokens.

Counts both record types the miner writes, because both reach the model and an
earlier version of this script counted only one:

``ranked_group``
    A win group: one task, several ranked candidates, and the counters the gate
    measured. Only the kernel source inside the candidates is trainable; the
    shapes, wall times and hardware counters wrapped around it are several
    times larger than the code, which is why sizing this from file bytes
    overstates it badly.

``repair`` / ``win``
    A conversation -- parent kernel, the error it produced, the fix -- carried
    in ``messages``. This is the shape v4 weighted at 2.0 in its loss, so it is
    emphatically training data.

Missing the second kind understated the corpus threefold: HIP read as 59.1M
when it was 182.5M, and the parity plan was built on the wrong number. The
error was easy to make because the base pass writes repair records first and
groups only afterwards, so a freshly started stream shows nothing but repair
and scores as zero.

Two figures for the group half, because the build has latitude between them:
the floor counts the best candidate only, the ceiling counts every candidate,
which is what a step-centric build emits when it turns an episode into one row
per revision. Repair rows are counted once either way -- there is no expansion
choice to make about a conversation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

#: Prompt the row carries in addition to the code, for group records. The v4
#: kernel rows average well above the code alone because each restates the
#: task, the reference and the arch.
PROMPT_CHARS = 2000

SKIP = ("seed_attempts", "repair_attempts", "events", "telemetry", "manifest")


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
        "v5frontierhip", "v5frontier_twins", "v5hardpool", "v5hip", "v5hippool",
        "v5pool_flydsl", "v5frontier", "v5pooltriton", "v5pool", "hipwave1",
        "hip_pool",
    ])
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    ap.add_argument("--by-root", action="store_true")
    args = ap.parse_args()

    cpt = args.chars_per_token
    floor: dict[str, int] = defaultdict(int)
    ceil: dict[str, int] = defaultdict(int)
    repair: dict[str, int] = defaultdict(int)
    groups: dict[str, int] = defaultdict(int)
    convos: dict[str, int] = defaultdict(int)
    per_root: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for root in args.roots:
        base = Path(args.data_dir) / root
        if not base.is_dir():
            continue
        for path in base.glob("**/*.jsonl"):
            if any(k in str(path) for k in SKIP):
                continue
            with path.open("r", errors="ignore") as fh:
                for line in fh:
                    if len(line) < 2:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    kind = row.get("type")
                    d = dialect_of(str(row.get("task_id") or ""), root)

                    if kind == "ranked_group":
                        srcs = [c.get("source") for c in (row.get("candidates") or [])
                                if isinstance(c, dict) and isinstance(c.get("source"), str)]
                        if not srcs:
                            continue
                        groups[d] += 1
                        floor[d] += max(len(x) for x in srcs) + PROMPT_CHARS
                        ceil[d] += sum(len(x) for x in srcs) + PROMPT_CHARS * len(srcs)
                        per_root[root][d] += (sum(len(x) for x in srcs)
                                              + PROMPT_CHARS * len(srcs)) / cpt
                    elif kind in ("repair", "win"):
                        msgs = row.get("messages")
                        if not isinstance(msgs, list):
                            continue
                        n = sum(len(m.get("content", "")) for m in msgs
                                if isinstance(m, dict))
                        if not n:
                            continue
                        convos[d] += 1
                        repair[d] += n
                        per_root[root][d] += n / cpt

    print(f"\n{'dialect':<10}{'groups':>8}{'convos':>9}"
          f"{'group floor':>14}{'group ceiling':>15}{'conversations':>15}"
          f"{'TOTAL (ceil)':>14}")
    print("-" * 85)
    tot = {}
    for d in ("Triton", "HIP", "FlyDSL"):
        total = (ceil[d] + repair[d]) / cpt
        tot[d] = total
        print(f"{d:<10}{groups[d]:>8,}{convos[d]:>9,}"
              f"{floor[d]/cpt:>14,.0f}{ceil[d]/cpt:>15,.0f}"
              f"{repair[d]/cpt:>15,.0f}{total:>14,.0f}")

    tri = tot.get("Triton", 0) or 1
    print("\nparity of mined data against mined Triton:")
    for d in ("HIP", "FlyDSL"):
        print(f"  {d:<8}{100*tot[d]/tri:>7.1f}%")

    if args.by_root:
        print("\nby root (ceiling tokens):")
        for root in sorted(per_root, key=lambda r: -sum(per_root[r].values())):
            v = per_root[root]
            print(f"  {root:<22}" + "  ".join(
                f"{k}={v[k]/1e6:>7.2f}M" for k in ("Triton", "HIP", "FlyDSL")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
