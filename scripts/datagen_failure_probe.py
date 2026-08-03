#!/usr/bin/env python3
"""Why a datagen wave's episodes are failing, split by cause.

Yield alone cannot tell you whether a wave is producing bad kernels or whether
the node is over-subscribed. Those need opposite responses: the first is a task
or teacher problem, the second means backing the worker count off. This splits
the failures so the difference is visible, and reports the recent-episode yield
separately from the lifetime figure so a change in worker count shows up instead
of being averaged away by everything that ran before it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/b05factory/agentic_v2")
    ap.add_argument("--recent", type=int, default=300,
                    help="also report yield over the last N episodes per shard")
    args = ap.parse_args()

    files = [f for f in glob.glob(os.path.join(args.dir, "shard_*.jsonl"))
             if "telemetry" not in f]
    reasons: Counter[str] = Counter()
    total = 0
    useful = 0
    recent_rows: list[dict] = []

    for path in files:
        rows = []
        with open(path) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
        recent_rows.extend(rows[-args.recent:])
        for row in rows:
            total += 1
            if _ok(row):
                useful += 1
            else:
                reasons[_why(row)] += 1

    print(f"episodes      : {total}")
    if total:
        print(f"reached correct: {useful} ({100.0 * useful / total:.1f}%)")
    print("\nfailure causes:")
    for reason, n in reasons.most_common(10):
        print(f"  {reason:24} {n}")

    if recent_rows:
        r_ok = sum(1 for r in recent_rows if _ok(r))
        print(f"\nrecent window ({len(recent_rows)} episodes): "
              f"{r_ok} correct ({100.0 * r_ok / len(recent_rows):.1f}%)")
    return 0


def _ok(row: dict) -> bool:
    turns = row.get("turns")
    if isinstance(turns, list) and any(
            isinstance(t, dict) and t.get("correct") for t in turns):
        return True
    return bool(row.get("success") or row.get("final_correct")
                or row.get("any_correct"))


def _why(row: dict) -> str:
    """Best available cause for an episode that never reached correctness.

    Infrastructure causes (timeout, OOM, the process being killed) are the ones
    that indicate over-subscription rather than a bad kernel, so they are named
    separately instead of collapsing into a single failure bucket.
    """
    turns = row.get("turns") or []
    texts = []
    for t in turns:
        if isinstance(t, dict):
            texts.append(str(t.get("error_text") or t.get("error") or ""))
    blob = " ".join(texts).lower() + " " + str(row.get("error", "")).lower()
    if "timeout" in blob or "timed out" in blob:
        return "timeout (over-subscription?)"
    if "out of memory" in blob or "oom" in blob:
        return "oom (over-subscription?)"
    if "killed" in blob or "signal" in blob:
        return "killed (over-subscription?)"
    if "compil" in blob or "syntaxerror" in blob or "nameerror" in blob:
        return "compile error"
    if "mismatch" in blob or "allclose" in blob or "snr" in blob:
        return "incorrect result"
    if not turns:
        return "no turns recorded"
    return "never reached correct"


if __name__ == "__main__":
    raise SystemExit(main())
