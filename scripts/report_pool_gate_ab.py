#!/usr/bin/env python
"""The before/after number for the pool-admission gate.

Reads each arm's telemetry and reports the rate that matters: the fraction of
episodes that never reached a correct kernel (``cat:attempt``), which is the 70%
the campaign was losing. Also reports the step-centric yield, because a lower
attempt rate is only worth having if it converts into training rows.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--json-out", default="")
    return p.parse_args(argv)


def _rows(path):
    out = []
    if not pathlib.Path(path).is_file():
        return out
    with pathlib.Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def summarize(arm_dir: pathlib.Path) -> dict:
    telemetry = _rows(arm_dir / "shard_000.telemetry.jsonl")
    trajectories = [r for r in _rows(arm_dir / "shard_000.jsonl")
                    if not r.get("_dropped")]
    cats = collections.Counter(str(r.get("category")) for r in telemetry)
    attempted = len(telemetry)
    correct_reached = sum(1 for r in telemetry
                          if str(r.get("category")) in ("success", "repair"))
    speedups = [r["best_speedup"] for r in telemetry
                if isinstance(r.get("best_speedup"), (int, float))]
    speedups.sort()

    steps = 0
    with_steps = 0
    try:
        from kore.data.step_centric import decompose

        rows, stats = decompose(trajectories)
        steps, with_steps = stats["steps"], stats["with_steps"]
    except Exception:  # noqa: BLE001 - reporting must not depend on the importer
        pass

    return {
        "attempted": attempted,
        "by_category": dict(cats),
        "attempt_rate": round(cats["attempt"] / attempted, 4) if attempted else None,
        "error_rate": round(cats["error"] / attempted, 4) if attempted else None,
        "reached_correct_rate": round(correct_reached / attempted, 4)
        if attempted else None,
        "trajectories": len(trajectories),
        "step_centric_rows": steps,
        "trajectories_with_steps": with_steps,
        "median_best_speedup": speedups[len(speedups) // 2] if speedups else None,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

    out_dir = pathlib.Path(args.out_dir)
    result = {arm: summarize(out_dir / arm) for arm in ("before", "after")}

    print()
    print(f"{'arm':8s} {'episodes':>9s} {'attempt%':>9s} {'error%':>8s} "
          f"{'correct%':>9s} {'steps':>7s} {'med speedup':>12s}")
    for arm in ("before", "after"):
        s = result[arm]
        if not s["attempted"]:
            print(f"{arm:8s} {'no data':>9s}")
            continue

        def pct(key):
            return f"{100.0 * s[key]:>8.1f}%" if s[key] is not None else f"{'-':>9s}"

        print(f"{arm:8s} {s['attempted']:>9d} {pct('attempt_rate')} "
              f"{pct('error_rate')[:8]:>8s} {pct('reached_correct_rate')} "
              f"{s['step_centric_rows']:>7d} "
              f"{s['median_best_speedup'] if s['median_best_speedup'] else '-':>12}")

    before, after = result["before"], result["after"]
    if before["attempt_rate"] is not None and after["attempt_rate"] is not None:
        delta = before["attempt_rate"] - after["attempt_rate"]
        print()
        print(f"attempt-rate change: {100 * before['attempt_rate']:.1f}% -> "
              f"{100 * after['attempt_rate']:.1f}%  ({100 * delta:+.1f} points)")
        result["attempt_rate_delta"] = round(delta, 4)

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
