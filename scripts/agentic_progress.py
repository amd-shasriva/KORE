#!/usr/bin/env python
"""Report live progress of the agentic datagen campaign from its own telemetry.

Reads the per-episode telemetry each node appends, so it reports what the run
actually did rather than what the scheduler thinks it is doing: episodes done,
realized rate per node, the success/repair/attempt mix, the measured-speedup
distribution, and disk consumed per thousand kept trajectories. That last number
is the one that decides whether the campaign fits on the volume, and it is only
knowable from the data already on disk.

Read-only.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        print(f"missing output directory: {out_dir}")
        return 2

    rows = []
    for telemetry in sorted(out_dir.glob("shard_*.telemetry.jsonl")):
        for line in telemetry.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    shards = sorted(out_dir.glob("shard_*.jsonl"))
    data_shards = [p for p in shards if not p.name.endswith(".telemetry.jsonl")]
    total_bytes = sum(p.stat().st_size for p in data_shards)

    categories = collections.Counter(row.get("category", "?") for row in rows)
    kept = sum(1 for row in rows if row.get("kept"))
    errors = sum(1 for row in rows if row.get("error"))
    durations = [row["wall_seconds"] for row in rows if row.get("wall_seconds")]
    speedups = [
        row["best_speedup"] for row in rows
        if isinstance(row.get("best_speedup"), (int, float))
    ]
    teacher_seconds = sum(row.get("teacher_seconds", 0.0) for row in rows)
    env_seconds = sum(row.get("env_seconds", 0.0) for row in rows)
    busy = sum(durations)
    hosts = {row.get("worker") for row in rows}

    free = shutil.disk_usage(str(out_dir)).free
    report = {
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out_dir": str(out_dir),
        "shards": len(data_shards),
        "episodes_recorded": len(rows),
        "kept": kept,
        "errors": errors,
        "by_category": dict(categories.most_common()),
        "episode_seconds_p50": round(_percentile(durations, 0.50), 1),
        "episode_seconds_p90": round(_percentile(durations, 0.90), 1),
        "teacher_share": round(teacher_seconds / busy, 3) if busy else 0.0,
        "env_share": round(env_seconds / busy, 3) if busy else 0.0,
        "bytes_on_disk": total_bytes,
        "gb_per_1k_kept": round(total_bytes / max(kept, 1) * 1000 / 1e9, 2),
        "free_gb": round(free / 1e9, 1),
        "speedup_p50": round(_percentile(speedups, 0.50), 3),
        "speedup_p90": round(_percentile(speedups, 0.90), 3),
        "speedup_p99": round(_percentile(speedups, 0.99), 3),
        "speedup_max": round(max(speedups), 3) if speedups else 0.0,
        "n_benched": len(speedups),
        "n_workers_seen": len(hosts),
    }

    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
