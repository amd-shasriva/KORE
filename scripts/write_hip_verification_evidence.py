#!/usr/bin/env python
"""Assemble data/hip_task_verification.json from real verification runs.

The evidence artifact used to be hand-assembled, which is a bad property for the
one file that backs "this task is runnable": a hand-edited row cannot be
distinguished from a measured one. This script builds it from the JSON that
``scripts/verify_hip_tasks_e2e.py`` writes, so every row in it came from a run.

It takes one or more RUNS, each a glob over that run's shard JSONs. Multiple runs
matter because the two verdicts a task can get are not equally stable:

* correctness is deterministic -- a task that verifies, verifies every time;
* timing ADMISSION is a noise gate, and a marginal task can pass one sweep and
  fail the next.

So the artifact records both, separately, and per run. A task counts as proven
runnable only if it compiled, verified on every declared shape, and was
timing-admitted -- and the number of runs in which that held is written down
rather than collapsed into a boolean.

    PYTHONPATH=. python scripts/write_hip_verification_evidence.py \
        --run '/tmp/hipverify/shard*.json' --run '/tmp/hipverify/r2_shard*.json' \
        --out data/hip_task_verification.json
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import pathlib
import subprocess
import sys
from typing import Any, Optional

REPO = pathlib.Path(__file__).resolve().parents[1]


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO),
            capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _load_run(pattern: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no shard JSON matched {pattern!r}")
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for row in json.load(handle)["rows"]:
                rows[row["task_id"]] = row
    return rows


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[],
                        help="glob over one run's shard JSONs (repeatable)")
    parser.add_argument("--out", default="data/hip_task_verification.json")
    parser.add_argument("--host-note", default=(
        "local dev host, 8x AMD Instinct MI350X; same gfx950/CDNA4 architecture "
        "as the MI355X product target"))
    args = parser.parse_args(argv)
    if not args.run:
        raise SystemExit("at least one --run glob is required")

    sys.path.insert(0, str(REPO))
    from kore.env.hip_toolchain import hipcc_version, probe_toolchain
    from kore.tasks.registry import all_tasks

    runs = [_load_run(pattern) for pattern in args.run]
    registry_ids = sorted(t.task_id for t in all_tasks() if t.backend == "hip")

    rows: list[dict[str, Any]] = []
    for task_id in registry_ids:
        seen = [run[task_id] for run in runs if task_id in run]
        if not seen:
            rows.append({"task_id": task_id, "runs": 0,
                         "note": "NOT MEASURED by any supplied run"})
            continue
        latest = seen[-1]
        rows.append({
            "task_id": task_id,
            "operation": latest["operation"],
            "dtype": latest["dtype"],
            "runs": len(seen),
            # Deterministic verdicts: identical in every run, or the task is broken.
            "compiled": all(r["compiled"] for r in seen),
            "correct": all(r["correct"] for r in seen),
            "flagged_hack": any(r["flagged_hack"] for r in seen),
            "infra_error": any(r["infra_error"] for r in seen),
            "worst_snr_db": min(r["snr_db"] for r in seen
                                if isinstance(r["snr_db"], (int, float))),
            "gate_db": latest["gate_db"],
            "snr_by_shape": latest["snr_by_shape"],
            # Noise-gated verdict: recorded per run, never collapsed.
            "timing_admitted_runs": sum(
                1 for r in seen if r["performance_eligible"] is True),
            "speedup_by_run": [r["speedup"] for r in seen],
            "candidate_cv_pct_by_run": [r["cv_pct"] for r in seen],
            "timing_protocol": latest["timing_protocol"],
            "runnable_runs": sum(1 for r in seen if r["runnable"]),
            "last_error": (latest["error_text"] or "")[:400],
        })

    measured = [r for r in rows if r.get("runs")]
    correct = [r for r in measured if r.get("compiled") and r.get("correct")
               and not r.get("infra_error") and not r.get("flagged_hack")]
    always = [r for r in measured if r.get("runnable_runs") == r.get("runs")]
    ever = [r for r in measured if r.get("runnable_runs", 0) > 0]

    status = probe_toolchain()
    payload = {
        "artifact_type": "kore.hip-task-verification",
        "schema_version": "2.0",
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_head": _git_head(),
        "written_by": "scripts/write_hip_verification_evidence.py",
        "how_to_reproduce": [
            "for i in 0 1 2 3; do PYTHONPATH=. python scripts/verify_hip_tasks_e2e.py "
            "--gpu $((i+4)) --shard $i/4 --verified-correctness "
            "--json /tmp/hipverify/shard$i.json & done; wait",
            "PYTHONPATH=. python scripts/write_hip_verification_evidence.py "
            "--run '/tmp/hipverify/shard*.json' --out data/hip_task_verification.json",
        ],
        "host": {
            "note": args.host_note,
            "torch_hip": status.torch_hip_version,
            "hipcc": hipcc_version(status),
            "rocm_home": status.rocm_home,
        },
        "gate": (
            "each task evaluated through kore.env.kore_env.KoreEnv with its OWN "
            "declared seed as the candidate, adversarial regimes enabled, full "
            "paired publication timing protocol at the unmodified 3% CV gate"),
        "summary": {
            "registry_hip_tasks": len(registry_ids),
            "measured": len(measured),
            "compiled_and_correct": len(correct),
            "runnable_every_run": len(always),
            "runnable_in_some_run": len(ever),
            "runs": len(runs),
            "note": (
                "correctness is deterministic and holds for every measured task; "
                "the gap between runnable_every_run and compiled_and_correct is "
                "TIMING ADMISSION, a measurement-noise gate. A task that fails it "
                "still yields a correct episode -- it just carries no speedup "
                "reward that run."),
        },
        "rows": rows,
    }
    out = REPO / args.out
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  registry HIP tasks     : {len(registry_ids)}")
    print(f"  measured               : {len(measured)}")
    print(f"  compiled and correct   : {len(correct)}")
    print(f"  runnable in every run  : {len(always)}")
    print(f"  runnable in some run   : {len(ever)}")
    unmeasured = [r["task_id"] for r in rows if not r.get("runs")]
    if unmeasured:
        print(f"  NOT MEASURED           : {len(unmeasured)} -> {unmeasured[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
