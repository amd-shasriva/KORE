#!/usr/bin/env python
"""Gate teacher-written HIP seeds on real gfx950 before any datagen runs on them.

``verify_hip_seeds.py`` gates the 65 hand-authored ops by reading
``hip_ops.HIP_OPS``, so it cannot see a materialized pool task. These seeds need
the same gate for the same reason: a seed that does not compile, or cannot clear
the task's own SNR threshold, is worse than no task at all. Datagen against it
can only ever score zero, and it reports as a model error rather than as the
broken task it is -- which is how a task-resolution bug once produced a 97% error
rate while claiming 13,277 episodes/hour.

Verification is deliberately the same path the environment uses: stage the seed
as the candidate artifact the backend declares (``kernel.hip``) and run the task's
own driver with ``--impl candidate``. Anything else would be testing a different
thing from what datagen will run.

    python scripts/verify_pool_hip_seeds.py --root data/pool_hip --json gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def verify_one(task_dir: Path, timeout: int) -> dict:
    """Compile and check one seed, returning a verdict row."""
    cfg = json.loads((task_dir / "task.yaml").read_text())
    rec = {"task_id": cfg.get("task_id"), "hip_twin_of": cfg.get("hip_twin_of"),
           "family": cfg.get("op_family")}
    seed = task_dir / cfg.get("seed_kernel_name", "seed_hip.hip")
    if not seed.is_file():
        return {**rec, "status": "no_seed"}

    # The driver reads the candidate under the backend's declared filename, so
    # stage it there rather than pointing the driver at the seed: datagen will
    # present candidates exactly this way.
    from kore.env.hip_toolchain import CANDIDATE_FILENAMES, HIP_BACKEND

    shutil.copy(seed, task_dir / CANDIDATE_FILENAMES[HIP_BACKEND])

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}{os.pathsep}{env.get('PYTHONPATH', '')}"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "driver.py", "--impl", "candidate"],
            cwd=str(task_dir), capture_output=True, text=True,
            timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {**rec, "status": "timeout", "seconds": timeout}
    out = (proc.stdout or "") + (proc.stderr or "")
    rec["seconds"] = round(time.time() - t0, 1)

    # The driver prints the verdict; trust it rather than re-deriving one here,
    # since it is the same authority datagen and the reward path use.
    low = out.lower()
    if "allclose: pass" in low or "verdict: pass" in low:
        rec["status"] = "pass"
    elif proc.returncode != 0:
        rec["status"] = "compile_or_run_fail"
        rec["error"] = out.strip()[-400:]
    else:
        rec["status"] = "incorrect"
        rec["error"] = out.strip()[-400:]
    for key in ("snr_db", "snr"):
        idx = low.rfind(key + ":")
        if idx >= 0:
            try:
                rec["snr_db"] = float(low[idx + len(key) + 1:].split()[0])
            except Exception:  # noqa: BLE001 - a missing number is not a failure
                pass
            break
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pool_hip")
    ap.add_argument("--json", default="")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    dirs = sorted((Path(args.root) / "tasks").glob("*__hip"))
    if args.limit:
        dirs = dirs[: args.limit]
    print(f"gating {len(dirs)} seed(s) on {os.uname().nodename}")

    rows = []
    counts: dict[str, int] = {}
    for i, d in enumerate(dirs, 1):
        r = verify_one(d, args.timeout)
        rows.append(r)
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(f"  [{i}/{len(dirs)}] {r['task_id']}: {r['status']}"
              + (f" snr={r['snr_db']}" if r.get("snr_db") is not None else ""),
              flush=True)

    n = len(rows) or 1
    print(f"\nyield: {counts.get('pass', 0)}/{len(rows)} "
          f"({100 * counts.get('pass', 0) / n:.0f}%) usable as tasks")
    print(f"breakdown: {counts}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"counts": counts, "rows": rows}, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
