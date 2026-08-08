#!/usr/bin/env python
"""Re-copy each twin's driver.py from the task it was twinned from.

A twin is materialized by copying reference.py and driver.py out of its source
task and writing a new task.yaml and seed. The copy is taken once, at seed
time, which means a later fix to a driver never reaches the twins already on
disk -- they keep running the version that was current when the teacher wrote
their seed.

That is not hypothetical. 24 hand-authored tasks carry their own candidate
loader inside driver.py, and every one of them looked for ``kernel.py`` beside
itself. A HIP twin stages ``kernel.hip`` and no ``kernel.py``, so those twins
died in the loader before anything was compiled and the gate recorded
compile_or_run_fail, which reads as a bad seed rather than a stale driver.
Fixing the source tasks left 34 already-materialized twins still broken.

Only driver.py is refreshed. reference.py is the oracle and for a
functionalized twin it is *generated* rather than copied -- overwriting it with
the source's would change what the twin is graded against, which is the one
thing that must not drift.

    python scripts/refresh_twin_drivers.py --roots data/registry_hip_frontier
    python scripts/refresh_twin_drivers.py --check     # report, write nothing
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: Where a twin's source task may live. The registry first: a pool id and a
#: registry id never collide, and the registry is the smaller, curated set.
SOURCE_ROOTS = ("kore/tasks", "data/task_pool/tasks")

TWIN_SUFFIX = re.compile(r"(__hipf|__hip|__flydsl)$")


def source_of(twin_dir: Path, roots=SOURCE_ROOTS) -> Path | None:
    base = TWIN_SUFFIX.sub("", twin_dir.name)
    if base == twin_dir.name:
        return None
    for root in roots:
        candidate = REPO / root / base
        if (candidate / "driver.py").is_file():
            return candidate
    return None


def refresh(roots, check: bool = False) -> tuple[int, int]:
    stale = fixed = 0
    for root in roots:
        tasks = (REPO / root) / "tasks"
        if not tasks.is_dir():
            continue
        for twin in sorted(tasks.iterdir()):
            dst = twin / "driver.py"
            if not dst.is_file():
                continue
            src = source_of(twin)
            if src is None:
                continue
            a, b = (src / "driver.py").read_text(errors="ignore"), dst.read_text(errors="ignore")
            if a == b:
                continue
            stale += 1
            print(f"  stale: {twin.relative_to(REPO)}")
            if not check:
                shutil.copy(src / "driver.py", dst)
                fixed += 1
    return stale, fixed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[
        "data/registry_hip_frontier", "data/registry_flydsl_frontier",
        "data/pool_hip_frontier", "data/pool_flydsl", "data/pool_hip",
        "data/pool_hip_f", "data/pool_hip_ok", "data/frontier_twins_ok",
    ])
    ap.add_argument("--check", action="store_true",
                    help="report drift without writing")
    args = ap.parse_args()

    stale, fixed = refresh(args.roots, args.check)
    print(f"twin drivers: {stale} stale"
          + (f", {fixed} refreshed" if not args.check else " (check only)"))
    # A stale driver under --check is a real finding for CI, but the loop calls
    # this without --check and must not treat "there was drift" as a failure.
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())
