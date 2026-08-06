#!/usr/bin/env python
"""Re-partition a shard directory whose manifest predates the checkout.

A manifest records the commit it was built at, and a datagen worker refuses to mine
a shard whose code has moved. The guard is right, but it means every commit
invalidates every manifest, and an invalidated submission dies instantly with
NonZeroExitCode -- which in the queue looks exactly like waiting a turn.

This lived in shell first and the shell was the problem. Reading the manifest with a
here-document nested inside a command substitution mis-parsed twice: once shifting
fields so ``n_shards`` came back wrong, and once failing outright with "here-document
delimited by end-of-file". Both times it rebuilt a seven-shard layout as a single
shard, which made every array index above 0 illegal and left a node "running" for
half an hour on "array index is outside manifest shard range".

Reading JSON in the language that has a JSON parser removes that whole class of
failure. Values that cannot be right abort instead of falling back to a default: a
stale manifest only blocks new submissions for one stream, which is visible and
recoverable, while a destroyed shard layout silently wastes nodes.

    python scripts/refresh_shards.py runs/shards_hippool [--check-only]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=REPO, text=True).strip()


def refresh(shard_dir: Path, check_only: bool = False) -> int:
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"{shard_dir.name}: no manifest", file=sys.stderr)
        return 2
    try:
        m = json.loads(manifest_path.read_text())
    except Exception as exc:  # noqa: BLE001 - a torn manifest is a real condition
        print(f"{shard_dir.name}: unreadable manifest ({exc})", file=sys.stderr)
        return 2

    head = _head()
    if m.get("repo_commit") == head:
        print(f"{shard_dir.name}: current at {head[:8]}")
        return 0

    n_shards = m.get("n_shards")
    src = m.get("source_task_file") or ""
    data_root = m.get("data_root") or ""
    task_pool = m.get("task_pool") or ""

    # The number of shard files on disk is the more trustworthy record: a previous
    # bad refresh can leave n_shards understated while every shard file survives,
    # and taking the smaller number would silently retire real shards.
    on_disk = len(list(shard_dir.glob("base_*.txt")))
    if isinstance(n_shards, int) and on_disk > n_shards:
        print(f"{shard_dir.name}: manifest says {n_shards} shard(s) but {on_disk} "
              f"exist on disk; trusting disk")
        n_shards = on_disk

    problems = []
    if not isinstance(n_shards, int) or n_shards < 1:
        problems.append(f"n_shards={n_shards!r}")
    if not src or not Path(src).is_file():
        problems.append(f"source_task_file={src!r}")
    if not data_root:
        problems.append("data_root empty")
    if problems:
        print(f"{shard_dir.name}: REFUSING to re-partition ({'; '.join(problems)})",
              file=sys.stderr)
        return 3

    print(f"{shard_dir.name}: manifest at {str(m.get('repo_commit'))[:8]} != "
          f"checkout {head[:8]} -> re-partitioning {n_shards} shard(s)")
    if check_only:
        return 1

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    if task_pool:
        env["KORE_TASK_POOL"] = task_pool
    else:
        env.pop("KORE_TASK_POOL", None)
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "partition_any_tasks.py"),
         "--task-file", src, "--out-dir", str(shard_dir),
         "--data-root", data_root, "--shards", str(n_shards),
         "--target", str(int(m.get("target_wins", 3))), "--skip-check"],
        cwd=REPO, env=env, capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-2000:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        return 4

    # Verify rather than trust: the whole point of this file is that a refresh once
    # silently reduced the shard count.
    after = json.loads(manifest_path.read_text())
    if after.get("n_shards") != n_shards:
        print(f"{shard_dir.name}: FATAL re-partition wrote "
              f"n_shards={after.get('n_shards')}, expected {n_shards}",
              file=sys.stderr)
        return 5
    print(f"{shard_dir.name}: refreshed, n_shards={n_shards} at {head[:8]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dirs", nargs="+")
    ap.add_argument("--check-only", action="store_true",
                    help="report staleness without rewriting anything")
    args = ap.parse_args()
    rc = 0
    for d in args.shard_dirs:
        p = Path(d)
        if not p.is_absolute():
            p = REPO / d
        rc = max(rc, refresh(p, args.check_only))
    return 0 if args.check_only else rc


if __name__ == "__main__":
    raise SystemExit(main())
