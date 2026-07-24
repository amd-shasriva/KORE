"""Verify dataset completeness and emit the remaining-undone list for cleanup.

Prints a one-line summary (fully-complete count, wins histogram, missing repair/
groups) and writes a private-runtime cleanup list with every task still short of
repair+groups+wins>=target. ``--cleanup-out`` selects an explicit owned state file.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path


def _records(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    records = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"invalid JSONL record {path}:{line_no}: expected object"
                )
            records.append(record)
    return records


def _distinct_wins(root: Path, task_id: str) -> int:
    sources = {
        str(record.get("final_source", "") or "").strip()
        for record in _records(root / "wins" / f"{task_id}.jsonl")
    }
    sources.discard("")
    return len(sources)


def _has_shard(root: Path, kind: str, task_id: str) -> bool:
    path = root / kind / f"{task_id}.jsonl"
    return not path.with_suffix(path.suffix + ".inprogress").exists() and bool(
        _records(path)
    )


def verify(root: Path, task_ids: list[str], target: int) -> tuple[dict, list[str]]:
    wins_hist = collections.Counter()
    missing_repair = missing_groups = fully_complete = 0
    undone = []
    for task_id in task_ids:
        wins = _distinct_wins(root, task_id)
        wins_hist[min(wins, target)] += 1
        repair = _has_shard(root, "repair", task_id)
        groups = _has_shard(root, "groups", task_id)
        if not repair:
            missing_repair += 1
        if not groups:
            missing_groups += 1
        if wins >= target and repair and groups:
            fully_complete += 1
        else:
            undone.append(task_id)
    summary = {
        "tasks": len(task_ids),
        "fully_complete": fully_complete,
        "wins_hist": dict(sorted(wins_hist.items())),
        "missing_repair": missing_repair,
        "missing_groups": missing_groups,
        "remaining_undone": len(undone),
    }
    return summary, undone


def _write_cleanup(path: Path, undone: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(",".join(undone))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("target", type=int)
    ap.add_argument("--tasks", default="", help="optional comma-separated task IDs")
    ap.add_argument("--prefix", default="genb_")
    ap.add_argument("--cleanup-out", default="")
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.target < 1:
        ap.error("target must be positive")

    from kore.tasks.registry import train_tasks

    if args.tasks.strip():
        tasks = list(dict.fromkeys(t for t in args.tasks.split(",") if t))
    else:
        tasks = [
            task.task_id
            for task in train_tasks()
            if task.task_id.startswith(args.prefix)
        ]
    summary, undone = verify(Path(args.root), tasks, args.target)
    cleanup_out = args.cleanup_out
    if not cleanup_out:
        from kore.ops.runtime import SecureRuntime

        cleanup_out = str(SecureRuntime().state_path("kf-verify/cleanup.txt"))
    _write_cleanup(Path(cleanup_out), undone)
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(
            f"VERIFY tasks={summary['tasks']} "
            f"fully_complete={summary['fully_complete']} "
            f"wins_hist={summary['wins_hist']} "
            f"missing_repair={summary['missing_repair']} "
            f"missing_groups={summary['missing_groups']} "
            f"remaining_undone={summary['remaining_undone']}"
        )
    return 1 if args.require_complete and undone else 0


if __name__ == "__main__":
    raise SystemExit(main())
