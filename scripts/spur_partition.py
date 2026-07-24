"""Build disjoint, cost-balanced task shards for SPUR burst datagen.

Each Slurm array element receives two comma-separated task lists:

* ``deep_NNN.txt``: tasks whose distinct verified-win count is below ``--target``.
* ``base_NNN.txt``: tasks missing a non-empty repair or ranked-groups shard.

The assignment is deterministic longest-processing-time (LPT) bin packing. Costs
model the bounded trajectory budget in ``deepen_wins.py`` plus the much larger
repair/groups generation budgets, balancing work rather than raw task counts.
All outputs are immutable run-specific files, so array elements never overlap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class WorkItem:
    task_id: str
    cost: int
    needs_deepen: bool
    needs_base: bool
    wins: int
    missing_repair: bool
    missing_groups: bool


def _canonical_hash(record: dict) -> str:
    source = str(record.get("final_source", "") or "").strip()
    payload = source or json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8", "ignore")).hexdigest()


def jsonl_record_count(path: Path) -> int:
    """Count object records, failing loudly on malformed/unsupported JSONL."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    count = 0
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
            count += 1
    return count


def distinct_wins(path: Path) -> int:
    """Count distinct win kernels, failing loudly on malformed JSONL."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    seen: set[str] = set()
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
            if not str(record.get("final_source", "") or "").strip():
                continue
            seen.add(_canonical_hash(record))
    return len(seen)


def shard_present(data_root: Path, kind: str, task_id: str) -> bool:
    path = data_root / kind / f"{task_id}.jsonl"
    marker = path.with_suffix(path.suffix + ".inprogress")
    return not marker.exists() and jsonl_record_count(path) > 0


def work_item(data_root: Path, task_id: str, target: int) -> WorkItem | None:
    wins = distinct_wins(data_root / "wins" / f"{task_id}.jsonl")
    need = max(0, target - wins)
    missing_repair = not shard_present(data_root, "repair", task_id)
    missing_groups = not shard_present(data_root, "groups", task_id)
    if not (need or missing_repair or missing_groups):
        return None

    # deepen_wins bounds attempts at max(need*3, need+2): 9/6/3 for 0/1/2 wins.
    deepen_cost = max(need * 3, need + 2) if need else 0
    # Repair can make up to 175 teacher/eval attempts; groups evaluates 120
    # candidates. Relative weights spread these expensive gaps across nodes.
    cost = deepen_cost + (9 if missing_repair else 0) + (6 if missing_groups else 0)
    return WorkItem(
        task_id=task_id,
        cost=cost,
        needs_deepen=bool(need),
        needs_base=missing_repair or missing_groups,
        wins=wins,
        missing_repair=missing_repair,
        missing_groups=missing_groups,
    )


def balanced_partition(items: Iterable[WorkItem], n_shards: int) -> list[list[WorkItem]]:
    """Deterministic LPT partition with disjoint, complete assignment."""
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    shards: list[list[WorkItem]] = [[] for _ in range(n_shards)]
    costs = [0] * n_shards
    ordered = sorted(items, key=lambda item: (-item.cost, item.task_id))
    for item in ordered:
        idx = min(range(n_shards), key=lambda i: (costs[i], len(shards[i]), i))
        shards[idx].append(item)
        costs[idx] += item.cost
    return shards


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--target", type=int, default=3)
    ap.add_argument("--prefix", default="genb_")
    args = ap.parse_args()

    from kore.tasks.registry import train_tasks

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    task_ids = sorted(
        task.task_id for task in train_tasks() if task.task_id.startswith(args.prefix)
    )
    items = [
        item
        for task_id in task_ids
        if (item := work_item(data_root, task_id, args.target)) is not None
    ]
    shards = balanced_partition(items, args.shards)

    manifest_shards = []
    for idx, shard in enumerate(shards):
        deep = [item.task_id for item in shard if item.needs_deepen]
        base = [item.task_id for item in shard if item.needs_base]
        _atomic_text(out_dir / f"deep_{idx:03d}.txt", ",".join(deep))
        _atomic_text(out_dir / f"base_{idx:03d}.txt", ",".join(base))
        summary = {
            "index": idx,
            "cost": sum(item.cost for item in shard),
            "tasks": len(shard),
            "deepen": len(deep),
            "base": len(base),
        }
        manifest_shards.append(summary)
        print(
            f"shard={idx:03d} cost={summary['cost']} tasks={len(shard)} "
            f"deepen={len(deep)} base={len(base)}"
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": _git_head(Path(__file__).resolve().parents[1]),
        "data_root": str(data_root),
        "target_wins": args.target,
        "n_train_tasks": len(task_ids),
        "n_work_items": len(items),
        "n_shards": args.shards,
        "totals": {
            "cost": sum(item.cost for item in items),
            "deepen": sum(item.needs_deepen for item in items),
            "base": sum(item.needs_base for item in items),
            "missing_repair": sum(item.missing_repair for item in items),
            "missing_groups": sum(item.missing_groups for item in items),
        },
        "shards": manifest_shards,
        "items": [asdict(item) for item in items],
    }
    _atomic_text(out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(
        "PARTITION "
        f"work={len(items)} shards={args.shards} "
        f"deepen={manifest['totals']['deepen']} base={manifest['totals']['base']} "
        f"cost_range={min((s['cost'] for s in manifest_shards), default=0)}.."
        f"{max((s['cost'] for s in manifest_shards), default=0)} "
        f"out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
