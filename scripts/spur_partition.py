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
import re
from typing import Iterable

# Frontier / compute-bound bias. Hard vendor-baseline kernels (aiter, hipBLAS,
# rocBLAS, CK, dequant matmul) and low-precision GEMM/attention/MoE ops are the
# scarce, high-value targets. Multiply their deepen cost so LPT packing devotes
# proportionally more of the fixed trajectory budget to them.
DEFAULT_FRONTIER_WEIGHT = 4.0
_FRONTIER_BASELINE_RE = re.compile(r"aiter|hipblas|rocblas|ck_|dequant_matmul", re.I)
_FRONTIER_DTYPES = ("fp8", "int8", "int4", "mxfp4", "w4a16")
_FRONTIER_FAMILIES = frozenset({"gemm", "attention", "moe"})


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


def _frontier_weight() -> float:
    """Multiplier applied to deepen cost for hard compute-bound tasks.

    Controlled by ``KORE_FRONTIER_WEIGHT`` (default ``DEFAULT_FRONTIER_WEIGHT``).
    A value <= 1 disables the bias. Invalid values fall back to the default.
    """
    raw = os.environ.get("KORE_FRONTIER_WEIGHT")
    if raw is None or not raw.strip():
        return DEFAULT_FRONTIER_WEIGHT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_FRONTIER_WEIGHT
    return value if value > 0 else DEFAULT_FRONTIER_WEIGHT


def _operator_family(operation: str) -> str | None:
    """Coarse operator family from a Task.operation string.

    Maps the fine-grained operation name onto the frontier families
    ``gemm`` / ``attention`` / ``moe`` used for cost biasing.
    """
    op = (operation or "").lower()
    if "moe" in op:
        return "moe"
    if "attn" in op or "attention" in op:
        return "attention"
    if "gemm" in op or "matmul" in op:
        return "gemm"
    return None


def is_frontier_task(
    comparison_baseline: str | None,
    dtype: str | None,
    operation: str | None,
) -> bool:
    """True if the task targets a hard compute-bound frontier kernel.

    Matches when the comparison baseline is a vendor library (aiter/hipBLAS/
    rocBLAS/CK/dequant matmul), OR the dtype is low precision (fp8/int8/int4/
    mxfp4/w4a16), OR the operator family is gemm/attention/moe.
    """
    baseline = comparison_baseline or ""
    if _FRONTIER_BASELINE_RE.search(baseline):
        return True
    dt = (dtype or "").lower()
    if any(token in dt for token in _FRONTIER_DTYPES):
        return True
    return _operator_family(operation) in _FRONTIER_FAMILIES


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


def work_item(
    data_root: Path,
    task_id: str,
    target: int,
    *,
    comparison_baseline: str | None = None,
    dtype: str | None = None,
    operation: str | None = None,
    frontier_weight: float | None = None,
) -> WorkItem | None:
    wins = distinct_wins(data_root / "wins" / f"{task_id}.jsonl")
    need = max(0, target - wins)
    missing_repair = not shard_present(data_root, "repair", task_id)
    missing_groups = not shard_present(data_root, "groups", task_id)
    if not (need or missing_repair or missing_groups):
        return None

    # deepen_wins bounds attempts at max(need*3, need+2): 9/6/3 for 0/1/2 wins.
    deepen_cost = max(need * 3, need + 2) if need else 0
    # Bias budget toward hard compute-bound frontier kernels so LPT packing
    # assigns them proportionally more of the fixed trajectory budget.
    frontier = is_frontier_task(comparison_baseline, dtype, operation)
    if frontier and deepen_cost:
        weight = _frontier_weight() if frontier_weight is None else frontier_weight
        deepen_cost = int(round(deepen_cost * weight))
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


def reverify_item(
    data_root: Path,
    task_id: str,
    *,
    comparison_baseline: str | None = None,
    dtype: str | None = None,
    operation: str | None = None,
    frontier_weight: float | None = None,
) -> WorkItem | None:
    """Cost-weighted work item for the REVERIFY stage.

    Selects any train task that already has a non-derived wins/groups/repair shard
    (reverify re-measures EXISTING kernels; it never generates). Cost models the
    re-evaluation budget: every ranked-groups candidate + every distinct win + every
    repair record is re-benched, so cost scales with the record volume, biased up for
    hard compute-bound frontier families (their benches dominate wall time).
    """
    wins = distinct_wins(data_root / "wins" / f"{task_id}.jsonl")
    n_groups = jsonl_record_count(data_root / "groups" / f"{task_id}.jsonl")
    n_repair = jsonl_record_count(data_root / "repair" / f"{task_id}.jsonl")
    if not (wins or n_groups or n_repair):
        return None
    # Groups candidates dominate (each is compiled + benched); wins/repair re-verify
    # once each. Base cost is the total records touched.
    base = n_groups + wins + n_repair
    frontier = is_frontier_task(comparison_baseline, dtype, operation)
    if frontier:
        weight = _frontier_weight() if frontier_weight is None else frontier_weight
        base = int(round(base * weight))
    return WorkItem(
        task_id=task_id,
        cost=max(1, base),
        needs_deepen=False,
        needs_base=False,
        wins=wins,
        missing_repair=not n_repair,
        missing_groups=not n_groups,
    )


def evolve_item(
    data_root: Path,
    task_id: str,
    target: int,
    *,
    comparison_baseline: str | None = None,
    dtype: str | None = None,
    operation: str | None = None,
    frontier_weight: float | None = None,
) -> WorkItem | None:
    """Cost-weighted work item for the EVOLVE stage.

    Selects tasks that are UNDER target distinct wins (need to be topped up) OR are
    hard compute-bound frontier families (re-pushed from 'beats eager' toward
    'beats vendor' even when already at target). Cost is the evolutionary search
    budget, biased up for frontier families.
    """
    wins = distinct_wins(data_root / "wins" / f"{task_id}.jsonl")
    n_evolve = distinct_wins(data_root / "wins" / f"{task_id}.evolve.jsonl")
    have = wins + n_evolve
    need = max(0, target - have)
    frontier = is_frontier_task(comparison_baseline, dtype, operation)
    if not (need or frontier):
        return None
    # Each evolve_task run is an island search (generations x candidates); budget a
    # few runs per under-target win plus a floor for frontier re-push.
    base = max(need * 4, 3 if frontier else 0)
    if frontier:
        weight = _frontier_weight() if frontier_weight is None else frontier_weight
        base = int(round(base * weight))
    return WorkItem(
        task_id=task_id,
        cost=max(1, base),
        needs_deepen=bool(need),
        needs_base=False,
        wins=have,
        missing_repair=False,
        missing_groups=False,
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
    ap.add_argument(
        "--mode",
        choices=("datagen", "reverify", "evolve"),
        default="datagen",
        help="datagen: deep_/base_ lists (default). reverify/evolve: single "
             "shard_NNN.txt list per node for the frontier stages.",
    )
    # Selection defaults to every registered train task (empty prefix). The
    # taxonomy split in kore/tasks/taxonomy.py is already the authority for what
    # may be trained on, so a prefix allowlist layered on top can only silently
    # drop work. An earlier eight-family default did exactly that: it reached
    # 1278 of 1289 train tasks and excluded eleven hand-authored vendor-lane
    # tasks (rmsnorm_aiter, layernorm_bf16, softmax_bf16, rope_bf16, ...) that
    # carry no matching prefix -- the highest-value tasks in the registry.
    # Narrow deliberately with --prefix or --task-file when that is the intent.
    ap.add_argument(
        "--prefix",
        default="",
        help="comma-separated task-id prefixes to include (empty string = all train tasks)",
    )
    # Optional explicit allowlist: one task id per line. When set it takes
    # precedence over --prefix, giving fully deterministic selection.
    ap.add_argument(
        "--task-file",
        default=None,
        help="file of explicit task ids (one per line); overrides --prefix",
    )
    # Attempt cap so a permanently-failing hard task cannot absorb the whole
    # budget. Recorded in the manifest for the datagen workers to honor.
    ap.add_argument(
        "--max-attempts-per-task",
        type=int,
        default=0,
        help="per-task attempt cap for downstream datagen (0 = unbounded)",
    )
    args = ap.parse_args()

    from kore.tasks.registry import train_tasks

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    tasks_by_id = {task.task_id: task for task in train_tasks()}
    if args.task_file:
        wanted = [
            line.strip()
            for line in Path(args.task_file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        missing = [tid for tid in wanted if tid not in tasks_by_id]
        if missing:
            raise SystemExit(
                f"--task-file lists unknown task ids: {sorted(set(missing))}"
            )
        selected = sorted(set(wanted))
    else:
        prefixes = tuple(p.strip() for p in args.prefix.split(","))
        # An empty prefix matches everything (str.startswith('') is True).
        selected = sorted(
            tid
            for tid in tasks_by_id
            if any(tid.startswith(pfx) for pfx in prefixes)
        )
    task_ids = selected

    frontier_weight = _frontier_weight()
    items = []
    for task_id in task_ids:
        task = tasks_by_id[task_id]
        if args.mode == "reverify":
            item = reverify_item(
                data_root, task_id,
                comparison_baseline=task.comparison_baseline,
                dtype=task.dtype, operation=task.operation,
                frontier_weight=frontier_weight,
            )
        elif args.mode == "evolve":
            item = evolve_item(
                data_root, task_id, args.target,
                comparison_baseline=task.comparison_baseline,
                dtype=task.dtype, operation=task.operation,
                frontier_weight=frontier_weight,
            )
        else:
            item = work_item(
                data_root, task_id, args.target,
                comparison_baseline=task.comparison_baseline,
                dtype=task.dtype, operation=task.operation,
                frontier_weight=frontier_weight,
            )
        if item is not None:
            items.append(item)
    shards = balanced_partition(items, args.shards)

    manifest_shards = []
    for idx, shard in enumerate(shards):
        if args.mode in ("reverify", "evolve"):
            # Single per-node task list; the stage worker does dynamic in-node load
            # balancing across its 8 GPUs, so one list is all it needs.
            ids = [item.task_id for item in shard]
            _atomic_text(out_dir / f"shard_{idx:03d}.txt", ",".join(ids))
            summary = {
                "index": idx,
                "cost": sum(item.cost for item in shard),
                "tasks": len(shard),
                "deepen": sum(item.needs_deepen for item in shard),
                "base": 0,
            }
            print(f"shard={idx:03d} cost={summary['cost']} tasks={len(shard)}")
        else:
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
            print(
                f"shard={idx:03d} cost={summary['cost']} tasks={len(shard)} "
                f"deepen={len(deep)} base={len(base)}"
            )
        manifest_shards.append(summary)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": _git_head(Path(__file__).resolve().parents[1]),
        "data_root": str(data_root),
        "mode": args.mode,
        "target_wins": args.target,
        "max_attempts_per_task": args.max_attempts_per_task,
        "frontier_weight": frontier_weight,
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
