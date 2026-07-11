"""Task data-coverage audit (Pillar 2): make "100% coverage" measurable + fixable.

"Cover 100% of everything" has two senses; this module owns the first and reports
on the second:

  1. DATA coverage — every TRAIN task must have non-empty ``repair`` + ``groups`` +
     ``wins`` shards. A task with a missing/empty shard is a hole: the policy never
     sees repair transitions, preferences, or a win demo for that operator. The
     shipped data had ~5-28 tasks short of full coverage.
  2. SPACE coverage — the op x dtype frontier the task generator emits (see
     ``kore.tasks.generate_ops.FAMILY_DTYPES``). Reported here per (family, dtype)
     so gaps (e.g. no generated fp8/int8 elementwise) are visible.

:func:`coverage_report` returns a structured report; :func:`undercovered_tasks`
lists exactly which tasks need (re)generation, so datagen can TARGET the holes
instead of blindly re-running everything. Pure (registry + filesystem), no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

# The three per-task datagen products that together constitute full coverage.
REQUIRED_KINDS: tuple[str, ...] = ("repair", "groups", "wins")


def _shard_count(data_root: Path, kind: str, task_id: str) -> int:
    """Number of JSONL lines in ``<data_root>/<kind>/<task_id>.jsonl`` (0 if absent)."""
    p = data_root / kind / f"{task_id}.jsonl"
    if not p.is_file():
        return 0
    try:
        return sum(1 for ln in p.read_text().splitlines() if ln.strip())
    except OSError:
        return 0


def task_coverage(data_root, task_ids: Iterable[str],
                  kinds: tuple[str, ...] = REQUIRED_KINDS) -> dict[str, dict]:
    """Per-task shard counts + a ``full`` flag. ``{task_id: {kind: n, ..., full: bool}}``."""
    data_root = Path(data_root)
    out: dict[str, dict] = {}
    for tid in task_ids:
        counts = {k: _shard_count(data_root, k, tid) for k in kinds}
        counts["full"] = all(counts[k] > 0 for k in kinds)
        out[tid] = counts
    return out


def _train_task_ids() -> list[str]:
    """Train (non-held-out) task ids from the registry; [] if unavailable."""
    try:
        from kore.tasks.registry import train_tasks
        return sorted(t.task_id for t in train_tasks())
    except Exception:  # noqa: BLE001
        return []


def undercovered_tasks(data_root, task_ids: Optional[Iterable[str]] = None,
                       kinds: tuple[str, ...] = REQUIRED_KINDS) -> dict[str, list[str]]:
    """``{task_id: [missing_kinds]}`` for every train task missing a required kind."""
    ids = list(task_ids) if task_ids is not None else _train_task_ids()
    cov = task_coverage(data_root, ids, kinds)
    return {tid: [k for k in kinds if c[k] == 0]
            for tid, c in cov.items() if not c["full"]}


def space_coverage() -> dict:
    """The generated op x dtype frontier (family -> dtypes) + per-dtype gaps."""
    try:
        from kore.tasks._genops import DTYPES
        from kore.tasks.generate_ops import FAMILY_DTYPES
    except Exception:  # noqa: BLE001
        return {}
    all_dtypes = set(DTYPES)
    per_family = {fam: {"emitted": list(dts),
                        "missing": sorted(all_dtypes - set(dts))}
                  for fam, dts in FAMILY_DTYPES.items()}
    return {"all_dtypes": sorted(all_dtypes), "per_family": per_family}


def coverage_report(data_root, task_ids: Optional[Iterable[str]] = None) -> dict:
    """Full data + space coverage report."""
    ids = list(task_ids) if task_ids is not None else _train_task_ids()
    cov = task_coverage(data_root, ids)
    n = len(ids)
    n_full = sum(1 for c in cov.values() if c["full"])
    per_kind = {k: sum(1 for c in cov.values() if c[k] > 0) for k in REQUIRED_KINDS}
    under = undercovered_tasks(data_root, ids)
    return {
        "n_train_tasks": n,
        "n_full_coverage": n_full,
        "coverage_pct": round(100.0 * n_full / n, 2) if n else 0.0,
        "per_kind_covered": per_kind,
        "per_kind_pct": {k: round(100.0 * v / n, 2) if n else 0.0
                         for k, v in per_kind.items()},
        "n_undercovered": len(under),
        "undercovered": under,
        "space": space_coverage(),
    }


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="KORE task data-coverage audit")
    p.add_argument("data_root", help="campaign data root (e.g. data/full14b)")
    p.add_argument("--json", action="store_true", help="emit the full JSON report")
    p.add_argument("--undercovered", action="store_true", help="list only the holes")
    a = p.parse_args(argv)
    rep = coverage_report(a.data_root)
    if a.json:
        print(json.dumps(rep, indent=2))
    elif a.undercovered:
        for tid, missing in sorted(rep["undercovered"].items()):
            print(f"{tid}: missing {', '.join(missing)}")
    else:
        print(f"train tasks: {rep['n_train_tasks']}  full coverage: "
              f"{rep['n_full_coverage']} ({rep['coverage_pct']}%)  "
              f"undercovered: {rep['n_undercovered']}")
        for k, pct in rep["per_kind_pct"].items():
            print(f"  {k}: {rep['per_kind_covered'][k]}/{rep['n_train_tasks']} ({pct}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
