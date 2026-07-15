"""Offline v1 -> v2 data upgrade (MAX REUSE, no regeneration).

Salvages the expensive, GPU-verified v1 datagen (``repair/``, ``wins/``,
``groups/``) and upgrades it to the v2 contract entirely OFFLINE, so a full v2
dataset is one CPU ``build`` stage away - you do NOT re-run the (GPU + teacher)
datagen for data you already have.

What it does (CPU only, idempotent, keeps ``.pre_normalize.bak`` backups):
  1. Contract-normalizes every RAW ``repair/*.jsonl`` + ``wins/*.jsonl`` record
     (legacy ``<think>/<answer>`` and raw-teacher ``CHANGE:`` -> canonical
     ANALYSIS/PROPOSED_CHANGE/FULL_KERNEL), so the build stage emits canonical SFT
     rows while REUSING the verified records. Derived shards (``_gold_*`` /
     ``_repair_*``) are skipped - the build stage re-mints them.
  2. Contract-normalizes the already-built ``sft/multicap.jsonl`` + ``dpo/pairs.jsonl``
     (belt-and-suspenders; the recommended flow rebuilds them).
  3. Reports data coverage (``kore.data.coverage``).

All verified scalar fields (snr_db / wall_us / speedup / preferences / failure_class)
are preserved byte-for-byte - only assistant *text* is re-rendered. ``groups/`` are
left as-is (candidate SOURCES only; the build stage wraps them canonically and
attaches provenance + in-context prompts).

CLI::

    python -m kore.data.upgrade_v1 data/full14b            # normalize raw + built shards
    python -m kore.data.upgrade_v1 data/full14b --raw-only # just the raw records
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from kore.data.normalize import normalize_file
from kore.obs import get_logger

log = get_logger("data.upgrade_v1")


def _shards(data_root: Path, sub: str) -> list[Path]:
    d = data_root / sub
    if not d.is_dir():
        return []
    # skip derived shards (rebuilt by the build stage) - only touch raw records
    return [p for p in sorted(d.glob("*.jsonl")) if not p.name.startswith("_")]


def upgrade_raw_records(data_root, backup: bool = True) -> dict:
    """Contract-normalize raw repair + wins shards in place. Returns stats."""
    data_root = Path(data_root)
    stats = {"repair_files": 0, "repair_changed": 0, "wins_files": 0, "wins_changed": 0}
    for sub, fkey, ckey in (("repair", "repair_files", "repair_changed"),
                            ("wins", "wins_files", "wins_changed")):
        for p in _shards(data_root, sub):
            r = normalize_file(p, in_place=True, backup=backup)
            stats[fkey] += 1
            stats[ckey] += int(r.get("changed", 0))
    log.event("upgrade_raw_records", **stats)
    return stats


def upgrade_built_shards(data_root, backup: bool = True) -> dict:
    """Contract-normalize the already-built multicap + pairs shards. Returns stats."""
    data_root = Path(data_root)
    out = {}
    for rel in ("sft/multicap.jsonl", "dpo/pairs.jsonl"):
        p = data_root / rel
        if p.is_file():
            r = normalize_file(p, in_place=True, backup=backup)
            out[rel] = {"rows": r.get("rows", 0), "changed": r.get("changed", 0)}
    log.event("upgrade_built_shards", **{k: v.get("changed") for k, v in out.items()})
    return out


def upgrade(data_root, raw_only: bool = False, backup: bool = True) -> dict:
    """Full offline v1 -> v2 upgrade. Returns a combined report + next-step guidance."""
    data_root = Path(data_root)
    raw = upgrade_raw_records(data_root, backup=backup)
    built = {} if raw_only else upgrade_built_shards(data_root, backup=backup)
    try:
        from kore.data.coverage import coverage_report
        cov = coverage_report(data_root)
        cov_brief = {k: cov[k] for k in ("n_train_tasks", "n_full_coverage",
                                         "coverage_pct", "n_undercovered")}
        undercovered = sorted(cov.get("undercovered", {}))
    except Exception as e:  # noqa: BLE001
        cov_brief, undercovered = {"error": str(e)}, []
    return {"raw": raw, "built": built, "coverage": cov_brief,
            "undercovered_tasks": undercovered}


_NEXT_STEPS = """\
v1 raw records upgraded to the v2 contract (verified metrics preserved). To finish v2:

  1. Rebuild (CPU, reuses ALL raw records; applies dedup + curation + provenance +
     in-context DPO prompts + gold-wins + repair-DPO):
       python scripts/run_campaign.py --only build --data-root {root} ...
  2. Contract-normalize the rebuilt shards (safety):
       python -m kore.data.normalize {root}/sft/multicap.jsonl {root}/dpo/pairs.jsonl --in-place
  3. Fill coverage holes + get rigor-verified/compile-baseline speedups + rocprof-
     grounded reasoning (GPU; parallel datagen RESUMES existing shards):
       python scripts/run_campaign.py --only datagen --data-root {root} --rigorous-verify ...
     (only the {n_under} undercovered/new tasks are (re)generated; existing shards skipped)
"""


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Offline v1->v2 data upgrade (max reuse)")
    p.add_argument("data_root", help="campaign data root (e.g. data/full14b)")
    p.add_argument("--raw-only", action="store_true", help="only normalize raw repair/wins records")
    p.add_argument("--no-backup", action="store_true", help="skip .pre_normalize.bak backups")
    a = p.parse_args(argv)
    rep = upgrade(a.data_root, raw_only=a.raw_only, backup=not a.no_backup)
    print(json.dumps(rep, indent=2))
    print(_NEXT_STEPS.format(root=a.data_root, n_under=len(rep["undercovered_tasks"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
