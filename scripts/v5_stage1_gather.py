#!/usr/bin/env python
"""Stage 1 of the v5 build: gather every mined record, dedup, thin, and cache.

The corpus this reads is 236,425 rows across 13 roots, and two measured
properties of it decide everything this script does.

Cross-root duplication. 1,440 of 2,649 task ids were mined into more than one
root, because the resume contract is scoped to a single data root -- a task
finished in v5frontier_twins looks untouched to a job pointed at v5frontierhip.
That is 82,542 rows, 35% of the corpus, and they are not near-duplicates but
independent generations, so no content hash collapses them. Deduping on the
representative source is what removes them.

Repair redundancy. The 130,378 repair rows carry only 14,106 distinct
(task, broken-kernel) problems -- 9.24 answers per problem, and 52.6% of the
rows sit on the 12.7% of problems that were answered 25 or more times. The
generator's quota counts accepted fixes, not distinct problems, so once a task's
mutators are exhausted it keeps re-answering bugs already in the shard. Several
correct fixes to one bug is real signal; twenty-five is a memorisation risk at
this scale, where the measured evidence is that quality filtering beats volume.

Writes a pickle of typed records so the later stages never re-scan 3.5 GB.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
import time
from pathlib import Path

REPO = Path("/home/shasriva/Kore-RL/KORE")
sys.path.insert(0, str(REPO))

#: Shards predate the production envelope, so they carry no schema_version and
#: only this reader mode accepts them. production_strict raises on every one.
READ_MODE = "legacy_quarantine"

#: Above this the win is not a kernel achievement, it is a broken baseline. The
#: corpus holds 769 rows above it, against 137 that were flagged at write time.
CREDIBLE_SPEEDUP_MAX = 10.0


def dialect(task_id: str) -> str:
    t = task_id or ""
    if t.endswith(("__hip", "__hipf")) or t.startswith("hip_"):
        return "HIP"
    if t.endswith("__flydsl"):
        return "FlyDSL"
    return "Triton"


def as_record(rec) -> dict:
    """A mapping view of a typed record, for the admission check."""
    if isinstance(rec, dict):
        return rec
    out = {}
    for k in ("task_id", "operation", "arch", "gpu", "dtype", "provenance_root"):
        v = getattr(rec, k, None)
        if v is not None:
            out[k] = v
    out.setdefault("arch", out.pop("gpu", None) or "gfx950")
    return out


def gather(roots: list[Path]) -> list:
    from kore.data.schemas import read_jsonl

    out: list = []
    per_root: dict[str, int] = {}
    for root in roots:
        n0 = len(out)
        for sub in ("repair", "wins", "groups"):
            d = root / sub
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.jsonl")):
                try:
                    out += read_jsonl(p, typed=True, mode=READ_MODE)
                except Exception as exc:  # noqa: BLE001 - a torn shard is not the corpus
                    print(f"  WARN {p}: {type(exc).__name__}: {exc}", file=sys.stderr)
        per_root[root.name] = len(out) - n0
        print(f"  {root.name:<22} {len(out) - n0:>8,}", flush=True)
    return out, per_root


def thin_repairs(repairs: list, keep: int, keep_scarce: int = 4) -> tuple[list, dict]:
    """Keep at most ``keep`` distinct fixes per broken kernel (``keep_scarce`` for FlyDSL).

    The corpus holds 9.24 answers per distinct problem, which is squarely in the
    regime where repetition costs real capability rather than merely wasting
    tokens: repeating a small fraction of a corpus a hundred times has been
    measured to degrade an 800M model to the quality of a 400M one, and the
    damage lands specifically on the copying and induction circuitry that
    generalisation runs on. More capable models treat semantically equivalent
    targets more like exact duplicates, not less, so a 30B is in the worse half
    of that finding.

    Two is the floor rather than one because what matters is the number of
    *distinct* solution paths, not samples: rejection-sampling work finds three
    paths already beats plain fine-tuning stably while the benefit of doubling
    the sample count falls away quickly, since extra samples yield no new
    problems. FlyDSL gets four because it is the one dialect where the corpus is
    thin enough that coverage, not redundancy, is the binding risk.

    Distinct fixes are preferred over repeats of one, and within that the
    higher-accuracy fix wins, so thinning removes repetition rather than lessons.
    Ordering is fully determined by the data so two runs agree.
    """
    from kore.data.prompts import extract_kernel

    groups: dict[tuple, list] = collections.defaultdict(list)
    for r in repairs:
        groups[(r.task_id, r.parent_hash)].append(r)

    before = collections.Counter(len(v) for v in groups.values())
    kept: list = []
    for _key, rs in groups.items():
        seen_fix: dict[str, object] = {}
        for r in rs:
            msgs = getattr(r, "messages", None) or []
            body = msgs[-1].get("content", "") if msgs else ""
            fix = extract_kernel(body) or body
            # First distinct fix wins its slot; a later repeat only replaces it
            # if it is measurably more accurate.
            prev = seen_fix.get(fix)
            if prev is None or (r.child_snr_db or -1e9) > (prev.child_snr_db or -1e9):
                seen_fix[fix] = r
        ranked = sorted(
            seen_fix.values(),
            key=lambda r: (-(r.child_snr_db or -1e9), str(r.failure_class or "")),
        )
        k = keep_scarce if dialect(_key[0]) == "FlyDSL" else keep
        kept.extend(ranked[:k])

    after = collections.Counter()
    kg: dict[tuple, int] = collections.Counter()
    for r in kept:
        kg[(r.task_id, r.parent_hash)] += 1
    after = collections.Counter(kg.values())
    return kept, {
        "problems": len(groups),
        "rows_before": len(repairs),
        "rows_after": len(kept),
        "hist_before": dict(sorted(before.items())),
        "hist_after": dict(sorted(after.items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "runs/v5_build/stage1.pkl"))
    ap.add_argument("--keep-per-problem", type=int, default=4,
                    help="max distinct fixes per broken kernel (common dialects)")
    ap.add_argument("--keep-per-problem-scarce", type=int, default=6,
                    help="same, for FlyDSL where coverage outranks redundancy")
    ap.add_argument("--cache-policy", choices=("strict", "audited"), default="audited",
                    help="cache the superset; stage 4 narrows it authoritatively")
    args = ap.parse_args()

    t0 = time.time()
    roots = sorted(d for d in (REPO / "data").iterdir()
                   if d.is_dir() and any((d / s).is_dir() for s in ("repair", "wins", "groups")))
    print(f"=== gathering {len(roots)} roots ===", flush=True)
    raw, per_root = gather(roots)
    print(f"  gathered {len(raw):,} records in {time.time() - t0:.0f}s\n", flush=True)

    from kore.data.build_datasets import dedup_by_source_hash
    from kore.data.schemas import RankedGroupRecord, RepairRecord, WinRecord

    n_before = len(raw)
    raw = dedup_by_source_hash(raw)
    print(f"=== cross-root dedup: {n_before:,} -> {len(raw):,} "
          f"({n_before - len(raw):,} duplicate copies removed) ===\n", flush=True)

    # Cache the SUPERSET, not a policy's view of it. Filtering here bakes one
    # policy into the artifact: the first build cached the strict view and
    # discarded 24,547 records that the audited policy would have admitted, so
    # asking the audited question later could only ever re-derive the strict
    # answer. Since audited is a superset of strict, caching the wider set lets a
    # single six-minute pass serve both, and stage 4's gate -- which is
    # authoritative anyway, and is the only stage that sees the benchmark index --
    # decides what actually reaches training.
    from kore.data.v5_policy import admits
    kept, held = [], collections.Counter()
    for r in raw:
        ok, why = admits(as_record(r), args.cache_policy)
        if ok:
            kept.append(r)
        else:
            held[why.split(":")[0]] += 1
    print(f"=== cache filter ({args.cache_policy}): removed {sum(held.values()):,}, "
          f"kept {len(kept):,} ===")
    for k, v in held.most_common(6):
        print(f"    {k:<30} {v:,}")
    print(flush=True)

    repairs = [r for r in kept if isinstance(r, RepairRecord)]
    wins = [r for r in kept if isinstance(r, WinRecord)]
    groups = [r for r in kept if isinstance(r, RankedGroupRecord)]

    # Drop wins whose speedup is not credible. A 9,380x "win" is a statement
    # about the baseline, and the flag written at generation time caught only
    # 137 of the 769 rows above the ceiling.
    n_w = len(wins)
    wins = [w for w in wins
            if (w.speedup is None) or (w.speedup <= CREDIBLE_SPEEDUP_MAX)]
    print(f"=== wins: {n_w:,} -> {len(wins):,} "
          f"({n_w - len(wins):,} above the {CREDIBLE_SPEEDUP_MAX}x credible ceiling) ===")

    thinned, stats = thin_repairs(repairs, args.keep_per_problem,
                                  args.keep_per_problem_scarce)
    print(f"=== repair thinning (keep {args.keep_per_problem}/problem, "
          f"{args.keep_per_problem_scarce} for FlyDSL) ===")
    print(f"  distinct problems : {stats['problems']:,}")
    print(f"  rows {stats['rows_before']:,} -> {stats['rows_after']:,} "
          f"({stats['rows_before'] / max(stats['problems'],1):.2f}x -> "
          f"{stats['rows_after'] / max(stats['problems'],1):.2f}x per problem)")

    for label, recs in (("repair", thinned), ("win", wins), ("group", groups)):
        d = collections.Counter(dialect(getattr(r, "task_id", "")) for r in recs)
        print(f"  {label:<7} {len(recs):>8,}  HIP={d['HIP']:,} Triton={d['Triton']:,} "
              f"FlyDSL={d['FlyDSL']:,}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump({"repairs": thinned, "wins": wins, "groups": groups}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "gathered": n_before, "after_dedup": len(raw),
        "cache_policy": args.cache_policy, "cache_removed": dict(held),
        "per_root": per_root, "repair": stats,
        "wins_kept": len(wins), "wins_dropped_implausible": n_w - len(wins),
        "groups": len(groups), "seconds": round(time.time() - t0, 1),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
