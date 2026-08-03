#!/usr/bin/env python
"""Turn raw agentic shards into a filtered, decontaminated SFT slice.

Three stages, in this order, because each is cheaper than the next and because
the contamination verdict has to be computed on exactly the rows that survive:

  quality        Kernel-Smith retention: keep trajectories that preserved
                 correctness and materially improved, not every trajectory that
                 ran. See kore/data/agentic_filter.py for what "improved" means
                 and why gain is measured inside the episode rather than against
                 the vendor kernel.
  reward hacking an absolute speedup cap plus a vendor-grade timing requirement.
                 The cap is derived from the observed distribution rather than
                 guessed; run with --report-only first and read the basis it
                 prints before pinning one.
  decontamination the repo's own decontam rules against the held-out eval, by
                 task id, by product family, and by the full HoldoutIndex
                 (exact / normalized AST / semantic graph / MinHash / directional
                 containment). Generation only draws from train_tasks(), so every
                 one of these should report zero -- which is the point. An
                 unchecked assumption here produces better eval numbers, not an
                 error.

--report-only computes and prints everything and writes nothing, so the cap can
be chosen from evidence before any slice is published.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _rows(paths):
    for path in paths:
        with path.open(errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and not row.get("_dropped"):
                    yield row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--glob", default="shard_*.jsonl")
    parser.add_argument("--out", default="")
    parser.add_argument("--stats-json", default="")
    parser.add_argument("--min-gain", type=float, default=1.15)
    parser.add_argument("--vendor-parity", type=float, default=1.0)
    parser.add_argument("--max-speedup", type=float, default=0.0,
                        help="0 = derive the cap from the observed distribution "
                             "and print the basis")
    parser.add_argument("--min-high-gain-revisions", type=int, default=1)
    parser.add_argument("--max-seq-tokens", type=int, default=17408)
    parser.add_argument("--allow-non-vendor-grade", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    from kore.data.agentic_filter import (
        FilterPolicy, analyze, classify, speedup_distribution, suggest_max_speedup,
    )

    in_dir = pathlib.Path(args.in_dir)
    paths = sorted(
        p for p in in_dir.glob(args.glob) if not p.name.endswith(".telemetry.jsonl")
    )
    if not paths:
        print(f"no shards matched {in_dir}/{args.glob}")
        return 2
    print(f"shards: {len(paths)}  ({sum(p.stat().st_size for p in paths)/1e9:.2f} GB)")

    # Pass 1: measure. The cap cannot be justified without the distribution, and
    # the distribution has to come from the data being filtered.
    observed: list[float] = []
    categories: collections.Counter = collections.Counter()
    gains: list[float] = []
    revisions: list[int] = []
    tokens: list[float] = []
    n_rows = 0
    for row in _rows(paths):
        n_rows += 1
        stats = analyze(row)
        categories[stats.category] += 1
        tokens.append(stats.est_tokens)
        if stats.best_speedup is not None:
            observed.append(stats.best_speedup)
        if stats.gain is not None:
            gains.append(stats.gain)
        revisions.append(stats.n_high_gain_revisions)

    distribution = speedup_distribution(observed)
    suggestion = suggest_max_speedup(observed)
    print(f"\nrows                : {n_rows:,}")
    print(f"category mix        : {dict(categories.most_common())}")
    print(f"measured speedups   : {json.dumps(distribution)}")
    print(f"in-episode gain     : {json.dumps(speedup_distribution(gains))}")
    print(f"high-gain revisions : mean="
          f"{(sum(revisions)/len(revisions) if revisions else 0):.2f} "
          f"zero={sum(1 for r in revisions if r == 0):,}")
    print(f"reward-hack cap     : {json.dumps(suggestion)}")

    max_speedup = args.max_speedup if args.max_speedup > 0 else suggestion["cap"]
    policy = FilterPolicy(
        min_gain=args.min_gain,
        vendor_parity=args.vendor_parity,
        max_speedup=max_speedup,
        min_high_gain_revisions=args.min_high_gain_revisions,
        max_seq_tokens=args.max_seq_tokens,
        require_vendor_grade=not args.allow_non_vendor_grade,
    )
    print(f"\napplying cap        : {max_speedup} "
          f"({'explicit' if args.max_speedup > 0 else suggestion['basis']})")

    # Pass 2: filter.
    kept_rows: list[dict] = []
    drops: collections.Counter = collections.Counter()
    kept_speedups: list[float] = []
    for row in _rows(paths):
        reason, stats = classify(row, policy)
        if reason is not None:
            drops[reason] += 1
            continue
        kept_rows.append({
            "messages": row.get("messages") or [],
            "task_id": stats.task_id,
            "_source": "kore_agentic_mt",
            "_speedup": round(stats.best_speedup, 4),
            "_gain": round(stats.gain, 4) if stats.gain else None,
            "_high_gain_revisions": stats.n_high_gain_revisions,
            "_turns": stats.turns,
            "_category": stats.category,
        })
        kept_speedups.append(stats.best_speedup)

    print("\nquality filter:")
    for reason, count in drops.most_common():
        print(f"  {count:>8,}  drop_{reason}")
    print(f"  {len(kept_rows):>8,}  KEPT")
    print(f"  kept speedups: {json.dumps(speedup_distribution(kept_speedups))}")

    # Pass 3: decontaminate what survived.
    from kore.data.decontam import (
        build_heldout_ngrams, decontaminate_chat_rows, heldout_families,
        heldout_task_ids, record_family,
    )

    heldout_ids = {t for t in heldout_task_ids() if t}
    heldout_fams = {f for f in heldout_families() if f}
    print(f"\nheld-out task ids   : {len(heldout_ids)}")
    print(f"held-out families   : {len(heldout_fams)}")

    id_hits: collections.Counter = collections.Counter()
    family_hits: collections.Counter = collections.Counter()
    surviving: list[dict] = []
    for row in kept_rows:
        text = "".join(
            str(m.get("content") or "") for m in row["messages"] if isinstance(m, dict))
        hit = None
        if row["task_id"] in heldout_ids:
            hit = row["task_id"]
        else:
            for task_id in heldout_ids:
                if task_id in text:
                    hit = task_id
                    break
        if hit:
            id_hits[hit] += 1
            continue
        try:
            family = record_family(row)
        except Exception:  # noqa: BLE001 - taxonomy gaps must not skip the id gate
            family = ""
        if family and family in heldout_fams:
            family_hits[family] += 1
            continue
        surviving.append(row)

    index = build_heldout_ngrams()
    clean, decontam_stats = decontaminate_chat_rows(surviving, heldout_ngrams=index)
    print(f"[ids]      dropped: {sum(id_hits.values()):,} {dict(id_hits.most_common(5))}")
    print(f"[families] dropped: {sum(family_hits.values()):,} "
          f"{dict(family_hits.most_common(5))}")
    print(f"[index]    dropped: {decontam_stats['n_dropped_contaminated']:,} "
          f"reasons={decontam_stats['drop_reasons']}")
    print(f"           references indexed: {decontam_stats['heldout_references']}")
    print(f"\nFINAL: {len(clean):,} trajectories")

    report = {
        "in_dir": str(in_dir),
        "shards": len(paths),
        "rows_read": n_rows,
        "category_mix": dict(categories.most_common()),
        "speedup_distribution": distribution,
        "gain_distribution": speedup_distribution(gains),
        "reward_hack_cap": max_speedup,
        "reward_hack_cap_basis": (
            "explicit" if args.max_speedup > 0 else suggestion["basis"]),
        "reward_hack_evidence": suggestion,
        "quality_drops": dict(drops),
        "kept_after_quality": len(kept_rows),
        "kept_speedup_distribution": speedup_distribution(kept_speedups),
        "decontam_id_hits": dict(id_hits),
        "decontam_family_hits": dict(family_hits),
        "decontam_index": {
            k: v for k, v in decontam_stats.items() if k != "evidence"},
        "final_rows": len(clean),
        "clean": (
            not id_hits and not family_hits
            and decontam_stats["n_dropped_contaminated"] == 0
        ),
    }
    if args.stats_json:
        pathlib.Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.stats_json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.stats_json}")

    if args.report_only:
        print("\nreport only, nothing written")
        return 0
    if not args.out:
        print("\nno --out given, nothing written")
        return 0

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in clean:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {len(clean):,} rows to {out} ({out.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
