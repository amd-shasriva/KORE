#!/usr/bin/env python
"""Build the v3 SFT mixture: the final training corpus for the product model.

Three sources, in order of how much they matter:

  1. AMD-NATIVE STEP-CENTRIC ROWS. Our own agentic trajectories, generated on
     MI355X through KoreEnv against AITER/hipBLASLt production baselines, then
     decomposed into single revisions keeping only the correctness-preserving,
     high-gain ones. This is the part nobody else has: every other kernel corpus
     is NVIDIA-generated and graded against torch-eager, which is a far easier
     bar than beating AMD's own hand-tuned libraries. Kernel-Smith's result is
     that step-centric supervision -- training a local improver rather than a
     one-shot generator -- is what took their 235B past Claude-4.6-opus.

  2. RECOVERED ROWS. multicap_full, multicap_kernel and the older agentic
     directory are re-cuts of the same corpus and ~90% duplicate v2, but a
     measured ~6.6k rows are genuinely new. Leaving free data on the floor
     because the filenames look redundant would be careless.

  3. THE v2 BASE. 61,122 rows already decontaminated and filtered.

Everything is deduplicated on message content and re-screened against the
held-out eval, because a row that entered through any path can still be
contaminated and the eval is the only thing standing behind every claim we make.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

CHARS_PER_TOKEN = 3.6


def _sig(rec: dict) -> str:
    msgs = rec.get("messages") or []
    txt = "".join(str(m.get("content") or "") for m in msgs if isinstance(m, dict))
    return hashlib.md5((txt or json.dumps(rec, sort_keys=True)).encode()).hexdigest()


def _rows(path: pathlib.Path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/b05factory/sft/multicap_v2.jsonl")
    ap.add_argument("--agentic-dir", default="data/b05factory/agentic_mt")
    ap.add_argument("--recover", nargs="*", default=[
        "data/b05factory/sft/multicap_full.jsonl",
        "data/b05factory/sft/multicap_kernel.jsonl",
    ])
    ap.add_argument("--hipkittens", default="data/b05factory/sft/hipkittens.jsonl",
                    help="HipKittens knowledge slice from "
                         "scripts/build_hipkittens_sft.py (skipped if absent)")
    ap.add_argument("--hipkittens-repeat", type=int, default=1,
                    help="times to repeat the HipKittens slice. It is a few dozen "
                         "very dense rows against a ~61k-row base, so at 1x it "
                         "cannot move behaviour; upsampling is the intended use")
    ap.add_argument("--out", default="data/b05factory/sft/multicap_v3.jsonl")
    ap.add_argument("--min-gain", type=float, default=0.05)
    ap.add_argument("--max-speedup", type=float, default=50.0)
    ap.add_argument("--max-seq-tokens", type=int, default=17408)
    ap.add_argument("--full-trajectories", dest="full_trajectories",
                    action="store_true", default=True,
                    help="also emit a full trajectory for each SUCCESSFUL episode "
                         "that yielded no step row (default: on)")
    ap.add_argument("--no-full-trajectories", dest="full_trajectories",
                    action="store_false",
                    help="step-centric rows only (the pre-measurement behaviour)")
    ap.add_argument("--all-full-trajectories", action="store_true",
                    help="emit a full trajectory for EVERY successful episode, not "
                         "just those with no step row (Dr. Kernel's setup; the "
                         "step row's messages are a prefix, so this duplicates "
                         "tokens content-hash dedup cannot see)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from kore.data.decontam import heldout_families, heldout_task_ids, record_family
    from kore.data.step_centric import decompose, decompose_with_trajectories

    ids = {t for t in heldout_task_ids() if t}
    fams = {f for f in heldout_families() if f}
    stats: collections.Counter = collections.Counter()
    seen: set[str] = set()
    out_rows: list[dict] = []

    def admit(rec: dict, bucket: str) -> bool:
        """One gate for every source, so nothing enters by a side door."""
        msgs = rec.get("messages") or []
        if not msgs:
            stats[f"{bucket}:drop_empty"] += 1
            return False
        txt = "".join(str(m.get("content") or "") for m in msgs if isinstance(m, dict))
        if len(txt) / CHARS_PER_TOKEN > args.max_seq_tokens:
            # Truncation lands mid-answer and teaches the model to start an
            # optimization it never finishes.
            stats[f"{bucket}:drop_too_long"] += 1
            return False
        if any(t in txt for t in ids):
            stats[f"{bucket}:drop_contaminated_id"] += 1
            return False
        try:
            fam = record_family(rec)
        except Exception:
            fam = ""
        if fam and fam in fams:
            stats[f"{bucket}:drop_contaminated_family"] += 1
            return False
        sig = _sig(rec)
        if sig in seen:
            stats[f"{bucket}:drop_duplicate"] += 1
            return False
        seen.add(sig)
        out_rows.append(rec)
        stats[f"{bucket}:kept"] += 1
        return True

    base = pathlib.Path(args.base)
    print(f"base: {base}")
    for rec in _rows(base):
        admit(rec, "base")

    print(f"\nAMD step-centric from {args.agentic_dir}")
    traj = []
    for p in sorted(pathlib.Path(args.agentic_dir).glob("*.jsonl")):
        if "telemetry" in p.name:
            continue          # per-attempt failures, not trajectories
        traj.extend(_rows(p))

    # Step-centric supervision alone is blind to a trajectory whose win was not a
    # REVISION. Measured on the overnight campaign: 3,475 episodes reached a
    # correct kernel and 1,942 of them produced no step row, 1,576 because they
    # were correct on their first turn -- and their median measured speedup is
    # 1.58x, so the discard is not a quality filter, it is a representation gap.
    # A never-correct trajectory is still never emitted.
    if args.full_trajectories:
        step_rows, step_stats = decompose_with_trajectories(
            traj, min_gain=args.min_gain, max_speedup=args.max_speedup,
            only_residual=not args.all_full_trajectories,
        )
    else:
        step_rows, step_stats = decompose(traj, min_gain=args.min_gain,
                                          max_speedup=args.max_speedup)
    print(f"  trajectories={step_stats['trajectories']:,} "
          f"with_steps={step_stats['with_steps']:,} steps={step_stats['steps']:,} "
          f"(fix={step_stats['fix_steps']:,} speedup={step_stats['speedup_steps']:,})")
    if args.full_trajectories:
        print(f"  reached_correct={step_stats['reached_correct']:,} "
              f"full_trajectories={step_stats['full_trajectories']:,} "
              f"skipped_has_steps={step_stats['full_skipped_has_steps']:,} "
              f"never_correct_dropped={step_stats['never_correct_dropped']:,}")
    for rec in step_rows:
        admit(rec, "amd_step" if rec.get("_source") == "kernel_step_centric"
              else "amd_full_traj")

    for rel in args.recover:
        print(f"\nrecover: {rel}")
        for rec in _rows(pathlib.Path(rel)):
            admit(rec, "recover")

    # HipKittens: CDNA4 kernel knowledge (MIT, arXiv:2511.08083). Admitted through
    # the same gate as everything else, then optionally repeated -- the repeat has
    # to happen AFTER admission because `admit` deduplicates on exact content, so
    # feeding the same row in twice would silently drop the second copy and the
    # requested weight would not apply.
    hk_path = pathlib.Path(args.hipkittens)
    if args.hipkittens and hk_path.exists():
        print(f"\nhipkittens: {hk_path}")
        first = len(out_rows)
        for rec in _rows(hk_path):
            admit(rec, "hipkittens")
        admitted = out_rows[first:]
        extra = max(0, int(args.hipkittens_repeat) - 1)
        for _ in range(extra):
            out_rows.extend(admitted)
        stats["hipkittens:repeated_copies"] = len(admitted) * extra
        if admitted:
            lic = {str((r.get("_provenance") or {}).get("license")) for r in admitted}
            print(f"  admitted {len(admitted):,} rows, repeat x{args.hipkittens_repeat} "
                  f"-> {len(admitted) * (extra + 1):,} in mixture; licence(s): "
                  f"{sorted(lic)}")
    elif args.hipkittens:
        print(f"\nhipkittens: {hk_path} absent; run "
              f"scripts/build_hipkittens_sft.py to build it (slice SKIPPED)")

    print("\n==== gate results ====")
    for k in sorted(stats):
        print(f"  {stats[k]:>8,}  {k}")
    chars = sum(len("".join(str(m.get('content') or '')
                            for m in (r.get('messages') or [])))
                for r in out_rows)
    print(f"\nv3: {len(out_rows):,} rows, ~{chars/CHARS_PER_TOKEN/1e6:.0f}M tokens")
    src = collections.Counter(r.get("_source", "?") for r in out_rows)
    for k, v in src.most_common(12):
        print(f"    {v:>7,}  {k}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as w:
        for rec in out_rows:
            w.write(json.dumps(rec) + "\n")
    print(f"\nwrote {len(out_rows):,} rows to {out} ({out.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
