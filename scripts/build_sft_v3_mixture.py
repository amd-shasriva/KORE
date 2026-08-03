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
    ap.add_argument("--out", default="data/b05factory/sft/multicap_v3.jsonl")
    ap.add_argument("--min-gain", type=float, default=0.05)
    ap.add_argument("--max-speedup", type=float, default=50.0)
    ap.add_argument("--max-seq-tokens", type=int, default=17408)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from kore.data.decontam import heldout_families, heldout_task_ids, record_family
    from kore.data.step_centric import decompose

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
    step_rows, step_stats = decompose(traj, min_gain=args.min_gain,
                                      max_speedup=args.max_speedup)
    print(f"  trajectories={step_stats['trajectories']:,} "
          f"with_steps={step_stats['with_steps']:,} steps={step_stats['steps']:,} "
          f"(fix={step_stats['fix_steps']:,} speedup={step_stats['speedup_steps']:,})")
    for rec in step_rows:
        admit(rec, "amd_step")

    for rel in args.recover:
        print(f"\nrecover: {rel}")
        for rec in _rows(pathlib.Path(rel)):
            admit(rec, "recover")

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
