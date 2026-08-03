#!/usr/bin/env python
"""Add multi-turn kernel-refinement trajectories to the SFT mixture.

Our mixture teaches one-shot generation: 52,646 of 56,493 rows are a single
user/assistant exchange, so only 6.4% show the model what to do with execution
feedback. Both frontier kernel systems of 2026 disagree with that shape.
Dr. Kernel (arXiv 2602.05885, ICML 2026) warm-starts on 5-turn trajectories
before multi-turn RL, and Kernel-Smith trains the model "as a strong local
improver inside the evolutionary loop rather than as a one-shot generator".
Dr. Kernel-14B is built on Qwen3-14B -- our base -- and beats Claude-4.5-Sonnet
and GPT-5 on the KernelBench Level-2 speedup rate.

So this mixes in hkust-nlp/drkernel-coldstart-8k (MIT): 8,920 trajectories where
each turn appends real KernelGYM feedback (compile / correctness / speedup /
profiling) and asks for a revision. The skill it teaches -- read a profiler
result, diagnose, revise -- is what the RL stage will then sharpen, and it is
hardware-agnostic even though the trajectories were collected on NVIDIA.

Three filters, in order:

  contamination  against the held-out eval, via the repo's own decontam rules.
                 These trajectories derive from cudaLLM-data and were validated
                 on KernelBench Level 2, so overlap is plausible rather than
                 hypothetical and has to be checked, not assumed.
  quality        keep trajectories that actually ended faster. A trajectory that
                 never achieves a speedup demonstrates the failure we are trying
                 to train out, so final_speedup is a label, not just metadata.
  budget         cap total added tokens, because SFT time scales with them and
                 the whole mixture has to fit the allocation.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

CHARS_PER_TOKEN = 3.6  # measured on this mixture; used only for budgeting


def _text(msgs) -> str:
    return "".join(str(m.get("content") or "") for m in msgs if isinstance(m, dict))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-mixture", default="data/b05factory/sft/multicap.jsonl")
    ap.add_argument("--out", default="data/b05factory/sft/multicap_v2.jsonl")
    ap.add_argument("--repo-id", default="hkust-nlp/drkernel-coldstart-8k")
    ap.add_argument("--min-speedup", type=float, default=1.0,
                    help="keep trajectories whose final kernel was at least this much faster")
    # A first pass kept a trajectory reporting 1541.94x. No fused Triton kernel
    # is three orders of magnitude faster than its Torch reference; that is the
    # measurement being gamed -- a decoy kernel that is never called, or real
    # computation skipped -- which is the exact reward hacking Dr. Kernel was
    # written to defeat. Training on those teaches the model to cheat, so cap
    # well above genuine fusion wins (typically 1-10x) and far below absurdity.
    ap.add_argument("--max-speedup", type=float, default=50.0)
    ap.add_argument("--max-seq-tokens", type=int, default=17408,
                    help="model's max_seq_length; longer trajectories get truncated mid-refinement")
    ap.add_argument("--max-add-tokens", type=float, default=90e6,
                    help="cap on tokens added, to keep SFT inside one allocation")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from kore.data.decontam import heldout_task_ids, heldout_families, record_family

    ids = {t for t in heldout_task_ids() if t}
    fams = {f for f in heldout_families() if f}

    from datasets import load_dataset

    print(f"loading {args.repo_id}", flush=True)
    ds = load_dataset(args.repo_id, split="train")
    print(f"  {len(ds):,} trajectories, columns={ds.column_names}", flush=True)

    kept, stats = [], collections.Counter()
    added_chars = 0
    speedups = []
    lengths = []
    # Highest-speedup first: if the token budget binds, spend it on the
    # trajectories that best demonstrate a real optimization.
    order = sorted(range(len(ds)), key=lambda i: -(ds[i]["final_speedup"] or 0.0))
    for i in order:
        row = ds[i]
        stats["seen"] += 1
        sp = row.get("final_speedup") or 0.0
        msgs = row.get("messages") or []
        if not msgs:
            stats["drop_empty"] += 1
            continue
        if sp < args.min_speedup:
            stats["drop_slow"] += 1
            continue
        if sp > args.max_speedup:
            stats["drop_implausible_speedup"] += 1
            continue
        txt = _text(msgs)
        est_tokens = len(txt) / CHARS_PER_TOKEN
        if est_tokens > args.max_seq_tokens:
            # Truncation would cut the trajectory mid-refinement, teaching the
            # model to start an optimization and never finish it.
            stats["drop_too_long"] += 1
            continue
        if any(t in txt for t in ids):
            stats["drop_contaminated_id"] += 1
            continue
        rec = {"messages": [dict(m) for m in msgs], "_source": "kernel_multiturn_refine",
               "_speedup": float(sp), "_rounds": int(row.get("num_rounds") or 0)}
        try:
            fam = record_family(rec)
        except Exception:
            fam = ""
        if fam and fam in fams:
            stats["drop_contaminated_family"] += 1
            continue
        if added_chars + len(txt) > args.max_add_tokens * CHARS_PER_TOKEN:
            stats["drop_budget"] += 1
            continue
        added_chars += len(txt)
        speedups.append(sp)
        lengths.append(est_tokens)
        kept.append(rec)

    print("\nfilter results:")
    for k in ("seen", "drop_empty", "drop_slow", "drop_implausible_speedup",
              "drop_too_long", "drop_contaminated_id",
              "drop_contaminated_family", "drop_budget"):
        print(f"  {stats[k]:>7,}  {k}")
    print(f"  {len(kept):>7,}  KEPT")
    if speedups:
        speedups.sort()

        def pct(p):
            return speedups[min(len(speedups) - 1, int(p * len(speedups)))]

        print(f"  speedup kept: min={speedups[0]:.2f} p50={pct(0.50):.2f} "
              f"p90={pct(0.90):.2f} p99={pct(0.99):.2f} max={speedups[-1]:.2f}")
    if lengths:
        lengths.sort()

        def lpct(p):
            return lengths[min(len(lengths) - 1, int(p * len(lengths)))]

        print(f"  est tokens/traj: p50={lpct(0.50):,.0f} p90={lpct(0.90):,.0f} "
              f"max={lengths[-1]:,.0f} (limit {args.max_seq_tokens:,})")
    print(f"  tokens added (est): {added_chars/CHARS_PER_TOKEN/1e6:.1f}M")

    base = pathlib.Path(args.base_mixture)
    base_rows = base_chars = 0
    with base.open() as f:
        for line in f:
            if line.strip():
                base_rows += 1
                base_chars += len(line)
    print(f"\nbase mixture: {base_rows:,} rows, ~{base_chars/CHARS_PER_TOKEN/1e6:.0f}M tokens")
    print(f"new mixture : {base_rows + len(kept):,} rows, "
          f"~{(base_chars + added_chars)/CHARS_PER_TOKEN/1e6:.0f}M tokens "
          f"(+{100*added_chars/max(base_chars,1):.0f}%)")

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as w:
        with base.open() as f:
            for line in f:
                if line.strip():
                    w.write(line if line.endswith("\n") else line + "\n")
                    n += 1
        for rec in kept:
            w.write(json.dumps(rec) + "\n")
            n += 1
    print(f"\nwrote {n:,} rows to {out} ({out.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
