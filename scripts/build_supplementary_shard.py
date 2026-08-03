#!/usr/bin/env python3
"""Select the tasks the main datagen wave under-samples, and shard them locally.

The main wave draws from a ~14.7k pool that is dominated by KernelBook-derived
convolution and GEMM modules, so its family mix came out convolution 4,888 /
gemm 3,843 / attention 354 / moe 45 / quantization 31 episodes. That ordering is
close to backwards for what this model is meant to be good at:

* AgentKernelArena's two highest published Opus bars are HIP, not Triton --
  torch2hip 6.89x and hip2hip 6.69x against triton2triton 2.13x -- so HIP tasks
  are where the benchmark is won or lost, and all 188 of ours now have gfx950
  execution evidence.
* Attention is where the hard AMD work lives (flash variants, MQA/GQA decode,
  sliding window, FP8 KV) and where HipKittens measures frontier models losing
  1.3-3.0x to hand-tuned C++.
* The base model is itself an MoE, and MXFP4/FP8 is the headline MI355X numeric
  feature. 45 and 31 episodes respectively is not coverage, it is a rounding
  error.

So this selects by family rather than sampling uniformly, and spends MORE
episodes per task than the main wave (which uses 6). That is deliberate: for a
scarce high-value task the marginal trajectory is worth far more than the
5,000th convolution, and the step-centric extractor needs several attempts at
the same task before it can tell a real revision from a lucky one.

Emits plain shard files for scripts/run_agentic_shard.py, so this runs on a
loose box with free GPUs and needs no scheduler.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Ordered by how badly the main wave under-serves them, most starved first.
# Substrings are matched against the task id, which is how these families are
# actually named in the registry.
PRIORITY_FAMILIES: dict[str, tuple[str, ...]] = {
    "attention": ("flash_attn", "attention", "attn"),
    "moe": ("moe", "expert", "topk_gate", "router"),
    "quantization": ("quant", "fp8", "mxfp4", "int8", "dequant", "scale_block"),
    "positional": ("rope", "alibi", "rotary", "positional"),
    "sparse": ("sparse", "block_sparse", "2to4"),
    "fusion": ("fusion", "fused"),
}


def classify(task_id: str) -> str | None:
    tid = task_id.lower()
    for family, keys in PRIORITY_FAMILIES.items():
        if any(k in tid for k in keys):
            return family
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--include-hip", action="store_true", default=True,
                    help="include every HIP task (AKA's highest-value category)")
    ap.add_argument("--no-include-hip", dest="include_hip", action="store_false")
    args = ap.parse_args()

    from kore.tasks.registry import all_tasks, is_heldout

    chosen: dict[str, str] = {}     # task_id -> why it was chosen
    for task in all_tasks():
        # Never train on a held-out task; the whole point of the split is that
        # the eval number means something.
        try:
            if is_heldout(task):
                continue
        except Exception:  # noqa: BLE001 - a task without a split decision
            pass
        family = classify(task.task_id)
        if family:
            chosen[task.task_id] = family
        elif args.include_hip and getattr(task, "backend", "") == "hip":
            chosen[task.task_id] = "hip"

    if not chosen:
        print("no tasks selected", file=sys.stderr)
        return 1

    from collections import Counter
    counts = Counter(chosen.values())
    print("selected tasks by reason:")
    for reason, n in counts.most_common():
        print(f"  {reason:14} {n}")
    print(f"  {'TOTAL':14} {len(chosen)}")

    out = pathlib.Path(args.shard_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(chosen)
    for i in range(args.shards):
        part = ids[i::args.shards]
        (out / f"shard_{i:03d}.txt").write_text("\n".join(part) + "\n")
        print(f"shard_{i:03d}.txt: {len(part)} tasks")

    (out / "manifest.json").write_text(json.dumps({
        "purpose": "supplementary wave for families the main wave under-samples",
        "n_tasks": len(ids),
        "by_reason": dict(counts),
        "shards": args.shards,
        "selection": {k: list(v) for k, v in PRIORITY_FAMILIES.items()},
        "include_hip": args.include_hip,
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
