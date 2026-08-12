#!/usr/bin/env python
"""Carve a held-out eval slice out of the v5 mixture, and remove it from training.

The run's stated top risk is instruction-following collapse, it was measured once
already at a higher learning rate, and until now there was no signal on it until
the run finished: 1,867 optimizer steps and ~30 hours before anyone could say
whether the model still follows instructions. This produces the missing signal.

Two properties matter more than the size of the slice.

**It is removed from training, not merely copied.** Sampling rows into an eval file
while leaving them in the training file measures memorisation. This rewrites the
mixture without them.

**It is labelled by capability, not pooled.** A single scalar eval loss over a
mixed slice cannot distinguish "kernels improved and chat collapsed" from "nothing
moved", and those call for opposite decisions. Rows are stratified and tagged so
per-capability loss is readable separately.

Also worth recording for whoever adds to this later: the split happens BEFORE the
trainer's repair up-weighting duplicates rows. With ``repair_loss_weight`` now 1.0
that duplication is off, but if it is ever raised again, splitting afterwards would
put one copy of a repair row in train and its twin in eval.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


def _msg_hash(messages) -> str:
    return hashlib.sha1(
        json.dumps(messages, sort_keys=True).encode("utf-8", "ignore")).hexdigest()

REPO = Path("/home/shasriva/Kore-RL/KORE")

#: Capability groups, and how many rows to hold out of each. Sized so the whole
#: slice is ~0.4% of the corpus: large enough for a stable per-group loss, small
#: enough that removing it costs nothing, and cheap enough to evaluate often.
#: Instruction-following and chat are over-sampled relative to their share of the
#: corpus, because they are what the run risks losing rather than what it teaches.
GROUPS: dict[str, tuple[tuple[str, ...], int]] = {
    # The capability being taught. Split across shapes so a regression in one
    # (say, dialect porting) is not hidden by the others.
    "kernel_generate": (("kernel_torch2hip", "kernel_triton2hip",
                         "kernel_instruction_hip", "kernel_torch2flydsl",
                         "kernel_triton2flydsl", "kernel_instruction_flydsl",
                         "kernel_torch2kernel", "kernel_instruction"), 240),
    "kernel_repair": (("kernel_repair",), 80),
    "kernel_optimize": (("kernel_gold_win", "kernel_win", "kernel_step_centric"), 80),
    # The capabilities being risked.
    "instruction_following": (("instruction_following",), 120),
    "chat": (("replay_chat", "general_chat", "chat"), 120),
    "general_code": (("replay_code", "general_code"), 120),
    "tool_use": (("replay_tool_use", "agentic_tooluse", "tool_use"), 80),
    "math": (("math_reasoning", "math"), 60),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=str(REPO / "data/v5_sft.jsonl"))
    ap.add_argument("--out-eval", default=str(REPO / "data/v5_eval.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.sft)
    rng = random.Random(args.seed)

    want: dict[str, int] = {}
    group_of: dict[str, str] = {}
    for group, (sources, n) in GROUPS.items():
        want[group] = n
        for s in sources:
            group_of[s] = group

    # Reservoir sample per group in one streaming pass: the file is 1.8GB and there
    # is no reason to hold it in memory to pick 900 rows out of it.
    picked: dict[str, list] = collections.defaultdict(list)
    seen: collections.Counter = collections.Counter()
    total = 0
    with src.open(errors="ignore") as fh:
        for lineno, line in enumerate(fh):
            if not line.strip():
                continue
            total += 1
            try:
                src_name = str(json.loads(line).get("_source") or "")
            except Exception:  # noqa: BLE001 - torn line
                continue
            group = group_of.get(src_name)
            if group is None:
                continue
            seen[group] += 1
            k = want[group]
            if len(picked[group]) < k:
                picked[group].append((lineno, line))
            else:
                j = rng.randint(0, seen[group] - 1)
                if j < k:
                    picked[group][j] = (lineno, line)

    # Hold out by CONTENT, not by line number. The mixture contains deliberate
    # duplicates -- scarce slices are upsampled, so ~20% of rows are a second copy
    # of another -- and removing one line leaves its twins in training. Measured on
    # the first attempt: 296 of 900 held-out rows still had a copy in train. Hashing
    # the message list catches every copy.
    hold_hashes: dict[str, str] = {}
    for group, rows in picked.items():
        for _lineno, line in rows:
            try:
                msgs = json.loads(line).get("messages")
            except Exception:  # noqa: BLE001
                continue
            hold_hashes[_msg_hash(msgs)] = group

    print(f"source rows: {total:,}")
    print(f"{'group':<24}{'available':>11}{'held out':>10}")
    for group in GROUPS:
        print(f"{group:<24}{seen[group]:>11,}{len(picked[group]):>10,}")
    n_hold = len(hold_hashes)
    print(f"{'TOTAL':<24}{'':>11}{n_hold:>10,}  "
          f"({100 * n_hold / max(1, total):.2f}% of the corpus)")

    # Write eval, tagged by group so per-capability loss is separable. Deduped by
    # the same hash, so the eval slice itself carries no repeats.
    out_eval = Path(args.out_eval)
    written: set = set()
    n_eval = 0
    with out_eval.open("w") as fh:
        for group, rows in picked.items():
            for _lineno, line in rows:
                rec = json.loads(line)
                key = _msg_hash(rec.get("messages"))
                if key in written:
                    continue
                written.add(key)
                rec["_eval_group"] = group
                fh.write(json.dumps(rec) + "\n")
                n_eval += 1

    # Rewrite training without the held-out rows.
    tmp = src.with_suffix(".jsonl.tmp")
    kept = removed = 0
    with src.open(errors="ignore") as fin, tmp.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                msgs = json.loads(line).get("messages")
            except Exception:  # noqa: BLE001
                continue
            if _msg_hash(msgs) in hold_hashes:
                removed += 1
                continue
            fout.write(line)
            kept += 1
    tmp.replace(src)
    print(f"\nwrote {out_eval}  ({n_eval:,} distinct rows)")
    print(f"rewrote {src}  ({kept:,} rows, was {total:,}; removed {removed:,} "
          f"including duplicate copies of held-out rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
