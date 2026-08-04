#!/usr/bin/env python3
"""Blend general replay into the v4 kernel mixture, to a token budget.

v4's AMD-native content is ~48k rows of multi-turn kernel work, and left alone it
would be ~87% of the training tokens against v3's ~9%. That is a large enough
distribution shift to cost instruction-following on a 30B instruct model, and the
failure is invisible in the loss curve -- you get a model that writes good kernels
and stops following directions.

Replay fixes it, and the literature is specific about how much. Ibrahim et al.
("Simple and Scalable Strategies to Continually Pre-train LLMs") find that even
5% replay sharply reduces forgetting while the benefit plateaus near 25-30%,
because what anchors the model is a steady gradient pointing back at the base
distribution, not a large one. So this targets:

    kernel   70%   the thing we are actually training
    chat     17%   preserves conversation and instruction-following
    coding   13%   preserves algorithmic reasoning under specialisation

Budgets are in TOKENS, not rows, because rows are wildly misleading here: in v3
math_reasoning was 8.6% of rows and 26% of tokens (10,632 tok/row) while
general_code was 8.6% of rows and 1.2% of tokens. Sampling to a row target would
have silently produced a completely different mixture than intended.

Sources are whatever is already in the local HF cache -- this host is air-gapped
at training time, so a source that needs a download is not a source.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

# (dataset, config, split, bucket, preferred text/message fields)
CANDIDATES = [
    ("allenai/tulu-3-sft-mixture", None, "train", "chat", ("messages",)),
    ("nvidia/OpenCodeInstruct", None, "train", "coding", ("input", "output")),
    ("open-thoughts/OpenThoughts3-1.2M", None, "train", "coding", ("conversations", "messages")),
    ("livecodebench/code_generation_lite", None, "test", "coding", ("question_content",)),
]


def est_tokens(text: str) -> int:
    """chars/3.6, calibrated against this run's tokenizer.

    v3's actual token count came in 37% above a chars/4 estimate (332M measured
    against 243M estimated), and code tokenises denser than prose. Budgeting with
    the uncorrected estimate would overshoot the mixture by a third.
    """
    return int(len(text) / 3.6)


def to_messages(row: dict, fields: tuple) -> list | None:
    """Coerce a source row into the chat schema the mixture uses."""
    for f in fields:
        val = row.get(f)
        if isinstance(val, list) and val and isinstance(val[0], dict):
            out = []
            for m in val:
                role = m.get("role") or m.get("from") or "user"
                content = m.get("content") or m.get("value") or ""
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant", "model"):
                    role = "assistant"
                if content:
                    out.append({"role": role, "content": str(content)})
            if len(out) >= 2:
                return out
    # Fall back to an instruction/response pair when the source is not chat-shaped.
    if len(fields) == 2:
        q, a = row.get(fields[0]), row.get(fields[1])
        if q and a:
            return [{"role": "user", "content": str(q)},
                    {"role": "assistant", "content": str(a)}]
    single = row.get(fields[0]) if fields else None
    if isinstance(single, str) and len(single) > 40:
        return None    # a prompt with no response teaches nothing under completion loss
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-tokens", type=int, required=True,
                    help="measured token count of the kernel side of v4")
    ap.add_argument("--chat-share", type=float, default=0.17)
    ap.add_argument("--coding-share", type=float, default=0.13)
    ap.add_argument("--kernel-share", type=float, default=0.70)
    ap.add_argument("--have-chat-tokens", type=int, default=0,
                    help="chat tokens already present in the base mixture")
    ap.add_argument("--have-coding-tokens", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    total = args.kernel_tokens / args.kernel_share
    need = {
        "chat": max(0, int(total * args.chat_share) - args.have_chat_tokens),
        "coding": max(0, int(total * args.coding_share) - args.have_coding_tokens),
    }
    print(f"kernel tokens   : {args.kernel_tokens:,}")
    print(f"implied total   : {total:,.0f}")
    print(f"need chat       : {need['chat']:,} (have {args.have_chat_tokens:,})")
    print(f"need coding     : {need['coding']:,} (have {args.have_coding_tokens:,})")
    if args.plan_only:
        return 0

    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        print(f"datasets unavailable: {exc}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    written = {"chat": 0, "coding": 0}
    rows_out = 0
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as fh:
        for name, cfg, split, bucket, fields in CANDIDATES:
            if written[bucket] >= need[bucket]:
                continue
            try:
                ds = load_dataset(name, cfg, split=split, streaming=True)
            except Exception as exc:  # noqa: BLE001 - not cached is a normal outcome
                print(f"  skip {name}: {str(exc)[:80]}")
                continue
            took = 0
            for row in ds:
                if written[bucket] >= need[bucket]:
                    break
                msgs = to_messages(row, fields)
                if not msgs:
                    continue
                n = sum(est_tokens(m["content"]) for m in msgs)
                if n < 30 or n > 16000:
                    continue      # mirrors the trainer's overlong drop
                fh.write(json.dumps({
                    "messages": msgs,
                    "_source": f"replay_{bucket}",
                    "_origin": name,
                }) + "\n")
                written[bucket] += n
                took += 1
                rows_out += 1
            print(f"  {name} [{bucket}]: {took} rows, {written[bucket]:,} tok cumulative")

    print(f"\nwrote {rows_out} rows -> {out}")
    for b in ("chat", "coding"):
        pct = 100.0 * written[b] / need[b] if need[b] else 100.0
        print(f"  {b}: {written[b]:,} / {need[b]:,} tokens ({pct:.0f}% of target)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
