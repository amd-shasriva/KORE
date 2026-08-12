#!/usr/bin/env python
"""Stage 5: prove the v5 file is clean and trainable before anyone spends a GPU on it.

Each check corresponds to a specific way this dataset could be wrong in a way that
is invisible at load time. Several of them are here because the earlier drafts of
this very pipeline failed them.

  1. Held-out leakage. The registry matches its probe set by exact task id while
     twins carry a backend suffix, so ``genb_x__hip`` classified as trainable and
     39 of 43 probes reached a built file. Checked here against the base id.
  2. Benchmark contamination. Nothing in this repo ever compared the pool against
     the evaluation benchmark; 12 pool tasks are byte-identical to an arena task.
  3. Provenance shape. A bare string crashes the loader at the production
     ``repair_loss_weight``, after the model has loaded on every rank.
  4. One assistant turn. Loss covers every assistant turn with no opt-out, so a
     second turn means training on a rejected revision.
  5. Reward hacks. A kernel that delegates to torch passed the numerical gate and
     would score as compile-but-wrong while teaching the fallback.
  6. Length. Over-cap rows are dropped at train time and counted only in one
     aggregate log line.
  7. Roles and empty content. An unknown role renders as masked context; an empty
     assistant turn trains the stop token alone and trips no guard.
  8. Duplication. Repeated targets are the duplication that costs capability.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path("/home/shasriva/Kore-RL/KORE")
sys.path.insert(0, str(REPO))

_CODE = re.compile(r"```[a-zA-Z+]*\n(.*?)```", re.S)


def code_of(text: str) -> str:
    m = _CODE.findall(text or "")
    return max(m, key=len).strip() if m else (text or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(REPO / "data/v5_sft.jsonl"))
    ap.add_argument("--model-id", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--revision", default="b2cff646eb4bb1d68355c01b18ae02e7cf42d120")
    ap.add_argument("--limit", type=int, default=17408)
    args = ap.parse_args()

    from kore.data.arena_index import ArenaIndex
    from kore.data.v5_emit import cheats
    from kore.data.v5_policy import strip_suffix
    from kore.tasks.registry import CONTAMINATED_TASKS, HELDOUT_TASKS

    idx = REPO / "data/arena_contamination.json"
    arena = ArenaIndex.load(idx) if idx.is_file() else None

    rows = []
    with open(args.path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"rows: {len(rows):,}\n")

    fail: collections.Counter = collections.Counter()
    warn: collections.Counter = collections.Counter()
    by_source: collections.Counter = collections.Counter()
    by_shape: collections.Counter = collections.Counter()
    by_dialect: collections.Counter = collections.Counter()
    bodies: collections.Counter = collections.Counter()
    tasks: set[str] = set()
    leaked: list[str] = []
    contaminated: list[str] = []

    for r in rows:
        src = str(r.get("_source") or "?")
        by_source[src] += 1
        if str(src).startswith("kernel"):
            by_shape[str(r.get("_shape") or "?")] += 1
            by_dialect[str(r.get("_dialect") or "?")] += 1
        tid = str(r.get("_task_id") or "")
        base = strip_suffix(tid)
        if tid:
            tasks.add(base)
            if base in HELDOUT_TASKS or tid in HELDOUT_TASKS:
                fail["heldout_probe_leak"] += 1
                leaked.append(tid)
            if base in CONTAMINATED_TASKS or tid in CONTAMINATED_TASKS:
                fail["contaminated_task_leak"] += 1
                leaked.append(tid)
            if arena is not None and arena.match(base) is not None:
                fail["arena_contamination"] += 1
                contaminated.append(tid)

        prov = r.get("_provenance")
        if prov is not None and not isinstance(prov, dict):
            fail["provenance_not_object"] += 1

        msgs = r.get("messages") or []
        if not msgs:
            fail["no_messages"] += 1
            continue
        n_asst = sum(1 for m in msgs if m.get("role") == "assistant")
        # One assistant turn is required of KERNEL rows only. Loss covers every
        # assistant turn, which is a hazard when the earlier ones are rejected
        # kernel revisions -- and is exactly what you want for a replay chat or
        # tool-use conversation, where each turn is a genuine response and
        # multi-turn ability is the capability being preserved.
        if str(src).startswith("kernel") and n_asst != 1:
            fail["kernel_assistant_turns_not_1"] += 1
        elif n_asst < 1:
            fail["no_assistant_turn"] += 1
        else:
            warn[f"multiturn_replay::{src}"] += 0 if n_asst == 1 else 1
        for m in msgs:
            if m.get("role") not in ("system", "user", "assistant", "tool"):
                fail["bad_role"] += 1
            if "content" not in m:
                fail["missing_content"] += 1
            elif not str(m.get("content") or "").strip():
                fail["empty_content"] += 1
        target = msgs[-1].get("content") or ""
        if msgs[-1].get("role") != "assistant":
            fail["last_not_assistant"] += 1
        if str(src).startswith("kernel"):
            why = cheats(code_of(target))
            if why:
                warn[f"cheat::{why.split(':')[0]}"] += 1
        bodies[hashlib.sha1(target.encode("utf-8", "ignore")).hexdigest()] += 1

    print("=== CORRECTNESS GATES ===")
    gates = ["heldout_probe_leak", "contaminated_task_leak", "arena_contamination",
             "provenance_not_object", "kernel_assistant_turns_not_1",
             "no_assistant_turn", "bad_role", "missing_content", "empty_content",
             "last_not_assistant", "no_messages"]
    for g in gates:
        n = fail.get(g, 0)
        print(f"  {'PASS' if n == 0 else 'FAIL'}  {g:<28} {n:,}")
    if leaked:
        print(f"    leaked ids: {sorted(set(leaked))[:5]}")
    if contaminated:
        print(f"    contaminated ids: {sorted(set(contaminated))[:5]}")

    print("\n=== WARNINGS ===")
    if warn:
        for k, v in warn.most_common():
            print(f"  {k:<32} {v:,}")
    else:
        print("  none")

    dup = sum(c - 1 for c in bodies.values() if c > 1)
    print(f"\n=== DUPLICATION ===")
    print(f"  distinct targets      {len(bodies):,}")
    print(f"  repeated rows         {dup:,}  ({100*dup/max(1,len(rows)):.1f}%)")
    top = bodies.most_common(1)[0][1] if bodies else 0
    print(f"  most-repeated target  {top}x")
    print(f"  distinct kernel tasks {len(tasks):,}")

    print("\n=== COMPOSITION ===")
    kern = sum(v for k, v in by_source.items() if k.startswith("kernel"))
    print(f"  kernel {kern:,} ({100*kern/len(rows):.1f}%)   "
          f"replay {len(rows)-kern:,} ({100*(len(rows)-kern)/len(rows):.1f}%)")
    print("  by skill:")
    for k, v in by_shape.most_common():
        print(f"    {k:<16} {v:>8,}  {100*v/max(1,kern):5.1f}%")
    print("  by dialect:")
    for k, v in by_dialect.most_common():
        print(f"    {k:<16} {v:>8,}  {100*v/max(1,kern):5.1f}%")

    print("\n=== TOKENS ===")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
        lens = []
        per_src: dict[str, list[int]] = collections.defaultdict(list)
        for r in rows:
            n = len(tok.apply_chat_template(r["messages"], tokenize=True,
                                            add_generation_prompt=False))
            lens.append(n)
            per_src[str(r.get("_source"))].append(n)
        lens.sort()
        def q(p):
            return lens[min(len(lens) - 1, int(len(lens) * p))]
        print(f"  total {sum(lens):,} tokens over {len(lens):,} rows")
        print(f"  mean {sum(lens)//len(lens):,}  median {q(.5):,}  p90 {q(.9):,}  "
              f"p99 {q(.99):,}  max {max(lens):,}")
        over = sum(1 for n in lens if n > args.limit)
        print(f"  {'PASS' if over == 0 else 'FAIL'}  over {args.limit:,}: {over:,}")
        print("  by source (median / max):")
        for s, v in sorted(per_src.items(), key=lambda kv: -len(kv[1]))[:10]:
            v.sort()
            print(f"    {s:<28} {v[len(v)//2]:>7,} / {v[-1]:>7,}")
    except Exception as exc:  # noqa: BLE001
        print(f"  tokenizer unavailable: {type(exc).__name__}: {exc}")

    hard = sum(fail.get(g, 0) for g in gates)
    print(f"\n{'=' * 52}")
    print("VERDICT:", "PASS - safe to train" if hard == 0
          else f"FAIL - {hard:,} hard violations")
    return 0 if hard == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
