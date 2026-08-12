#!/usr/bin/env python
"""Drop assistant targets that were cut off mid-generation, and rebalance eval.

The length gate in the build pipeline asks "is this row under the token cap", which
a truncated row passes by construction -- truncation is *how* it got under the cap.
So 552 math_reasoning rows shipped with their chain-of-thought severed mid-word:

    ...at n=11:11^3 ≡ 5 mod 17. So r ≡ 11 mod 17.\\n\\nThus let r=1
    ...1/6 ≈ 0.166666..., 1/9 ≈ 0.111111...,
    ...then g(x_6)= (g(x

As SFT targets these are worse than useless. Every one teaches that a plausible way
to finish a hard problem is to emit sixteen thousand tokens and stop mid-token
without an end-of-turn, and because they are the longest rows in the corpus they
carry 9.27M tokens -- 1.86% of all tokens, but a far larger share of the ~38.6%
that assistant-only loss actually trains on. A model that learns non-termination
from them fails in a way that is expensive to detect and unpleasant to serve.

Measured to be confined to math_reasoning: 552 truncated of 696 rows above 16,000
tokens corpus-wide, with the other 144 being kernel rows that legitimately end on a
brace or fence. The control confirms the signature rather than the heuristic being
loose -- 94% of capped math rows lack terminal punctuation against 3% of short ones.

Eval is repaired rather than merely shrunk: dropping truncated rows there would
leave the math group at ~29 rows, too few for a stable per-group loss, so clean rows
are drawn from the training pool to restore the group size and then removed from
training to keep the split disjoint.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

REPO = Path("/home/shasriva/Kore-RL/KORE")

#: A finished answer ends on sentence or code punctuation. Checked against the last
#: 40 characters so trailing whitespace or a short closing token does not matter.
_TERMINAL = re.compile(r'[.!?}\)\]>"\'`;:]\s*$|\\\]$|```\s*$')


def is_truncated(messages) -> bool:
    """True when the final assistant turn looks cut off mid-generation."""
    finals = [m.get("content") or "" for m in messages
              if isinstance(m, dict) and m.get("role") == "assistant"]
    if not finals:
        return False
    tail = finals[-1].rstrip()
    if not tail:
        return False
    return not _TERMINAL.search(tail[-40:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=str(REPO / "data/v5_sft.jsonl"))
    ap.add_argument("--eval", dest="ev", default=str(REPO / "data/v5_eval.jsonl"))
    #: Only rows long enough to have plausibly hit the cap. A short answer without
    #: terminal punctuation is usually a legitimate style (a bare number, a code
    #: fragment), not a truncation, and 17 of 552 short math rows match the pattern
    #: harmlessly -- so gating on length is what keeps this precise.
    ap.add_argument("--min-tokens", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sft_path, ev_path = Path(args.sft), Path(args.ev)

    # Pass 1: read eval, drop truncated, note how many of each group to restore.
    ev_rows, ev_dropped = [], collections.Counter()
    for line in ev_path.open(errors="ignore"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if (rec.get("_tokens", 0) >= args.min_tokens
                and is_truncated(rec.get("messages") or [])):
            ev_dropped[rec.get("_eval_group", "?")] += 1
            continue
        ev_rows.append(rec)
    print(f"eval: dropped {sum(ev_dropped.values())} truncated "
          f"({dict(ev_dropped)}), {len(ev_rows)} remain")

    # Pass 2: stream training, dropping truncated rows and collecting clean
    # candidates for the eval groups that need backfilling.
    ev_hashes = {json.dumps(r.get("messages"), sort_keys=True) for r in ev_rows}
    need = dict(ev_dropped)
    kept, dropped_train = [], 0
    pools: dict[str, list] = collections.defaultdict(list)
    src_to_group = {"math_reasoning": "math", "math": "math"}
    for line in sft_path.open(errors="ignore"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if (rec.get("_tokens", 0) >= args.min_tokens
                and is_truncated(rec.get("messages") or [])):
            dropped_train += 1
            continue
        group = src_to_group.get(str(rec.get("_source") or ""))
        if group and need.get(group):
            pools[group].append(rec)
        kept.append(rec)
    print(f"train: dropped {dropped_train} truncated, {len(kept)} remain")

    # Backfill eval from the clean pool, then remove those rows from training so the
    # two stay disjoint by content (including the mixture's upsampled duplicates).
    promoted_keys: set = set()
    for group, count in need.items():
        pool = [r for r in pools.get(group, [])
                if json.dumps(r.get("messages"), sort_keys=True) not in ev_hashes]
        rng.shuffle(pool)
        take = pool[:count]
        for rec in take:
            rec = dict(rec)
            rec["_eval_group"] = group
            ev_rows.append(rec)
            promoted_keys.add(json.dumps(rec.get("messages"), sort_keys=True))
        print(f"eval backfill {group}: wanted {count}, took {len(take)} "
              f"from {len(pool)} clean candidates")
    before = len(kept)
    kept = [r for r in kept
            if json.dumps(r.get("messages"), sort_keys=True) not in promoted_keys]
    print(f"train: removed {before - len(kept)} promoted-to-eval rows "
          f"(incl. duplicate copies)")

    with sft_path.open("w") as fh:
        for rec in kept:
            fh.write(json.dumps(rec) + "\n")
    with ev_path.open("w") as fh:
        for rec in ev_rows:
            fh.write(json.dumps(rec) + "\n")

    groups = collections.Counter(r.get("_eval_group") for r in ev_rows)
    print(f"\nwrote {sft_path}  ({len(kept):,} rows)")
    print(f"wrote {ev_path}  ({len(ev_rows):,} rows, groups={dict(sorted(groups.items()))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
