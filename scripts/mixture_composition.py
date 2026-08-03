#!/usr/bin/env python3
"""Break an SFT mixture down by source, with row and token counts.

Planning the next mixture needs the current one measured rather than recalled:
the AMD-native fraction is the number the whole retrain is justified by, and it
is only meaningful next to the token counts, because an AMD-native multi-turn
kernel row is several times longer than a general-reasoning row. Counting rows
alone overstates how much of the model's gradient the general data is actually
receiving, and counting tokens alone hides how few distinct AMD examples there
are.

Tokenises with the model's own tokenizer when it is available and falls back to
a chars/4 estimate, saying which was used -- an estimate labelled as one is
fine, an estimate mistaken for a measurement is not.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict


def source_of(row: dict) -> str:
    """Best available provenance label for a mixture row."""
    for key in ("source", "_source", "origin", "dataset", "kind"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    prov = row.get("provenance")
    if isinstance(prov, dict):
        for key in ("source", "generator", "kind", "origin"):
            val = prov.get(key)
            if isinstance(val, str) and val:
                return val
    # Fall back to structural signals rather than guessing a name.
    if row.get("task_id"):
        return "amd-native (has task_id)"
    return "unlabelled"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--model", default="", help="tokenizer to use; blank = estimate")
    args = ap.parse_args()

    tok = None
    if args.model:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model)
        except Exception as exc:  # noqa: BLE001
            print(f"# tokenizer unavailable ({exc}); estimating chars/4")

    rows = Counter()
    chars: defaultdict[str, int] = defaultdict(int)
    tokens: defaultdict[str, int] = defaultdict(int)

    with open(args.path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            src = source_of(row)
            rows[src] += 1
            text = json.dumps(row.get("messages") or row)
            chars[src] += len(text)
            if tok is not None:
                tokens[src] += len(tok.encode(text, add_special_tokens=False))

    total_rows = sum(rows.values())
    unit = "tokens" if tok is not None else "tokens(est chars/4)"
    print(f"{'source':<40} {'rows':>8} {'share':>7} {unit:>18} {'tok/row':>9}")
    grand_tok = 0
    for src, n in rows.most_common():
        t = tokens[src] if tok is not None else chars[src] // 4
        grand_tok += t
        print(f"{src[:40]:<40} {n:>8} {100.0*n/total_rows:>6.1f}% "
              f"{t:>18,} {t//max(n,1):>9,}")
    print(f"{'TOTAL':<40} {total_rows:>8} {100.0:>6.1f}% {grand_tok:>18,} "
          f"{grand_tok//max(total_rows,1):>9,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
