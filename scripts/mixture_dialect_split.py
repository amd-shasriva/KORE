#!/usr/bin/env python3
"""Rows and tokens of an SFT mixture, split by capability and kernel dialect.

The parity question -- how far HIP and FlyDSL trail Triton -- has to be asked
of the mixture that is actually trained on, not of the mined roots beside it.
Measuring the roots gets it wrong twice over: the win-group records there are
mostly candidate metadata that never reaches the model, and the same data is
present again in every mixture built from it, so a directory listing charges
for it two or three times.

Dialect is read from the text the model sees. A mixture row has no twin suffix
to read -- it is a chat transcript by then -- so the marker has to be the code
itself: ``@triton.jit`` and ``tl.`` for Triton, ``flyc.jit`` for FlyDSL,
``__global__`` and the HIP runtime calls for HIP.

Tokens are chars/4 unless a tokenizer is given, and the estimate is labelled as
one. The v4 config records both for the same file -- 606M estimated against
830M real -- so the ratio to a true count is about 1.37x.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

TRITON = re.compile(r"@triton\.jit|\btl\.(load|store|program_id)|import triton")
FLYDSL = re.compile(r"flyc\.jit|\bflydsl\b|import flydsl")
HIP = re.compile(r"__global__|hipLaunchKernelGGL|#include <hip|hipMalloc|__hip")


def text_of(row: dict) -> str:
    """Everything in the row the model is trained on."""
    parts: list[str] = []
    for key in ("messages", "prompt", "chosen"):
        val = row.get(key)
        if isinstance(val, list):
            for m in val:
                if isinstance(m, dict) and isinstance(m.get("content"), str):
                    parts.append(m["content"])
                elif isinstance(m, str):
                    parts.append(m)
        elif isinstance(val, str):
            parts.append(val)
    if isinstance(row.get("text"), str):
        parts.append(row["text"])
    return "\n".join(parts)


def dialect_of(text: str) -> str:
    """Which kernel language this row teaches, or none.

    Counted rather than short-circuited: a HIP row routinely quotes the Triton
    original it is a twin of, so first-match-wins labelled HIP rows as Triton.
    """
    scores = {
        "FlyDSL": len(FLYDSL.findall(text)),
        "HIP": len(HIP.findall(text)),
        "Triton": len(TRITON.findall(text)),
    }
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] else "non-kernel"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    args = ap.parse_args()

    cpt = args.chars_per_token
    chars: dict[str, int] = defaultdict(int)
    rows: dict[str, int] = defaultdict(int)
    total_rows = 0

    for p in args.paths:
        path = Path(p)
        if not path.exists():
            print(f"missing: {path}")
            continue
        with path.open("r", errors="ignore") as fh:
            for line in fh:
                if len(line) < 2:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                text = text_of(row)
                d = dialect_of(text)
                chars[d] += len(text)
                rows[d] += 1
                total_rows += 1

    total = sum(chars.values()) or 1
    print(f"\nrows: {total_rows:,}")
    print(f"{'bucket':<12}{'tokens':>14}{'share':>9}{'rows':>11}")
    print("-" * 46)
    for d in ("Triton", "HIP", "FlyDSL", "non-kernel"):
        print(f"{d:<12}{chars[d]/cpt:>14,.0f}{100*chars[d]/total:>8.1f}%{rows[d]:>11,}")
    print(f"{'TOTAL':<12}{total/cpt:>14,.0f}{100:>8.1f}%{total_rows:>11,}")

    kern = sum(chars[d] for d in ("Triton", "HIP", "FlyDSL")) or 1
    print(f"\nkernel tokens: {kern/cpt:,.0f}  "
          f"({100*kern/total:.1f}% of the mixture)")
    tri = chars["Triton"] or 1
    print("parity against Triton, within the kernel half:")
    for d in ("HIP", "FlyDSL"):
        print(f"  {d:<8}{100*chars[d]/tri:>7.1f}%   "
              f"({(tri - chars[d])/cpt:>12,.0f} tokens behind)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
