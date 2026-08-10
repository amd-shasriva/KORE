#!/usr/bin/env python3
"""Token share per kernel dialect across every mined corpus on disk.

Parity is the question the mining plan is steered by -- Triton has been mined
since the beginning and HIP and FlyDSL were added later, so "how far behind are
they" decides where the miners go. Answering it from row counts is misleading:
an agentic multi-turn HIP row is several times longer than a single-turn Triton
one, so rows understate HIP's share of the gradient and tokens do not.

Dialect is read from the task id suffix, which is where the twin path records
it: ``__hip``/``__hipf`` for HIP, ``__flydsl`` for FlyDSL, and a bare id for the
Triton original the twin was made from.

Sizes here are hundreds of thousands of rows across ~12GB, so the pass avoids
json.loads: the id is pulled with a regex and the raw line length is the token
proxy. chars/4 is an estimate and is labelled as one.

Quarantined and backup paths are skipped -- they are excluded from training, so
counting them would overstate the corpus.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

TASK_ID = re.compile(rb'"task_id"\s*:\s*"([^"]*)"')
SKIP_PARTS = ("_quarantine", ".bak", "telemetry", "ledger_backup")


def dialect_of(task_id: bytes) -> str:
    if task_id.endswith(b"__hip") or task_id.endswith(b"__hipf"):
        return "HIP"
    if task_id.endswith(b"__flydsl"):
        return "FlyDSL"
    if task_id.endswith(b"__triton"):
        return "Triton"
    return "Triton"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    args = ap.parse_args()

    chars: dict[str, int] = defaultdict(int)
    rows: dict[str, int] = defaultdict(int)
    per_root: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    files = [
        p for p in Path(args.data_dir).glob("*/**/*.jsonl")
        if not any(s in str(p) for s in SKIP_PARTS)
    ]
    print(f"scanning {len(files)} files", flush=True)

    for i, path in enumerate(files, 1):
        root = path.relative_to(args.data_dir).parts[0]
        # The filename is the task id for the per-task layouts, which lets a
        # whole file be classified once instead of per line.
        stem = path.stem.encode()
        file_dialect = dialect_of(stem) if b"__" in stem else None
        try:
            with path.open("rb") as fh:
                for line in fh:
                    if len(line) < 2:
                        continue
                    if file_dialect is not None:
                        d = file_dialect
                    else:
                        m = TASK_ID.search(line)
                        d = dialect_of(m.group(1)) if m else "Triton"
                    chars[d] += len(line)
                    rows[d] += 1
                    per_root[root][d] += len(line)
        except OSError as exc:
            print(f"  skip {path}: {exc}", file=sys.stderr)
        if i % 200 == 0:
            print(f"  {i}/{len(files)} files", flush=True)

    cpt = args.chars_per_token
    total = sum(chars.values()) or 1
    print("\n=== token share by dialect (chars/%.0f estimate) ===" % cpt)
    print(f"{'dialect':<10}{'tokens':>14}{'share':>9}{'rows':>12}")
    for d in ("Triton", "HIP", "FlyDSL"):
        print(f"{d:<10}{chars[d]/cpt:>14,.0f}{100*chars[d]/total:>8.1f}%{rows[d]:>12,}")
    print(f"{'TOTAL':<10}{total/cpt:>14,.0f}{100:>8.1f}%{sum(rows.values()):>12,}")

    tri = chars["Triton"] or 1
    print("\n=== parity against Triton ===")
    for d in ("HIP", "FlyDSL"):
        print(f"  {d:<8} {100*chars[d]/tri:>6.1f}% of Triton"
              f"   ({(tri - chars[d])/cpt:>14,.0f} tokens behind)")

    print("\n=== by root (tokens) ===")
    for root in sorted(per_root, key=lambda r: -sum(per_root[r].values())):
        v = per_root[root]
        tot = sum(v.values())
        if tot / cpt < 100_000:
            continue
        print(f"  {root:<26} T={v['Triton']/cpt:>12,.0f}  "
              f"H={v['HIP']/cpt:>11,.0f}  F={v['FlyDSL']/cpt:>10,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
