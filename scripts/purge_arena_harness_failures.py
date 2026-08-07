#!/usr/bin/env python
"""Drop arena rows that a harness bug scored, so resume re-judges only those.

A partial ledger is the only record of work already paid for, so the wrong move
after fixing a harness bug is to delete it and re-run all 413 tasks -- that
throws away hundreds of GPU-hours of verdicts that were never in question. The
right move is to remove exactly the rows whose verdict came from the bug, because
the arena's resume reads the ledgers to decide what is left and will re-score
anything absent.

Which rows those are is not a judgement call; each fixed bug leaves a signature:

  triton_collision   torch2flydsl candidates compiled and then failed every shape
                     on "aiter gluon kernels require triton>=3.6.0, found 3.5.1".
                     Two triton distributions both owned site-packages/triton and
                     the stray 3.5.1 had overwritten the 3.6.0 that torch pins.

  missing_extension_contract
                     torch2hip ships all 57 targets as zero-byte .hip files and
                     its loader reads ext.forward, but the prompt never said so,
                     so candidates were kernels without a pybind11 module and the
                     build failed before numerics. hip2hip, whose targets ship
                     non-empty, scored normally on the same toolchain.

Two rules make this safe to run unattended:

  * A row that was CORRECT is never purged, whatever its signature. A success is
    not a harness artifact, and re-rolling it can only lose points.
  * Ledgers are copied aside before being rewritten, and the rewrite is atomic,
    so an interrupted purge cannot leave a torn ledger.

    python scripts/purge_arena_harness_failures.py --out runs/aka_full_v4 --arm v4
    python scripts/purge_arena_harness_failures.py --out runs/aka_base --arm base --apply
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import shutil
import time

#: Signature -> (predicate, human reason). A predicate sees the parsed row and
#: returns True when that row's verdict is attributable to the named bug.
#:
#: Deliberately narrow. Matching on task_type alone would also sweep up rows that
#: failed for real reasons, and those are data.


def _is_triton_collision(row: dict) -> bool:
    err = (row.get("error") or "")
    return ("gluon kernels require triton" in err
            or "module_aiter_core" in err and "triton" in err)


def _is_missing_extension_contract(row: dict) -> bool:
    """A torch2hip candidate that never compiled.

    Every torch2hip target is empty, so the contract was missing on all of them;
    a candidate that failed to compile could not have been judged on numerics.
    Compiled-but-wrong is left alone: that verdict is about the kernel.
    """
    if (row.get("task_type") or "") != "torch2hip":
        return False
    return not row.get("compiled")


SIGNATURES = {
    "triton_collision": (
        _is_triton_collision,
        "aiter gluon refused triton 3.5.1 (fixed: triton-rocm 3.6.0 restored)",
    ),
    "missing_extension_contract": (
        _is_missing_extension_contract,
        "torch2hip built with no pybind11 module (fixed: prompt states the contract)",
    ),
}


def _rows(path: pathlib.Path):
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line), line
        except Exception:  # noqa: BLE001 - a torn last line after a kill
            continue


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="ledger directory, e.g. runs/aka_base")
    ap.add_argument("--arm", required=True, help="arm name, e.g. v4 or base")
    ap.add_argument("--signature", action="append", default=None,
                    choices=sorted(SIGNATURES),
                    help="restrict to one signature (default: all)")
    ap.add_argument("--apply", action="store_true",
                    help="actually rewrite the ledgers (default is a dry run)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    ledgers = sorted(out.glob(f"results_{args.arm}*.partial.jsonl"))
    if not ledgers:
        print(f"no ledgers under {out} for arm {args.arm}")
        return 1

    active = args.signature or sorted(SIGNATURES)
    print(f"=== {out}  arm={args.arm}  ledgers={len(ledgers)} ===")
    print("signatures: " + ", ".join(active))

    per_sig = collections.Counter()
    per_cat = collections.Counter()
    kept_correct = collections.Counter()
    total = purged = 0
    plan: list[tuple[pathlib.Path, list[str]]] = []

    for led in ledgers:
        keep: list[str] = []
        for row, raw in _rows(led):
            total += 1
            hit = None
            for name in active:
                pred, _ = SIGNATURES[name]
                try:
                    if pred(row):
                        hit = name
                        break
                except Exception:  # noqa: BLE001 - a malformed row is not a match
                    continue
            # A success is never a harness artifact.
            if hit and row.get("correct"):
                kept_correct[hit] += 1
                hit = None
            if hit:
                purged += 1
                per_sig[hit] += 1
                per_cat[row.get("task_type") or "?"] += 1
            else:
                keep.append(raw)
        plan.append((led, keep))

    print(f"\nrows: {total} total, {purged} attributable to a fixed bug, "
          f"{total - purged} kept")
    if per_sig:
        print("\nby signature:")
        for name, n in per_sig.most_common():
            print(f"  {n:4}  {name}  -- {SIGNATURES[name][1]}")
    if per_cat:
        print("\nby task type:")
        for cat, n in per_cat.most_common():
            print(f"  {n:4}  {cat}")
    if kept_correct:
        print("\nkept despite matching (they were CORRECT -- never re-roll a win):")
        for name, n in kept_correct.most_common():
            print(f"  {n:4}  {name}")

    if not args.apply:
        print("\ndry run; nothing written. re-run with --apply to purge.")
        return 0
    if not purged:
        print("\nnothing to purge.")
        return 0

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = out / f"ledger_backup_{args.arm}_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for led, keep in plan:
        shutil.copy2(led, backup / led.name)
        tmp = led.with_suffix(led.suffix + ".rewriting")
        with tmp.open("w") as fh:
            for raw in keep:
                fh.write(raw + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, led)   # atomic: a kill mid-purge leaves the old ledger
    print(f"\npurged {purged} row(s). originals copied to {backup}")
    print("the next arena run will re-score exactly those tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
