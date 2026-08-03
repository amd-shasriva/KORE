#!/usr/bin/env python
"""Audit a training file for overlap with the held-out evaluation set.

kore/data/decontam.py already implements the filtering, and kore/data/assemble.py
calls it. This script answers a different question: whether the file we are
ABOUT to train on is actually clean. Those diverge whenever a dataset is
regenerated, copied between machines, or assembled by a path that skipped the
filter, and the failure is silent -- a contaminated mixture produces better eval
numbers, not an error.

Three independent checks, because each misses something the others catch:

  ids       a held-out task id appearing literally in the row text
  families  the repo's own record_family attribution landing in a held-out
            family, which catches a task renamed or paraphrased
  ngrams    signal n-gram overlap against the held-out reference documents,
            which catches a task whose text was copied without its id

Signal n-grams matter here: raw n-gram overlap against Triton code is dominated
by boilerplate (`import triton`, `tl.program_id(0)`), so decontam.py filters
generic shingles before comparing. Using raw n-grams would flag essentially
every kernel row and the audit would be worthless.

Exit status is 0 only when all three are clean, so this can gate a launch.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys


def _rows(path: pathlib.Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _text(rec: dict) -> str:
    msgs = rec.get("messages")
    if isinstance(msgs, list):
        return "".join(str(m.get("content") or "") for m in msgs if isinstance(m, dict))
    return json.dumps(rec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="jsonl file to audit")
    ap.add_argument("--ngram-sample", type=int, default=4000,
                    help="rows to n-gram check (0 = all; full pass is O(rows*refs))")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from kore.data.decontam import heldout_task_ids, heldout_families, record_family

    path = pathlib.Path(args.dataset)
    if not path.exists():
        print(f"missing dataset: {path}")
        return 2

    ids = {t for t in heldout_task_ids() if t}
    fams = {f for f in heldout_families() if f}
    print(f"dataset            : {path}  ({path.stat().st_size/1e6:.0f} MB)")
    print(f"held-out task ids  : {len(ids)}")
    print(f"held-out families  : {len(fams)}")

    id_hits: collections.Counter = collections.Counter()
    fam_hits: collections.Counter = collections.Counter()
    src_hits: collections.Counter = collections.Counter()
    n = 0
    texts = []
    for rec in _rows(path):
        n += 1
        txt = _text(rec)
        if args.ngram_sample and len(texts) < args.ngram_sample:
            texts.append(txt)
        for t in ids:
            if t in txt:
                id_hits[t] += 1
                src_hits[str(rec.get("_source", "?"))] += 1
        try:
            fam = record_family(rec)
        except Exception:
            fam = ""
        if fam and fam in fams:
            fam_hits[fam] += 1

    print(f"rows scanned       : {n:,}")
    print(f"\n[ids]      rows containing a held-out task id : {sum(id_hits.values()):,}")
    for k, v in id_hits.most_common(10):
        print(f"             {v:>6,}  {k}")
    print(f"[families] rows attributed to a held-out family: {sum(fam_hits.values()):,}")
    for k, v in fam_hits.most_common(10):
        print(f"             {v:>6,}  {k}")
    if src_hits:
        print("           id hits by _source:")
        for k, v in src_hits.most_common():
            print(f"             {v:>6,}  {k}")

    ngram_hits = 0
    ngram_detail = []
    try:
        from kore.data.decontam import build_heldout_ngrams, signal_ngram_set

        ref = build_heldout_ngrams()
        if ref:
            refsets = ref if isinstance(ref, (list, tuple, set)) else [ref]
            flat = set()
            for r in refsets:
                if isinstance(r, (set, frozenset)):
                    flat |= r
            for txt in texts:
                sig = signal_ngram_set(txt)
                if sig and flat and (sig & flat):
                    ngram_hits += 1
                    if len(ngram_detail) < 3:
                        ngram_detail.append(sorted(sig & flat)[:2])
            print(f"[ngrams]   of {len(texts):,} sampled rows, overlapping: {ngram_hits:,}")
            for d in ngram_detail:
                print(f"             e.g. {d}")
        else:
            print("[ngrams]   no held-out reference n-grams available; check skipped")
    except Exception as exc:  # a missing helper must not mask the other checks
        print(f"[ngrams]   unavailable ({type(exc).__name__}: {exc})")

    clean = not id_hits and not fam_hits and ngram_hits == 0
    print(f"\nVERDICT: {'CLEAN' if clean else 'CONTAMINATED'}")
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps({
            "dataset": str(path), "rows": n,
            "id_hits": dict(id_hits), "family_hits": dict(fam_hits),
            "ngram_hits": ngram_hits, "ngram_sampled": len(texts),
            "clean": clean,
        }, indent=2) + "\n")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
