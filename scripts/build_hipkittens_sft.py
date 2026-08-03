#!/usr/bin/env python
"""Build the HipKittens SFT slice, decontaminated and deduplicated.

Emits ``data/b05factory/sft/hipkittens.jsonl`` plus a JSON report, so the slice
can be sized into the v3 mixture by ``scripts/build_sft_v3_mixture.py``.

Three gates, applied in this order, because each catches something the others do
not:

  contamination  a held-out task id appearing literally in a row, the repo's own
                 ``record_family`` attribution landing in a held-out family, and
                 signal-n-gram overlap against the held-out reference documents.
                 Same three checks as ``scripts/audit_decontamination.py``, run
                 BEFORE the slice is written rather than after, so a contaminated
                 row never reaches disk.

  cross-corpus   near-duplicate overlap against the existing mixture. Exact
  dedup          content hashing is not enough on its own: these rows are newly
                 authored prose, so they will never hash-collide with v2 while
                 still being capable of restating a v2 row's content. Directional
                 containment against the existing corpus is the check that
                 actually applies.

  length         the mixture drops rows over 17,408 tokens because truncation
                 lands mid-answer; enforced here too so the slice is known-good
                 in isolation.

Exit status is non-zero when any row is contaminated, so this can gate a build.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import pathlib
import sys

CHARS_PER_TOKEN = 3.6


def _text(rec: dict) -> str:
    return "".join(
        str(m.get("content") or "")
        for m in (rec.get("messages") or [])
        if isinstance(m, dict)
    )


def _sig(rec: dict) -> str:
    """Exact-content signature, matching build_sft_v3_mixture._sig."""
    txt = _text(rec)
    return hashlib.md5((txt or json.dumps(rec, sort_keys=True)).encode()).hexdigest()


def _iter_corpus(paths: list[pathlib.Path]):
    """Yield rows from .jsonl, .jsonl.gz, or split .gz.partNN sets."""
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        else:  # a directory of gzip parts to be concatenated then decompressed
            continue


def _iter_gz_parts(parts: list[pathlib.Path]):
    """Decompress a concatenated set of ``*.gz.partNN`` shards, streaming.

    The released corpus is stored split, and reassembling it to disk costs
    hundreds of MB we do not need: we only ever read it once.
    """
    if not parts:
        return
    import io

    class _Chain(io.RawIOBase):
        def __init__(self, files):
            self._files = list(files)
            self._fh = None

        def readable(self):
            return True

        def readinto(self, b):
            while True:
                if self._fh is None:
                    if not self._files:
                        return 0
                    self._fh = self._files.pop(0).open("rb")
                n = self._fh.readinto(b)
                if n:
                    return n
                self._fh.close()
                self._fh = None

    with gzip.open(io.BufferedReader(_Chain(sorted(parts))), "rt",
                   errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hk-root", default="",
                    help="HipKittens checkout (default: $KORE_HIPKITTENS_ROOT or "
                         "~/third_party/HipKittens)")
    ap.add_argument("--out", default="data/b05factory/sft/hipkittens.jsonl")
    ap.add_argument("--report", default="data/b05factory/sft/hipkittens_report.json")
    ap.add_argument("--against", nargs="*", default=[
        "data/b05factory/sft/multicap_v2.jsonl",
    ], help="existing mixture file(s) to dedup against")
    ap.add_argument("--against-parts", nargs="*", default=[
        "data/release/sft/multicap.jsonl.gz.part*",
    ], help="glob(s) of split gzip corpus parts to dedup against")
    ap.add_argument("--max-seq-tokens", type=int, default=17408)
    ap.add_argument("--dedup-threshold", type=float, default=0.5,
                    help="directional containment above which a row is judged a "
                         "restatement of an existing corpus row")
    ap.add_argument("--dedup-sample", type=int, default=0,
                    help="cap corpus rows compared (0 = all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from kore.data.decontam import (
        analyze_text_contamination,
        build_heldout_ngrams,
        heldout_families,
        heldout_task_ids,
        record_family,
    )
    from kore.data.dedup import token_shingles
    from kore.data.hipkittens import build_rows

    print("== build ==")
    rows, stats = build_rows(args.hk_root or None)
    print(f"generated {stats['rows']} rows "
          f"(~{stats['est_tokens']:,} tokens) from commit {stats['commit'][:12]} "
          f"[{stats['license']}]")
    for k, v in sorted(stats["per_qa_type"].items()):
        print(f"    {v:>4}  {k}")
    print(f"  dropped near-duplicate within slice: {stats['dropped_near_duplicate']}")
    print(f"  swizzles verified conflict-free: "
          f"{stats['swizzles_verified_conflict_free']}/{len(stats['conflict_degrees'])}")

    # ---------------- gate 1: contamination ---------------- #
    print("\n== contamination ==")
    ids = {t for t in heldout_task_ids() if t}
    fams = {f for f in heldout_families() if f}
    print(f"held-out task ids: {len(ids)}   families: {len(fams)}")
    id_hits: collections.Counter = collections.Counter()
    fam_hits: collections.Counter = collections.Counter()
    ngram_hits: list[dict] = []
    try:
        index = build_heldout_ngrams()
    except Exception as exc:  # noqa: BLE001
        print(f"  [ngrams] index unavailable ({type(exc).__name__}: {exc})")
        index = None

    clean: list[dict] = []
    for row in rows:
        txt = _text(row)
        bad = False
        for t in ids:
            if t in txt:
                id_hits[t] += 1
                bad = True
        try:
            fam = record_family(row)
        except Exception:  # noqa: BLE001
            fam = ""
        if fam and fam in fams:
            fam_hits[fam] += 1
            bad = True
        if index is not None and getattr(index, "references", ()):
            match = analyze_text_contamination(txt, index)
            if match is not None:
                ngram_hits.append({
                    "qa_type": row.get("_qa_type"), "reason": match.reason,
                    "score": round(match.score, 4),
                    "reference_id": match.reference_id,
                })
                bad = True
        if not bad:
            clean.append(row)

    print(f"  [ids]      rows containing a held-out task id : {sum(id_hits.values())}")
    for k, v in id_hits.most_common(5):
        print(f"               {v:>4}  {k}")
    print(f"  [families] rows attributed to a held-out family: {sum(fam_hits.values())}")
    for k, v in fam_hits.most_common(5):
        print(f"               {v:>4}  {k}")
    if index is not None:
        print(f"  [ngrams]   references indexed: {len(getattr(index, 'references', ()))}"
              f"  overlapping rows: {len(ngram_hits)}")
        for h in ngram_hits[:3]:
            print(f"               e.g. {h}")
    contaminated = sum(id_hits.values()) + sum(fam_hits.values()) + len(ngram_hits)
    print(f"  VERDICT: {'CLEAN' if contaminated == 0 else 'CONTAMINATED'}")

    # ---------------- gate 2: length ---------------- #
    kept: list[dict] = []
    too_long = 0
    for row in clean:
        if len(_text(row)) / CHARS_PER_TOKEN > args.max_seq_tokens:
            too_long += 1
            continue
        kept.append(row)
    if too_long:
        print(f"\n== length ==\n  dropped over {args.max_seq_tokens} tokens: {too_long}")

    # ---------------- gate 3: dedup vs the existing mixture ---------------- #
    print("\n== dedup against the existing mixture ==")
    corpus_paths = [repo / p for p in args.against]
    part_paths: list[pathlib.Path] = []
    for pattern in args.against_parts:
        part_paths.extend(sorted((repo).glob(pattern)))
    present = [p for p in corpus_paths if p.exists()]
    missing = [p for p in corpus_paths if not p.exists()]
    for p in present:
        print(f"  corpus: {p.relative_to(repo)}")
    if part_paths:
        print(f"  corpus: {len(part_paths)} gzip part(s) "
              f"({part_paths[0].name.rsplit('.', 1)[0]}*)")
    # Report what we were asked to compare against but could not find, rather
    # than quietly comparing against less and calling the slice deduplicated.
    for p in missing:
        print(f"  MISSING (not compared): {p.relative_to(repo)}")
    if not present and not part_paths:
        print("  WARNING: no existing corpus found; cross-corpus dedup SKIPPED. "
              "The slice may restate rows already in the mixture.")

    row_shingles = [token_shingles(_text(r), 8) for r in kept]
    exact = {_sig(r) for r in kept}
    dup_exact = 0
    dup_near: list[dict] = []
    near_flag = [0.0] * len(kept)
    n_corpus = 0
    for rec in list(_iter_corpus(present)) + list(_iter_gz_parts(part_paths)):
        n_corpus += 1
        if args.dedup_sample and n_corpus > args.dedup_sample:
            break
        ctxt = _text(rec)
        if _sig(rec) in exact:
            dup_exact += 1
        csh = token_shingles(ctxt, 8)
        if not csh:
            continue
        for i, sh in enumerate(row_shingles):
            if not sh:
                continue
            inter = len(sh & csh)
            if not inter:
                continue
            # Denominator is OUR row: "how much of this new row already exists".
            score = inter / len(sh)
            if score > near_flag[i]:
                near_flag[i] = score
    print(f"  corpus rows compared: {n_corpus:,}")
    print(f"  exact content collisions: {dup_exact}")
    final: list[dict] = []
    for i, row in enumerate(kept):
        if near_flag[i] >= args.dedup_threshold:
            dup_near.append({"qa_type": row.get("_qa_type"),
                             "containment": round(near_flag[i], 3)})
            continue
        final.append(row)
    worst = max(near_flag) if near_flag else 0.0
    print(f"  max containment of any new row in the existing corpus: {worst:.3f}")
    print(f"  dropped as restatements (>= {args.dedup_threshold}): {len(dup_near)}")
    for d in dup_near[:5]:
        print(f"      {d}")

    # ---------------- report ---------------- #
    chars = sum(len(_text(r)) for r in final)
    report = {
        "hipkittens_commit": stats["commit"],
        "license": stats["license"],
        "generated_rows": stats["rows"],
        "per_qa_type": stats["per_qa_type"],
        "dropped_near_duplicate_within_slice": stats["dropped_near_duplicate"],
        "swizzles_verified_conflict_free": stats["swizzles_verified_conflict_free"],
        "conflict_degrees": stats["conflict_degrees"],
        "contamination": {
            "id_hits": dict(id_hits),
            "family_hits": dict(fam_hits),
            "ngram_hits": ngram_hits,
            "clean": contaminated == 0,
        },
        "dropped_too_long": too_long,
        "cross_corpus": {
            "compared": ([str(p.relative_to(repo)) for p in present]
                         + [str(p.relative_to(repo)) for p in part_paths]),
            "requested_but_missing": [str(p.relative_to(repo)) for p in missing],
            "corpus_rows_compared": n_corpus,
            "exact_collisions": dup_exact,
            "max_containment": round(worst, 4),
            "threshold": args.dedup_threshold,
            "dropped": dup_near,
        },
        "final_rows": len(final),
        "final_chars": chars,
        "final_est_tokens": int(chars / CHARS_PER_TOKEN),
    }

    print("\n== result ==")
    print(f"  final rows  : {len(final)}")
    print(f"  final tokens: ~{report['final_est_tokens']:,} "
          f"(at {CHARS_PER_TOKEN} chars/token)")
    print(f"  mean tokens/row: {report['final_est_tokens'] // max(1, len(final)):,}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0 if contaminated == 0 else 1

    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as w:
        for row in final:
            w.write(json.dumps(row) + "\n")
    rep = repo / args.report
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {len(final)} rows to {args.out} ({out.stat().st_size/1e3:.0f} kB)")
    print(f"wrote report to {args.report}")
    return 0 if contaminated == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
