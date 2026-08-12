#!/usr/bin/env python
"""Build the frozen pool-task -> AgentKernelArena contamination table.

Run once and commit the artifact. Every v5 stage screens against it, so rebuilding
is an explicit act with a reviewable diff rather than a silent change in what got
excluded from training.

    python scripts/v5_build_arena_index.py --out data/arena_contamination.json

The table scores every pool task against every arena task that ships a parseable
PyTorch source, and records each pool task's single best match. Screening then
applies a threshold at read time, so lowering it later does not require a rebuild
and the near-misses stay visible for audit.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

REPO = Path("/home/shasriva/Kore-RL/KORE")
sys.path.insert(0, str(REPO))

ARENA = Path("/home/shasriva/third_party/AgentKernelArena")
POOL = REPO / "data" / "task_pool"

#: Arena PyTorch sources sit under a few names depending on the suite.
ARENA_SOURCE_GLOBS = ("pytorch_code_module/*.py", "model.py", "*_model.py",
                      "reference.py", "original.py")


def arena_documents() -> tuple[dict[str, str], list[str]]:
    """Every arena task id -> its PyTorch source text, plus the tasks lacking one.

    ``source_file_path`` is a YAML *list*, so it has to be parsed rather than
    regexed; reading it as a scalar silently finds only a third of the corpus and
    leaves the rest unscreened, which is worse than not screening at all because
    it looks like coverage.
    """
    import yaml

    out: dict[str, str] = {}
    missing: list[str] = []
    tasks_dir = ARENA / "tasks"
    if not tasks_dir.is_dir():
        return out, missing
    for cfg in sorted(tasks_dir.rglob("config.yaml")):
        td = cfg.parent
        tid = str(td.relative_to(tasks_dir))
        try:
            raw = yaml.safe_load(cfg.read_text(errors="ignore")) or {}
        except Exception:  # noqa: BLE001 - a malformed config is not a source
            raw = {}
        declared = raw.get("source_file_path")
        if isinstance(declared, str):
            declared = [declared]
        text = ""
        for rel in (declared or []):
            cand = td / str(rel)
            if cand.is_file() and cand.suffix == ".py":
                text = cand.read_text(errors="ignore")
                break
        if not text:
            for pat in ARENA_SOURCE_GLOBS:
                hits = sorted(td.glob(pat))
                if hits:
                    text = hits[0].read_text(errors="ignore")
                    break
        if text.strip():
            out[tid] = text
        else:
            missing.append(tid)
    return out, missing


def pool_documents() -> dict[str, str]:
    """Every pool task id -> the PyTorch module that defines its semantics."""
    out: dict[str, str] = {}
    manifest = POOL / "pool.jsonl"
    if manifest.is_file():
        with manifest.open(errors="ignore") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001 - torn line
                    continue
                tid = str(d.get("task_id") or d.get("id") or "")
                src = d.get("module_source") or d.get("pytorch_source") or ""
                if tid and isinstance(src, str) and src.strip():
                    out[tid] = src
    # Fall back to reading task directories for anything the manifest missed.
    tasks_dir = POOL / "tasks"
    if tasks_dir.is_dir():
        for td in sorted(tasks_dir.iterdir()):
            if td.name in out or not td.is_dir():
                continue
            ref = td / "reference.py"
            if ref.is_file():
                text = ref.read_text(errors="ignore")
                if text.strip():
                    out[td.name] = text
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data/arena_contamination.json"))
    ap.add_argument("--report-top", type=int, default=25)
    args = ap.parse_args()

    from kore.data.arena_index import (DEFAULT_THRESHOLD, build_document_frequency,
                                       exact_hash, jaccard, normalize, shingles)

    t0 = time.time()
    arena_raw, arena_missing = arena_documents()
    pool_raw = pool_documents()
    print(f"arena sources: {len(arena_raw):,}   pool sources: {len(pool_raw):,}",
          flush=True)
    # Tasks with no PyTorch source cannot be screened this way at all. Naming them
    # is the honest alternative to letting them look covered.
    unscreened: collections.Counter = collections.Counter(
        t.split("/")[0] for t in arena_missing)
    print(f"arena tasks with NO python source (unscreened): {len(arena_missing)} "
          f"{dict(unscreened)}", flush=True)

    arena_norm, arena_sh, arena_exact = {}, {}, {}
    for tid, src in arena_raw.items():
        n = normalize(src)
        if not n:
            continue
        arena_norm[tid] = n
        arena_sh[tid] = shingles(n)
        arena_exact.setdefault(exact_hash(n), []).append(tid)
    print(f"arena parseable: {len(arena_norm):,} "
          f"({len(arena_raw) - len(arena_norm):,} unparseable/non-Python)", flush=True)

    pool_norm, pool_sh = {}, {}
    for tid, src in pool_raw.items():
        n = normalize(src)
        if not n:
            continue
        pool_norm[tid] = n
        pool_sh[tid] = shingles(n)
    print(f"pool parseable : {len(pool_norm):,}", flush=True)

    # Structural boilerplate is measured on the pool, which is large enough for the
    # document frequency to be meaningful; the arena is far too small to estimate it.
    common = build_document_frequency(pool_sh.values())
    print(f"boilerplate shingles dropped: {len(common):,}", flush=True)
    for d in (arena_sh, pool_sh):
        for k in list(d):
            d[k] = d[k] - common

    # Inverted index over arena shingles so each pool task only scores against
    # arena tasks it shares rare content with, rather than all of them.
    inv: dict[str, list[str]] = collections.defaultdict(list)
    for tid, sh in arena_sh.items():
        for s in sh:
            inv[s].append(tid)

    matches: dict[str, list] = {}
    n_exact = 0
    for tid, sh in pool_sh.items():
        cand: collections.Counter = collections.Counter()
        for s in sh:
            for a in inv.get(s, ()):
                cand[a] += 1
        best, best_score = None, 0.0
        for a, _ in cand.most_common(50):
            sc = jaccard(sh, arena_sh[a])
            if sc > best_score:
                best, best_score = a, sc
        # Exact identity overrides the shingle score, which the DF filter can
        # depress on a short module by removing most of its content.
        hit = arena_exact.get(exact_hash(pool_norm[tid]))
        if hit:
            best, best_score = hit[0], 1.0
            n_exact += 1
        if best is not None and best_score > 0.05:
            matches[tid] = [best, round(best_score, 4)]

    blocking = {k: v for k, v in matches.items() if v[1] >= DEFAULT_THRESHOLD}
    arena_hit = {v[0] for v in blocking.values()}
    table = {
        "meta": {
            "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "arena_root": str(ARENA),
            "arena_tasks_total": len(arena_raw),
            "arena_tasks_scored": len(arena_norm),
            "pool_tasks_scored": len(pool_norm),
            "boilerplate_shingles": len(common),
            "threshold": DEFAULT_THRESHOLD,
            "exact_matches": n_exact,
            "pool_tasks_blocked": len(blocking),
            "arena_tasks_implicated": len(arena_hit),
            "arena_tasks_unscreened": len(arena_missing),
            "arena_unscreened_by_category": dict(unscreened),
        },
        "matches": matches,
        "unscreened": sorted(arena_missing),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1, sort_keys=True))

    print(f"\n=== arena contamination index ===")
    for k, v in table["meta"].items():
        print(f"  {k:<26} {v}")
    print(f"\ntop {args.report_top} matches:")
    for tid, (a, sc) in sorted(matches.items(), key=lambda kv: -kv[1][1])[:args.report_top]:
        print(f"  {sc:5.3f}  {a:<52} {tid}")
    by_cat: collections.Counter = collections.Counter(
        a.split("/")[0] + "/" + (a.split("/")[1] if "/" in a[len(a.split('/')[0]) + 1:] else "")
        for a in arena_hit)
    print("\nimplicated arena tasks by category:")
    for k, v in by_cat.most_common():
        print(f"  {k:<38} {v}")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
