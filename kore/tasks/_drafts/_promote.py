#!/usr/bin/env python3
"""Promote GPU-verified draft tasks into the live registry.

Reads the per-family ``_report_<fam>.json`` written by ``_gpu_verify_all.py`` and
promotes every task that is:
  (a) correctness PASS (seed compiles + candidate matches the fp32 oracle >= 25 dB), AND
  (b) has a WORKING vendor baseline (``baseline_runs``), AND
  (c) whose baseline matches the oracle (``base==oracle`` SNR >= 25 dB) OR is a
      documented perf-only-bar task (dense baseline intentionally != windowed oracle),
      OR has no numeric baseline-vs-oracle compare (grad-tuple tasks whose driver
      baseline still runs).

For each promoted task it copies the family's shared ``_<fam>_common.py`` into
``kore/tasks/`` once, then MOVES the task dir ``_drafts/<fam>/<id>/`` -> ``kore/tasks/<id>/``.
Held tasks stay staged. Prints the promote/hold breakdown. Never overwrites a live task.
"""
import glob
import json
import os
import shutil
import sys

REPO = os.getcwd()
if not os.path.isdir(os.path.join(REPO, "kore", "tasks", "_drafts")):
    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DRAFTS = os.path.join(REPO, "kore", "tasks", "_drafts")
LIVE = os.path.join(REPO, "kore", "tasks")
FAMILIES = ("attention", "quant", "moe", "training")
SNR_GATE = 25.0
# Tasks whose baseline is a DOCUMENTED perf-only bar (dense vendor kernel that
# intentionally computes something different from the specialized oracle).
EXPECTED_PERF_ONLY = {
    "flash_attn_sliding_decode_bf16",   # dense decode vs windowed oracle (by design)
    "flash_attn_sink_prefill_bf16",     # dense causal vs sink oracle (AITER sink_ptr != oracle)
}
REQUIRED_FILES = ("task.yaml", "reference.py", "seed_triton.py", "driver.py")


def promotable(r: dict) -> bool:
    if r.get("status") != "PASS":
        return False
    if not r.get("baseline_runs"):
        return False
    if r["task"] in EXPECTED_PERF_ONLY:
        return True
    bos = r.get("baseline_oracle_snr")
    if bos is None:            # driver baseline runs but no numeric compare (e.g. grad tuples)
        return True
    return bos >= SNR_GATE


def main():
    # Merge ALL reports; later files (e.g. _report_refix.json, alphabetically after the
    # per-family ones) override earlier records for the same task, so re-verified fixes win.
    merged = {}
    for rep in sorted(glob.glob(os.path.join(DRAFTS, "_report_*.json"))):
        try:
            for r in json.load(open(rep)):
                merged[r["task"]] = r
        except Exception as e:  # noqa: BLE001
            print(f"WARN: could not read {rep}: {e}")

    promoted, held = [], []
    commons_done = set()
    for tid, r in sorted(merged.items()):
        fam = r.get("family")
        src = os.path.join(DRAFTS, fam, tid) if fam else None
        dst = os.path.join(LIVE, tid)
        if os.path.exists(dst):
            continue  # already live (promoted earlier)
        if not src or not os.path.isdir(src):
            continue  # not staged here (already moved / stale record)
        if not promotable(r):
            held.append((fam, tid, r.get("status"), r.get("baseline_runs"),
                         r.get("baseline_oracle_snr")))
            continue
        if not all(os.path.exists(os.path.join(src, f)) for f in REQUIRED_FILES):
            held.append((fam, tid, "MISSING_FILES", None, None))
            continue
        common = f"_{fam}_common.py"
        csrc, cdst = os.path.join(DRAFTS, fam, common), os.path.join(LIVE, common)
        if fam not in commons_done and os.path.exists(csrc):
            shutil.copy2(csrc, cdst)
            commons_done.add(fam)
            print(f"copied {common} -> kore/tasks/")
        for junk in ("kernel.py",):
            p = os.path.join(src, junk)
            if os.path.exists(p):
                os.remove(p)
        pc = os.path.join(src, "__pycache__")
        if os.path.isdir(pc):
            shutil.rmtree(pc)
        shutil.move(src, dst)
        promoted.append((fam, tid))
        print(f"PROMOTED {fam}/{tid} -> kore/tasks/{tid}/")

    print("\n==================== PROMOTION SUMMARY ====================")
    print(f"PROMOTED {len(promoted)}:")
    for fam, tid in promoted:
        print(f"  + {fam:9s} {tid}")
    print(f"HELD {len(held)}:")
    for fam, tid, st, br, bos in held:
        print(f"  - {fam:9s} {tid}  (status={st} baseline_runs={br} base==oracle={bos})")


if __name__ == "__main__":
    main()
