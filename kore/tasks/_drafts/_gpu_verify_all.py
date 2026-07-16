#!/usr/bin/env python3
"""On-gfx950 verification runner for EVERY staged draft task.

For each task under ``kore/tasks/_drafts/<family>/<id>/`` this runs the promotion
gates from that family's ``VERIFICATION_CHECKLIST.md`` and records the outcome:

  1. SEED COMPILES + CORRECT: ``driver.py`` correctness (the seed candidate vs the
     fp32 oracle in ``reference.py``); we parse ``SNR`` (must be >= 25 dB), the
     ``allclose`` flag, and any ``CANDIDATE_ERROR`` (a compile / runtime failure).
  2. VENDOR BASELINE RUNS: ``driver.py --bench-mode --impl reference`` -- the REAL
     AITER / hipBLASLt / framework baseline must run and print ``median_ms`` (i.e.
     the ``comparison_baseline`` symbol exists and accepts this layout/dtype).
  3. BASELINE == ORACLE (perf-bar integrity): import the task's ``reference`` and
     compare ``baseline_output`` vs ``reference_output`` (SNR). A mismatch (< 25 dB)
     means the vendor computes something different from the oracle -- the speed bar
     would be unfair; it is FLAGGED (not auto-failed) for a baseline fix per the
     checklist ("never weaken the oracle").

It writes ``kore/tasks/_drafts/_gpu_verify_report.json`` and prints a summary. It
does NOT promote anything and does NOT touch live tasks; each ``kernel.py`` it
creates is removed. Run from the repo root with the KORE venv AFTER the campaign is
stopped (it needs the GPUs):

    KORE_PY=~/kore-venv/bin/python ~/kore-venv/bin/python kore/tasks/_drafts/_gpu_verify_all.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys

# repo root = 4 levels up from kore/tasks/_drafts/_gpu_verify_all.py
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DRAFTS_REL = os.path.join("kore", "tasks", "_drafts")
# Robustness: if the layout ever changes, trust the launch cwd (always the repo root).
if not os.path.isdir(os.path.join(REPO, DRAFTS_REL)):
    REPO = os.getcwd()
FAMILIES = ("attention", "quant", "moe", "training")
PY = os.environ.get("KORE_PY", sys.executable)
SNR_GATE = 25.0
PER_CALL_TIMEOUT_S = int(os.environ.get("KORE_VERIFY_TIMEOUT", "1200"))

# Compare the vendor baseline against the fp32 oracle, in a clean subprocess so 31
# reference modules never collide in sys.modules. Mirrors driver.py's path setup.
_BASELINE_SNIPPET = r'''
import os, sys, math
tdir = sys.argv[1]
sys.path.insert(0, tdir)                       # reference.py
sys.path.insert(0, os.path.dirname(tdir))      # _<family>_common.py
import torch, reference as ref
shape = ref.parse_shape("default")
inp = ref.get_inputs(shape, device="cuda", seed=0)
base = ref.baseline_output(shape, inp)
orc = ref.reference_output(shape, inp)
def snr(o, r):
    o = o.float(); r = r.float()
    num = float((r * r).sum()); den = float(((o - r) ** 2).sum())
    if den == 0.0:
        return 999.0
    if num == 0.0:
        return -999.0
    return 10.0 * math.log10(num / den)
if isinstance(base, (tuple, list)):
    s = min(snr(b, r) for b, r in zip(base, orc))
else:
    s = snr(base, orc)
print("BASELINE_ORACLE_SNR: %.2f" % s)
'''


def _run(cmd, timeout=PER_CALL_TIMEOUT_S):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -9, "", "TIMEOUT after %ss" % timeout


def _last_err(err):
    lines = [ln for ln in (err or "").strip().splitlines() if ln.strip()]
    return lines[-1][:200] if lines else None


def verify(fam, tdir):
    tdir = os.path.normpath(tdir)
    tid = os.path.basename(tdir)
    res = {"family": fam, "task": tid}
    seed = os.path.join(tdir, "seed_triton.py")
    drv = os.path.join(tdir, "driver.py")
    kern = os.path.join(tdir, "kernel.py")
    if not (os.path.exists(seed) and os.path.exists(drv)):
        res["status"] = "NO_SEED_OR_DRIVER"
        return res
    shutil.copy(seed, kern)
    try:
        rc, out, err = _run([PY, drv])
        m = re.search(r"SNR:\s*([-\d.]+)", out)
        res["snr"] = float(m.group(1)) if m else None
        m = re.search(r"allclose:\s*(True|False)", out)
        res["allclose"] = (m.group(1) == "True") if m else None
        m = re.search(r"CANDIDATE_ERROR:\s*(.*)", out)
        res["cand_error"] = m.group(1).strip()[:200] if m else None
        if res["snr"] is None and not res["cand_error"]:
            res["cand_error"] = _last_err(err) or "no SNR printed (driver failure)"

        rc2, out2, err2 = _run([PY, drv, "--bench-mode", "--impl", "reference"])
        res["baseline_runs"] = (rc2 == 0 and "median_ms" in out2)
        if not res["baseline_runs"]:
            res["baseline_error"] = _last_err(err2) or _last_err(out2)

        rc3, out3, err3 = _run([PY, "-c", _BASELINE_SNIPPET, tdir])
        m = re.search(r"BASELINE_ORACLE_SNR:\s*([-\d.]+)", out3)
        res["baseline_oracle_snr"] = float(m.group(1)) if m else None
        if res["baseline_oracle_snr"] is None:
            res["baseline_oracle_error"] = _last_err(err3)

        corr_ok = bool(res["allclose"]) and res["snr"] is not None and res["snr"] >= SNR_GATE
        res["status"] = "PASS" if corr_ok else "FAIL"
        # perf-bar integrity is a flag, not a hard fail (checklist: fix baseline, keep oracle)
        bos = res.get("baseline_oracle_snr")
        res["baseline_mismatch"] = bool(res["baseline_runs"] and bos is not None and bos < SNR_GATE)
    finally:
        if os.path.exists(kern):
            os.remove(kern)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help="comma-separated subset of %s" % (FAMILIES,))
    ap.add_argument("--out", default=os.path.join(REPO, DRAFTS_REL, "_gpu_verify_report.json"))
    a = ap.parse_args()
    fams = [f.strip() for f in a.families.split(",") if f.strip()]

    results = []
    for fam in fams:
        for tdir in sorted(glob.glob(os.path.join(REPO, DRAFTS_REL, fam, "*", ""))):
            if "__pycache__" in tdir:
                continue
            if not os.path.exists(os.path.join(tdir, "task.yaml")):
                continue
            tid = os.path.basename(os.path.normpath(tdir))
            print(f"[verify] {fam}/{tid} ...", flush=True)
            r = verify(fam, tdir)
            results.append(r)
            extra = ""
            if r.get("cand_error"):
                extra += f"  candErr={r['cand_error']}"
            if r.get("baseline_error"):
                extra += f"  baseErr={r['baseline_error']}"
            print(f"    -> {r['status']}  snr={r.get('snr')} allclose={r.get('allclose')} "
                  f"baseline_runs={r.get('baseline_runs')} base==oracle_snr={r.get('baseline_oracle_snr')}"
                  f"{extra}", flush=True)

    out = a.out
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    npass = sum(1 for r in results if r["status"] == "PASS")
    nmis = sum(1 for r in results if r.get("baseline_mismatch"))
    print("\n==================== SUMMARY ====================")
    for r in results:
        flag = " [BASELINE!=ORACLE]" if r.get("baseline_mismatch") else ""
        print(f"  {r['status']:5s} {r['family']:9s} {r['task']}{flag}")
    print(f"\n  {npass}/{len(results)} PASS ; {nmis} baseline!=oracle flags")
    print(f"  report: {out}")


if __name__ == "__main__":
    main()
