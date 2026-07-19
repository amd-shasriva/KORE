#!/usr/bin/env python3
"""On-gfx950 verification for the ``genb_*`` breadth tasks (genops ABI).

For each ``kore/tasks/genb_*/`` task it runs the same two gates the datagen relies
on, in a clean subprocess (so the 66 reference modules never collide in
``sys.modules``):

  1. SEED COMPILES + CORRECT: copy ``seed_triton.py`` -> ``kernel.py`` and run
     ``driver.py`` (the shared ``_genops.driver_main`` correctness path: the seed
     candidate vs the fp32 ``ref_fn`` oracle). Parse ``SNR`` (>= the task gate),
     ``allclose``, and any ``CANDIDATE_ERROR`` (a compile / runtime failure).
  2. BASELINE RUNS: ``driver.py --bench-mode --impl reference`` -- the torch
     ``baseline_fn`` (the perf bar a fused kernel must beat) must run + print
     ``median_ms``.

Non-destructive: every ``kernel.py`` it creates is removed. Writes
``runs/breadth_verify_report.json``. Run from the repo root on a FREE gpu (not the
factory's 3,4,6, not a container's), with the KORE venv:

    HIP_VISIBLE_DEVICES=7 KORE_PY=~/kore-venv/bin/python \
        ~/kore-venv/bin/python scripts/verify_breadth.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

REPO = os.getcwd()
PY = os.environ.get("KORE_PY", sys.executable)
PER_CALL_TIMEOUT_S = int(os.environ.get("KORE_VERIFY_TIMEOUT", "900"))


def _run(cmd, timeout=PER_CALL_TIMEOUT_S):
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", ".")
    # honest adversarial-input correctness (matches the factory datagen env)
    env.setdefault("KORE_VERIFIED_CORRECTNESS", "1")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=REPO, env=env)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -9, "", f"TIMEOUT after {timeout}s"


def _last_err(err):
    lines = [ln for ln in (err or "").strip().splitlines() if ln.strip()]
    return lines[-1][:220] if lines else None


def _task_meta(tdir: str) -> tuple[float, str]:
    """Read the task's snr_threshold + the 'minimal' shape spec string.

    The generic ``_genops`` driver defaults ``--shape`` to a hardcoded ``{M,N}``
    (``_parse_shape('default')``); the factory/env ALWAYS passes an explicit
    ``--shape`` built from the task's shapes (``Shape.as_args()``), so a faithful
    verify must do the same or non-``{M,N}`` ops raise ``KeyError`` in get_inputs.
    We use the ``minimal`` shape: it exercises compile+correctness cheaply (and keeps
    the O(N^2) sort/top-p starter seeds fast)."""
    gate, shape_str = 25.0, ""
    try:
        import yaml
        meta = yaml.safe_load(open(os.path.join(tdir, "task.yaml")).read())
        gate = float(meta.get("snr_threshold", meta.get("targets", {}).get("snr_db", 25.0)))
        shp = (meta.get("shapes", {}) or {}).get("minimal", {}) or {}
        shape_str = ",".join(f"{k}={v}" for k, v in shp.items())
    except Exception:  # noqa: BLE001
        pass
    return gate, shape_str


def verify(tdir: str) -> dict:
    tdir = os.path.normpath(tdir)
    tid = os.path.basename(tdir)
    res = {"task": tid}
    seed = os.path.join(tdir, "seed_triton.py")
    drv = os.path.join(tdir, "driver.py")
    kern = os.path.join(tdir, "kernel.py")
    if not (os.path.exists(seed) and os.path.exists(drv)):
        res["status"] = "NO_SEED_OR_DRIVER"
        return res
    gate, shape_str = _task_meta(tdir)
    res["snr_gate"] = gate
    res["shape"] = shape_str or "default"
    shape_args = ["--shape", shape_str] if shape_str else []
    shutil.copy(seed, kern)
    try:
        rc, out, err = _run([PY, drv, *shape_args])
        m = re.search(r"SNR:\s*([-\d.]+)", out)
        res["snr"] = float(m.group(1)) if m else None
        m = re.search(r"allclose:\s*(True|False)", out)
        res["allclose"] = (m.group(1) == "True") if m else None
        m = re.search(r"CANDIDATE_ERROR:\s*(.*)", out)
        res["cand_error"] = m.group(1).strip()[:220] if m else None
        if res["snr"] is None and not res["cand_error"]:
            res["cand_error"] = _last_err(err) or "no SNR printed (driver failure)"

        rc2, out2, err2 = _run([PY, drv, "--bench-mode", "--impl", "reference", *shape_args])
        res["baseline_runs"] = (rc2 == 0 and "median_ms" in out2)
        if not res["baseline_runs"]:
            res["baseline_error"] = _last_err(err2) or _last_err(out2)

        corr_ok = bool(res["allclose"]) and res["snr"] is not None and res["snr"] >= gate
        res["status"] = "PASS" if corr_ok else "FAIL"
    finally:
        if os.path.exists(kern):
            os.remove(kern)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="kore/tasks/genb_*/",
                    help="task dir glob (default all genb_*)")
    ap.add_argument("--out", default=os.path.join(REPO, "runs/breadth_verify_report.json"))
    ap.add_argument("--nshards", type=int, default=1,
                    help="split the task list into N shards (parallel verify across GPUs)")
    ap.add_argument("--shard", type=int, default=0, help="which shard [0, nshards)")
    a = ap.parse_args()

    tdirs = sorted(glob.glob(os.path.join(REPO, a.glob)))
    if a.nshards > 1:
        tdirs = tdirs[a.shard::a.nshards]
    results = []
    for tdir in tdirs:
        if "__pycache__" in tdir:
            continue
        if not os.path.exists(os.path.join(tdir, "task.yaml")):
            continue
        tid = os.path.basename(os.path.normpath(tdir))
        print(f"[verify] {tid} ...", flush=True)
        r = verify(tdir)
        results.append(r)
        extra = ""
        if r.get("cand_error"):
            extra += f"  candErr={r['cand_error']}"
        if r.get("baseline_error"):
            extra += f"  baseErr={r['baseline_error']}"
        print(f"    -> {r['status']} snr={r.get('snr')} gate={r.get('snr_gate')} "
              f"allclose={r.get('allclose')} baseline_runs={r.get('baseline_runs')}{extra}",
              flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)

    npass = sum(1 for r in results if r["status"] == "PASS")
    print("\n==================== SUMMARY ====================")
    for r in results:
        print(f"  {r['status']:5s} {r['task']}")
    print(f"\n  {npass}/{len(results)} PASS")
    print(f"  status breakdown: {dict(Counter(r['status'] for r in results))}")
    print(f"  report: {a.out}")


if __name__ == "__main__":
    main()
