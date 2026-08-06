#!/usr/bin/env python
"""Gate teacher-written HIP seeds on real gfx950 before any datagen runs on them.

``verify_hip_seeds.py`` gates the 65 hand-authored ops by reading
``hip_ops.HIP_OPS``, so it cannot see a materialized pool task. These seeds need
the same gate for the same reason: a seed that does not compile, or cannot clear
the task's own SNR threshold, is worse than no task at all. Datagen against it
can only ever score zero, and it reports as a model error rather than as the
broken task it is -- which is how a task-resolution bug once produced a 97% error
rate while claiming 13,277 episodes/hour.

Verification is deliberately the same path the environment uses: stage the seed
as the candidate artifact the backend declares (``kernel.hip``) and run the task's
own driver with ``--impl candidate``. Anything else would be testing a different
thing from what datagen will run.

    python scripts/verify_pool_hip_seeds.py --root data/pool_hip --json gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def verify_one(task_dir: Path, timeout: int, gpu: int | None = None) -> dict:
    """Compile and check one seed, returning a verdict row."""
    cfg = json.loads((task_dir / "task.yaml").read_text())
    rec = {"task_id": cfg.get("task_id"), "hip_twin_of": cfg.get("hip_twin_of"),
           "family": cfg.get("op_family")}
    seed = task_dir / cfg.get("seed_kernel_name", "seed_hip.hip")
    if not seed.is_file():
        return {**rec, "status": "no_seed"}

    # The driver reads the candidate under the backend's declared filename, so
    # stage it there rather than pointing the driver at the seed: datagen will
    # present candidates exactly this way.
    from kore.env.hip_toolchain import CANDIDATE_FILENAMES, HIP_BACKEND

    shutil.copy(seed, task_dir / CANDIDATE_FILENAMES[HIP_BACKEND])

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}{os.pathsep}{env.get('PYTHONPATH', '')}"
    if gpu is not None:
        # One device per worker. Without this every worker builds and runs on
        # device 0, which serializes the part that needs a GPU at all and leaves
        # seven idle.
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # Give each worker its own MIOpen databases. MIOpen keeps a SQLite
        # performance-db and a compiled-kernel cache, and eight workers sharing
        # them raise miopenStatusInternalError from inside the *reference*
        # convolution -- so a perfectly good seed is recorded as a broken task.
        # It accounted for 2,875 of 2,892 failures here, ~44% of the
        # functionalized set, and it grows with the run as contention builds,
        # which is why an early sample looked healthy at 87% yield.
        mio = Path(env.get("TMPDIR", "/tmp")) / f"miopen_w{gpu}"
        (mio / "db").mkdir(parents=True, exist_ok=True)
        (mio / "cache").mkdir(parents=True, exist_ok=True)
        env["MIOPEN_USER_DB_PATH"] = str(mio / "db")
        env["MIOPEN_CUSTOM_CACHE_DIR"] = str(mio / "cache")
        env["MIOPEN_DISABLE_CACHE"] = "0"

    # A private compile cache per seed, discarded once the verdict is in. Every
    # seed is a different source, so the cache can never hit -- retaining it only
    # accumulates. Left shared, eight workers compiling thousands of extensions
    # filled the node's 123 GB /tmp and the run ended with 3,534 seeds recorded as
    # "No space left on device", which is indistinguishable in the ledger from a
    # kernel that genuinely does not build.
    cache = Path(env.get("TMPDIR", "/tmp")) / "kore_gate_cache" / rec["task_id"]
    cache.mkdir(parents=True, exist_ok=True)
    env["KORE_COMPILE_CACHE_DIR"] = str(cache)
    env["TORCH_EXTENSIONS_DIR"] = str(cache)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "driver.py", "--impl", "candidate"],
            cwd=str(task_dir), capture_output=True, text=True,
            timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {**rec, "status": "timeout", "seconds": timeout}
    finally:
        # Reclaim the build whatever the outcome, including on timeout: a seed
        # that hangs is exactly the one leaving the largest partial build behind.
        shutil.rmtree(cache, ignore_errors=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    rec["seconds"] = round(time.time() - t0, 1)

    # The driver prints the verdict; trust it rather than re-deriving one here,
    # since it is the same authority datagen and the reward path use.
    #
    # It prints a Python bool -- `allclose: True` -- not the word "pass". Matching
    # on "allclose: pass" can never fire, so every seed was reported incorrect
    # whatever it actually did, and a batch reporting SNR 999 (a perfect match)
    # read as 0% yield. The gate was rejecting good seeds.
    low = out.lower()
    if "allclose: true" in low or "verdict: pass" in low:
        rec["status"] = "pass"
    else:
        rec["status"] = ("compile_or_run_fail" if proc.returncode != 0
                         else "incorrect")
        # Keep the compiler's own diagnostics, not just the tail. hipcc prints
        # the error first and ninja's "build stopped" last, so a short tail
        # captures only the fact that something failed and never what.
        errs = [ln for ln in out.splitlines()
                if "error:" in ln.lower() or "warning: " in ln.lower()]
        rec["diagnostics"] = errs[:12]
        rec["error"] = out.strip()[-2000:]
    for key in ("snr_db", "snr"):
        idx = low.rfind(key + ":")
        if idx >= 0:
            try:
                rec["snr_db"] = float(low[idx + len(key) + 1:].split()[0])
            except Exception:  # noqa: BLE001 - a missing number is not a failure
                pass
            break
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pool_hip")
    ap.add_argument("--json", default="")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8,
                    help="seeds gated at once, one GPU each. Gating is a compile "
                         "plus a short run, so serial gating is what starves "
                         "datagen of tasks")
    ap.add_argument("--resume", action="store_true",
                    help="skip seeds already decided in the output report")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    # "*__hip*" so functionalized twins (__hipf) are gated on the same path as
    # parameter-free ones. They differ only in how many tensors the entry takes,
    # which is the driver's business, not the gate's.
    dirs = sorted((Path(args.root) / "tasks").glob("*__hip*"))

    # Carry forward verdicts from an earlier run of this root. Gating thousands of
    # seeds outlives one job's time limit, and re-deciding a seed the last job
    # already decided would mean the sweep never finishes.
    rows: list[dict] = []
    if args.resume and args.json and Path(args.json).is_file():
        try:
            rows = json.loads(Path(args.json).read_text()).get("rows", [])
        except Exception:  # noqa: BLE001 - a truncated report just means redo it
            rows = []
        done = {r.get("task_id") for r in rows}
        before = len(dirs)
        dirs = [d for d in dirs if d.name not in done]
        print(f"resuming: {before - len(dirs)} already decided, {len(dirs)} to go")
    if args.limit:
        dirs = dirs[: args.limit]
    print(f"gating {len(dirs)} seed(s) on {os.uname().nodename} "
          f"with {args.workers} worker(s)")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1

    def _emit(i: int, r: dict) -> None:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(f"  [{i}/{len(dirs)}] {r['task_id']}: {r['status']}"
              + (f" snr={r['snr_db']}" if r.get("snr_db") is not None else ""),
              flush=True)

    def _checkpoint() -> None:
        """Write the report as we go. A gate job that is preempted at 90% must not
        throw away the verdicts it already has."""
        if args.json:
            tmp = Path(args.json).with_suffix(".tmp")
            tmp.write_text(json.dumps({"counts": counts, "rows": rows}, indent=2))
            tmp.replace(args.json)

    if args.workers > 1 and dirs:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n_gpu = max(1, args.workers)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(verify_one, d, args.timeout, i % n_gpu): d
                    for i, d in enumerate(dirs)}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                rows.append(r)
                _emit(i, r)
                if i % 20 == 0:
                    _checkpoint()
    else:
        for i, d in enumerate(dirs, 1):
            r = verify_one(d, args.timeout, 0)
            rows.append(r)
            _emit(i, r)
            if i % 20 == 0:
                _checkpoint()

    n = len(rows) or 1
    print(f"\nyield: {counts.get('pass', 0)}/{len(rows)} "
          f"({100 * counts.get('pass', 0) / n:.0f}%) usable as tasks")
    print(f"breakdown: {counts}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"counts": counts, "rows": rows}, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
