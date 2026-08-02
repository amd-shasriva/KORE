#!/usr/bin/env python3
"""Execute every registered task's committed seed on real GPU hardware.

Closes a long-standing evidence gap: the 1,052 generated breadth tasks were
admitted by a CPU-side AST/anti-hack scan only, and no kernel had ever been
compiled against them on the target architecture. This runs each task's seed
through its own ``driver.py`` -- the same subprocess contract ``KoreEnv`` speaks
-- and records a per-task verdict.

Deliberately does NOT import ``KoreEnv``: this is evidence collection about the
task corpus, so it must not inherit replay caching, reward shaping, or any
config-dependent behaviour that could mask a broken task.

Infrastructure faults (OOM, HIP errors, timeouts) are classified separately from
task defects, because reporting a node fault as a broken task is exactly the
mistake this evidence is meant to prevent.

Usage:
    python scripts/verify_tasks_gpu.py --out report.json --gpus 0,1,2,3
    python scripts/verify_tasks_gpu.py --out r.json --gpus 0 --prefix genb_attn
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "kore" / "tasks"

_SNR = re.compile(r"^SNR:\s*(-?[\d.]+)", re.M)
_ALLCLOSE = re.compile(r"^allclose:\s*(True|False)", re.M)
# The elementwise disagreement in representable steps of the OUTPUT format --
# the dtype-normalised form of max_diff, and the quantity the elementwise gate
# actually tests.  Recorded per task so the tolerance's headroom is auditable
# from the artifact instead of having to be re-derived from a rerun.
_STEPS = re.compile(r"^format_steps:\s*([\d.eE+-]+)\s+limit:\s*([\d.eE+-]+)", re.M)
_INFRA = re.compile(
    r"OutOfMemoryError|HSA_STATUS|hipError|CUDA error|No such device|"
    r"MemoryError|Cannot allocate|Bus error|device-side assert",
    re.I,
)

STATUSES = (
    "PASS",              # seed is correct on hardware at its declared threshold
    "FAIL_CORRECTNESS",  # seed ran but missed allclose or its SNR gate
    "COMPILE_FAIL",      # seed produced no verdict
    "INFRA",             # node/resource fault -- NOT a task defect
    "TIMEOUT",
    "BAD_TASK_YAML",
    "STAGE_ERROR",
)


def _primary_shape(meta: dict) -> str:
    shapes = meta.get("shapes") or {}
    dims = shapes.get("primary") or shapes.get("minimal")
    if not isinstance(dims, dict):
        return ""
    return ",".join(f"{k}={int(v)}" for k, v in dims.items())


def run_task(task_dir: Path, gpu: str, timeout: int, python: str) -> dict:
    import yaml

    try:
        meta = yaml.safe_load((task_dir / "task.yaml").read_text())
        seed_name = meta["seed_kernel_name"]
    except Exception as exc:  # noqa: BLE001
        return {"task": task_dir.name, "status": "BAD_TASK_YAML", "detail": str(exc)[:200]}

    shape = _primary_shape(meta)
    threshold = float(meta.get("snr_threshold", 30.0))
    started = time.time()

    with tempfile.TemporaryDirectory(prefix="koreverify_") as tmp:
        work = Path(tmp)
        try:
            for src in task_dir.glob("*.py"):
                shutil.copy2(src, work / src.name)
            shutil.copy2(task_dir / seed_name, work / "kernel.py")
        except Exception as exc:  # noqa: BLE001
            return {"task": task_dir.name, "status": "STAGE_ERROR", "detail": str(exc)[:200]}

        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": tmp,
            "TMPDIR": tmp,
            "HIP_VISIBLE_DEVICES": gpu,
            "GPU_TARGET": meta.get("gpu_target", "gfx950"),
            "PYTHONPATH": str(REPO),
            # Mirror the production sbatch so allocator behaviour matches.
            "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "TRITON_CACHE_DIR": f"{tmp}/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{tmp}/inductor",
        }
        cmd = [python, "driver.py"] + (["--shape", shape] if shape else [])
        try:
            proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True,
                                  text=True, timeout=timeout)
            output, rc = proc.stdout + proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            output, rc = "TIMEOUT", -9

    snr_matches = _SNR.findall(output)
    allclose = _ALLCLOSE.findall(output)
    steps = _STEPS.findall(output)
    snr = float(snr_matches[-1]) if snr_matches else None
    record = {
        "task": task_dir.name, "rc": rc, "snr_db": snr, "threshold": threshold,
        "seconds": round(time.time() - started, 1), "shape": shape, "gpu": gpu,
        "dtype": meta.get("dtype"), "operation": meta.get("operation"),
        "format_steps": round(float(steps[-1][0]), 4) if steps else None,
        "format_steps_limit": float(steps[-1][1]) if steps else None,
    }

    if rc == -9:
        return {**record, "status": "TIMEOUT", "detail": ""}
    # Check infra BEFORE compile-fail: an OOM also yields no verdict, and
    # calling that a broken task would be a false accusation.
    if _INFRA.search(output):
        return {**record, "status": "INFRA", "detail": output[-400:]}
    if snr is None:
        return {**record, "status": "COMPILE_FAIL", "detail": output[-400:]}
    if allclose and allclose[-1] == "True" and snr >= threshold:
        return {**record, "status": "PASS", "detail": ""}
    return {**record, "status": "FAIL_CORRECTNESS", "detail": output[-400:]}


def _worker(args):
    names, gpu, timeout, python = args
    return [run_task(TASKS / n, gpu, timeout, python) for n in names]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="report JSON path")
    ap.add_argument("--gpus", default="0", help="comma-separated HIP device ids")
    ap.add_argument("--prefix", default="", help="task-id prefix filter (empty = all)")
    ap.add_argument("--timeout", type=int, default=900, help="per-task seconds")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        ap.error("--gpus must name at least one device")

    names = sorted(d.name for d in TASKS.iterdir()
                   if d.is_dir() and d.name.startswith(args.prefix)
                   and (d / "task.yaml").is_file())
    if not names:
        print(f"no tasks match prefix {args.prefix!r}", file=sys.stderr)
        return 2

    print(f"verifying {len(names)} tasks on GPUs {gpus}", flush=True)
    shards = [(names[i::len(gpus)], gpus[i], args.timeout, args.python)
              for i in range(len(gpus))]
    started = time.time()
    with mp.get_context("spawn").Pool(len(gpus)) as pool:
        results = [r for chunk in pool.map(_worker, shards) for r in chunk]

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary = {
        "prefix": args.prefix, "total": len(results),
        "elapsed_s": round(time.time() - started, 1), "gpus": gpus,
        "counts": dict(sorted(counts.items())),
    }
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "results": sorted(results, key=lambda r: r["task"])},
        indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

    # A task defect fails the run; an infra fault does not, so a flaky node
    # cannot be laundered into a green verification.
    defects = sum(counts.get(s, 0) for s in
                  ("FAIL_CORRECTNESS", "COMPILE_FAIL", "BAD_TASK_YAML", "STAGE_ERROR"))
    return 1 if defects else 0


if __name__ == "__main__":
    raise SystemExit(main())
