"""P0 falsification test: is the roofline-residual paradigm physically real on this node?

Runs a 1-3 hour, three-question test BEFORE committing the node to a multi-day RL run:

  (a) Does SOL-attainment  eta = T_min / T_measured  predict speedup-vs-vendor?
      -> the roofline bound is a meaningful yardstick.               [needs timing]
  (b) Does the residual  (T_measured - T_min)  decompose cleanly into
      counter-derived stall / memory / overhead terms (regression R^2)?
      -> the "named gradient" is real, not drowned by cross-terms.   [needs PMC]
  (c) Along an improving trajectory of kernels, does the dominant residual term
      fall while wall-clock stays ~flat?
      -> a dense learning signal exists INSIDE the deceptive valley. [needs PMC + traj]

It prints a verdict box: GO / PARTIAL / FALLBACK / PIVOT.

  GO       (a,b,c pass)      -> build the residual-descent reward + generalization
                               harness; commit the node to the full run.
  PARTIAL  (a,b pass, c weak)-> bound sound, in-valley signal thin; add trajectory
                               kernels (--max-kernels-per-task) and re-check.
  FALLBACK (a pass, b noisy) -> use the bounded-eta framing; refine the decomposition.
  PIVOT    (a fails)         -> roofline bound doesn't predict speedup for these ops;
                               the pure deceptiveness-measurement paper is the play.
                               *** BEFORE accepting PIVOT, re-check the peak constants
                               (KORE_PEAK_*): a wrong peak is the #1 false-negative. ***

Usage:
    python -m kore.analysis.p0_sol --tasks rmsnorm_aiter,gemm_bf16 --dry-run
    python -m kore.analysis.p0_sol --tasks gemm_bf16,softmax_bf16,gelu_tanh_bf16 \
        --warmup 10 --iters 50 --out runs/p0_report.json
    python -m kore.analysis.p0_sol --out runs/p0_full.json         # full sweep

Dry-run is CPU-only (no GPU): it computes the roofline for every task and pulls any
timing already in the replay cache, so you can prove wiring before spending GPU time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from kore.analysis.rooflines import (
    Roofline,
    detect_arch,
    resolve_peaks,
    roofline,
    shape_to_str,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_MEDIAN = re.compile(r"median_ms:\s*([-\d.eE]+)")
_SNR = re.compile(r"SNR:\s*([-\d.eE]+)")
_ALLCLOSE = re.compile(r"allclose:\s*(True|False)", re.IGNORECASE)
# emitted once per process by kore.tasks.aiter_ref._mark_baseline; lets us honestly
# label each check-(a) baseline as aiter_vendor / hipblaslt_vendor / framework.
_BASELINE_TAG = re.compile(r"KORE_BASELINE_IMPL:(\w+)")


# --------------------------------------------------------------------------- #
# pure stats (no scipy; numpy optional)
# --------------------------------------------------------------------------- #
def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average rank (1-based) for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = math.sqrt(sum((a - mx) ** 2 for a in x))
    syy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy)


def spearman(x: list[float], y: list[float]) -> Optional[float]:
    """Spearman rank correlation (ties -> average ranks). None if undefined."""
    if len(x) != len(y) or len(x) < 2:
        return None
    return _pearson(_rank(x), _rank(y))


def ols_r2(X: list[list[float]], y: list[float]) -> Optional[float]:
    """R^2 of an ordinary-least-squares fit y ~ [X | 1]. Needs numpy; None if absent
    or under-determined."""
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    if len(y) < 3 or len(X) != len(y):
        return None
    A = np.array([row + [1.0] for row in X], dtype=float)
    b = np.array(y, dtype=float)
    if A.shape[0] <= A.shape[1]:  # under-determined -> trivial/meaningless R^2
        return None
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:  # noqa: BLE001
        return None
    pred = A @ coef
    ss_res = float(((b - pred) ** 2).sum())
    ss_tot = float(((b - b.mean()) ** 2).sum())
    if ss_tot == 0:
        return None
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------- #
# GPU measurement: stage kernel, run driver for correctness + candidate/vendor
# timing, and (optionally) profile PMC counters via rocprofv3. Mirrors KoreEnv's
# proven staging (start_new_session + killpg on timeout, last-match verdict parse).
# --------------------------------------------------------------------------- #
@dataclass
class KernelMeasure:
    task_id: str
    label: str
    correct: bool
    snr_db: Optional[float]
    cand_ms: Optional[float]          # candidate median wall time
    vendor_ms: Optional[float]        # production baseline median (AITER/hipBLASLt/torch)
    t_min_ms: float                   # roofline lower bound
    eta: Optional[float]              # T_min / cand_ms  (SOL attainment)
    speedup: Optional[float]          # vendor_ms / cand_ms
    residual_ms: Optional[float]      # cand_ms - t_min_ms
    counters: dict = field(default_factory=dict)
    stall_frac: Optional[float] = None       # MemUnitStalled / 100
    occupancy: Optional[float] = None         # OccupancyPercent / 100
    baseline_type: Optional[str] = None       # aiter_vendor | hipblaslt_vendor | framework
    error: Optional[str] = None


def _exec(cmd: list[str], cwd: Path, env: dict, timeout: int) -> tuple[int, str, bool]:
    p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, (out or "") + "\n" + (err or ""), False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            out, err = p.communicate(timeout=10)
        except Exception:  # noqa: BLE001
            out, err = "", ""
        return -9, (out or "") + "\n" + (err or ""), True


def _last(pat: re.Pattern, text: str):
    ms = list(pat.finditer(text))
    return ms[-1] if ms else None


def _stage(task, source: str) -> Path:
    wd = Path(tempfile.mkdtemp(prefix=f"p0_{task.task_id}_"))
    for p in task.dir.glob("*.py"):
        shutil.copy(p, wd / p.name)
    (wd / "kernel.py").write_text(source)
    return wd


def _proc_env(device: str, arch: str) -> dict:
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = device
    env["GPU_TARGET"] = arch
    # p0 process runs under the venv (kore installed editable), so `import kore.*`
    # already resolves in the child; add REPO_ROOT to PYTHONPATH belt-and-suspenders.
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _bench(driver: Path, shape_args: list[str], impl: str, warmup: int, iters: int,
           env: dict, timeout: int) -> tuple[Optional[float], Optional[str]]:
    """Return (median_ms, baseline_type). ``baseline_type`` is only populated for the
    ``reference`` impl (the candidate kernel never emits the sentinel)."""
    cmd = [sys.executable, str(driver), "--bench-mode", "--impl", impl,
           "--warmup", str(warmup), "--iters", str(iters), *shape_args]
    rc, out, timed = _exec(cmd, driver.parent, env, timeout)
    if timed or rc != 0:
        return None, None
    m = _last(_MEDIAN, out)
    tag = _last(_BASELINE_TAG, out)
    return (float(m.group(1)) if m else None), (tag.group(1) if tag else None)


def _bench_repeated(driver: Path, shape_args: list[str], impl: str, warmup: int,
                    iters: int, env: dict, timeout: int,
                    reps: int) -> tuple[Optional[float], Optional[str], list[float]]:
    """Run ``_bench`` ``reps`` times (fresh process each = reseeded schedule/thermal
    state) and return (median-of-run-medians, baseline_type, all run medians).

    ``reps==1`` reproduces the single-shot path exactly. Repeats tighten each
    (kernel, shape) point against run-to-run timing noise; the check-level 95% CI
    comes from bootstrapping over the SET of points, not from these repeats."""
    meds: list[float] = []
    tag: Optional[str] = None
    for _ in range(max(1, reps)):
        m, t = _bench(driver, shape_args, impl, warmup, iters, env, timeout)
        if m is not None:
            meds.append(m)
        if t:
            tag = t
    if not meds:
        return None, tag, []
    s = sorted(meds)
    return s[len(s) // 2], tag, meds


def _correctness(driver: Path, shape_args: list[str], env: dict,
                 timeout: int) -> tuple[Optional[float], Optional[bool], Optional[str]]:
    rc, out, timed = _exec([sys.executable, str(driver), *shape_args], driver.parent, env, timeout)
    if timed:
        return None, None, "timeout"
    snr_m = _last(_SNR, out)
    ac_m = _last(_ALLCLOSE, out)
    snr = float(snr_m.group(1)) if snr_m else None
    ac = (ac_m.group(1).lower() == "true") if ac_m else None
    err = None if (snr_m or ac_m) else out.strip()[-400:]
    return snr, ac, err


# gfx950/CDNA4 derived metrics (verified via `rocprofv3 --list-avail`). These are
# the named residual components: occupancy, memory-stall, MFMA (compute) utilization,
# and active cycles. NB: gfx950 renamed the raw counters (e.g.
# SQ_INSTS_VALU_MFMA_MOPS_BF16, not ..._MFMA_BF16), so the old SQ_* names collect
# nothing — we request the derived metrics instead.
_PMC_METRICS = ["OccupancyPercent", "MemUnitStalled", "MfmaUtil", "GRBM_GUI_ACTIVE"]


def _profile_pmc(driver: Path, shape_args: list[str], env: dict, timeout: int) -> dict:
    """Run the candidate under rocprofv3 --pmc; return the main kernel's counters.

    rocprofv3's gfx950 CSV is LONG format (one row per dispatch x counter, columns
    ``Kernel_Name, Counter_Name, Counter_Value, Start_Timestamp, End_Timestamp``).
    We group by kernel, pick the one with the largest total GPU time (the compute
    kernel, not input setup), and return its mean counter values. Best-effort: any
    failure returns {} so checks (b)/(c) degrade gracefully.
    """
    if not shutil.which("rocprofv3"):
        return {}
    import csv as _csv
    import glob as _glob
    outdir = Path(tempfile.mkdtemp(prefix="p0_pmc_"))
    cmd = ["rocprofv3", "--pmc", *_PMC_METRICS, "-d", str(outdir), "--output-format", "csv",
           "--", sys.executable, str(driver), "--bench-mode", "--impl", "candidate",
           "--warmup", "2", "--iters", "5", *shape_args]
    try:
        rc, out, timed = _exec(cmd, driver.parent, env, timeout)
        if timed or rc != 0:
            return {}
        csvs = _glob.glob(str(outdir / "**" / "*counter_collection.csv"), recursive=True) \
            + _glob.glob(str(outdir / "*counter_collection.csv"))
        per_kernel: dict[str, dict] = {}
        for cp in csvs:
            try:
                with open(cp, newline="") as f:
                    for row in _csv.DictReader(f):
                        kn = row.get("Kernel_Name") or ""
                        cname = row.get("Counter_Name") or ""
                        if not kn or not cname:
                            continue
                        try:
                            cval = float(row.get("Counter_Value", "nan"))
                            dur = float(row.get("End_Timestamp", 0)) - float(row.get("Start_Timestamp", 0))
                        except (TypeError, ValueError):
                            continue
                        d = per_kernel.setdefault(kn, {"dur": 0.0, "vals": {}})
                        d["dur"] += max(dur, 0.0)
                        d["vals"].setdefault(cname, []).append(cval)
            except Exception:  # noqa: BLE001
                continue
        if not per_kernel:
            return {}
        main = max(per_kernel.values(), key=lambda d: d["dur"])
        return {name: (sum(v) / len(v)) for name, v in main["vals"].items() if v}
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def _decompose(counters: dict) -> tuple[Optional[float], Optional[float]]:
    """Named residual fractions from gfx950 derived metrics: (stall_frac, occupancy).

    ``MemUnitStalled`` and ``OccupancyPercent`` are rocprofv3 derived metrics on a
    0..100 scale; we normalize to 0..1. ``stall_frac`` is the memory-stall fraction
    (time the memory unit was stalled) and ``occupancy`` is achieved GPU occupancy.
    These are the counter-derived regressors for the residual decomposition (check b):
    residual time is expected to grow with stall and with the occupancy *deficit*.
    """
    stall = counters.get("MemUnitStalled")
    occ = counters.get("OccupancyPercent")
    stall_frac = (float(stall) / 100.0) if stall is not None else None
    occupancy = (float(occ) / 100.0) if occ is not None else None
    return stall_frac, occupancy


def measure_kernel(task, label: str, source: str, shape, peaks: dict, arch: str,
                   warmup: int, iters: int, device: str, do_pmc: bool,
                   timeout: int = 300, reseeds: int = 1) -> KernelMeasure:
    dims = shape.dims
    rf = roofline(task.task_id, task.operation, task.dtype, shape_to_str(dims), dims, peaks, arch)
    t_min = rf.t_min_ms if rf else float("nan")
    wd = _stage(task, source)
    env = _proc_env(device, arch)
    try:
        driver = wd / "driver.py"
        snr, ac, err = _correctness(driver, shape.as_args(), env, timeout)
        correct = bool(ac) and (snr is not None) and (snr >= float(getattr(task, "snr_threshold", 25.0)))
        cand_ms = None
        if correct:
            cand_ms, _, _ = _bench_repeated(driver, shape.as_args(), "candidate", warmup, iters, env, timeout, reseeds)
        vendor_ms, baseline_type, _ = _bench_repeated(driver, shape.as_args(), "reference", warmup, iters, env, timeout, reseeds)
        counters = {}
        stall_frac = occupancy = None
        if do_pmc and correct and cand_ms:
            counters = _profile_pmc(driver, shape.as_args(), env, timeout)
            stall_frac, occupancy = _decompose(counters)
        eta = (t_min / cand_ms) if (cand_ms and t_min == t_min) else None
        speedup = (vendor_ms / cand_ms) if (vendor_ms and cand_ms) else None
        residual = (cand_ms - t_min) if (cand_ms and t_min == t_min) else None
        return KernelMeasure(
            task_id=task.task_id, label=label, correct=correct, snr_db=snr,
            cand_ms=cand_ms, vendor_ms=vendor_ms, t_min_ms=t_min, eta=eta,
            speedup=speedup, residual_ms=residual, counters=counters,
            stall_frac=stall_frac, occupancy=occupancy, baseline_type=baseline_type,
            error=err if not correct else None,
        )
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# --------------------------------------------------------------------------- #
# trajectory assembly: seed + cached group candidates (for check c ordering)
# --------------------------------------------------------------------------- #
def trajectory_sources(task, max_kernels: int, generate_variants: bool = True) -> list[tuple[str, str]]:
    """(label, source) trajectory: seed, cached group candidates, then generated
    optimization variants (tile/vectorize/pipeline/num_warps sweeps of the seed).

    The variants give check (c) a real trajectory of correct-but-varying-speed
    kernels: they change schedule knobs (tile sizes, num_warps, num_stages,
    vectorization) that move the kernel through the physical state space while
    preserving correctness, which is exactly the "in-valley" motion (c) tests.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    seed = None
    try:
        seed = task.seed_source
        out.append(("seed", seed))
        seen.add(seed.strip())
    except Exception:  # noqa: BLE001
        pass
    groups = REPO_ROOT / "data" / "groups" / f"{task.task_id}.jsonl"
    if groups.exists():
        for line in groups.read_text().splitlines():
            if len(out) >= max_kernels:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for cand in rec.get("candidates", []) or []:
                src = cand.get("source") or ""
                if src and src.strip() not in seen:
                    out.append((f"cand{len(out)}", src))
                    seen.add(src.strip())
                    if len(out) >= max_kernels:
                        break
    if generate_variants and seed and len(out) < max_kernels:
        import random
        try:
            from kore.data.mutate import apply_operator, list_operators
            ops = list_operators("optimize")
            rng = random.Random(1234)
            frontier = [seed]
            while len(out) < max_kernels and frontier:
                base = frontier.pop(0)
                for op in ops:
                    if len(out) >= max_kernels:
                        break
                    try:
                        variant, _ = apply_operator(op, base, rng)
                    except Exception:  # noqa: BLE001
                        continue
                    if variant and variant.strip() not in seen:
                        out.append((f"opt:{op}", variant))
                        seen.add(variant.strip())
                        frontier.append(variant)
        except Exception:  # noqa: BLE001 - mutate unavailable -> seed/groups only
            pass
    return out[:max_kernels]


# --------------------------------------------------------------------------- #
# the three checks
# --------------------------------------------------------------------------- #
def _baseline_table(measures: list[KernelMeasure]) -> tuple[list[dict], dict]:
    """Per-operator {operator, baseline_type, speedup(median)} + a type->count map.

    Aggregates over trajectory variants: the baseline (reference) is per-operator, so
    we report the median speedup of that operator's timed kernels and the baseline type
    actually used (aiter_vendor > hipblaslt_vendor > framework precedence if mixed)."""
    from statistics import median
    _prec = {"aiter_vendor": 3, "hipblaslt_vendor": 2, "framework": 1}
    by_op: dict[str, dict] = {}
    for m in measures:
        if m.speedup is None:
            continue
        d = by_op.setdefault(m.task_id, {"speedups": [], "baseline_type": None})
        d["speedups"].append(m.speedup)
        bt = m.baseline_type
        if bt and _prec.get(bt, 0) > _prec.get(d["baseline_type"] or "", 0):
            d["baseline_type"] = bt
    table = [{"operator": k, "baseline_type": v["baseline_type"],
              "speedup": median(v["speedups"])} for k, v in sorted(by_op.items())]
    comp: dict[str, int] = {}
    for row in table:
        comp[row["baseline_type"] or "unknown"] = comp.get(row["baseline_type"] or "unknown", 0) + 1
    return table, comp


def check_a(measures: list[KernelMeasure]) -> dict:
    pts = [(m.eta, m.speedup) for m in measures if m.eta and m.speedup]
    table, comp = _baseline_table(measures)
    if len(pts) < 3:
        return {"rho": None, "n": len(pts), "verdict": "SKIP",
                "note": "need >=3 kernels with both eta and vendor speedup",
                "by_operator": table, "baseline_composition": comp}
    rho = spearman([p[0] for p in pts], [p[1] for p in pts])
    verdict = "PASS" if (rho is not None and rho >= 0.5 and len(pts) >= 5) else "WEAK"
    return {"rho": rho, "n": len(pts), "verdict": verdict,
            "by_operator": table, "baseline_composition": comp}


def check_b(measures: list[KernelMeasure]) -> dict:
    # regress the residual time on counter-derived "time lost" terms:
    #   t_stall = stall_frac * measured ; t_occ_deficit = (1 - occupancy) * measured
    rows = [m for m in measures if m.stall_frac is not None and m.occupancy is not None
            and m.residual_ms is not None and m.cand_ms]
    if len(rows) < 5:
        return {"r2": None, "n": len(rows), "verdict": "SKIP",
                "note": "need >=5 kernels with PMC counters + residual"}
    X = [[m.stall_frac * m.cand_ms, (1.0 - m.occupancy) * m.cand_ms] for m in rows]
    y = [m.residual_ms for m in rows]
    r2 = ols_r2(X, y)
    if r2 is None:
        return {"r2": None, "n": len(rows), "verdict": "WEAK", "note": "numpy absent / under-determined"}
    verdict = "PASS" if r2 >= 0.7 else "WEAK"
    return {"r2": r2, "n": len(rows), "verdict": verdict}


def _dominant_residual(m: KernelMeasure) -> float:
    """Dominant named residual component: max(stall fraction, occupancy deficit)."""
    return max(m.stall_frac or 0.0, 1.0 - (m.occupancy if m.occupancy is not None else 1.0))


def check_c(per_task: dict[str, list[KernelMeasure]]) -> dict:
    flat_tol = 0.10  # wall-clock "flat" if |d wall|/wall < 10%
    in_valley = 0
    monotone = 0
    tasks_used = 0
    for tid, ms in per_task.items():
        traj = [m for m in ms if m.correct and m.cand_ms and m.stall_frac is not None]
        if len(traj) < 3:
            continue
        # order by SOL attainment (eta) ascending = "improvement" direction; robust
        # when no vendor baseline exists (speedup may be None for aiter-only tasks).
        traj.sort(key=lambda m: (m.eta or 0.0))
        tasks_used += 1
        for a, b in zip(traj, traj[1:]):
            dwall = abs((b.cand_ms - a.cand_ms) / a.cand_ms) if a.cand_ms else 1.0
            if dwall < flat_tol:
                in_valley += 1
                if _dominant_residual(b) < _dominant_residual(a):  # dominant residual term falls
                    monotone += 1
    if in_valley < 3:
        return {"frac": None, "in_valley_pairs": in_valley, "tasks": tasks_used,
                "verdict": "SKIP", "note": "need >=3 flat-wall adjacent pairs across tasks"}
    frac = monotone / in_valley
    verdict = "PASS" if frac >= 0.6 else "WEAK"
    return {"frac": frac, "in_valley_pairs": in_valley, "tasks": tasks_used, "verdict": verdict}


def _select_shapes(task, n: int) -> list:
    """Up to ``n`` REPRESENTATIVE shapes for a task (primary first, then validation_*).

    The per-task ``minimal`` shape is a tiny correctness-only shape (e.g. M=64,N=512)
    that is launch/overhead-bound: the operator's mandatory work is nanoseconds while
    the kernel-launch floor is microseconds, so eta ~ 0 regardless of kernel quality
    and the roofline SOL model does not apply. Including it pollutes check (a) with an
    uncorrelated cluster, so representative shapes deliberately EXCLUDE ``minimal``
    (documented limitation: roofline predicts speedup only in the work-bound regime).

    ``n==1`` yields just the primary shape -> the single-shot behavior."""
    ordered: list = []
    prim = task.shape("primary")
    if prim is not None:
        ordered.append(prim)
    for s in task.shapes:
        if s.name in ("primary", "minimal") or s in ordered:
            continue
        ordered.append(s)
    if not ordered and task.shapes:
        ordered = [s for s in task.shapes if s.name != "minimal"] or list(task.shapes)
    return ordered[:max(1, n)]


def _bootstrap_check_cis(all_measures: list, per_task: dict, B: int,
                         seed: int = 12345) -> tuple:
    """Percentile bootstrap 95% CIs for (rho, R^2, frac) by resampling the check
    inputs with replacement B times and re-running the exact check functions.

    check (a)/(b): resample the kernel-measure points; check (c): resample the
    per-(task,shape) trajectories. Returns (rho_ci, r2_ci, frac_ci); each is
    ``[lo, hi]`` or None if too few valid resamples."""
    import random
    rng = random.Random(seed)
    rho_s: list[float] = []
    r2_s: list[float] = []
    frac_s: list[float] = []
    n = len(all_measures)
    keys = list(per_task.keys())
    for _ in range(max(1, B)):
        if n:
            samp = [all_measures[rng.randrange(n)] for _ in range(n)]
            ra = check_a(samp)
            if ra.get("rho") is not None:
                rho_s.append(ra["rho"])
            rb = check_b(samp)
            if rb.get("r2") is not None:
                r2_s.append(rb["r2"])
        if keys:
            samp_pt = {}
            for i in range(len(keys)):
                pk = keys[rng.randrange(len(keys))]
                samp_pt[f"{pk}#{i}"] = per_task[pk]
            rc = check_c(samp_pt)
            if rc.get("frac") is not None:
                frac_s.append(rc["frac"])

    def pctl(xs: list[float]):
        if len(xs) < 20:
            return None
        xs = sorted(xs)
        lo = xs[int(0.025 * (len(xs) - 1))]
        hi = xs[int(0.975 * (len(xs) - 1))]
        return [round(lo, 4), round(hi, 4)]

    return pctl(rho_s), pctl(r2_s), pctl(frac_s)


def decide(a: dict, b: dict, c: dict, dry_run: bool) -> str:
    if dry_run:
        return "DRY_RUN"
    av, bv, cv = a["verdict"], b["verdict"], c["verdict"]
    if av == "PASS" and bv == "PASS" and cv == "PASS":
        return "GO"
    if av == "PASS" and bv == "PASS":
        return "PARTIAL"
    if av == "PASS":
        return "FALLBACK"
    if av in ("WEAK", "SKIP"):
        return "PIVOT?" if av == "WEAK" else "INSUFFICIENT_DATA"
    return "PIVOT"


# --------------------------------------------------------------------------- #
# dry-run: roofline table + eta from cached replay (no GPU)
# --------------------------------------------------------------------------- #
def _mine_replay_eta(task, peaks: dict, arch: str, replay_dir: Path) -> list[KernelMeasure]:
    path = replay_dir / f"replay_{task.task_id}.jsonl"
    if not path.exists():
        return []
    out: list[KernelMeasure] = []
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = rec.get("key", "")
        val = rec.get("value", {}) or {}
        if "::" not in key:
            continue
        shape_str = key.split("::", 1)[1]
        dims = {m.group(1): int(m.group(2))
                for m in re.finditer(r"([A-Za-z_]\w*)=(-?\d+)", shape_str)}
        rf = roofline(task.task_id, task.operation, task.dtype, shape_str, dims, peaks, arch)
        if rf is None:
            continue
        wbs = val.get("wall_by_shape") or {}
        bbs = val.get("baseline_by_shape") or {}
        cand_ms = float(wbs.get(shape_str)) if wbs.get(shape_str) else (
            float(val["wall_ms"]) if val.get("wall_ms") else None)
        vendor_ms = float(bbs.get(shape_str)) if bbs.get(shape_str) else (
            float(val["baseline_ms"]) if val.get("baseline_ms") else None)
        if not cand_ms:
            continue
        out.append(KernelMeasure(
            task_id=task.task_id, label=f"cache:{shape_str}",
            correct=bool(val.get("validation_passed")), snr_db=val.get("snr_db"),
            cand_ms=cand_ms, vendor_ms=vendor_ms, t_min_ms=rf.t_min_ms,
            eta=rf.t_min_ms / cand_ms, speedup=(vendor_ms / cand_ms) if vendor_ms else None,
            residual_ms=cand_ms - rf.t_min_ms,
        ))
    return out


# --------------------------------------------------------------------------- #
# report / verdict box
# --------------------------------------------------------------------------- #
def _fmt(v, pct=False, suffix=""):
    if v is None:
        return "-"
    return (f"{v*100:.1f}%" if pct else f"{v:.3f}") + suffix


def render(report: dict) -> str:
    L: list[str] = []
    arch = report["arch"]
    p = report["peaks"]
    L.append(f"# P0 roofline / SOL  --  arch={arch}  {'(DRY RUN)' if report['dry_run'] else ''}")
    L.append(f"# peaks: HBM {p['hbm_bytes_per_s']/1e12:.1f} TB/s | bf16 "
             f"{p['bf16_flops_per_s']/1e15:.2f} PF/s | fp8 {p['fp8_flops_per_s']/1e15:.2f} PF/s")
    L.append("")
    hdr = f"{'task':26s} {'dtype':6s} {'bound':7s} {'AI(F/B)':>9s} {'T_min':>10s}"
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in report["rooflines"]:
        L.append(f"{r['task_id']:26s} {r['dtype']:6s} {r['bound']:7s} "
                 f"{r['arithmetic_intensity']:9.2f} {r['t_min_ms']*1e3:8.1f}us")
    if report["unmodeled"]:
        L.append("\nunmodeled: " + ", ".join(report["unmodeled"]))

    meas = report.get("measures") or []
    timed = [m for m in meas if m.get("eta")]
    if timed:
        L.append("\n## measured (eta = T_min/measured; speedup = vendor/candidate)")
        h2 = f"{'task':22s} {'label':10s} {'ok':3s} {'eta':>7s} {'speedup':>8s} {'stall':>7s}"
        L.append(h2)
        L.append("-" * len(h2))
        for m in timed:
            L.append(f"{m['task_id']:22s} {m['label'][:10]:10s} "
                     f"{'Y' if m['correct'] else 'n':3s} {_fmt(m['eta'], pct=True):>7s} "
                     f"{_fmt(m['speedup'], suffix='x'):>8s} {_fmt(m['stall_frac'], pct=True):>7s}")

    a, b, c = report["checks"]["a"], report["checks"]["b"], report["checks"]["c"]

    def _ci(d):
        ci = d.get("ci95")
        return f" 95%CI[{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""

    tbl = a.get("by_operator") or []
    if tbl:
        L.append("\n## check-(a) baselines (operator -> baseline_type, median speedup)")
        h3 = f"{'operator':22s} {'baseline_type':18s} {'speedup':>8s}"
        L.append(h3)
        L.append("-" * len(h3))
        for row in tbl:
            L.append(f"{row['operator']:22s} {str(row['baseline_type']):18s} "
                     f"{_fmt(row['speedup'], suffix='x'):>8s}")
        comp = a.get("baseline_composition") or {}
        L.append("baseline composition: " + ", ".join(f"{k}={v}" for k, v in sorted(comp.items())))
    L.append("\n" + "=" * 60)
    L.append(f"(a) eta predicts speedup   : rho={_fmt(a.get('rho'))} (n={a.get('n')}){_ci(a)}   -> {a['verdict']}")
    L.append(f"(b) residual decomp R^2    : {_fmt(b.get('r2'))} (n={b.get('n')}){_ci(b)}        -> {b['verdict']}")
    L.append(f"(c) monotone-in-valley frac: {_fmt(c.get('frac'))} "
             f"(pairs={c.get('in_valley_pairs')}){_ci(c)}  -> {c['verdict']}")
    L.append(f"DECISION: {report['decision']}")
    L.append("=" * 60)
    if report["dry_run"]:
        L.append("DRY RUN: only (a) is meaningful (from cached data); run on-GPU for (b)/(c).")
    for note in (a.get("note"), b.get("note"), c.get("note")):
        if note:
            L.append(f"  note: {note}")
    return "\n".join(L)


def run(tasks: list[str], arch: str, peaks: dict, warmup: int, iters: int,
        max_kernels: int, device: str, dry_run: bool, do_pmc: bool,
        replay_dir: Path, shapes_per_task: int = 1, reseeds: int = 1,
        bootstrap: int = 0) -> dict:
    from kore.tasks.registry import all_tasks, get_task

    task_objs = [get_task(t) for t in tasks] if tasks else all_tasks()

    rooflines: list[dict] = []
    unmodeled: list[str] = []
    all_measures: list[KernelMeasure] = []
    per_task: dict[str, list[KernelMeasure]] = {}

    for t in task_objs:
        prim = t.shape("primary") or (t.shapes[0] if t.shapes else None)
        if prim is None:
            continue
        rf = roofline(t.task_id, t.operation, t.dtype, shape_to_str(prim.dims), prim.dims, peaks, arch)
        if rf is None:
            unmodeled.append(f"{t.task_id} ({t.operation})")
            continue
        rooflines.append(asdict(rf))

        if dry_run:
            ms = _mine_replay_eta(t, peaks, arch, replay_dir)
            per_task[t.task_id] = ms
            all_measures.extend(ms)
        else:
            shapes = _select_shapes(t, shapes_per_task)
            for label, src in trajectory_sources(t, max_kernels):
                for shp in shapes:
                    m = measure_kernel(t, f"{label}@{shp.name}", src, shp, peaks, arch,
                                       warmup, iters, device, do_pmc, reseeds=reseeds)
                    # one trajectory per (task, shape) keeps check (c)'s flat-wall
                    # ordering coherent; check (a)/(b) pool all points.
                    per_task.setdefault(f"{t.task_id}@{shp.name}", []).append(m)
                    all_measures.append(m)

    a = check_a(all_measures)
    b = check_b(all_measures)
    c = check_c(per_task)
    if bootstrap and not dry_run:
        rho_ci, r2_ci, frac_ci = _bootstrap_check_cis(all_measures, per_task, bootstrap)
        if rho_ci:
            a["ci95"] = rho_ci
        if r2_ci:
            b["ci95"] = r2_ci
        if frac_ci:
            c["ci95"] = frac_ci
    decision = decide(a, b, c, dry_run)
    return {
        "arch": arch, "peaks": peaks, "dry_run": dry_run,
        "shapes_per_task": shapes_per_task, "reseeds": reseeds, "bootstrap": bootstrap,
        "rooflines": rooflines, "unmodeled": unmodeled,
        "measures": [asdict(m) for m in all_measures],
        "checks": {"a": a, "b": b, "c": c},
        "decision": decision,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P0 roofline/SOL falsification test")
    ap.add_argument("--tasks", default=None, help="comma-separated task ids (default: all)")
    ap.add_argument("--arch", default=None, help="gfx950 (default, auto-detected) | gfx942")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-kernels-per-task", type=int, default=8, dest="max_kernels")
    ap.add_argument("--shapes-per-task", type=int, default=1, dest="shapes_per_task",
                    help="measure each operator at up to N shapes (>=3 for CI)")
    ap.add_argument("--reseeds", type=int, default=1,
                    help="repeat each (kernel,shape) timing N times (median-of-medians)")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="bootstrap resamples for 95%% CIs on all three checks (e.g. 1000)")
    ap.add_argument("--device", default="0", help="HIP_VISIBLE_DEVICES")
    ap.add_argument("--dry-run", action="store_true", help="CPU-only: roofline + cached eta")
    ap.add_argument("--no-pmc", action="store_true", help="skip rocprofv3 PMC (checks b/c weaker)")
    ap.add_argument("--replay-dir", default="runs")
    ap.add_argument("--out", default="runs/p0_report.json")
    args = ap.parse_args(argv)

    arch = args.arch or detect_arch()
    peaks = resolve_peaks(arch)
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else []
    report = run(tasks, arch, peaks, args.warmup, args.iters, args.max_kernels,
                 args.device, args.dry_run, not args.no_pmc, Path(args.replay_dir),
                 shapes_per_task=args.shapes_per_task, reseeds=args.reseeds,
                 bootstrap=args.bootstrap)
    text = render(report)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[p0_sol] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
