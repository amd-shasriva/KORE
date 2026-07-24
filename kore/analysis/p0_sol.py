"""Preregistered P0 falsification of roofline and residual reward semantics.

The primary analyses remove the shared ``T_candidate`` denominator that inflated
the original in-sample statistics:

* check (a) compares eta against a T_candidate-only predictor and a
  denominator-preserving T_min permutation null;
* check (b) predicts the normalized gap on held-out task clusters, compares
  T_candidate/intercept baselines, performs leave-family-out evaluation, and
  retains raw residual R² only as a leakage diagnostic;
* check (c) preserves trajectory collection order instead of sorting by eta.

Task-cluster bootstrap intervals, preregistered thresholds, and Benjamini-
Hochberg correction are mandatory before an operator family can authorize
empirical shaping.  A failure leaves the physical model available for
conservative integrity rejection/pruning only.

Usage:
    python -m kore.analysis.p0_sol --tasks rmsnorm_aiter,gemm_bf16 --dry-run
    python -m kore.analysis.p0_sol --tasks gemm_bf16,softmax_bf16,gelu_tanh_bf16 \
        --warmup 10 --iters 50 --out runs/p0_report.json
    python -m kore.analysis.p0_sol --out runs/p0_full.json         # full sweep

``--reanalyze`` is CPU-only and applies the full controls to an existing report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from kore.analysis.roofline import (
    ModelError,
    PhysicalModel,
    make_physical_model,
    model_from_peak_mapping,
)
from kore.analysis.rooflines import (
    Roofline,
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
    model_fingerprint: Optional[str] = None
    family: Optional[str] = None
    shape_id: Optional[str] = None


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
# nothing - we request the derived metrics instead.
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
    def percent(value):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 100.0
        ):
            return None
        return float(value) / 100.0

    stall_frac = percent(stall)
    occupancy = percent(occ)
    return stall_frac, occupancy


def measure_kernel(task, label: str, source: str, shape, peaks: dict, arch: str,
                   warmup: int, iters: int, device: str, do_pmc: bool,
                   timeout: int = 300, reseeds: int = 1,
                   model: Optional[PhysicalModel] = None) -> KernelMeasure:
    dims = shape.dims
    rf = roofline(
        task.task_id, task.operation, task.dtype, shape_to_str(dims), dims,
        peaks, arch, model=model)
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
        try:
            from kore.eval.generalization import family_of
            family = family_of(task.task_id)
        except Exception:  # noqa: BLE001
            family = None
        return KernelMeasure(
            task_id=task.task_id, label=label, correct=correct, snr_db=snr,
            cand_ms=cand_ms, vendor_ms=vendor_ms, t_min_ms=t_min, eta=eta,
            speedup=speedup, residual_ms=residual, counters=counters,
            stall_frac=stall_frac, occupancy=occupancy, baseline_type=baseline_type,
            error=err if not correct else None,
            model_fingerprint=(model.fingerprint if model else getattr(rf, "model_fingerprint", None)),
            family=family, shape_id=getattr(shape, "name", None),
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
# Preregistered, leakage-controlled validation
# --------------------------------------------------------------------------- #
PREREGISTRATION: dict[str, Any] = {
    "schema": "kore.p0-validation.v2",
    "primary_target": "normalized_gap=(T_candidate-T_min)/T_candidate",
    "cluster_unit": "task_id",
    "cv": "deterministic five-fold task-cluster",
    "null": "within-task joint feature permutation preserving T_candidate",
    "multiple_testing": "Benjamini-Hochberg FDR",
    "alpha": 0.05,
    "min_points": 30,
    "min_task_clusters": 6,
    "min_normalized_cv_r2": 0.10,
    "min_increment_over_baseline": 0.05,
    "check_a_min_rho": 0.50,
    "check_a_min_increment_over_tcand": 0.05,
    "check_c_min_pairs": 20,
    "check_c_min_fraction": 0.60,
}


def _number(value: Any, *, positive: bool = False) -> bool:
    good = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    return bool(good and (not positive or float(value) > 0.0))


def _fit(X: list[list[float]], y: list[float]) -> Optional[list[float]]:
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    if len(X) != len(y) or len(y) < 2:
        return None
    A = np.array([list(row) + [1.0] for row in X], dtype=float)
    b = np.array(y, dtype=float)
    if A.shape[0] <= A.shape[1] or not np.isfinite(A).all() or not np.isfinite(b).all():
        return None
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:  # noqa: BLE001
        return None
    return [float(value) for value in coef]


def _predict(coef: Optional[list[float]], X: list[list[float]]) -> Optional[list[float]]:
    if coef is None:
        return None
    return [
        sum(weight * value for weight, value in zip(coef[:-1], row)) + coef[-1]
        for row in X
    ]


def _r2(y: list[float], pred: Optional[list[float]]) -> Optional[float]:
    if pred is None or len(y) != len(pred) or len(y) < 2:
        return None
    mean = sum(y) / len(y)
    total = sum((value - mean) ** 2 for value in y)
    if total <= 1e-30:
        return None
    return 1.0 - sum((value - fitted) ** 2 for value, fitted in zip(y, pred)) / total


def _cluster_cv(
    rows: list[dict],
    design: Callable[[dict], list[float]],
    target: Callable[[dict], float],
    *,
    folds: int = 5,
    seed: int = 1729,
) -> dict:
    groups = sorted(
        {str(row.get("_group") or row["task_id"]) for row in rows},
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    if len(groups) < 3:
        return {"r2": None, "fold_r2": [], "n_groups": len(groups)}
    fold_count = min(max(2, folds), len(groups))
    assignments = {group: index % fold_count for index, group in enumerate(groups)}
    all_y: list[float] = []
    all_pred: list[float] = []
    fold_scores: list[Optional[float]] = []
    for fold in range(fold_count):
        train = [
            row for row in rows
            if assignments[str(row.get("_group") or row["task_id"])] != fold
        ]
        test = [
            row for row in rows
            if assignments[str(row.get("_group") or row["task_id"])] == fold
        ]
        X_train, y_train = [design(row) for row in train], [target(row) for row in train]
        X_test, y_test = [design(row) for row in test], [target(row) for row in test]
        coef = _fit(X_train, y_train)
        predicted = _predict(coef, X_test)
        if predicted is None:
            continue
        all_y.extend(y_test)
        all_pred.extend(predicted)
        fold_scores.append(_r2(y_test, predicted))
    return {
        "r2": _r2(all_y, all_pred),
        "fold_r2": fold_scores,
        "n_groups": len(groups),
        "n_predictions": len(all_y),
    }


def _percentile_ci(values: list[float]) -> Optional[list[float]]:
    finite = sorted(value for value in values if _number(value))
    if len(finite) < 20:
        return None
    lo = finite[int(0.025 * (len(finite) - 1))]
    hi = finite[int(0.975 * (len(finite) - 1))]
    return [round(lo, 6), round(hi, 6)]


def _cluster_bootstrap(
    rows: list[dict],
    metric: Callable[[list[dict]], Optional[float]],
    samples: int,
    seed: int,
) -> Optional[list[float]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["task_id"]), []).append(row)
    names = sorted(groups)
    if samples < 20 or len(names) < 2:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        sampled: list[dict] = []
        for index in range(len(names)):
            name = names[rng.randrange(len(names))]
            for row in groups[name]:
                copied = dict(row)
                copied["_group"] = f"{name}#{index}"
                sampled.append(copied)
        value = metric(sampled)
        if value is not None and _number(value):
            values.append(float(value))
    return _percentile_ci(values)


def _permuted_features(rows: list[dict], rng: random.Random) -> list[dict]:
    """Jointly permute normalized features within task; keep each T_candidate."""
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["task_id"]), []).append(index)
    out = [dict(row) for row in rows]
    for indices in groups.values():
        features = [(rows[index]["stall"], rows[index]["occ_deficit"]) for index in indices]
        rng.shuffle(features)
        for index, (stall, occ_deficit) in zip(indices, features):
            out[index]["stall"] = stall
            out[index]["occ_deficit"] = occ_deficit
    return out


def _permutation_test(
    rows: list[dict],
    metric: Callable[[list[dict]], Optional[float]],
    samples: int,
    seed: int,
) -> dict:
    observed = metric(rows)
    if observed is None or samples <= 0:
        return {"observed": observed, "p_value": None, "null_median": None, "null_ci95": None}
    rng = random.Random(seed)
    null: list[float] = []
    for _ in range(samples):
        value = metric(_permuted_features(rows, rng))
        if value is not None and _number(value):
            null.append(float(value))
    if not null:
        return {"observed": observed, "p_value": None, "null_median": None, "null_ci95": None}
    ordered = sorted(null)
    p_value = (1 + sum(value >= observed for value in ordered)) / (len(ordered) + 1)
    return {
        "observed": observed,
        "p_value": p_value,
        "null_median": ordered[len(ordered) // 2],
        "null_ci95": _percentile_ci(ordered),
        "n_permutations": len(ordered),
    }


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
    return check_a_rigorous(measures)


def check_a_rigorous(
    measures: list[KernelMeasure],
    *,
    permutations: int = 200,
    bootstrap: int = 200,
    seed: int = 173,
) -> dict:
    rows = [
        {
            "task_id": measure.task_id,
            "eta": float(measure.eta),
            "speedup": float(measure.speedup),
            "cand_ms": float(measure.cand_ms),
            "t_min_ms": float(measure.t_min_ms),
        }
        for measure in measures
        if _number(measure.eta, positive=True)
        and _number(measure.speedup, positive=True)
        and _number(measure.cand_ms, positive=True)
        and _number(measure.t_min_ms, positive=True)
    ]
    table, comp = _baseline_table(measures)
    if len(rows) < 3:
        return {"rho": None, "n": len(rows), "verdict": "SKIP",
                "note": "need >=3 finite kernels with eta, T_candidate, and speedup",
                "by_operator": table, "baseline_composition": comp}
    speedups = [row["speedup"] for row in rows]
    rho = spearman([row["eta"] for row in rows], speedups)
    tcand_rho = spearman([1.0 / row["cand_ms"] for row in rows], speedups)
    increment = (
        rho - tcand_rho if rho is not None and tcand_rho is not None else None
    )
    full_cv = _cluster_cv(
        rows,
        lambda row: [math.log(row["eta"])],
        lambda row: math.log(row["speedup"]),
        seed=seed,
    )
    tcand_cv = _cluster_cv(
        rows,
        lambda row: [math.log(1.0 / row["cand_ms"])],
        lambda row: math.log(row["speedup"]),
        seed=seed,
    )

    # Preserve every candidate denominator and speedup; only the physical
    # numerator T_min is shuffled within task.
    rng = random.Random(seed)
    null: list[float] = []
    by_task: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_task.setdefault(row["task_id"], []).append(index)
    for _ in range(max(0, permutations)):
        eta_perm = [0.0] * len(rows)
        for indices in by_task.values():
            numerators = [rows[index]["t_min_ms"] for index in indices]
            rng.shuffle(numerators)
            for index, numerator in zip(indices, numerators):
                eta_perm[index] = numerator / rows[index]["cand_ms"]
        perm_rho = spearman(eta_perm, speedups)
        if perm_rho is not None and tcand_rho is not None:
            null.append(perm_rho - tcand_rho)
    p_value = (
        (1 + sum(value >= increment for value in null)) / (len(null) + 1)
        if null and increment is not None
        else None
    )

    def rho_metric(sampled):
        return spearman(
            [row["eta"] for row in sampled], [row["speedup"] for row in sampled]
        )

    def increment_metric(sampled):
        observed = rho_metric(sampled)
        baseline = spearman(
            [1.0 / row["cand_ms"] for row in sampled],
            [row["speedup"] for row in sampled],
        )
        return observed - baseline if observed is not None and baseline is not None else None

    rho_ci = _cluster_bootstrap(rows, rho_metric, bootstrap, seed + 1)
    increment_ci = _cluster_bootstrap(rows, increment_metric, bootstrap, seed + 2)
    eligible = bool(
        rho is not None
        and increment is not None
        and len(rows) >= PREREGISTRATION["min_points"]
        and len({row["task_id"] for row in rows}) >= PREREGISTRATION["min_task_clusters"]
        and rho >= PREREGISTRATION["check_a_min_rho"]
        and increment >= PREREGISTRATION["check_a_min_increment_over_tcand"]
        and full_cv["r2"] is not None
        and tcand_cv["r2"] is not None
        and full_cv["r2"] - tcand_cv["r2"] >= PREREGISTRATION["min_increment_over_baseline"]
        and p_value is not None
        and p_value <= PREREGISTRATION["alpha"]
    )
    return {
        "rho": rho,
        "n": len(rows),
        "n_task_clusters": len({row["task_id"] for row in rows}),
        "rho_ci95_task_bootstrap": rho_ci,
        "tcand_only_rho": tcand_rho,
        "increment_over_tcand": increment,
        "increment_ci95_task_bootstrap": increment_ci,
        "full_log_cv_r2": full_cv["r2"],
        "tcand_only_log_cv_r2": tcand_cv["r2"],
        "denominator_preserving_null": {
            "p_value": p_value,
            "null_median_increment": (
                sorted(null)[len(null) // 2] if null else None
            ),
            "n_permutations": len(null),
        },
        "p_value": p_value,
        "p_adjusted": p_value,
        "_eligible": eligible,
        "verdict": "PASS" if eligible else "FAIL",
        "by_operator": table,
        "baseline_composition": comp,
    }


def _counter_rows(measures: list[KernelMeasure]) -> tuple[list[dict], int]:
    rows: list[dict] = []
    impossible = 0
    try:
        from kore.eval.generalization import family_of
    except Exception:  # noqa: BLE001
        family_of = lambda task_id: None
    for measure in measures:
        values = (
            measure.stall_frac,
            measure.occupancy,
            measure.residual_ms,
            measure.cand_ms,
            measure.t_min_ms,
        )
        if not all(_number(value) for value in values):
            continue
        stall, occupancy, residual, candidate, t_min = map(float, values)
        if not (0.0 <= stall <= 1.0 and 0.0 <= occupancy <= 1.0 and candidate > 0.0):
            continue
        if residual < -1e-12 or t_min > candidate * (1.0 + 1e-9):
            impossible += 1
            continue
        rows.append({
            "task_id": measure.task_id,
            "family": measure.family or family_of(measure.task_id),
            "stall": stall,
            "occ_deficit": 1.0 - occupancy,
            "cand_ms": candidate,
            "residual_ms": max(residual, 0.0),
            "gap": max(0.0, min(1.0, residual / candidate)),
        })
    return rows, impossible


def _normalized_cv_r2(rows: list[dict]) -> Optional[float]:
    return _cluster_cv(
        rows,
        lambda row: [row["stall"], row["occ_deficit"]],
        lambda row: row["gap"],
    )["r2"]


def _family_evidence(
    rows: list[dict], permutations: int, bootstrap: int, seed: int
) -> dict[str, dict]:
    families: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("family"):
            families.setdefault(str(row["family"]), []).append(row)
    out: dict[str, dict] = {}
    for index, (family, points) in enumerate(sorted(families.items())):
        family_seed = seed + index
        metric = lambda sample, fold_seed=family_seed: _cluster_cv(
            sample,
            lambda row: [row["stall"], row["occ_deficit"]],
            lambda row: row["gap"],
            seed=fold_seed,
        )["r2"]
        full = _cluster_cv(
            points,
            lambda row: [row["stall"], row["occ_deficit"]],
            lambda row: row["gap"],
            seed=family_seed,
        )
        baseline = _cluster_cv(
            points,
            lambda row: [math.log(row["cand_ms"])],
            lambda row: row["gap"],
            seed=seed + index,
        )
        perm = _permutation_test(
            points,
            metric,
            permutations,
            seed + 100 + index,
        )
        ci = _cluster_bootstrap(
            points, metric, bootstrap, seed + 200 + index
        )
        coef = _fit(
            [[row["stall"], row["occ_deficit"]] for row in points],
            [row["gap"] for row in points],
        )
        out[family] = {
            "n_points": len(points),
            "n_task_clusters": len({row["task_id"] for row in points}),
            "normalized_cv_r2": full["r2"],
            "baseline_cv_r2": baseline["r2"],
            "ci95": ci,
            "p_value": perm["p_value"],
            "p_adjusted": perm["p_value"],
            "coefficients": coef,
            "verdict": "FAIL",
        }
    return out


def check_b(
    measures: list[KernelMeasure],
    *,
    permutations: int = 200,
    bootstrap: int = 200,
    seed: int = 271,
) -> dict:
    rows, impossible = _counter_rows(measures)
    if len(rows) < 5:
        return {
            "r2": None,
            "n": len(rows),
            "verdict": "SKIP",
            "note": "need >=5 finite, physically valid counter points",
            "excluded_super_sol": impossible,
            "family_evidence": {},
        }
    normalized_design = lambda row: [row["stall"], row["occ_deficit"]]
    normalized_target = lambda row: row["gap"]
    raw_design = lambda row: [
        row["stall"] * row["cand_ms"],
        row["occ_deficit"] * row["cand_ms"],
    ]
    raw_target = lambda row: row["residual_ms"]
    tcand_design = lambda row: [row["cand_ms"]]
    normalized_tcand_design = lambda row: [math.log(row["cand_ms"])]

    raw_coef = _fit([raw_design(row) for row in rows], [raw_target(row) for row in rows])
    raw_r2 = _r2(
        [raw_target(row) for row in rows],
        _predict(raw_coef, [raw_design(row) for row in rows]),
    )
    raw_tcand_coef = _fit(
        [tcand_design(row) for row in rows], [raw_target(row) for row in rows]
    )
    raw_tcand_r2 = _r2(
        [raw_target(row) for row in rows],
        _predict(raw_tcand_coef, [tcand_design(row) for row in rows]),
    )
    normalized_coef = _fit(
        [normalized_design(row) for row in rows],
        [normalized_target(row) for row in rows],
    )
    normalized_in_sample = _r2(
        [normalized_target(row) for row in rows],
        _predict(normalized_coef, [normalized_design(row) for row in rows]),
    )
    normalized_cv = _cluster_cv(rows, normalized_design, normalized_target, seed=seed)
    tcand_cv = _cluster_cv(rows, normalized_tcand_design, normalized_target, seed=seed)
    intercept_cv = _cluster_cv(rows, lambda row: [], normalized_target, seed=seed)
    raw_cv = _cluster_cv(rows, raw_design, raw_target, seed=seed)
    raw_tcand_cv = _cluster_cv(rows, tcand_design, raw_target, seed=seed)

    normalized_metric = lambda sample: _cluster_cv(
        sample, normalized_design, normalized_target, seed=seed
    )["r2"]
    normalized_perm = _permutation_test(
        rows, normalized_metric, permutations, seed + 1
    )
    raw_perm = _permutation_test(
        rows,
        lambda sample: _r2(
            [raw_target(row) for row in sample],
            _predict(
                _fit(
                    [raw_design(row) for row in sample],
                    [raw_target(row) for row in sample],
                ),
                [raw_design(row) for row in sample],
            ),
        ),
        permutations,
        seed + 2,
    )
    normalized_ci = _cluster_bootstrap(
        rows, normalized_metric, bootstrap, seed + 3
    )

    # Leave one entire operator family out.
    lofo: dict[str, dict] = {}
    for family in sorted({row["family"] for row in rows if row.get("family")}):
        train = [row for row in rows if row["family"] != family]
        test = [row for row in rows if row["family"] == family]
        coef = _fit(
            [normalized_design(row) for row in train],
            [normalized_target(row) for row in train],
        )
        lofo[family] = {
            "n_test": len(test),
            "r2": _r2(
                [normalized_target(row) for row in test],
                _predict(coef, [normalized_design(row) for row in test]),
            ),
        }

    baseline_r2 = tcand_cv["r2"]
    increment = (
        normalized_cv["r2"] - baseline_r2
        if normalized_cv["r2"] is not None and baseline_r2 is not None
        else None
    )
    eligible = bool(
        len(rows) >= PREREGISTRATION["min_points"]
        and len({row["task_id"] for row in rows}) >= PREREGISTRATION["min_task_clusters"]
        and normalized_cv["r2"] is not None
        and normalized_cv["r2"] >= PREREGISTRATION["min_normalized_cv_r2"]
        and increment is not None
        and increment >= PREREGISTRATION["min_increment_over_baseline"]
        and normalized_ci is not None
        and normalized_ci[0] > 0.0
        and normalized_perm["p_value"] is not None
        and normalized_perm["p_value"] <= PREREGISTRATION["alpha"]
    )
    return {
        # Back-compat key, now explicitly labeled a non-primary diagnostic.
        "r2": raw_r2,
        "n": len(rows),
        "n_task_clusters": len({row["task_id"] for row in rows}),
        "excluded_super_sol": impossible,
        "raw_in_sample": {
            "named_r2": raw_r2,
            "tcand_only_r2": raw_tcand_r2,
            "named_task_cv_r2": raw_cv["r2"],
            "tcand_only_task_cv_r2": raw_tcand_cv["r2"],
            "denominator_preserving_null": raw_perm,
        },
        "normalized_primary": {
            "target": "(T_candidate-T_min)/T_candidate",
            "in_sample_r2": normalized_in_sample,
            "task_cluster_cv_r2": normalized_cv["r2"],
            "fold_r2": normalized_cv["fold_r2"],
            "tcand_only_cv_r2": tcand_cv["r2"],
            "intercept_only_cv_r2": intercept_cv["r2"],
            "increment_over_tcand": increment,
            "ci95_task_bootstrap": normalized_ci,
            "denominator_preserving_null": normalized_perm,
            "coefficients": normalized_coef,
        },
        "leave_family_out": lofo,
        "family_evidence": _family_evidence(
            rows, max(20, permutations // 2), bootstrap, seed + 50
        ),
        "p_value": normalized_perm["p_value"],
        "p_adjusted": normalized_perm["p_value"],
        "_eligible": eligible,
        "verdict": "PASS" if eligible else "FAIL",
    }


def _dominant_residual(m: KernelMeasure) -> float:
    """Dominant named residual component: max(stall fraction, occupancy deficit)."""
    return max(m.stall_frac or 0.0, 1.0 - (m.occupancy if m.occupancy is not None else 1.0))


def check_c(
    per_task: dict[str, list[KernelMeasure]],
    *,
    bootstrap: int = 200,
    seed: int = 911,
) -> dict:
    flat_tol = 0.10  # wall-clock "flat" if |d wall|/wall < 10%
    in_valley = 0
    monotone = 0
    tasks_used = 0
    for tid, ms in per_task.items():
        traj = [m for m in ms if m.correct and m.cand_ms and m.stall_frac is not None]
        if len(traj) < 3:
            continue
        # Preserve preregistered collection order (seed -> candidate trajectory).
        # Sorting by eta would define "improvement" using the outcome under test.
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
    p_value = sum(
        math.comb(in_valley, successes) for successes in range(monotone, in_valley + 1)
    ) / (2.0 ** in_valley)

    rng = random.Random(seed)
    keys = sorted(per_task)
    boot: list[float] = []
    if bootstrap >= 20 and keys:
        for _ in range(bootstrap):
            sample = {
                f"{key}#{index}": per_task[key]
                for index, key in enumerate(rng.choice(keys) for _ in keys)
            }
            result = check_c(sample, bootstrap=0, seed=seed)
            if result.get("frac") is not None:
                boot.append(result["frac"])
    ci = _percentile_ci(boot)
    eligible = bool(
        in_valley >= PREREGISTRATION["check_c_min_pairs"]
        and frac >= PREREGISTRATION["check_c_min_fraction"]
        and ci is not None
        and ci[0] > 0.5
        and p_value <= PREREGISTRATION["alpha"]
    )
    return {
        "frac": frac,
        "in_valley_pairs": in_valley,
        "tasks": tasks_used,
        "ci95_task_bootstrap": ci,
        "p_value": p_value,
        "p_adjusted": p_value,
        "_eligible": eligible,
        "verdict": "PASS" if eligible else "FAIL",
    }


def _bh_adjust(p_values: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank in range(total, 0, -1):
        key, p_value = ordered[rank - 1]
        running = min(running, p_value * total / rank)
        adjusted[key] = min(1.0, running)
    return adjusted


def apply_multiple_testing(a: dict, b: dict, c: dict) -> None:
    tests: list[tuple[str, float]] = []
    for key, check in (("a", a), ("b", b), ("c", c)):
        if _number(check.get("p_value")):
            tests.append((key, float(check["p_value"])))
    for family, evidence in (b.get("family_evidence") or {}).items():
        if _number(evidence.get("p_value")):
            tests.append((f"family:{family}", float(evidence["p_value"])))
    adjusted = _bh_adjust(tests) if tests else {}
    alpha = float(PREREGISTRATION["alpha"])
    for key, check in (("a", a), ("b", b), ("c", c)):
        check["p_adjusted"] = adjusted.get(key)
        if check.get("_eligible") and check["p_adjusted"] is not None and check["p_adjusted"] <= alpha:
            check["verdict"] = "PASS"
        elif check.get("verdict") != "SKIP":
            check["verdict"] = "FAIL"
    for family, evidence in (b.get("family_evidence") or {}).items():
        evidence["p_adjusted"] = adjusted.get(f"family:{family}")
        ci = evidence.get("ci95")
        cv = evidence.get("normalized_cv_r2")
        baseline = evidence.get("baseline_cv_r2")
        evidence["verdict"] = "PASS" if (
            evidence.get("n_points", 0) >= 20
            and evidence.get("n_task_clusters", 0) >= 3
            and cv is not None
            and baseline is not None
            and cv >= 0.10
            and cv - baseline >= 0.05
            and ci is not None
            and ci[0] > 0.0
            and evidence["p_adjusted"] is not None
            and evidence["p_adjusted"] <= alpha
        ) else "FAIL"


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
    """Compatibility summary of the new task-cluster bootstraps."""
    a = check_a_rigorous(all_measures, permutations=0, bootstrap=B, seed=seed)
    b = check_b(all_measures, permutations=0, bootstrap=B, seed=seed + 1)
    c = check_c(per_task, bootstrap=B, seed=seed + 2)
    return (
        a.get("rho_ci95_task_bootstrap"),
        (b.get("normalized_primary") or {}).get("ci95_task_bootstrap"),
        c.get("ci95_task_bootstrap"),
    )


def decide(a: dict, b: dict, c: dict, dry_run: bool) -> str:
    if dry_run:
        return "DRY_RUN"
    av, bv, cv = a["verdict"], b["verdict"], c["verdict"]
    if av == "PASS" and bv == "PASS" and cv == "PASS":
        return "GO"
    if av == "PASS" and bv == "PASS":
        return "EVIDENCE_PARTIAL"
    if av == "SKIP" or bv == "SKIP":
        return "INSUFFICIENT_DATA"
    return "INTEGRITY_ONLY"


def _clean_json(value):
    if isinstance(value, dict):
        return {
            key: _clean_json(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_json(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _fingerprint_payload(payload: dict) -> str:
    encoded = json.dumps(
        _clean_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _measure_from_dict(raw: dict) -> Optional[KernelMeasure]:
    fields = set(KernelMeasure.__dataclass_fields__)
    try:
        return KernelMeasure(**{key: value for key, value in raw.items() if key in fields})
    except (TypeError, ValueError):
        return None


def reanalyze_report(
    report: dict,
    *,
    permutations: int = 1000,
    bootstrap: int = 1000,
    seed: int = 20260723,
) -> dict:
    """Apply the preregistered controls to an existing measurement report."""
    measures = [
        measure
        for measure in (
            _measure_from_dict(raw) for raw in (report.get("measures") or [])
        )
        if measure is not None
    ]
    per_task: dict[str, list[KernelMeasure]] = {}
    for measure in measures:
        shape_id = measure.shape_id
        if not shape_id and "@" in measure.label:
            shape_id = measure.label.rsplit("@", 1)[-1]
        per_task.setdefault(f"{measure.task_id}@{shape_id or 'unknown'}", []).append(measure)
    a = check_a_rigorous(
        measures, permutations=permutations, bootstrap=bootstrap, seed=seed
    )
    b = check_b(
        measures, permutations=permutations, bootstrap=bootstrap, seed=seed + 1
    )
    c = check_c(per_task, bootstrap=bootstrap, seed=seed + 2)
    apply_multiple_testing(a, b, c)
    checks = _clean_json({"a": a, "b": b, "c": c})

    model = report.get("model") or {}
    model_fingerprint = str(
        model.get("fingerprint")
        or next(
            (
                measure.model_fingerprint
                for measure in measures
                if measure.model_fingerprint
            ),
            "",
        )
    )
    fingerprint_status = "verified" if model_fingerprint.startswith("sha256:") else "legacy-unfingerprinted"
    analysis_fingerprint = _fingerprint_payload({
        "schema": PREREGISTRATION["schema"],
        "model_fingerprint": model_fingerprint or None,
        "measures": report.get("measures") or [],
        "checks": checks,
        "preregistration": PREREGISTRATION,
    })

    # Only passing families with complete held-out statistics become deployable
    # shaping evidence. Legacy reports cannot authorize shaping.
    deployable: dict[str, dict] = {}
    if fingerprint_status == "verified":
        for family, evidence in (checks["b"].get("family_evidence") or {}).items():
            if evidence.get("verdict") != "PASS":
                continue
            if evidence.get("ci95") is None or evidence.get("coefficients") is None:
                continue
            deployable[family] = {
                **evidence,
                "family": family,
                "report_fingerprint": analysis_fingerprint,
                "model_fingerprint": model_fingerprint,
            }

    updated = dict(report)
    updated.update({
        "validation_schema": PREREGISTRATION["schema"],
        "preregistration": dict(PREREGISTRATION),
        "model_fingerprint_status": fingerprint_status,
        "checks": checks,
        "decision": decide(a, b, c, bool(report.get("dry_run"))),
        "analysis_fingerprint": analysis_fingerprint,
        "shaping_evidence": {
            "status": "available" if deployable else "disabled",
            "families": deployable,
            "unsupported_families": sorted(
                (checks["b"].get("family_evidence") or {}).keys()
            ),
        },
        "validation_resamples": {
            "permutations": permutations,
            "task_cluster_bootstrap": bootstrap,
            "seed": seed,
        },
    })
    updated["evidence_fingerprint"] = _fingerprint_payload(updated)
    return _clean_json(updated)


# --------------------------------------------------------------------------- #
# dry-run: roofline table + eta from cached replay (no GPU)
# --------------------------------------------------------------------------- #
def _mine_replay_eta(task, peaks: dict, arch: str, replay_dir: Path,
                     model: Optional[PhysicalModel] = None) -> list[KernelMeasure]:
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
        rf = roofline(
            task.task_id, task.operation, task.dtype, shape_str, dims, peaks, arch,
            model=model)
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
            model_fingerprint=(model.fingerprint if model else rf.model_fingerprint),
            shape_id=shape_str,
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
    model = report.get("model") or {}
    arch = model.get("architecture") or report.get("arch", "unknown")
    sku = model.get("sku") or "unidentified"
    fingerprint = model.get("fingerprint") or "legacy-unfingerprinted"
    L.append(
        f"# P0 leakage-controlled validation -- {sku}/{arch} "
        f"{'(DRY RUN)' if report.get('dry_run') else ''}"
    )
    L.append(f"# model: {fingerprint}")
    L.append(f"# preregistration: {report.get('validation_schema', PREREGISTRATION['schema'])}")

    p = report.get("peaks") or {}
    if _number(p.get("hbm_bytes_per_s"), positive=True):
        pieces = [f"HBM {p['hbm_bytes_per_s']/1e12:.3f} TB/s"]
        for dtype in ("bf16", "fp8"):
            key = f"{dtype}_flops_per_s"
            if _number(p.get(key), positive=True):
                pieces.append(f"{dtype} {p[key]/1e15:.3f} PF/s")
        L.append("# peaks: " + " | ".join(pieces))

    L.append("")
    hdr = f"{'task':26s} {'dtype':6s} {'bound':7s} {'AI(F/B)':>9s} {'T_min':>10s}"
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in report.get("rooflines") or []:
        L.append(f"{r['task_id']:26s} {r['dtype']:6s} {r['bound']:7s} "
                 f"{r['arithmetic_intensity']:9.2f} {r['t_min_ms']*1e3:8.1f}us")
    if report.get("unmodeled"):
        L.append("\nunavailable (unsupported model): " + ", ".join(report["unmodeled"]))

    a, b, c = report["checks"]["a"], report["checks"]["b"], report["checks"]["c"]
    primary = b.get("normalized_primary") or {}
    raw = b.get("raw_in_sample") or {}
    L.append("\n" + "=" * 60)
    L.append(
        "(a) roofline beyond Tcand: "
        f"rho={_fmt(a.get('rho'))}, Tcand={_fmt(a.get('tcand_only_rho'))}, "
        f"delta={_fmt(a.get('increment_over_tcand'))}, "
        f"q={_fmt(a.get('p_adjusted'))} -> {a['verdict']}"
    )
    L.append(
        "(b) normalized held-out: "
        f"R2={_fmt(primary.get('task_cluster_cv_r2'))}, "
        f"Tcand baseline={_fmt(primary.get('tcand_only_cv_r2'))}, "
        f"q={_fmt(b.get('p_adjusted'))} -> {b['verdict']}"
    )
    L.append(
        "    raw in-sample diagnostic: "
        f"named R2={_fmt(raw.get('named_r2'))}, "
        f"Tcand-only R2={_fmt(raw.get('tcand_only_r2'))}, "
        f"null median={_fmt((raw.get('denominator_preserving_null') or {}).get('null_median'))}"
    )
    L.append(
        "(c) collection-order valley: "
        f"frac={_fmt(c.get('frac'))}, pairs={c.get('in_valley_pairs')}, "
        f"q={_fmt(c.get('p_adjusted'))} -> {c['verdict']}"
    )
    L.append(f"DECISION: {report['decision']}")
    shaping = report.get("shaping_evidence") or {}
    L.append(
        "SHAPING: "
        + (
            "enabled for " + ", ".join(sorted((shaping.get("families") or {}).keys()))
            if shaping.get("families")
            else "disabled; no family passed held-out evidence"
        )
    )
    L.append("=" * 60)
    for note in (a.get("note"), b.get("note"), c.get("note")):
        if note:
            L.append(f"  note: {note}")
    return "\n".join(L)


def run(tasks: list[str], arch: str, peaks: dict, warmup: int, iters: int,
        max_kernels: int, device: str, dry_run: bool, do_pmc: bool,
        replay_dir: Path, shapes_per_task: int = 1, reseeds: int = 1,
        bootstrap: int = 0, permutations: int = 200,
        model: Optional[PhysicalModel] = None) -> dict:
    from kore.tasks.registry import all_tasks, get_task

    if model is None:
        sku = str(peaks.get("sku") or ("mi350x" if arch == "gfx950" else "mi300x")).lower()
        model = model_from_peak_mapping(
            peaks,
            sku=sku,
            source=str(peaks.get("calibration_source") or "legacy-explicit-mapping"),
        )
    if model.architecture != arch:
        raise ModelError(
            f"run architecture {arch!r} does not match model {model.architecture!r}")
    task_objs = [get_task(t) for t in tasks] if tasks else all_tasks()

    rooflines: list[dict] = []
    unmodeled: list[str] = []
    all_measures: list[KernelMeasure] = []
    per_task: dict[str, list[KernelMeasure]] = {}

    for t in task_objs:
        prim = t.shape("primary") or (t.shapes[0] if t.shapes else None)
        if prim is None:
            continue
        rf = roofline(
            t.task_id, t.operation, t.dtype, shape_to_str(prim.dims), prim.dims,
            peaks, arch, model=model)
        if rf is None:
            unmodeled.append(f"{t.task_id} ({t.operation})")
            continue
        rooflines.append(asdict(rf))

        if dry_run:
            ms = _mine_replay_eta(t, peaks, arch, replay_dir, model=model)
            per_task[t.task_id] = ms
            all_measures.extend(ms)
        else:
            shapes = _select_shapes(t, shapes_per_task)
            for label, src in trajectory_sources(t, max_kernels):
                for shp in shapes:
                    m = measure_kernel(t, f"{label}@{shp.name}", src, shp, peaks, arch,
                                       warmup, iters, device, do_pmc, reseeds=reseeds,
                                       model=model)
                    # one trajectory per (task, shape) keeps check (c)'s flat-wall
                    # ordering coherent; check (a)/(b) pool all points.
                    per_task.setdefault(f"{t.task_id}@{shp.name}", []).append(m)
                    all_measures.append(m)

    base_report = {
        "arch": arch,
        "model": model.as_dict(),
        "peaks": {
            "hbm_bytes_per_s": model.hbm_bytes_per_s,
            **{
                f"{dtype}_flops_per_s": value
                for dtype, value in model.compute_flops_per_s.items()
            },
        },
        "dry_run": dry_run,
        "shapes_per_task": shapes_per_task,
        "reseeds": reseeds,
        "bootstrap": bootstrap,
        "rooflines": rooflines, "unmodeled": unmodeled,
        "measures": [asdict(m) for m in all_measures],
    }
    return reanalyze_report(
        base_report,
        permutations=permutations,
        bootstrap=bootstrap,
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P0 roofline/SOL falsification test")
    ap.add_argument("--tasks", default=None, help="comma-separated task ids (default: all)")
    ap.add_argument("--sku", default="mi350x", help="explicit hardware SKU")
    ap.add_argument("--arch", default=None, help="optional consistency check (e.g. gfx950)")
    ap.add_argument("--calibration", default=None,
                    help="fingerprint-safe kore.runtime-calibration.v1 JSON")
    ap.add_argument("--expect-model-fingerprint", default=None)
    ap.add_argument("--reanalyze", default=None,
                    help="CPU-only: apply controls to an existing P0 JSON")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-kernels-per-task", type=int, default=8, dest="max_kernels")
    ap.add_argument("--shapes-per-task", type=int, default=1, dest="shapes_per_task",
                    help="measure each operator at up to N shapes (>=3 for CI)")
    ap.add_argument("--reseeds", type=int, default=1,
                    help="repeat each (kernel,shape) timing N times (median-of-medians)")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="task-cluster bootstrap resamples (preregistered: 1000)")
    ap.add_argument("--permutations", type=int, default=1000,
                    help="denominator-preserving null permutations (preregistered: 1000)")
    ap.add_argument("--device", default="0", help="HIP_VISIBLE_DEVICES")
    ap.add_argument("--dry-run", action="store_true", help="CPU-only: roofline + cached eta")
    ap.add_argument("--no-pmc", action="store_true", help="skip rocprofv3 PMC (checks b/c weaker)")
    ap.add_argument("--replay-dir", default="runs")
    ap.add_argument("--out", default="runs/p0_report.json")
    args = ap.parse_args(argv)

    if args.reanalyze:
        source = json.loads(Path(args.reanalyze).read_text())
        report = reanalyze_report(
            source,
            permutations=args.permutations,
            bootstrap=args.bootstrap,
        )
    else:
        model = make_physical_model(
            args.sku,
            args.calibration,
            expected_fingerprint=args.expect_model_fingerprint,
        )
        if args.arch and args.arch != model.architecture:
            ap.error(
                f"--arch {args.arch!r} conflicts with {model.sku}/{model.architecture}")
        arch = model.architecture
        peaks = {
            "hbm_bytes_per_s": model.hbm_bytes_per_s,
            "sku": model.sku,
            "architecture": model.architecture,
            "calibration_source": model.calibration_source,
            "calibration_id": model.calibration_id,
            **{
                f"{dtype}_flops_per_s": value
                for dtype, value in model.compute_flops_per_s.items()
            },
        }
        tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else []
        report = run(
            tasks, arch, peaks, args.warmup, args.iters, args.max_kernels,
            args.device, args.dry_run, not args.no_pmc, Path(args.replay_dir),
            shapes_per_task=args.shapes_per_task, reseeds=args.reseeds,
            bootstrap=args.bootstrap, permutations=args.permutations, model=model)
    text = render(report)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[p0_sol] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
