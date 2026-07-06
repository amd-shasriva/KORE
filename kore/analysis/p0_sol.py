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
    stall_frac: Optional[float] = None
    mem_frac: Optional[float] = None
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
           env: dict, timeout: int) -> Optional[float]:
    cmd = [sys.executable, str(driver), "--bench-mode", "--impl", impl,
           "--warmup", str(warmup), "--iters", str(iters), *shape_args]
    rc, out, timed = _exec(cmd, driver.parent, env, timeout)
    if timed or rc != 0:
        return None
    m = _last(_MEDIAN, out)
    return float(m.group(1)) if m else None


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


def _profile_pmc(driver: Path, shape_args: list[str], counters: list[str],
                 env: dict, timeout: int) -> dict:
    """Run the candidate under rocprofv3 --pmc; return aggregated counter dict.
    Best-effort: any failure returns {} so checks (b)/(c) degrade gracefully."""
    if not shutil.which("rocprofv3"):
        return {}
    outdir = Path(tempfile.mkdtemp(prefix="p0_pmc_"))
    cmd = ["rocprofv3", "--pmc", *counters, "-d", str(outdir), "--output-format", "csv",
           "--", sys.executable, str(driver), "--bench-mode", "--impl", "candidate",
           "--warmup", "2", "--iters", "5", *shape_args]
    try:
        rc, out, timed = _exec(cmd, driver.parent, env, timeout)
        if timed or rc != 0:
            return {}
        try:
            from kore.verifier.parsers.rocprofv3 import parse_rocprofv3_csv
        except Exception:  # noqa: BLE001
            return {}
        agg: dict[str, float] = {}
        import glob as _glob
        csvs = _glob.glob(str(outdir / "*.csv")) + _glob.glob(str(outdir / "**/*.csv"), recursive=True)
        for cp in csvs:
            try:
                for k in parse_rocprofv3_csv(cp):
                    for name, val in k.counters.items():
                        agg[name] = agg.get(name, 0.0) + float(val)
            except Exception:  # noqa: BLE001
                continue
        return agg
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def _decompose(counters: dict) -> tuple[Optional[float], Optional[float]]:
    """First-order stall / memory fractions from SQ counters (0..1). None if absent.

    stall_frac ~ wait cycles vs issued work; mem_frac ~ vmem vs (vmem+mfma). These
    are the counter-derived regressors for the residual decomposition (check b).
    Deliberately simple + documented; refine with derived occupancy metrics later.
    """
    def c(*names):
        for n in names:
            if n in counters:
                return float(counters[n])
        return 0.0

    wait_any = c("SQ_WAIT_INST_ANY")
    wait_lds = c("SQ_WAIT_INST_LDS")
    wait_vmem = c("SQ_WAIT_INST_VMEM")
    mfma = c("SQ_INSTS_VALU_MFMA_BF16", "SQ_INSTS_VALU_MFMA_F16", "SQ_INSTS_VALU_MFMA_F32")
    valu = c("SQ_INSTS_VALU")
    vmem = c("SQ_INSTS_VMEM")
    work = mfma + valu + vmem
    wait = wait_any + wait_lds + wait_vmem
    stall_frac = (wait / (wait + work)) if (wait + work) > 0 else None
    mem_frac = (vmem / (vmem + mfma)) if (vmem + mfma) > 0 else None
    return stall_frac, mem_frac


def measure_kernel(task, label: str, source: str, shape, peaks: dict, arch: str,
                   warmup: int, iters: int, device: str, do_pmc: bool,
                   timeout: int = 300) -> KernelMeasure:
    dims = shape.dims
    rf = roofline(task.task_id, task.operation, task.dtype, shape_to_str(dims), dims, peaks, arch)
    t_min = rf.t_min_ms if rf else float("nan")
    wd = _stage(task, source)
    env = _proc_env(device, arch)
    try:
        driver = wd / "driver.py"
        snr, ac, err = _correctness(driver, shape.as_args(), env, timeout)
        correct = bool(ac) and (snr is not None) and (snr >= float(getattr(task, "snr_threshold", 25.0)))
        cand_ms = _bench(driver, shape.as_args(), "candidate", warmup, iters, env, timeout) if correct else None
        vendor_ms = _bench(driver, shape.as_args(), "reference", warmup, iters, env, timeout)
        counters = {}
        stall_frac = mem_frac = None
        if do_pmc and correct and cand_ms:
            from kore.verifier.pmc import COUNTER_SETS
            counters = _profile_pmc(driver, shape.as_args(), COUNTER_SETS["full"], env, timeout)
            stall_frac, mem_frac = _decompose(counters)
        eta = (t_min / cand_ms) if (cand_ms and t_min == t_min) else None
        speedup = (vendor_ms / cand_ms) if (vendor_ms and cand_ms) else None
        residual = (cand_ms - t_min) if (cand_ms and t_min == t_min) else None
        return KernelMeasure(
            task_id=task.task_id, label=label, correct=correct, snr_db=snr,
            cand_ms=cand_ms, vendor_ms=vendor_ms, t_min_ms=t_min, eta=eta,
            speedup=speedup, residual_ms=residual, counters=counters,
            stall_frac=stall_frac, mem_frac=mem_frac, error=err if not correct else None,
        )
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# --------------------------------------------------------------------------- #
# trajectory assembly: seed + cached group candidates (for check c ordering)
# --------------------------------------------------------------------------- #
def trajectory_sources(task, max_kernels: int) -> list[tuple[str, str]]:
    """(label, source) list: seed first, then distinct cached group candidates."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
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
    return out[:max_kernels]


# --------------------------------------------------------------------------- #
# the three checks
# --------------------------------------------------------------------------- #
def check_a(measures: list[KernelMeasure]) -> dict:
    pts = [(m.eta, m.speedup) for m in measures if m.eta and m.speedup]
    if len(pts) < 3:
        return {"rho": None, "n": len(pts), "verdict": "SKIP",
                "note": "need >=3 kernels with both eta and vendor speedup"}
    rho = spearman([p[0] for p in pts], [p[1] for p in pts])
    verdict = "PASS" if (rho is not None and rho >= 0.5 and len(pts) >= 5) else "WEAK"
    return {"rho": rho, "n": len(pts), "verdict": verdict}


def check_b(measures: list[KernelMeasure]) -> dict:
    rows = [(m.stall_frac, m.mem_frac, m.residual_ms) for m in measures
            if m.stall_frac is not None and m.mem_frac is not None and m.residual_ms is not None]
    if len(rows) < 5:
        return {"r2": None, "n": len(rows), "verdict": "SKIP",
                "note": "need >=5 kernels with PMC counters + residual"}
    X = [[r[0], r[1]] for r in rows]
    y = [r[2] for r in rows]
    r2 = ols_r2(X, y)
    if r2 is None:
        return {"r2": None, "n": len(rows), "verdict": "WEAK", "note": "numpy absent / under-determined"}
    verdict = "PASS" if r2 >= 0.7 else "WEAK"
    return {"r2": r2, "n": len(rows), "verdict": verdict}


def check_c(per_task: dict[str, list[KernelMeasure]]) -> dict:
    flat_tol = 0.10  # wall-clock "flat" if |d wall|/wall < 10%
    in_valley = 0
    monotone = 0
    tasks_used = 0
    for tid, ms in per_task.items():
        traj = [m for m in ms if m.correct and m.cand_ms and m.stall_frac is not None]
        if len(traj) < 3:
            continue
        traj.sort(key=lambda m: (m.speedup or 0.0))  # ascending "improvement"
        tasks_used += 1
        for a, b in zip(traj, traj[1:]):
            dwall = abs((b.cand_ms - a.cand_ms) / a.cand_ms) if a.cand_ms else 1.0
            if dwall < flat_tol:
                in_valley += 1
                if (b.stall_frac or 0) < (a.stall_frac or 0):  # dominant residual term falls
                    monotone += 1
    if in_valley < 3:
        return {"frac": None, "in_valley_pairs": in_valley, "tasks": tasks_used,
                "verdict": "SKIP", "note": "need >=3 flat-wall adjacent pairs across tasks"}
    frac = monotone / in_valley
    verdict = "PASS" if frac >= 0.6 else "WEAK"
    return {"frac": frac, "in_valley_pairs": in_valley, "tasks": tasks_used, "verdict": verdict}


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
    L.append("\n" + "=" * 60)
    L.append(f"(a) eta predicts speedup   : rho={_fmt(a.get('rho'))} (n={a.get('n')})   -> {a['verdict']}")
    L.append(f"(b) residual decomp R^2    : {_fmt(b.get('r2'))} (n={b.get('n')})        -> {b['verdict']}")
    L.append(f"(c) monotone-in-valley frac: {_fmt(c.get('frac'))} "
             f"(pairs={c.get('in_valley_pairs')})  -> {c['verdict']}")
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
        replay_dir: Path) -> dict:
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
        else:
            ms = []
            for label, src in trajectory_sources(t, max_kernels):
                m = measure_kernel(t, label, src, prim, peaks, arch, warmup, iters, device, do_pmc)
                ms.append(m)
        per_task[t.task_id] = ms
        all_measures.extend(ms)

    a = check_a(all_measures)
    b = check_b(all_measures)
    c = check_c(per_task)
    decision = decide(a, b, c, dry_run)
    return {
        "arch": arch, "peaks": peaks, "dry_run": dry_run,
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
                 args.device, args.dry_run, not args.no_pmc, Path(args.replay_dir))
    text = render(report)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[p0_sol] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
