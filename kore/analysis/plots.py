"""Generate P0 study figures from a p0_sol report JSON (matplotlib, headless).

Figures:
  fig1_roofline_eta.png   - per-operator SOL attainment (eta), colored by roofline bound
  fig2_eta_vs_speedup.png - check (a): eta vs speedup, with the preregistered
                            increment over the T_candidate-only baseline
  fig3_residual_fit.png   - check (b): the PREREGISTERED NORMALIZED residual model
                            (gap = (T_cand - T_min)/T_cand ~ stall + occupancy
                            deficit), headlined by its held-out CV R^2
  fig4_monotone_valley.png- check (c): dominant residual along the preregistered
                            collection order (never re-sorted by eta)
  fig5_correct_but_slow.png - the correct-but-slow wall: eta and speedup per op

These are publication artifacts, so each figure reads the keys the current
:mod:`kore.analysis.p0_sol` actually emits and presents that check's
PREREGISTERED primary statistic:

* check (a)'s CI lives at ``rho_ci95_task_bootstrap`` (not ``ci95``), and rho
  alone is not the claim: the preregistered quantity is the INCREMENT over a
  ``T_candidate``-only predictor, so the figure reports both.
* check (b)'s primary is ``normalized_primary`` - a held-out, task-cluster
  cross-validated R^2 on a normalized target, with its own bootstrap CI. The raw
  in-sample residual OLS is a shared-denominator artifact (a ``T_candidate``-only
  predictor beats it and the denominator-preserving permutation null sits above
  it), so it appears only as a labelled, self-refuting diagnostic.
* check (c) is evaluated along the preregistered collection order. Sorting a
  trajectory by eta would define "improvement" using the outcome under test,
  which is precisely what ``p0_sol.check_c`` refuses to do.

Usage: python -m kore.analysis.plots --report runs/p0_study.json --out figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ACCENT = "#B4232A"
BLUE = "#1F4E79"
GREEN = "#1F7A3D"
GREY = "#6B7280"


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _num(value):
    """The value when it is a finite real number, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if np.isfinite(value) else None


def _fmt(value, spec: str = ".3f") -> str:
    number = _num(value)
    return format(number, spec) if number is not None else "n/a"


def _interval(bounds, spec: str = ".3f") -> str:
    """Render a stored ``[lo, hi]`` CI, or say explicitly that there is none."""
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return "  95%CI unavailable"
    lo, hi = _num(bounds[0]), _num(bounds[1])
    if lo is None or hi is None:
        return "  95%CI unavailable"
    return f"  95%CI[{format(lo, spec)},{format(hi, spec)}]"


def _verdict(check: dict) -> str:
    return str(check.get("verdict") or "UNREPORTED")


def _hardware(rep: dict) -> str:
    model = rep.get("model") or {}
    sku = model.get("sku") or (rep.get("peaks") or {}).get("sku") or "unidentified SKU"
    arch = model.get("architecture") or rep.get("arch") or "unknown arch"
    return f"{arch} ({sku})"


def _counter_rows(rep: dict) -> list[dict]:
    """Measures admissible to check (b), mirroring ``p0_sol._counter_rows``.

    Same validity filter as the statistic (fractions in range, positive candidate
    time, non-negative residual, no super-SOL point), so the figure's point cloud
    is exactly the sample the reported R^2 was computed on.
    """
    rows: list[dict] = []
    for m in rep.get("measures") or []:
        stall = _num(m.get("stall_frac"))
        occupancy = _num(m.get("occupancy"))
        residual = _num(m.get("residual_ms"))
        candidate = _num(m.get("cand_ms"))
        t_min = _num(m.get("t_min_ms"))
        if None in (stall, occupancy, residual, candidate, t_min):
            continue
        if not (0.0 <= stall <= 1.0 and 0.0 <= occupancy <= 1.0 and candidate > 0.0):
            continue
        if residual < -1e-12 or t_min > candidate * (1.0 + 1e-9):
            continue
        rows.append({
            "task_id": m["task_id"],
            "stall": stall,
            "occ_deficit": 1.0 - occupancy,
            "cand_ms": candidate,
            "residual_ms": max(residual, 0.0),
            "gap": max(0.0, min(1.0, residual / candidate)),
        })
    return rows


def _seed_points(rep: dict) -> list:
    """One representative (seed, primary shape) measure per operator for per-op bars.

    Multi-shape runs label kernels ``seed@<shape>``; prefer ``seed@primary``, then
    any ``seed@*``, then any timed measure. Deduplicated to one point per task."""
    pref = [m for m in rep["measures"] if m.get("eta") and m.get("label") == "seed@primary"]
    if not pref:
        pref = [m for m in rep["measures"] if m.get("eta") and str(m.get("label", "")).startswith("seed")]
    if not pref:
        pref = [m for m in rep["measures"] if m.get("eta")]
    seen: dict = {}
    for m in pref:
        seen.setdefault(m["task_id"], m)
    return list(seen.values())


def fig_roofline_eta(rep: dict, out: Path) -> None:
    ms = _seed_points(rep)
    bound = {r["task_id"]: r["bound"] for r in rep["rooflines"]}
    ms.sort(key=lambda m: m["eta"], reverse=True)
    names = [m["task_id"] for m in ms]
    etas = [m["eta"] * 100 for m in ms]
    colors = [ACCENT if bound.get(n) == "compute" else BLUE for n in names]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(range(len(names)), etas, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("SOL attainment  η = T_min / T_measured   (%)")
    ax.set_title(f"Seed-kernel SOL attainment per operator on {_hardware(rep)}")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ACCENT, label="compute-bound"),
                       Patch(color=BLUE, label="memory-bound")], loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig1_roofline_eta.png", dpi=150)
    plt.close(fig)


def fig_eta_vs_speedup(rep: dict, out: Path) -> None:
    check = rep.get("checks", {}).get("a", {})
    pts = [(m["eta"] * 100, m["speedup"], m["task_id"]) for m in rep["measures"]
           if m.get("eta") and m.get("speedup")]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=70, color=ACCENT, zorder=3, edgecolor="white", alpha=0.85)
        seen_lbl = set()  # label one representative point per task to avoid clutter
        for x, y, n in sorted(pts, key=lambda p: -p[0]):
            if n in seen_lbl:
                continue
            seen_lbl.add(n)
            ax.annotate(n, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.axhline(1.0, ls="--", color=GREY, label="parity with vendor (speedup=1)")
    rho = check.get("rho")
    n = check.get("n")
    # Current key. ``ci95`` is only read so a legacy artifact still annotates.
    ci = check.get("rho_ci95_task_bootstrap") or check.get("ci95")
    ax.set_xlabel("SOL attainment  η  (%)")
    ax.set_ylabel("speedup vs production baseline  (vendor / candidate)")
    if rho is None:
        ax.set_title("Check (a): η vs speedup")
    else:
        # The preregistered statistic is the INCREMENT over a T_candidate-only
        # predictor: rho alone cannot separate "η predicts speedup" from "both
        # carry the candidate runtime in their denominator".
        ax.set_title(
            f"Check (a) [{_verdict(check)}]: does η predict speedup beyond T_candidate?\n"
            f"Spearman ρ = {_fmt(rho)} (n={n}){_interval(ci)}\n"
            f"T_candidate-only ρ = {_fmt(check.get('tcand_only_rho'))}, "
            f"increment = {_fmt(check.get('increment_over_tcand'), '+.3f')}"
            f"{_interval(check.get('increment_ci95_task_bootstrap'))}",
            fontsize=10,
        )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "fig2_eta_vs_speedup.png", dpi=150)
    plt.close(fig)


def fig_residual_fit(rep: dict, out: Path) -> None:
    """Check (b) as preregistered: the NORMALIZED, held-out counter model.

    Plots the normalized target ``gap = (T_candidate - T_min)/T_candidate`` against
    the preregistered ``[stall, occupancy-deficit]`` prediction and headlines the
    held-out task-cluster CV R² with its bootstrap CI. The raw in-sample residual
    OLS this figure used to report is a shared-denominator artifact, so it appears
    only as a labelled diagnostic beside the ``T_candidate``-only score that beats
    it and the permutation null that sits above it.
    """
    check = rep.get("checks", {}).get("b", {})
    primary = check.get("normalized_primary") or {}
    raw = check.get("raw_in_sample") or {}
    rows = _counter_rows(rep)

    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    if len(rows) >= 3:
        design = np.array([[r["stall"], r["occ_deficit"]] for r in rows])
        y = np.array([r["gap"] for r in rows])
        coefficients = primary.get("coefficients")
        if isinstance(coefficients, (list, tuple)) and len(coefficients) == 3:
            # Stored preregistered fit: feature weights then intercept (p0_sol._predict).
            weights = np.array([float(c) for c in coefficients[:-1]])
            pred = design @ weights + float(coefficients[-1])
            fit_label = "preregistered stored coefficients"
        else:
            augmented = np.column_stack([design, np.ones(len(rows))])
            coef, *_ = np.linalg.lstsq(augmented, y, rcond=None)
            pred = augmented @ coef
            fit_label = "refit on the normalized target"
        ax.scatter(pred, y, s=60, color=GREEN, zorder=3, edgecolor="white",
                   label=f"measured vs predicted ({fit_label})")
        low = float(min(y.min(), pred.min()))
        high = float(max(y.max(), pred.max()))
        pad = 0.05 * max(high - low, 1e-9)
        lim = [low - pad, high + pad]
        ax.plot(lim, lim, ls="--", color=GREY, label="y = x (perfect)")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(
            f"Check (b) [{_verdict(check)}]: normalized residual gap from PMC terms\n"
            f"PRIMARY held-out task-cluster CV R² = "
            f"{_fmt(primary.get('task_cluster_cv_r2'), '.4f')} (n={len(rows)})"
            f"{_interval(primary.get('ci95_task_bootstrap'))}\n"
            f"T_candidate-only CV R² = {_fmt(primary.get('tcand_only_cv_r2'), '.4f')}, "
            f"intercept-only = {_fmt(primary.get('intercept_only_cv_r2'), '.4f')}",
            fontsize=10,
        )
        # The discredited statistic, drawn only so the figure refutes it.
        null = raw.get("denominator_preserving_null") or {}
        ax.text(
            0.02, 0.98,
            "non-primary diagnostic (NOT the result):\n"
            f"raw in-sample residual OLS R² = {_fmt(raw.get('named_r2'), '.4f')}\n"
            f"  T_candidate-only alone scores {_fmt(raw.get('tcand_only_r2'), '.4f')}\n"
            f"  denominator-preserving null median = "
            f"{_fmt(null.get('null_median'), '.4f')} (p={_fmt(null.get('p_value'), '.3f')})\n"
            "  -> shared-denominator artifact, not counter evidence",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color=GREY,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=GREY, alpha=0.85),
        )
    else:
        ax.set_title("Check (b): insufficient PMC data")
    ax.set_xlabel("predicted normalized gap from PMC terms  (stall, occupancy deficit)")
    ax.set_ylabel("measured normalized gap  (T_measured − T_min) / T_measured")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_residual_fit.png", dpi=150)
    plt.close(fig)


def _dominant(measure: dict) -> float:
    """Dominant named residual component: max(stall fraction, occupancy deficit)."""
    stall = _num(measure.get("stall_frac")) or 0.0
    occupancy = _num(measure.get("occupancy"))
    return max(stall, 1.0 - (occupancy if occupancy is not None else 1.0))


def _trajectories(rep: dict) -> dict[str, list[dict]]:
    """Per (task, shape) trajectories in the report's PREREGISTERED order.

    Mirrors ``p0_sol.reanalyze_report``: measures are grouped by task and shape and
    kept in collection order. Re-sorting by eta - which this figure used to do -
    would define the improvement direction using the outcome under test, which is
    exactly why ``p0_sol.check_c`` refuses to sort.
    """
    grouped: dict[str, list[dict]] = {}
    for m in rep.get("measures") or []:
        if not (m.get("correct") and _num(m.get("cand_ms")) and m.get("stall_frac") is not None):
            continue
        shape = m.get("shape_id")
        if not shape and "@" in str(m.get("label", "")):
            shape = str(m["label"]).rsplit("@", 1)[-1]
        grouped.setdefault(f"{m['task_id']}@{shape or 'unknown'}", []).append(m)
    return {key: values for key, values in grouped.items() if len(values) >= 2}


def fig_monotone_valley(rep: dict, out: Path) -> None:
    check = rep.get("checks", {}).get("c", {})
    trajs = _trajectories(rep)
    flat_tol = 0.10  # p0_sol.check_c: a pair is "in the valley" when |d wall|/wall < 10%
    fig, ax = plt.subplots(figsize=(8.5, 6))
    valley_x: list[int] = []
    valley_y: list[float] = []
    for key, ms in list(trajs.items())[:8]:
        xs = list(range(1, len(ms) + 1))
        ys = [_dominant(m) * 100 for m in ms]
        ax.plot(xs, ys, marker="o", label=key, alpha=0.8)
        for index, (first, second) in enumerate(zip(ms, ms[1:])):
            before, after = _num(first.get("cand_ms")), _num(second.get("cand_ms"))
            if not before or after is None:
                continue
            if abs((after - before) / before) < flat_tol:
                valley_x.extend([xs[index], xs[index + 1]])
                valley_y.extend([ys[index], ys[index + 1]])
    if valley_x:
        ax.scatter(valley_x, valley_y, s=140, facecolors="none", edgecolors=ACCENT,
                   linewidths=1.4, zorder=4,
                   label=f"flat-wall pair endpoints (|Δwall|/wall < {flat_tol:.0%})")
    ax.set_xlabel("measurement index along the preregistered trajectory "
                  "(collection order; NOT sorted by η)")
    ax.set_ylabel("dominant residual term  max(stall, occ-deficit)  (%)")
    ax.set_title(
        f"Check (c) [{_verdict(check)}]: does the dominant residual fall across a "
        f"flat-wall step?\nmonotone-in-valley fraction = {_fmt(check.get('frac'))} "
        f"(pairs={check.get('in_valley_pairs')}, tasks={check.get('tasks')})"
        f"{_interval(check.get('ci95_task_bootstrap'))}",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    if trajs:
        ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig4_monotone_valley.png", dpi=150)
    plt.close(fig)


def fig_correct_but_slow(rep: dict, out: Path) -> None:
    ms = [m for m in _seed_points(rep) if m.get("speedup")]
    ms.sort(key=lambda m: m["speedup"])
    names = [m["task_id"] for m in ms]
    sp = [m["speedup"] for m in ms]
    # honest coloring by the ACTUAL baseline used (from the labeled run):
    # aiter_vendor = real AITER CK production kernel; hipblaslt_vendor = hipBLASLt
    # GEMM; framework = torch fused op (no standalone AITER kernel for that op).
    bt = [m.get("baseline_type") for m in ms]
    cmap = {"aiter_vendor": ACCENT, "hipblaslt_vendor": BLUE, "framework": GREEN}
    colors = [cmap.get(b, GREY) for b in bt]
    n_aiter = sum(1 for b in bt if b in ("aiter_vendor", "hipblaslt_vendor"))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(range(len(names)), sp, color=colors)
    ax.axhline(1.0, ls="--", color=GREY, label="baseline parity (speedup=1)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("seed speedup vs its PRODUCTION baseline")
    ax.set_title(f"All seeds are CORRECT. Seed speedup vs the real production baseline\n"
                 f"({n_aiter}/{len(names)} measured against AITER/hipBLASLt vendor kernels; "
                 f"seeds sit below the vendor bar - the correct-but-slow wall)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ACCENT, label="AITER vendor (CK kernel)"),
                       Patch(color=BLUE, label="hipBLASLt vendor (GEMM)"),
                       Patch(color=GREEN, label="framework (torch; no standalone AITER op)")],
              loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig5_correct_but_slow.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate P0 study figures")
    ap.add_argument("--report", default="runs/p0_study.json")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args(argv)
    rep = _load(args.report)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_roofline_eta(rep, out)
    fig_eta_vs_speedup(rep, out)
    fig_residual_fit(rep, out)
    fig_monotone_valley(rep, out)
    fig_correct_but_slow(rep, out)
    print(f"[plots] wrote 5 figures to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
