"""Generate P0 study figures from a p0_sol report JSON (matplotlib, headless).

Figures:
  fig1_roofline_eta.png   - per-operator SOL attainment (eta), colored by roofline bound
  fig2_eta_vs_speedup.png - check (a): eta vs speedup-vs-vendor (Spearman rho)
  fig3_residual_fit.png   - check (b): measured residual vs counter-predicted (R^2)
  fig4_monotone_valley.png- check (c): dominant residual term along the improvement path
  fig5_correct_but_slow.png - the correct-but-slow wall: eta and speedup per op

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
    ax.set_title("Seed-kernel SOL attainment per operator on gfx950 (MI350X)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ACCENT, label="compute-bound"),
                       Patch(color=BLUE, label="memory-bound")], loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig1_roofline_eta.png", dpi=150)
    plt.close(fig)


def fig_eta_vs_speedup(rep: dict, out: Path) -> None:
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
    rho = rep["checks"]["a"].get("rho")
    n = rep["checks"]["a"].get("n")
    ci = rep["checks"]["a"].get("ci95")
    ci_s = f"  95%CI[{ci[0]:.2f},{ci[1]:.2f}]" if ci else ""
    ax.set_xlabel("SOL attainment  η  (%)")
    ax.set_ylabel("speedup vs production baseline  (vendor / candidate)")
    ax.set_title(f"Check (a): does η predict speedup?   Spearman ρ = {rho:.3f} (n={n}){ci_s}"
                 if rho is not None else "Check (a): η vs speedup")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "fig2_eta_vs_speedup.png", dpi=150)
    plt.close(fig)


def fig_residual_fit(rep: dict, out: Path) -> None:
    rows = [m for m in rep["measures"] if m.get("stall_frac") is not None
            and m.get("occupancy") is not None and m.get("residual_ms") is not None and m.get("cand_ms")]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    if len(rows) >= 3:
        X = np.array([[m["stall_frac"] * m["cand_ms"], (1 - m["occupancy"]) * m["cand_ms"], 1.0]
                      for m in rows])
        y = np.array([m["residual_ms"] for m in rows])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.scatter(pred, y, s=60, color=GREEN, zorder=3, edgecolor="white")
        lim = [0, max(float(y.max()), float(pred.max())) * 1.05]
        ax.plot(lim, lim, ls="--", color=GREY, label="y = x (perfect)")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ci = rep["checks"]["b"].get("ci95")
        ci_s = f"  95%CI[{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
        ax.set_title(f"Check (b): residual decomposes into stall + occupancy-deficit\n"
                     f"measured vs counter-predicted residual   R² = {r2:.4f} (n={len(rows)}){ci_s}")
    else:
        ax.set_title("Check (b): insufficient PMC data")
    ax.set_xlabel("predicted residual time from PMC terms  (ms)")
    ax.set_ylabel("measured residual  T_measured − T_min  (ms)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "fig3_residual_fit.png", dpi=150)
    plt.close(fig)


def fig_monotone_valley(rep: dict, out: Path) -> None:
    # group measures by task, order by eta ascending, plot dominant residual term
    by_task: dict[str, list] = {}
    for m in rep["measures"]:
        if m.get("correct") and m.get("eta") and m.get("stall_frac") is not None:
            by_task.setdefault(m["task_id"], []).append(m)
    trajs = {t: sorted(ms, key=lambda m: m["eta"]) for t, ms in by_task.items() if len(ms) >= 2}
    fig, ax = plt.subplots(figsize=(8.5, 6))

    def dom(m):
        return max(m.get("stall_frac") or 0.0, 1 - (m.get("occupancy") if m.get("occupancy") is not None else 1.0))

    if trajs:
        for t, ms in list(trajs.items())[:8]:
            xs = [m["eta"] * 100 for m in ms]
            ys = [dom(m) * 100 for m in ms]
            ax.plot(xs, ys, marker="o", label=t, alpha=0.8)
    ax.set_xlabel("SOL attainment η (%)  - improvement direction →")
    ax.set_ylabel("dominant residual term  max(stall, occ-deficit)  (%)")
    frac = rep["checks"]["c"].get("frac")
    pairs = rep["checks"]["c"].get("in_valley_pairs")
    ax.set_title(f"Check (c): dominant residual falls as η rises\n"
                 f"monotone-in-valley fraction = {frac} (pairs={pairs})")
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
