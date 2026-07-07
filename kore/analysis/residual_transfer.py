"""CRUX EXPERIMENT: does the named-residual structure transfer ACROSS operator families?

P0 check (b) proved the runtime residual ``R = T_measured - T_min`` is reconstructed from
the NAMED counter terms (memory-stall time + occupancy-deficit time) with R^2 ~ 0.98 when
POOLED across operators. The "paradigm" claim (an operator-INDEPENDENT residual latent that
enables zero-shot cross-family transfer) requires strictly more than that pooled fit. This
module measures the single number that decides paradigm-vs-strong-combination:

  Test A  Leave-One-operator-Family-Out (LOFO) out-of-family R^2 of the residual
          decomposition -- fit the named-term -> residual map on all OTHER families,
          predict a held-out family, score R^2 on it. (Raw form = check (b); normalized
          form removes the kernel-size confound: (1 - eta) ~ stall_frac + occupancy_deficit.)
  Test B  Coefficient stability of the decomposition across LOFO folds (operator-independent
          => stable betas).
  Test C  Family-decodability from the residual latent (stall, occupancy, eta): if a
          nearest-centroid classifier recovers the family far above the majority baseline,
          families occupy DISTINCT residual regions (operator-SEPARABLE latent, weakening the
          "shared manifold" story); if near baseline, the latent is operator-agnostic (shared).

Reading:
  * High LOFO R^2 (>~0.7) AND low family-decodability  -> shared, universal residual latent:
    the paradigm's necessary condition holds; the residual-predictive world model is well-founded.
  * High LOFO R^2 but high decodability -> the decomposition transfers, but families live on
    different residual patches (KernelSight-LM/PipeWeave-style per-family structure): partial.
  * Low/negative LOFO R^2 -> the decomposition is operator-SPECIFIC: retreat to the
    (still-publishable) combination paper; do NOT claim an operator-independent latent.

CPU-only, no GPU, no external deps beyond numpy. Reads the PMC'd P0 measures
(``data/p0_study_final.json`` by default). Touches nothing live.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Optional

from kore.eval.generalization import family_of


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _load_points(report_path: Path) -> list[dict]:
    """Extract PMC-usable residual points from a p0_sol JSON report."""
    data = json.loads(Path(report_path).read_text())
    ms = data.get("measures", []) if isinstance(data, dict) else (data or [])
    pts = []
    for m in ms:
        if not m.get("correct"):
            continue
        st, oc = m.get("stall_frac"), m.get("occupancy")
        rr, cm = m.get("residual_ms"), m.get("cand_ms")
        tmin = m.get("t_min_ms")
        if st is None or oc is None or rr is None or not cm or cm <= 0:
            continue
        fam = family_of(m.get("task_id", ""))
        if fam is None:
            continue
        eta = (tmin / cm) if (tmin is not None and cm) else (m.get("eta"))
        pts.append({
            "task_id": m["task_id"], "family": fam,
            "stall": float(st), "occ_deficit": max(0.0, 1.0 - float(oc)),
            "cand_ms": float(cm), "residual_ms": float(rr),
            "eta": float(eta) if eta is not None else None,
        })
    return pts


# --------------------------------------------------------------------------- #
# tiny numpy OLS + R^2 (no sklearn dependency; matches p0_sol.ols_r2 conventions)
# --------------------------------------------------------------------------- #
def _fit_ols(X, y):
    import numpy as np
    A = np.array([row + [1.0] for row in X], dtype=float)
    b = np.array(y, dtype=float)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coef  # [w0, w1, ..., intercept]


def _predict(coef, X):
    import numpy as np
    A = np.array([row + [1.0] for row in X], dtype=float)
    return A @ coef


def _r2(y_true, y_pred) -> Optional[float]:
    import numpy as np
    y = np.array(y_true, dtype=float)
    p = np.array(y_pred, dtype=float)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 1e-30:
        return None
    ss_res = float(((y - p) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def _design(pts, normalized: bool):
    """Return (X, y). raw: y=residual_ms, X=[stall*T, occdef*T] (= check b).
    normalized: y = 1-eta = residual/cand, X = [stall, occdef] (size-confound removed)."""
    X, y = [], []
    for p in pts:
        if normalized:
            X.append([p["stall"], p["occ_deficit"]])
            y.append(p["residual_ms"] / p["cand_ms"])
        else:
            X.append([p["stall"] * p["cand_ms"], p["occ_deficit"] * p["cand_ms"]])
            y.append(p["residual_ms"])
    return X, y


# --------------------------------------------------------------------------- #
# Test A + B: leave-one-family-out transfer of the residual decomposition
# --------------------------------------------------------------------------- #
def lofo_transfer(pts, normalized: bool) -> dict:
    fams = sorted({p["family"] for p in pts})
    Xall, yall = _design(pts, normalized)
    pooled_coef = _fit_ols(Xall, yall)
    pooled_r2 = _r2(yall, _predict(pooled_coef, Xall))

    per_family = {}
    coefs = []
    for f in fams:
        test = [p for p in pts if p["family"] == f]
        train = [p for p in pts if p["family"] != f]
        if len(test) < 3 or len(train) < 4:
            continue
        Xtr, ytr = _design(train, normalized)
        Xte, yte = _design(test, normalized)
        coef = _fit_ols(Xtr, ytr)
        coefs.append(coef)
        r2_out = _r2(yte, _predict(coef, Xte))                 # trained elsewhere -> test on F
        # in-family ceiling: fit on F, eval on F (in-sample) -> is F predictable at all?
        r2_in = _r2(yte, _predict(_fit_ols(Xte, yte), Xte)) if len(test) >= 4 else None
        per_family[f] = {"n_test": len(test), "r2_out_of_family": r2_out,
                         "r2_in_family_ceiling": r2_in}

    # coefficient stability across folds (operator-independent => low CV)
    import numpy as np
    stab = None
    if len(coefs) >= 2:
        C = np.array(coefs)
        mean = C.mean(axis=0)
        std = C.std(axis=0)
        stab = {"beta_mean": [round(x, 4) for x in mean.tolist()],
                "beta_std": [round(x, 4) for x in std.tolist()],
                "beta_cv": [round(abs(s / m), 3) if abs(m) > 1e-9 else None
                            for s, m in zip(std, mean)]}
    outs = [v["r2_out_of_family"] for v in per_family.values() if v["r2_out_of_family"] is not None]
    return {"normalized": normalized, "pooled_in_sample_r2": pooled_r2,
            "per_family": per_family,
            "median_out_of_family_r2": (median(outs) if outs else None),
            "min_out_of_family_r2": (min(outs) if outs else None),
            "n_families_scored": len(outs), "coef_stability": stab}


# --------------------------------------------------------------------------- #
# Test C: is the operator family decodable from the residual latent?
# --------------------------------------------------------------------------- #
def family_decodability(pts) -> dict:
    """Leave-one-out nearest-centroid classification of family from the standardized
    residual latent (stall, occupancy_deficit, 1-eta). Accuracy >> majority baseline
    => families are separable in residual space (operator-specific patches)."""
    import numpy as np
    feats = np.array([[p["stall"], p["occ_deficit"],
                       (1.0 - p["eta"]) if p["eta"] is not None else 0.0] for p in pts], dtype=float)
    fams = [p["family"] for p in pts]
    mu, sd = feats.mean(axis=0), feats.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (feats - mu) / sd
    uniq = sorted(set(fams))
    y = np.array([uniq.index(f) for f in fams])
    correct = 0
    for i in range(len(Z)):
        mask = np.ones(len(Z), dtype=bool); mask[i] = False
        cents = []
        for c in range(len(uniq)):
            m = mask & (y == c)
            cents.append(Z[m].mean(axis=0) if m.any() else np.full(Z.shape[1], 1e9))
        d = [float(((Z[i] - cen) ** 2).sum()) for cen in cents]
        if int(np.argmin(d)) == y[i]:
            correct += 1
    acc = correct / len(Z)
    counts = {f: fams.count(f) for f in uniq}
    majority = max(counts.values()) / len(fams)
    return {"loo_accuracy": round(acc, 3), "majority_baseline": round(majority, 3),
            "n_classes": len(uniq), "chance": round(1.0 / len(uniq), 3),
            "separable_signal": round(acc - majority, 3)}


def _verdict(a_norm: dict, decode: dict) -> str:
    r = a_norm.get("median_out_of_family_r2")
    mn = a_norm.get("min_out_of_family_r2")
    sep = decode.get("separable_signal")
    if r is None:
        return "INCONCLUSIVE (insufficient data)"
    if r >= 0.7 and (mn is None or mn >= 0.3):
        base = "PARADIGM SUPPORTED: the named-residual decomposition transfers across held-out families"
    elif r >= 0.4:
        base = "PARTIAL: the decomposition transfers moderately; operator-independence is weak-to-moderate"
    else:
        base = "PARADIGM NOT SUPPORTED: residual structure is operator-specific -> use the combination framing"
    shared = "shared/agnostic latent" if (sep is not None and sep < 0.25) else "operator-separable latent (per-family patches)"
    return f"{base}; residual latent looks like a {shared}."


def run(report_path: Path) -> dict:
    pts = _load_points(report_path)
    a_raw = lofo_transfer(pts, normalized=False)
    a_norm = lofo_transfer(pts, normalized=True)
    decode = family_decodability(pts)
    from collections import Counter
    return {
        "n_points": len(pts),
        "per_family_counts": dict(Counter(p["family"] for p in pts)),
        "testA_raw_checkb_form": a_raw,
        "testA_normalized_sol_gap_form": a_norm,
        "testC_family_decodability": decode,
        "verdict": _verdict(a_norm, decode),
    }


def render(res: dict) -> str:
    L = ["# CRUX EXPERIMENT: cross-operator-family transfer of the named residual",
         f"# points={res['n_points']}  families={res['per_family_counts']}", ""]
    for key, title in [("testA_normalized_sol_gap_form", "Test A (normalized: (1-eta) ~ stall + occ_deficit)  [PRIMARY]"),
                       ("testA_raw_checkb_form", "Test A (raw check-(b) form: residual_ms ~ stall*T + occdef*T)")]:
        a = res[key]
        L.append(f"## {title}")
        L.append(f"  pooled in-sample R^2 = {a['pooled_in_sample_r2']}")
        L.append(f"  {'family':12s} {'n':>3s} {'R2_out_of_family':>17s} {'R2_in_family':>13s}")
        for f, v in sorted(a["per_family"].items()):
            ro = f"{v['r2_out_of_family']:.3f}" if v["r2_out_of_family"] is not None else "-"
            ri = f"{v['r2_in_family_ceiling']:.3f}" if v["r2_in_family_ceiling"] is not None else "-"
            L.append(f"  {f:12s} {v['n_test']:3d} {ro:>17s} {ri:>13s}")
        L.append(f"  --> MEDIAN out-of-family R^2 = {a['median_out_of_family_r2']}  (min = {a['min_out_of_family_r2']})")
        if a.get("coef_stability"):
            L.append(f"  coef stability betas={a['coef_stability']['beta_mean']} CV={a['coef_stability']['beta_cv']}")
        L.append("")
    d = res["testC_family_decodability"]
    L.append("## Test C (family decodable from residual latent?)")
    L.append(f"  LOO accuracy = {d['loo_accuracy']}  vs majority baseline {d['majority_baseline']} "
             f"(chance {d['chance']}, {d['n_classes']} classes)  -> separable signal {d['separable_signal']}")
    L.append("")
    L.append("=" * 70)
    L.append(f"VERDICT: {res['verdict']}")
    L.append("=" * 70)
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Crux experiment: cross-family residual transfer")
    ap.add_argument("--report", default="data/p0_study_final.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = run(Path(args.report))
    print(render(res))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\n[residual_transfer] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
