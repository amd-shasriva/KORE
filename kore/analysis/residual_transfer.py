"""Compatibility report for the canonical P0 residual validation.

All statistics live in :mod:`kore.analysis.p0_sol`; this module no longer carries
an independent OLS/LOFO implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kore.analysis.p0_sol import reanalyze_report


def run(
    report_path: Path,
    *,
    permutations: int = 1000,
    bootstrap: int = 1000,
    seed: int = 20260723,
) -> dict:
    source = json.loads(Path(report_path).read_text())
    report = reanalyze_report(
        source,
        permutations=permutations,
        bootstrap=bootstrap,
        seed=seed,
    )
    check = report["checks"]["b"]
    primary = check.get("normalized_primary") or {}
    lofo = check.get("leave_family_out") or {}
    return {
        "validation_schema": report["validation_schema"],
        "analysis_fingerprint": report["analysis_fingerprint"],
        "model_fingerprint_status": report["model_fingerprint_status"],
        "n_points": check.get("n", 0),
        "raw_in_sample": check.get("raw_in_sample"),
        "normalized_primary": primary,
        "leave_family_out": lofo,
        "family_evidence": check.get("family_evidence") or {},
        "shaping_evidence": report.get("shaping_evidence"),
        "verdict": (
            "SUPPORTED_FOR_SHAPING"
            if check.get("verdict") == "PASS"
            else "NOT_SUPPORTED_FOR_SHAPING"
        ),
        # Expose the canonical result for exact wrapper-equivalence tests.
        "canonical_check": check,
    }


def render(result: dict) -> str:
    primary = result.get("normalized_primary") or {}
    raw = result.get("raw_in_sample") or {}
    lines = [
        "# Residual transfer (canonical leakage-controlled P0)",
        f"# schema={result.get('validation_schema')}",
        f"# points={result.get('n_points')} model={result.get('model_fingerprint_status')}",
        "",
        "Raw in-sample diagnostic:",
        f"  named R2={raw.get('named_r2')}",
        f"  Tcand-only R2={raw.get('tcand_only_r2')}",
        f"  denominator-null median="
        f"{(raw.get('denominator_preserving_null') or {}).get('null_median')}",
        "",
        "Normalized task-cluster held-out primary:",
        f"  R2={primary.get('task_cluster_cv_r2')}",
        f"  Tcand-only R2={primary.get('tcand_only_cv_r2')}",
        f"  CI95={primary.get('ci95_task_bootstrap')}",
        "",
        "Leave-family-out:",
    ]
    for family, values in sorted((result.get("leave_family_out") or {}).items()):
        lines.append(f"  {family}: n={values.get('n_test')} R2={values.get('r2')}")
    lines.extend([
        "",
        f"VERDICT: {result.get('verdict')}",
        "Empirical shaping is disabled for every family not explicitly listed as passing.",
    ])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Canonical leakage-controlled residual transfer report")
    parser.add_argument("--report", default="data/p0_study_final.json")
    parser.add_argument("--out", default=None)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    result = run(
        Path(args.report),
        permutations=args.permutations,
        bootstrap=args.bootstrap,
    )
    print(render(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\n[residual_transfer] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
