"""Offline, fail-closed Qwen3-32B model/resource preflight.

Example:
    PYTHONPATH=. python scripts/preflight_qwen3_32b.py \
      --model-path /models/Qwen3-32B \
      --revision <full-hub-commit-sha> \
      --scratch-path /scratch \
      --measured-profile measured-peak.json \
      --output preflight.json

The default Qwen3-32B profile intentionally contains ``revision=MEASURE``.
Omitting ``--revision`` therefore fails.  Omitting a matching measured peak
profile emits analytical lower bounds but never reports that the workload fits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from kore.policy.model_spec import (
    ModelSpec,
    ModelSpecError,
    QWEN3_32B_PROFILE,
)
from kore.policy.resources import (
    InsufficientResourcesError,
    MeasuredPeakProfile,
    ResourcePreflightError,
    UnresolvedProductionFieldError,
    collect_resource_snapshot,
    evaluate_resource_preflight,
    load_resource_snapshot,
)


def _emit(payload: dict[str, Any], output: Optional[str]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    print(text, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a local Qwen3-32B checkpoint and measured resource profile "
            "without loading the model"
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--revision",
        help="immutable 40/64-hex Hub commit (required; profile defaults to MEASURE)",
    )
    parser.add_argument("--scratch-path", required=True)
    parser.add_argument(
        "--resources-json",
        help="recorded resource snapshot; otherwise probe this host",
    )
    parser.add_argument(
        "--measured-profile",
        help="measured peak JSON tied to this model/environment profile",
    )
    parser.add_argument("--headroom-fraction", type=float, default=0.10)
    parser.add_argument("--output", help="also write the complete report JSON here")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model_spec = ModelSpec.from_local_checkpoint(
            args.model_path,
            revision=args.revision,
            expected=QWEN3_32B_PROFILE,
        )
        resources = (
            load_resource_snapshot(args.resources_json)
            if args.resources_json
            else collect_resource_snapshot(args.model_path, args.scratch_path)
        )
        measured = (
            MeasuredPeakProfile.from_json(args.measured_profile)
            if args.measured_profile
            else None
        )
        report = evaluate_resource_preflight(
            model_spec,
            resources,
            measured,
            headroom_fraction=args.headroom_fraction,
        )
        _emit(report.to_dict(), args.output)

        if report.status == "unresolved":
            raise UnresolvedProductionFieldError("; ".join(report.reasons))
        if report.status == "insufficient":
            raise InsufficientResourcesError("; ".join(report.reasons))
        # Analytical lower bounds can prove impossibility, never sufficiency.
        report.assert_production_ready()
        return 0
    except (ModelSpecError, ResourcePreflightError, OSError) as exc:
        if "report" not in locals():
            _emit(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "production_ready": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                args.output,
            )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
