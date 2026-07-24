#!/usr/bin/env python3
"""Development-only legacy supervisor with owned process state.

Production datagen moved to the SPUR supervisor.  This compatibility entrypoint
has no pattern-based process discovery: one run ID owns one child process group,
and success requires both exit code zero and strict manifest/artifact checks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from kore.ops.campaign import CampaignSpec, CampaignSupervisor
from kore.ops.runtime import SecureRuntime, deprecated_entrypoint


MIGRATION = (
    "use scripts/spur_supervise_datagen.py for production datagen and a "
    "scheduler-approved launcher for training stages"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("KORE_REPO", str(Path.cwd())))
    parser.add_argument("--python", default=os.environ.get("KORE_PY", sys.executable))
    parser.add_argument("--runtime-dir", default=os.environ.get("KORE_RUNTIME_DIR"))
    parser.add_argument("--run-id", default=os.environ.get("KORE_RUN_ID"))
    parser.add_argument(
        "--data-root", default=os.environ.get("KORE_DATA_ROOT", "data/full14b")
    )
    parser.add_argument(
        "--stages",
        default=os.environ.get(
            "KORE_SUP_STAGES", "build,sft,dpo,grpo,soup,eval"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=os.environ.get("KORE_SUP_FORCE", "0") == "1",
    )
    parser.add_argument(
        "--workers", default=os.environ.get("KORE_DATAGEN_WORKERS", "64")
    )
    parser.add_argument(
        "--sft-total", default=os.environ.get("KORE_SUP_SFT_TOTAL", "13000")
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.environ.get("KORE_SUP_POLL_S", "120")),
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=float(os.environ.get("KORE_SUP_COOLDOWN_S", "90")),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=int(os.environ.get("KORE_SUP_MAX_RETRIES", "12")),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _command(args: argparse.Namespace, python: Path) -> tuple[str, ...]:
    command = [
        str(python),
        "scripts/run_campaign.py",
        "--model",
        "Qwen/Qwen3-14B",
        "--full-ft",
        "--use-hf",
        "--teacher",
        "claude",
        "--adaptive-steps",
    ]
    if args.force:
        command.append("--force")
    command.extend(
        [
            "--stages",
            args.stages,
            "--sft-total",
            str(args.sft_total),
            "--gpu-ids",
            "0,1,2,3,4,5,6,7",
            "--datagen-workers",
            str(args.workers),
            "--ground-reasoning",
            "--profile-reward",
            "0.15",
            "--data-root",
            args.data_root,
            "--midtrain-out",
            "runs/full/midtrain",
            "--sft-out",
            "runs/full/sft",
            "--dpo-out",
            "runs/full/dpo",
            "--grpo-out",
            "runs/full/grpo",
            "--soup-out",
            "runs/full/soup",
        ]
    )
    return tuple(command)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not deprecated_entrypoint(
        "scripts/kore_supervise.py", MIGRATION, dry_run=args.dry_run
    ):
        repo = Path(args.repo).resolve()
        print(
            json.dumps(
                {
                    "command": list(_command(args, Path(args.python))),
                    "repo": str(repo),
                    "required_stages": args.stages.split(","),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.max_attempts < 1 or args.poll_seconds <= 0 or args.cooldown_seconds < 0:
        raise SystemExit("attempt and timing values must be positive")
    repo = Path(args.repo).resolve()
    python = Path(args.python).expanduser().resolve()
    environment = os.environ.copy()
    environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
    for key in (
        "KORE_VERIFIED_CORRECTNESS",
        "KORE_COMPILE_BASELINE",
        "KORE_SHAPE_AUGMENT",
        "KORE_BENCH_COLD",
        "KORE_ROOFLINE_GATE",
        "KORE_MINTER_EVOLVE_GRAMMAR",
    ):
        environment.setdefault(key, "1")
    environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    stages = tuple(stage for stage in args.stages.split(",") if stage)
    spec = CampaignSpec(
        repo=repo,
        python=python,
        data_root=Path(args.data_root),
        command=_command(args, python),
        required_stages=stages,
        log_dir=repo / "runs" / "full" / "logs",
        log_prefix="campaign_foldin",
        environment=environment,
        poll_seconds=args.poll_seconds,
        cooldown_seconds=args.cooldown_seconds,
        max_attempts=args.max_attempts,
        run_id=args.run_id,
    )
    return CampaignSupervisor(spec, SecureRuntime(args.runtime_dir)).run()


if __name__ == "__main__":
    raise SystemExit(main())
