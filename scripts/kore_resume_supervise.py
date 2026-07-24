#!/usr/bin/env python3
"""Consume a private pause sentinel and run the legacy owned supervisor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from kore.ops.campaign import CampaignSpec, CampaignSupervisor
from kore.ops.runtime import SecureRuntime, SecurityError, deprecated_entrypoint


MIGRATION = (
    "use scripts/spur_supervise_datagen.py for production datagen, then launch "
    "training through the site scheduler after a strict dataset verification"
)
DEFAULT_STAGES = "build,midtrain,sft,dpo,grpo,soup,eval"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("KORE_REPO", str(Path.cwd())))
    parser.add_argument("--python", default=os.environ.get("KORE_PY", sys.executable))
    parser.add_argument("--runtime-dir", default=os.environ.get("KORE_RUNTIME_DIR"))
    parser.add_argument("--run-id", default=os.environ.get("KORE_RUN_ID"))
    parser.add_argument(
        "--data-root", default=os.environ.get("KORE_DATA_ROOT", "data/full14b")
    )
    parser.add_argument("--stages", default=DEFAULT_STAGES)
    parser.add_argument(
        "--workers", default=os.environ.get("KORE_DATAGEN_WORKERS", "64")
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
    parser.add_argument(
        "--sentinel-timeout",
        type=float,
        default=float(os.environ.get("KORE_SENTINEL_TIMEOUT_S", "86400")),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _command(args: argparse.Namespace, python: Path) -> tuple[str, ...]:
    return (
        str(python),
        "scripts/run_campaign.py",
        "--model",
        "Qwen/Qwen3-14B",
        "--full-ft",
        "--use-hf",
        "--teacher",
        "claude",
        "--adaptive-steps",
        "--force",
        "--stages",
        args.stages,
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
    )


def _consume_pause(runtime: SecureRuntime, timeout: float, poll: float) -> dict | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = runtime.consume_sentinel("paused-after-datagen")
        if value is not None:
            return value
        time.sleep(min(poll, max(0.05, deadline - time.monotonic())))
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not deprecated_entrypoint(
        "scripts/kore_resume_supervise.py", MIGRATION, dry_run=args.dry_run
    ):
        print(
            json.dumps(
                {
                    "consume_sentinel": "paused-after-datagen",
                    "command": list(_command(args, Path(args.python))),
                    "required_stages": args.stages.split(","),
                },
                sort_keys=True,
            )
        )
        return 0
    if (
        args.max_attempts < 1
        or args.poll_seconds <= 0
        or args.cooldown_seconds < 0
        or args.sentinel_timeout <= 0
    ):
        print("ERROR: attempt and timing values must be positive", file=sys.stderr)
        return 2
    repo = Path(args.repo).resolve()
    python = Path(args.python).expanduser().resolve()
    try:
        runtime = SecureRuntime(args.runtime_dir)
        pause = _consume_pause(runtime, args.sentinel_timeout, args.poll_seconds)
    except SecurityError as exc:
        print(f"ERROR: unsafe pause sentinel: {exc}", file=sys.stderr)
        return 74
    if pause is None:
        print("ERROR: pause sentinel was absent until the bounded timeout", file=sys.stderr)
        return 4
    print(
        f"RESUME_SUP consumed pause sentinel run_id={pause.get('run_id', '?')} "
        f"task_sha256={pause.get('task_sha256', '?')}",
        flush=True,
    )
    environment = os.environ.copy()
    environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
    environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    for key in (
        "KORE_VERIFIED_CORRECTNESS",
        "KORE_COMPILE_BASELINE",
        "KORE_SHAPE_AUGMENT",
        "KORE_BENCH_COLD",
    ):
        environment.setdefault(key, "1")
    stages = tuple(stage for stage in args.stages.split(",") if stage)
    spec = CampaignSpec(
        repo=repo,
        python=python,
        data_root=Path(args.data_root),
        command=_command(args, python),
        required_stages=stages,
        log_dir=repo / "runs" / "full" / "logs",
        log_prefix="resume_campaign",
        environment=environment,
        poll_seconds=args.poll_seconds,
        cooldown_seconds=args.cooldown_seconds,
        max_attempts=args.max_attempts,
        run_id=args.run_id,
    )
    return CampaignSupervisor(spec, runtime).run()


if __name__ == "__main__":
    raise SystemExit(main())
