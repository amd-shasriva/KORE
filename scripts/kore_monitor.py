#!/usr/bin/env python3
"""Read-only monitor for identity-owned campaign state and incremental logs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

from kore.ops.runtime import (
    IncrementalLogReader,
    ProcessIdentity,
    SecureRuntime,
    SecurityError,
    identity_matches,
)
from kore.ops.verify import verify_campaign


STAGE_RE = re.compile(r"STAGE  campaign \(\w+\): stage (start|done): (\w+)")
ERR_RE = re.compile(
    r"Traceback \(most recent call last\)| ERROR |OutOfMemoryError|out of memory|"
    r"HIP out of memory|CUDA error|HIP error|torch\.cuda\.OutOfMemory|CalledProcessError"
)
GATE_RE = re.compile(
    r"retention[^\n]*(FAIL|regress|below)|GATE[^\n]*FAIL|hard-stop|hard stop"
)
R429_RE = re.compile(r"429|rate.?limit|RateLimit|overloaded|retries=[1-9]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", default=os.environ.get("KORE_RUNTIME_DIR"))
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.environ.get("KORE_MONITOR_POLL_S", "180")),
    )
    parser.add_argument("--rate-limit-step", type=int, default=40)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _child(runtime: SecureRuntime, active: dict) -> ProcessIdentity | None:
    state = active.get("child_state")
    if not isinstance(state, str) or not state:
        return None
    try:
        value = runtime.read_json(state)
    except FileNotFoundError:
        return None
    identity = value.get("identity")
    if not isinstance(identity, dict):
        raise SecurityError(f"child state has no identity: {state}")
    return ProcessIdentity.from_json(identity)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "source": "private active/campaign.json",
                    "process_discovery": "PID/start-time/run-ID identity only",
                    "log_mode": "incremental",
                },
                sort_keys=True,
            )
        )
        return 0
    if args.poll_seconds <= 0 or args.rate_limit_step < 1:
        print("ERROR: poll values must be positive", file=sys.stderr)
        return 2
    try:
        runtime = SecureRuntime(args.runtime_dir, create=False)
    except SecurityError as exc:
        print(f"ERROR: no safe runtime state: {exc}", file=sys.stderr)
        return 4
    reader: IncrementalLogReader | None = None
    reader_path: str | None = None
    stage: tuple[str, str] | None = None
    errors = gates = rate_limits = alerted_rate_limits = 0
    polls = 0
    print("MONITOR start", flush=True)
    while True:
        try:
            active = runtime.read_json(Path("active") / "campaign.json")
            child = _child(runtime, active)
        except (FileNotFoundError, SecurityError, KeyError, TypeError, ValueError) as exc:
            print(f"ERROR: invalid active campaign state: {exc}", file=sys.stderr)
            return 4
        log_path = active.get("log_path")
        if isinstance(log_path, str) and log_path and log_path != reader_path:
            reader_path = log_path
            reader = IncrementalLogReader(log_path)
        for line in reader.read_lines() if reader is not None else []:
            matches = STAGE_RE.findall(line)
            if matches and matches[-1] != stage:
                stage = matches[-1]
                print(f"ALERT STAGE {stage[1]}:{stage[0]}", flush=True)
            new_errors = len(ERR_RE.findall(line))
            if new_errors:
                errors += new_errors
                print(f"ALERT ERROR_DETECTED total={errors}", flush=True)
            new_gates = len(GATE_RE.findall(line))
            if new_gates:
                gates += new_gates
                print(f"ALERT RETENTION_GATE total={gates}", flush=True)
            rate_limits += len(R429_RE.findall(line))
        if rate_limits >= alerted_rate_limits + args.rate_limit_step:
            alerted_rate_limits = rate_limits
            print(f"ALERT RATE_LIMIT_STORM total={rate_limits}", flush=True)

        running = False
        reason = "no child state"
        if child is not None:
            running, reason = identity_matches(child)
        if not running:
            repo = active.get("repo")
            data_root = active.get("data_root")
            stages = active.get("required_stages")
            if isinstance(repo, str) and isinstance(data_root, str) and isinstance(stages, list):
                status = verify_campaign(repo, data_root, [str(item) for item in stages])
            else:
                status = None
            if status is not None and status.ok and active.get("phase") == "succeeded":
                print("ALERT CAMPAIGN_COMPLETE (strict artifacts verified)", flush=True)
                print("MONITOR exit", flush=True)
                return 0
            if args.once:
                print(f"MONITOR not running: {reason}", flush=True)
                return 3
        stage_text = f"{stage[1]}:{stage[0]}" if stage else "none"
        print(
            f"HEARTBEAT n={polls} run_id={active.get('run_id', '?')} "
            f"pid={child.pid if child else None} running={running} "
            f"stage={stage_text} errs={errors} r429={rate_limits}",
            flush=True,
        )
        polls += 1
        if args.once:
            return 0 if running else 3
        if not running:
            print(f"ALERT CAMPAIGN_DIED ({reason})", flush=True)
            print("MONITOR exit", flush=True)
            return 3
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
