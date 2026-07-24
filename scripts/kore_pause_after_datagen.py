#!/usr/bin/env python3
"""Development-only pause guard for one identity-owned legacy run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import time

from kore.ops.runtime import (
    IncrementalLogReader,
    ProcessIdentity,
    SecureRuntime,
    SecurityError,
    deprecated_entrypoint,
    terminate_owned,
)
from kore.ops.verify import verify_task_shards


MIGRATION = (
    "let the SPUR wave finish, verify it with scripts/_kf_verify.py, and use "
    "scheduler-native dependencies instead of stopping a live production run"
)


def build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("KORE_REPO", str(repo)))
    parser.add_argument("--runtime-dir", default=os.environ.get("KORE_RUNTIME_DIR"))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--data-root", default=os.environ.get("KORE_DATA_ROOT", "data/full14b")
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.environ.get("KORE_PAUSE_POLL_S", "30")),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.environ.get("KORE_PAUSE_TIMEOUT_S", "86400")),
    )
    parser.add_argument("--term-timeout", type=float, default=30.0)
    parser.add_argument("--kill-timeout", type=float, default=5.0)
    parser.add_argument("--confirm-stop-owned-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _valid_group(path: Path) -> bool:
    marker = path.with_suffix(path.suffix + ".inprogress")
    if marker.exists() or marker.is_symlink():
        return False
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        return False
    try:
        with path.open() as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    return bool(records) and all(isinstance(record, dict) for record in records)


def _identity(value: object, label: str) -> ProcessIdentity:
    if not isinstance(value, dict):
        raise SecurityError(f"active state has no {label} process identity")
    try:
        return ProcessIdentity.from_json(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise SecurityError(f"invalid {label} process identity: {exc}") from exc


def _stop_active(
    runtime: SecureRuntime,
    active: dict,
    *,
    term_timeout: float,
    kill_timeout: float,
) -> tuple[bool, str]:
    supervisor = _identity(active.get("supervisor"), "supervisor")
    result = terminate_owned(
        supervisor, term_timeout=term_timeout, kill_timeout=kill_timeout
    )
    if not result.stopped:
        return False, f"supervisor stop failed: {result.reason}"
    child_state = active.get("child_state")
    if isinstance(child_state, str) and child_state:
        try:
            child_record = runtime.read_json(child_state)
        except FileNotFoundError:
            child_record = {}
        child_value = child_record.get("identity")
        if child_value is not None:
            child = _identity(child_value, "child")
            result = terminate_owned(
                child, term_timeout=term_timeout, kill_timeout=kill_timeout
            )
            if not result.stopped:
                return False, f"child stop failed: {result.reason}"
    return True, "owned supervisor and child stopped"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not deprecated_entrypoint(
        "scripts/kore_pause_after_datagen.py", MIGRATION, dry_run=args.dry_run
    ):
        print(
            json.dumps(
                {
                    "action": "observe one owned run; stop only after strict task-set verification",
                    "sentinel": "paused-after-datagen",
                    "signals": ["TERM", "bounded wait", "KILL"],
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.confirm_stop_owned_run:
        print(
            "ERROR: destructive action requires --confirm-stop-owned-run",
            file=sys.stderr,
        )
        return 65
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        print("ERROR: poll and timeout values must be positive", file=sys.stderr)
        return 2
    repo = Path(args.repo).resolve()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = repo / data_root
    try:
        runtime = SecureRuntime(args.runtime_dir)
        active = runtime.read_json(Path("active") / "campaign.json")
        run_id = str(active.get("run_id", ""))
        if not run_id:
            raise SecurityError("active campaign state has no run_id")
        if args.run_id and args.run_id != run_id:
            raise SecurityError(
                f"active run_id mismatch: expected={args.run_id} actual={run_id}"
            )
        existing = runtime.peek_sentinel("paused-after-datagen")
        if existing is not None:
            if existing.get("run_id") == run_id:
                print("PAUSE_GUARD run is already paused", flush=True)
                return 0
            runtime.clear_sentinel("paused-after-datagen")

        from kore.tasks.registry import train_tasks

        tasks = [task.task_id for task in train_tasks()]
        task_set = runtime.store_task_set(
            Path("runs") / run_id / "task-set.json", tasks
        )
        groups = data_root / "groups"
        log_value = active.get("log_path")
        reader = (
            IncrementalLogReader(str(log_value))
            if isinstance(log_value, str) and log_value
            else None
        )
        build_started = False
        deadline = time.monotonic() + args.timeout_seconds
        while time.monotonic() < deadline:
            if reader is not None:
                build_started = build_started or any(
                    "stage start: build" in line for line in reader.read_lines()
                )
            complete = sum(
                _valid_group(groups / f"{task_id}.jsonl")
                for task_id in task_set.task_ids
            )
            print(
                f"PAUSE_GUARD groups={complete}/{task_set.count} "
                f"task_sha256={task_set.sha256} build_started={build_started}",
                flush=True,
            )
            if complete == task_set.count or build_started:
                break
            time.sleep(args.poll_seconds)
        else:
            print("ERROR: pause boundary was not reached before timeout", file=sys.stderr)
            return 4

        stopped, reason = _stop_active(
            runtime,
            active,
            term_timeout=args.term_timeout,
            kill_timeout=args.kill_timeout,
        )
        if not stopped:
            print(f"ERROR: {reason}", file=sys.stderr)
            return 5
        status = verify_task_shards(
            data_root, task_set.task_ids, target_wins=0, kinds=("groups",)
        )
        if not status.ok:
            print(
                "ERROR: owned run stopped at build boundary, but strict datagen "
                f"verification failed: {'; '.join(status.errors[:5])}",
                file=sys.stderr,
            )
            return 6
        runtime.write_sentinel(
            "paused-after-datagen",
            {
                "schema": 1,
                "run_id": run_id,
                "task_count": task_set.count,
                "task_sha256": task_set.sha256,
                "build_started": build_started,
                "created_ns": time.time_ns(),
            },
        )
    except (OSError, SecurityError, ValueError) as exc:
        print(f"ERROR: pause guard safety check failed: {exc}", file=sys.stderr)
        return 74
    print(f"PAUSE_GUARD ALERT PAUSED_AFTER_DATAGEN run_id={run_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
