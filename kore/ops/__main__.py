"""Command-line access to the operational safety helpers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .runtime import (
    LockBusy,
    OwnedProcess,
    ProcessIdentity,
    SecureRuntime,
    SecurityError,
    identity_matches,
    new_run_id,
    open_append_log,
    task_set_identity,
    terminate_owned,
)
from .verify import (
    status_json,
    verify_campaign,
    verify_grpo_config,
    verify_model_artifact,
    verify_sft_gate,
    verify_task_shards,
)


def _command_tail(value: list[str]) -> list[str]:
    command = list(value)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("a command is required after --")
    return command


def _run_owned(args: argparse.Namespace) -> int:
    try:
        command = _command_tail(args.command)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    run_id = args.run_id or new_run_id(args.name)
    env = os.environ.copy()
    for item in args.env:
        if "=" not in item:
            print(f"error: --env must be KEY=VALUE, got {item!r}", file=sys.stderr)
            return 2
        key, value = item.split("=", 1)
        if not key:
            print("error: --env key must not be empty", file=sys.stderr)
            return 2
        env[key] = value
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "run_id": run_id,
                    "name": args.name,
                    "cwd": str(Path(args.cwd).resolve()),
                    "log": str(Path(args.log).resolve()) if args.log else None,
                    "command": command,
                },
                sort_keys=True,
            )
        )
        return 0

    runtime = SecureRuntime(args.runtime_dir)
    state_relative = Path("runs") / run_id / f"{args.name}.json"
    log_handle = None
    try:
        with runtime.lock(args.name):
            if args.log:
                log_handle = open_append_log(args.log)
            try:
                child = OwnedProcess.spawn(
                    command,
                    run_id=run_id,
                    runtime=runtime,
                    state_relative=state_relative,
                    cwd=args.cwd,
                    env=env,
                    stdout=log_handle,
                )
            except FileNotFoundError:
                print(f"error: command not found: {command[0]}", file=sys.stderr)
                return 127

            stop_signal: list[int] = []

            def request_stop(signum: int, _frame: object) -> None:
                stop_signal.append(signum)

            previous = {
                signum: signal.signal(signum, request_stop)
                for signum in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                while child.poll() is None and not stop_signal:
                    time.sleep(0.2)
                if stop_signal:
                    result = child.terminate(
                        term_timeout=args.term_timeout,
                        kill_timeout=args.kill_timeout,
                    )
                    if not result.stopped:
                        print(
                            f"error: owned process did not stop: {result.reason}",
                            file=sys.stderr,
                        )
                        return 70
                    return 128 + stop_signal[0]
                return child.wait()
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
    except LockBusy as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 73
    except SecurityError as exc:
        print(f"error: operational safety check failed: {exc}", file=sys.stderr)
        return 74
    finally:
        if log_handle is not None:
            log_handle.close()


def _load_identity(runtime: SecureRuntime, state: str) -> ProcessIdentity:
    value = runtime.read_json(state)
    identity = value.get("identity")
    if not isinstance(identity, dict):
        raise SecurityError(f"process state has no identity object: {state}")
    return ProcessIdentity.from_json(identity)


def _status_process(args: argparse.Namespace) -> int:
    try:
        runtime = SecureRuntime(args.runtime_dir, create=False)
        identity = _load_identity(runtime, args.state)
        matched, reason = identity_matches(identity)
    except FileNotFoundError:
        print(json.dumps({"running": False, "reason": "state missing"}))
        return 3
    except SecurityError as exc:
        print(json.dumps({"running": False, "reason": str(exc)}))
        return 4
    print(
        json.dumps(
            {
                "running": matched,
                "reason": reason,
                "run_id": identity.run_id,
                "pid": identity.pid,
                "pgid": identity.pgid,
            },
            sort_keys=True,
        )
    )
    return 0 if matched else 3


def _stop_process(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "runtime_dir": args.runtime_dir,
                    "state": args.state,
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        runtime = SecureRuntime(args.runtime_dir, create=False)
        identity = _load_identity(runtime, args.state)
        result = terminate_owned(
            identity,
            term_timeout=args.term_timeout,
            kill_timeout=args.kill_timeout,
        )
    except (FileNotFoundError, SecurityError) as exc:
        print(f"error: cannot stop owned process: {exc}", file=sys.stderr)
        return 4
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0 if result.stopped else 5


def _task_ids(args: argparse.Namespace) -> list[str]:
    if args.tasks and args.tasks_file:
        raise ValueError("use only one of --tasks or --tasks-file")
    if args.tasks_file:
        text = Path(args.tasks_file).read_text()
    elif args.tasks:
        text = args.tasks
    else:
        raise ValueError("--tasks or --tasks-file is required")
    return [item.strip() for item in text.replace("\n", ",").split(",") if item.strip()]


def _verify(args: argparse.Namespace) -> int:
    try:
        if args.verify_kind == "campaign":
            status = verify_campaign(
                args.repo,
                args.data_root,
                [item for item in args.required_stages.split(",") if item],
            )
        elif args.verify_kind == "model":
            status = verify_model_artifact(args.path, repo=args.repo)
        elif args.verify_kind == "sft-gate":
            status = verify_sft_gate(args.manifest, args.candidate, repo=args.repo)
        elif args.verify_kind == "grpo-config":
            status = verify_grpo_config(args.config, repo=args.repo)
        elif args.verify_kind == "task-shards":
            status = verify_task_shards(
                args.data_root,
                _task_ids(args),
                target_wins=args.target_wins,
                kinds=tuple(item for item in args.kinds.split(",") if item),
            )
        elif args.verify_kind == "task-set":
            identity = task_set_identity(_task_ids(args))
            print(
                json.dumps(
                    {
                        "ok": True,
                        "count": identity.count,
                        "sha256": identity.sha256,
                        "task_ids": list(identity.task_ids),
                    },
                    sort_keys=True,
                )
            )
            return 0
        else:  # pragma: no cover - argparse constrains this
            raise ValueError(f"unknown verifier: {args.verify_kind}")
    except (OSError, ValueError, SecurityError) as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)], "details": {}}))
        return 2
    print(json.dumps(status_json(status), sort_keys=True))
    return 0 if status.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kore.ops",
        description="Owned-process and strict artifact safety helpers.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    run = sub.add_parser("run", help="run a command in an owned process group")
    run.add_argument("--runtime-dir")
    run.add_argument("--run-id")
    run.add_argument("--name", required=True)
    run.add_argument("--cwd", default=".")
    run.add_argument("--log")
    run.add_argument("--env", action="append", default=[])
    run.add_argument("--term-timeout", type=float, default=15.0)
    run.add_argument("--kill-timeout", type=float, default=5.0)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(func=_run_owned)

    status = sub.add_parser("status", help="check a persisted process identity")
    status.add_argument("--runtime-dir")
    status.add_argument("--state", required=True)
    status.set_defaults(func=_status_process)

    stop = sub.add_parser("stop", help="stop one persisted owned process")
    stop.add_argument("--runtime-dir")
    stop.add_argument("--state", required=True)
    stop.add_argument("--term-timeout", type=float, default=15.0)
    stop.add_argument("--kill-timeout", type=float, default=5.0)
    stop.add_argument("--dry-run", action="store_true")
    stop.set_defaults(func=_stop_process)

    verify = sub.add_parser("verify", help="strictly verify completion artifacts")
    verify_sub = verify.add_subparsers(dest="verify_kind", required=True)

    campaign = verify_sub.add_parser("campaign")
    campaign.add_argument("--repo", default=".")
    campaign.add_argument("--data-root", required=True)
    campaign.add_argument("--required-stages", required=True)
    campaign.set_defaults(func=_verify)

    model = verify_sub.add_parser("model")
    model.add_argument("--repo", default=".")
    model.add_argument("--path", required=True)
    model.set_defaults(func=_verify)

    sft = verify_sub.add_parser("sft-gate")
    sft.add_argument("--repo", default=".")
    sft.add_argument("--manifest", required=True)
    sft.add_argument("--candidate", required=True)
    sft.set_defaults(func=_verify)

    grpo = verify_sub.add_parser("grpo-config")
    grpo.add_argument("--repo", default=".")
    grpo.add_argument("--config", required=True)
    grpo.set_defaults(func=_verify)

    shards = verify_sub.add_parser("task-shards")
    shards.add_argument("--data-root", required=True)
    shards.add_argument("--tasks")
    shards.add_argument("--tasks-file")
    shards.add_argument("--target-wins", type=int, default=1)
    shards.add_argument("--kinds", default="repair,groups,wins")
    shards.set_defaults(func=_verify)

    task_set = verify_sub.add_parser("task-set")
    task_set.add_argument("--tasks")
    task_set.add_argument("--tasks-file")
    task_set.set_defaults(func=_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
