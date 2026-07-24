"""Drive resumable SPUR datagen waves to verified completion.

This lightweight login-node supervisor submits exactly one wave at a time,
waits for all ``kore-factory`` children to leave the queue, verifies durable
progress, and repartitions the remaining work. It never cancels jobs and aborts
instead of looping when the scheduler is unavailable or progress stalls.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def factory_jobs_active(squeue_output: str) -> bool:
    return any(
        len(fields := line.split()) >= 3 and fields[2].startswith("kore-fac")
        for line in squeue_output.splitlines()[1:]
    )


def progress_score(summary: dict) -> int:
    """Monotonic count of completed stage units, including partial win progress."""
    wins = sum(int(count) * int(bucket) for bucket, count in summary["wins_hist"].items())
    base = (
        2 * int(summary["tasks"])
        - int(summary["missing_repair"])
        - int(summary["missing_groups"])
    )
    return wins + base


def _json_line(output: str) -> dict:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError(f"command did not emit a JSON object: {output[-500:]}")


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.repo = Path(args.repo).resolve()
        self.log_path = Path(args.log).resolve()
        self.state_path = self.log_path.with_suffix(".state.json")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
        print(line, flush=True)
        with self.log_path.open("a") as fh:
            fh.write(line + "\n")

    def run(
        self,
        command: list[str],
        *,
        check: bool = True,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess:
        self.log("RUN " + " ".join(command))
        try:
            result = subprocess.run(
                command,
                cwd=self.repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"command timed out after {timeout}s: {' '.join(command)}"
            ) from exc
        if result.stdout:
            with self.log_path.open("a") as fh:
                fh.write(result.stdout)
                if not result.stdout.endswith("\n"):
                    fh.write("\n")
        if check and result.returncode:
            raise RuntimeError(
                f"command failed rc={result.returncode}: {' '.join(command)}\n"
                f"{result.stdout[-1000:]}"
            )
        return result

    def verify(self) -> dict:
        cleanup = self.state_path.with_suffix(".cleanup.txt")
        result = self.run(
            [
                self.args.python,
                "scripts/_kf_verify.py",
                self.args.data_root,
                str(self.args.target),
                "--json",
                "--cleanup-out",
                str(cleanup),
            ],
            timeout=600,
        )
        summary = _json_line(result.stdout)
        self.log(
            "VERIFY "
            f"complete={summary['fully_complete']}/{summary['tasks']} "
            f"remaining={summary['remaining_undone']} score={progress_score(summary)}"
        )
        return summary

    def queue(self) -> str:
        result = self.run(
            ["squeue", "-u", self.args.user],
            check=False,
            timeout=self.args.queue_timeout_seconds,
        )
        if result.returncode:
            raise RuntimeError(f"squeue failed rc={result.returncode}")
        return result.stdout

    def wait_for_wave(self) -> None:
        time.sleep(self.args.submission_grace_seconds)
        failures = 0
        seen_active = False
        visibility_deadline = time.monotonic() + self.args.visibility_timeout_seconds
        while True:
            try:
                output = self.queue()
                failures = 0
            except RuntimeError as exc:
                failures += 1
                self.log(f"WARN scheduler query failed ({failures}): {exc}")
                if failures >= self.args.max_scheduler_failures:
                    raise
                time.sleep(self.args.poll_seconds)
                continue
            active = factory_jobs_active(output)
            if active:
                seen_active = True
                self.log("WAIT factory wave remains active")
                time.sleep(self.args.poll_seconds)
                continue
            if seen_active:
                return
            if time.monotonic() >= visibility_deadline:
                self.log("WAIT no factory child became visible before deadline")
                return
            time.sleep(min(self.args.poll_seconds, 10))

    def submit_wave(self) -> str:
        result = self.run(
            [
                "env",
                f"KORE_REPO={self.repo}",
                f"KORE_PY={self.args.python}",
                f"KORE_DATA_ROOT={self.args.data_root}",
                f"KORE_WINS_TARGET={self.args.target}",
                "bash",
                "scripts/spur_submit_datagen.sh",
                str(self.args.shards),
                str(self.args.wave_nodes),
            ],
            timeout=600,
        )
        match = re.search(r"\bjob_id=(\d+)", result.stdout)
        if not match:
            if "Dataset is already complete" in result.stdout:
                return "complete"
            raise RuntimeError("submission output did not contain job_id")
        job_id = match.group(1)
        self.log(f"SUBMITTED job_id={job_id}")
        return job_id

    def write_state(self, state: dict) -> None:
        tmp = self.state_path.with_name(f".{self.state_path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, self.state_path)
        finally:
            tmp.unlink(missing_ok=True)

    def supervise(self) -> int:
        lock_path = self.repo / "runs" / ".spur_supervisor.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.log("FATAL another SPUR supervisor is active")
                return 3

            if factory_jobs_active(self.queue()):
                self.log("FATAL pre-existing factory jobs detected")
                return 3

            previous = self.verify()
            stalled = 0
            for wave in range(1, self.args.max_waves + 1):
                if int(previous["remaining_undone"]) == 0:
                    self.log("COMPLETE dataset verification passed")
                    return 0
                job_id = self.submit_wave()
                if job_id == "complete":
                    final = self.verify()
                    return 0 if int(final["remaining_undone"]) == 0 else 4

                self.write_state(
                    {"wave": wave, "job_id": job_id, "before": previous}
                )
                self.wait_for_wave()
                current = self.verify()
                before_score = progress_score(previous)
                after_score = progress_score(current)
                if after_score <= before_score:
                    stalled += 1
                    self.log(
                        f"WARN no durable progress in wave={wave} "
                        f"stalled={stalled}/{self.args.max_stalled_waves}"
                    )
                    if stalled >= self.args.max_stalled_waves:
                        self.log("FATAL progress stalled; refusing further submissions")
                        return 5
                else:
                    stalled = 0
                previous = current
            self.log("FATAL maximum wave count reached before completion")
            return 6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/shasriva/Kore-RL/KORE")
    ap.add_argument("--python", default="/home/shasriva/kore-venv/bin/python")
    ap.add_argument("--data-root", default="/home/shasriva/Kore-RL/KORE/data/b05factory")
    ap.add_argument("--target", type=int, default=3)
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--wave-nodes", type=int, default=64)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--submission-grace-seconds", type=int, default=15)
    ap.add_argument("--queue-timeout-seconds", type=int, default=30)
    ap.add_argument("--visibility-timeout-seconds", type=int, default=120)
    ap.add_argument("--max-scheduler-failures", type=int, default=5)
    ap.add_argument("--max-stalled-waves", type=int, default=2)
    ap.add_argument("--max-waves", type=int, default=12)
    ap.add_argument("--user", default=os.environ.get("USER", ""))
    ap.add_argument(
        "--log",
        default=(
            "/home/shasriva/Kore-RL/KORE/runs/"
            f"spur-supervisor-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.log"
        ),
    )
    args = ap.parse_args()
    for name in (
        "target",
        "shards",
        "wave_nodes",
        "poll_seconds",
        "submission_grace_seconds",
        "queue_timeout_seconds",
        "visibility_timeout_seconds",
        "max_scheduler_failures",
        "max_stalled_waves",
        "max_waves",
    ):
        if getattr(args, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if args.wave_nodes > args.shards:
        args.wave_nodes = args.shards
    if not args.user:
        ap.error("--user is required")
    return Supervisor(args).supervise()


if __name__ == "__main__":
    sys.exit(main())
