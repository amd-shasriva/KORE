"""Drive resumable SPUR datagen waves to verified completion.

This lightweight login-node supervisor submits exactly one wave at a time,
waits for all ``kore-factory`` children to leave the queue, verifies durable
progress, and repartitions the remaining work. It never cancels jobs and aborts
instead of looping when the scheduler is unavailable or progress stalls.

Two properties keep a flaky control plane from duplicating a live 64-node wave:
a reply that cannot be read is a distinct state from an empty queue (and is
retried, never acted on), and a wave is only considered drained after several
consecutive empty replies. Verification covers exactly the task families
``scripts/spur_partition.py`` shards, so completion and stall detection see all
of the work rather than the breadth slice.
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

try:
    from scripts._kf_verify import PARTITION_TASK_PREFIXES
except ImportError:  # run as `python scripts/spur_supervise_datagen.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts._kf_verify import PARTITION_TASK_PREFIXES

# Tri-state classification of a scheduler reply. "unreadable" exists so a
# control-plane error blob can never be mistaken for "no jobs are queued".
QUEUE_ACTIVE = "active"
QUEUE_EMPTY = "empty"
QUEUE_UNREADABLE = "unreadable"

# Positive markers of a scheduler that could not answer. SPUR's control plane
# intermittently replies to ``squeue`` with an anyhow/tower cause chain instead
# of a job table, sometimes at exit code 0:
#
#     Error: failed to connect to spurctld
#     Caused by:
#         0: transport error
#         3: Connection refused (os error 111)
#
# Detection is deliberately POSITIVE (match the error shape) rather than
# negative (anything that isn't a job table): a genuinely empty SPUR queue may
# legitimately print nothing at all, or a bare header, or a "no jobs" notice,
# and treating any of those as an error would stall the campaign at the end of
# every wave. None of these markers can appear in a squeue row, whose third
# whitespace field is the job name.
_QUEUE_ERROR_RE = re.compile(
    r"""
      ^\s*error\b
    | ^\s*caused \s+ by:
    | \b os \s+ error \s+ \d+
    | \b transport \s+ error \b
    | \b connection \s+ (?: refused | reset | timed \s+ out ) \b
    | \b failed \s+ to \s+ connect \b
    | \b slurm_load_jobs \s+ error \b
    | \b unable \s+ to \s+ contact \b
    | \b socket \s+ timed \s+ out \b
    | \b no \s+ leader \s+ elected \b
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def classify_queue_reply(squeue_output: str) -> str:
    """Classify a ``squeue`` reply as active / empty / unreadable."""
    text = squeue_output or ""
    if _QUEUE_ERROR_RE.search(text):
        return QUEUE_UNREADABLE
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[2].startswith("kore-fac"):
            return QUEUE_ACTIVE
    return QUEUE_EMPTY


def factory_jobs_active(squeue_output: str) -> bool:
    """True only when the reply positively shows a live ``kore-fac*`` job.

    Callers that must not confuse an unreadable scheduler with an empty queue
    use :func:`classify_queue_reply` instead.
    """
    return classify_queue_reply(squeue_output) == QUEUE_ACTIVE


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
        # --prefix must name the SAME families spur_partition.py shards. With the
        # verifier's own genb_ default the completion test and the stall score
        # would only see breadth tasks, so the supervisor would log COMPLETE with
        # vendor/gemm/attention/MoE work outstanding, or score a wave that only
        # moved those families as a stall and stop submitting.
        cleanup = self.state_path.with_suffix(".cleanup.txt")
        result = self.run(
            [
                self.args.python,
                "scripts/_kf_verify.py",
                self.args.data_root,
                str(self.args.target),
                "--prefix",
                self.args.verify_prefix,
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
        # The spur controller intermittently fails to answer: sometimes with a
        # non-zero exit, sometimes with an error blob at rc=0. BOTH are transient
        # failures subject to this bounded retry -- an unreadable reply must never
        # reach the callers as "the queue is empty".
        last = ""
        attempts = self.args.queue_attempts
        for attempt in range(attempts):
            result = self.run(
                ["squeue", "-u", self.args.user],
                check=False,
                timeout=self.args.queue_timeout_seconds,
            )
            unreadable = classify_queue_reply(result.stdout) == QUEUE_UNREADABLE
            if not result.returncode and not unreadable:
                return result.stdout
            reason = "unreadable reply" if unreadable else f"rc={result.returncode}"
            last = reason
            self.log(
                f"WARN squeue transient {reason} "
                f"(attempt {attempt + 1}/{attempts}); retrying"
            )
            time.sleep(min(30, 5 * (attempt + 1)))
        raise RuntimeError(f"squeue failed ({last or '?'}) after {attempts} attempts")

    def queue_state(self) -> str:
        """``QUEUE_ACTIVE`` or ``QUEUE_EMPTY``; raises when the reply is unreadable."""
        return classify_queue_reply(self.queue())

    def wait_for_wave(self) -> None:
        time.sleep(self.args.submission_grace_seconds)
        failures = 0
        empty_polls = 0
        seen_active = False
        needed = self.args.empty_polls_to_finish
        visibility_deadline = time.monotonic() + self.args.visibility_timeout_seconds
        while True:
            try:
                state = self.queue_state()
                failures = 0
            except RuntimeError as exc:
                failures += 1
                # A scheduler we could not read tells us nothing about the wave,
                # so drop any partial debounce rather than counting toward it.
                empty_polls = 0
                self.log(f"WARN scheduler query failed ({failures}): {exc}")
                if failures >= self.args.max_scheduler_failures:
                    raise
                time.sleep(self.args.poll_seconds)
                continue
            if state == QUEUE_ACTIVE:
                seen_active = True
                empty_polls = 0
                self.log("WAIT factory wave remains active")
                time.sleep(self.args.poll_seconds)
                continue
            empty_polls += 1
            if seen_active:
                # Debounce: a single empty poll can also be a scheduler that
                # answered without listing a still-running wave, and returning
                # early there verifies a half-finished wave and repartitions on
                # top of live workers.
                if empty_polls >= needed:
                    self.log(f"WAIT factory wave drained ({empty_polls} empty polls)")
                    return
                self.log(f"WAIT queue empty {empty_polls}/{needed}; debouncing")
                time.sleep(self.args.poll_seconds)
                continue
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

            # queue_state() raises when the scheduler cannot be read, so this
            # guard can only fall through on a reply that positively shows no
            # factory job -- never on an error blob that merely looks empty.
            if self.queue_state() == QUEUE_ACTIVE:
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
    ap.add_argument("--queue-attempts", type=int, default=6)
    ap.add_argument(
        "--empty-polls-to-finish",
        type=int,
        default=3,
        help="consecutive empty scheduler replies required to call a wave done",
    )
    ap.add_argument("--visibility-timeout-seconds", type=int, default=120)
    ap.add_argument("--max-scheduler-failures", type=int, default=5)
    ap.add_argument("--max-stalled-waves", type=int, default=2)
    ap.add_argument("--max-waves", type=int, default=12)
    ap.add_argument(
        "--verify-prefix",
        default=PARTITION_TASK_PREFIXES,
        help="comma-separated task-id prefixes to verify; defaults to the family "
             "set scripts/spur_partition.py shards",
    )
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
        "queue_attempts",
        "empty_polls_to_finish",
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
