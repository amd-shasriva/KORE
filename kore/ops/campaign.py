"""Owned-process campaign supervision used by quarantined legacy wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import Mapping, Sequence

from .runtime import (
    ArtifactStatus,
    IncrementalLogReader,
    LockBusy,
    OwnedProcess,
    ProcessIdentity,
    RunPhase,
    SecureRuntime,
    SecurityError,
    SupervisorStateMachine,
    capture_process_identity,
    identity_matches,
    new_run_id,
    open_append_log,
)
from .verify import verify_campaign


_STAGE_RE = re.compile(r"STAGE  campaign \(\w+\): stage (start|done): (\w+)")
_ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)| ERROR |out of memory|HIP out of memory|"
    r"CUDA error|HIP error|OutOfMemoryError|CalledProcessError"
)
_GATE_RE = re.compile(
    r"retention[^\n]*(FAIL|regress|below)|GATE[^\n]*FAIL|hard-stop|hard stop",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CampaignSpec:
    repo: Path
    python: Path
    data_root: Path
    command: tuple[str, ...]
    required_stages: tuple[str, ...]
    log_dir: Path
    log_prefix: str
    environment: Mapping[str, str] = field(default_factory=dict)
    poll_seconds: float = 120.0
    cooldown_seconds: float = 90.0
    max_attempts: int = 12
    term_timeout: float = 30.0
    kill_timeout: float = 5.0
    run_id: str | None = None


class CampaignSupervisor:
    """Bounded supervisor whose only child is an identity-owned process group."""

    def __init__(self, spec: CampaignSpec, runtime: SecureRuntime):
        self.spec = spec
        self.runtime = runtime
        self.run_id = spec.run_id or os.environ.get("KORE_RUN_ID") or new_run_id(
            spec.log_prefix
        )
        self.machine = SupervisorStateMachine(spec.max_attempts)
        self.stop_requested = False
        self.child: OwnedProcess | None = None
        self.log_path: Path | None = None
        self.current_stage: tuple[str, str] | None = None
        self.errors = 0
        self.gate_failures = 0

    @property
    def active_relative(self) -> Path:
        return Path("active") / "campaign.json"

    def _log(self, message: str) -> None:
        print(message, flush=True)

    def _validate_dependencies(self) -> ArtifactStatus:
        errors = []
        if not self.spec.repo.is_dir():
            errors.append(f"repository directory is missing: {self.spec.repo}")
        if not self.spec.python.is_file() or not os.access(self.spec.python, os.X_OK):
            errors.append(f"Python interpreter is missing or not executable: {self.spec.python}")
        campaign = self.spec.repo / "scripts" / "run_campaign.py"
        if not campaign.is_file():
            errors.append(f"campaign entrypoint is missing: {campaign}")
        if not self.spec.command:
            errors.append("campaign command is empty")
        if not self.spec.required_stages:
            errors.append("strict verifier stage list is empty")
        return (
            ArtifactStatus.failure(*errors)
            if errors
            else ArtifactStatus.success()
        )

    def _active_state(
        self,
        supervisor: ProcessIdentity,
        *,
        child_state: str | None = None,
    ) -> dict:
        return {
            "schema": 1,
            "run_id": self.run_id,
            "phase": self.machine.phase.value,
            "attempt": self.machine.attempt,
            "reason": self.machine.reason,
            "repo": str(self.spec.repo),
            "data_root": str(self.spec.data_root),
            "required_stages": list(self.spec.required_stages),
            "log_path": str(self.log_path) if self.log_path else None,
            "child_state": child_state,
            "supervisor": supervisor.to_json(),
            "updated_ns": time.time_ns(),
        }

    def _write_active(
        self, supervisor: ProcessIdentity, *, child_state: str | None = None
    ) -> None:
        self.runtime.write_json(
            self.active_relative,
            self._active_state(supervisor, child_state=child_state),
        )

    def _refuse_live_previous_run(self) -> None:
        try:
            previous = self.runtime.read_json(self.active_relative)
        except FileNotFoundError:
            return
        child_state = previous.get("child_state")
        if not isinstance(child_state, str) or not child_state:
            return
        try:
            child_record = self.runtime.read_json(child_state)
            identity_value = child_record.get("identity")
            if not isinstance(identity_value, dict):
                return
            identity = ProcessIdentity.from_json(identity_value)
            matched, _reason = identity_matches(identity)
        except (FileNotFoundError, SecurityError, KeyError, TypeError, ValueError):
            return
        if matched:
            raise SecurityError(
                "a previously owned campaign process is still active; "
                f"inspect or stop state {child_state} instead of starting another"
            )

    def _observe_lines(self, lines: Sequence[str]) -> None:
        for line in lines:
            stages = _STAGE_RE.findall(line)
            if stages:
                stage = stages[-1]
                if stage != self.current_stage:
                    self.current_stage = stage
                    self._log(f"ALERT STAGE {stage[1]}:{stage[0]}")
            new_errors = len(_ERROR_RE.findall(line))
            if new_errors:
                self.errors += new_errors
                self._log(f"ALERT ERROR_DETECTED total={self.errors}")
            new_gates = len(_GATE_RE.findall(line))
            if new_gates:
                self.gate_failures += new_gates
                self._log(f"ALERT RETENTION_GATE total={self.gate_failures}")

    def _run_attempt(
        self, supervisor: ProcessIdentity, attempt: int
    ) -> ArtifactStatus:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.log_path = self.spec.log_dir / (
            f"{self.spec.log_prefix}_{timestamp}_{attempt:02d}_{self.run_id[-8:]}.log"
        )
        child_relative = Path("runs") / self.run_id / f"attempt-{attempt:03d}.json"
        env = os.environ.copy()
        env.update(self.spec.environment)
        self.spec.log_dir.mkdir(parents=True, exist_ok=True)
        with open_append_log(self.log_path) as log_handle:
            self.child = OwnedProcess.spawn(
                self.spec.command,
                run_id=self.run_id,
                runtime=self.runtime,
                state_relative=child_relative,
                cwd=self.spec.repo,
                env=env,
                stdout=log_handle,
            )
            self._write_active(supervisor, child_state=str(child_relative))
            reader = IncrementalLogReader(self.log_path)
            while self.child.poll() is None and not self.stop_requested:
                time.sleep(self.spec.poll_seconds)
                self._observe_lines(reader.read_lines())
                stage = (
                    f"{self.current_stage[1]}:{self.current_stage[0]}"
                    if self.current_stage
                    else "none"
                )
                self._log(
                    f"HEARTBEAT run_id={self.run_id} attempt={attempt} "
                    f"stage={stage} errs={self.errors}"
                )
            if self.stop_requested and self.child.poll() is None:
                result = self.child.terminate(
                    term_timeout=self.spec.term_timeout,
                    kill_timeout=self.spec.kill_timeout,
                )
                if not result.stopped:
                    return ArtifactStatus.failure(
                        f"owned child did not stop: {result.reason}"
                    )
            returncode = self.child.wait()
            self._observe_lines(reader.read_lines(final=True))
        self.machine.process_exited(returncode)
        status = verify_campaign(
            self.spec.repo, self.spec.data_root, self.spec.required_stages
        )
        if self.gate_failures:
            status = ArtifactStatus.failure(
                f"retention gate failure observed ({self.gate_failures})",
                *status.errors,
                details=status.details,
            )
        return status

    def run(self) -> int:
        dependency_status = self._validate_dependencies()
        if not dependency_status.ok:
            self.machine.fail("; ".join(dependency_status.errors))
            for error in dependency_status.errors:
                self._log(f"FATAL {error}")
            return 127
        os.environ["KORE_RUN_ID"] = self.run_id
        supervisor = capture_process_identity(
            os.getpid(),
            self.run_id,
            command=(sys.executable, *sys.argv),
            owns_process_group=False,
            marker_required=False,
        )
        try:
            with self.runtime.lock("campaign-supervisor"):
                self._refuse_live_previous_run()
                self._write_active(supervisor)
                previous_handlers = {
                    signum: signal.signal(
                        signum,
                        lambda _signum, _frame: setattr(self, "stop_requested", True),
                    )
                    for signum in (signal.SIGINT, signal.SIGTERM)
                }
                try:
                    while self.machine.attempt < self.machine.max_attempts:
                        attempt = self.machine.begin_attempt()
                        self.errors = 0
                        self.gate_failures = 0
                        self.current_stage = None
                        self._log(
                            f"ALERT LAUNCH run_id={self.run_id} "
                            f"attempt={attempt}/{self.machine.max_attempts}"
                        )
                        status = self._run_attempt(supervisor, attempt)
                        if self.stop_requested:
                            self.machine.stop("stop requested")
                            self._write_active(supervisor)
                            self._log("ALERT SUPERVISOR_STOPPED")
                            return 130
                        if self.machine.verification_finished(status):
                            self._write_active(supervisor)
                            self._log(
                                f"ALERT CAMPAIGN_COMPLETE run_id={self.run_id} "
                                f"attempt={attempt}"
                            )
                            return 0
                        self._write_active(supervisor)
                        self._log(
                            f"ALERT CAMPAIGN_INCOMPLETE attempt={attempt}: "
                            f"{self.machine.reason}"
                        )
                        if self.machine.phase is RunPhase.GAVE_UP:
                            break
                        time.sleep(self.spec.cooldown_seconds)
                    self._write_active(supervisor)
                    self._log(
                        f"ALERT SUPERVISOR_GIVEUP run_id={self.run_id}: "
                        f"{self.machine.reason}"
                    )
                    return 6
                finally:
                    for signum, handler in previous_handlers.items():
                        signal.signal(signum, handler)
        except LockBusy:
            self._log("FATAL another campaign supervisor holds the runtime lock")
            return 73
        except (OSError, SecurityError, ValueError) as exc:
            if self.machine.phase not in (
                RunPhase.SUCCEEDED,
                RunPhase.GAVE_UP,
                RunPhase.STOPPED,
            ):
                self.machine.fail(str(exc))
            self._log(f"FATAL operational safety check failed: {exc}")
            return 74
