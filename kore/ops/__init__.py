"""Operational process, state, and artifact safety helpers.

The training implementation deliberately does not depend on this package.  It is
for launchers, supervisors, and diagnostics that need a small, stdlib-only
control plane around a run.
"""

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
    TaskSetIdentity,
    TerminationResult,
    deprecated_entrypoint,
    new_run_id,
    task_set_identity,
)

__all__ = [
    "ArtifactStatus",
    "IncrementalLogReader",
    "LockBusy",
    "OwnedProcess",
    "ProcessIdentity",
    "RunPhase",
    "SecureRuntime",
    "SecurityError",
    "SupervisorStateMachine",
    "TaskSetIdentity",
    "TerminationResult",
    "deprecated_entrypoint",
    "new_run_id",
    "task_set_identity",
]
