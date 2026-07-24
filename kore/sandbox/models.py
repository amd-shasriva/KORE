"""Typed policy, request, response, verdict, and status models."""

from __future__ import annotations

import math
import re
import secrets
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from kore.sandbox.canonical import (
    canonical_json_bytes,
    policy_digest,
    runtime_digest,
    source_digest,
    task_digest,
    toolchain_digest,
)
from kore.sandbox.errors import (
    DigestMismatch,
    PolicyViolation,
    SourceTooLarge,
    UnsupportedIsolationMode,
)


_HEX256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9._:@/+~-]+$")


class IsolationMode(str, Enum):
    """Repository-visible execution modes.

    The local subprocess mode is a compatibility backend for trusted code. It
    is not a security sandbox.
    """

    TRUSTED_SUBPROCESS = "trusted-subprocess"
    EXTERNAL_BROKER = "external-broker"


class TrustLevel(str, Enum):
    TRUSTED_CODE = "trusted-code"
    UNTRUSTED = "untrusted"


class ExecutionKind(str, Enum):
    LEGACY_PYTHON = "legacy-python"
    HSACO_LAUNCH_PLAN = "hsaco-launch-plan"


class ExecutionStatus(str, Enum):
    OK = "ok"
    CANDIDATE_ERROR = "candidate-error"
    TIMEOUT = "timeout"
    INFRA_ERROR = "infra-error"
    POLICY_VIOLATION = "policy-violation"
    GPU_FAULT = "gpu-fault"
    GPU_QUARANTINED = "gpu-quarantined"
    BROKER_UNAVAILABLE = "broker-unavailable"
    UNSUPPORTED_ISOLATION = "unsupported-isolation"
    INVALID_VERDICT = "invalid-verdict"


class GpuHealthStatus(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULTED = "faulted"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class ResourceBudget:
    """Bounded resources carried with every candidate request.

    The trusted local backend enforces wall/CPU/file/open-file/output limits.
    Memory, process-count, GPU, and cgroup limits require the external broker;
    their presence here is a contract, not a claim of local enforcement.
    """

    wall_time_seconds: float = 300.0
    cpu_time_seconds: int = 300
    max_source_bytes: int = 256 * 1024
    max_task_bytes: int = 8 * 1024 * 1024
    max_control_bytes: int = 1024 * 1024
    max_output_bytes: int = 1024 * 1024
    max_file_bytes: int = 64 * 1024 * 1024
    max_rss_bytes: int = 16 * 1024 * 1024 * 1024
    max_processes: int = 64
    max_open_files: int = 256
    max_scratch_bytes: int = 1024 * 1024 * 1024
    max_launches: int = 128

    def __post_init__(self) -> None:
        values = {
            "wall_time_seconds": self.wall_time_seconds,
            "cpu_time_seconds": self.cpu_time_seconds,
            "max_source_bytes": self.max_source_bytes,
            "max_task_bytes": self.max_task_bytes,
            "max_control_bytes": self.max_control_bytes,
            "max_output_bytes": self.max_output_bytes,
            "max_file_bytes": self.max_file_bytes,
            "max_rss_bytes": self.max_rss_bytes,
            "max_processes": self.max_processes,
            "max_open_files": self.max_open_files,
            "max_scratch_bytes": self.max_scratch_bytes,
            "max_launches": self.max_launches,
        }
        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
            if name != "wall_time_seconds" and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.max_control_bytes < 1024:
            raise ValueError("max_control_bytes must be at least 1024")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResourceBudget":
        return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})


@dataclass(frozen=True)
class IsolationPolicy:
    """Fail-closed execution policy."""

    mode: IsolationMode | str = IsolationMode.TRUSTED_SUBPROCESS
    trust_level: TrustLevel | str = TrustLevel.TRUSTED_CODE
    production: bool = False
    require_signed_verdict: bool = False
    allow_legacy_python: bool = True
    approved_broker_id: Optional[str] = None
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    schema_version: int = 1

    def __post_init__(self) -> None:
        try:
            mode = self.mode if isinstance(self.mode, IsolationMode) else IsolationMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise UnsupportedIsolationMode(f"unsupported isolation mode: {self.mode!r}") from exc
        try:
            trust = (
                self.trust_level
                if isinstance(self.trust_level, TrustLevel)
                else TrustLevel(self.trust_level)
            )
        except (TypeError, ValueError) as exc:
            raise PolicyViolation(f"unsupported trust level: {self.trust_level!r}") from exc
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "trust_level", trust)
        if isinstance(self.budget, Mapping):
            object.__setattr__(self, "budget", ResourceBudget.from_dict(self.budget))
        elif not isinstance(self.budget, ResourceBudget):
            raise PolicyViolation("policy budget must be a ResourceBudget")

        if self.schema_version != 1:
            raise PolicyViolation(f"unsupported policy schema: {self.schema_version}")
        protected = self.production or trust is TrustLevel.UNTRUSTED
        if protected and mode is not IsolationMode.EXTERNAL_BROKER:
            raise PolicyViolation(
                "production/untrusted candidates require external-broker isolation"
            )
        if protected and not self.require_signed_verdict:
            raise PolicyViolation(
                "production/untrusted candidates require a signed broker verdict"
            )
        if self.production and self.allow_legacy_python:
            raise PolicyViolation("legacy Python execution is non-production only")
        if mode is IsolationMode.EXTERNAL_BROKER and not self.approved_broker_id:
            raise PolicyViolation("external-broker mode requires an approved broker id")

    @property
    def backend_label(self) -> str:
        if self.mode is IsolationMode.TRUSTED_SUBPROCESS:
            return "trusted-code-only"
        return "external-broker"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IsolationPolicy":
        data = dict(value)
        if "budget" in data and not isinstance(data["budget"], ResourceBudget):
            data["budget"] = ResourceBudget.from_dict(data["budget"])
        return cls(**data)


@dataclass(frozen=True)
class DigestSet:
    task: str
    source: str
    policy: str
    toolchain: str
    runtime: str

    def __post_init__(self) -> None:
        for name in ("task", "source", "policy", "toolchain", "runtime"):
            if not _HEX256.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    @classmethod
    def compute(
        cls,
        *,
        task_id: str,
        task_descriptor: Mapping[str, Any],
        source: str,
        policy: IsolationPolicy,
        toolchain_descriptor: Mapping[str, Any],
        runtime_descriptor: Mapping[str, Any],
    ) -> "DigestSet":
        return cls(
            task=task_digest(task_id, task_descriptor),
            source=source_digest(source),
            policy=policy_digest(policy),
            toolchain=toolchain_digest(toolchain_descriptor),
            runtime=runtime_digest(runtime_descriptor),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DigestSet":
        return cls(**{name: str(value[name]) for name in cls.__dataclass_fields__})


def fresh_nonce() -> str:
    """Generate a 256-bit request nonce without registering it."""

    return secrets.token_hex(32)


@dataclass(frozen=True)
class SandboxRequest:
    """Canonical candidate execution request.

    ``argv``, ``working_directory``, and ``environment`` exist solely for the
    trusted compatibility backend. Production launch-plan requests must not
    carry those host-process controls.
    """

    request_id: str
    nonce: str
    task_id: str
    source: str
    digests: DigestSet
    policy: IsolationPolicy
    execution_kind: ExecutionKind | str
    argv: tuple[str, ...] = ()
    working_directory: Optional[str] = None
    environment: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    launch_plan: Optional[Mapping[str, Any]] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        try:
            kind = (
                self.execution_kind
                if isinstance(self.execution_kind, ExecutionKind)
                else ExecutionKind(self.execution_kind)
            )
        except (TypeError, ValueError) as exc:
            raise PolicyViolation(f"unsupported execution kind: {self.execution_kind!r}") from exc
        object.__setattr__(self, "execution_kind", kind)
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "environment", dict(self.environment))

        if self.schema_version != 1:
            raise PolicyViolation(f"unsupported request schema: {self.schema_version}")
        for name, value in (
            ("request_id", self.request_id),
            ("nonce", self.nonce),
            ("task_id", self.task_id),
        ):
            if not value or len(value) > 256 or not _TOKEN.fullmatch(value):
                raise PolicyViolation(f"invalid {name}")
        if len(self.nonce) < 32:
            raise PolicyViolation("nonce must contain at least 128 bits of encoded entropy")
        source_bytes = len(self.source.encode("utf-8"))
        if source_bytes > self.policy.budget.max_source_bytes:
            raise SourceTooLarge(
                f"candidate source is {source_bytes} bytes; "
                f"limit is {self.policy.budget.max_source_bytes}"
            )
        if self.digests.source != source_digest(self.source):
            raise DigestMismatch("request source digest does not match source")
        if self.digests.policy != policy_digest(self.policy):
            raise DigestMismatch("request policy digest does not match policy")
        for arg in self.argv:
            if not isinstance(arg, str) or "\x00" in arg:
                raise PolicyViolation("argv entries must be NUL-free strings")
        for key, value in self.environment.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, str)
                or "\x00" in key
                or "\x00" in value
            ):
                raise PolicyViolation("environment must contain NUL-free strings")
        if self.timeout_seconds is not None:
            if (
                isinstance(self.timeout_seconds, bool)
                or not math.isfinite(float(self.timeout_seconds))
                or self.timeout_seconds <= 0
                or self.timeout_seconds > self.policy.budget.wall_time_seconds
            ):
                raise PolicyViolation(
                    "request timeout must be positive and within the wall-time budget"
                )

        if kind is ExecutionKind.LEGACY_PYTHON:
            if not self.policy.allow_legacy_python:
                raise PolicyViolation("legacy Python execution is disabled by policy")
            if not self.argv or not self.working_directory:
                raise PolicyViolation("legacy Python execution requires argv and working_directory")
            if self.launch_plan is not None:
                raise PolicyViolation("legacy Python request cannot include a launch plan")
        else:
            if self.launch_plan is None:
                raise PolicyViolation("HSACO execution requires a launch plan")
            from kore.sandbox.launch_plan import LaunchCaps, LaunchPlan

            plan = (
                self.launch_plan
                if isinstance(self.launch_plan, LaunchPlan)
                else LaunchPlan.from_dict(self.launch_plan)
            )
            plan.validate(
                LaunchCaps(
                    max_launches=self.policy.budget.max_launches,
                    max_scratch_bytes_per_launch=self.policy.budget.max_scratch_bytes,
                    max_total_scratch_bytes=self.policy.budget.max_scratch_bytes,
                )
            )
            object.__setattr__(self, "launch_plan", plan.to_dict())
            if self.policy.production and (
                self.argv or self.working_directory is not None or self.environment
            ):
                raise PolicyViolation(
                    "production launch plans cannot carry host argv/cwd/environment"
                )

        control_size = len(canonical_json_bytes(self.to_dict()))
        if control_size > self.policy.budget.max_control_bytes:
            raise PolicyViolation(
                f"request control payload is {control_size} bytes; "
                f"limit is {self.policy.budget.max_control_bytes}"
            )

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        task_descriptor: Mapping[str, Any],
        source: str,
        policy: IsolationPolicy,
        toolchain_descriptor: Mapping[str, Any],
        runtime_descriptor: Mapping[str, Any],
        execution_kind: ExecutionKind = ExecutionKind.LEGACY_PYTHON,
        request_id: Optional[str] = None,
        nonce: Optional[str] = None,
        argv: tuple[str, ...] = (),
        working_directory: Optional[str] = None,
        environment: Optional[Mapping[str, str]] = None,
        timeout_seconds: Optional[float] = None,
        launch_plan: Optional[Mapping[str, Any]] = None,
    ) -> "SandboxRequest":
        return cls(
            request_id=request_id or uuid.uuid4().hex,
            nonce=nonce or fresh_nonce(),
            task_id=task_id,
            source=source,
            digests=DigestSet.compute(
                task_id=task_id,
                task_descriptor=task_descriptor,
                source=source,
                policy=policy,
                toolchain_descriptor=toolchain_descriptor,
                runtime_descriptor=runtime_descriptor,
            ),
            policy=policy,
            execution_kind=execution_kind,
            argv=argv,
            working_directory=working_directory,
            environment=environment or {},
            timeout_seconds=timeout_seconds,
            launch_plan=launch_plan,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "nonce": self.nonce,
            "task_id": self.task_id,
            "source": self.source,
            "digests": self.digests,
            "policy": self.policy,
            "execution_kind": self.execution_kind,
            "argv": self.argv,
            "working_directory": self.working_directory,
            "environment": self.environment,
            "timeout_seconds": self.timeout_seconds,
            "launch_plan": self.launch_plan,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SandboxRequest":
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            request_id=str(value["request_id"]),
            nonce=str(value["nonce"]),
            task_id=str(value["task_id"]),
            source=str(value["source"]),
            digests=DigestSet.from_dict(value["digests"]),
            policy=IsolationPolicy.from_dict(value["policy"]),
            execution_kind=str(value["execution_kind"]),
            argv=tuple(str(arg) for arg in value.get("argv", ())),
            working_directory=(
                str(value["working_directory"])
                if value.get("working_directory") is not None
                else None
            ),
            environment={
                str(key): str(val) for key, val in value.get("environment", {}).items()
            },
            timeout_seconds=(
                float(value["timeout_seconds"])
                if value.get("timeout_seconds") is not None
                else None
            ),
            launch_plan=value.get("launch_plan"),
        )


@dataclass(frozen=True)
class SandboxVerdict:
    request_id: str
    nonce: str
    status: ExecutionStatus | str
    digests: DigestSet
    backend: str
    gpu_health: GpuHealthStatus | str = GpuHealthStatus.UNKNOWN
    exit_code: Optional[int] = None
    message: str = ""
    elapsed_seconds: float = 0.0
    output_digest: Optional[str] = None
    output_truncated: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        try:
            status = self.status if isinstance(self.status, ExecutionStatus) else ExecutionStatus(self.status)
            health = (
                self.gpu_health
                if isinstance(self.gpu_health, GpuHealthStatus)
                else GpuHealthStatus(self.gpu_health)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid verdict status") from exc
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "gpu_health", health)
        if self.schema_version != 1:
            raise ValueError(f"unsupported verdict schema: {self.schema_version}")
        if not self.backend or len(self.backend) > 128:
            raise ValueError("invalid verdict backend")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if self.output_digest is not None and not _HEX256.fullmatch(self.output_digest):
            raise ValueError("output_digest must be a lowercase SHA-256 digest")
        if status is ExecutionStatus.GPU_FAULT and health not in (
            GpuHealthStatus.FAULTED,
            GpuHealthStatus.DEGRADED,
        ):
            raise ValueError("GPU fault verdict requires faulted/degraded GPU health")
        if (
            status is ExecutionStatus.GPU_QUARANTINED
            and health is not GpuHealthStatus.QUARANTINED
        ):
            raise ValueError("GPU quarantine verdict requires quarantined GPU health")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "nonce": self.nonce,
            "status": self.status,
            "digests": self.digests,
            "backend": self.backend,
            "gpu_health": self.gpu_health,
            "exit_code": self.exit_code,
            "message": self.message,
            "elapsed_seconds": self.elapsed_seconds,
            "output_digest": self.output_digest,
            "output_truncated": self.output_truncated,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SandboxVerdict":
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            request_id=str(value["request_id"]),
            nonce=str(value["nonce"]),
            status=str(value["status"]),
            digests=DigestSet.from_dict(value["digests"]),
            backend=str(value["backend"]),
            gpu_health=str(value.get("gpu_health", GpuHealthStatus.UNKNOWN.value)),
            exit_code=(
                int(value["exit_code"]) if value.get("exit_code") is not None else None
            ),
            message=str(value.get("message", "")),
            elapsed_seconds=float(value.get("elapsed_seconds", 0.0)),
            output_digest=(
                str(value["output_digest"])
                if value.get("output_digest") is not None
                else None
            ),
            output_truncated=bool(value.get("output_truncated", False)),
        )


@dataclass(frozen=True)
class SignedVerdict:
    verdict: SandboxVerdict
    key_id: str
    algorithm: str
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.to_dict(),
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SignedVerdict":
        return cls(
            verdict=SandboxVerdict.from_dict(value["verdict"]),
            key_id=str(value["key_id"]),
            algorithm=str(value["algorithm"]),
            signature=str(value["signature"]),
        )


@dataclass(frozen=True)
class SandboxResponse:
    verdict: SandboxVerdict
    stdout: str = ""
    stderr: str = ""
    signed_verdict: Optional[SignedVerdict] = None

    @property
    def status(self) -> ExecutionStatus:
        return self.verdict.status

    @property
    def attested(self) -> bool:
        return self.signed_verdict is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.to_dict(),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "signed_verdict": (
                self.signed_verdict.to_dict() if self.signed_verdict is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SandboxResponse":
        signed_raw = value.get("signed_verdict")
        return cls(
            verdict=SandboxVerdict.from_dict(value["verdict"]),
            stdout=str(value.get("stdout", "")),
            stderr=str(value.get("stderr", "")),
            signed_verdict=(
                SignedVerdict.from_dict(signed_raw) if signed_raw is not None else None
            ),
        )
