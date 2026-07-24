"""Configuration for the repository side of the sandbox boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from kore.sandbox.models import IsolationMode, IsolationPolicy, ResourceBudget, TrustLevel


def _bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _ids(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class SandboxConfig:
    """Process configuration; call :meth:`policy` for validated semantics."""

    mode: IsolationMode | str = IsolationMode.TRUSTED_SUBPROCESS
    trust_level: TrustLevel | str = TrustLevel.TRUSTED_CODE
    production: bool = False
    require_signed_verdict: bool = False
    allow_legacy_python: bool = True
    broker_socket: Optional[Path] = None
    broker_id: Optional[str] = None
    broker_approved: bool = False
    broker_allowed_uids: tuple[int, ...] = ()
    broker_allowed_gids: tuple[int, ...] = ()
    broker_timeout_seconds: float = 10.0
    broker_max_frame_bytes: int = 4 * 1024 * 1024
    budget: ResourceBudget = field(default_factory=ResourceBudget)

    def __post_init__(self) -> None:
        if self.broker_socket is not None:
            object.__setattr__(self, "broker_socket", Path(self.broker_socket))
        object.__setattr__(
            self, "broker_allowed_uids", tuple(int(uid) for uid in self.broker_allowed_uids)
        )
        object.__setattr__(
            self, "broker_allowed_gids", tuple(int(gid) for gid in self.broker_allowed_gids)
        )
        if self.broker_timeout_seconds <= 0 or self.broker_max_frame_bytes <= 0:
            raise ValueError("broker timeout and frame bound must be positive")

    def policy(self) -> IsolationPolicy:
        return IsolationPolicy(
            mode=self.mode,
            trust_level=self.trust_level,
            production=self.production,
            require_signed_verdict=self.require_signed_verdict,
            allow_legacy_python=self.allow_legacy_python,
            approved_broker_id=self.broker_id,
            budget=self.budget,
        )

    @classmethod
    def from_env(cls, environment: Optional[Mapping[str, str]] = None) -> "SandboxConfig":
        env = os.environ if environment is None else environment
        production = _bool(
            env.get("KORE_SANDBOX_PRODUCTION", "0"),
            name="KORE_SANDBOX_PRODUCTION",
        )
        trust = env.get("KORE_SANDBOX_TRUST", TrustLevel.TRUSTED_CODE.value)
        protected = production or trust == TrustLevel.UNTRUSTED.value
        signed_default = "1" if protected else "0"
        legacy_default = "0" if production else "1"

        budget = ResourceBudget(
            wall_time_seconds=float(env.get("KORE_SANDBOX_WALL_SECONDS", "300")),
            cpu_time_seconds=int(env.get("KORE_SANDBOX_CPU_SECONDS", "300")),
            max_source_bytes=int(env.get("KORE_SANDBOX_MAX_SOURCE_BYTES", str(256 * 1024))),
            max_task_bytes=int(env.get("KORE_SANDBOX_MAX_TASK_BYTES", str(8 * 1024 * 1024))),
            max_control_bytes=int(
                env.get("KORE_SANDBOX_MAX_CONTROL_BYTES", str(1024 * 1024))
            ),
            max_output_bytes=int(
                env.get("KORE_SANDBOX_MAX_OUTPUT_BYTES", str(1024 * 1024))
            ),
            max_file_bytes=int(
                env.get("KORE_SANDBOX_MAX_FILE_BYTES", str(64 * 1024 * 1024))
            ),
            max_rss_bytes=int(
                env.get("KORE_SANDBOX_MAX_RSS_BYTES", str(16 * 1024 * 1024 * 1024))
            ),
            max_processes=int(env.get("KORE_SANDBOX_MAX_PROCESSES", "64")),
            max_open_files=int(env.get("KORE_SANDBOX_MAX_OPEN_FILES", "256")),
            max_scratch_bytes=int(
                env.get("KORE_SANDBOX_MAX_SCRATCH_BYTES", str(1024 * 1024 * 1024))
            ),
            max_launches=int(env.get("KORE_SANDBOX_MAX_LAUNCHES", "128")),
        )
        socket_value = env.get("KORE_SANDBOX_BROKER_SOCKET")
        broker_id = env.get("KORE_SANDBOX_BROKER_ID")
        return cls(
            mode=env.get("KORE_ISOLATION_MODE", IsolationMode.TRUSTED_SUBPROCESS.value),
            trust_level=trust,
            production=production,
            require_signed_verdict=_bool(
                env.get("KORE_SANDBOX_REQUIRE_SIGNED", signed_default),
                name="KORE_SANDBOX_REQUIRE_SIGNED",
            ),
            allow_legacy_python=_bool(
                env.get("KORE_SANDBOX_ALLOW_LEGACY_PYTHON", legacy_default),
                name="KORE_SANDBOX_ALLOW_LEGACY_PYTHON",
            ),
            broker_socket=Path(socket_value) if socket_value else None,
            broker_id=broker_id,
            broker_approved=_bool(
                env.get("KORE_SANDBOX_BROKER_APPROVED", "0"),
                name="KORE_SANDBOX_BROKER_APPROVED",
            ),
            broker_allowed_uids=_ids(env.get("KORE_SANDBOX_BROKER_UIDS", "")),
            broker_allowed_gids=_ids(env.get("KORE_SANDBOX_BROKER_GIDS", "")),
            broker_timeout_seconds=float(
                env.get("KORE_SANDBOX_BROKER_TIMEOUT_SECONDS", "10")
            ),
            broker_max_frame_bytes=int(
                env.get("KORE_SANDBOX_BROKER_MAX_FRAME_BYTES", str(4 * 1024 * 1024))
            ),
            budget=budget,
        )
