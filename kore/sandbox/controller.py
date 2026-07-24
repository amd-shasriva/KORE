"""Isolation controllers and the trusted compatibility subprocess backend."""

from __future__ import annotations

import os
import re
import resource
import selectors
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Protocol, Sequence

from kore.sandbox.broker import (
    BrokerClientProtocol,
    PeerCredentialPolicy,
    UnixBrokerClient,
)
from kore.sandbox.canonical import output_digest
from kore.sandbox.config import SandboxConfig
from kore.sandbox.environment import assert_safe_candidate_environment
from kore.sandbox.errors import (
    BrokerUnavailable,
    OutputTooLarge,
    PeerCredentialError,
    PolicyViolation,
    SandboxConfigurationError,
    SizeLimitExceeded,
    VerdictVerificationError,
)
from kore.sandbox.models import (
    ExecutionKind,
    ExecutionStatus,
    GpuHealthStatus,
    IsolationMode,
    IsolationPolicy,
    SandboxRequest,
    SandboxResponse,
    SandboxVerdict,
)
from kore.sandbox.signing import (
    NonceRegistry,
    VerdictSignatureVerifier,
    verify_signed_verdict,
)


_GPU_QUARANTINE = re.compile(r"\b(?:gpu|device).{0,40}quarantin", re.IGNORECASE | re.DOTALL)
_GPU_FAULT = re.compile(
    r"(?:amdgpu.{0,80}(?:gpu|vm|page) fault|"
    r"HSA_STATUS_ERROR_EXCEPTION|hipErrorHardware|"
    r"uncorrectable.{0,30}ECC|\bXid\b|GPU reset)",
    re.IGNORECASE | re.DOTALL,
)
_INFRA_ERROR = re.compile(
    r"(?:hipError|HIP error|out of memory|No space left on device|"
    r"Input/output error|cannot open shared object file|"
    r"ModuleNotFoundError:.*(?:torch|aiter|triton|rocm)|"
    r"ImportError:.*(?:torch|aiter|triton|rocm|libamdhip|librocm))",
    re.IGNORECASE,
)


class CleanupHook(Protocol):
    """External cgroup/device cleanup hook.

    Implementations are supplied by the host/broker deployment. Repository code
    intentionally does not create or administer cgroups.
    """

    def cleanup(self, *, pid: int, status: ExecutionStatus) -> None:
        ...


class IsolationController(ABC):
    """Execution boundary selected by an :class:`IsolationPolicy`."""

    policy: IsolationPolicy
    backend_label: str

    @abstractmethod
    def execute(self, request: SandboxRequest) -> SandboxResponse:
        """Execute or fail closed with a typed response."""


class TrustedSubprocessController(IsolationController):
    """Compatibility backend for trusted candidate code only.

    Process groups, rlimits, environment filtering, and bounded output are
    hygiene controls. They are not host isolation.
    """

    backend_label = "trusted-code-only"

    def __init__(
        self,
        policy: Optional[IsolationPolicy] = None,
        *,
        cleanup_hooks: Sequence[CleanupHook] = (),
    ):
        self.policy = policy or IsolationPolicy()
        if self.policy.mode is not IsolationMode.TRUSTED_SUBPROCESS:
            raise SandboxConfigurationError("trusted controller requires trusted-subprocess mode")
        if self.policy.production or self.policy.require_signed_verdict:
            raise SandboxConfigurationError(
                "trusted subprocess cannot satisfy production signed-verdict policy"
            )
        self._cleanup_hooks = tuple(cleanup_hooks)

    def execute(self, request: SandboxRequest) -> SandboxResponse:
        if request.policy != self.policy:
            return _failure_response(
                request,
                ExecutionStatus.POLICY_VIOLATION,
                "request policy differs from controller policy",
                self.backend_label,
            )
        if request.execution_kind is not ExecutionKind.LEGACY_PYTHON:
            return _failure_response(
                request,
                ExecutionStatus.POLICY_VIOLATION,
                "trusted subprocess only supports non-production legacy Python",
                self.backend_label,
            )
        try:
            assert_safe_candidate_environment(request.environment)
        except PolicyViolation as exc:
            return _failure_response(
                request,
                ExecutionStatus.POLICY_VIOLATION,
                str(exc),
                self.backend_label,
            )

        started = time.monotonic()
        process: Optional[subprocess.Popen[bytes]] = None
        status = ExecutionStatus.INFRA_ERROR
        health = GpuHealthStatus.UNKNOWN
        message = ""
        stdout = b""
        stderr = b""
        output_truncated = False
        returncode: Optional[int] = None
        try:
            process = subprocess.Popen(
                request.argv,
                cwd=request.working_directory,
                env=dict(request.environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                start_new_session=True,
                preexec_fn=_budget_preexec(request.policy),
            )
            stdout, stderr, timed_out, output_truncated = _communicate_bounded(
                process,
                timeout_seconds=(
                    request.timeout_seconds or request.policy.budget.wall_time_seconds
                ),
                max_output_bytes=request.policy.budget.max_output_bytes,
            )
            returncode = process.returncode
            status, health, message = classify_execution(
                output=(stdout + b"\n" + stderr).decode("utf-8", "replace"),
                returncode=returncode,
                timed_out=timed_out,
                output_limit_exceeded=output_truncated,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            status = ExecutionStatus.INFRA_ERROR
            message = f"unable to start trusted candidate process: {exc}"
        finally:
            if process is not None:
                _kill_process_group(process.pid)
                for hook in self._cleanup_hooks:
                    try:
                        hook.cleanup(pid=process.pid, status=status)
                    except Exception as exc:  # noqa: BLE001 - cleanup must fail closed
                        status = ExecutionStatus.INFRA_ERROR
                        health = GpuHealthStatus.UNKNOWN
                        message = f"cleanup hook failed: {exc}"

        verdict = SandboxVerdict(
            request_id=request.request_id,
            nonce=request.nonce,
            status=status,
            digests=request.digests,
            backend=self.backend_label,
            gpu_health=health,
            exit_code=returncode,
            message=message,
            elapsed_seconds=max(0.0, time.monotonic() - started),
            output_digest=output_digest(stdout, stderr),
            output_truncated=output_truncated,
        )
        return SandboxResponse(
            verdict=verdict,
            stdout=stdout.decode("utf-8", "replace"),
            stderr=stderr.decode("utf-8", "replace"),
        )


class BrokerIsolationController(IsolationController):
    """Fail-closed client for an approved out-of-process broker."""

    def __init__(
        self,
        *,
        policy: IsolationPolicy,
        client: BrokerClientProtocol,
        verifier: Optional[VerdictSignatureVerifier],
        nonces: Optional[NonceRegistry] = None,
    ):
        if policy.mode is not IsolationMode.EXTERNAL_BROKER:
            raise SandboxConfigurationError("broker controller requires external-broker mode")
        if not policy.approved_broker_id or client.broker_id != policy.approved_broker_id:
            raise SandboxConfigurationError("broker client identity is not policy-approved")
        if policy.require_signed_verdict and verifier is None:
            raise SandboxConfigurationError("signed verdict policy requires a verifier")
        if policy.production and not bool(getattr(verifier, "production_approved", False)):
            raise SandboxConfigurationError(
                "production policy requires a production-approved pluggable verifier"
            )
        self.policy = policy
        self.client = client
        self.verifier = verifier
        self.nonces = nonces or NonceRegistry()
        self.backend_label = client.broker_id

    def execute(self, request: SandboxRequest) -> SandboxResponse:
        if request.policy != self.policy:
            return _failure_response(
                request,
                ExecutionStatus.POLICY_VIOLATION,
                "request policy differs from controller policy",
                self.backend_label,
            )
        try:
            self.nonces.register(request.nonce)
        except VerdictVerificationError as exc:
            return _failure_response(
                request,
                ExecutionStatus.INVALID_VERDICT,
                str(exc),
                self.backend_label,
            )

        try:
            response = self.client.execute(request)
        except BrokerUnavailable as exc:
            self.nonces.discard(request.nonce)
            return _failure_response(
                request,
                ExecutionStatus.BROKER_UNAVAILABLE,
                str(exc),
                self.backend_label,
            )
        except (PeerCredentialError, PolicyViolation, SizeLimitExceeded) as exc:
            self.nonces.discard(request.nonce)
            return _failure_response(
                request,
                ExecutionStatus.POLICY_VIOLATION,
                str(exc),
                self.backend_label,
            )
        except Exception as exc:  # noqa: BLE001 - broker faults must fail closed
            self.nonces.discard(request.nonce)
            return _failure_response(
                request,
                ExecutionStatus.BROKER_UNAVAILABLE,
                f"approved broker client failed: {exc}",
                self.backend_label,
            )

        if not isinstance(response, SandboxResponse):
            self.nonces.discard(request.nonce)
            return _failure_response(
                request,
                ExecutionStatus.INVALID_VERDICT,
                "approved broker returned an invalid response type",
                self.backend_label,
            )
        verdict = response.verdict
        if (
            verdict.request_id != request.request_id
            or verdict.nonce != request.nonce
            or verdict.digests != request.digests
            or verdict.backend != self.backend_label
        ):
            self.nonces.discard(request.nonce)
            return _failure_response(
                request,
                ExecutionStatus.INVALID_VERDICT,
                "broker response is not bound to this request and approved broker",
                self.backend_label,
            )

        try:
            stdout_bytes = response.stdout.encode("utf-8")
            stderr_bytes = response.stderr.encode("utf-8")
        except UnicodeEncodeError:
            self.nonces.discard(request.nonce)
            return _failure_response(
                request,
                ExecutionStatus.INVALID_VERDICT,
                "broker output is not valid Unicode",
                self.backend_label,
            )
        output_size = len(stdout_bytes) + len(stderr_bytes)
        if output_size > request.policy.budget.max_output_bytes:
            self.nonces.discard(request.nonce)
            return _failure_response(
                request,
                ExecutionStatus.POLICY_VIOLATION,
                str(OutputTooLarge("broker response exceeded output budget")),
                self.backend_label,
            )

        if request.policy.require_signed_verdict:
            if response.signed_verdict is None or self.verifier is None:
                self.nonces.discard(request.nonce)
                return _failure_response(
                    request,
                    ExecutionStatus.INVALID_VERDICT,
                    "approved broker returned no signed verdict",
                    self.backend_label,
                )
            try:
                verified = verify_signed_verdict(
                    response.signed_verdict,
                    request,
                    self.verifier,
                    self.nonces,
                )
                if response.verdict != verified:
                    raise VerdictVerificationError(
                        "response envelope differs from signed verdict"
                    )
                expected_output = output_digest(stdout_bytes, stderr_bytes)
                if verified.output_digest != expected_output:
                    raise VerdictVerificationError(
                        "signed output digest does not match broker output"
                    )
            except VerdictVerificationError as exc:
                self.nonces.discard(request.nonce)
                return _failure_response(
                    request,
                    ExecutionStatus.INVALID_VERDICT,
                    str(exc),
                    self.backend_label,
                )
        else:
            self.nonces.discard(request.nonce)
        return response


def create_isolation_controller(
    config: SandboxConfig,
    *,
    verifier: Optional[VerdictSignatureVerifier] = None,
    broker_client: Optional[BrokerClientProtocol] = None,
    cleanup_hooks: Sequence[CleanupHook] = (),
) -> IsolationController:
    """Construct exactly the configured backend; never fall back."""

    policy = config.policy()
    if policy.mode is IsolationMode.TRUSTED_SUBPROCESS:
        return TrustedSubprocessController(policy, cleanup_hooks=cleanup_hooks)
    if policy.mode is not IsolationMode.EXTERNAL_BROKER:
        # IsolationPolicy currently catches this first; keep the branch explicit
        # for future enum additions.
        raise SandboxConfigurationError(f"unsupported isolation mode: {policy.mode}")
    if not config.broker_approved:
        raise SandboxConfigurationError("external broker is not explicitly approved")
    if broker_client is None:
        if config.broker_socket is None:
            raise SandboxConfigurationError("external broker socket is not configured")
        try:
            peer_policy = PeerCredentialPolicy(
                allowed_uids=frozenset(config.broker_allowed_uids),
                allowed_gids=frozenset(config.broker_allowed_gids),
            )
        except ValueError as exc:
            raise SandboxConfigurationError(
                "external broker requires approved peer uid/gid credentials"
            ) from exc
        try:
            broker_client = UnixBrokerClient(
                socket_path=config.broker_socket,
                broker_id=policy.approved_broker_id or "",
                peer_policy=peer_policy,
                timeout_seconds=config.broker_timeout_seconds,
                max_frame_bytes=config.broker_max_frame_bytes,
            )
        except ValueError as exc:
            raise SandboxConfigurationError("invalid external broker configuration") from exc
    return BrokerIsolationController(
        policy=policy,
        client=broker_client,
        verifier=verifier,
    )


def classify_execution(
    *,
    output: str,
    returncode: Optional[int],
    timed_out: bool = False,
    output_limit_exceeded: bool = False,
) -> tuple[ExecutionStatus, GpuHealthStatus, str]:
    """Classify candidate, timeout, infra, policy, and GPU outcomes distinctly."""

    if output_limit_exceeded:
        return (
            ExecutionStatus.POLICY_VIOLATION,
            GpuHealthStatus.UNKNOWN,
            "candidate output exceeded policy budget",
        )
    if timed_out:
        return ExecutionStatus.TIMEOUT, GpuHealthStatus.UNKNOWN, "candidate timed out"
    if _GPU_QUARANTINE.search(output):
        return (
            ExecutionStatus.GPU_QUARANTINED,
            GpuHealthStatus.QUARANTINED,
            _message_tail(output),
        )
    if _GPU_FAULT.search(output):
        return ExecutionStatus.GPU_FAULT, GpuHealthStatus.FAULTED, _message_tail(output)
    if (
        _INFRA_ERROR.search(output)
        or (returncode is not None and returncode < 0)
        or returncode == 137
    ):
        return ExecutionStatus.INFRA_ERROR, GpuHealthStatus.UNKNOWN, _message_tail(output)
    if returncode not in (0, None):
        return ExecutionStatus.CANDIDATE_ERROR, GpuHealthStatus.UNKNOWN, _message_tail(output)
    if returncode is None:
        return ExecutionStatus.INFRA_ERROR, GpuHealthStatus.UNKNOWN, "process had no exit status"
    return ExecutionStatus.OK, GpuHealthStatus.UNKNOWN, ""


def _failure_response(
    request: SandboxRequest,
    status: ExecutionStatus,
    message: str,
    backend: str,
) -> SandboxResponse:
    verdict = SandboxVerdict(
        request_id=request.request_id,
        nonce=request.nonce,
        status=status,
        digests=request.digests,
        backend=backend,
        message=message,
        output_digest=output_digest(b"", b""),
    )
    return SandboxResponse(verdict=verdict)


def _budget_preexec(policy: IsolationPolicy):
    budget = policy.budget

    def apply() -> None:  # pragma: no cover - runs in child
        os.umask(0o077)
        _set_soft_limit(resource.RLIMIT_CPU, budget.cpu_time_seconds)
        _set_soft_limit(resource.RLIMIT_FSIZE, budget.max_file_bytes)
        _set_soft_limit(resource.RLIMIT_NOFILE, budget.max_open_files)
        if hasattr(resource, "RLIMIT_RSS"):
            _set_soft_limit(resource.RLIMIT_RSS, budget.max_rss_bytes)
        _set_soft_limit(resource.RLIMIT_CORE, 0)
        # RLIMIT_NPROC is per UID and therefore cannot safely implement a
        # per-candidate process budget here. The external broker must use cgroup
        # pids.max. RLIMIT_AS is also omitted because ROCm reserves large VA ranges.

    return apply


def _set_soft_limit(kind: int, requested: int) -> None:  # pragma: no cover - child only
    try:
        _, hard = resource.getrlimit(kind)
        value = int(requested)
        if hard != resource.RLIM_INFINITY:
            value = min(value, int(hard))
        resource.setrlimit(kind, (value, hard))
    except (OSError, ValueError):
        pass


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _communicate_bounded(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
    max_output_bytes: int,
) -> tuple[bytes, bytes, bool, bool]:
    """Drain stdout/stderr without ever buffering beyond the combined cap."""

    if process.stdout is None or process.stderr is None:
        raise RuntimeError("bounded communication requires stdout/stderr pipes")
    selector = selectors.DefaultSelector()
    streams = {process.stdout.fileno(): ("stdout", process.stdout), process.stderr.fileno(): ("stderr", process.stderr)}
    for fd, (name, stream) in streams.items():
        os.set_blocking(fd, False)
        selector.register(stream, selectors.EVENT_READ, data=name)

    chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    total = 0
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    exceeded = False
    killed_at: Optional[float] = None
    try:
        while selector.get_map():
            now = time.monotonic()
            if not timed_out and not exceeded and now >= deadline:
                timed_out = True
                _kill_process_group(process.pid)
                killed_at = now
            wait = 0.05
            if not timed_out and not exceeded:
                wait = max(0.0, min(wait, deadline - now))
            events = selector.select(wait)
            for key, _ in events:
                try:
                    data = os.read(key.fd, 64 * 1024)
                except BlockingIOError:
                    continue
                if not data:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                available = max(0, max_output_bytes - total)
                if available:
                    kept = data[:available]
                    chunks[key.data].append(kept)
                    total += len(kept)
                if len(data) > available:
                    exceeded = True
                    _kill_process_group(process.pid)
                    killed_at = killed_at or time.monotonic()
            if process.poll() is not None and not events:
                # EOF notifications normally arrive on the next selector tick.
                continue
            if killed_at is not None and time.monotonic() - killed_at > 1.0:
                break
    finally:
        for key in list(selector.get_map().values()):
            try:
                selector.unregister(key.fileobj)
            except Exception:
                pass
            try:
                key.fileobj.close()
            except Exception:
                pass
        selector.close()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _kill_process_group(process.pid)
            timed_out = True
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # An uninterruptible host process is an infrastructure failure;
                # the external cgroup cleanup hook remains responsible for it.
                pass
    return b"".join(chunks["stdout"]), b"".join(chunks["stderr"]), timed_out, exceeded


def _message_tail(output: str, length: int = 800) -> str:
    stripped = output.strip()
    if not stripped:
        return "candidate process failed"
    return stripped[-length:]
