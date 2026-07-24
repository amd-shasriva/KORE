from __future__ import annotations

import os
import socket
import struct
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from kore.sandbox.broker import (
    PeerCredentialPolicy,
    UnixBrokerClient,
    encode_frame,
    peer_credentials,
    recv_frame,
)
from kore.sandbox.canonical import (
    canonical_json_bytes,
    output_digest,
    source_digest,
)
from kore.sandbox.config import SandboxConfig
from kore.sandbox.controller import (
    BrokerIsolationController,
    TrustedSubprocessController,
    classify_execution,
    create_isolation_controller,
)
from kore.sandbox.environment import build_candidate_environment
from kore.sandbox.errors import (
    DigestMismatch,
    FrameTooLarge,
    PeerCredentialError,
    PolicyViolation,
    ReplayDetected,
    SandboxConfigurationError,
    SignatureVerificationError,
    SourceTooLarge,
    UnsupportedIsolationMode,
    VerdictVerificationError,
)
from kore.sandbox.models import (
    DigestSet,
    ExecutionKind,
    ExecutionStatus,
    GpuHealthStatus,
    IsolationMode,
    IsolationPolicy,
    ResourceBudget,
    SandboxRequest,
    SandboxResponse,
    SandboxVerdict,
    TrustLevel,
)
from kore.sandbox.signing import (
    EphemeralLocalHMAC,
    NonceRegistry,
    sign_verdict,
    verify_signed_verdict,
)


def _environment(tmp_path: Path) -> dict[str, str]:
    return build_candidate_environment(
        base_environment={"PATH": os.environ.get("PATH", "")},
        private_root=tmp_path / "private",
        project_root=Path(__file__).resolve().parents[1],
        gpu_target="gfx950",
        gpu="0",
    )


def _request(
    tmp_path: Path,
    *,
    policy: IsolationPolicy | None = None,
    source: str = "print('ok')",
    command: str = "print('ok')",
    timeout_seconds: float | None = None,
    nonce: str | None = None,
) -> SandboxRequest:
    policy = policy or IsolationPolicy()
    return SandboxRequest.create(
        task_id="cpu-test",
        task_descriptor={"dtype": "fp32", "shapes": [{"n": 1}]},
        source=source,
        policy=policy,
        toolchain_descriptor={"python": sys.version_info[:3]},
        runtime_descriptor={"target": "cpu-test"},
        execution_kind=ExecutionKind.LEGACY_PYTHON,
        argv=(sys.executable, "-c", command),
        working_directory=str(tmp_path),
        environment=_environment(tmp_path),
        timeout_seconds=timeout_seconds,
        nonce=nonce,
    )


def _broker_policy(*, production: bool = False) -> IsolationPolicy:
    return IsolationPolicy(
        mode=IsolationMode.EXTERNAL_BROKER,
        trust_level=TrustLevel.UNTRUSTED,
        production=production,
        require_signed_verdict=True,
        allow_legacy_python=not production,
        approved_broker_id="test-broker",
    )


def _verdict(
    request: SandboxRequest,
    *,
    status: ExecutionStatus = ExecutionStatus.OK,
    nonce: str | None = None,
    digests: DigestSet | None = None,
) -> SandboxVerdict:
    return SandboxVerdict(
        request_id=request.request_id,
        nonce=nonce or request.nonce,
        status=status,
        digests=digests or request.digests,
        backend=request.policy.approved_broker_id or "trusted-code-only",
        gpu_health=(
            GpuHealthStatus.FAULTED
            if status is ExecutionStatus.GPU_FAULT
            else GpuHealthStatus.UNKNOWN
        ),
        exit_code=0,
        output_digest=output_digest(b"", b""),
    )


def test_candidate_environment_strips_secrets_and_uses_private_paths(tmp_path):
    base = {
        "PATH": "/usr/bin",
        "LANG": "C.UTF-8",
        "OPENAI_API_KEY": "secret",
        "HTTPS_PROXY": "http://proxy",
        "NO_PROXY": "*",
        "SLURM_JOB_ID": "42",
        "SSH_AUTH_SOCK": "/tmp/agent",
        "LD_PRELOAD": "/tmp/inject.so",
        "PYTHONUSERBASE": "/home/user/.local",
        "ROCR_VISIBLE_DEVICES": "7",
        "UNRELATED_AMBIENT": "also-not-inherited",
    }

    env = build_candidate_environment(
        base_environment=base,
        private_root=tmp_path / "private",
        project_root=tmp_path / "repo",
        gpu_target="gfx950",
        gpu="2",
    )

    for forbidden in (
        "OPENAI_API_KEY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "SLURM_JOB_ID",
        "SSH_AUTH_SOCK",
        "LD_PRELOAD",
        "PYTHONUSERBASE",
        "ROCR_VISIBLE_DEVICES",
        "UNRELATED_AMBIENT",
    ):
        assert forbidden not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["HIP_VISIBLE_DEVICES"] == "2"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    roots = {
        Path(env["HOME"]),
        Path(env["TMPDIR"]),
        Path(env["XDG_CACHE_HOME"]),
        Path(env["TRITON_CACHE_DIR"]),
        Path(env["TORCHINDUCTOR_CACHE_DIR"]),
    }
    assert all(path.is_relative_to(tmp_path / "private") for path in roots)
    assert all(path.stat().st_mode & 0o777 == 0o700 for path in roots)
    assert Path(env["HOME"]) != Path(env["TMPDIR"])


def test_canonical_serialization_and_digests_are_deterministic():
    assert canonical_json_bytes({"b": 2, "a": [3, 1]}) == canonical_json_bytes(
        {"a": [3, 1], "b": 2}
    )
    assert source_digest("abc") == source_digest("abc")
    assert source_digest("abc") != source_digest("abc\n")


def test_oversized_outbound_and_inbound_frames_are_rejected():
    with pytest.raises(FrameTooLarge):
        encode_frame(b"x" * 9, max_frame_bytes=8)

    reader, writer = socket.socketpair()
    try:
        writer.sendall(struct.pack("!I", 9))
        with pytest.raises(FrameTooLarge):
            recv_frame(reader, max_frame_bytes=8)
    finally:
        reader.close()
        writer.close()


def test_peer_credentials_are_validated_on_unix_socket():
    left, right = socket.socketpair()
    try:
        credentials = peer_credentials(left)
        PeerCredentialPolicy(allowed_uids=frozenset({os.getuid()})).validate(credentials)
        with pytest.raises(PeerCredentialError):
            PeerCredentialPolicy(
                allowed_uids=frozenset({os.getuid() + 100000})
            ).validate(credentials)
    finally:
        left.close()
        right.close()


def test_oversized_source_is_rejected_before_execution(tmp_path):
    policy = IsolationPolicy(budget=ResourceBudget(max_source_bytes=8))
    with pytest.raises(SourceTooLarge):
        _request(tmp_path, policy=policy, source="x" * 9)


def test_oversized_control_payload_is_rejected(tmp_path):
    policy = IsolationPolicy(
        budget=ResourceBudget(
            max_source_bytes=2048,
            max_control_bytes=1024,
        )
    )
    with pytest.raises(PolicyViolation, match="control payload"):
        _request(tmp_path, policy=policy, source="x" * 900)


def test_trusted_backend_bounds_candidate_output(tmp_path):
    policy = IsolationPolicy(
        budget=ResourceBudget(
            wall_time_seconds=2,
            cpu_time_seconds=2,
            max_output_bytes=128,
        )
    )
    request = _request(
        tmp_path,
        policy=policy,
        command="import sys; sys.stdout.write('x' * 4096); sys.stdout.flush()",
    )

    response = TrustedSubprocessController(policy).execute(request)

    assert response.status is ExecutionStatus.POLICY_VIOLATION
    assert response.verdict.output_truncated
    assert len(response.stdout.encode()) + len(response.stderr.encode()) <= 128


def test_trusted_backend_rejects_non_allowlisted_environment_before_spawn(tmp_path):
    policy = IsolationPolicy()
    request = _request(tmp_path, policy=policy)
    unsafe = replace(
        request,
        environment={**request.environment, "OPENAI_API_KEY": "must-not-cross"},
    )

    response = TrustedSubprocessController(policy).execute(unsafe)

    assert response.status is ExecutionStatus.POLICY_VIOLATION


def test_nonce_signature_tampering_and_replay_are_rejected(tmp_path):
    signer = EphemeralLocalHMAC(b"k" * 32)
    nonces = NonceRegistry()
    nonce = nonces.issue()
    request = _request(tmp_path, policy=_broker_policy(), nonce=nonce)
    signed = sign_verdict(_verdict(request), signer)

    assert verify_signed_verdict(signed, request, signer, nonces).status is ExecutionStatus.OK
    with pytest.raises(ReplayDetected):
        verify_signed_verdict(signed, request, signer, nonces)

    tamper_nonces = NonceRegistry()
    tamper_nonce = tamper_nonces.issue()
    tamper_request = _request(tmp_path, policy=_broker_policy(), nonce=tamper_nonce)
    bad_signature = replace(
        sign_verdict(_verdict(tamper_request), signer),
        signature="00" * 32,
    )
    with pytest.raises(SignatureVerificationError):
        verify_signed_verdict(bad_signature, tamper_request, signer, tamper_nonces)

    wrong_nonce = "f" * 64
    signed_wrong_nonce = sign_verdict(
        _verdict(tamper_request, nonce=wrong_nonce),
        signer,
    )
    with pytest.raises(VerdictVerificationError):
        verify_signed_verdict(
            signed_wrong_nonce,
            tamper_request,
            signer,
            tamper_nonces,
        )


@pytest.mark.parametrize("digest_name", ["task", "source", "policy", "toolchain", "runtime"])
def test_each_verdict_digest_is_bound_to_request(tmp_path, digest_name):
    signer = EphemeralLocalHMAC(b"d" * 32)
    nonces = NonceRegistry()
    nonce = nonces.issue()
    request = _request(tmp_path, policy=_broker_policy(), nonce=nonce)
    bad_digests = replace(request.digests, **{digest_name: "f" * 64})
    signed = sign_verdict(_verdict(request, digests=bad_digests), signer)

    with pytest.raises(DigestMismatch):
        verify_signed_verdict(signed, request, signer, nonces)


def test_unsupported_isolation_mode_fails_closed():
    with pytest.raises(UnsupportedIsolationMode):
        SandboxConfig(mode="wishful-sandbox").policy()


def test_external_mode_requires_explicit_broker_approval(tmp_path):
    config = SandboxConfig(
        mode=IsolationMode.EXTERNAL_BROKER,
        trust_level=TrustLevel.UNTRUSTED,
        require_signed_verdict=True,
        broker_socket=tmp_path / "broker.sock",
        broker_id="test-broker",
        broker_allowed_uids=(os.getuid(),),
        broker_approved=False,
    )
    with pytest.raises(SandboxConfigurationError, match="not explicitly approved"):
        create_isolation_controller(
            config,
            verifier=EphemeralLocalHMAC(b"a" * 32),
        )


def test_untrusted_subprocess_policy_is_rejected():
    with pytest.raises(PolicyViolation):
        IsolationPolicy(
            mode=IsolationMode.TRUSTED_SUBPROCESS,
            trust_level=TrustLevel.UNTRUSTED,
            require_signed_verdict=True,
        )


def test_missing_broker_socket_is_unavailable_not_local_fallback(tmp_path):
    policy = _broker_policy()
    request = _request(tmp_path, policy=policy)
    client = UnixBrokerClient(
        socket_path=tmp_path / "missing.sock",
        broker_id="test-broker",
        peer_policy=PeerCredentialPolicy(allowed_uids=frozenset({os.getuid()})),
    )
    controller = BrokerIsolationController(
        policy=policy,
        client=client,
        verifier=EphemeralLocalHMAC(b"b" * 32),
    )

    response = controller.execute(request)

    assert response.status is ExecutionStatus.BROKER_UNAVAILABLE
    assert not response.attested


def test_broker_controller_accepts_matching_signed_verdict(tmp_path):
    policy = _broker_policy()
    signer = EphemeralLocalHMAC(b"s" * 32)

    class SignedClient:
        broker_id = "test-broker"

        def execute(self, request):
            verdict = _verdict(request)
            return SandboxResponse(
                verdict=verdict,
                signed_verdict=sign_verdict(verdict, signer),
            )

    request = _request(tmp_path, policy=policy)
    response = BrokerIsolationController(
        policy=policy,
        client=SignedClient(),
        verifier=signer,
    ).execute(request)

    assert response.status is ExecutionStatus.OK
    assert response.attested


def test_production_rejects_ephemeral_hmac_verifier():
    class NeverCalledClient:
        broker_id = "test-broker"

        def execute(self, request):
            raise AssertionError("must not execute")

    with pytest.raises(SandboxConfigurationError):
        BrokerIsolationController(
            policy=_broker_policy(production=True),
            client=NeverCalledClient(),
            verifier=EphemeralLocalHMAC(b"b" * 32),
        )


def test_statuses_separate_candidate_timeout_infra_policy_and_gpu_fault():
    assert classify_execution(
        output="SyntaxError", returncode=1
    )[0] is ExecutionStatus.CANDIDATE_ERROR
    assert classify_execution(
        output="", returncode=-9, timed_out=True
    )[0] is ExecutionStatus.TIMEOUT
    assert classify_execution(
        output="hipErrorOutOfMemory", returncode=1
    )[0] is ExecutionStatus.INFRA_ERROR
    assert classify_execution(
        output="", returncode=0, output_limit_exceeded=True
    )[0] is ExecutionStatus.POLICY_VIOLATION
    gpu_status, gpu_health, _ = classify_execution(
        output="amdgpu: VM fault; GPU reset", returncode=1
    )
    assert gpu_status is ExecutionStatus.GPU_FAULT
    assert gpu_health is GpuHealthStatus.FAULTED
    quarantined, quarantine_health, _ = classify_execution(
        output="device has been quarantined", returncode=1
    )
    assert quarantined is ExecutionStatus.GPU_QUARANTINED
    assert quarantine_health is GpuHealthStatus.QUARANTINED


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("ordinary candidate failure", ExecutionStatus.CANDIDATE_ERROR),
        ("hipErrorOutOfMemory", ExecutionStatus.INFRA_ERROR),
        ("amdgpu: VM fault; GPU reset", ExecutionStatus.GPU_FAULT),
        ("device has been quarantined", ExecutionStatus.GPU_QUARANTINED),
    ],
)
def test_trusted_backend_preserves_typed_failure_status(tmp_path, message, expected):
    policy = IsolationPolicy()
    command = f"import sys; sys.stderr.write({message!r}); sys.exit(1)"

    response = TrustedSubprocessController(policy).execute(
        _request(tmp_path, policy=policy, command=command)
    )

    assert response.status is expected


def test_candidate_timeout_is_reported_by_trusted_backend(tmp_path):
    policy = IsolationPolicy(
        budget=ResourceBudget(wall_time_seconds=1, cpu_time_seconds=1)
    )
    request = _request(
        tmp_path,
        policy=policy,
        command="import time; time.sleep(10)",
        timeout_seconds=0.05,
    )

    response = TrustedSubprocessController(policy).execute(request)

    assert response.status is ExecutionStatus.TIMEOUT


def test_trusted_backend_invokes_host_cleanup_hook(tmp_path):
    calls = []

    class RecordingCleanup:
        def cleanup(self, *, pid, status):
            calls.append((pid, status))

    policy = IsolationPolicy()
    response = TrustedSubprocessController(
        policy,
        cleanup_hooks=(RecordingCleanup(),),
    ).execute(_request(tmp_path, policy=policy))

    assert response.status is ExecutionStatus.OK
    assert len(calls) == 1
    assert calls[0][0] > 0
    assert calls[0][1] is ExecutionStatus.OK


def test_trusted_legacy_compatibility_backend_runs_cpu_command(tmp_path):
    policy = IsolationPolicy()
    request = _request(tmp_path, policy=policy, command="print('legacy-ok')")

    response = TrustedSubprocessController(policy).execute(request)

    assert response.status is ExecutionStatus.OK
    assert response.verdict.backend == "trusted-code-only"
    assert response.stdout.strip() == "legacy-ok"
    assert not response.attested
