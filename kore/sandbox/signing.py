"""Nonce lifecycle and pluggable signed-verdict verification."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import OrderedDict
from typing import Protocol

from kore.sandbox.canonical import canonical_json_bytes
from kore.sandbox.errors import (
    DigestMismatch,
    ReplayDetected,
    SignatureVerificationError,
    UnknownNonce,
    VerdictVerificationError,
)
from kore.sandbox.models import (
    ExecutionStatus,
    GpuHealthStatus,
    SandboxRequest,
    SandboxVerdict,
    SignedVerdict,
)


EPHEMERAL_HMAC_ALGORITHM = "hmac-sha256-ephemeral-local"


class VerdictSignatureVerifier(Protocol):
    """Production-pluggable signature verifier.

    Production implementations should bind ``key_id`` to a managed asymmetric
    key or hardware-backed service. Repository code does not provide that key
    management.
    """

    production_approved: bool

    def verify(
        self,
        payload: bytes,
        signature: bytes,
        *,
        key_id: str,
        algorithm: str,
    ) -> bool:
        ...


class VerdictSigner(Protocol):
    """Signer interface implemented by the out-of-process broker."""

    key_id: str
    algorithm: str
    production_approved: bool

    def sign(self, payload: bytes) -> bytes:
        ...


class EphemeralLocalHMAC:
    """In-memory HMAC signer/verifier for local tests only.

    It is intentionally marked ``production_approved = False``. A shared secret
    in the client process cannot establish a production trust boundary.
    """

    algorithm = EPHEMERAL_HMAC_ALGORITHM
    production_approved = False

    def __init__(self, key: bytes | None = None, *, key_id: str = "ephemeral-local"):
        self._key = key or secrets.token_bytes(32)
        if len(self._key) < 32:
            raise ValueError("ephemeral HMAC key must contain at least 256 bits")
        self.key_id = key_id

    def sign(self, payload: bytes) -> bytes:
        return hmac.new(self._key, payload, hashlib.sha256).digest()

    def verify(
        self,
        payload: bytes,
        signature: bytes,
        *,
        key_id: str,
        algorithm: str,
    ) -> bool:
        if key_id != self.key_id or algorithm != self.algorithm:
            return False
        expected = self.sign(payload)
        return hmac.compare_digest(expected, signature)


class NonceRegistry:
    """Thread-safe, bounded one-shot nonce registry."""

    def __init__(self, *, ttl_seconds: float = 600.0, max_entries: int = 8192):
        if ttl_seconds <= 0 or max_entries <= 0:
            raise ValueError("nonce ttl and capacity must be positive")
        self._ttl = float(ttl_seconds)
        self._max_entries = int(max_entries)
        self._pending: OrderedDict[str, float] = OrderedDict()
        self._consumed: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def issue(self) -> str:
        nonce = secrets.token_hex(32)
        self.register(nonce)
        return nonce

    def register(self, nonce: str) -> None:
        if len(nonce) < 32:
            raise ValueError("nonce is too short")
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            if nonce in self._consumed or nonce in self._pending:
                raise ReplayDetected("nonce was already registered or consumed")
            self._pending[nonce] = now + self._ttl
            while len(self._pending) > self._max_entries:
                self._pending.popitem(last=False)

    def consume(self, nonce: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            if nonce in self._consumed:
                raise ReplayDetected("signed verdict nonce was already consumed")
            if nonce not in self._pending:
                raise UnknownNonce("signed verdict nonce was not issued")
            self._pending.pop(nonce)
            self._consumed[nonce] = now + self._ttl
            while len(self._consumed) > self._max_entries:
                self._consumed.popitem(last=False)

    def discard(self, nonce: str) -> None:
        """Forget a request that ended without an accepted verdict."""

        with self._lock:
            self._pending.pop(nonce, None)

    def _purge(self, now: float) -> None:
        for registry in (self._pending, self._consumed):
            while registry:
                _, expiry = next(iter(registry.items()))
                if expiry > now:
                    break
                registry.popitem(last=False)


def sign_verdict(verdict: SandboxVerdict, signer: VerdictSigner) -> SignedVerdict:
    payload = canonical_json_bytes(verdict.to_dict())
    signature = signer.sign(payload)
    return SignedVerdict(
        verdict=verdict,
        key_id=signer.key_id,
        algorithm=signer.algorithm,
        signature=signature.hex(),
    )


def verify_signed_verdict(
    signed: SignedVerdict,
    request: SandboxRequest,
    verifier: VerdictSignatureVerifier,
    nonces: NonceRegistry,
) -> SandboxVerdict:
    """Verify signature, request binding, GPU status consistency, and nonce.

    The nonce is consumed only after all cryptographic and semantic checks pass.
    """

    try:
        signature = bytes.fromhex(signed.signature)
    except ValueError as exc:
        raise SignatureVerificationError("verdict signature is not hexadecimal") from exc
    payload = canonical_json_bytes(signed.verdict.to_dict())
    try:
        valid_signature = verifier.verify(
            payload,
            signature,
            key_id=signed.key_id,
            algorithm=signed.algorithm,
        )
    except Exception as exc:  # noqa: BLE001 - verifier failures must fail closed
        raise SignatureVerificationError("verdict verifier failed") from exc
    if not valid_signature:
        raise SignatureVerificationError("verdict signature verification failed")

    verdict = signed.verdict
    if verdict.request_id != request.request_id:
        raise VerdictVerificationError("verdict request id does not match request")
    if verdict.nonce != request.nonce:
        raise VerdictVerificationError("verdict nonce does not match request")
    if verdict.digests != request.digests:
        raise DigestMismatch("verdict digest set does not match request")
    if (
        request.policy.approved_broker_id is not None
        and verdict.backend != request.policy.approved_broker_id
    ):
        raise VerdictVerificationError("verdict was not issued by the approved broker")
    if verdict.status is ExecutionStatus.GPU_FAULT and verdict.gpu_health not in (
        GpuHealthStatus.FAULTED,
        GpuHealthStatus.DEGRADED,
    ):
        raise VerdictVerificationError("GPU fault verdict lacks faulted GPU health")
    if (
        verdict.status is ExecutionStatus.GPU_QUARANTINED
        and verdict.gpu_health is not GpuHealthStatus.QUARANTINED
    ):
        raise VerdictVerificationError("GPU quarantine verdict lacks quarantined health")

    nonces.consume(verdict.nonce)
    return verdict
