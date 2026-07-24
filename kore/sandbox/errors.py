"""Typed failures for the repository-side sandbox boundary."""

from __future__ import annotations


class SandboxError(RuntimeError):
    """Base class for sandbox boundary failures."""


class SandboxConfigurationError(SandboxError):
    """The requested isolation policy cannot be satisfied."""


class UnsupportedIsolationMode(SandboxConfigurationError):
    """The configured isolation mode is unknown."""


class PolicyViolation(SandboxError):
    """A request violates its declared isolation policy."""


class SizeLimitExceeded(PolicyViolation):
    """A bounded source, frame, or output exceeded its limit."""


class SourceTooLarge(SizeLimitExceeded):
    """Candidate source exceeds the policy budget."""


class FrameTooLarge(SizeLimitExceeded):
    """A broker control frame exceeds its configured bound."""


class OutputTooLarge(SizeLimitExceeded):
    """Candidate output exceeds its configured bound."""


class BrokerUnavailable(SandboxError):
    """The approved external broker could not be reached."""


class PeerCredentialError(SandboxError):
    """The connected Unix peer is not an approved broker identity."""


class VerdictVerificationError(SandboxError):
    """A signed broker verdict failed verification."""


class SignatureVerificationError(VerdictVerificationError):
    """The verdict signature is absent, unsupported, or invalid."""


class DigestMismatch(VerdictVerificationError):
    """A verdict does not bind to the request digests."""


class NonceError(VerdictVerificationError):
    """Base class for nonce lifecycle failures."""


class UnknownNonce(NonceError):
    """A verdict used a nonce that was never issued."""


class ReplayDetected(NonceError):
    """A previously consumed nonce was presented again."""


class InvalidLaunchPlan(PolicyViolation):
    """A declarative GPU launch plan failed validation."""
