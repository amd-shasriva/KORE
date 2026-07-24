"""Bounded Unix-socket client protocol for an external isolation broker."""

from __future__ import annotations

import json
import os
import socket
import stat
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from kore.sandbox.canonical import canonical_json_bytes
from kore.sandbox.errors import (
    BrokerUnavailable,
    FrameTooLarge,
    OutputTooLarge,
    PeerCredentialError,
    PolicyViolation,
)
from kore.sandbox.models import SandboxRequest, SandboxResponse


FRAME_HEADER = struct.Struct("!I")
DEFAULT_MAX_FRAME_BYTES = 4 * 1024 * 1024


def encode_frame(payload: bytes, *, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> bytes:
    if not payload:
        raise PolicyViolation("empty broker frames are not permitted")
    if len(payload) > max_frame_bytes:
        raise FrameTooLarge(
            f"broker frame is {len(payload)} bytes; limit is {max_frame_bytes}"
        )
    return FRAME_HEADER.pack(len(payload)) + payload


def send_frame(
    sock: socket.socket,
    payload: bytes,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> None:
    sock.sendall(encode_frame(payload, max_frame_bytes=max_frame_bytes))


def recv_frame(
    sock: socket.socket,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> bytes:
    header = _recv_exact(sock, FRAME_HEADER.size)
    (length,) = FRAME_HEADER.unpack(header)
    if length == 0:
        raise PolicyViolation("empty broker frames are not permitted")
    if length > max_frame_bytes:
        raise FrameTooLarge(f"broker frame declares {length} bytes; limit is {max_frame_bytes}")
    return _recv_exact(sock, length)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise BrokerUnavailable("broker closed the socket mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass(frozen=True)
class PeerCredentials:
    pid: int
    uid: int
    gid: int


@dataclass(frozen=True)
class PeerCredentialPolicy:
    """Approved Unix identity for the external broker."""

    allowed_uids: frozenset[int] = field(default_factory=frozenset)
    allowed_gids: frozenset[int] = field(default_factory=frozenset)
    allowed_pids: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_uids", frozenset(self.allowed_uids))
        object.__setattr__(self, "allowed_gids", frozenset(self.allowed_gids))
        object.__setattr__(self, "allowed_pids", frozenset(self.allowed_pids))
        if not self.allowed_uids and not self.allowed_gids:
            raise ValueError("peer policy must approve at least one uid or gid")
        if any(value < 0 for value in self.allowed_uids | self.allowed_gids):
            raise ValueError("approved peer uid/gid values cannot be negative")
        if any(value <= 0 for value in self.allowed_pids):
            raise ValueError("approved peer pid values must be positive")

    def validate(self, credentials: PeerCredentials) -> None:
        if self.allowed_uids and credentials.uid not in self.allowed_uids:
            raise PeerCredentialError(f"broker uid {credentials.uid} is not approved")
        if self.allowed_gids and credentials.gid not in self.allowed_gids:
            raise PeerCredentialError(f"broker gid {credentials.gid} is not approved")
        if self.allowed_pids and credentials.pid not in self.allowed_pids:
            raise PeerCredentialError(f"broker pid {credentials.pid} is not approved")


def peer_credentials(sock: socket.socket) -> PeerCredentials:
    """Read Linux ``SO_PEERCRED`` from a connected Unix socket."""

    if not hasattr(socket, "SO_PEERCRED"):
        raise PeerCredentialError("SO_PEERCRED is unavailable on this platform")
    size = struct.calcsize("3i")
    raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, size)
    pid, uid, gid = struct.unpack("3i", raw)
    return PeerCredentials(pid=pid, uid=uid, gid=gid)


class BrokerClientProtocol(Protocol):
    broker_id: str

    def execute(self, request: SandboxRequest) -> SandboxResponse:
        ...


class UnixBrokerClient:
    """Client only. No privileged broker/device policy lives in this package."""

    def __init__(
        self,
        *,
        socket_path: Path,
        broker_id: str,
        peer_policy: PeerCredentialPolicy,
        timeout_seconds: float = 10.0,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        reject_world_writable_socket: bool = True,
    ):
        path = Path(socket_path)
        if not path.is_absolute():
            raise ValueError("broker socket path must be absolute")
        if timeout_seconds <= 0 or max_frame_bytes <= 0:
            raise ValueError("broker timeout and frame bound must be positive")
        self.socket_path = path
        self.broker_id = broker_id
        self.peer_policy = peer_policy
        self.timeout_seconds = float(timeout_seconds)
        self.max_frame_bytes = int(max_frame_bytes)
        self.reject_world_writable_socket = reject_world_writable_socket

    def execute(self, request: SandboxRequest) -> SandboxResponse:
        self._validate_socket_path()
        payload = canonical_json_bytes({"type": "execute", "request": request.to_dict()})
        request_frame_limit = min(
            self.max_frame_bytes,
            request.policy.budget.max_control_bytes,
        )
        if len(payload) > request_frame_limit:
            raise FrameTooLarge(
                f"broker request is {len(payload)} bytes; limit is {request_frame_limit}"
            )

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_seconds)
        try:
            try:
                sock.connect(str(self.socket_path))
            except OSError as exc:
                raise BrokerUnavailable(
                    f"approved broker is unavailable at {self.socket_path}"
                ) from exc
            self.peer_policy.validate(peer_credentials(sock))
            send_frame(sock, payload, max_frame_bytes=request_frame_limit)
            response_payload = recv_frame(sock, max_frame_bytes=self.max_frame_bytes)
        except socket.timeout as exc:
            raise BrokerUnavailable("approved broker timed out") from exc
        except OSError as exc:
            raise BrokerUnavailable("approved broker transport failed") from exc
        finally:
            sock.close()

        try:
            envelope = json.loads(response_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PolicyViolation("broker returned invalid JSON") from exc
        if not isinstance(envelope, dict) or envelope.get("type") != "result":
            raise PolicyViolation("broker returned an invalid response envelope")
        try:
            response = SandboxResponse.from_dict(envelope["response"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PolicyViolation("broker returned an invalid typed response") from exc

        try:
            output_bytes = len(response.stdout.encode("utf-8")) + len(
                response.stderr.encode("utf-8")
            )
        except UnicodeEncodeError as exc:
            raise PolicyViolation("broker output is not valid Unicode") from exc
        if output_bytes > request.policy.budget.max_output_bytes:
            raise OutputTooLarge(
                f"broker output is {output_bytes} bytes; "
                f"limit is {request.policy.budget.max_output_bytes}"
            )
        return response

    def _validate_socket_path(self) -> None:
        try:
            info = self.socket_path.lstat()
        except OSError as exc:
            raise BrokerUnavailable(
                f"approved broker socket does not exist: {self.socket_path}"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISSOCK(info.st_mode):
            raise PeerCredentialError("broker path is not a direct Unix socket")
        if self.peer_policy.allowed_uids and info.st_uid not in self.peer_policy.allowed_uids:
            raise PeerCredentialError(f"broker socket owner uid {info.st_uid} is not approved")
        if self.peer_policy.allowed_gids and info.st_gid not in self.peer_policy.allowed_gids:
            raise PeerCredentialError(f"broker socket owner gid {info.st_gid} is not approved")
        if self.reject_world_writable_socket and info.st_mode & stat.S_IWOTH:
            raise PeerCredentialError("broker socket must not be world-writable")
