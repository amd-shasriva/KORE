"""Stdlib-only safety primitives for operational wrappers.

The central rule is that a process may be controlled only through state created
for a concrete run ID.  A recorded PID is never sufficient by itself: the Linux
process start time, uid, process group, cgroup, and inherited ``KORE_RUN_ID``
marker are checked again immediately before a signal is sent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import BinaryIO, Callable, Iterable, Mapping, Protocol, Sequence


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


class SecurityError(RuntimeError):
    """A filesystem or process identity failed a safety check."""


class LockBusy(RuntimeError):
    """A non-blocking run lock is already held."""


@dataclass(frozen=True)
class ArtifactStatus:
    ok: bool
    errors: tuple[str, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def success(cls, **details: object) -> "ArtifactStatus":
        return cls(True, (), details)

    @classmethod
    def failure(
        cls, *errors: str, details: Mapping[str, object] | None = None
    ) -> "ArtifactStatus":
        return cls(False, tuple(errors), details or {})


def _validate_name(value: str, *, what: str = "name") -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"unsafe {what}: {value!r}")
    return value


def _mode(path_stat: os.stat_result) -> int:
    return stat.S_IMODE(path_stat.st_mode)


def _check_owned_directory(path: Path, *, private: bool) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise SecurityError(f"runtime directory is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SecurityError(f"runtime path is not a real directory: {path}")
    if info.st_uid != os.getuid():
        raise SecurityError(
            f"runtime directory owner mismatch: {path} uid={info.st_uid}"
        )
    if private and _mode(info) & 0o077:
        raise SecurityError(
            f"runtime directory must not grant group/other access: "
            f"{path} mode={_mode(info):04o}"
        )


def _check_owned_regular(info: os.stat_result, path: Path) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise SecurityError(f"state path is not a regular file: {path}")
    if info.st_uid != os.getuid():
        raise SecurityError(f"state file owner mismatch: {path} uid={info.st_uid}")
    if _mode(info) & 0o077:
        raise SecurityError(
            f"state file must be private: {path} mode={_mode(info):04o}"
        )


class SecureFileLock:
    """An owned, no-symlink, private advisory lockfile."""

    def __init__(self, path: Path, *, blocking: bool = False):
        self.path = path
        self.blocking = blocking
        self._fd: int | None = None

    def __enter__(self) -> "SecureFileLock":
        flags = os.O_RDWR | os.O_CREAT | _NOFOLLOW | _CLOEXEC
        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise SecurityError(f"cannot safely open lockfile {self.path}: {exc}") from exc
        try:
            os.fchmod(fd, 0o600)
            _check_owned_regular(os.fstat(fd), self.path)
            operation = fcntl.LOCK_EX
            if not self.blocking:
                operation |= fcntl.LOCK_NB
            try:
                fcntl.flock(fd, operation)
            except BlockingIOError as exc:
                raise LockBusy(f"lock already held: {self.path}") from exc
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


class SecureRuntime:
    """Private runtime state rooted below XDG_RUNTIME_DIR or /tmp.

    Every directory is owned by the current uid and mode 0700.  Every state file
    is opened with ``O_NOFOLLOW``, checked for current ownership, and mode 0600.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None, *, create: bool = True):
        if path is None:
            configured = os.environ.get("KORE_RUNTIME_DIR")
            if configured:
                path = configured
            else:
                xdg = os.environ.get("XDG_RUNTIME_DIR")
                path = (
                    Path(xdg) / "kore-ops"
                    if xdg
                    else Path(tempfile.gettempdir()) / f"kore-ops-{os.getuid()}"
                )
        self.path = Path(path).expanduser()
        if create:
            self._ensure_private_dir(self.path)
        else:
            _check_owned_directory(self.path, private=True)

    @staticmethod
    def _ensure_private_dir(path: Path) -> None:
        parent = path.parent
        if not parent.exists():
            raise SecurityError(f"runtime parent does not exist: {parent}")
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        _check_owned_directory(path, private=True)
        # Tighten a directory created through an unusual umask, then re-check.
        os.chmod(path, 0o700, follow_symlinks=False)
        _check_owned_directory(path, private=True)

    def _parts(self, relative: str | os.PathLike[str]) -> tuple[str, ...]:
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError(f"runtime path must be relative: {relative}")
        parts = candidate.parts
        if not parts or any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"unsafe runtime path: {relative}")
        for part in parts:
            _validate_name(part, what="runtime path component")
        return parts

    def ensure_dir(self, relative: str | os.PathLike[str]) -> Path:
        current = self.path
        for part in self._parts(relative):
            current = current / part
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            _check_owned_directory(current, private=True)
            os.chmod(current, 0o700, follow_symlinks=False)
        return current

    def state_path(self, relative: str | os.PathLike[str]) -> Path:
        parts = self._parts(relative)
        parent = self.path if len(parts) == 1 else self.ensure_dir(Path(*parts[:-1]))
        return parent / parts[-1]

    def lock(self, name: str, *, blocking: bool = False) -> SecureFileLock:
        _validate_name(name, what="lock name")
        directory = self.ensure_dir("locks")
        return SecureFileLock(directory / f"{name}.lock", blocking=blocking)

    def write_bytes(self, relative: str | os.PathLike[str], payload: bytes) -> Path:
        target = self.state_path(relative)
        if target.exists() or target.is_symlink():
            info = target.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise SecurityError(f"refusing symlink state target: {target}")
            _check_owned_regular(info, target)
        fd, raw_tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        tmp = Path(raw_tmp)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
            _check_owned_regular(target.lstat(), target)
            dir_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | _CLOEXEC)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        return target

    def write_json(
        self, relative: str | os.PathLike[str], value: Mapping[str, object]
    ) -> Path:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        return self.write_bytes(relative, payload)

    def read_bytes(self, relative: str | os.PathLike[str]) -> bytes:
        target = self.state_path(relative)
        flags = os.O_RDONLY | _NOFOLLOW | _CLOEXEC
        try:
            fd = os.open(target, flags)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise SecurityError(f"cannot safely read state {target}: {exc}") from exc
        try:
            _check_owned_regular(os.fstat(fd), target)
            with os.fdopen(fd, "rb", closefd=False) as handle:
                return handle.read()
        finally:
            os.close(fd)

    def read_json(self, relative: str | os.PathLike[str]) -> dict:
        try:
            value = json.loads(self.read_bytes(relative))
        except json.JSONDecodeError as exc:
            raise SecurityError(f"invalid JSON state {relative}: {exc}") from exc
        if not isinstance(value, dict):
            raise SecurityError(f"JSON state must contain an object: {relative}")
        return value

    def unlink(self, relative: str | os.PathLike[str], *, missing_ok: bool = True) -> None:
        target = self.state_path(relative)
        try:
            info = target.lstat()
        except FileNotFoundError:
            if missing_ok:
                return
            raise
        if stat.S_ISLNK(info.st_mode):
            raise SecurityError(f"refusing to unlink symlink state: {target}")
        _check_owned_regular(info, target)
        target.unlink()

    def write_sentinel(self, name: str, value: Mapping[str, object]) -> Path:
        _validate_name(name, what="sentinel name")
        return self.write_json(Path("sentinels") / f"{name}.json", value)

    def peek_sentinel(self, name: str) -> dict | None:
        _validate_name(name, what="sentinel name")
        try:
            return self.read_json(Path("sentinels") / f"{name}.json")
        except FileNotFoundError:
            return None

    def clear_sentinel(self, name: str) -> None:
        _validate_name(name, what="sentinel name")
        self.unlink(Path("sentinels") / f"{name}.json", missing_ok=True)

    def consume_sentinel(self, name: str) -> dict | None:
        """Atomically claim, validate, read, and remove a sentinel."""

        _validate_name(name, what="sentinel name")
        source = self.state_path(Path("sentinels") / f"{name}.json")
        try:
            info = source.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(info.st_mode):
            raise SecurityError(f"refusing symlink sentinel: {source}")
        _check_owned_regular(info, source)
        claim = source.with_name(
            f"consume-{source.name}-{os.getpid()}-{secrets.token_hex(4)}"
        )
        os.replace(source, claim)
        relative_claim = claim.relative_to(self.path)
        try:
            return self.read_json(relative_claim)
        finally:
            self.unlink(relative_claim, missing_ok=True)

    def store_task_set(
        self, relative: str | os.PathLike[str], task_ids: Iterable[str]
    ) -> "TaskSetIdentity":
        identity = task_set_identity(task_ids)
        value = {
            "schema": 1,
            "count": identity.count,
            "sha256": identity.sha256,
            "task_ids": list(identity.task_ids),
        }
        try:
            existing = self.read_json(relative)
        except FileNotFoundError:
            self.write_json(relative, value)
            return identity
        if existing != value:
            raise SecurityError(
                f"immutable task-set state changed at {self.state_path(relative)}"
            )
        return identity


@dataclass(frozen=True)
class TaskSetIdentity:
    task_ids: tuple[str, ...]
    count: int
    sha256: str


def task_set_identity(task_ids: Iterable[str]) -> TaskSetIdentity:
    raw = [str(task_id).strip() for task_id in task_ids]
    if not raw or any(not task_id for task_id in raw):
        raise ValueError("task set must contain non-empty task IDs")
    if len(set(raw)) != len(raw):
        raise ValueError("task set contains duplicate task IDs")
    ordered = tuple(sorted(raw))
    digest = hashlib.sha256(("\n".join(ordered) + "\n").encode()).hexdigest()
    return TaskSetIdentity(ordered, len(ordered), digest)


def new_run_id(prefix: str = "run") -> str:
    clean = re.sub(r"[^a-z0-9-]+", "-", prefix.lower()).strip("-") or "run"
    clean = clean[:40]
    return f"{clean}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{secrets.token_hex(4)}"


@dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    uid: int
    start_ticks: int
    pgid: int
    cgroup: str
    run_id: str | None


@dataclass(frozen=True)
class ProcessIdentity:
    run_id: str
    pid: int
    uid: int
    start_ticks: int
    pgid: int
    cgroup: str
    command: tuple[str, ...] = ()
    owns_process_group: bool = True
    marker_required: bool = True

    def to_json(self) -> dict:
        value = asdict(self)
        value["command"] = list(self.command)
        return value

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "ProcessIdentity":
        return cls(
            run_id=str(value["run_id"]),
            pid=int(value["pid"]),
            uid=int(value["uid"]),
            start_ticks=int(value["start_ticks"]),
            pgid=int(value["pgid"]),
            cgroup=str(value.get("cgroup", "")),
            command=tuple(str(item) for item in value.get("command", [])),
            owns_process_group=bool(value.get("owns_process_group", True)),
            marker_required=bool(value.get("marker_required", True)),
        )


def _parse_proc_stat(text: str) -> tuple[int, int]:
    close = text.rfind(")")
    if close < 0:
        raise ValueError("malformed /proc stat")
    fields = text[close + 2 :].split()
    # fields[0] is field 3 (state); pgrp is field 5; starttime is field 22.
    return int(fields[2]), int(fields[19])


class ProcessBackend(Protocol):
    def snapshot(self, pid: int) -> ProcessSnapshot | None: ...

    def group_members(self, pgid: int) -> list[ProcessSnapshot]: ...

    def signal_pid(self, pid: int, sig: int) -> None: ...

    def signal_group(self, pgid: int, sig: int) -> None: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class LinuxProcessBackend:
    def __init__(self, proc_root: str | os.PathLike[str] = "/proc"):
        self.proc_root = Path(proc_root)

    def snapshot(self, pid: int) -> ProcessSnapshot | None:
        if pid <= 1:
            return None
        base = self.proc_root / str(pid)
        try:
            info = base.stat()
            pgid, start_ticks = _parse_proc_stat((base / "stat").read_text())
            cgroup = (base / "cgroup").read_text(errors="replace").strip()
            environ = (base / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            return None
        except (OSError, ValueError) as exc:
            raise SecurityError(f"cannot inspect process {pid}: {exc}") from exc
        marker = None
        for item in environ:
            if item.startswith(b"KORE_RUN_ID="):
                marker = item.split(b"=", 1)[1].decode(errors="replace")
                break
        return ProcessSnapshot(pid, info.st_uid, start_ticks, pgid, cgroup, marker)

    def group_members(self, pgid: int) -> list[ProcessSnapshot]:
        members = []
        try:
            entries = list(self.proc_root.iterdir())
        except OSError as exc:
            raise SecurityError(f"cannot inspect process table: {exc}") from exc
        for entry in entries:
            if not entry.name.isdigit():
                continue
            snapshot = self.snapshot(int(entry.name))
            if snapshot is not None and snapshot.pgid == pgid:
                members.append(snapshot)
        return sorted(members, key=lambda item: item.pid)

    def signal_pid(self, pid: int, sig: int) -> None:
        os.kill(pid, sig)

    def signal_group(self, pgid: int, sig: int) -> None:
        os.killpg(pgid, sig)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def capture_process_identity(
    pid: int,
    run_id: str,
    *,
    command: Sequence[str] = (),
    owns_process_group: bool = True,
    marker_required: bool = True,
    backend: ProcessBackend | None = None,
    timeout: float = 1.0,
) -> ProcessIdentity:
    backend = backend or LinuxProcessBackend()
    deadline = backend.monotonic() + timeout
    snapshot = backend.snapshot(pid)
    while snapshot is not None and (
        (marker_required and snapshot.run_id != run_id)
        or (owns_process_group and snapshot.pgid != pid)
    ):
        if backend.monotonic() >= deadline:
            break
        backend.sleep(0.01)
        snapshot = backend.snapshot(pid)
    if snapshot is None:
        raise SecurityError(f"process exited before identity capture: pid={pid}")
    if snapshot.uid != os.getuid():
        raise SecurityError(f"spawned process uid mismatch: pid={pid}")
    if marker_required and snapshot.run_id != run_id:
        raise SecurityError(
            f"spawned process run marker mismatch: pid={pid} "
            f"expected={run_id!r} actual={snapshot.run_id!r}"
        )
    if owns_process_group and snapshot.pgid != pid:
        raise SecurityError(
            f"spawned process does not own its process group: "
            f"pid={pid} pgid={snapshot.pgid}"
        )
    return ProcessIdentity(
        run_id=run_id,
        pid=pid,
        uid=snapshot.uid,
        start_ticks=snapshot.start_ticks,
        pgid=snapshot.pgid,
        cgroup=snapshot.cgroup,
        command=tuple(command),
        owns_process_group=owns_process_group,
        marker_required=marker_required,
    )


def identity_matches(
    identity: ProcessIdentity, *, backend: ProcessBackend | None = None
) -> tuple[bool, str]:
    backend = backend or LinuxProcessBackend()
    snapshot = backend.snapshot(identity.pid)
    if snapshot is None:
        return False, "process absent"
    expected = (
        identity.uid,
        identity.start_ticks,
        identity.pgid,
        identity.cgroup,
    )
    actual = (snapshot.uid, snapshot.start_ticks, snapshot.pgid, snapshot.cgroup)
    if actual != expected:
        return False, "pid identity mismatch"
    if identity.marker_required and snapshot.run_id != identity.run_id:
        return False, "run marker mismatch"
    return True, "matched"


def _owned_group(
    identity: ProcessIdentity, backend: ProcessBackend
) -> tuple[bool, list[ProcessSnapshot], str]:
    members = backend.group_members(identity.pgid)
    if not members:
        return True, [], "group absent"
    for member in members:
        if member.uid != identity.uid:
            return False, members, f"group contains foreign uid pid={member.pid}"
        if identity.marker_required and member.run_id != identity.run_id:
            return False, members, f"group contains foreign run marker pid={member.pid}"
        if member.cgroup != identity.cgroup:
            return False, members, f"group member changed cgroup pid={member.pid}"
    return True, members, "owned"


@dataclass(frozen=True)
class TerminationResult:
    stopped: bool
    term_sent: bool
    kill_sent: bool
    refused: bool
    reason: str


def terminate_owned(
    identity: ProcessIdentity,
    *,
    term_timeout: float = 15.0,
    kill_timeout: float = 5.0,
    poll_interval: float = 0.1,
    backend: ProcessBackend | None = None,
) -> TerminationResult:
    """TERM, bounded wait, then KILL an identity-verified owned process."""

    backend = backend or LinuxProcessBackend()
    matched, reason = identity_matches(identity, backend=backend)
    if not matched:
        if reason == "process absent" and identity.owns_process_group:
            owned, members, group_reason = _owned_group(identity, backend)
            if not owned:
                return TerminationResult(False, False, False, True, group_reason)
            if not members:
                return TerminationResult(True, False, False, False, reason)
        elif reason == "process absent":
            return TerminationResult(True, False, False, False, reason)
        else:
            return TerminationResult(False, False, False, True, reason)

    def remaining() -> tuple[bool, str]:
        if identity.owns_process_group:
            owned, members, group_reason = _owned_group(identity, backend)
            if not owned:
                return True, group_reason
            return bool(members), group_reason
        live, live_reason = identity_matches(identity, backend=backend)
        if not live and live_reason == "process absent":
            return False, live_reason
        return live, live_reason

    if identity.owns_process_group:
        owned, _members, group_reason = _owned_group(identity, backend)
        if not owned:
            return TerminationResult(False, False, False, True, group_reason)
        backend.signal_group(identity.pgid, signal.SIGTERM)
    else:
        backend.signal_pid(identity.pid, signal.SIGTERM)

    deadline = backend.monotonic() + max(0.0, term_timeout)
    while backend.monotonic() < deadline:
        live, live_reason = remaining()
        if not live:
            return TerminationResult(True, True, False, False, live_reason)
        if "foreign" in live_reason or "changed" in live_reason:
            return TerminationResult(False, True, False, True, live_reason)
        backend.sleep(poll_interval)

    live, live_reason = remaining()
    if not live:
        return TerminationResult(True, True, False, False, live_reason)
    if "foreign" in live_reason or "changed" in live_reason:
        return TerminationResult(False, True, False, True, live_reason)

    if identity.owns_process_group:
        backend.signal_group(identity.pgid, signal.SIGKILL)
    else:
        backend.signal_pid(identity.pid, signal.SIGKILL)
    deadline = backend.monotonic() + max(0.0, kill_timeout)
    while backend.monotonic() < deadline:
        live, live_reason = remaining()
        if not live:
            return TerminationResult(True, True, True, False, live_reason)
        backend.sleep(poll_interval)
    live, live_reason = remaining()
    return TerminationResult(
        not live,
        True,
        True,
        False,
        live_reason if not live else "process survived SIGKILL deadline",
    )


def open_append_log(path: str | os.PathLike[str]) -> BinaryIO:
    """Open an owned regular log for append without following symlinks."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    parent_info = target.parent.lstat()
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise SecurityError(f"log parent is not a real directory: {target.parent}")
    if parent_info.st_uid != os.getuid():
        raise SecurityError(f"log parent owner mismatch: {target.parent}")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | _NOFOLLOW | _CLOEXEC
    try:
        fd = os.open(target, flags, 0o600)
    except OSError as exc:
        raise SecurityError(f"cannot safely open log {target}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise SecurityError(f"log is not an owned regular file: {target}")
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, "ab", buffering=0)
    except BaseException:
        os.close(fd)
        raise


class IncrementalLogReader:
    """Read only bytes appended since the previous call.

    Rotation and truncation reset the offset.  Partial trailing lines remain
    buffered until completed, avoiding duplicate regex matches across polls.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.inode: tuple[int, int] | None = None
        self.offset = 0
        self._partial = b""

    def read_lines(self, *, final: bool = False) -> list[str]:
        flags = os.O_RDONLY | _NOFOLLOW | _CLOEXEC
        try:
            fd = os.open(self.path, flags)
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise SecurityError(f"cannot safely read log {self.path}: {exc}") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise SecurityError(f"log is not a regular file: {self.path}")
            if info.st_uid != os.getuid():
                raise SecurityError(f"log owner mismatch: {self.path}")
            identity = (info.st_dev, info.st_ino)
            if self.inode != identity or info.st_size < self.offset:
                self.inode = identity
                self.offset = 0
                self._partial = b""
            os.lseek(fd, self.offset, os.SEEK_SET)
            chunks = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            self.offset = os.lseek(fd, 0, os.SEEK_CUR)
        finally:
            os.close(fd)
        data = self._partial + b"".join(chunks)
        pieces = data.splitlines(keepends=True)
        if pieces and not pieces[-1].endswith((b"\n", b"\r")) and not final:
            self._partial = pieces.pop()
        else:
            self._partial = b""
        return [piece.decode("utf-8", "replace").rstrip("\r\n") for piece in pieces]


class RunPhase(str, Enum):
    INITIAL = "initial"
    RUNNING = "running"
    VERIFYING = "verifying"
    WAITING = "waiting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    GAVE_UP = "gave_up"
    STOPPED = "stopped"


@dataclass
class SupervisorStateMachine:
    max_attempts: int
    phase: RunPhase = RunPhase.INITIAL
    attempt: int = 0
    returncode: int | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")

    def begin_attempt(self) -> int:
        if self.phase not in (RunPhase.INITIAL, RunPhase.WAITING):
            raise RuntimeError(f"cannot launch from phase {self.phase.value}")
        if self.attempt >= self.max_attempts:
            raise RuntimeError("attempt budget exhausted")
        self.attempt += 1
        self.returncode = None
        self.reason = ""
        self.phase = RunPhase.RUNNING
        return self.attempt

    def process_exited(self, returncode: int) -> None:
        if self.phase is not RunPhase.RUNNING:
            raise RuntimeError(f"cannot record exit from phase {self.phase.value}")
        self.returncode = int(returncode)
        self.phase = RunPhase.VERIFYING

    def verification_finished(self, status: ArtifactStatus) -> bool:
        if self.phase is not RunPhase.VERIFYING:
            raise RuntimeError(f"cannot verify from phase {self.phase.value}")
        if self.returncode == 0 and status.ok:
            self.phase = RunPhase.SUCCEEDED
            self.reason = "strict artifact verification passed"
            return True
        failures = list(status.errors)
        if self.returncode != 0:
            failures.insert(0, f"process exited rc={self.returncode}")
        self.reason = "; ".join(failures) or "completion verification failed"
        self.phase = (
            RunPhase.WAITING
            if self.attempt < self.max_attempts
            else RunPhase.GAVE_UP
        )
        return False

    def fail(self, reason: str) -> None:
        if self.phase in (RunPhase.SUCCEEDED, RunPhase.GAVE_UP, RunPhase.STOPPED):
            raise RuntimeError(f"cannot fail terminal phase {self.phase.value}")
        self.phase = RunPhase.FAILED
        self.reason = reason

    def stop(self, reason: str) -> None:
        if self.phase is RunPhase.SUCCEEDED:
            raise RuntimeError("cannot stop a completed run")
        self.phase = RunPhase.STOPPED
        self.reason = reason

    def to_json(self) -> dict:
        return {
            "schema": 1,
            "max_attempts": self.max_attempts,
            "attempt": self.attempt,
            "phase": self.phase.value,
            "returncode": self.returncode,
            "reason": self.reason,
        }


class OwnedProcess:
    """A subprocess in its own session with persisted, verified ownership."""

    def __init__(
        self,
        process: subprocess.Popen,
        identity: ProcessIdentity,
        runtime: SecureRuntime,
        state_relative: str | os.PathLike[str],
    ):
        self.process = process
        self.identity = identity
        self.runtime = runtime
        self.state_relative = state_relative

    @classmethod
    def spawn(
        cls,
        command: Sequence[str],
        *,
        run_id: str,
        runtime: SecureRuntime,
        state_relative: str | os.PathLike[str],
        cwd: str | os.PathLike[str],
        env: Mapping[str, str] | None = None,
        stdout: int | BinaryIO | None = None,
        stderr: int | BinaryIO | None = subprocess.STDOUT,
    ) -> "OwnedProcess":
        if not command:
            raise ValueError("owned process command must not be empty")
        _validate_name(run_id, what="run ID")
        child_env = dict(os.environ if env is None else env)
        child_env["KORE_RUN_ID"] = run_id
        process = subprocess.Popen(
            list(command),
            cwd=os.fspath(cwd),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            identity = capture_process_identity(
                process.pid, run_id, command=command, owns_process_group=True
            )
        except BaseException:
            # The child is ours because Popen just created it, but do not use a
            # broad pattern if identity capture failed.
            try:
                process.terminate()
                process.wait(timeout=2)
            except BaseException:
                try:
                    process.kill()
                except BaseException:
                    pass
            raise
        owned = cls(process, identity, runtime, state_relative)
        owned._write_state("running", None)
        return owned

    def _write_state(self, status: str, returncode: int | None) -> None:
        self.runtime.write_json(
            self.state_relative,
            {
                "schema": 1,
                "status": status,
                "returncode": returncode,
                "updated_ns": time.time_ns(),
                "identity": self.identity.to_json(),
            },
        )

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self, timeout: float | None = None) -> int:
        returncode = self.process.wait(timeout=timeout)
        if self.identity.owns_process_group:
            backend = LinuxProcessBackend()
            owned, members, reason = _owned_group(self.identity, backend)
            if not owned:
                raise SecurityError(f"cannot clean residual process group: {reason}")
            if members:
                result = terminate_owned(self.identity, backend=backend)
                if not result.stopped:
                    raise SecurityError(
                        f"residual owned process group did not stop: {result.reason}"
                    )
        self._write_state("exited", returncode)
        return returncode

    def terminate(
        self, *, term_timeout: float = 15.0, kill_timeout: float = 5.0
    ) -> TerminationResult:
        result = terminate_owned(
            self.identity, term_timeout=term_timeout, kill_timeout=kill_timeout
        )
        if result.stopped:
            try:
                self.process.wait(timeout=kill_timeout)
            except subprocess.TimeoutExpired:
                pass
            self._write_state("terminated", self.process.poll())
        return result


def deprecated_entrypoint(
    script: str,
    migration: str,
    *,
    dry_run: bool = False,
    override_env: str = "KORE_ALLOW_DEPRECATED_DEV",
) -> bool:
    """Enforce the development-only quarantine for a deprecated entrypoint.

    Returns ``False`` for a side-effect-free dry-run.  A real invocation exits
    with EX_USAGE unless the explicit development override is exactly ``1``.
    """

    message = (
        f"{script} is deprecated and disabled for production. "
        f"Migration: {migration}. "
        f"Development-only override: {override_env}=1."
    )
    if dry_run:
        print(f"DRY-RUN: {message}")
        return False
    if os.environ.get(override_env) != "1":
        print(f"ERROR: {message}", file=sys.stderr)
        raise SystemExit(64)
    print(f"WARNING: development override enabled: {script}", file=sys.stderr)
    return True
