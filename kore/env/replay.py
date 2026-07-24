"""Versioned, process-safe JSONL replay storage.

An evaluation result is reusable only under the exact evaluation contract that
produced it.  The caller supplies that contract as JSON-compatible ``context``;
it is part of both the key and the persisted record.  The environment's context
contains shapes, validation/benchmark capabilities, task fingerprints,
architecture, rigor settings, and observation-producing thresholds.

Migration policy
----------------
Schema-1 records (the historical ``(task_id, source)`` cache) do not contain
enough provenance to prove which shapes or rigor produced an observation.  They
are therefore deliberately ignored and recomputed on demand.  New schema
records are appended to the same JSONL file, so migration is non-destructive and
does not require an unsafe inference from observation fields.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from kore.reward.reward import Observation


REPLAY_SCHEMA_VERSION = 2
"""On-disk/key schema. Bump when record validation or key semantics change."""

LEGACY_MIGRATION_POLICY = "ignore-and-recompute"
"""Schema-1 entries are never promoted because their capabilities are unknown."""

_UNSCOPED_CONTEXT = {
    "contract_version": 0,
    "namespace": "unscoped-public-api",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OBS_FIELDS = {f.name for f in fields(Observation)}
_OBS_MAP_FIELDS = ("wall_by_shape", "baseline_by_shape", "snr_by_shape")
_OBS_BOOL_FIELDS = ("compiled", "validation_passed", "flagged_hack", "infra_error")
_OBS_NUMBER_FIELDS = (
    "snr_db",
    "wall_ms",
    "baseline_ms",
    "cv_pct",
    "profile_efficiency",
)


def _obs_from_dict(rec: Mapping[str, Any]) -> Observation:
    """Decode a current Observation, rejecting unsafe/malformed cache payloads."""
    if not isinstance(rec, Mapping):
        raise TypeError("observation payload must be an object")
    values = {k: v for k, v in rec.items() if k in _OBS_FIELDS}
    if "compiled" not in values:
        raise ValueError("observation payload is missing compiled")
    for name in _OBS_BOOL_FIELDS:
        if name in values and not isinstance(values[name], bool):
            raise TypeError(f"observation field {name} must be boolean")
    for name in _OBS_MAP_FIELDS:
        if name not in values:
            continue
        mapping = values[name]
        if not isinstance(mapping, dict):
            raise TypeError(f"observation field {name} must be an object")
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise TypeError(f"observation field {name} must have string keys")
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or (isinstance(value, float) and math.isnan(value))):
                raise TypeError(f"observation field {name} must contain numbers")
    for name in _OBS_NUMBER_FIELDS:
        value = values.get(name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and math.isnan(value))
        ):
            raise TypeError(f"observation field {name} must be numeric or null")
    if "dtype" in values and not isinstance(values["dtype"], str):
        raise TypeError("observation field dtype must be a string")
    for name in ("error_text", "hack_reason"):
        if name in values and values[name] is not None and not isinstance(values[name], str):
            raise TypeError(f"observation field {name} must be a string or null")
    obs = Observation(**values)
    if obs.infra_error:
        raise ValueError("infrastructure failures are not replayable")
    return obs


def kernel_hash(source: str) -> str:
    """Content hash of a kernel source (stable id used across datagen)."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _normalise_context(context: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a detached, canonical JSON object for hashing/persistence."""
    value: Any
    if context is None:
        # Backward-compatible direct ReplayCache use remains practical, but lives
        # in an isolated namespace that KoreEnv never queries.
        value = _UNSCOPED_CONTEXT
    elif is_dataclass(context):
        value = asdict(context)
    elif hasattr(context, "to_dict"):
        value = context.to_dict()
    else:
        value = context
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (RecursionError, TypeError, ValueError) as exc:
        raise TypeError("replay context must be a finite JSON object") from exc
    if not isinstance(decoded, dict):
        raise TypeError("replay context must be a JSON object")
    return decoded


def _key_from_source_hash(
    task_id: str,
    source_sha256: str,
    context: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "task_id": task_id,
        "source_sha256": source_sha256,
        "context": context,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_key(
    task_id: str,
    source: str,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    """Versioned key for an exact ``(task, source, evaluation-context)``."""
    return _key_from_source_hash(
        str(task_id),
        kernel_hash(source),
        _normalise_context(context),
    )


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:  # pragma: no cover - defensive regular-file guard
            raise OSError("short replay-cache write")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    """Persist a newly-created directory entry when the platform supports it."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class ReplayCache:
    """Append-only replay cache safe for threads, processes, and torn tails.

    Every append holds an advisory process lock, uses ``O_APPEND``, terminates a
    pre-existing malformed tail before writing, and fsyncs the data file.  Reads
    take a shared process lock and incrementally ingest records written by other
    live processes.
    """

    migration_policy = LEGACY_MIGRATION_POLICY

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock_path = self.path.with_name(self.path.name + ".lock")
        self._mem: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._offset = 0
        self._identity: Optional[tuple[int, int]] = None
        self._ignored = {"legacy": 0, "incompatible": 0, "malformed": 0}
        self._refresh()

    @property
    def ignored_records(self) -> dict[str, int]:
        """Counts of records rejected while loading (diagnostic only)."""
        with self._lock:
            return dict(self._ignored)

    @contextmanager
    def _process_lock(self, exclusive: bool) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _refresh(self) -> None:
        # A replay cache is an optimization: unavailable storage must not prevent
        # a fresh evaluation from running.
        with self._lock:
            try:
                with self._process_lock(exclusive=False):
                    self._read_new_records_locked()
            except OSError:
                return

    def _read_new_records_locked(self) -> None:
        """Read complete newline-terminated records while process-locked."""
        try:
            f = self.path.open("rb")
        except FileNotFoundError:
            if self._identity is not None:
                self._mem.clear()
                self._offset = 0
                self._identity = None
            return

        with f:
            stat = os.fstat(f.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if self._identity != identity or stat.st_size < self._offset:
                self._mem.clear()
                self._offset = 0
                self._identity = identity

            f.seek(self._offset)
            while True:
                start = f.tell()
                line = f.readline()
                if not line:
                    self._offset = start
                    break
                if not line.endswith(b"\n"):
                    # An unterminated tail may be a crashed append. Leave the
                    # offset before it so a later delimiter/completion is seen.
                    self._offset = start
                    break
                self._consume_line(line)
                self._offset = f.tell()

    def _consume_line(self, raw_line: bytes) -> None:
        line = raw_line.strip()
        if not line:
            return
        try:
            rec = json.loads(line)
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError):
            self._ignored["malformed"] += 1
            return
        if not isinstance(rec, dict):
            self._ignored["malformed"] += 1
            return

        version = rec.get("schema_version")
        if version != REPLAY_SCHEMA_VERSION:
            bucket = "legacy" if version is None or version == 1 else "incompatible"
            self._ignored[bucket] += 1
            return

        try:
            key = rec["key"]
            task_id = rec["task_id"]
            source_sha256 = rec["source_sha256"]
            context = _normalise_context(rec["context"])
            if not isinstance(key, str) or not isinstance(task_id, str):
                raise TypeError("invalid key/task id")
            if not isinstance(source_sha256, str) or not _SHA256_RE.fullmatch(source_sha256):
                raise ValueError("invalid source digest")
            expected = _key_from_source_hash(task_id, source_sha256, context)
            if key != expected:
                raise ValueError("record key does not match provenance")
            obs = _obs_from_dict(rec["obs"])
        except (KeyError, RecursionError, TypeError, ValueError):
            self._ignored["malformed"] += 1
            return
        self._mem[key] = asdict(obs)

    def get(
        self,
        task_id: str,
        source: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Observation]:
        context_obj = _normalise_context(context)
        key = _key_from_source_hash(str(task_id), kernel_hash(source), context_obj)
        self._refresh()
        with self._lock:
            rec = self._mem.get(key)
            if rec is None:
                return None
            try:
                return _obs_from_dict(rec)
            except (TypeError, ValueError):
                # Defensive against accidental in-process mutation.
                self._mem.pop(key, None)
                return None

    def put(
        self,
        task_id: str,
        source: str,
        obs: Observation,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Durably append ``obs``; infra/storage failures remain uncached."""
        if obs.infra_error:
            return

        context_obj = _normalise_context(context)
        task_id = str(task_id)
        source_sha256 = kernel_hash(source)
        key = _key_from_source_hash(task_id, source_sha256, context_obj)
        try:
            # Validate before touching disk. This also rejects malformed objects
            # passed through a dynamically-typed caller.
            obs_rec = asdict(obs)
            _obs_from_dict(obs_rec)
            record = {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "key": key,
                "task_id": task_id,
                "source_sha256": source_sha256,
                "context": context_obj,
                "obs": obs_rec,
            }
            payload = (
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
        except (RecursionError, TypeError, ValueError):
            return

        with self._lock:
            try:
                with self._process_lock(exclusive=True):
                    self._read_new_records_locked()
                    created = not self.path.exists()
                    fd = os.open(
                        self.path,
                        os.O_RDWR | os.O_CREAT | os.O_APPEND,
                        0o600,
                    )
                    try:
                        stat = os.fstat(fd)
                        prefix = b""
                        if stat.st_size and os.pread(fd, 1, stat.st_size - 1) != b"\n":
                            # Never concatenate a valid record onto a torn tail.
                            prefix = b"\n"
                        _write_all(fd, prefix + payload)
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    if created:
                        _fsync_directory(self.path.parent)
                    # Re-read from the last complete offset. This ingests both a
                    # newly-delimited old tail and our just-written record.
                    self._read_new_records_locked()
            except OSError:
                return

    def __len__(self) -> int:
        self._refresh()
        with self._lock:
            return len(self._mem)
