"""KORE data-generation record schemas and durable JSONL I/O.

Four record types feed the capability curriculum:
  - ``RepairRecord``  (Stage 1, repair-weighted SFT): a broken -> fixed turn,
    conditioned on the exact verifier error.
  - ``RankedGroupRecord`` (Stage 2, RFT + DPO): a group of candidates for one
    parent with a ranking and the derived preference pairs.
  - ``WinRecord`` (Stage 3, multi-turn evolve): a full winning trajectory.
  - ``AgenticTrajectoryRecord``: a multi-turn tool-use episode (resolved lazily
    from :mod:`kore.agent.schema` to avoid an import cycle).

Every record is a plain dataclass with symmetric ``to_dict``/``from_dict`` so it
round-trips losslessly through JSONL. Production record admission is strict and
versioned; the explicitly named ``read_jsonl_legacy`` path is the only tolerant
reader and is intended for quarantine/migration tooling.

``write_jsonl`` is generic (training rows without a KORE ``type`` are supported)
but durable: it writes a unique temporary file in the destination directory,
flushes and fsyncs it, atomically replaces the destination, then fsyncs the
directory. Known KORE records are stamped with the current schema version.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Union

if TYPE_CHECKING:
    from kore.agent.schema import AgenticTrajectoryRecord

_LOG = logging.getLogger(__name__)

GPU_DEFAULT = "gfx950"  # KORE target = MI350X/CDNA4 (matches registry.TRAIN_ARCH)
RECORD_SCHEMA_VERSION = 1
SCHEMA_VERSION_FIELD = "schema_version"


@dataclass
class RepairRecord:
    """A single repair turn: parent kernel failed, teacher fixed it."""

    task_id: str
    failure_class: str          # "compile_fail" | "snr_fail"
    parent_hash: str
    error_text: str
    messages: list[dict]        # [{"role": ..., "content": ...}, ...]
    child_snr_db: float | None = None
    type: str = "repair"
    operator: str = "repair"
    gpu: str = GPU_DEFAULT
    # Leakage provenance (KORE Sec 4.4): the source op/arch/shape this record was
    # generated from, used for leakage-aware train/val/test splitting.
    operation: str | None = None
    arch: str | None = None
    shape: str | None = None
    schema_version: ClassVar[int] = RECORD_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {**asdict(self), SCHEMA_VERSION_FIELD: self.schema_version}

    @classmethod
    def from_dict(cls, d: dict) -> "RepairRecord":
        return cls(
            task_id=d["task_id"],
            failure_class=d["failure_class"],
            parent_hash=d["parent_hash"],
            error_text=d.get("error_text", ""),
            messages=list(d.get("messages", [])),
            child_snr_db=d.get("child_snr_db"),
            type=d.get("type", "repair"),
            operator=d.get("operator", "repair"),
            gpu=d.get("gpu", GPU_DEFAULT),
            operation=d.get("operation"),
            arch=d.get("arch"),
            shape=d.get("shape"),
        )


@dataclass
class RankedGroupRecord:
    """A parent plus k ranked candidates and the derived preference pairs."""

    task_id: str
    parent_id: str
    candidates: list[dict]      # [{"source", "wall_us", "snr_db", "rank"}, ...]
    preferences: list[list[int]]  # [[chosen_idx, rejected_idx], ...]
    type: str = "ranked_group"
    gpu: str = GPU_DEFAULT
    # Leakage provenance (KORE Sec 4.4).
    operation: str | None = None
    arch: str | None = None
    shape: str | None = None
    # rocprofv3 counters for the rank-0 (best) candidate, when collected at datagen
    # (Pillar 4, KORE_GROUND_REASONING=1). Enables profiler-grounded gold-win reasoning.
    counters: dict | None = None
    # rocprofv3 counters + wall for a representative SLOWER-correct candidate (the
    # "parent" the win improves on), so gold-win reasoning can narrate a real
    # PROFILE(parent)->...->MEASURE(best) delta instead of misattributing the winner's.
    parent_counters: dict | None = None
    parent_wall_us: float | None = None
    schema_version: ClassVar[int] = RECORD_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {**asdict(self), SCHEMA_VERSION_FIELD: self.schema_version}

    @classmethod
    def from_dict(cls, d: dict) -> "RankedGroupRecord":
        return cls(
            task_id=d["task_id"],
            parent_id=d["parent_id"],
            candidates=list(d.get("candidates", [])),
            preferences=[list(p) for p in d.get("preferences", [])],
            type=d.get("type", "ranked_group"),
            gpu=d.get("gpu", GPU_DEFAULT),
            operation=d.get("operation"),
            arch=d.get("arch"),
            shape=d.get("shape"),
            counters=d.get("counters"),
            parent_counters=d.get("parent_counters"),
            parent_wall_us=d.get("parent_wall_us"),
        )


@dataclass
class WinRecord:
    """A full winning multi-turn trajectory (initial -> final, wall improved)."""

    task_id: str
    trajectory: list[dict]      # list of chat messages across turns
    initial_wall_us: float | None
    final_wall_us: float | None
    speedup: float | None
    final_source: str
    snr_db: float | None = None
    type: str = "win"
    gpu: str = GPU_DEFAULT
    # Leakage provenance (KORE Sec 4.4).
    operation: str | None = None
    arch: str | None = None
    shape: str | None = None
    schema_version: ClassVar[int] = RECORD_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {**asdict(self), SCHEMA_VERSION_FIELD: self.schema_version}

    @classmethod
    def from_dict(cls, d: dict) -> "WinRecord":
        return cls(
            task_id=d["task_id"],
            trajectory=list(d.get("trajectory", [])),
            initial_wall_us=d.get("initial_wall_us"),
            final_wall_us=d.get("final_wall_us"),
            speedup=d.get("speedup"),
            final_source=d.get("final_source", ""),
            snr_db=d.get("snr_db"),
            type=d.get("type", "win"),
            gpu=d.get("gpu", GPU_DEFAULT),
            operation=d.get("operation"),
            arch=d.get("arch"),
            shape=d.get("shape"),
        )


Record = Union[
    RepairRecord,
    RankedGroupRecord,
    WinRecord,
    "AgenticTrajectoryRecord",
]

_TYPE_TO_CLASS = {
    "repair": RepairRecord,
    "ranked_group": RankedGroupRecord,
    "win": WinRecord,
}
_KNOWN_RECORD_TYPES = frozenset((*_TYPE_TO_CLASS, "agentic"))
_MESSAGE_ROLES = frozenset(("system", "user", "assistant", "tool"))


class RecordValidationError(ValueError):
    """A KORE record violates the current strict schema."""


class JsonlValidationError(ValueError):
    """A JSONL line is malformed or fails strict record validation."""


@dataclass(frozen=True)
class ShardValidation:
    """Stable facts computed while strictly validating one JSONL shard."""

    record_count: int
    sha256: str


@dataclass(frozen=True)
class _PreparedJsonl:
    """A fully written and fsynced same-directory temporary JSONL file."""

    target_path: Path
    temp_path: Path
    record_count: int
    sha256: str


def _record_class(record_type: Any):
    """Resolve a record class lazily to avoid the agent-schema import cycle."""
    if record_type == "agentic":
        from kore.agent.schema import AgenticTrajectoryRecord

        return AgenticTrajectoryRecord
    return _TYPE_TO_CLASS.get(record_type)


def _validation_error(path: str, message: str) -> RecordValidationError:
    return RecordValidationError(f"{path}: {message}")


def _validate_json_tree(value: Any, path: str = "record") -> None:
    """Reject values JSON cannot represent portably, especially NaN/Inf."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _validation_error(path, "NaN and infinity are not allowed")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _validation_error(path, "object keys must be strings")
            _validate_json_tree(item, f"{path}.{key}")
        return
    raise _validation_error(path, f"unsupported JSON value {type(value).__name__}")


def _require_dict(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise _validation_error(path, "must be an object")
    return value


def _require_list(value: Any, path: str, *, nonempty: bool = False) -> list:
    if not isinstance(value, list):
        raise _validation_error(path, "must be a list")
    if nonempty and not value:
        raise _validation_error(path, "must not be empty")
    return value


def _require_string(mapping: dict, key: str, path: str, *,
                    nonempty: bool = True) -> str:
    if key not in mapping:
        raise _validation_error(path, f"missing required field {key!r}")
    value = mapping[key]
    if not isinstance(value, str):
        raise _validation_error(f"{path}.{key}", "must be a string")
    if nonempty and not value.strip():
        raise _validation_error(f"{path}.{key}", "must not be empty")
    return value


def _require_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _validation_error(path, "must be an integer")
    if minimum is not None and value < minimum:
        raise _validation_error(path, f"must be >= {minimum}")
    return value


def _validate_optional_number(mapping: dict, key: str, path: str,
                              *, positive: bool = False) -> None:
    value = mapping.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _validation_error(f"{path}.{key}", "must be a number or null")
    if not math.isfinite(float(value)):
        raise _validation_error(f"{path}.{key}", "must be finite")
    if positive and value <= 0:
        raise _validation_error(f"{path}.{key}", "must be positive")


def _validate_messages(value: Any, path: str) -> None:
    # Empty transcripts remain representable for source-only champion records and
    # failed/no-turn episodes; every message that is present is fully validated.
    messages = _require_list(value, path)
    for index, raw_message in enumerate(messages):
        message_path = f"{path}[{index}]"
        message = _require_dict(raw_message, message_path)
        role = _require_string(message, "role", message_path)
        if role not in _MESSAGE_ROLES:
            raise _validation_error(
                f"{message_path}.role", f"unknown message role {role!r}")
        _require_string(message, "content", message_path)


def _validate_repair(d: dict) -> None:
    failure_class = _require_string(d, "failure_class", "record")
    if failure_class not in ("compile_fail", "snr_fail"):
        raise _validation_error(
            "record.failure_class", f"unknown failure class {failure_class!r}")
    _require_string(d, "parent_hash", "record")
    _require_string(d, "error_text", "record", nonempty=False)
    if "messages" not in d:
        raise _validation_error("record", "missing required field 'messages'")
    _validate_messages(d["messages"], "record.messages")
    _validate_optional_number(d, "child_snr_db", "record")


def _validate_ranked_group(d: dict) -> None:
    _require_string(d, "parent_id", "record")
    if "candidates" not in d:
        raise _validation_error("record", "missing required field 'candidates'")
    candidates = _require_list(
        d["candidates"], "record.candidates", nonempty=True)
    ranks: list[int] = []
    for index, raw_candidate in enumerate(candidates):
        candidate_path = f"record.candidates[{index}]"
        candidate = _require_dict(raw_candidate, candidate_path)
        _require_string(candidate, "source", candidate_path)
        if "rank" not in candidate:
            raise _validation_error(candidate_path, "missing required field 'rank'")
        ranks.append(_require_int(
            candidate["rank"], f"{candidate_path}.rank", minimum=0))
        for numeric_key in (
            "wall_us", "snr_db", "speedup", "baseline_wall_us",
        ):
            _validate_optional_number(candidate, numeric_key, candidate_path)

    expected_ranks = set(range(len(candidates)))
    if set(ranks) != expected_ranks or len(set(ranks)) != len(ranks):
        raise _validation_error(
            "record.candidates",
            f"ranks must be unique and contiguous 0..{len(candidates) - 1}")

    if "preferences" not in d:
        raise _validation_error("record", "missing required field 'preferences'")
    preferences = _require_list(d["preferences"], "record.preferences")
    seen_pairs: set[tuple[int, int]] = set()
    for index, pair in enumerate(preferences):
        pair_path = f"record.preferences[{index}]"
        if not isinstance(pair, list) or len(pair) != 2:
            raise _validation_error(pair_path, "must be [chosen_idx, rejected_idx]")
        chosen = _require_int(pair[0], f"{pair_path}[0]", minimum=0)
        rejected = _require_int(pair[1], f"{pair_path}[1]", minimum=0)
        if chosen >= len(candidates) or rejected >= len(candidates):
            raise _validation_error(pair_path, "candidate index is out of range")
        if chosen == rejected:
            raise _validation_error(pair_path, "cannot prefer a candidate to itself")
        if ranks[chosen] >= ranks[rejected]:
            raise _validation_error(
                pair_path, "chosen candidate must have a better (lower) rank")
        key = (chosen, rejected)
        if key in seen_pairs:
            raise _validation_error(pair_path, "duplicate preference")
        seen_pairs.add(key)

    for optional_dict in ("counters", "parent_counters"):
        value = d.get(optional_dict)
        if value is not None and not isinstance(value, dict):
            raise _validation_error(
                f"record.{optional_dict}", "must be an object or null")
    _validate_optional_number(d, "parent_wall_us", "record")


def _validate_win(d: dict) -> None:
    if "trajectory" not in d:
        raise _validation_error("record", "missing required field 'trajectory'")
    _validate_messages(d["trajectory"], "record.trajectory")
    _require_string(d, "final_source", "record")
    for numeric_key in (
        "initial_wall_us", "final_wall_us", "speedup", "snr_db",
    ):
        _validate_optional_number(d, numeric_key, "record")


def _validate_agentic(d: dict) -> None:
    if "messages" not in d:
        raise _validation_error("record", "missing required field 'messages'")
    _validate_messages(d["messages"], "record.messages")
    if "tool_trace" not in d:
        raise _validation_error("record", "missing required field 'tool_trace'")
    tool_trace = _require_list(d["tool_trace"], "record.tool_trace")
    for index, trace in enumerate(tool_trace):
        _require_dict(trace, f"record.tool_trace[{index}]")
    _require_string(d, "best_kernel", "record", nonempty=False)
    _validate_optional_number(d, "best_reward", "record")
    turns_to_best = d.get("turns_to_best")
    if turns_to_best is not None:
        _require_int(turns_to_best, "record.turns_to_best", minimum=0)
    if not isinstance(d.get("success"), bool):
        raise _validation_error("record.success", "must be a boolean")
    for list_key in ("reflections", "phase_trace"):
        items = _require_list(d.get(list_key), f"record.{list_key}")
        for index, item in enumerate(items):
            _require_dict(item, f"record.{list_key}[{index}]")
    if not isinstance(d.get("provenance"), dict):
        raise _validation_error("record.provenance", "must be an object")


_TYPE_VALIDATORS = {
    "repair": _validate_repair,
    "ranked_group": _validate_ranked_group,
    "win": _validate_win,
    "agentic": _validate_agentic,
}


def validate_record_dict(
    d: Any,
    *,
    expected_task_id: str | None = None,
    expected_type: str | None = None,
) -> dict:
    """Strictly validate one current-version KORE record.

    Unknown top-level metadata is retained for forward-compatible provenance,
    but record type, schema version, required structure and all numeric values
    are checked. ``expected_task_id`` and ``expected_type`` bind a record to its
    containing shard.
    """
    d = _require_dict(d, "record")
    _validate_json_tree(d)
    version = d.get(SCHEMA_VERSION_FIELD)
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != RECORD_SCHEMA_VERSION
    ):
        raise _validation_error(
            f"record.{SCHEMA_VERSION_FIELD}",
            f"expected {RECORD_SCHEMA_VERSION}, got {version!r}")
    record_type = d.get("type")
    if record_type not in _KNOWN_RECORD_TYPES:
        raise _validation_error("record.type", f"unknown record type {record_type!r}")
    if expected_type is not None and record_type != expected_type:
        raise _validation_error(
            "record.type", f"expected {expected_type!r}, got {record_type!r}")
    task_id = _require_string(d, "task_id", "record")
    if expected_task_id is not None and task_id != expected_task_id:
        raise _validation_error(
            "record.task_id", f"expected {expected_task_id!r}, got {task_id!r}")
    _TYPE_VALIDATORS[record_type](d)
    return d


def record_from_dict(
    d: dict,
    *,
    expected_task_id: str | None = None,
    expected_type: str | None = None,
    validate: bool = True,
) -> Record:
    """Dispatch a raw dict to its typed record class.

    Validation is strict by default. ``validate=False`` exists solely for the
    explicit legacy reader below.
    """
    if validate:
        validate_record_dict(
            d, expected_task_id=expected_task_id, expected_type=expected_type)
    elif not isinstance(d, dict):
        raise TypeError(f"record must be a dict, got {type(d)!r}")
    record_type = d.get("type")
    cls = _record_class(record_type)
    if cls is None:
        raise RecordValidationError(f"unknown record type: {record_type!r}")
    return cls.from_dict(d)


def record_to_dict(rec: Any) -> dict:
    """Convert a dataclass-like record to a detached JSON object.

    Known KORE record types are stamped with the current schema version. The
    function remains generic for training rows that do not carry a ``type``.
    """
    if hasattr(rec, "to_dict"):
        raw = rec.to_dict()
    elif isinstance(rec, dict):
        raw = rec
    else:
        raise TypeError(f"cannot serialize {type(rec)!r} to a record dict")
    if not isinstance(raw, dict):
        raise TypeError(
            f"{type(rec)!r}.to_dict() returned {type(raw)!r}, expected dict")
    d = dict(raw)
    if d.get("type") in _KNOWN_RECORD_TYPES:
        d.setdefault(SCHEMA_VERSION_FIELD, RECORD_SCHEMA_VERSION)
    return d


# Backwards-compatible private spelling used by older internal callers.
_to_dict = record_to_dict


def _record_line(
    rec: Any,
    *,
    validate_records: bool,
    expected_task_id: str | None,
    expected_type: str | None,
) -> bytes:
    d = record_to_dict(rec)
    _validate_json_tree(d)
    if validate_records:
        validate_record_dict(
            d, expected_task_id=expected_task_id, expected_type=expected_type)
    try:
        text = json.dumps(
            d,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(f"record is not JSON serializable: {exc}") from exc
    return text.encode("utf-8") + b"\n"


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(directory), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_replace(temp_path: Path, target_path: Path) -> None:
    if temp_path.parent.resolve() != target_path.parent.resolve():
        raise ValueError("atomic replacement requires a same-directory temporary file")
    os.replace(temp_path, target_path)
    _fsync_directory(target_path.parent)


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> Path:
    """Durably replace ``path`` with ``data`` using a unique local temp file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    open_fd = fd
    try:
        with os.fdopen(fd, "wb") as stream:
            open_fd = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _durable_replace(temp_path, path)
        return path
    except BaseException:
        if open_fd >= 0:
            os.close(open_fd)
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Union[str, Path], value: Any) -> Path:
    """Durably write one finite JSON value with canonical key ordering."""
    _validate_json_tree(value, "json")
    try:
        data = (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(f"value is not JSON serializable: {exc}") from exc
    return atomic_write_bytes(path, data)


def _prepare_jsonl(
    path: Union[str, Path],
    records: Iterable[Any],
    *,
    validate_records: bool = False,
    expected_task_id: str | None = None,
    expected_type: str | None = None,
) -> _PreparedJsonl:
    """Write and fsync a unique temp file without publishing it yet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    count = 0
    open_fd = fd
    try:
        with os.fdopen(fd, "wb") as stream:
            open_fd = -1
            for rec in records:
                line = _record_line(
                    rec,
                    validate_records=validate_records,
                    expected_task_id=expected_task_id,
                    expected_type=expected_type,
                )
                stream.write(line)
                digest.update(line)
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        return _PreparedJsonl(
            target_path=path,
            temp_path=temp_path,
            record_count=count,
            sha256=digest.hexdigest(),
        )
    except BaseException:
        if open_fd >= 0:
            os.close(open_fd)
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _commit_prepared_jsonl(prepared: _PreparedJsonl) -> Path:
    _durable_replace(prepared.temp_path, prepared.target_path)
    return prepared.target_path


def write_jsonl(
    path: Union[str, Path],
    records: Iterable[Any],
    *,
    validate_records: bool = False,
    expected_task_id: str | None = None,
    expected_type: str | None = None,
) -> Path:
    """Atomically and durably replace a JSONL file.

    The generic default accepts arbitrary dict-shaped training rows. Production
    KORE shard writers pass ``validate_records=True`` plus expected bindings.
    """
    prepared = _prepare_jsonl(
        path,
        records,
        validate_records=validate_records,
        expected_task_id=expected_task_id,
        expected_type=expected_type,
    )
    try:
        return _commit_prepared_jsonl(prepared)
    finally:
        try:
            prepared.temp_path.unlink()
        except FileNotFoundError:
            pass


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token!r}")


def _decode_json_record(raw: bytes, path: Path, lineno: int) -> dict:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JsonlValidationError(
            f"{path} line {lineno}: invalid UTF-8: {exc}") from exc
    if not text.strip():
        raise JsonlValidationError(f"{path} line {lineno}: blank lines are not allowed")
    try:
        value = json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise JsonlValidationError(
            f"{path} line {lineno}: malformed JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise JsonlValidationError(
            f"{path} line {lineno}: record must be an object, "
            f"got {type(value).__name__}")
    try:
        _validate_json_tree(value)
    except RecordValidationError as exc:
        raise JsonlValidationError(f"{path} line {lineno}: {exc}") from exc
    return value


def read_jsonl(
    path: Union[str, Path],
    typed: bool = True,
    *,
    expected_task_id: str | None = None,
    expected_type: str | None = None,
) -> list:
    """Strictly read a JSONL file.

    Bad JSON, non-object lines, non-finite values, unknown/unversioned record
    types and schema violations fail closed. Use ``read_jsonl_legacy`` only in
    quarantine or migration code that deliberately skips bad legacy rows.
    """
    path = Path(path)
    if not path.exists():
        return []
    out: list = []
    with path.open("rb") as stream:
        for lineno, raw in enumerate(stream, start=1):
            d = _decode_json_record(raw, path, lineno)
            try:
                if typed:
                    out.append(record_from_dict(
                        d,
                        expected_task_id=expected_task_id,
                        expected_type=expected_type,
                    ))
                else:
                    if expected_task_id is not None or expected_type is not None:
                        validate_record_dict(
                            d,
                            expected_task_id=expected_task_id,
                            expected_type=expected_type,
                        )
                    out.append(d)
            except (KeyError, TypeError, ValueError) as exc:
                raise JsonlValidationError(
                    f"{path} line {lineno}: invalid record: {exc}") from exc
    return out


def validate_jsonl_shard(
    path: Union[str, Path],
    *,
    expected_task_id: str,
    expected_type: str,
) -> ShardValidation:
    """Validate every line and hash the exact bytes from one file descriptor."""
    path = Path(path)
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for lineno, raw in enumerate(stream, start=1):
            digest.update(raw)
            if not raw.endswith(b"\n"):
                raise JsonlValidationError(
                    f"{path} line {lineno}: truncated line (missing newline)")
            d = _decode_json_record(raw, path, lineno)
            try:
                validate_record_dict(
                    d,
                    expected_task_id=expected_task_id,
                    expected_type=expected_type,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise JsonlValidationError(
                    f"{path} line {lineno}: invalid record: {exc}") from exc
            count += 1
    return ShardValidation(record_count=count, sha256=digest.hexdigest())


def read_jsonl_legacy(
    path: Union[str, Path],
    typed: bool = True,
) -> list:
    """Tolerantly read legacy JSONL for quarantine/migration only.

    Missing schema versions are accepted and bad rows are logged and skipped.
    This function must never be used to decide whether a production shard is
    complete.
    """
    path = Path(path)
    if not path.exists():
        return []
    out: list = []
    with path.open("rb") as stream:
        for lineno, raw in enumerate(stream, start=1):
            try:
                text = raw.decode("utf-8").strip()
                if not text:
                    continue
                d = json.loads(text, parse_constant=_reject_json_constant)
                if not isinstance(d, dict):
                    raise TypeError(f"record must be an object, got {type(d).__name__}")
                _validate_json_tree(d)
                if typed and d.get("type") in _KNOWN_RECORD_TYPES:
                    out.append(record_from_dict(d, validate=False))
                else:
                    out.append(d)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                _LOG.warning(
                    "quarantining malformed legacy record in %s line %d: %s",
                    path,
                    lineno,
                    exc,
                )
    return out


__all__ = [
    "GPU_DEFAULT",
    "JsonlValidationError",
    "RECORD_SCHEMA_VERSION",
    "RecordValidationError",
    "RepairRecord",
    "RankedGroupRecord",
    "SCHEMA_VERSION_FIELD",
    "ShardValidation",
    "WinRecord",
    "atomic_write_bytes",
    "atomic_write_json",
    "read_jsonl",
    "read_jsonl_legacy",
    "record_from_dict",
    "record_to_dict",
    "validate_jsonl_shard",
    "validate_record_dict",
    "write_jsonl",
]
