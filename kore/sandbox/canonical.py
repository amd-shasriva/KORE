"""Canonical serialization and domain-separated sandbox digests."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


CANONICAL_FORMAT = "kore-canonical-json-v1"
DIGEST_ALGORITHM = "sha256"


def _canonical_value(value: Any) -> Any:
    """Convert supported values to a deterministic JSON value.

    Mappings must have string keys. Sets and arbitrary objects are deliberately
    rejected: accepting their process-dependent iteration/repr would make the
    signed bytes ambiguous.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not permit NaN or infinity")
        return value
    if isinstance(value, bytes):
        return {"$bytes_b64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
            if field.metadata.get("canonical", True)
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mappings require string keys")
        return {
            key: _canonical_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported canonical JSON type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize *value* to stable UTF-8 JSON bytes."""

    normalized = _canonical_value(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_text(value: Any) -> str:
    """Text form of :func:`canonical_json_bytes`."""

    return canonical_json_bytes(value).decode("utf-8")


def digest_value(domain: str, value: Any) -> str:
    """Return a domain-separated SHA-256 digest of a canonical value."""

    if not domain or "\x00" in domain:
        raise ValueError("digest domain must be a non-empty NUL-free string")
    prefix = f"kore-sandbox:{domain}:v1\x00".encode("ascii")
    return hashlib.sha256(prefix + canonical_json_bytes(value)).hexdigest()


def task_digest(task_id: str, descriptor: Mapping[str, Any]) -> str:
    return digest_value("task", {"task_id": task_id, "descriptor": descriptor})


def source_digest(source: str) -> str:
    return digest_value("source", source)


def policy_digest(policy: Any) -> str:
    return digest_value("policy", policy)


def toolchain_digest(descriptor: Mapping[str, Any]) -> str:
    return digest_value("toolchain", descriptor)


def runtime_digest(descriptor: Mapping[str, Any]) -> str:
    return digest_value("runtime", descriptor)


def output_digest(stdout: bytes, stderr: bytes) -> str:
    return digest_value("output", {"stdout": stdout, "stderr": stderr})
