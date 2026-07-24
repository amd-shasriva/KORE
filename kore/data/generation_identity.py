"""Deterministic identity envelope for generated-data shard contracts.

The identity deliberately binds inputs that can change record meaning: task files
and split, generator/prompt/evaluator source, resolved configuration and rigor,
teacher backend plus immutable revision, runtime software/hardware, seeds, and
all non-secret behavioral environment variables. A receipt from another identity
is never silently reused.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

DATA_LANE_VERSION = "kore-datagen-v1"
GENERATION_IDENTITY_VERSION = 1

_ROOT = Path(__file__).resolve().parents[2]
_GENERATOR_SOURCES = {
    "repair": ("kore/data/gen_repair.py", "kore/data/mutate.py"),
    "groups": ("kore/data/gen_groups.py",),
    "wins": ("kore/data/gen_wins.py", "kore/data/grounded_reasoning.py"),
    "agentic": (
        "kore/data/gen_agentic.py",
        "kore/agent/harness.py",
        "kore/agent/tools.py",
        "kore/agent/format.py",
    ),
}
_COMMON_SOURCES = (
    "kore/config.py",
    "kore/data/generation_identity.py",
    "kore/data/parallel_datagen.py",
    "kore/data/schemas.py",
    "kore/data/prompts.py",
    "kore/data/teacher.py",
    "kore/data/verify_rigor.py",
    "kore/policy/format.py",
    "kore/env/kore_env.py",
    "kore/reward/reward.py",
    "kore/tasks/registry.py",
)
_BEHAVIOR_ENV_EXACT = {
    "GPU_TARGET",
    "ROCM_PATH",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "AMD_LLM_API_VERSION",
    "AMD_LLM_GATEWAY_URL",
}
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


class GenerationIdentityError(ValueError):
    """Required immutable generation identity could not be resolved."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def identity_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def source_manifest(paths: Iterable[Path]) -> dict:
    files: dict[str, str] = {}
    for raw_path in sorted({Path(path).resolve() for path in paths}):
        if not raw_path.is_file():
            raise GenerationIdentityError(f"identity source file missing: {raw_path}")
        try:
            name = str(raw_path.relative_to(_ROOT))
        except ValueError:
            name = str(raw_path)
        files[name] = file_sha256(raw_path)
    return {"files": files, "digest": identity_digest(files)}


def _task_files(task: Any) -> list[Path]:
    task_dir = Path(getattr(task, "dir", ""))
    if not task_dir.is_dir():
        raise GenerationIdentityError(
            f"task {getattr(task, 'task_id', None)!r} has no source directory")
    files = [
        path
        for path in task_dir.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() in {".py", ".yaml", ".yml", ".json"}
    ]
    if not files:
        raise GenerationIdentityError(f"task source directory is empty: {task_dir}")
    return files


def task_identity(task: Any) -> dict:
    from kore.tasks.registry import (
        heldout_tasks,
        is_heldout,
        operator_family,
        train_tasks,
    )

    task_id = getattr(task, "task_id", None)
    if not isinstance(task_id, str) or not task_id:
        raise GenerationIdentityError("task identity requires a non-empty task_id")
    files = source_manifest(_task_files(task))
    identity = {
        "task_id": task_id,
        "operation": getattr(task, "operation", None),
        "operator_family": operator_family(task),
        "dtype": getattr(task, "dtype", None),
        "backend": getattr(task, "backend", None),
        "gpu_target": getattr(task, "gpu_target", None),
        "comparison_baseline": getattr(task, "comparison_baseline", None),
        "snr_threshold": getattr(task, "snr_threshold", None),
        "shapes": _jsonable(getattr(task, "shapes", [])),
        "registry_split": "heldout" if is_heldout(task) else "train",
        "registry_membership": {
            "train": sorted(item.task_id for item in train_tasks()),
            "heldout": sorted(item.task_id for item in heldout_tasks()),
        },
        "files": files,
    }
    identity["digest"] = identity_digest(identity)
    return identity


def code_identity(kind: str) -> dict:
    if kind not in _GENERATOR_SOURCES:
        raise GenerationIdentityError(f"unknown generator kind {kind!r}")
    paths = [_ROOT / name for name in (*_COMMON_SOURCES, *_GENERATOR_SOURCES[kind])]
    return source_manifest(paths)


def resolved_config_identity(task: Any) -> dict:
    from kore.config import CONFIG
    from kore.data.verify_rigor import RIGOR_ENV

    config = _jsonable(CONFIG)
    rigor = {name: os.environ.get(name) for name in sorted(RIGOR_ENV)}
    evaluation = {
        "task_snr_threshold": getattr(task, "snr_threshold", None),
        "comparison_baseline": getattr(task, "comparison_baseline", None),
        "full_validation": True,
        "multi_shape": True,
        "config": config,
        "rigor": rigor,
    }
    evaluation["digest"] = identity_digest(evaluation)
    return evaluation


def behavioral_environment() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, value in os.environ.items():
        if not (name.startswith("KORE_") or name in _BEHAVIOR_ENV_EXACT):
            continue
        if any(marker in name.upper() for marker in _SECRET_MARKERS):
            continue
        out[name] = value
    return dict(sorted(out.items()))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def software_identity() -> dict:
    packages = {
        name: _package_version(name)
        for name in ("kore", "torch", "triton", "pyyaml", "anthropic", "openai")
    }
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def hardware_identity(task: Any) -> dict:
    explicit = os.environ.get("KORE_HARDWARE_ID")
    return {
        "architecture": getattr(task, "gpu_target", None),
        "hardware_id": explicit or platform.node(),
        "machine": platform.machine(),
        "visible_devices": {
            name: os.environ.get(name)
            for name in (
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "rocm_path": os.environ.get("ROCM_PATH", "/opt/rocm"),
    }


def resolve_teacher_identity(
    teacher_kind: str,
    model: Any,
    immutable_revision: str | None,
) -> dict:
    kind = (teacher_kind or "").lower()
    if kind in ("anthropic", "opus"):
        kind = "claude"
    resolved_model = (
        str(model)
        if model is not None
        else os.environ.get("KORE_TEACHER_MODEL")
    )
    if kind == "stub":
        resolved_model = resolved_model or "stub"
        immutable_revision = immutable_revision or file_sha256(
            _ROOT / "kore/data/teacher.py")
    if not resolved_model:
        if kind == "claude":
            resolved_model = "claude-opus-4.8"
        else:
            raise GenerationIdentityError(
                f"teacher backend {kind!r} requires an explicit model")
    revision = immutable_revision or os.environ.get("KORE_TEACHER_REVISION")
    if not revision:
        raise GenerationIdentityError(
            "teacher identity requires --model-teacher-revision or "
            "KORE_TEACHER_REVISION; a mutable model label is insufficient")
    return {
        "backend": kind,
        "model": resolved_model,
        "immutable_revision": revision,
        "api_version": os.environ.get("AMD_LLM_API_VERSION"),
        "endpoint": os.environ.get("AMD_LLM_GATEWAY_URL"),
        "resilient": True,
    }


def build_generation_identity(
    *,
    kind: str,
    task: Any,
    teacher_kind: str,
    model_teacher: Any,
    model_teacher_revision: str | None,
    seed: int,
) -> dict:
    identity = {
        "identity_version": GENERATION_IDENTITY_VERSION,
        "data_lane_version": DATA_LANE_VERSION,
        "kind": kind,
        "task": task_identity(task),
        "code": code_identity(kind),
        "evaluation": resolved_config_identity(task),
        "teacher": resolve_teacher_identity(
            teacher_kind, model_teacher, model_teacher_revision),
        "hardware": hardware_identity(task),
        "software": software_identity(),
        "seeds": {"generator_seed": int(seed)},
        "behavioral_environment": behavioral_environment(),
    }
    identity["digest"] = identity_digest(identity)
    return identity


def validate_generation_identity(identity: Any, *, task_id: str, kind: str) -> dict:
    if not isinstance(identity, dict):
        raise GenerationIdentityError("generation identity must be an object")
    digest = identity.get("digest")
    payload = {key: value for key, value in identity.items() if key != "digest"}
    if digest != identity_digest(payload):
        raise GenerationIdentityError("generation identity digest mismatch")
    if identity.get("identity_version") != GENERATION_IDENTITY_VERSION:
        raise GenerationIdentityError("unsupported generation identity version")
    if identity.get("data_lane_version") != DATA_LANE_VERSION:
        raise GenerationIdentityError("generation data-lane version mismatch")
    if identity.get("kind") != kind:
        raise GenerationIdentityError("generation identity kind mismatch")
    task = identity.get("task")
    if not isinstance(task, dict) or task.get("task_id") != task_id:
        raise GenerationIdentityError("generation identity task mismatch")
    if not isinstance(identity.get("code"), dict):
        raise GenerationIdentityError("generation identity lacks source hashes")
    teacher = identity.get("teacher")
    if not isinstance(teacher, dict) or not teacher.get("immutable_revision"):
        raise GenerationIdentityError("generation identity lacks immutable teacher revision")
    if not isinstance(identity.get("evaluation"), dict):
        raise GenerationIdentityError("generation identity lacks evaluation contract")
    if kind not in _GENERATOR_SOURCES:
        raise GenerationIdentityError(f"unknown generator kind {kind!r}")
    return identity


__all__ = [
    "DATA_LANE_VERSION",
    "GENERATION_IDENTITY_VERSION",
    "GenerationIdentityError",
    "behavioral_environment",
    "build_generation_identity",
    "canonical_json",
    "code_identity",
    "file_sha256",
    "hardware_identity",
    "identity_digest",
    "resolve_teacher_identity",
    "software_identity",
    "source_manifest",
    "task_identity",
    "validate_generation_identity",
]
