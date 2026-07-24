"""Strict, side-effect-free completion checks for operational wrappers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from typing import Iterable, Mapping

from .runtime import ArtifactStatus, task_set_identity


_MODEL_CONFIGS = ("config.json", "adapter_config.json")
_MODEL_WEIGHTS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def _regular(path: Path, *, nonempty: bool = True) -> str | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return f"missing artifact: {path}"
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return f"artifact is not a real regular file: {path}"
    if nonempty and info.st_size == 0:
        return f"artifact is empty: {path}"
    return None


def _load_json(path: Path) -> tuple[object | None, str | None]:
    error = _regular(path)
    if error:
        return None, error
    try:
        return json.loads(path.read_text()), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON artifact {path}: {exc}"


def _validate_jsonl(path: Path, *, require_records: bool = True) -> tuple[int, str | None]:
    error = _regular(path, nonempty=require_records)
    if error:
        return 0, error
    count = 0
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return 0, (
                        f"invalid JSONL artifact {path}:{line_number}: "
                        "record is not an object"
                    )
                count += 1
    except (OSError, json.JSONDecodeError) as exc:
        return 0, f"invalid JSONL artifact {path}: {exc}"
    if require_records and count == 0:
        return 0, f"JSONL artifact has no records: {path}"
    return count, None


def _resolve(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo / path


def verify_model_artifact(path: str | Path, *, repo: str | Path = ".") -> ArtifactStatus:
    root = _resolve(Path(repo).resolve(), path)
    try:
        info = root.lstat()
    except FileNotFoundError:
        return ArtifactStatus.failure(f"model artifact directory is missing: {root}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return ArtifactStatus.failure(
            f"model artifact is not a real directory: {root}"
        )
    errors = []
    if not any((root / name).is_file() for name in _MODEL_CONFIGS):
        errors.append(
            f"model artifact has no {' or '.join(_MODEL_CONFIGS)}: {root}"
        )
    if not any((root / name).is_file() for name in _MODEL_WEIGHTS):
        errors.append(f"model artifact has no final model weights: {root}")
    incomplete = sorted(root.glob("*.inprogress")) + sorted(root.glob("*.tmp"))
    if incomplete:
        errors.append(
            "model artifact contains incomplete markers: "
            + ", ".join(str(path) for path in incomplete)
        )
    if errors:
        return ArtifactStatus.failure(*errors)
    return ArtifactStatus.success(path=str(root))


def _distinct_wins(path: Path) -> tuple[int, str | None]:
    count, error = _validate_jsonl(path)
    if error:
        return 0, error
    seen = set()
    try:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                source = str(value.get("final_source", "") or "").strip()
                if source:
                    seen.add(hashlib.sha256(source.encode()).hexdigest())
    except (OSError, json.JSONDecodeError) as exc:
        return 0, f"invalid wins artifact {path}: {exc}"
    if count and not seen:
        return 0, f"wins artifact has no final_source records: {path}"
    return len(seen), None


def _stage_event_done(path: Path, stage: str) -> tuple[bool, str | None]:
    _count, error = _validate_jsonl(path)
    if error:
        return False, error
    try:
        with path.open() as handle:
            events = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid campaign event artifact {path}: {exc}"
    done = any(
        event.get("stage") == stage and event.get("status") in ("done", "skipped")
        for event in events
    )
    return done, None if done else f"no completed {stage} event in {path}"


def verify_task_shards(
    data_root: str | Path,
    task_ids: Iterable[str],
    *,
    target_wins: int = 1,
    kinds: tuple[str, ...] = ("repair", "groups", "wins"),
) -> ArtifactStatus:
    root = Path(data_root).resolve()
    if target_wins < 0:
        raise ValueError("target_wins must be non-negative")
    identity = task_set_identity(task_ids)
    errors: list[str] = []
    checked = 0
    for task_id in identity.task_ids:
        for kind in kinds:
            path = root / kind / f"{task_id}.jsonl"
            marker = path.with_suffix(path.suffix + ".inprogress")
            if marker.exists() or marker.is_symlink():
                errors.append(f"incomplete shard marker present: {marker}")
                continue
            if kind == "wins":
                wins, error = _distinct_wins(path)
                if error:
                    errors.append(error)
                elif wins < target_wins:
                    errors.append(
                        f"wins below target for {task_id}: {wins}/{target_wins}"
                    )
            else:
                _count, error = _validate_jsonl(path)
                if error:
                    errors.append(error)
            checked += 1
    details = {
        "data_root": str(root),
        "task_count": identity.count,
        "task_sha256": identity.sha256,
        "shards_checked": checked,
        "target_wins": target_wins,
    }
    if errors:
        return ArtifactStatus.failure(*errors, details=details)
    return ArtifactStatus.success(**details)


def verify_sft_gate(
    manifest_path: str | Path,
    candidate: str | Path,
    *,
    repo: str | Path = ".",
) -> ArtifactStatus:
    repo_path = Path(repo).resolve()
    manifest_file = _resolve(repo_path, manifest_path)
    value, error = _load_json(manifest_file)
    if error:
        return ArtifactStatus.failure(error)
    if not isinstance(value, dict):
        return ArtifactStatus.failure(f"manifest is not a JSON object: {manifest_file}")
    errors = []
    if "sft" not in set(value.get("done_stages") or []):
        errors.append("manifest does not mark sft complete")
    expected = _resolve(repo_path, candidate).resolve()
    recorded_value = value.get("sft_ckpt")
    if not recorded_value:
        errors.append("manifest has no sft_ckpt")
    else:
        recorded = _resolve(repo_path, str(recorded_value)).resolve()
        if recorded != expected:
            errors.append(
                f"manifest sft_ckpt mismatch: recorded={recorded} expected={expected}"
            )
    model = verify_model_artifact(expected, repo=repo_path)
    errors.extend(model.errors)
    if errors:
        return ArtifactStatus.failure(*errors)
    return ArtifactStatus.success(manifest=str(manifest_file), candidate=str(expected))


def verify_grpo_config(
    config_path: str | Path, *, repo: str | Path = "."
) -> ArtifactStatus:
    repo_path = Path(repo).resolve()
    config_file = _resolve(repo_path, config_path)
    value, error = _load_json(config_file)
    if error:
        return ArtifactStatus.failure(error)
    if not isinstance(value, dict):
        return ArtifactStatus.failure(f"GRPO config is not a JSON object: {config_file}")
    output = value.get("output_dir")
    if not output:
        return ArtifactStatus.failure(f"GRPO config has no output_dir: {config_file}")
    status = verify_model_artifact(str(output), repo=repo_path)
    if not status.ok:
        return status
    return ArtifactStatus.success(config=str(config_file), output_dir=str(output))


def verify_campaign(
    repo: str | Path,
    data_root: str | Path,
    required_stages: Iterable[str],
) -> ArtifactStatus:
    repo_path = Path(repo).resolve()
    root = _resolve(repo_path, data_root).resolve()
    stages = tuple(dict.fromkeys(str(stage).strip() for stage in required_stages))
    if not stages or any(not stage for stage in stages):
        raise ValueError("required_stages must be non-empty")
    manifest_file = root / "campaign_manifest.json"
    value, error = _load_json(manifest_file)
    if error:
        return ArtifactStatus.failure(error)
    if not isinstance(value, dict):
        return ArtifactStatus.failure(f"manifest is not a JSON object: {manifest_file}")
    done = set(value.get("done_stages") or [])
    errors: list[str] = []
    details: dict[str, object] = {
        "manifest": str(manifest_file),
        "required_stages": list(stages),
    }
    for stage in stages:
        if stage not in done:
            errors.append(f"manifest does not mark stage complete: {stage}")
            continue
        if stage == "build":
            for relative in ("sft/multicap.jsonl", "dpo/pairs.jsonl"):
                _count, item_error = _validate_jsonl(root / relative)
                if item_error:
                    errors.append(item_error)
        elif stage in ("midtrain", "sft", "dpo", "grpo"):
            key = f"{stage}_ckpt"
            artifact = value.get(key)
            if not artifact:
                errors.append(f"manifest has no {key}")
            else:
                model = verify_model_artifact(str(artifact), repo=repo_path)
                errors.extend(model.errors)
        elif stage == "soup":
            artifact = value.get("final")
            if not artifact:
                errors.append("manifest has no final soup artifact")
            else:
                model = verify_model_artifact(str(artifact), repo=repo_path)
                errors.extend(model.errors)
        elif stage == "eval":
            _report, item_error = _load_json(root / "eval" / "bakeoff.json")
            if item_error:
                errors.append(item_error)
        elif stage == "agentic":
            candidates = sorted((root / "agentic").glob("*.jsonl"))
            if not candidates:
                errors.append(f"no agentic artifacts under {root / 'agentic'}")
            for candidate in candidates:
                _count, item_error = _validate_jsonl(candidate)
                if item_error:
                    errors.append(item_error)
        elif stage == "datagen":
            task_ids = value.get("train_tasks")
            if not isinstance(task_ids, list) or not task_ids:
                errors.append("manifest has no immutable train task set for datagen")
            else:
                datagen = verify_task_shards(root, [str(item) for item in task_ids])
                errors.extend(datagen.errors)
                details.update(datagen.details)
        elif stage == "reverify":
            _done, item_error = _stage_event_done(
                root / "campaign_events.jsonl", "reverify"
            )
            if item_error:
                errors.append(item_error)
        elif stage == "evolve":
            candidates = sorted((root / "wins").glob("*.evolve.jsonl"))
            if not candidates:
                errors.append(f"no evolve artifacts under {root / 'wins'}")
        else:
            errors.append(f"no strict verifier is defined for stage: {stage}")
    if errors:
        return ArtifactStatus.failure(*errors, details=details)
    return ArtifactStatus.success(**details)


def status_json(status: ArtifactStatus) -> dict:
    return {
        "ok": status.ok,
        "errors": list(status.errors),
        "details": dict(status.details),
    }
