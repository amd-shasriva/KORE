"""Narrow provenance helpers for the fail-closed campaign manifest.

The campaign owns persistence and compatibility policy.  This module only
provides deterministic identities and streaming SHA-256 digests so those checks
do not need to load model checkpoints or datasets into memory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional


DIGEST_PREFIX = "sha256:"
SOURCE_PATHS = ("scripts", "kore", "configs", "pyproject.toml")
TOKENIZER_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


class LineageError(RuntimeError):
    """A provenance identity could not be established exactly."""


def _stable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _stable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _stable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_stable(v) for v in value), key=lambda v: canonical_json(v))
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return _stable(vars(value))
    return repr(value)


def canonical_json(value: Any) -> str:
    """Canonical JSON used by every manifest contract digest."""
    try:
        return json.dumps(
            _stable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LineageError(f"value is not canonically serializable: {exc}") from exc


def object_digest(value: Any) -> str:
    return DIGEST_PREFIX + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: os.PathLike[str] | str, *, chunk_size: int = 8 << 20) -> str:
    """Hash one file in bounded memory."""
    p = Path(path)
    if not p.is_file():
        raise LineageError(f"required provenance file is missing: {p}")
    h = hashlib.sha256()
    try:
        with p.open("rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
    except OSError as exc:
        raise LineageError(f"cannot read provenance file {p}: {exc}") from exc
    return DIGEST_PREFIX + h.hexdigest()


def digest_files(
    paths: Iterable[os.PathLike[str] | str],
    *,
    root: Optional[os.PathLike[str] | str] = None,
) -> dict:
    """Return a path-sensitive Merkle-style digest for regular files."""
    files = sorted({Path(p).resolve() for p in paths}, key=lambda p: str(p))
    if not files:
        raise LineageError("cannot digest an empty file set")
    base = Path(root).resolve() if root is not None else None
    entries = []
    for p in files:
        try:
            rel = str(p.relative_to(base)) if base is not None else str(p)
        except ValueError:
            rel = str(p)
        entries.append({"path": rel, "size": p.stat().st_size, "sha256": file_digest(p)})
    return {
        "algorithm": "sha256",
        "digest": object_digest(entries),
        "total_bytes": sum(e["size"] for e in entries),
        "files": entries,
    }


def digest_tree(
    root: os.PathLike[str] | str,
    *,
    include: Optional[Callable[[Path], bool]] = None,
) -> dict:
    base = Path(root).resolve()
    if not base.is_dir():
        raise LineageError(f"required provenance directory is missing: {base}")
    paths = [
        p for p in base.rglob("*")
        if p.is_file() and (include is None or include(p.relative_to(base)))
    ]
    return digest_files(paths, root=base)


def _git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LineageError(f"cannot establish git source identity: {exc}") from exc


def git_source_identity(repo_root: os.PathLike[str] | str) -> dict:
    """Bind a commit plus staged/unstaged/untracked source content."""
    repo = Path(repo_root).resolve()
    commit = _git(repo, "rev-parse", "HEAD").decode().strip()
    status = _git(
        repo, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *SOURCE_PATHS,
    )
    diff = _git(repo, "diff", "--binary", "--no-ext-diff", "HEAD", "--", *SOURCE_PATHS)
    untracked_raw = _git(
        repo, "ls-files", "--others", "--exclude-standard", "-z", "--", *SOURCE_PATHS,
    )
    untracked = sorted(x.decode("utf-8", "surrogateescape") for x in untracked_raw.split(b"\0") if x)
    h = hashlib.sha256()
    h.update(commit.encode())
    h.update(b"\0")
    h.update(diff)
    for rel in untracked:
        h.update(b"\0untracked\0")
        h.update(rel.encode("utf-8", "surrogateescape"))
        h.update(b"\0")
        p = repo / rel
        if p.is_file():
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(8 << 20), b""):
                    h.update(chunk)
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_status_digest": DIGEST_PREFIX + hashlib.sha256(status).hexdigest(),
        "content_digest": DIGEST_PREFIX + h.hexdigest(),
        "scope": list(SOURCE_PATHS),
    }


def _version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_arches() -> list[str]:
    try:
        text = subprocess.run(
            ["rocminfo"], check=True, capture_output=True, text=True, timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted(set(re.findall(r"\bgfx[0-9a-f]+\b", text, flags=re.IGNORECASE)))


def runtime_identity() -> dict:
    """Stable runtime/hardware identity plus a resume compatibility digest."""
    packages = {
        name: _version(name)
        for name in (
            "accelerate", "datasets", "huggingface-hub", "numpy", "peft",
            "safetensors", "torch", "transformers", "triton", "trl",
        )
    }
    rocm_version = None
    for candidate in (Path("/opt/rocm/.info/version"), Path("/opt/rocm/.info/version-dev")):
        try:
            if candidate.is_file():
                rocm_version = candidate.read_text().strip()
                break
        except OSError:
            pass
    compatibility = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "packages": packages,
        "rocm_version": rocm_version,
        "gpu_arches": _gpu_arches(),
    }
    return {
        **compatibility,
        "executable": sys.executable,
        "hostname": socket.gethostname(),
        "visibility": {
            k: os.environ.get(k)
            for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
            if os.environ.get(k) is not None
        },
        "compatibility_digest": object_digest(compatibility),
    }


def architecture_signature(config: Mapping[str, Any]) -> dict:
    keys = (
        "model_type", "architectures", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "vocab_size", "max_position_embeddings", "tie_word_embeddings",
    )
    return {k: _stable(config.get(k)) for k in keys if config.get(k) is not None}


def _tokenizer_files(snapshot: Path) -> list[Path]:
    return [
        p for p in snapshot.iterdir()
        if p.is_file() and (
            p.name in TOKENIZER_NAMES
            or p.name.startswith(("tokenizer.", "vocab."))
            or p.suffix in {".model", ".tiktoken"}
        )
    ]


def resolve_model_snapshot(model_id: str, revision: Optional[str] = None) -> tuple[dict, dict, str]:
    """Resolve a local model or immutable Hugging Face snapshot exactly.

    Returns ``(model_identity, tokenizer_identity, load_path)``.  Remote models
    are materialized through the HF cache at the resolved commit so every later
    ``from_pretrained(load_path)`` call is pinned without changing trainer APIs.
    """
    requested = str(model_id)
    local = Path(requested).expanduser()
    if local.exists():
        snapshot = local.resolve()
        tree = digest_tree(
            snapshot,
            include=lambda rel: not any(part.startswith(("checkpoint-", "retention_cache"))
                                        for part in rel.parts),
        )
        resolved = f"local-{tree['digest'].split(':', 1)[1]}"
        kind = "local"
        content_digest = tree["digest"]
    else:
        try:
            from huggingface_hub import HfApi, snapshot_download

            info = HfApi().model_info(requested, revision=revision)
            resolved = str(info.sha or "")
            if not re.fullmatch(r"[0-9a-fA-F]{40,64}", resolved):
                raise LineageError(
                    f"Hugging Face did not return an immutable commit for {requested!r}"
                )
            snapshot = Path(snapshot_download(repo_id=requested, revision=resolved)).resolve()
        except LineageError:
            raise
        except Exception as exc:  # noqa: BLE001 - turn all resolution failures fail-closed
            raise LineageError(
                f"cannot resolve/materialize exact model revision {requested!r}@{revision or 'main'}: {exc}"
            ) from exc
        kind = "huggingface"
        content_digest = object_digest({"repo_id": requested, "commit": resolved})

    config_path = snapshot / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise LineageError(f"model snapshot has no readable config.json: {snapshot}: {exc}") from exc
    if not isinstance(config, dict) or not config:
        raise LineageError(f"model config is empty or invalid: {config_path}")
    weights = (
        sorted(snapshot.glob("*.safetensors"))
        + sorted(snapshot.glob("pytorch_model*.bin"))
        + sorted(snapshot.glob("adapter_model*.bin"))
    )
    if not weights or any(path.stat().st_size <= 0 for path in weights):
        raise LineageError(f"model snapshot has no non-empty weight files: {snapshot}")
    tok_files = _tokenizer_files(snapshot)
    if not tok_files:
        raise LineageError(f"model snapshot has no tokenizer provenance files: {snapshot}")
    tok_digest = digest_files(tok_files, root=snapshot)
    model_identity = {
        "requested_id": requested,
        "requested_revision": revision,
        "resolved_revision": resolved,
        "kind": kind,
        "snapshot_path": str(snapshot),
        "content_digest": content_digest,
        "config_digest": file_digest(config_path),
        "architecture": architecture_signature(config),
    }
    tokenizer_identity = {
        "requested_id": requested,
        "requested_revision": revision,
        "resolved_revision": resolved,
        "kind": kind,
        "snapshot_path": str(snapshot),
        "content_digest": tok_digest["digest"],
        "files": tok_digest["files"],
    }
    return model_identity, tokenizer_identity, str(snapshot)


__all__ = [
    "LineageError",
    "architecture_signature",
    "canonical_json",
    "digest_files",
    "digest_tree",
    "file_digest",
    "git_source_identity",
    "object_digest",
    "resolve_model_snapshot",
    "runtime_identity",
]
