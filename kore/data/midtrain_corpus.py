"""Leakage-safe Stage-0 corpus assembly.

Production builds are fail-closed and reproducible:

* repository/dataset lineages are split and held out before pairs or chunks are
  derived;
* every output row carries source provenance and SHA-256 ancestry;
* external datasets and the model tokenizer use explicit immutable revisions;
* the full frozen benchmark-text artifact is mandatory;
* chunk admission is measured with the actual tokenizer, not a four-character
  estimate; and
* exact/near dedup is source-channel aware, so intentional weighting channels
  survive while duplicate origins remain auditable.

Tests and local smoke work must opt into ``development_mode=True``. Development
uses an exact UTF-8 byte tokenizer and bundled smoke benchmark references, and
labels both choices in every build report.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import random
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from kore.data.general_replay import REPLAY_KINDS, load_general_replay
from kore.obs import get_logger

log = get_logger("data.midtrain_corpus")

SOURCE_METADATA_SCHEMA_VERSION = "1.0"

# Deprecated compatibility constant. It remains importable for callers that used
# it for estimates, but corpus admission never uses it.
CHARS_PER_TOKEN = 4

_HIP_EXTS = (".cu", ".cuh", ".hip", ".cpp", ".cc", ".hpp", ".h")
_ASM_EXTS = (".s", ".S")
_DOC_EXTS = (".md", ".rst")
_TRITON_MARKERS = ("import triton", "triton.jit", "triton.language", "tl.")
_SKIP_DIR_PARTS = frozenset({
    "__pycache__", ".git", "node_modules", ".venv", "venv", "build", "dist",
    ".mypy_cache", ".pytest_cache", ".egg-info", "_drafts",
})
_DOC_KEYWORDS = (
    "rocprof", "rocprofiler", "omniperf", "tuning", "perf", "optimize",
    "occupancy", "roofline", "benchmark", "profil", "triton", "hip", "rocm",
    "kernel", "composable", "ck_tile", "tensile", "hipblaslt", "rocblas",
    "aiter", "amd", "gpu", "instinct", "mi300", "mi325", "mi350", "mi355",
    "mi200", "gfx", "cdna", "wavefront", "wave", "lds", "vgpr", "mfma",
    "wmma", "isa", "swizzle", "tile", "matmul", "gemm", "attention", "quant",
    "fp8", "bf16", "mxfp4", "mxfp8", "microscaling",
)
_HELDOUT_CONCEPT_RE = re.compile(
    r"(mla|multi.?head.?latent|flashmla|paged?[_.\-]?(attn|attention|kv|cache|decode|prefill))",
    re.IGNORECASE,
)
_KORE_AUTHORED_SOURCES = frozenset({"kore_tasks", "pytorch_triton_pairs"})
_DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "amd_kernels": 2.0,
    "pytorch_triton_pairs": 2.0,
    "kore_tasks": 1.5,
    "kernelbook": 1.5,
}
_MUTABLE_REVISIONS = frozenset({"", "main", "master", "head", "latest"})


@dataclass(frozen=True)
class SourceRoot:
    path: Path
    repository_url: str
    commit: str
    license: str
    source_id: str
    lineage_id: str
    verified: bool
    source_timestamp: Optional[str] = None
    development_mode: bool = False
    path_prefix: str = ""


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    text: str
    source: str
    metadata: Mapping[str, Any]


class _DevelopmentByteTokenizer:
    """Exact offline tokenizer for explicitly labeled development builds."""

    name_or_path = "kore/development-utf8-bytes"
    revision = "development-byte-v1"

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list((text or "").encode("utf-8"))


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _norm_hash(text: str) -> str:
    normalized = " ".join((text or "").split())
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _immutable_revision(value: Any) -> bool:
    return str(value or "").strip().lower() not in _MUTABLE_REVISIONS


def _safe_git(path: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _license_from_root(path: Path) -> str:
    try:
        candidates = sorted(
            child for child in path.iterdir()
            if child.is_file() and child.name.lower().startswith(("license", "copying"))
        )
    except OSError:
        return ""
    if not candidates:
        return ""
    try:
        head = candidates[0].read_text(encoding="utf-8", errors="ignore")[:4096]
    except OSError:
        return f"FILE:{candidates[0].name}"
    match = re.search(r"SPDX-License-Identifier:\s*([A-Za-z0-9_.+\-]+)", head)
    return match.group(1) if match else f"FILE:{candidates[0].name}"


def _is_skippable(path: Path) -> bool:
    if set(path.parts) & _SKIP_DIR_PARTS:
        return True
    return any(part.endswith(".egg-info") for part in path.parts)


def _is_heldout_concept(path: Any) -> bool:
    try:
        return bool(_HELDOUT_CONCEPT_RE.search(str(path)))
    except Exception:  # noqa: BLE001
        return False


def _is_heldout_task_dir(dir_name: str) -> bool:
    try:
        from kore.data.decontam import _family_of, heldout_families, heldout_task_ids

        return dir_name in heldout_task_ids() or _family_of(dir_name) in heldout_families()
    except Exception:  # noqa: BLE001
        # Fail closed for the explicit held-out names even in a minimal import env.
        return _is_heldout_concept(dir_name)


def _read_text_info(path: Path, max_chars: int) -> Optional[tuple[str, str, bool]]:
    try:
        raw = path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError, ValueError):
        return None
    raw = raw.strip()
    if not raw:
        return None
    root_hash = _sha256(raw)
    truncated = len(raw) > max_chars
    return raw[:max_chars] if truncated else raw, root_hash, truncated


def _read_text(path: Path, max_chars: int) -> Optional[str]:
    info = _read_text_info(path, max_chars)
    return info[0] if info else None


def _kore_task_root() -> Optional[Path]:
    import kore

    root = Path(kore.__file__).resolve().parent / "tasks"
    return root if root.is_dir() else None


def discover_repo_roots() -> list[Path]:
    """Locate local source containers without reading or mutating them."""
    import kore

    candidates: list[Path] = []
    if os.environ.get("KORE_REPOS_DIR"):
        candidates.append(Path(os.environ["KORE_REPOS_DIR"]))
    package = Path(kore.__file__).resolve()
    candidates.extend([
        Path.cwd() / "repos",
        Path.cwd().parent / "repos",
        package.parents[1] / "repos",
        package.parents[2] / "repos",
    ])
    seen: set[Path] = set()
    out: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved in seen or not resolved.is_dir():
                continue
        except OSError:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _load_source_catalog(
    artifact: Optional[str | os.PathLike | Mapping[str, Any]],
) -> dict:
    if artifact is None:
        artifact = os.environ.get("KORE_SOURCE_METADATA")
    if artifact is None:
        return {
            "schema_version": SOURCE_METADATA_SCHEMA_VERSION,
            "sources": [],
            "datasets": [],
            "holdouts": {},
            "_base_dir": str(Path.cwd()),
        }
    if isinstance(artifact, Mapping):
        obj = dict(artifact)
        base_dir = Path.cwd()
    else:
        path = Path(artifact)
        if not path.is_file():
            raise FileNotFoundError(f"source metadata artifact unavailable: {path}")
        obj = json.loads(path.read_text(encoding="utf-8"))
        base_dir = path.resolve().parent
    validate_source_metadata_artifact(obj)
    obj["_base_dir"] = str(base_dir)
    return obj


def validate_source_metadata_artifact(obj: Mapping[str, Any]) -> None:
    """Validate the source metadata schema used by production corpus builds."""
    if str(obj.get("schema_version")) != SOURCE_METADATA_SCHEMA_VERSION:
        raise ValueError(f"unsupported source metadata schema: {obj.get('schema_version')!r}")
    for index, source in enumerate(obj.get("sources", ())):
        if not isinstance(source, dict):
            raise ValueError(f"sources[{index}] must be an object")
        for key in ("local_path", "repository_url", "commit", "license", "lineage_id"):
            if not str(source.get(key) or "").strip():
                raise ValueError(f"sources[{index}] missing {key}")
        if source.get("verified") is not True:
            raise ValueError(f"sources[{index}] is not verified")
    for index, dataset in enumerate(obj.get("datasets", ())):
        if not isinstance(dataset, dict):
            raise ValueError(f"datasets[{index}] must be an object")
        for key in ("dataset_id", "revision", "license"):
            if not str(dataset.get(key) or "").strip():
                raise ValueError(f"datasets[{index}] missing {key}")
        if not _immutable_revision(dataset.get("revision")):
            raise ValueError(f"datasets[{index}] has mutable revision")
        if dataset.get("verified") is not True:
            raise ValueError(f"datasets[{index}] is not verified")


def _catalog_sources(catalog: Mapping[str, Any]) -> dict[Path, dict]:
    base = Path(str(catalog.get("_base_dir") or Path.cwd()))
    out: dict[Path, dict] = {}
    for source in catalog.get("sources", ()):
        try:
            path = Path(str(source["local_path"]))
            resolved = (base / path).resolve() if not path.is_absolute() else path.resolve()
        except (KeyError, OSError):
            continue
        metadata = dict(source)
        metadata["_resolved_local_path"] = str(resolved)
        out[resolved] = metadata
    return out


def _catalog_source_for(path: Path, sources: Mapping[Path, dict]) -> Optional[dict]:
    """Return the nearest catalog root containing ``path``."""
    resolved = path.resolve()
    matches = [
        (root, metadata)
        for root, metadata in sources.items()
        if root == resolved or root in resolved.parents
    ]
    if not matches:
        return None
    return dict(max(matches, key=lambda item: len(item[0].parts))[1])


def _dataset_metadata(catalog: Mapping[str, Any], dataset_id: str) -> Optional[dict]:
    for dataset in catalog.get("datasets", ()):
        if isinstance(dataset, dict) and dataset.get("dataset_id") == dataset_id:
            return dict(dataset)
    return None


def _lineage_paths(path: Path) -> list[Path]:
    """Split a source container into repository lineages before file derivation."""
    if (path / ".git").exists():
        return [path]
    try:
        children = sorted(
            child for child in path.iterdir()
            if child.is_dir() and not _is_skippable(child)
        )
    except OSError:
        return []
    # ``repos/`` containers and parents with Git children are lineage containers.
    if path.name.lower() in {"repos", "sources", "repositories"} or any(
        (child / ".git").exists() for child in children
    ):
        return children
    return [path]


def _root_from_path(
    path: Path,
    explicit: Optional[Mapping[str, Any]],
    *,
    development_mode: bool,
) -> SourceRoot:
    resolved = path.resolve()
    explicit = dict(explicit or {})
    repository_url = str(explicit.get("repository_url") or _safe_git(resolved, "remote", "get-url", "origin"))
    commit = str(explicit.get("commit") or _safe_git(resolved, "rev-parse", "HEAD"))
    license_id = str(explicit.get("license") or _license_from_root(resolved))
    source_id = str(explicit.get("source_id") or repository_url or resolved)
    lineage_id = str(explicit.get("lineage_id") or f"{source_id}@{commit or 'unversioned'}")
    explicit_verified = explicit.get("verified")
    inferred_verified = bool(repository_url and commit and license_id)
    if explicit_verified is False:
        # An explicit failed verification is authoritative in every mode.
        verified = False
    elif development_mode:
        verified = True
        repository_url = repository_url or f"development-local://{resolved}"
        commit = commit or "development-unpinned"
        license_id = license_id or "DEVELOPMENT-UNKNOWN"
    else:
        verified = explicit_verified is True or inferred_verified
    path_prefix = ""
    explicit_root = explicit.get("_resolved_local_path") or explicit.get("local_path")
    if explicit_root:
        try:
            base = Path(str(explicit_root))
            if not base.is_absolute():
                base = base.resolve()
            path_prefix = resolved.relative_to(base.resolve()).as_posix()
            if path_prefix == ".":
                path_prefix = ""
        except (OSError, ValueError):
            path_prefix = ""
    return SourceRoot(
        path=resolved,
        repository_url=repository_url,
        commit=commit,
        license=license_id,
        source_id=source_id,
        lineage_id=lineage_id,
        verified=verified,
        source_timestamp=explicit.get("source_timestamp") or explicit.get("timestamp"),
        development_mode=development_mode,
        path_prefix=path_prefix,
    )


def _resolve_source_roots(
    roots: Iterable[Any],
    catalog: Mapping[str, Any],
    *,
    development_mode: bool,
) -> tuple[list[SourceRoot], int]:
    catalog_by_path = _catalog_sources(catalog)
    specs: list[SourceRoot] = []
    excluded = 0
    seen: set[Path] = set()
    for value in roots:
        descriptor = dict(value) if isinstance(value, Mapping) else {}
        raw_path = descriptor.get("path") or descriptor.get("local_path") if descriptor else value
        path = Path(raw_path)
        if not path.is_dir():
            continue
        for lineage_path in _lineage_paths(path.resolve()):
            if lineage_path in seen or "_drafts" in lineage_path.parts:
                continue
            seen.add(lineage_path)
            explicit = (
                catalog_by_path.get(lineage_path)
                or descriptor
                or _catalog_source_for(path.resolve(), catalog_by_path)
            )
            if explicit and lineage_path != path.resolve() and lineage_path not in catalog_by_path:
                # A container-level descriptor still yields distinct child
                # lineages. Source-level holdouts can use the shared source_id;
                # lineage-level holdouts remain repository-specific.
                explicit = dict(explicit)
                suffix = lineage_path.relative_to(path.resolve()).as_posix()
                base_lineage = str(explicit.get("lineage_id") or explicit.get("source_id") or path)
                explicit["lineage_id"] = f"{base_lineage}:{suffix}"
            root = _root_from_path(lineage_path, explicit, development_mode=development_mode)
            if not root.verified or (explicit and explicit.get("holdout") is True):
                excluded += 1
                continue
            specs.append(root)
    return sorted(specs, key=lambda item: (item.lineage_id, str(item.path))), excluded


def _metadata_for_file(
    root: SourceRoot,
    path: Path,
    text: str,
    root_hash: str,
    *,
    source: str,
    truncated: bool,
) -> dict:
    try:
        local_rel = path.relative_to(root.path).as_posix()
    except ValueError:
        local_rel = path.as_posix()
    rel = (
        (Path(root.path_prefix) / local_rel).as_posix()
        if root.path_prefix else local_rel
    )
    from kore.data.decontam import _family_of

    return {
        "schema_version": SOURCE_METADATA_SCHEMA_VERSION,
        "repository_url": root.repository_url,
        "commit": root.commit,
        "path": rel,
        "license": root.license,
        "row_id": rel,
        "source_id": root.source_id,
        "lineage_id": root.lineage_id,
        "family": _family_of(rel),
        "source_timestamp": root.source_timestamp,
        "verified": root.verified,
        "development_mode": root.development_mode,
        "root_content_hash": root_hash,
        "content_hash": _sha256(text),
        "truncated": truncated,
        "derivation": ["source_file"],
    }


def _source_is_heldout(root: SourceRoot, policy) -> bool:
    from kore.data.decontam import is_contaminated_record

    return is_contaminated_record({
        "operation": "",
        "source_metadata": {
            "source_id": root.source_id,
            "lineage_id": root.lineage_id,
            "source_timestamp": root.source_timestamp,
        },
    }, policy)


def _collect_documents(
    roots: Iterable[SourceRoot],
    exts: tuple[str, ...],
    source: str,
    max_files: int,
    scan_budget: int,
    content_filter: Optional[Callable[[str], bool]] = None,
    max_chars_per_file: int = 200_000,
    policy=None,
) -> tuple[list[SourceDocument], int]:
    candidates: list[tuple[str, SourceRoot, Path]] = []
    dropped_lineage = 0
    for root in roots:
        if _source_is_heldout(root, policy):
            dropped_lineage += 1
            continue
        for ext in exts:
            for path in root.path.rglob(f"*{ext}"):
                if _is_skippable(path) or not path.is_file():
                    continue
                try:
                    rel = path.relative_to(root.path).as_posix()
                except ValueError:
                    rel = path.as_posix()
                # Family/concept holdout happens before reading/chunk derivation.
                if _is_heldout_concept(rel):
                    dropped_lineage += 1
                    continue
                candidates.append((f"{root.lineage_id}\0{rel}", root, path))
    candidates.sort(key=lambda item: item[0])
    out: list[SourceDocument] = []
    for _, root, path in candidates[:scan_budget]:
        if len(out) >= max_files:
            break
        info = _read_text_info(path, max_chars_per_file)
        if info is None:
            continue
        text, root_hash, truncated = info
        if content_filter is not None and not content_filter(text):
            continue
        metadata = _metadata_for_file(
            root, path, text, root_hash, source=source, truncated=truncated,
        )
        out.append(SourceDocument(path, text, source, metadata))
    return out, dropped_lineage


def _collect_files(
    roots: Iterable[Path],
    exts: tuple[str, ...],
    max_files: int,
    scan_budget: int,
    content_filter: Optional[Callable[[str], bool]] = None,
    max_chars_per_file: int = 200_000,
) -> list[tuple[Path, str]]:
    """Backward-compatible path/text collector used by external callers."""
    specs = [
        _root_from_path(Path(root), None, development_mode=True)
        for root in roots if Path(root).is_dir()
    ]
    docs, _ = _collect_documents(
        specs, exts, "compat", max_files, scan_budget, content_filter,
        max_chars_per_file, policy={"families": (), "task_ids": ()},
    )
    return [(doc.path, doc.text) for doc in docs]


def _normalize_document(doc: SourceDocument) -> SourceDocument:
    if doc.source not in _KORE_AUTHORED_SOURCES:
        return doc
    from kore.data.arch_normalize import normalize_text

    normalized = normalize_text(doc.text)
    if normalized == doc.text:
        return doc
    metadata = dict(doc.metadata)
    metadata["parent_content_hash"] = metadata.get("content_hash")
    metadata["content_hash"] = _sha256(normalized)
    metadata["derivation"] = [*metadata.get("derivation", ()), "arch_normalize"]
    return replace(doc, text=normalized, metadata=metadata)


def _derive_pair(task_id: str, reference: SourceDocument, seed: SourceDocument) -> SourceDocument:
    text = (
        f"# PyTorch reference implementation ({task_id})\n\n{reference.text}\n\n"
        f"# Equivalent Triton kernel for {task_id}\n\n{seed.text}\n"
    )
    metadata = dict(seed.metadata)
    pair_path = (Path(str(seed.metadata.get("path") or task_id)).parent / "pair.py").as_posix()
    metadata.update({
        "path": pair_path,
        "row_id": pair_path,
        "family": reference.metadata.get("family") or seed.metadata.get("family"),
        "content_hash": _sha256(text),
        "root_content_hash": reference.metadata.get("root_content_hash"),
        "parent_content_hashes": [
            reference.metadata.get("content_hash"),
            seed.metadata.get("content_hash"),
        ],
        "parent_paths": [
            reference.metadata.get("path"),
            seed.metadata.get("path"),
        ],
        "derivation": ["source_pair"],
    })
    return SourceDocument(seed.path.parent / "pair.py", text, "pytorch_triton_pairs", metadata)


def chunk_text(text: str, budget_chars: int) -> list[str]:
    """Legacy deterministic character chunker (not used for corpus admission)."""
    if budget_chars <= 0:
        return [text] if text else []
    chunks: list[str] = []
    buffer: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        while len(line) > budget_chars:
            if buffer:
                chunks.append("".join(buffer))
                buffer, size = [], 0
            chunks.append(line[:budget_chars])
            line = line[budget_chars:]
        if buffer and size + len(line) > budget_chars:
            chunks.append("".join(buffer))
            buffer, size = [], 0
        buffer.append(line)
        size += len(line)
    if buffer:
        chunks.append("".join(buffer))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _token_ids(tokenizer: Any, text: str) -> list:
    if hasattr(tokenizer, "encode"):
        try:
            # Match the trainer/tokenizer default, including BOS/EOS wrappers.
            value = tokenizer.encode(text, add_special_tokens=True)
        except TypeError:
            value = tokenizer.encode(text)
    elif callable(tokenizer):
        value = tokenizer(text)
    else:
        raise TypeError("tokenizer must expose encode(text) or be callable")
    if isinstance(value, dict):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError("tokenizer did not return an input-id sequence")
    return list(value)


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(_token_ids(tokenizer, text))


def _largest_token_prefix(text: str, max_tokens: int, tokenizer: Any) -> int:
    low, high, best = 1, len(text), 0
    while low <= high:
        middle = (low + high) // 2
        if count_tokens(tokenizer, text[:middle]) <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best == 0:
        raise ValueError("tokenizer emits more than max_tokens for one character")
    return best


def chunk_text_tokens(text: str, max_tokens: int, tokenizer: Any) -> list[str]:
    """Line-aware chunks measured with ``tokenizer``; every chunk is rechecked."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not text:
        return []
    chunks: list[str] = []
    buffer = ""
    for raw_line in text.splitlines(keepends=True):
        line = raw_line
        while line and count_tokens(tokenizer, line) > max_tokens:
            if buffer.strip():
                chunks.append(buffer.strip())
                buffer = ""
            split_at = _largest_token_prefix(line, max_tokens, tokenizer)
            chunks.append(line[:split_at].strip())
            line = line[split_at:]
        candidate = buffer + line
        if buffer and count_tokens(tokenizer, candidate) > max_tokens:
            chunks.append(buffer.strip())
            buffer = line
        else:
            buffer = candidate
    if buffer.strip():
        chunks.append(buffer.strip())
    clean = [chunk for chunk in chunks if chunk]
    if any(count_tokens(tokenizer, chunk) > max_tokens for chunk in clean):
        raise AssertionError("token chunk exceeded max_seq_length")
    return clean


def _resolve_tokenizer(
    config: Any,
    tokenizer: Any,
    tokenizer_id: Optional[str],
    tokenizer_revision: Optional[str],
    *,
    development_mode: bool,
) -> tuple[Any, dict]:
    if tokenizer is None and development_mode:
        tokenizer = _DevelopmentByteTokenizer()
        tokenizer_id = tokenizer.name_or_path
        tokenizer_revision = tokenizer.revision
    else:
        tokenizer_id = tokenizer_id or getattr(tokenizer, "name_or_path", None) or getattr(
            config, "model_id", None
        )
        tokenizer_revision = (
            tokenizer_revision
            or getattr(tokenizer, "revision", None)
            or os.environ.get("KORE_TOKENIZER_REVISION")
        )
    if not development_mode and not _immutable_revision(tokenizer_revision):
        raise ValueError("production midtrain requires an immutable tokenizer_revision")
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("transformers is required to load the pinned tokenizer") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id or getattr(config, "model_id"),
            revision=tokenizer_revision,
            local_files_only=True,
        )
    # Smoke-check the adapter before source traversal.
    count_tokens(tokenizer, "KORE tokenizer admission check")
    return tokenizer, {
        "id": str(tokenizer_id),
        "revision": str(tokenizer_revision),
        "mode": "development-byte" if development_mode and isinstance(
            tokenizer, _DevelopmentByteTokenizer
        ) else "pinned-tokenizer",
        "admission_unit": "tokens",
    }


def _messages_to_text(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role", "")).strip() or "user"
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _stable_stream_select(
    iterable: Iterable[Any],
    n: int,
    seed: int,
    convert: Callable[[dict, int], Optional[tuple[str, str]]],
    *,
    scan_multiplier: int = 8,
) -> list[tuple[str, str]]:
    """Select lowest seeded content hashes from a pinned deterministic stream."""
    if n <= 0:
        return []
    heap: list[tuple[int, str, tuple[str, str]]] = []
    seen: set[str] = set()
    budget = max(n * scan_multiplier, 256)
    for index, raw in enumerate(iterable):
        if index >= budget:
            break
        converted = convert(dict(raw), index)
        if converted is None:
            continue
        row_id, text = converted
        payload_hash = hashlib.sha256(
            json.dumps([row_id, text], ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        if payload_hash in seen:
            continue
        seen.add(payload_hash)
        score = int(hashlib.sha256(f"{seed}:{payload_hash}".encode()).hexdigest(), 16)
        item = (-score, payload_hash, (row_id, text))
        if len(heap) < n:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    selected = [(-neg_score, payload_hash, value) for neg_score, payload_hash, value in heap]
    selected.sort(key=lambda item: (item[0], item[1]))
    return [value for _, _, value in selected]


def _load_kernelbook_pairs(
    n: int,
    max_chars: int,
    *,
    revision: Optional[str] = None,
    seed: int = 0,
) -> list[tuple[Path, str]]:
    if not _immutable_revision(revision):
        raise ValueError("KernelBook requires a pinned revision")
    from datasets import load_dataset

    dataset = load_dataset(
        "GPUMODE/KernelBook", split="train", streaming=True, revision=revision,
    )

    def convert(example: dict, index: int) -> Optional[tuple[str, str]]:
        python = example.get("python_code") or example.get("pytorch_code")
        triton = example.get("triton_code") or example.get("original_triton_code")
        if not (isinstance(python, str) and isinstance(triton, str)
                and python.strip() and triton.strip()):
            return None
        row_id = str(example.get("id") or example.get("row_id") or index)
        text = (
            f"# PyTorch module\n\n{python.strip()[:max_chars]}\n\n"
            f"# Equivalent Triton kernel\n\n{triton.strip()[:max_chars]}\n"
        )
        return row_id, text

    return [
        (Path(f"kernelbook/{row_id}.py"), text)
        for row_id, text in _stable_stream_select(dataset, n, seed, convert)
    ]


def _load_amd_kernels(
    n: int,
    max_chars: int,
    *,
    revision: Optional[str] = None,
    seed: int = 0,
) -> list[tuple[Path, str]]:
    if not _immutable_revision(revision):
        raise ValueError("kernelbot-data requires a pinned revision")
    from datasets import load_dataset

    dataset = load_dataset(
        "GPUMODE/kernelbot-data",
        "amd_successful_submissions",
        split="train",
        streaming=True,
        revision=revision,
    )

    def convert(example: dict, index: int) -> Optional[tuple[str, str]]:
        if example.get("run_passed") is False:
            return None
        code = example.get("code")
        if isinstance(code, (bytes, bytearray)):
            code = code.decode("utf-8", errors="ignore")
        if not isinstance(code, str) or not code.strip():
            return None
        row_id = str(example.get("id") or example.get("submission_id") or index)
        return row_id, code.strip()[:max_chars]

    return [
        (Path(f"amd_kernels/{row_id}.py"), text)
        for row_id, text in _stable_stream_select(dataset, n, seed, convert)
    ]


def _hf_documents(
    values: Iterable[tuple[Path, str]],
    source: str,
    dataset_id: str,
    dataset_meta: Mapping[str, Any],
) -> list[SourceDocument]:
    out: list[SourceDocument] = []
    revision = str(dataset_meta["revision"])
    for path, text in values:
        row_id = path.stem
        digest = _sha256(text)
        metadata = {
            "schema_version": SOURCE_METADATA_SCHEMA_VERSION,
            "repository_url": dataset_meta.get("repository_url")
            or f"https://huggingface.co/datasets/{dataset_id}",
            "commit": revision,
            "dataset_revision": revision,
            "path": path.as_posix(),
            "license": dataset_meta["license"],
            "row_id": row_id,
            "source_id": dataset_id,
            "lineage_id": f"{dataset_id}@{revision}",
            "family": "",
            "verified": True,
            "root_content_hash": digest,
            "content_hash": digest,
            "derivation": ["dataset_row"],
        }
        out.append(SourceDocument(path, text, source, metadata))
    return out


def _quality_filter_documents(
    collected: list[tuple[str, list[SourceDocument]]],
) -> tuple[list[tuple[str, list[SourceDocument]]], dict]:
    from kore.data.corpus_quality import code_quality_reason, doc_quality_reason

    result: list[tuple[str, list[SourceDocument]]] = []
    all_reasons: dict[str, dict[str, int]] = {}
    for source, documents in collected:
        reason_fn = doc_quality_reason if source == "docs" else code_quality_reason
        kept: list[SourceDocument] = []
        reasons: dict[str, int] = {}
        for document in documents:
            reason = reason_fn(document.text, document.path)
            if reason is None:
                kept.append(document)
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
        result.append((source, kept))
        if reasons:
            all_reasons[source] = reasons
            log.info(
                "midtrain quality filter",
                source=source,
                kept=len(kept),
                dropped=len(documents) - len(kept),
                reasons=reasons,
            )
    return result, all_reasons


def _chunk_metadata(document: SourceDocument, chunk: str, index: int) -> dict:
    metadata = dict(document.metadata)
    source_row_id = str(metadata.get("row_id") or metadata.get("path") or "")
    metadata["source_row_id"] = source_row_id
    metadata["row_id"] = f"{source_row_id}#chunk={index}"
    metadata["parent_content_hash"] = metadata.get("content_hash")
    metadata["content_hash"] = _sha256(chunk)
    metadata["derivation"] = [*metadata.get("derivation", ()), "token_chunk"]
    return metadata


def _origin(metadata: Mapping[str, Any]) -> dict:
    return {
        key: metadata.get(key)
        for key in (
            "repository_url", "commit", "path", "license", "source_row_id",
            "row_id", "root_content_hash", "content_hash", "lineage_id",
        )
    }


def _merge_origins(winner: dict, group: list[dict]) -> dict:
    metadata = dict(winner.get("source_metadata") or {})
    origins: list[dict] = []
    seen: set[str] = set()
    for row in group:
        item = _origin(row.get("source_metadata") or {})
        key = json.dumps(item, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            origins.append(item)
    metadata["origins"] = origins
    winner["source_metadata"] = metadata
    return winner


def _near_dedup_corpus(
    rows: list[dict],
    threshold: float = 0.7,
    num_perm: int = 64,
    bands: int = 16,
) -> list[dict]:
    """Source-aware structural/MinHash dedup preserving weighted channels."""
    del num_perm, bands
    from kore.data.dedup import dedup_near

    short = [row for row in rows if len(row.get("text", "")) < 200]
    candidates = [row for row in rows if len(row.get("text", "")) >= 200]
    for order, row in enumerate(rows):
        row["_dedup_order"] = order
    kept, _ = dedup_near(
        candidates,
        source_key="text",
        fuzzy_threshold=threshold,
        partition_key="source",
        merge=_merge_origins,
    )
    output = sorted(short + kept, key=lambda row: row["_dedup_order"])
    for row in rows:
        row.pop("_dedup_order", None)
    for row in output:
        row.pop("_dedup_order", None)
    return output


def _source_weights() -> dict[str, float]:
    weights = dict(_DEFAULT_SOURCE_WEIGHTS)
    for part in os.environ.get("KORE_MIDTRAIN_WEIGHTS", "").split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            weights[key.strip()] = max(1.0, float(value))
        except ValueError:
            continue
    return weights


def _replicas_for(row: Mapping[str, Any], weight: float, seed: int) -> int:
    whole = max(1, int(weight))
    fraction = weight - int(weight)
    if fraction <= 0:
        return whole
    digest = str((row.get("source_metadata") or {}).get("content_hash") or _sha256(
        str(row.get("text") or "")
    ))
    value = int(hashlib.sha256(f"{seed}:{row.get('source')}:{digest}".encode()).hexdigest(), 16)
    unit = value / float(2**256 - 1)
    return whole + int(unit < fraction)


def _weighted_rows(rows: list[dict], weights: Mapping[str, float], seed: int) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        weight = float(weights.get(str(row.get("source")), 1.0))
        replicas = _replicas_for(row, weight, seed)
        for replica in range(replicas):
            copied = dict(row)
            metadata = dict(copied.get("source_metadata") or {})
            metadata["sampling_weight"] = weight
            metadata["sampling_replica"] = replica
            copied["source_metadata"] = metadata
            output.append(copied)
    return output


def _counts(rows: Iterable[Mapping[str, Any]], known: Iterable[str] = ()) -> dict[str, int]:
    result = {source: 0 for source in known}
    for row in rows:
        source = str(row.get("source") or "")
        result[source] = result.get(source, 0) + 1
    return result


def _doc_relevant(document: SourceDocument) -> bool:
    tail = "/".join(part.lower() for part in document.path.parts[-3:])
    return any(keyword in tail for keyword in _DOC_KEYWORDS) or any(
        keyword in document.text[:2000].lower() for keyword in _DOC_KEYWORDS
    )


def _load_pinned_replay(
    kind: str,
    n: int,
    seed: int,
    catalog: Mapping[str, Any],
) -> list[dict]:
    """Load replay directly from a pinned HF revision with stable hash sampling."""
    from datasets import load_dataset
    from kore.data.general_replay import HF_SOURCES, _formatter_for, _row_chars

    specs = HF_SOURCES[kind]
    specs = specs if isinstance(specs, list) else [specs]
    errors: list[str] = []
    for spec in specs:
        dataset_id = str(spec["path"])
        metadata = _dataset_metadata(catalog, dataset_id)
        if not metadata or metadata.get("verified") is not True:
            errors.append(f"{dataset_id}: no verified pinned metadata")
            continue
        revision = metadata.get("revision")
        if not _immutable_revision(revision):
            errors.append(f"{dataset_id}: mutable revision")
            continue
        try:
            dataset = load_dataset(
                dataset_id,
                spec.get("config"),
                split=spec.get("split", "train"),
                streaming=True,
                revision=revision,
            )
            formatter = _formatter_for(kind, spec)
            max_chars = spec.get("max_row_chars")

            def convert(example: dict, index: int) -> Optional[tuple[str, str]]:
                row = formatter(example, spec, kind)
                if row is None or (max_chars and _row_chars(row) > int(max_chars)):
                    return None
                text = _messages_to_text(row.get("messages", []))
                if not text:
                    return None
                row_id = str(example.get("id") or example.get("row_id") or index)
                return row_id, json.dumps(row["messages"], ensure_ascii=False)

            selected = _stable_stream_select(dataset, n, seed, convert, scan_multiplier=20)
            rows: list[dict] = []
            for row_id, payload in selected:
                messages = json.loads(payload)
                text = _messages_to_text(messages)
                rows.append({
                    "messages": messages,
                    "_source": kind,
                    "_source_metadata": {
                        "schema_version": SOURCE_METADATA_SCHEMA_VERSION,
                        "repository_url": metadata.get("repository_url")
                        or f"https://huggingface.co/datasets/{dataset_id}",
                        "commit": str(revision),
                        "dataset_revision": str(revision),
                        "path": f"{dataset_id}/{spec.get('split', 'train')}",
                        "license": metadata["license"],
                        "row_id": row_id,
                        "source_id": dataset_id,
                        "lineage_id": f"{dataset_id}@{revision}",
                        "verified": True,
                        "root_content_hash": _sha256(text),
                        "content_hash": _sha256(text),
                        "derivation": ["dataset_row", "chat_render"],
                    },
                })
            if rows:
                return rows
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{dataset_id}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"no pinned replay source for {kind}: {'; '.join(errors)}")


def _development_replay_metadata(kind: str, text: str, index: int) -> dict:
    digest = _sha256(text)
    return {
        "schema_version": SOURCE_METADATA_SCHEMA_VERSION,
        "repository_url": "development-bundled://kore/general-replay",
        "commit": "development-bundled",
        "path": f"replay_samples/{kind}.jsonl",
        "license": "DEVELOPMENT-INTERNAL",
        "row_id": f"{kind}:{index}:{digest[-12:]}",
        "source_id": f"bundled-replay:{kind}",
        "lineage_id": f"bundled-replay:{kind}",
        "verified": True,
        "development_mode": True,
        "root_content_hash": digest,
        "content_hash": digest,
        "derivation": ["bundled_row", "chat_render"],
    }


def build_midtrain_corpus(
    out_path,
    config,
    seed: int = 0,
    use_hf: bool = False,
    *,
    source_roots: Optional[list] = None,
    task_root=None,
    max_files_per_source: int = 20000,
    scan_budget: int = 200000,
    max_chars_per_file: int = 200_000,
    development_mode: bool = False,
    source_metadata: Optional[str | os.PathLike | Mapping[str, Any]] = None,
    benchmark_artifact: Optional[str | os.PathLike | Mapping[str, Any]] = None,
    tokenizer: Any = None,
    tokenizer_id: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    replay_loader: Optional[Callable[..., list[dict]]] = None,
    replay_replacement_attempts: int = 5,
) -> dict:
    """Build the provenance-preserving Stage-0 JSONL corpus.

    Production is the default. It requires a full frozen benchmark artifact and
    a pinned tokenizer. ``development_mode=True`` is an explicit, report-visible
    opt-in for offline tests and smoke builds.
    """
    from kore.data.decontam import (
        HoldoutIndex,
        HoldoutPolicy,
        analyze_text_contamination,
        eval_benchmark_references,
        heldout_source_references,
        load_frozen_benchmark_artifact,
    )

    development_mode = bool(development_mode or _truthy(os.environ.get("KORE_MIDTRAIN_DEVELOPMENT")))
    out_path = Path(out_path)
    catalog = _load_source_catalog(source_metadata)
    holdout_config = dict(catalog.get("holdouts") or {})
    default_policy = HoldoutPolicy.default()
    configured = HoldoutPolicy.coerce(holdout_config)
    policy = HoldoutPolicy(
        families=default_policy.families | configured.families,
        task_ids=default_policy.task_ids | configured.task_ids,
        source_ids=configured.source_ids,
        lineage_ids=configured.lineage_ids,
        training_cutoff=configured.training_cutoff,
    )

    benchmark_input = benchmark_artifact or os.environ.get("KORE_DECONTAM_BENCHMARK_ARTIFACT")
    if development_mode and not benchmark_input:
        benchmark_refs = eval_benchmark_references(development_mode=True)
        benchmark_report = {
            "scope": "smoke",
            "mode": "development",
            "artifact_hash": None,
            "revisions": {},
        }
    else:
        if not benchmark_input:
            raise FileNotFoundError(
                "production midtrain requires a full frozen benchmark artifact "
                "(benchmark_artifact or KORE_DECONTAM_BENCHMARK_ARTIFACT)"
            )
        frozen = load_frozen_benchmark_artifact(benchmark_input)  # type: ignore[arg-type]
        benchmark_refs = frozen.references
        benchmark_report = {
            "scope": frozen.scope,
            "mode": frozen.mode,
            "artifact_hash": frozen.artifact_hash,
            "revisions": dict(frozen.revisions),
        }

    tokenizer, tokenizer_report = _resolve_tokenizer(
        config,
        tokenizer,
        tokenizer_id,
        tokenizer_revision,
        development_mode=development_mode,
    )
    max_tokens = int(config.max_seq_length)
    if max_tokens <= 0:
        raise ValueError("config.max_seq_length must be positive")

    scale = 1.0
    try:
        scale = max(0.01, float(os.environ.get("KORE_MIDTRAIN_SCALE", "1.0") or 1.0))
    except (TypeError, ValueError):
        pass

    def cap(env_name: str, default: float) -> int:
        value = os.environ.get(env_name)
        try:
            return max(1, int(float(value))) if value else max(1, int(default))
        except (TypeError, ValueError):
            return max(1, int(default))

    max_files_per_source = cap(
        "KORE_MIDTRAIN_MAX_FILES", max_files_per_source * scale,
    )
    scan_budget = cap("KORE_MIDTRAIN_SCAN_BUDGET", scan_budget * scale)
    fraction = float(getattr(config, "general_replay_frac", 0.15) or 0.0)

    raw_roots = source_roots if source_roots is not None else discover_repo_roots()
    repo_roots, n_unverified = _resolve_source_roots(
        raw_roots, catalog, development_mode=development_mode,
    )
    task_path = Path(task_root) if task_root is not None else _kore_task_root()

    collected: list[tuple[str, list[SourceDocument]]] = []
    dropped_lineage = 0

    # Split each task directory into a lineage and gate it before deriving pairs.
    task_documents: list[SourceDocument] = []
    pair_documents: list[SourceDocument] = []
    if task_path is not None and task_path.is_dir():
        task_catalog = _catalog_source_for(task_path, _catalog_sources(catalog))
        task_parent = _root_from_path(
            task_path, task_catalog, development_mode=development_mode,
        )
        per_task: dict[str, dict[str, SourceDocument]] = {}
        try:
            children = sorted(child for child in task_path.iterdir() if child.is_dir())
        except OSError:
            children = []
        for task_dir in children:
            if task_dir.name == "_drafts" or _is_heldout_task_dir(task_dir.name):
                dropped_lineage += 1
                continue
            task_root_spec = replace(
                task_parent,
                path=task_dir.resolve(),
                source_id=f"{task_parent.source_id}:task:{task_dir.name}",
                lineage_id=f"{task_parent.lineage_id}:task:{task_dir.name}",
                path_prefix=(
                    (Path(task_parent.path_prefix) / task_dir.name).as_posix()
                    if task_parent.path_prefix else task_dir.name
                ),
            )
            if not task_root_spec.verified:
                n_unverified += 1
                continue
            docs, dropped = _collect_documents(
                [task_root_spec],
                (".py",),
                "kore_tasks",
                max_files_per_source,
                scan_budget,
                content_filter=lambda text: len(text) > 20,
                max_chars_per_file=max_chars_per_file,
                policy=policy,
            )
            dropped_lineage += dropped
            normalized = [_normalize_document(doc) for doc in docs]
            task_documents.extend(normalized)
            per_task[task_dir.name] = {doc.path.name: doc for doc in normalized}

        # Root-level infrastructure is a separate lineage, never a task derivative.
        root_spec = replace(
            task_parent,
            source_id=f"{task_parent.source_id}:task-infrastructure",
            lineage_id=f"{task_parent.lineage_id}:task-infrastructure",
        )
        if root_spec.verified:
            try:
                root_files = sorted(path for path in task_path.glob("*.py") if path.is_file())
            except OSError:
                root_files = []
            for path in root_files[:max_files_per_source]:
                info = _read_text_info(path, max_chars_per_file)
                if info is None:
                    continue
                text, root_hash, truncated = info
                metadata = _metadata_for_file(
                    root_spec, path, text, root_hash, source="kore_tasks", truncated=truncated,
                )
                task_documents.append(_normalize_document(
                    SourceDocument(path, text, "kore_tasks", metadata)
                ))

        for task_id, docs in sorted(per_task.items()):
            reference, seed_doc = docs.get("reference.py"), docs.get("seed_triton.py")
            if reference is not None and seed_doc is not None:
                pair_documents.append(_derive_pair(task_id, reference, seed_doc))
    collected.append(("kore_tasks", task_documents))
    collected.append(("pytorch_triton_pairs", pair_documents))

    # Pinned external datasets. No revision means no network call.
    kernelbook_docs: list[SourceDocument] = []
    amd_docs: list[SourceDocument] = []
    if use_hf:
        kernelbook_meta = _dataset_metadata(catalog, "GPUMODE/KernelBook")
        amd_meta = _dataset_metadata(catalog, "GPUMODE/kernelbot-data")
        if not kernelbook_meta or not amd_meta:
            raise ValueError("use_hf requires verified pinned metadata for KernelBook and kernelbot-data")
        kb_max = cap("KORE_MIDTRAIN_KERNELBOOK_MAX", max_files_per_source)
        amd_max = cap("KORE_MIDTRAIN_AMD_MAX", max(max_files_per_source, 60000 * scale))
        kernelbook_docs = _hf_documents(
            _load_kernelbook_pairs(
                kb_max,
                max_chars_per_file,
                revision=kernelbook_meta["revision"],
                seed=seed + 101,
            ),
            "kernelbook",
            "GPUMODE/KernelBook",
            kernelbook_meta,
        )
        amd_docs = _hf_documents(
            _load_amd_kernels(
                amd_max,
                max_chars_per_file,
                revision=amd_meta["revision"],
                seed=seed + 102,
            ),
            "amd_kernels",
            "GPUMODE/kernelbot-data",
            amd_meta,
        )
    collected.append(("kernelbook", kernelbook_docs))
    collected.append(("amd_kernels", amd_docs))

    for source, extensions, content_filter, scan_multiplier in (
        ("triton", (".py",), lambda text: any(marker in text for marker in _TRITON_MARKERS), 1),
        ("rocm_hip", _HIP_EXTS, None, 1),
        ("amd_asm", _ASM_EXTS, None, 1),
        ("docs", _DOC_EXTS, None, 3),
    ):
        docs, dropped = _collect_documents(
            repo_roots,
            extensions,
            source,
            max_files_per_source,
            scan_budget * scan_multiplier,
            content_filter=content_filter,
            max_chars_per_file=max_chars_per_file,
            policy=policy,
        )
        dropped_lineage += dropped
        if source == "docs":
            docs = [doc for doc in docs if _doc_relevant(doc)][:max_files_per_source]
        collected.append((source, docs))

    if os.environ.get("KORE_MIDTRAIN_QUALITY", "1") != "0":
        collected, quality_reasons = _quality_filter_documents(collected)
    else:
        if not development_mode:
            raise ValueError("production midtrain may not disable source quality filtering")
        quality_reasons = {}

    source_names = [source for source, _ in collected]
    rows: list[dict] = []
    exact: dict[tuple[str, str], dict] = {}
    n_dropped_exact = 0
    for source, documents in collected:
        for document in documents:
            for chunk_index, chunk in enumerate(
                chunk_text_tokens(document.text, max_tokens, tokenizer)
            ):
                metadata = _chunk_metadata(document, chunk, chunk_index)
                row = {"text": chunk, "source": source, "source_metadata": metadata}
                key = (source, _norm_hash(chunk))
                if key in exact:
                    n_dropped_exact += 1
                    exact[key] = _merge_origins(exact[key], [exact[key], row])
                    continue
                exact[key] = row
                rows.append(row)

    if not rows and not development_mode:
        raise RuntimeError(
            "production midtrain found no verified, quality-approved source documents"
        )

    n_near = 0
    if os.environ.get("KORE_MIDTRAIN_NEAR_DEDUP", "1") != "0" and len(rows) > 1:
        pair_sources = {"pytorch_triton_pairs", "kernelbook"}
        exempt = [row for row in rows if row["source"] in pair_sources]
        candidates = [row for row in rows if row["source"] not in pair_sources]
        kept = _near_dedup_corpus(candidates, threshold=0.7)
        n_near = len(candidates) - len(kept)
        # ``_merge_origins`` may return a copied representative, so identity-based
        # filtering would erase every deduped source channel. Source grouping is
        # deterministic; concatenate the exempt pair channels with the kept raw
        # channels and let final source counts reflect the actual representatives.
        rows = exempt + kept
    elif os.environ.get("KORE_MIDTRAIN_NEAR_DEDUP", "1") == "0" and not development_mode:
        raise ValueError("production midtrain may not disable near dedup")

    references = [*heldout_source_references(), *benchmark_refs]
    holdout_index = HoldoutIndex(references, n=8, policy=policy)
    clean_rows: list[dict] = []
    decontam_evidence: list[dict] = []
    for index, row in enumerate(rows):
        metadata = row["source_metadata"]
        match = analyze_text_contamination(
            row["text"],
            holdout_index,
            metadata=metadata,
            family=str(metadata.get("family") or ""),
            containment_threshold=0.78,
        )
        if match is None:
            clean_rows.append(row)
        else:
            item = match.to_dict()
            item.update({"row_index": index, "source": row["source"]})
            decontam_evidence.append(item)
    rows = clean_rows

    weights = _source_weights()
    before_weight = len(rows)
    if os.environ.get("KORE_MIDTRAIN_WEIGHTING", "1") != "0":
        rows = _weighted_rows(rows, weights, seed + 7)
    elif not development_mode:
        raise ValueError("production midtrain may not disable deterministic weighting")
    n_weighted = len(rows) - before_weight
    n_kernel = len(rows)

    # Replay target is based on the weighted kernel total. Replacement requests
    # are explicit and bounded when dedup/decontam rejects the first response.
    n_general_target = (
        round(fraction / (1.0 - fraction) * n_kernel)
        if 0.0 < fraction < 1.0 and n_kernel > 0 else 0
    )
    if n_general_target and not development_mode and not use_hf:
        raise ValueError(
            "production general replay requires use_hf=True with pinned dataset metadata"
        )
    seen_replay = {_norm_hash(str(row["text"])) for row in rows}
    replay_requests = 0
    replay_replacements = 0
    n_general = 0
    kinds = list(REPLAY_KINDS)
    base = n_general_target // len(kinds) if kinds else 0
    remainder = n_general_target - base * len(kinds)
    targets = {kind: base + int(index < remainder) for index, kind in enumerate(kinds)}
    load_replay = replay_loader or load_general_replay

    for kind_index, kind in enumerate(kinds):
        target = targets[kind]
        accepted = 0
        for attempt in range(max(1, int(replay_replacement_attempts))):
            remaining = target - accepted
            if remaining <= 0:
                break
            request_n = remaining if attempt == 0 else max(remaining * 2, target)
            replay_requests += 1
            replay_replacements += int(attempt > 0)
            request_seed = seed + 1 + kind_index + attempt * 1009
            if use_hf:
                replay = _load_pinned_replay(kind, request_n, request_seed, catalog)
            else:
                replay = load_replay(kind, request_n, seed=request_seed, use_hf=False)
            for row_index, replay_row in enumerate(replay):
                text = _messages_to_text(replay_row.get("messages", []))
                if not text:
                    continue
                base_metadata = (
                    replay_row.get("_source_metadata")
                    or replay_row.get("source_metadata")
                    or _development_replay_metadata(kind, text, row_index)
                )
                document = SourceDocument(
                    Path(str(base_metadata.get("path") or f"replay/{kind}")),
                    text,
                    "general_replay",
                    dict(base_metadata),
                )
                for chunk_index, chunk in enumerate(
                    chunk_text_tokens(text, max_tokens, tokenizer)
                ):
                    if accepted >= target:
                        break
                    normalized_hash = _norm_hash(chunk)
                    if normalized_hash in seen_replay:
                        continue
                    metadata = _chunk_metadata(document, chunk, chunk_index)
                    match = analyze_text_contamination(
                        chunk,
                        holdout_index,
                        metadata=metadata,
                        containment_threshold=0.78,
                    )
                    if match is not None:
                        item = match.to_dict()
                        item.update({"source": "general_replay", "replay_kind": kind})
                        decontam_evidence.append(item)
                        continue
                    seen_replay.add(normalized_hash)
                    rows.append({
                        "text": chunk,
                        "source": "general_replay",
                        "source_metadata": metadata,
                    })
                    accepted += 1
                    n_general += 1

    counts = _counts(rows, [*source_names, "general_replay"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    total = len(rows)
    reason_counts: dict[str, int] = {}
    for item in decontam_evidence:
        reason = str(item["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    report = {
        "out_path": str(out_path),
        "build_mode": "development" if development_mode else "production",
        "source_metadata_schema_version": SOURCE_METADATA_SCHEMA_VERSION,
        "total": total,
        "counts": counts,
        "n_dropped_dup": n_dropped_exact,
        "n_dropped_near_dup": n_near,
        "n_dropped_decontam": len(decontam_evidence),
        "decontam_reasons": dict(sorted(reason_counts.items())),
        "decontam_evidence": decontam_evidence,
        "n_dropped_source_lineage": dropped_lineage,
        "n_excluded_unverified_sources": n_unverified,
        "n_weighted_added": n_weighted,
        "general_frac": round((n_general / total) if total else 0.0, 4),
        "general_target": n_general_target,
        "general_underfill": max(0, n_general_target - n_general),
        "replay_requests": replay_requests,
        "replay_replacement_requests": replay_replacements,
        "max_seq_length": max_tokens,
        "budget_chars": None,
        "tokenizer": tokenizer_report,
        "benchmark_artifact": benchmark_report,
        "repo_roots": [str(root.path) for root in repo_roots],
        "source_lineages": [root.lineage_id for root in repo_roots],
        "corpus_scale": scale,
        "max_files_per_source": max_files_per_source,
        "quality_drop_reasons": quality_reasons,
    }
    log.info(
        "midtrain corpus built",
        out=str(out_path),
        mode=report["build_mode"],
        total=total,
        general_frac=report["general_frac"],
        dropped_dup=n_dropped_exact,
        dropped_near_dup=n_near,
        dropped_decontam=len(decontam_evidence),
        **{f"n_{key}": value for key, value in counts.items()},
    )
    return report


__all__ = [
    "SOURCE_METADATA_SCHEMA_VERSION",
    "CHARS_PER_TOKEN",
    "SourceRoot",
    "SourceDocument",
    "discover_repo_roots",
    "chunk_text",
    "chunk_text_tokens",
    "count_tokens",
    "validate_source_metadata_artifact",
    "build_midtrain_corpus",
]
