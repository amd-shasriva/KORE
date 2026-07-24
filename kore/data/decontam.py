"""Leakage-safe, provenance-aware dataset decontamination.

The old gate compared every candidate with one union of held-out n-grams and
divided by the candidate document length. That had two unsafe failure modes:

* common Triton scaffolding (``import triton``, ``@triton.jit``, ``tl.load``)
  could reject an unrelated training kernel; and
* a held-out kernel pasted into a long document could evade the gate because
  unrelated candidate text diluted the overlap.

This module indexes held-out documents as separate lineage clusters and applies
ordered, auditable checks: family/source/time policy, declared ancestry, exact
SHA-256, normalized AST, semantic graph, MinHash, and directional containment.
The containment denominator is always the held-out reference. Generic Triton
boilerplate is removed from fuzzy evidence, but exact and declared descendants
of held-out roots remain unconditional drops.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from kore.data.dedup import (
    content_hash,
    graph_fingerprint,
    minhash_signature,
    normalized_ast_fingerprint,
    semantic_graph_features,
)


FROZEN_BENCHMARK_ARTIFACT_TYPE = "kore.frozen-benchmark-texts"
FROZEN_BENCHMARK_SCHEMA_VERSION = "1.0"
DEFAULT_BENCHMARKS = (
    "mmlu",
    "humaneval",
    "livecodebench",
    "ifeval",
    "bfcl",
    "mtbench",
)

_IMMUTABLE_REVISION_DENYLIST = frozenset({"", "main", "master", "head", "latest"})
_GENERIC_TRITON_IDENTIFIERS = frozenset({
    # syntax/import scaffolding
    "as", "def", "from", "import", "return", "triton", "language", "jit", "autotune",
    "heuristics", "tl", "constexpr",
    # ubiquitous kernel coordinates and primitive calls
    "program_id", "arange", "load", "store", "zeros", "full", "where", "multiple_of",
    "max_contiguous", "float16", "bfloat16", "float32", "int32", "int64",
    # generic parameter stems; held-out concepts need evidence beyond these
    "x", "y", "z", "out", "input", "output", "ptr", "x_ptr", "y_ptr", "out_ptr",
    "n", "m", "k", "pid", "offs", "offsets", "mask", "block", "block_size",
    "block_m", "block_n", "block_k", "stride", "num_warps", "num_stages",
})
_PYTHON_KEYWORDS = frozenset({
    "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
    "del", "elif", "else", "except", "false", "finally", "for", "from", "global",
    "if", "import", "in", "is", "lambda", "none", "nonlocal", "not", "or", "pass",
    "raise", "return", "true", "try", "while", "with", "yield",
})
_MIN_GRAPH_FEATURES = 12
_MIN_SIGNAL_SHINGLES = 3
_DEFAULT_CONTAINMENT_THRESHOLD = 0.78
_DEFAULT_MINHASH_THRESHOLD = 0.84


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def heldout_task_ids() -> frozenset[str]:
    from kore.tasks.registry import heldout_tasks

    try:
        return frozenset(task.task_id for task in heldout_tasks())
    except Exception:  # noqa: BLE001 - registry unavailable in minimal environments
        return frozenset()


@lru_cache(maxsize=1)
def heldout_families() -> frozenset[str]:
    from kore.tasks.registry import HELDOUT_FAMILIES

    return frozenset(HELDOUT_FAMILIES)


def _family_of(op_or_task: str) -> str:
    """Infer the operator family from an operation/task/path string."""
    op = (op_or_task or "").lower()
    if "mla" in op or "latent_attn" in op or "latent_attention" in op:
        return "mla"
    if "paged" in op:
        return "paged_attention"
    if "attn" in op or "attention" in op:
        return "attention"
    if "topk" in op:
        return "moe_router"
    if "moe" in op:
        return "moe"
    if "rmsnorm" in op:
        return "rmsnorm"
    if "layernorm" in op:
        return "layernorm"
    if "gemm" in op or "matmul" in op:
        return "gemm"
    if "quant" in op:
        return "quant"
    if "rope" in op:
        return "rope"
    if "softmax" in op:
        return "softmax"
    if "gelu" in op or "silu" in op or "relu" in op:
        return "activation"
    return op or "other"


def _record_dict(rec: Any) -> dict:
    if isinstance(rec, dict):
        return rec
    if hasattr(rec, "to_dict"):
        try:
            value = rec.to_dict()
            return value if isinstance(value, dict) else {}
        except Exception:  # noqa: BLE001
            return {}
    return getattr(rec, "__dict__", {}) or {}


def _record_metadata(rec: Any) -> dict:
    row = _record_dict(rec)
    for key in ("source_metadata", "_source_metadata", "_provenance", "provenance"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return {}


def record_family(rec: Any) -> str:
    row = _record_dict(rec)
    meta = _record_metadata(rec)
    value = (
        row.get("operation")
        or row.get("task_id")
        or meta.get("family")
        or meta.get("operator_family")
        or meta.get("path")
        or ""
    )
    return _family_of(str(value))


@dataclass(frozen=True)
class HoldoutPolicy:
    """Metadata gates applied before content similarity.

    ``training_cutoff`` is an ISO-8601 timestamp. Sources newer than that cutoff
    are held out. Entire source IDs or lineage IDs can also be reserved.
    """

    families: frozenset[str] = field(default_factory=frozenset)
    task_ids: frozenset[str] = field(default_factory=frozenset)
    source_ids: frozenset[str] = field(default_factory=frozenset)
    lineage_ids: frozenset[str] = field(default_factory=frozenset)
    training_cutoff: Optional[str] = None

    @classmethod
    def default(cls) -> "HoldoutPolicy":
        return cls(families=heldout_families(), task_ids=heldout_task_ids())

    @classmethod
    def coerce(cls, value: Optional["HoldoutPolicy | Mapping[str, Any]"]) -> "HoldoutPolicy":
        if value is None:
            return cls.default()
        if isinstance(value, cls):
            return value
        return cls(
            families=frozenset(value.get("families", value.get("heldout_families", ()))),
            task_ids=frozenset(value.get("task_ids", value.get("heldout_task_ids", ()))),
            source_ids=frozenset(value.get("source_ids", value.get("heldout_sources", ()))),
            lineage_ids=frozenset(value.get("lineage_ids", value.get("heldout_lineages", ()))),
            training_cutoff=value.get("training_cutoff") or value.get("time_cutoff"),
        )


def is_contaminated_record(
    rec: Any,
    policy: Optional[HoldoutPolicy | Mapping[str, Any]] = None,
) -> bool:
    """True when record metadata belongs to a held-out partition."""
    row = _record_dict(rec)
    meta = _record_metadata(rec)
    gate = HoldoutPolicy.coerce(policy)
    task_id = str(row.get("task_id") or meta.get("task_id") or "")
    if task_id and task_id in gate.task_ids:
        return True
    if record_family(rec) in gate.families:
        return True
    source_id = str(meta.get("source_id") or meta.get("dataset") or "")
    lineage_id = str(meta.get("lineage_id") or meta.get("source_lineage") or "")
    if source_id and source_id in gate.source_ids:
        return True
    if lineage_id and lineage_id in gate.lineage_ids:
        return True
    return _after_cutoff(meta, gate.training_cutoff)


def decontaminate_records(
    records: Iterable[Any],
    policy: Optional[HoldoutPolicy | Mapping[str, Any]] = None,
) -> tuple[list, dict]:
    clean: list = []
    reasons: Counter[str] = Counter()
    for record in records:
        if is_contaminated_record(record, policy):
            reasons["metadata_holdout"] += 1
        else:
            clean.append(record)
    return clean, {
        "n_dropped_heldout": sum(reasons.values()),
        "n_kept": len(clean),
        "drop_reasons": dict(reasons),
    }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]", text or "")


def ngram_set(text: str, n: int = 8) -> set[str]:
    toks = _tokens(text)
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _is_generic_triton_shingle(shingle: str) -> bool:
    identifiers = {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", shingle)
    }
    if not identifiers:
        return True
    meaningful = identifiers - _GENERIC_TRITON_IDENTIFIERS - _PYTHON_KEYWORDS
    # Generated local names (tmp_0, v17, arg3) do not make boilerplate semantic.
    meaningful = {
        token for token in meaningful
        if not re.fullmatch(r"(?:v|tmp|arg|var|off|idx|i|j)\d*", token)
    }
    return not meaningful


def signal_ngram_set(text: str, n: int = 8) -> set[str]:
    """N-grams eligible as fuzzy leakage evidence.

    Exact/AST/graph checks still see the full document. Only generic Triton
    scaffolding is suppressed from fuzzy containment evidence.
    """
    return {gram for gram in ngram_set(text, n) if not _is_generic_triton_shingle(gram)}


@dataclass(frozen=True)
class ReferenceDocument:
    reference_id: str
    text: str
    family: str = ""
    source_id: str = ""
    lineage_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        return content_hash(self.text)


@dataclass(frozen=True)
class FrozenBenchmarkArtifact:
    references: tuple[ReferenceDocument, ...]
    revisions: Mapping[str, str]
    artifact_hash: str
    scope: str = "full"
    mode: str = "production"


@dataclass(frozen=True)
class ContaminationMatch:
    reason: str
    reference_id: Optional[str] = None
    score: float = 1.0
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "reason": self.reason,
            "reference_id": self.reference_id,
            "score": round(float(self.score), 6),
            "evidence": dict(self.evidence),
        }


def _coerce_reference(
    value: str | Mapping[str, Any] | ReferenceDocument,
    index: int,
    prefix: str,
) -> ReferenceDocument:
    if isinstance(value, ReferenceDocument):
        return value
    if isinstance(value, str):
        return ReferenceDocument(
            reference_id=f"{prefix}:{index}",
            text=value,
            source_id=prefix,
            lineage_id=f"{prefix}:{index}",
        )
    text = str(value.get("text") or value.get("content") or "")
    reference_id = str(value.get("reference_id") or value.get("row_id") or f"{prefix}:{index}")
    family = str(value.get("family") or value.get("operator_family") or "")
    source_id = str(value.get("source_id") or value.get("dataset") or prefix)
    lineage_id = str(value.get("lineage_id") or value.get("source_lineage") or reference_id)
    return ReferenceDocument(reference_id, text, family, source_id, lineage_id, dict(value))


class HoldoutIndex(set):
    """Set-compatible n-gram index retaining per-reference lineage clusters."""

    def __init__(
        self,
        references: Iterable[ReferenceDocument],
        n: int = 8,
        policy: Optional[HoldoutPolicy | Mapping[str, Any]] = None,
    ) -> None:
        self.n = int(n)
        self.references = tuple(ref for ref in references if ref.text)
        self.policy = HoldoutPolicy.coerce(policy)
        self.reference_ngrams: dict[str, set[str]] = {}
        self.signal_ngrams: dict[str, set[str]] = {}
        self.reference_by_id: dict[str, ReferenceDocument] = {}
        self.signal_postings: dict[str, set[str]] = {}
        self.required_signal_shingles: dict[str, int] = {}
        self.reference_by_hash: dict[str, ReferenceDocument] = {}
        self.reference_by_ast: dict[str, ReferenceDocument] = {}
        self.reference_by_graph: dict[str, ReferenceDocument] = {}
        self.graph_feature_counts: dict[str, int] = {}
        self.lineage_hashes: dict[str, set[str]] = {}
        union: set[str] = set()
        for ref in self.references:
            grams = ngram_set(ref.text, self.n)
            signal = signal_ngram_set(ref.text, self.n)
            self.reference_by_id[ref.reference_id] = ref
            self.reference_ngrams[ref.reference_id] = grams
            self.signal_ngrams[ref.reference_id] = signal
            self.required_signal_shingles[ref.reference_id] = min(
                max(_MIN_SIGNAL_SHINGLES, math.ceil(len(signal) * 0.25)),
                8,
                len(signal),
            ) if signal else 0
            for gram in signal:
                self.signal_postings.setdefault(gram, set()).add(ref.reference_id)
            union.update(grams)
            self.reference_by_hash[ref.hash] = ref
            lineage = ref.lineage_id or ref.reference_id
            self.lineage_hashes.setdefault(lineage, set()).add(ref.hash)
            ast_fp = normalized_ast_fingerprint(ref.text)
            if ast_fp:
                self.reference_by_ast.setdefault(ast_fp, ref)
            graph_fp = graph_fingerprint(ref.text)
            if graph_fp:
                self.reference_by_graph.setdefault(graph_fp, ref)
                self.graph_feature_counts[ref.reference_id] = len(semantic_graph_features(ref.text))
        super().__init__(union)

    @property
    def root_hashes(self) -> frozenset[str]:
        return frozenset(self.reference_by_hash)


def _reference_from_task_file(path: Path, task_id: str) -> Optional[ReferenceDocument]:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError):
        return None
    return ReferenceDocument(
        reference_id=f"kore-task:{task_id}:{path.name}",
        text=text,
        family=_family_of(task_id),
        source_id="kore-heldout-tasks",
        lineage_id=f"kore-task:{task_id}",
        metadata={"path": str(path), "task_id": task_id, "verified": True},
    )


@lru_cache(maxsize=1)
def heldout_source_references() -> tuple[ReferenceDocument, ...]:
    """Verified Python roots from every held-out KORE task lineage."""
    out: list[ReferenceDocument] = []
    try:
        from kore.tasks.registry import TASKS_DIR

        for task_id in sorted(heldout_task_ids()):
            task_dir = Path(TASKS_DIR) / task_id
            if not task_dir.is_dir() or "_drafts" in task_dir.parts:
                continue
            for path in sorted(task_dir.glob("*.py")):
                ref = _reference_from_task_file(path, task_id)
                if ref is not None:
                    out.append(ref)
    except Exception:  # noqa: BLE001 - optional task registry at corpus-build time
        return tuple()
    return tuple(out)


@lru_cache(maxsize=1)
def heldout_source_texts() -> tuple[str, ...]:
    return tuple(ref.text for ref in heldout_source_references())


def build_heldout_ngrams(
    n: int = 8,
    extra_sources: Optional[
        Iterable[str | Mapping[str, Any] | ReferenceDocument]
    ] = None,
    *,
    policy: Optional[HoldoutPolicy | Mapping[str, Any]] = None,
) -> HoldoutIndex:
    """Build a set-compatible, per-lineage held-out index."""
    refs = list(heldout_source_references())
    for index, source in enumerate(extra_sources or ()):
        ref = _coerce_reference(source, index, "extra-holdout")
        if ref.text:
            refs.append(ref)
    return HoldoutIndex(refs, n=n, policy=policy)


def _canonical_artifact_hash(obj: Mapping[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _immutable_revision(value: Any) -> bool:
    revision = str(value or "").strip()
    return revision.lower() not in _IMMUTABLE_REVISION_DENYLIST


def _verified_hash(text: str, expected: Any, label: str) -> str:
    actual = content_hash(text)
    want = str(expected or "")
    if want and not want.startswith("sha256:"):
        want = "sha256:" + want
    if not want:
        raise ValueError(f"{label}: missing content_hash")
    if want != actual:
        raise ValueError(f"{label}: content_hash mismatch ({want} != {actual})")
    return actual


def load_frozen_benchmark_artifact(
    artifact: str | os.PathLike | Mapping[str, Any],
    *,
    require_benchmarks: Iterable[str] = DEFAULT_BENCHMARKS,
) -> FrozenBenchmarkArtifact:
    """Validate and load a frozen *full* benchmark-text artifact.

    No network access occurs. Every benchmark must carry an immutable revision;
    every record must include the text and its verified SHA-256.
    """
    if isinstance(artifact, Mapping):
        obj = dict(artifact)
    else:
        path = Path(artifact)
        if not path.is_file():
            raise FileNotFoundError(f"frozen benchmark artifact unavailable: {path}")
        obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("artifact_type") != FROZEN_BENCHMARK_ARTIFACT_TYPE:
        raise ValueError("not a KORE frozen benchmark-text artifact")
    if str(obj.get("schema_version")) != FROZEN_BENCHMARK_SCHEMA_VERSION:
        raise ValueError(f"unsupported benchmark artifact schema: {obj.get('schema_version')!r}")
    if obj.get("scope") != "full":
        raise ValueError("production decontamination requires scope='full', not smoke/subset")

    raw_benches = obj.get("benchmarks")
    if isinstance(raw_benches, dict):
        benches = [{"name": name, **dict(value)} for name, value in raw_benches.items()]
    elif isinstance(raw_benches, list):
        benches = [dict(value) for value in raw_benches if isinstance(value, dict)]
    else:
        raise ValueError("benchmark artifact must contain a benchmarks mapping/list")

    references: list[ReferenceDocument] = []
    revisions: dict[str, str] = {}
    present: set[str] = set()
    for bench in benches:
        name = str(bench.get("name") or "")
        dataset = str(bench.get("dataset") or bench.get("repository_url") or "")
        revision = str(bench.get("revision") or "")
        split = str(bench.get("split") or "")
        license_id = str(bench.get("license") or "")
        if (not name or not dataset or not split or not license_id
                or not _immutable_revision(revision)):
            raise ValueError(f"benchmark {name or '<unnamed>'}: unpinned source metadata")
        records = bench.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"benchmark {name}: full artifact has no records")
        present.add(name)
        revisions[name] = revision
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"benchmark {name} record {index}: expected object")
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"benchmark {name} record {index}: missing text")
            row_id = str(record.get("row_id") or record.get("id") or "")
            if not row_id:
                raise ValueError(f"benchmark {name} record {index}: missing row_id")
            digest = _verified_hash(text, record.get("content_hash"), f"{name}:{row_id}")
            meta = {
                "repository_url": dataset,
                "revision": revision,
                "split": split,
                "row_id": row_id,
                "content_hash": digest,
                "license": record.get("license") or license_id,
                "verified": True,
                "scope": "full",
            }
            references.append(ReferenceDocument(
                reference_id=f"benchmark:{name}:{row_id}",
                text=text,
                source_id=f"benchmark:{name}",
                lineage_id=f"benchmark:{name}@{revision}",
                metadata=meta,
            ))
    missing = set(require_benchmarks) - present
    if missing:
        raise ValueError(f"full benchmark artifact missing: {sorted(missing)}")
    return FrozenBenchmarkArtifact(
        references=tuple(references),
        revisions=revisions,
        artifact_hash=_canonical_artifact_hash(obj),
    )


def _development_benchmark_references() -> tuple[ReferenceDocument, ...]:
    out: list[ReferenceDocument] = []
    try:
        from kore.eval.retention import DEFAULT_BENCHES as benches
        from kore.eval.retention import load_bench
    except Exception:  # noqa: BLE001
        return tuple()
    fields = ("question", "prompt", "text", "instruction")
    for name in benches:
        try:
            items = load_bench(name)
        except Exception:  # noqa: BLE001
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            text = next(
                (item[field] for field in fields
                 if isinstance(item.get(field), str) and item[field].strip()),
                "",
            )
            if not text:
                continue
            row_id = str(item.get("id") or item.get("task_id") or index)
            out.append(ReferenceDocument(
                reference_id=f"benchmark-smoke:{name}:{row_id}",
                text=text,
                source_id=f"benchmark-smoke:{name}",
                lineage_id=f"benchmark-smoke:{name}",
                metadata={
                    "row_id": row_id,
                    "content_hash": content_hash(text),
                    "scope": "smoke",
                    "verified": False,
                    "development_mode": True,
                },
            ))
    return tuple(out)


def eval_benchmark_references(
    artifact: Optional[str | os.PathLike | Mapping[str, Any]] = None,
    *,
    development_mode: bool = False,
) -> tuple[ReferenceDocument, ...]:
    """Return frozen full benchmark references, or explicit dev smoke refs."""
    artifact = artifact or os.environ.get("KORE_DECONTAM_BENCHMARK_ARTIFACT")
    development_mode = bool(
        development_mode or _truthy(os.environ.get("KORE_DECONTAM_DEVELOPMENT"))
    )
    if artifact:
        return load_frozen_benchmark_artifact(artifact).references
    if development_mode:
        return _development_benchmark_references()
    raise FileNotFoundError(
        "production decontamination requires KORE_DECONTAM_BENCHMARK_ARTIFACT "
        "(scope=full); pass development_mode=True only for labeled smoke builds"
    )


def eval_benchmark_texts(
    artifact: Optional[str | os.PathLike | Mapping[str, Any]] = None,
    *,
    development_mode: bool = False,
) -> tuple[str, ...]:
    """Compatibility view of :func:`eval_benchmark_references`."""
    return tuple(
        ref.text
        for ref in eval_benchmark_references(artifact, development_mode=development_mode)
    )


def _parse_time(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _after_cutoff(metadata: Mapping[str, Any], cutoff: Optional[str]) -> bool:
    if not cutoff:
        return False
    source_time = (
        metadata.get("timestamp")
        or metadata.get("source_timestamp")
        or metadata.get("published_at")
        or metadata.get("created_at")
    )
    source_dt, cutoff_dt = _parse_time(source_time), _parse_time(cutoff)
    return bool(source_dt and cutoff_dt and source_dt > cutoff_dt)


def _ancestor_hashes(metadata: Mapping[str, Any]) -> set[str]:
    values: list[Any] = []
    for key in (
        "root_content_hash",
        "source_root_hash",
        "parent_content_hash",
        "parent_hash",
        "ancestor_content_hashes",
        "ancestor_hashes",
        "parent_content_hashes",
    ):
        value = metadata.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(value)
        elif value:
            values.append(value)
    lineage = metadata.get("lineage")
    if isinstance(lineage, dict):
        for key in ("root_content_hash", "root_hash", "parent_hash", "ancestor_hashes"):
            value = lineage.get(key)
            if isinstance(value, (list, tuple, set)):
                values.extend(value)
            elif value:
                values.append(value)
    out = set()
    for value in values:
        digest = str(value)
        if digest and not digest.startswith("sha256:") and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            digest = "sha256:" + digest.lower()
        if digest:
            out.add(digest)
    return out


def _metadata_match(
    metadata: Mapping[str, Any],
    family: str,
    task_id: str,
    policy: HoldoutPolicy,
) -> Optional[ContaminationMatch]:
    if task_id and task_id in policy.task_ids:
        return ContaminationMatch("heldout_task", evidence={"task_id": task_id})
    if family and family in policy.families:
        return ContaminationMatch("heldout_family", evidence={"family": family})
    source_id = str(metadata.get("source_id") or metadata.get("dataset") or "")
    lineage_id = str(metadata.get("lineage_id") or metadata.get("source_lineage") or "")
    if source_id and source_id in policy.source_ids:
        return ContaminationMatch("heldout_source", evidence={"source_id": source_id})
    if lineage_id and lineage_id in policy.lineage_ids:
        return ContaminationMatch("heldout_source_lineage", evidence={"lineage_id": lineage_id})
    if _after_cutoff(metadata, policy.training_cutoff):
        return ContaminationMatch(
            "time_holdout",
            evidence={
                "source_timestamp": metadata.get("timestamp")
                or metadata.get("source_timestamp")
                or metadata.get("published_at"),
                "training_cutoff": policy.training_cutoff,
            },
        )
    return None


def analyze_text_contamination(
    text: str,
    holdout: HoldoutIndex,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    family: str = "",
    task_id: str = "",
    containment_threshold: float = _DEFAULT_CONTAINMENT_THRESHOLD,
    minhash_threshold: float = _DEFAULT_MINHASH_THRESHOLD,
) -> Optional[ContaminationMatch]:
    """Return the first decisive leakage reason and non-text evidence."""
    metadata = dict(metadata or {})
    policy_match = _metadata_match(metadata, family, task_id, holdout.policy)
    if policy_match is not None:
        return policy_match
    if not text or not holdout.references:
        return None

    ancestors = _ancestor_hashes(metadata)
    lineage_hits = ancestors & set(holdout.root_hashes)
    if lineage_hits:
        digest = sorted(lineage_hits)[0]
        ref = holdout.reference_by_hash[digest]
        return ContaminationMatch(
            "heldout_lineage_descendant",
            ref.reference_id,
            evidence={"ancestor_content_hash": digest, "lineage_id": ref.lineage_id},
        )

    digest = content_hash(text)
    if digest in holdout.reference_by_hash:
        ref = holdout.reference_by_hash[digest]
        return ContaminationMatch(
            "exact_content",
            ref.reference_id,
            evidence={"content_hash": digest, "lineage_id": ref.lineage_id},
        )

    ast_fp = normalized_ast_fingerprint(text)
    if ast_fp and ast_fp in holdout.reference_by_ast:
        ref = holdout.reference_by_ast[ast_fp]
        if len(semantic_graph_features(ref.text)) >= _MIN_GRAPH_FEATURES:
            return ContaminationMatch(
                "normalized_ast",
                ref.reference_id,
                evidence={"ast_fingerprint": ast_fp, "lineage_id": ref.lineage_id},
            )

    graph_fp = graph_fingerprint(text)
    if graph_fp and graph_fp in holdout.reference_by_graph:
        ref = holdout.reference_by_graph[graph_fp]
        n_features = holdout.graph_feature_counts.get(ref.reference_id, 0)
        if n_features >= _MIN_GRAPH_FEATURES:
            return ContaminationMatch(
                "semantic_graph",
                ref.reference_id,
                evidence={
                    "graph_fingerprint": graph_fp,
                    "graph_features": n_features,
                    "lineage_id": ref.lineage_id,
                },
            )

    candidate_signal = signal_ngram_set(text, holdout.n)
    candidate_sig: Optional[tuple[int, ...]] = None
    # Inverted postings avoid an O(corpus_rows × full_benchmark_rows) scan.
    # References sharing no meaningful shingle cannot satisfy fuzzy evidence.
    candidate_hits: Counter[str] = Counter()
    for gram in candidate_signal:
        for reference_id in holdout.signal_postings.get(gram, ()):
            candidate_hits[reference_id] += 1
    candidate_ids = sorted(
        candidate_hits,
        key=lambda reference_id: (
            -candidate_hits[reference_id],
            reference_id,
        ),
    )
    for reference_id in candidate_ids:
        ref = holdout.reference_by_id[reference_id]
        reference_signal = holdout.signal_ngrams.get(ref.reference_id, set())
        if not reference_signal:
            continue
        shared = candidate_signal & reference_signal
        containment = len(shared) / len(reference_signal)
        required = holdout.required_signal_shingles[ref.reference_id]
        if containment >= max(0.0, float(containment_threshold)) and len(shared) >= required:
            return ContaminationMatch(
                "directional_containment",
                ref.reference_id,
                containment,
                {
                    "shared_signal_shingles": len(shared),
                    "reference_signal_shingles": len(reference_signal),
                    "candidate_signal_shingles": len(candidate_signal),
                    "containment_denominator": "reference",
                    "cluster": ref.lineage_id or ref.reference_id,
                },
            )

        # MinHash is a near-duplicate check, not a boilerplate check. Require
        # meaningful shared signal in addition to the estimated Jaccard.
        if len(shared) < required or containment < 0.50:
            continue
        if candidate_sig is None:
            candidate_sig = minhash_signature(text)
        ref_sig = minhash_signature(ref.text)
        similarity = (
            sum(a == b for a, b in zip(candidate_sig, ref_sig)) / len(candidate_sig)
            if candidate_sig and len(candidate_sig) == len(ref_sig)
            else 0.0
        )
        if similarity >= minhash_threshold:
            return ContaminationMatch(
                "minhash_near_duplicate",
                ref.reference_id,
                similarity,
                {
                    "minhash_similarity": round(similarity, 6),
                    "directional_containment": round(containment, 6),
                    "shared_signal_shingles": len(shared),
                    "cluster": ref.lineage_id or ref.reference_id,
                },
            )
    return None


def contaminated_by_text(
    text: str,
    heldout_ngrams: set[str] | HoldoutIndex,
    n: int = 8,
    threshold: float = _DEFAULT_CONTAINMENT_THRESHOLD,
) -> bool:
    """Boolean compatibility wrapper using directional reference containment."""
    if not heldout_ngrams:
        return False
    if isinstance(heldout_ngrams, HoldoutIndex):
        return analyze_text_contamination(
            text,
            heldout_ngrams,
            containment_threshold=max(float(threshold), _DEFAULT_CONTAINMENT_THRESHOLD),
        ) is not None
    grams = ngram_set(text, n)
    if not grams:
        return False
    # A legacy plain set is one reference cluster. The denominator is the
    # held-out set, never the candidate.
    overlap = len(grams & set(heldout_ngrams)) / len(heldout_ngrams)
    return overlap >= float(threshold)


def _row_match(
    row: Mapping[str, Any],
    text: str,
    holdout: set[str] | HoldoutIndex,
    n: int,
    threshold: float,
) -> Optional[ContaminationMatch]:
    if isinstance(holdout, HoldoutIndex):
        meta = _record_metadata(row)
        family = record_family(row)
        task_id = str(row.get("task_id") or meta.get("task_id") or "")
        return analyze_text_contamination(
            text,
            holdout,
            metadata=meta,
            family=family,
            task_id=task_id,
            containment_threshold=max(float(threshold), _DEFAULT_CONTAINMENT_THRESHOLD),
        )
    if contaminated_by_text(text, holdout, n, threshold):
        return ContaminationMatch(
            "directional_containment",
            score=threshold,
            evidence={"legacy_union_index": True, "containment_denominator": "reference"},
        )
    return None


def _stats(clean: list, evidence: list[dict], holdout: set) -> dict:
    reasons = Counter(item["reason"] for item in evidence)
    return {
        "n_dropped_contaminated": len(evidence),
        "n_kept": len(clean),
        "heldout_ngrams": len(holdout),
        "heldout_references": len(getattr(holdout, "references", ())),
        "drop_reasons": dict(sorted(reasons.items())),
        "evidence": evidence,
    }


def decontaminate_chat_rows(
    rows: Iterable[dict],
    n: int = 8,
    threshold: float = _DEFAULT_CONTAINMENT_THRESHOLD,
    heldout_ngrams: Optional[set[str] | HoldoutIndex] = None,
) -> tuple[list[dict], dict]:
    holdout = heldout_ngrams if heldout_ngrams is not None else build_heldout_ngrams(n)
    clean: list[dict] = []
    evidence: list[dict] = []
    for index, row in enumerate(rows):
        text = " ".join(
            str(message.get("content", ""))
            for message in row.get("messages", [])
            if isinstance(message, dict)
        )
        match = _row_match(row, text, holdout, n, threshold)
        if match is None:
            clean.append(row)
            continue
        item = match.to_dict()
        item.update({"row_index": index, "source": row.get("_source") or row.get("source")})
        evidence.append(item)
    return clean, _stats(clean, evidence, holdout)


def decontaminate_corpus(
    rows: Iterable[dict],
    text_key: str = "text",
    n: int = 8,
    threshold: float = _DEFAULT_CONTAINMENT_THRESHOLD,
    extra_sources: Optional[
        Iterable[str | Mapping[str, Any] | ReferenceDocument]
    ] = None,
    *,
    heldout_ngrams: Optional[set[str] | HoldoutIndex] = None,
    policy: Optional[HoldoutPolicy | Mapping[str, Any]] = None,
) -> tuple[list[dict], dict]:
    """Drop contaminated rows and report stable reason/evidence records."""
    if heldout_ngrams is not None:
        holdout = heldout_ngrams
    elif policy is None:
        # Keep monkeypatch/backward compatibility with the historical two-arg
        # builder while the native path still supports an explicit policy.
        holdout = build_heldout_ngrams(n, extra_sources=extra_sources)
    else:
        holdout = build_heldout_ngrams(n, extra_sources=extra_sources, policy=policy)
    clean: list[dict] = []
    evidence: list[dict] = []
    for index, row in enumerate(rows):
        text = str(row.get(text_key, ""))
        match = _row_match(row, text, holdout, n, threshold)
        if match is None:
            clean.append(row)
            continue
        item = match.to_dict()
        item.update({"row_index": index, "source": row.get("source") or row.get("_source")})
        evidence.append(item)
    return clean, _stats(clean, evidence, holdout)


__all__ = [
    "FROZEN_BENCHMARK_ARTIFACT_TYPE",
    "FROZEN_BENCHMARK_SCHEMA_VERSION",
    "DEFAULT_BENCHMARKS",
    "HoldoutPolicy",
    "ReferenceDocument",
    "FrozenBenchmarkArtifact",
    "ContaminationMatch",
    "HoldoutIndex",
    "heldout_task_ids",
    "heldout_families",
    "heldout_source_references",
    "heldout_source_texts",
    "record_family",
    "is_contaminated_record",
    "decontaminate_records",
    "ngram_set",
    "signal_ngram_set",
    "build_heldout_ngrams",
    "load_frozen_benchmark_artifact",
    "eval_benchmark_references",
    "eval_benchmark_texts",
    "analyze_text_contamination",
    "contaminated_by_text",
    "decontaminate_corpus",
    "decontaminate_chat_rows",
]
