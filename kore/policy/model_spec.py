"""Offline model identity and checkpoint compatibility validation.

This module deliberately does not import torch, transformers, huggingface_hub, or
safetensors.  It validates a local Hugging Face checkpoint from JSON files and
the safetensors headers, so callers can reject the wrong model before any GPU
runtime is initialized.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional


UNRESOLVED = "MEASURE"
_PINNED_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


class ModelSpecError(ValueError):
    """Base class for fail-closed model specification errors."""


class FloatingRevisionError(ModelSpecError):
    """Raised when a branch, tag, or unresolved revision is supplied."""


class ArchitectureMismatchError(ModelSpecError):
    """Raised when config.json does not match the expected architecture."""


class CheckpointCompatibilityError(ModelSpecError):
    """Raised when safetensors metadata is missing or shape-incompatible."""


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def canonical_profile_hash(value: Any) -> str:
    """Return a deterministic SHA-256 over a JSON-compatible profile."""

    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_pinned_revision(revision: Optional[str]) -> str:
    """Validate and normalize an immutable Hugging Face commit revision.

    Branches and tags are mutable, including apparently versioned tags.  Only a
    full 40-character SHA-1 or 64-character SHA-256 is accepted.
    """

    candidate = (revision or "").strip()
    if not _PINNED_REVISION_RE.fullmatch(candidate):
        raise FloatingRevisionError(
            "model revision must be a full 40- or 64-hex commit hash; "
            f"got {revision!r}"
        )
    return candidate.lower()


def _required_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArchitectureMismatchError(
            f"config field {key!r} must be a positive integer, got {value!r}"
        )
    return value


_DECODER_CLASSES = {
    "qwen3": "Qwen3DecoderLayer",
    "qwen2": "Qwen2DecoderLayer",
    "llama": "LlamaDecoderLayer",
    "mistral": "MistralDecoderLayer",
}


@dataclass(frozen=True)
class ArchitectureSpec:
    """Architecture fields that determine model/checkpoint compatibility."""

    model_type: str
    architecture: str
    decoder_class: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ArchitectureSpec":
        model_type = config.get("model_type")
        if not isinstance(model_type, str) or not model_type:
            raise ArchitectureMismatchError("config.json has no valid model_type")

        architectures = config.get("architectures")
        if (
            not isinstance(architectures, list)
            or not architectures
            or not isinstance(architectures[0], str)
        ):
            raise ArchitectureMismatchError(
                "config.json must declare a non-empty architectures list"
            )

        decoder_class = _DECODER_CLASSES.get(model_type)
        declared_decoder = config.get("decoder_layer_class")
        if decoder_class is None:
            if not isinstance(declared_decoder, str) or not declared_decoder:
                raise ArchitectureMismatchError(
                    f"no trusted decoder class mapping for model_type {model_type!r}"
                )
            decoder_class = declared_decoder
        elif declared_decoder is not None and declared_decoder != decoder_class:
            raise ArchitectureMismatchError(
                f"decoder_layer_class {declared_decoder!r} is incompatible with "
                f"model_type {model_type!r} (expected {decoder_class!r})"
            )

        hidden_size = _required_int(config, "hidden_size")
        attention_heads = _required_int(config, "num_attention_heads")
        kv_heads = _required_int(config, "num_key_value_heads")
        if attention_heads % kv_heads:
            raise ArchitectureMismatchError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        head_dim_raw = config.get("head_dim")
        if head_dim_raw is None:
            if hidden_size % attention_heads:
                raise ArchitectureMismatchError(
                    "head_dim is absent and hidden_size is not divisible by "
                    "num_attention_heads"
                )
            head_dim = hidden_size // attention_heads
        elif (
            isinstance(head_dim_raw, bool)
            or not isinstance(head_dim_raw, int)
            or head_dim_raw <= 0
        ):
            raise ArchitectureMismatchError(
                f"config field 'head_dim' must be a positive integer, got {head_dim_raw!r}"
            )
        else:
            head_dim = head_dim_raw

        return cls(
            model_type=model_type,
            architecture=architectures[0],
            decoder_class=decoder_class,
            hidden_size=hidden_size,
            intermediate_size=_required_int(config, "intermediate_size"),
            num_hidden_layers=_required_int(config, "num_hidden_layers"),
            num_attention_heads=attention_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            vocab_size=_required_int(config, "vocab_size"),
            max_position_embeddings=_required_int(
                config, "max_position_embeddings"
            ),
        )

    def assert_matches(self, expected: "ArchitectureSpec") -> None:
        mismatches = []
        for field_name in self.__dataclass_fields__:
            actual_value = getattr(self, field_name)
            expected_value = getattr(expected, field_name)
            if actual_value != expected_value:
                mismatches.append(
                    f"{field_name}: expected {expected_value!r}, got {actual_value!r}"
                )
        if mismatches:
            raise ArchitectureMismatchError(
                "model architecture mismatch: " + "; ".join(mismatches)
            )


@dataclass(frozen=True)
class ModelProfile:
    """Expected model identity before local checkpoint inspection."""

    name: str
    model_id: str
    revision: str
    architecture: ArchitectureSpec
    expected_parameter_count: int | str = UNRESOLVED

    def with_revision(self, revision: str) -> "ModelProfile":
        return replace(self, revision=validate_pinned_revision(revision))

    def validate_resolved(self) -> None:
        validate_pinned_revision(self.revision)
        if (
            isinstance(self.expected_parameter_count, bool)
            or not isinstance(self.expected_parameter_count, int)
            or self.expected_parameter_count <= 0
        ):
            raise ModelSpecError(
                "expected_parameter_count must be resolved from an audited "
                "checkpoint/profile before production use"
            )

    @property
    def profile_hash(self) -> str:
        return canonical_profile_hash(
            {"schema_version": 1, "kind": "expected-model-profile", **asdict(self)}
        )


QWEN3_32B_PROFILE = ModelProfile(
    name="qwen3-32b",
    model_id="Qwen/Qwen3-32B",
    # Deliberately fail-closed.  A deployment must supply the immutable Hub commit.
    revision=UNRESOLVED,
    architecture=ArchitectureSpec(
        model_type="qwen3",
        architecture="Qwen3ForCausalLM",
        decoder_class="Qwen3DecoderLayer",
        hidden_size=5120,
        intermediate_size=25600,
        num_hidden_layers=64,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        max_position_embeddings=40960,
    ),
    # Derived from the complete dense Qwen3-32B tensor shapes, and rechecked
    # against local safetensors metadata by ModelSpec.from_local_checkpoint.
    expected_parameter_count=32_762_123_264,
)


@dataclass(frozen=True)
class FileDigest:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ModelFileFingerprints:
    config: FileDigest
    tokenizer: tuple[FileDigest, ...]
    generation: tuple[FileDigest, ...]
    safetensors_index: Optional[FileDigest]
    safetensors_shards: tuple[FileDigest, ...]

    @property
    def manifest_hash(self) -> str:
        return canonical_profile_hash(
            {"schema_version": 1, "kind": "model-file-manifest", **asdict(self)}
        )


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    data_offsets: tuple[int, int]
    parameter_count: int
    storage_bytes: int


@dataclass(frozen=True)
class CheckpointMetadata:
    index_path: Optional[str]
    shard_paths: tuple[str, ...]
    tensors: tuple[TensorMetadata, ...]
    parameter_count: int
    tensor_storage_bytes: int
    index_total_size: Optional[int]

    def tensor_map(self) -> dict[str, TensorMetadata]:
        return {tensor.name: tensor for tensor in self.tensors}


def _sha256_file(path: Path, *, relative_to: Path) -> FileDigest:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return FileDigest(
        path=path.relative_to(relative_to).as_posix(),
        size_bytes=size,
        sha256=digest.hexdigest(),
    )


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelSpecError(f"cannot read valid {description} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelSpecError(f"{description} at {path} must contain a JSON object")
    return value


_SAFETENSORS_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E4M3FN": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2": 8,
    "F8_E5M2FNUZ": 8,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
    "F4": 4,
    "F4_E2M1": 4,
}


def read_safetensors_metadata(path: str | Path) -> tuple[TensorMetadata, ...]:
    """Read and validate one safetensors header without materializing tensors."""

    shard_path = Path(path)
    try:
        file_size = shard_path.stat().st_size
        with shard_path.open("rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) != 8:
                raise CheckpointCompatibilityError(
                    f"{shard_path} is too short for a safetensors header"
                )
            header_len = int.from_bytes(raw_len, "little", signed=False)
            if header_len <= 1 or header_len > 256 * 1024 * 1024:
                raise CheckpointCompatibilityError(
                    f"{shard_path} has invalid safetensors header length {header_len}"
                )
            raw_header = handle.read(header_len)
            if len(raw_header) != header_len:
                raise CheckpointCompatibilityError(
                    f"{shard_path} has a truncated safetensors header"
                )
    except OSError as exc:
        raise CheckpointCompatibilityError(
            f"cannot read safetensors shard {shard_path}: {exc}"
        ) from exc

    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointCompatibilityError(
            f"{shard_path} has invalid safetensors header JSON: {exc}"
        ) from exc
    if not isinstance(header, dict):
        raise CheckpointCompatibilityError(
            f"{shard_path} safetensors header is not an object"
        )

    data_size = file_size - 8 - header_len
    tensors: list[TensorMetadata] = []
    intervals: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise CheckpointCompatibilityError(
                f"{shard_path} contains malformed tensor metadata"
            )
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype, str):
            raise CheckpointCompatibilityError(
                f"tensor {name!r} in {shard_path} has no dtype"
            )
        if (
            not isinstance(shape, list)
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
                for dim in shape
            )
        ):
            raise CheckpointCompatibilityError(
                f"tensor {name!r} in {shard_path} has invalid shape {shape!r}"
            )
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset < 0
                for offset in offsets
            )
            or offsets[1] < offsets[0]
        ):
            raise CheckpointCompatibilityError(
                f"tensor {name!r} in {shard_path} has invalid data_offsets"
            )
        start, end = offsets
        if end > data_size:
            raise CheckpointCompatibilityError(
                f"tensor {name!r} in {shard_path} extends past the shard"
            )
        count = math.prod(shape)
        storage_bytes = end - start
        bits = _SAFETENSORS_DTYPE_BITS.get(dtype)
        if bits is None:
            raise CheckpointCompatibilityError(
                f"tensor {name!r} in {shard_path} has unsupported dtype {dtype!r}"
            )
        expected_storage_bytes = (count * bits + 7) // 8
        if storage_bytes != expected_storage_bytes:
            raise CheckpointCompatibilityError(
                f"tensor {name!r} in {shard_path} stores {storage_bytes} bytes, "
                f"but dtype={dtype} shape={shape} requires {expected_storage_bytes}"
            )
        tensors.append(
            TensorMetadata(
                name=name,
                dtype=dtype,
                shape=tuple(shape),
                shard=shard_path.name,
                data_offsets=(start, end),
                parameter_count=count,
                storage_bytes=storage_bytes,
            )
        )
        intervals.append((start, end, name))

    for (_, previous_end, previous_name), (start, _, name) in zip(
        sorted(intervals), sorted(intervals)[1:]
    ):
        if start < previous_end:
            raise CheckpointCompatibilityError(
                f"overlapping tensors {previous_name!r} and {name!r} in {shard_path}"
            )
    return tuple(sorted(tensors, key=lambda tensor: tensor.name))


def inspect_safetensors_checkpoint(model_path: str | Path) -> CheckpointMetadata:
    """Inventory an indexed or single-shard checkpoint using headers only."""

    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise CheckpointCompatibilityError(
            f"local checkpoint directory does not exist: {root}"
        )

    index_candidates = sorted(root.glob("*.safetensors.index.json"))
    if len(index_candidates) > 1:
        raise CheckpointCompatibilityError(
            f"multiple safetensors index files found under {root}"
        )

    index_path: Optional[Path] = index_candidates[0] if index_candidates else None
    index_total_size: Optional[int] = None
    weight_map: Optional[dict[str, str]] = None
    if index_path is not None:
        index = _read_json(index_path, description="safetensors index")
        raw_weight_map = index.get("weight_map")
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise CheckpointCompatibilityError(
                f"{index_path} has no non-empty weight_map"
            )
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in raw_weight_map.items()):
            raise CheckpointCompatibilityError(
                f"{index_path} contains an invalid weight_map"
            )
        weight_map = dict(raw_weight_map)
        metadata = index.get("metadata", {})
        if metadata is not None and not isinstance(metadata, dict):
            raise CheckpointCompatibilityError(
                f"{index_path} metadata must be an object"
            )
        raw_total = (metadata or {}).get("total_size")
        if raw_total is not None:
            if isinstance(raw_total, bool) or not isinstance(raw_total, int) or raw_total < 0:
                raise CheckpointCompatibilityError(
                    f"{index_path} metadata.total_size must be a non-negative integer"
                )
            index_total_size = raw_total
        shard_names = sorted(set(weight_map.values()))
    else:
        shards = sorted(root.glob("*.safetensors"))
        if len(shards) != 1:
            raise CheckpointCompatibilityError(
                f"expected one safetensors shard or one index under {root}; "
                f"found {len(shards)} unindexed shards"
            )
        shard_names = [shards[0].name]

    all_tensors: dict[str, TensorMetadata] = {}
    for shard_name in shard_names:
        candidate = (root / shard_name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise CheckpointCompatibilityError(
                f"unsafe shard path in safetensors index: {shard_name!r}"
            ) from exc
        if not candidate.is_file():
            raise CheckpointCompatibilityError(
                f"safetensors index references missing shard {shard_name!r}"
            )
        for tensor in read_safetensors_metadata(candidate):
            if tensor.name in all_tensors:
                raise CheckpointCompatibilityError(
                    f"tensor {tensor.name!r} occurs in multiple shards"
                )
            all_tensors[tensor.name] = tensor

    if weight_map is not None:
        actual_names = set(all_tensors)
        indexed_names = set(weight_map)
        if actual_names != indexed_names:
            missing = sorted(indexed_names - actual_names)
            extra = sorted(actual_names - indexed_names)
            raise CheckpointCompatibilityError(
                "safetensors index/header tensor mismatch: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        wrong_shards = [
            name
            for name, shard in weight_map.items()
            if all_tensors[name].shard != shard
        ]
        if wrong_shards:
            raise CheckpointCompatibilityError(
                "safetensors index maps tensors to the wrong shard: "
                + ", ".join(sorted(wrong_shards)[:5])
            )

    tensors = tuple(sorted(all_tensors.values(), key=lambda tensor: tensor.name))
    parameter_count = sum(tensor.parameter_count for tensor in tensors)
    storage_bytes = sum(tensor.storage_bytes for tensor in tensors)
    if index_total_size is not None and index_total_size != storage_bytes:
        raise CheckpointCompatibilityError(
            "safetensors index metadata.total_size does not match tensor headers: "
            f"{index_total_size} != {storage_bytes}"
        )
    return CheckpointMetadata(
        index_path=index_path.name if index_path is not None else None,
        shard_paths=tuple(shard_names),
        tensors=tensors,
        parameter_count=parameter_count,
        tensor_storage_bytes=storage_bytes,
        index_total_size=index_total_size,
    )


_TOKENIZER_FILE_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}


def fingerprint_model_files(
    model_path: str | Path,
    checkpoint: Optional[CheckpointMetadata] = None,
) -> ModelFileFingerprints:
    """Hash all identity-bearing model files using stable relative paths."""

    root = Path(model_path).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ModelSpecError(f"missing required model config: {config_path}")

    tokenizer_paths = sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name in _TOKENIZER_FILE_NAMES
            or path.name.startswith("tokenizer.")
            or path.name.endswith(".tiktoken")
        )
    )
    if not tokenizer_paths:
        raise ModelSpecError(f"no tokenizer files found under {root}")

    generation_paths = sorted(root.glob("generation_config*.json"))
    if not generation_paths:
        raise ModelSpecError(f"no generation_config JSON found under {root}")

    checkpoint = checkpoint or inspect_safetensors_checkpoint(root)
    index_digest = (
        _sha256_file(root / checkpoint.index_path, relative_to=root)
        if checkpoint.index_path is not None
        else None
    )
    shard_digests = tuple(
        _sha256_file(root / shard, relative_to=root)
        for shard in checkpoint.shard_paths
    )
    return ModelFileFingerprints(
        config=_sha256_file(config_path, relative_to=root),
        tokenizer=tuple(
            _sha256_file(path, relative_to=root) for path in tokenizer_paths
        ),
        generation=tuple(
            _sha256_file(path, relative_to=root) for path in generation_paths
        ),
        safetensors_index=index_digest,
        safetensors_shards=shard_digests,
    )


def _expect_shape(
    tensors: Mapping[str, TensorMetadata],
    name: str,
    expected: tuple[int, ...],
    *,
    required: bool = True,
) -> None:
    tensor = tensors.get(name)
    if tensor is None:
        if required:
            raise CheckpointCompatibilityError(
                f"checkpoint is missing required tensor {name!r}"
            )
        return
    if tensor.shape != expected:
        raise CheckpointCompatibilityError(
            f"tensor {name!r} has shape {tensor.shape}, expected {expected}"
        )


def validate_checkpoint_compatibility(
    architecture: ArchitectureSpec,
    checkpoint: CheckpointMetadata,
    config: Mapping[str, Any],
) -> None:
    """Validate Qwen-style layer coverage and key tensor dimensions."""

    tensors = checkpoint.tensor_map()
    if not tensors:
        raise CheckpointCompatibilityError("checkpoint contains no tensors")

    hidden = architecture.hidden_size
    vocab = architecture.vocab_size
    intermediate = architecture.intermediate_size
    q_width = architecture.num_attention_heads * architecture.head_dim
    kv_width = architecture.num_key_value_heads * architecture.head_dim

    _expect_shape(tensors, "model.embed_tokens.weight", (vocab, hidden))
    _expect_shape(tensors, "model.norm.weight", (hidden,))
    _expect_shape(
        tensors,
        "lm_head.weight",
        (vocab, hidden),
        required=not bool(config.get("tie_word_embeddings", False)),
    )

    layer_indices = {
        int(match.group(1))
        for name in tensors
        if (match := _LAYER_RE.match(name)) is not None
    }
    expected_indices = set(range(architecture.num_hidden_layers))
    if layer_indices != expected_indices:
        missing = sorted(expected_indices - layer_indices)
        extra = sorted(layer_indices - expected_indices)
        raise CheckpointCompatibilityError(
            "checkpoint decoder layer coverage mismatch: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    for layer in range(architecture.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        expected_shapes = {
            f"{prefix}.self_attn.q_proj.weight": (q_width, hidden),
            f"{prefix}.self_attn.k_proj.weight": (kv_width, hidden),
            f"{prefix}.self_attn.v_proj.weight": (kv_width, hidden),
            f"{prefix}.self_attn.o_proj.weight": (hidden, q_width),
            f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
            f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
            f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            f"{prefix}.input_layernorm.weight": (hidden,),
            f"{prefix}.post_attention_layernorm.weight": (hidden,),
        }
        for name, shape in expected_shapes.items():
            _expect_shape(tensors, name, shape)
        _expect_shape(
            tensors,
            f"{prefix}.self_attn.q_norm.weight",
            (architecture.head_dim,),
            required=False,
        )
        _expect_shape(
            tensors,
            f"{prefix}.self_attn.k_norm.weight",
            (architecture.head_dim,),
            required=False,
        )


@dataclass(frozen=True)
class ModelSpec:
    """Fully resolved, locally verified model identity."""

    model_id: str
    revision: str
    checkpoint_path: str
    architecture: ArchitectureSpec
    checkpoint: CheckpointMetadata
    files: ModelFileFingerprints

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision", validate_pinned_revision(self.revision))
        if self.checkpoint.parameter_count <= 0:
            raise CheckpointCompatibilityError(
                "resolved ModelSpec checkpoint must contain parameters"
            )

    @classmethod
    def from_local_checkpoint(
        cls,
        model_path: str | Path,
        *,
        revision: Optional[str] = None,
        expected: Optional[ModelProfile | ArchitectureSpec] = None,
        model_id: Optional[str] = None,
    ) -> "ModelSpec":
        """Inspect and validate a local checkpoint with no network/GPU imports."""

        expected_revision: Optional[str] = None
        expected_architecture: Optional[ArchitectureSpec] = None
        expected_parameter_count: Optional[int] = None
        expected_model_id: Optional[str] = None
        if isinstance(expected, ModelProfile):
            expected_revision = expected.revision
            expected_architecture = expected.architecture
            expected_model_id = expected.model_id
            if isinstance(expected.expected_parameter_count, int):
                expected_parameter_count = expected.expected_parameter_count
        elif isinstance(expected, ArchitectureSpec):
            expected_architecture = expected

        resolved_revision = validate_pinned_revision(
            revision if revision is not None else expected_revision
        )
        if (
            revision is not None
            and expected_revision not in (None, UNRESOLVED)
            and validate_pinned_revision(expected_revision) != resolved_revision
        ):
            raise ModelSpecError(
                "explicit revision does not match the expected model profile"
            )

        root = Path(model_path).expanduser().resolve()
        config = _read_json(root / "config.json", description="model config")
        architecture = ArchitectureSpec.from_config(config)
        if expected_architecture is not None:
            architecture.assert_matches(expected_architecture)

        checkpoint = inspect_safetensors_checkpoint(root)
        validate_checkpoint_compatibility(architecture, checkpoint, config)
        if (
            expected_parameter_count is not None
            and checkpoint.parameter_count != expected_parameter_count
        ):
            raise CheckpointCompatibilityError(
                "checkpoint parameter count does not match expected profile: "
                f"{checkpoint.parameter_count} != {expected_parameter_count}"
            )
        files = fingerprint_model_files(root, checkpoint)
        return cls(
            model_id=model_id or expected_model_id or root.name,
            revision=resolved_revision,
            checkpoint_path=str(root),
            architecture=architecture,
            checkpoint=checkpoint,
            files=files,
        )

    @property
    def parameter_count(self) -> int:
        return self.checkpoint.parameter_count

    @property
    def profile_hash(self) -> str:
        """Stable model identity hash; deliberately excludes absolute local path."""

        return canonical_profile_hash(
            {
                "schema_version": 1,
                "kind": "resolved-model-spec",
                "model_id": self.model_id,
                "revision": self.revision,
                "architecture": asdict(self.architecture),
                "parameter_count": self.parameter_count,
                "tensor_storage_bytes": self.checkpoint.tensor_storage_bytes,
                "file_manifest_hash": self.files.manifest_hash,
            }
        )

    @property
    def fingerprint(self) -> str:
        """Compatibility alias for the resolved model profile hash."""

        return self.profile_hash

    def validate_for_load(
        self, model_id: str | Path, *, revision: Optional[str] = None
    ) -> None:
        """Ensure a serving request still targets this verified checkpoint."""

        requested = Path(model_id).expanduser()
        if not requested.exists():
            raise ModelSpecError(
                "a local ModelSpec cannot authorize a remote/model-id load; "
                "load the exact validated checkpoint_path"
            )
        if requested.resolve() != Path(self.checkpoint_path):
            raise ModelSpecError(
                f"serving path {requested.resolve()} differs from validated "
                f"checkpoint {self.checkpoint_path}"
            )
        # Close the preflight-to-load TOCTOU window. This is deliberately
        # completed before serve.load_generate imports torch/vLLM.
        current = ModelSpec.from_local_checkpoint(
            requested,
            revision=self.revision,
            expected=self.architecture,
            model_id=self.model_id,
        )
        if current.profile_hash != self.profile_hash:
            raise ModelSpecError(
                "local checkpoint files changed after ModelSpec validation"
            )
        if revision is not None and validate_pinned_revision(revision) != self.revision:
            raise ModelSpecError(
                "serving revision differs from the validated model revision"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "revision": self.revision,
            "checkpoint_path": self.checkpoint_path,
            "architecture": asdict(self.architecture),
            "parameter_count": self.parameter_count,
            "tensor_storage_bytes": self.checkpoint.tensor_storage_bytes,
            "checkpoint": _jsonable(self.checkpoint),
            "files": _jsonable(self.files),
            "file_manifest_hash": self.files.manifest_hash,
            "profile_hash": self.profile_hash,
        }


def load_model_spec(
    model_path: str | Path,
    *,
    revision: Optional[str] = None,
    expected: Optional[ModelProfile | ArchitectureSpec] = None,
    model_id: Optional[str] = None,
) -> ModelSpec:
    """Functional wrapper around :meth:`ModelSpec.from_local_checkpoint`."""

    return ModelSpec.from_local_checkpoint(
        model_path,
        revision=revision,
        expected=expected,
        model_id=model_id,
    )


__all__ = [
    "UNRESOLVED",
    "ArchitectureMismatchError",
    "ArchitectureSpec",
    "CheckpointCompatibilityError",
    "CheckpointMetadata",
    "FileDigest",
    "FloatingRevisionError",
    "ModelFileFingerprints",
    "ModelProfile",
    "ModelSpec",
    "ModelSpecError",
    "QWEN3_32B_PROFILE",
    "TensorMetadata",
    "canonical_profile_hash",
    "fingerprint_model_files",
    "inspect_safetensors_checkpoint",
    "load_model_spec",
    "read_safetensors_metadata",
    "validate_checkpoint_compatibility",
    "validate_pinned_revision",
]
