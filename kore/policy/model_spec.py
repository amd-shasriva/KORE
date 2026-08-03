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
import os
import re
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional


UNRESOLVED = "MEASURE"
_PINNED_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.(\d+)\.")


class ModelSpecError(ValueError):
    """Base class for fail-closed model specification errors."""


class FloatingRevisionError(ModelSpecError):
    """Raised when a branch, tag, or unresolved revision is supplied."""


class UnpinnedModelError(ModelSpecError):
    """Raised when a production load supplies no immutable revision at all."""


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
    "qwen3_moe": "Qwen3MoeDecoderLayer",
    "qwen2": "Qwen2DecoderLayer",
    "llama": "LlamaDecoderLayer",
    "mistral": "MistralDecoderLayer",
}

# Model types whose decoder block holds a routed expert FFN instead of one dense
# MLP. Their checkpoints carry ``mlp.experts.<i>.*`` + ``mlp.gate`` where a dense
# model carries ``mlp.gate_proj``/``up_proj``/``down_proj``, so shape validation
# has to branch. Membership here is a claim that the expert layout below matches
# that family's modeling code, which is why it is an allowlist and not a
# substring test on the model type.
_MOE_MODEL_TYPES = frozenset({"qwen3_moe"})


@dataclass(frozen=True)
class MoESpec:
    """Sparse-mixture-of-experts fields that change a checkpoint's tensor set.

    ``decoder_sparse_step`` and ``mlp_only_layers`` decide, per layer, whether the
    FFN is a routed expert bank or a plain dense MLP; :meth:`is_sparse_layer`
    mirrors ``Qwen3MoeDecoderLayer.__init__`` exactly so validation expects the
    same tensors the modeling code will look for.
    """

    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    decoder_sparse_step: int
    mlp_only_layers: tuple[int, ...]

    def is_sparse_layer(self, layer_idx: int) -> bool:
        return (
            layer_idx not in self.mlp_only_layers
            and self.num_experts > 0
            and (layer_idx + 1) % self.decoder_sparse_step == 0
        )


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
    # Defaulted and last so every existing dense ArchitectureSpec literal still
    # constructs. Profile hashes DO shift (asdict now emits "moe": null), which
    # is safe because a profile hash is only ever compared to one recomputed by
    # the same code -- validate_for_load's TOCTOU re-check -- never to a value
    # persisted by an older build.
    moe: Optional[MoESpec] = None

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
            moe=cls._moe_from_config(config, model_type),
        )

    @staticmethod
    def _moe_from_config(
        config: Mapping[str, Any], model_type: str
    ) -> Optional[MoESpec]:
        """Read the expert layout, fail-closed, for a known sparse model type.

        A missing or nonsensical expert field is an error rather than a silent
        fall-through to the dense path: validating a 30B MoE against dense
        ``mlp.gate_proj`` shapes would reject every real shard, and validating it
        against nothing at all would accept a truncated expert bank.
        """

        if model_type not in _MOE_MODEL_TYPES:
            return None
        raw_only = config.get("mlp_only_layers", [])
        if not isinstance(raw_only, (list, tuple)) or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in raw_only
        ):
            raise ArchitectureMismatchError(
                f"config field 'mlp_only_layers' must be a list of layer indices, "
                f"got {raw_only!r}"
            )
        return MoESpec(
            num_experts=_required_int(config, "num_experts"),
            num_experts_per_tok=_required_int(config, "num_experts_per_tok"),
            moe_intermediate_size=_required_int(config, "moe_intermediate_size"),
            # The modeling code computes (layer_idx + 1) % decoder_sparse_step,
            # so a zero here is a ZeroDivisionError at model construction.
            decoder_sparse_step=_required_int(config, "decoder_sparse_step"),
            mlp_only_layers=tuple(sorted(raw_only)),
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


#: The production backbone. Sparse MoE: 30.5B total, ~3.3B active per token
#: (128 experts, top-8, every layer routed because decoder_sparse_step=1 and
#: mlp_only_layers is empty). Chosen over a dense 30B-class model because no
#: Qwen at this size ships a Base variant, so the continued-pretraining recipe
#: cannot transfer and the instruct -> SFT -> RL path is the only one available.
QWEN3_CODER_30B_A3B_PROFILE = ModelProfile(
    name="qwen3-coder-30b-a3b-instruct",
    model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    revision="b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
    architecture=ArchitectureSpec(
        model_type="qwen3_moe",
        architecture="Qwen3MoeForCausalLM",
        decoder_class="Qwen3MoeDecoderLayer",
        hidden_size=2048,
        # 6144 is the DENSE ffn width, which this checkpoint never instantiates
        # (mlp_only_layers is empty), but it stays part of the identity because
        # transformers reads it and a change to it would be a different model.
        intermediate_size=6144,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=151936,
        max_position_embeddings=262144,
        moe=MoESpec(
            num_experts=128,
            num_experts_per_tok=8,
            moe_intermediate_size=768,
            decoder_sparse_step=1,
            mlp_only_layers=(),
        ),
    ),
    # Measured from the downloaded snapshot's safetensors headers on the cluster
    # (16 shards, 18,867 tensors, all BF16), and independently reproduced from
    # the config arithmetic; the Hub's own safetensors metadata agrees.
    expected_parameter_count=30_532_122_624,
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


def _contained_checkpoint_path(root: Path, name: str) -> Path:
    """Join an index-declared shard name to ``root`` without resolving symlinks.

    Containment is enforced on the declared name, which is the only part an
    untrusted index controls: an absolute path or any ``..`` component is
    rejected. The link target is deliberately not required to live under the
    checkpoint, because a Hugging Face cache snapshot is a farm of symlinks into
    a sibling ``blobs/`` directory - resolving them would reject every offline
    snapshot the training jobs actually load.
    """

    declared = PurePosixPath(str(name))
    if (
        not name
        or declared.is_absolute()
        or not declared.parts
        or ".." in declared.parts
        or "\\" in str(name)
    ):
        raise CheckpointCompatibilityError(
            f"unsafe shard path in safetensors index: {name!r}"
        )
    return root.joinpath(*declared.parts)


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
        candidate = _contained_checkpoint_path(root, shard_name)
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

    # Experts BEYOND num_experts are never loaded and never routed to, so the
    # per-expert shape checks below (which only look up the experts the config
    # promises) cannot see them. One pass over the names catches that.
    if architecture.moe is not None:
        overflow = sorted(
            {
                int(match.group(1))
                for name in tensors
                if (match := _EXPERT_RE.match(name)) is not None
                and int(match.group(1)) >= architecture.moe.num_experts
            }
        )
        if overflow:
            raise CheckpointCompatibilityError(
                "checkpoint contains expert indices beyond the configured "
                f"num_experts={architecture.moe.num_experts}: {overflow[:8]}"
            )

    moe = architecture.moe
    for layer in range(architecture.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        expected_shapes = {
            f"{prefix}.self_attn.q_proj.weight": (q_width, hidden),
            f"{prefix}.self_attn.k_proj.weight": (kv_width, hidden),
            f"{prefix}.self_attn.v_proj.weight": (kv_width, hidden),
            f"{prefix}.self_attn.o_proj.weight": (hidden, q_width),
            f"{prefix}.input_layernorm.weight": (hidden,),
            f"{prefix}.post_attention_layernorm.weight": (hidden,),
        }
        if moe is not None and moe.is_sparse_layer(layer):
            # Routed FFN: one router plus num_experts independent MLPs at
            # moe_intermediate_size. Every expert is required, because a bank
            # that is short a few experts still loads and still generates
            # plausible text - the router just never selects the missing ones.
            expert_width = moe.moe_intermediate_size
            expected_shapes[f"{prefix}.mlp.gate.weight"] = (moe.num_experts, hidden)
            for expert in range(moe.num_experts):
                expert_prefix = f"{prefix}.mlp.experts.{expert}"
                expected_shapes.update({
                    f"{expert_prefix}.gate_proj.weight": (expert_width, hidden),
                    f"{expert_prefix}.up_proj.weight": (expert_width, hidden),
                    f"{expert_prefix}.down_proj.weight": (hidden, expert_width),
                })
        else:
            expected_shapes.update({
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            })
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


def _expected_fields(
    expected: Optional["ModelProfile | ArchitectureSpec"],
) -> tuple[Optional[str], Optional[ArchitectureSpec], Optional[int], Optional[str]]:
    """Unpack (revision, architecture, parameter count, model id) expectations."""

    if isinstance(expected, ModelProfile):
        parameter_count = (
            expected.expected_parameter_count
            if isinstance(expected.expected_parameter_count, int)
            and not isinstance(expected.expected_parameter_count, bool)
            else None
        )
        return (
            expected.revision,
            expected.architecture,
            parameter_count,
            expected.model_id,
        )
    if isinstance(expected, ArchitectureSpec):
        return None, expected, None, None
    return None, None, None, None


@dataclass(frozen=True)
class CheckpointInspection:
    """Header-only inspection of a local checkpoint: no hashing, no GPU imports.

    This is the cheap tier of local verification. It reads ``config.json`` and
    the safetensors headers, so it still rejects a wrong architecture, a missing
    decoder layer, or a truncated shard before a multi-hour job loads 14B of
    weights - but it reads no tensor bytes, so it deliberately does NOT establish
    content identity. Only :class:`ModelSpec` does that, and only :class:`ModelSpec`
    may back a production claim.
    """

    model_id: str
    revision: Optional[str]
    checkpoint_path: str
    architecture: ArchitectureSpec
    checkpoint: CheckpointMetadata

    @property
    def parameter_count(self) -> int:
        return self.checkpoint.parameter_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "checkpoint-inspection",
            "model_id": self.model_id,
            "revision": self.revision,
            "checkpoint_path": self.checkpoint_path,
            "architecture": asdict(self.architecture),
            "parameter_count": self.parameter_count,
            "tensor_storage_bytes": self.checkpoint.tensor_storage_bytes,
            "shard_paths": list(self.checkpoint.shard_paths),
        }


def inspect_local_checkpoint(
    model_path: str | Path,
    *,
    revision: Optional[str] = None,
    expected: Optional["ModelProfile | ArchitectureSpec"] = None,
    model_id: Optional[str] = None,
) -> CheckpointInspection:
    """Validate architecture and checkpoint shapes without hashing any bytes."""

    (
        _expected_revision,
        expected_architecture,
        expected_parameter_count,
        expected_model_id,
    ) = _expected_fields(expected)
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
    return CheckpointInspection(
        model_id=model_id or expected_model_id or root.name,
        revision=validate_pinned_revision(revision) if revision is not None else None,
        checkpoint_path=str(root),
        architecture=architecture,
        checkpoint=checkpoint,
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

        expected_revision, _, _, _ = _expected_fields(expected)
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

        inspection = inspect_local_checkpoint(
            model_path,
            revision=resolved_revision,
            expected=expected,
            model_id=model_id,
        )
        root = Path(inspection.checkpoint_path)
        files = fingerprint_model_files(root, inspection.checkpoint)
        return cls(
            model_id=inspection.model_id,
            revision=resolved_revision,
            checkpoint_path=inspection.checkpoint_path,
            architecture=inspection.architecture,
            checkpoint=inspection.checkpoint,
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


# --------------------------------------------------------------------------- #
# Model identity for training entrypoints
#
# The training stages load their model with ``from_pretrained(model_id)``, which
# resolves to whatever the Hugging Face cache happens to hold. This layer turns
# that into an explicit, auditable decision:
#
#   * PRODUCTION requires an immutable commit, requires it to be resolvable from
#     a local snapshot (the jobs run with ``HF_HUB_OFFLINE=1``), and verifies the
#     checkpoint's content fingerprint. Anything missing is a hard failure with
#     an actionable message.
#   * DEVELOPMENT reports the same facts and keeps going. A config with no
#     revision loads exactly as it did before this module existed, so a job that
#     is already in flight cannot be broken by resuming into this code.
#
# An explicitly configured revision that is *malformed* (a branch, a tag, a short
# hash) fails in BOTH modes: that is a config bug, not a missing pin.
# --------------------------------------------------------------------------- #
PRODUCTION = "production"
DEVELOPMENT = "development"
_IDENTITY_MODES = (DEVELOPMENT, PRODUCTION)

VERIFY_NONE = "none"
VERIFY_METADATA = "metadata"
VERIFY_FINGERPRINT = "fingerprint"
_VERIFY_LEVELS = (VERIFY_NONE, VERIFY_METADATA, VERIFY_FINGERPRINT)

REVISION_ENV = "KORE_MODEL_REVISION"
REF_REVISION_ENV = "KORE_REF_MODEL_REVISION"
IDENTITY_MODE_ENV = "KORE_MODEL_IDENTITY_MODE"
IDENTITY_VERIFY_ENV = "KORE_MODEL_IDENTITY_VERIFY"
_OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_HF_CACHE_ENV = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")

# Launch-config keys that configure identity instead of being stage dataclass
# fields. The stage builders pop these before ``Config(**payload)`` so a pinned
# config cannot crash the strict dataclass parse.
IDENTITY_CONFIG_KEYS = (
    "model_revision",
    "ref_model_revision",
    "model_identity_mode",
    "model_identity_verify",
)


def _environ(environ: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def hub_offline(environ: Optional[Mapping[str, str]] = None) -> bool:
    """True when the Hub is unreachable by policy, so a pin must resolve locally."""

    env = _environ(environ)
    return any(_truthy(env.get(key)) for key in _OFFLINE_ENV)


def resolve_identity_mode(
    mode: Optional[str] = None, *, environ: Optional[Mapping[str, str]] = None
) -> str:
    """Resolve the identity mode; ``production`` is a one-way opt-in.

    Either the launch config or ``KORE_MODEL_IDENTITY_MODE`` can ask for
    production, and neither can silently downgrade the other.
    """

    env = _environ(environ)
    selected = []
    for value in (mode, env.get(IDENTITY_MODE_ENV)):
        text = str(value or "").strip().lower()
        if not text:
            continue
        if text not in _IDENTITY_MODES:
            raise ModelSpecError(
                "model identity mode must be one of "
                f"{list(_IDENTITY_MODES)}; got {value!r}"
            )
        selected.append(text)
    return PRODUCTION if PRODUCTION in selected else DEVELOPMENT


def resolve_verify_level(
    verify: Optional[str] = None,
    *,
    mode: str = DEVELOPMENT,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve how deeply a local checkpoint is checked.

    Production fingerprints the checkpoint (every identity-bearing file is
    hashed, which is what makes the pre-load re-check meaningful). Development
    defaults to the header-only tier so startup stays cheap.
    """

    env = _environ(environ)
    raw = verify if verify not in (None, "") else env.get(IDENTITY_VERIFY_ENV)
    text = str(raw or "").strip().lower()
    if not text:
        return VERIFY_FINGERPRINT if mode == PRODUCTION else VERIFY_METADATA
    if text not in _VERIFY_LEVELS:
        raise ModelSpecError(
            "model identity verification must be one of "
            f"{list(_VERIFY_LEVELS)}; got {raw!r}"
        )
    return text


def hf_cache_roots(
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[Path, ...]:
    """Candidate Hugging Face hub cache roots, in resolution order."""

    env = _environ(environ)
    candidates: list[Path] = []
    for key in _HF_CACHE_ENV:
        value = str(env.get(key) or "").strip()
        if value:
            candidates.append(Path(value).expanduser())
    home = str(env.get("HF_HOME") or "").strip()
    if home:
        candidates.append(Path(home).expanduser() / "hub")
    xdg = str(env.get("XDG_CACHE_HOME") or "").strip()
    base = Path(xdg).expanduser() if xdg else Path("~/.cache").expanduser()
    candidates.append(base / "huggingface" / "hub")
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            roots.append(candidate)
    return tuple(roots)


def hf_repo_cache_dirname(model_id: str) -> str:
    """Cache directory name for a Hub repo id (``org/name`` -> ``models--org--name``)."""

    return "models--" + str(model_id).strip().strip("/").replace("/", "--")


def resolve_local_snapshot(
    model_id: str,
    revision: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[Path]:
    """Map an immutable Hub commit to its local snapshot directory, offline.

    Returns ``None`` when no cache root holds that exact commit, which is the
    fact a fail-closed production preflight needs: ``HF_HUB_OFFLINE=1`` means a
    revision that is not already on disk cannot be loaded at all.
    """

    pinned = validate_pinned_revision(revision)
    dirname = hf_repo_cache_dirname(model_id)
    for root in hf_cache_roots(environ):
        candidate = root / dirname / "snapshots" / pinned
        if (candidate / "config.json").is_file():
            return candidate
    return None


@dataclass(frozen=True)
class ModelIdentity:
    """The resolved answer to "exactly which weights is this stage loading?"."""

    role: str
    stage: str
    model_id: str
    mode: str
    verify: str
    revision: Optional[str]
    pin_load: bool
    local_path: Optional[str]
    inspection: Optional[CheckpointInspection]
    spec: Optional[ModelSpec]
    notes: tuple[str, ...]

    @property
    def pinned(self) -> bool:
        return self.revision is not None

    @property
    def load_kwargs(self) -> dict[str, str]:
        """Kwargs to splat into every ``from_pretrained`` call for this model.

        Empty when nothing is pinned, so an unconfigured stage keeps its exact
        pre-existing load behaviour.
        """

        if self.pin_load and self.revision:
            return {"revision": self.revision}
        return {}

    def validate_before_load(self) -> None:
        """Re-verify the checkpoint immediately before the framework loads it.

        Closes the preflight-to-load TOCTOU window via
        :meth:`ModelSpec.validate_for_load`, which re-fingerprints the directory
        and rejects any file that changed since verification. A no-op when no
        fingerprint was taken (development, or a Hub id with no local snapshot),
        so it can be called unconditionally.
        """

        if self.spec is None:
            return
        self.spec.validate_for_load(
            self.spec.checkpoint_path, revision=self.revision
        )

    def log_fields(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model_id": self.model_id,
            "mode": self.mode,
            "verify": self.verify,
            "revision": self.revision,
            "revision_pinned_at_load": self.pin_load,
            "local_path": self.local_path,
            "parameter_count": (
                self.inspection.parameter_count
                if self.inspection is not None
                else None
            ),
            "model_profile_hash": (
                self.spec.profile_hash if self.spec is not None else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "model-identity",
            **self.log_fields(),
            "stage": self.stage,
            "notes": list(self.notes),
            "inspection": (
                self.inspection.to_dict() if self.inspection is not None else None
            ),
        }


def _local_directory_identity(
    directory: Path,
    *,
    model_id: str,
    configured_revision: Optional[str],
    stage: str,
    role: str,
    mode: str,
    verify: Optional[str],
    expected: Optional[ModelProfile | ArchitectureSpec],
    environ: Mapping[str, str],
) -> ModelIdentity:
    """Identity for a local checkpoint directory, e.g. a previous stage's output.

    A directory has no Hub commit and ``transformers`` ignores ``revision`` for
    it, so a configured Hub revision is reported as ignored rather than claimed
    as this checkpoint's identity - the ``midtrain -> sft -> dpo`` handoff must
    not be labelled with the base model's commit. The path is the identity here;
    what preflight adds is architecture and shape verification of the handoff,
    which is what catches loading the wrong stage's output.
    """

    notes: list[str] = []
    if configured_revision not in (None, "", UNRESOLVED):
        notes.append(
            f"{model_id!r} is a local checkpoint directory, so the configured "
            f"revision {str(configured_revision)[:12]}... is IGNORED: a directory "
            "has no Hub commit and from_pretrained does not accept one for it. "
            "Identity here is the path plus the verified architecture"
        )
    verify_level = resolve_verify_level(verify, mode=mode, environ=environ)
    if verify_level == VERIFY_FINGERPRINT:
        # A fingerprint is only meaningful against a recorded baseline, and KORE
        # mints no immutable id for stage outputs, so there is nothing to compare.
        verify_level = VERIFY_METADATA
        notes.append(
            "fingerprint verification needs an immutable revision to bind to; "
            "a local checkpoint directory is verified at the metadata tier"
        )
    inspection: Optional[CheckpointInspection] = None
    if verify_level != VERIFY_NONE:
        try:
            inspection = inspect_local_checkpoint(
                directory, expected=expected, model_id=model_id
            )
        except (ModelSpecError, OSError) as exc:
            if mode == PRODUCTION:
                raise
            notes.append(
                f"local checkpoint verification ({verify_level}) failed and was "
                f"skipped in development mode: {exc}"
            )
            verify_level = VERIFY_NONE
    return ModelIdentity(
        role=role,
        stage=stage,
        model_id=model_id,
        mode=mode,
        verify=verify_level,
        revision=None,
        pin_load=False,
        local_path=str(directory.resolve()),
        inspection=inspection,
        spec=None,
        notes=tuple(notes),
    )


def resolve_model_identity(
    model_id: str,
    *,
    revision: Optional[str] = None,
    stage: str = "train",
    role: str = "policy",
    mode: Optional[str] = None,
    verify: Optional[str] = None,
    expected: Optional[ModelProfile | ArchitectureSpec] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> ModelIdentity:
    """Resolve, and in production enforce, the identity of one model to load.

    A Hub repo id must name an immutable commit that resolves to a local snapshot
    (production) or is reported as unpinned (development). A local checkpoint
    directory is identified by its path and verified architecture instead - see
    :func:`_local_directory_identity`.
    """

    env = _environ(environ)
    resolved_mode = resolve_identity_mode(mode, environ=env)
    revision_env = REF_REVISION_ENV if role == "reference" else REVISION_ENV
    revision_key = "ref_model_revision" if role == "reference" else "model_revision"
    raw_revision = revision if revision not in (None, "") else env.get(revision_env)
    notes: list[str] = []

    # An empty model id must not be read as the working directory.
    resolved_id = str(model_id or "").strip()
    candidate_dir = Path(resolved_id).expanduser() if resolved_id else None
    if candidate_dir is not None and candidate_dir.is_dir():
        return _local_directory_identity(
            candidate_dir,
            model_id=resolved_id,
            configured_revision=raw_revision,
            stage=stage,
            role=role,
            mode=resolved_mode,
            verify=verify,
            expected=expected,
            environ=env,
        )

    if raw_revision in (None, "", UNRESOLVED):
        message = (
            f"no immutable revision is configured for the {role} model "
            f"{model_id!r}: set the launch-config key {revision_key!r} or "
            f"${revision_env} to the full 40-hex Hub commit the training data "
            "was built for (DATASET_STATUS.md records it)"
        )
        if resolved_mode == PRODUCTION:
            raise UnpinnedModelError(
                f"{stage}: {message}. Production mode refuses to load whatever "
                "the Hugging Face cache happens to hold."
            )
        notes.append(
            message
            + "; development mode loads the cache's default snapshot unpinned"
        )
        return ModelIdentity(
            role=role,
            stage=stage,
            model_id=str(model_id),
            mode=resolved_mode,
            verify=VERIFY_NONE,
            revision=None,
            pin_load=False,
            local_path=None,
            inspection=None,
            spec=None,
            notes=tuple(notes),
        )

    # A configured-but-mutable ref is a config defect, so it fails in both modes.
    pinned = validate_pinned_revision(raw_revision)

    pin_load = True
    local_path: Optional[Path] = resolve_local_snapshot(
        model_id, pinned, environ=env
    )
    if local_path is None:
        searched = ", ".join(str(root) for root in hf_cache_roots(env))
        message = (
            f"revision {pinned} of {model_id!r} is not present in any local "
            f"Hugging Face cache (searched: {searched})"
        )
        if resolved_mode == PRODUCTION:
            raise ModelSpecError(
                f"{stage}: {message}. Download that exact commit, or point "
                "$HF_HOME/$HF_HUB_CACHE at the cache that holds it."
            )
        if hub_offline(env):
            # Pinning an uncached commit under HF_HUB_OFFLINE=1 would turn a
            # working development run into a hard load failure, so degrade.
            pin_load = False
            notes.append(
                message
                + " and the Hub is offline; proceeding UNPINNED instead of "
                "failing the load (set KORE_MODEL_IDENTITY_MODE=production "
                "to make this fatal)"
            )
        else:
            notes.append(
                message + "; keeping the pin so the Hub resolves that exact commit"
            )

    verify_level = resolve_verify_level(verify, mode=resolved_mode, environ=env)
    if local_path is None and verify_level != VERIFY_NONE:
        notes.append(
            "no local checkpoint directory to verify; identity assurance is "
            "limited to the revision pin itself"
        )
        verify_level = VERIFY_NONE

    inspection: Optional[CheckpointInspection] = None
    spec: Optional[ModelSpec] = None
    if verify_level != VERIFY_NONE:
        try:
            if verify_level == VERIFY_FINGERPRINT:
                spec = ModelSpec.from_local_checkpoint(
                    local_path,
                    revision=pinned,
                    expected=expected,
                    model_id=str(model_id),
                )
                inspection = CheckpointInspection(
                    model_id=spec.model_id,
                    revision=spec.revision,
                    checkpoint_path=spec.checkpoint_path,
                    architecture=spec.architecture,
                    checkpoint=spec.checkpoint,
                )
            else:
                inspection = inspect_local_checkpoint(
                    local_path,
                    revision=pinned,
                    expected=expected,
                    model_id=str(model_id),
                )
        except (ModelSpecError, OSError) as exc:
            if resolved_mode == PRODUCTION:
                raise
            notes.append(
                f"local checkpoint verification ({verify_level}) failed and was "
                f"skipped in development mode: {exc}"
            )
            verify_level = VERIFY_NONE
            inspection = None
            spec = None

    return ModelIdentity(
        role=role,
        stage=stage,
        model_id=str(model_id),
        mode=resolved_mode,
        verify=verify_level,
        revision=pinned,
        pin_load=pin_load,
        local_path=str(local_path) if local_path is not None else None,
        inspection=inspection,
        spec=spec,
        notes=tuple(notes),
    )


def model_identity_for_config(
    config: Any,
    *,
    stage: str,
    role: str = "policy",
    model_id: Optional[str] = None,
    expected: Optional[ModelProfile | ArchitectureSpec] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> ModelIdentity:
    """Resolve identity from a stage config's optional identity attributes.

    The stage dataclasses in :mod:`kore.policy.configs` carry no identity fields,
    so every attribute is read with a ``getattr`` default: a config written before
    the pin existed - including one an in-flight job restarts from - resolves to
    an unpinned development identity and loads exactly as it did before.
    """

    attribute = "ref_model_revision" if role == "reference" else "model_revision"
    resolved_id = model_id if model_id is not None else getattr(config, "model_id", "")
    return resolve_model_identity(
        resolved_id,
        revision=getattr(config, attribute, None),
        stage=stage,
        role=role,
        mode=getattr(config, "model_identity_mode", None),
        verify=getattr(config, "model_identity_verify", None),
        expected=expected,
        environ=environ,
    )


def log_model_identity(logger: Any, identity: ModelIdentity) -> None:
    """Emit one structured identity line, plus every degradation as a warning."""

    logger.info("model identity resolved", **identity.log_fields())
    for note in identity.notes:
        logger.warn(
            "model identity is not fully pinned",
            role=identity.role,
            model_id=identity.model_id,
            mode=identity.mode,
            note=note,
        )


def split_runtime_settings(
    payload: Mapping[str, Any], keys: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a launch-config mapping into dataclass fields and runtime settings."""

    settings = {key: payload[key] for key in keys if key in payload}
    fields = {key: value for key, value in payload.items() if key not in settings}
    return fields, settings


def apply_runtime_settings(config: Any, settings: Mapping[str, Any]) -> Any:
    """Attach runtime settings to a stage config as plain attributes."""

    for key, value in settings.items():
        if value is not None:
            setattr(config, key, value)
    return config


__all__ = [
    "DEVELOPMENT",
    "IDENTITY_CONFIG_KEYS",
    "IDENTITY_MODE_ENV",
    "IDENTITY_VERIFY_ENV",
    "PRODUCTION",
    "REF_REVISION_ENV",
    "REVISION_ENV",
    "UNRESOLVED",
    "VERIFY_FINGERPRINT",
    "VERIFY_METADATA",
    "VERIFY_NONE",
    "ArchitectureMismatchError",
    "ArchitectureSpec",
    "CheckpointCompatibilityError",
    "CheckpointInspection",
    "CheckpointMetadata",
    "FileDigest",
    "FloatingRevisionError",
    "ModelFileFingerprints",
    "ModelIdentity",
    "ModelProfile",
    "ModelSpec",
    "ModelSpecError",
    "MoESpec",
    "QWEN3_32B_PROFILE",
    "QWEN3_CODER_30B_A3B_PROFILE",
    "TensorMetadata",
    "UnpinnedModelError",
    "apply_runtime_settings",
    "canonical_profile_hash",
    "fingerprint_model_files",
    "hf_cache_roots",
    "hf_repo_cache_dirname",
    "hub_offline",
    "inspect_local_checkpoint",
    "inspect_safetensors_checkpoint",
    "load_model_spec",
    "log_model_identity",
    "model_identity_for_config",
    "read_safetensors_metadata",
    "resolve_identity_mode",
    "resolve_local_snapshot",
    "resolve_model_identity",
    "resolve_verify_level",
    "split_runtime_settings",
    "validate_checkpoint_compatibility",
    "validate_pinned_revision",
]
