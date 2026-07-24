"""Declarative, semantics-preserving shape policies and frozen split manifests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from kore.tasks.base import Shape, Task

POLICY_SCHEMA_VERSION = 2
MANIFEST_SCHEMA_VERSION = 1
ShapeKey = tuple[tuple[str, int], ...]

BOUNDARY_REGIMES = ("small", "large", "non_power_of_two", "tail")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _shape_dict(shape: Shape) -> dict:
    return {"name": shape.name, "dims": dict(shape.dims)}


def _shape_from_dict(value: Mapping[str, object]) -> Shape:
    return Shape(str(value["name"]), {
        str(k): int(v) for k, v in dict(value["dims"]).items()
    })


def _copy_shapes(shapes: Iterable[Shape]) -> tuple[Shape, ...]:
    return tuple(Shape(shape.name, dict(shape.dims)) for shape in shapes)


def shape_key(shape: Shape | Mapping[str, int]) -> ShapeKey:
    """Canonical equality key used for comparisons, never output ordering."""
    dims = shape.dims if isinstance(shape, Shape) else shape
    return tuple(sorted((str(k), int(v)) for k, v in dims.items()))


def _shape_volume(dims: Mapping[str, int]) -> int:
    volume = 1
    for value in dims.values():
        if isinstance(value, int) and not isinstance(value, bool):
            volume *= max(1, value)
    return volume


@dataclass(frozen=True)
class AtomicShapeTransform:
    """One atomic mutation; all ``dims`` change together or none do."""

    name: str
    dims: tuple[str, ...]
    relation: str = "independent"  # independent | equal
    multiple: int = 1
    minimum: int = 2
    regimes: tuple[str, ...] = BOUNDARY_REGIMES
    misalign_to: int = 8
    misalign_dim: Optional[str] = None
    preserve_singleton: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dims": list(self.dims),
            "relation": self.relation,
            "multiple": self.multiple,
            "minimum": self.minimum,
            "regimes": list(self.regimes),
            "misalign_to": self.misalign_to,
            "misalign_dim": self.misalign_dim,
            "preserve_singleton": self.preserve_singleton,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AtomicShapeTransform":
        return cls(
            name=str(value["name"]),
            dims=tuple(str(dim) for dim in value.get("dims", ())),
            relation=str(value.get("relation", "independent")),
            multiple=int(value.get("multiple", 1)),
            minimum=int(value.get("minimum", 2)),
            regimes=tuple(str(x) for x in value.get("regimes", BOUNDARY_REGIMES)),
            misalign_to=int(value.get("misalign_to", 8)),
            misalign_dim=(
                str(value["misalign_dim"]) if value.get("misalign_dim") else None
            ),
            preserve_singleton=bool(value.get("preserve_singleton", True)),
        )


@dataclass(frozen=True)
class ShapeConstraint:
    """Serializable invariant checked on declarations, train, and hidden shapes."""

    kind: str  # equal | divisible | divides | le
    dims: tuple[str, ...]
    value: Optional[int] = None

    def to_dict(self) -> dict:
        out = {"kind": self.kind, "dims": list(self.dims)}
        if self.value is not None:
            out["value"] = self.value
        return out

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ShapeConstraint":
        return cls(
            kind=str(value["kind"]),
            dims=tuple(str(dim) for dim in value.get("dims", ())),
            value=int(value["value"]) if value.get("value") is not None else None,
        )


@dataclass(frozen=True)
class ShapeAugmentationPolicy:
    """Resolved policy with an explicit eligibility status and constraints.

    ``mutable_dims`` remains for API compatibility and represents independent
    one-dimensional transforms. New declarations should use ``transforms``.
    """

    mutable_dims: tuple[str, ...] = ()
    enabled: bool = False
    source: str = "none"
    reason: str = "no explicit shape policy"
    policy_id: str = "none"
    status: str = "ineligible"  # eligible | ineligible
    transforms: tuple[AtomicShapeTransform, ...] = ()
    constraints: tuple[ShapeConstraint, ...] = ()
    authored_hidden_shapes: tuple[Shape, ...] = ()
    max_elements: int = 0
    schema_version: int = POLICY_SCHEMA_VERSION

    @property
    def claim_eligible(self) -> bool:
        return self.status == "eligible" and (
            bool(self.effective_transforms) or bool(self.authored_hidden_shapes)
        )

    @property
    def effective_transforms(self) -> tuple[AtomicShapeTransform, ...]:
        if self.transforms:
            return self.transforms
        return tuple(
            AtomicShapeTransform(name=f"independent_{dim.lower()}", dims=(dim,))
            for dim in self.mutable_dims
        )

    @classmethod
    def disabled(
        cls,
        reason: str,
        *,
        source: str = "metadata",
        policy_id: str = "explicit_ineligible",
    ) -> "ShapeAugmentationPolicy":
        return cls(
            enabled=False,
            source=source,
            reason=reason,
            policy_id=policy_id,
            status="ineligible",
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "status": self.status,
            "enabled": self.enabled,
            "source": self.source,
            "reason": self.reason,
            "transforms": [item.to_dict() for item in self.effective_transforms],
            "constraints": [item.to_dict() for item in self.constraints],
            "authored_hidden_shapes": [
                _shape_dict(shape) for shape in self.authored_hidden_shapes
            ],
            "max_elements": self.max_elements,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ShapeAugmentationPolicy":
        status = str(value.get("status", "ineligible"))
        return cls(
            enabled=bool(value.get("enabled", status == "eligible")),
            source=str(value.get("source", "metadata")),
            reason=str(value.get("reason", "explicit metadata policy")),
            policy_id=str(value.get("policy_id", "metadata")),
            status=status,
            transforms=tuple(
                AtomicShapeTransform.from_dict(item)
                for item in value.get("transforms", ())
            ),
            constraints=tuple(
                ShapeConstraint.from_dict(item)
                for item in value.get("constraints", ())
            ),
            authored_hidden_shapes=tuple(
                _shape_from_dict(item)
                for item in value.get("authored_hidden_shapes", ())
            ),
            max_elements=int(value.get("max_elements", 0)),
            schema_version=int(value.get("schema_version", POLICY_SCHEMA_VERSION)),
        )


def policy_digest(policy: ShapeAugmentationPolicy) -> str:
    return _digest(policy.to_dict())


@dataclass(frozen=True)
class FrozenShapeSplit:
    """Serializable lineage artifact consumed by hidden evaluation."""

    schema_version: int
    task_id: str
    policy: ShapeAugmentationPolicy
    policy_digest: str
    task_digest: str
    task_file_digest: str
    engine_digest: str
    code_identity: str
    created_at: str
    seed: int
    prompt_shapes: tuple[Shape, ...]
    train_shapes: tuple[Shape, ...]
    hidden_shapes: tuple[Shape, ...]
    content_hash: str

    @property
    def train_keys(self) -> frozenset[ShapeKey]:
        return frozenset(shape_key(shape) for shape in self.train_shapes)

    @property
    def prompt_keys(self) -> frozenset[ShapeKey]:
        return frozenset(shape_key(shape) for shape in self.prompt_shapes)

    @property
    def hidden_keys(self) -> frozenset[ShapeKey]:
        return frozenset(shape_key(shape) for shape in self.hidden_shapes)

    def to_dict(self, *, include_hash: bool = True) -> dict:
        value = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "policy": self.policy.to_dict(),
            "policy_digest": self.policy_digest,
            "task_digest": self.task_digest,
            "task_file_digest": self.task_file_digest,
            "engine_digest": self.engine_digest,
            "code_identity": self.code_identity,
            "created_at": self.created_at,
            "seed": self.seed,
            "prompt_shapes": [_shape_dict(shape) for shape in self.prompt_shapes],
            "train_shapes": [_shape_dict(shape) for shape in self.train_shapes],
            "hidden_shapes": [_shape_dict(shape) for shape in self.hidden_shapes],
        }
        if include_hash:
            value["content_hash"] = self.content_hash
        return value

    def computed_hash(self) -> str:
        return _digest(self.to_dict(include_hash=False))

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FrozenShapeSplit":
        manifest = cls(
            schema_version=int(value["schema_version"]),
            task_id=str(value["task_id"]),
            policy=ShapeAugmentationPolicy.from_dict(value["policy"]),
            policy_digest=str(value["policy_digest"]),
            task_digest=str(value["task_digest"]),
            task_file_digest=str(value["task_file_digest"]),
            engine_digest=str(value["engine_digest"]),
            code_identity=str(value["code_identity"]),
            created_at=str(value["created_at"]),
            seed=int(value["seed"]),
            prompt_shapes=tuple(
                _shape_from_dict(item) for item in value.get("prompt_shapes", ())
            ),
            train_shapes=tuple(
                _shape_from_dict(item) for item in value.get("train_shapes", ())
            ),
            hidden_shapes=tuple(
                _shape_from_dict(item) for item in value.get("hidden_shapes", ())
            ),
            content_hash=str(value["content_hash"]),
        )
        if manifest.content_hash != manifest.computed_hash():
            raise ValueError("frozen shape manifest content hash mismatch")
        return manifest

    @classmethod
    def read(cls, path: str | Path) -> "FrozenShapeSplit":
        return cls.from_dict(json.loads(Path(path).read_text()))


_HAND_AUTHORED_OPERATIONS = frozenset({
    "flash_attn_backward", "flash_attn_chunked_prefill", "flash_attn_decode",
    "flash_attn_decode_fp8", "flash_attn_fp8", "flash_attn_headdim_prefill",
    "flash_attn_mha_prefill", "flash_attn_mqa_decode", "flash_attn_mqa_prefill",
    "flash_attn_noncausal_fp8", "flash_attn_noncausal_prefill",
    "flash_attn_prefill", "flash_attn_sink_prefill", "flash_attn_sliding",
    "flash_attn_sliding_decode", "flash_attn_varlen",
    "flash_attn_varlen_noncausal", "fused_add_rmsnorm", "fused_moe_silu",
    "fused_rmsnorm_quant_fp8", "fused_silu_mul_quant_fp8", "gelu_tanh",
    "gemm_backward", "gemm", "gemm_fp8", "gemm_fp8_a8w8_blockscale",
    "gemm_fp8_a8w8_pertensor", "gemm_fp8_a8w8_pertoken",
    "gemm_fp8_requant_epilogue", "gemm_int8_a8w8", "gemm_mxfp4",
    "gemm_mxfp4_a4w4", "gemm_w4a16", "gemm_w4a16_g128", "gemm_w4a8_fp8",
    "layernorm_backward", "layernorm", "mla_decode", "moe_batched_gemm",
    "moe_biased_grouped_topk", "moe_gelu", "moe_grouped_gemm",
    "moe_grouped_gemm_fp8", "moe_permute", "moe_sum_combine",
    "moe_topk_softmax_norenorm", "paged_attn_decode", "quant_fp8_pertoken",
    "rmsnorm", "rmsnorm_backward", "rope", "silu_and_mul",
    "softmax_backward", "softmax", "topk_softmax",
})

_GENERATED_FAMILIES = frozenset({
    "unary", "binary", "reduce", "fusion", "gemm_fusion",
})


def _common_dims(shapes: Sequence[Shape]) -> tuple[str, ...]:
    if not shapes:
        return ()
    return tuple(
        dim for dim in shapes[0].dims
        if all(dim in shape.dims for shape in shapes)
    )


def _known_source(
    operation: str,
    raw: Mapping[str, object],
    source: Optional[str],
) -> bool:
    if source and source.startswith("generator:"):
        return True
    if bool(raw.get("generated", False)):
        family = str(raw.get("op_family", ""))
        return (
            family.startswith("breadth_")
            or family.startswith("vendor_")
            or family in _GENERATED_FAMILIES
        )
    return operation in _HAND_AUTHORED_OPERATIONS


def _constraints_for(operation: str, dims: tuple[str, ...]) -> tuple[ShapeConstraint, ...]:
    constraints: list[ShapeConstraint] = []
    dimset = set(dims)
    if {"H", "HKV"} <= dimset:
        constraints.append(ShapeConstraint("divides", ("H", "HKV")))
    if {"H", "KV"} <= dimset:
        constraints.append(ShapeConstraint("divides", ("H", "KV")))
    if {"E", "n_groups"} <= dimset:
        constraints.append(ShapeConstraint("divides", ("E", "n_groups")))
    if {"topk", "E"} <= dimset:
        constraints.append(ShapeConstraint("le", ("topk", "E")))
    if {"topk_group", "n_groups"} <= dimset:
        constraints.append(ShapeConstraint("le", ("topk_group", "n_groups")))
    if {"cap", "M"} <= dimset:
        constraints.append(ShapeConstraint("le", ("cap", "M")))
    if "groupnorm" in operation and "N" in dimset:
        constraints.append(ShapeConstraint("divisible", ("N",), 32))
    if operation.startswith("qx_") and "block2d" in operation:
        constraints.extend((
            ShapeConstraint("divisible", ("M",), 128),
            ShapeConstraint("divisible", ("K",), 128),
        ))
    elif operation.startswith("qx_") and (
        "block128" in operation or "double_quant" in operation
        or "int4_" in operation
    ):
        constraints.append(ShapeConstraint("divisible", ("K",), 128))
    elif operation.startswith("qx_") and "mxfp4" in operation:
        constraints.append(ShapeConstraint("divisible", ("K",), 32))
    if operation == "block_sparse_matmul":
        constraints.extend((
            ShapeConstraint("divisible", ("K",), 32),
            ShapeConstraint("divisible", ("N",), 32),
        ))
    if operation == "sparse_2to4_apply":
        constraints.append(ShapeConstraint("divisible", ("N",), 4))
    if "blockscale" in operation and {"N", "K"} <= dimset:
        constraints.extend((
            ShapeConstraint("divisible", ("N",), 128),
            ShapeConstraint("divisible", ("K",), 128),
        ))
    if "K" in dimset and not operation.startswith("qx_"):
        if "g128" in operation or ("int4" in operation and "group" in operation):
            constraints.append(ShapeConstraint("divisible", ("K",), 128))
        elif "mxfp4" in operation:
            constraints.append(ShapeConstraint("divisible", ("K",), 32))
        elif "int4" in operation or "w4a" in operation:
            constraints.append(ShapeConstraint("divisible", ("K",), 2))
    return tuple(constraints)


def _transform_for(
    operation: str,
    shapes: Sequence[Shape],
    dims: tuple[str, ...],
) -> tuple[str, AtomicShapeTransform] | None:
    dimset = set(dims)

    if {"SQ", "SK"} <= dimset and all(
        shape.dims["SQ"] == shape.dims["SK"] for shape in shapes
    ):
        return "self_attention_equal_qk", AtomicShapeTransform(
            name="self_sequence",
            dims=("SQ", "SK"),
            relation="equal",
            regimes=("small", "large", "non_power_of_two", "attention_tail"),
        )

    if any(marker in operation for marker in ("attn", "attention", "mla")):
        for dim in ("Skv", "SKctx", "SK", "SMAX", "S"):
            if dim in dimset:
                alignment = 16 if "paged" in operation else 8
                tail = "page_tail" if "paged" in operation else "attention_tail"
                return "attention_context", AtomicShapeTransform(
                    name="context",
                    dims=(dim,),
                    regimes=("small", "large", "non_power_of_two", tail),
                    misalign_to=alignment,
                )

    if "dim0" in operation and "N" in dimset:
        return "dim0_outer_extent", AtomicShapeTransform(
            name="outer_columns", dims=("N",))

    sequence_markers = (
        "ssm", "scan", "cumsum", "cumprod", "linear_attention",
        "conv1d", "causal_conv", "batchnorm", "instancenorm",
    )
    if "L" in dimset and any(marker in operation for marker in sequence_markers):
        chunk_dim = "chunk" if "chunk" in dimset else None
        tail = "chunk_tail" if chunk_dim else "sequence_tail"
        return "sequence_extent", AtomicShapeTransform(
            name="sequence",
            dims=("L",),
            regimes=("small", "large", "non_power_of_two", tail),
            misalign_dim=chunk_dim,
        )

    spatial = ("conv", "pool", "interpolate", "im2col", "col2im")
    if "W" in dimset and any(marker in operation for marker in spatial):
        return "spatial_width", AtomicShapeTransform(
            name="spatial_width",
            dims=("W",),
            regimes=("small", "large", "non_power_of_two", "spatial_tail"),
        )

    if "S" in dimset and (
        "rope" in operation or "norm_qk" in operation or "attn" in operation
    ):
        return "sequence_extent", AtomicShapeTransform(
            name="sequence",
            dims=("S",),
            regimes=("small", "large", "non_power_of_two", "sequence_tail"),
        )

    if {"G", "N"} <= dimset:
        return "multi_tensor_width", AtomicShapeTransform(
            name="tensor_width",
            dims=("N",),
            regimes=("small", "large", "non_power_of_two", "reduction_tail"),
        )

    if operation.startswith("cv_winograd_input") and "N" in dimset:
        return "winograd_tiles", AtomicShapeTransform(
            name="tile_count", dims=("N",))
    if operation.startswith("cv_winograd_filter") and "Cout" in dimset:
        return "winograd_filters", AtomicShapeTransform(
            name="output_channels", dims=("Cout",))

    if "M" in dimset:
        multiple = 128 if operation.startswith("qx_") and "block2d" in operation else 1
        tail = "block_tail" if multiple > 1 else "row_tail"
        return "outer_rows", AtomicShapeTransform(
            name="outer_rows",
            dims=("M",),
            multiple=multiple,
            regimes=("small", "large", "non_power_of_two", tail),
            misalign_to=multiple * 2 if multiple > 1 else 8,
        )
    if "m" in dimset:
        return "expert_rows", AtomicShapeTransform(name="expert_rows", dims=("m",))
    if "R" in dimset:
        return "outer_rows", AtomicShapeTransform(name="outer_rows", dims=("R",))
    if "T" in dimset:
        return "token_count", AtomicShapeTransform(name="tokens", dims=("T",))
    if "Cout" in dimset:
        return "output_channels", AtomicShapeTransform(
            name="output_channels", dims=("Cout",))
    if "N" in dimset:
        return "outer_extent", AtomicShapeTransform(name="outer_extent", dims=("N",))
    return None


def _policy_from_metadata(
    value: object,
    shapes: Sequence[Shape],
) -> ShapeAugmentationPolicy:
    if value is False or value is None or value in ("none", "disabled"):
        return ShapeAugmentationPolicy.disabled(
            "task metadata explicitly marks hidden-shape claims ineligible")
    if not isinstance(value, Mapping):
        return ShapeAugmentationPolicy.disabled(
            "invalid shape_policy metadata; expected a mapping")
    policy = ShapeAugmentationPolicy.from_dict(value)
    if policy.max_elements <= 0 and shapes:
        policy = replace(
            policy,
            max_elements=max(_shape_volume(shape.dims) for shape in shapes) * 4,
        )
    return policy


def _resolve_engine_policy(
    operation: str,
    shapes: Sequence[Shape],
    *,
    raw: Mapping[str, object],
    source: Optional[str] = None,
) -> ShapeAugmentationPolicy:
    if not shapes:
        return ShapeAugmentationPolicy.disabled("task has no declared shapes")
    if not _known_source(operation, raw, source):
        return ShapeAugmentationPolicy.disabled(
            "operation is absent from explicit generator/hand-authored policy catalogs",
            source="engine:catalog",
            policy_id="catalog_unknown",
        )
    dims = _common_dims(shapes)
    resolved = _transform_for(operation, shapes, dims)
    if resolved is None:
        return ShapeAugmentationPolicy.disabled(
            f"no authored atomic transform for dimensions {dims}",
            source="engine:catalog",
            policy_id="catalog_no_transform",
        )
    rule, transform = resolved
    transform = replace(
        transform,
        minimum=min(
            shape.dims[dim]
            for shape in shapes
            for dim in transform.dims
        ),
    )
    max_elements = max(_shape_volume(shape.dims) for shape in shapes) * 4
    constraints = list(_constraints_for(operation, dims))
    for query_dim, context_dim in (
        ("SQ", "SK"), ("SQ", "SKctx"), ("Sq", "Skv"),
    ):
        if {query_dim, context_dim} <= set(dims) and all(
            shape.dims[query_dim] <= shape.dims[context_dim] for shape in shapes
        ):
            constraints.append(
                ShapeConstraint("le", (query_dim, context_dim)))
    if transform.relation == "equal":
        constraints.append(ShapeConstraint("equal", transform.dims))
    return ShapeAugmentationPolicy(
        enabled=True,
        source=source or f"engine:{rule}",
        reason=f"explicit {rule} transform with frozen non-target dimensions",
        policy_id=rule,
        status="eligible",
        transforms=(transform,),
        constraints=tuple(constraints),
        max_elements=max_elements,
    )


def augmentation_policy(task: Task) -> ShapeAugmentationPolicy:
    """Resolve metadata first, then an explicit source-engine catalog rule."""
    shapes = tuple(getattr(task, "shapes", ()) or ())
    raw = getattr(task, "raw", {}) or {}
    if "shape_policy" in raw:
        return _policy_from_metadata(raw["shape_policy"], shapes)
    if "shape_augmentation" in raw:
        legacy = raw["shape_augmentation"]
        if isinstance(legacy, Mapping) and "mutable_dims" in legacy:
            mutable = legacy.get("mutable_dims", ())
            if isinstance(mutable, str):
                mutable = (mutable,)
            policy = ShapeAugmentationPolicy(
                mutable_dims=tuple(str(dim) for dim in mutable),
                enabled=True,
                source="metadata:legacy",
                reason=str(legacy.get("reason", "legacy explicit mutable dimensions")),
                policy_id="legacy_independent",
                status="eligible",
                max_elements=max(
                    (_shape_volume(shape.dims) for shape in shapes), default=1) * 4,
            )
            return policy
        return _policy_from_metadata(legacy, shapes)
    return _resolve_engine_policy(
        str(getattr(task, "operation", "") or "").lower(),
        shapes,
        raw=raw,
    )


augmentation_policy_for_task = augmentation_policy


def policy_metadata_for_shapes(
    operation: str,
    shape_spec: Mapping[str, object],
    *,
    source: str,
) -> dict:
    """Policy metadata emitted by task source generators."""
    shapes: list[Shape] = [
        Shape("minimal", dict(shape_spec["minimal"])),
        Shape("primary", dict(shape_spec["primary"])),
    ]
    shapes.extend(
        Shape(f"validation_{index}", dict(dims))
        for index, dims in enumerate(shape_spec.get("validation", ()))
    )
    policy = _resolve_engine_policy(
        operation.lower(),
        shapes,
        raw={"generated": True, "op_family": source.removeprefix("generator:")},
        source=source,
    )
    return policy.to_dict()


def shape_policy_yaml_lines(
    operation: str,
    shape_spec: Mapping[str, object],
    *,
    source: str,
) -> list[str]:
    metadata = policy_metadata_for_shapes(operation, shape_spec, source=source)
    return yaml.safe_dump(
        {"shape_policy": metadata},
        sort_keys=False,
        default_flow_style=False,
    ).strip().splitlines()


def constraint_errors(
    dims: Mapping[str, int],
    policy: ShapeAugmentationPolicy,
) -> tuple[str, ...]:
    errors: list[str] = []
    for constraint in policy.constraints:
        if any(dim not in dims for dim in constraint.dims):
            errors.append(f"{constraint.kind} references absent dims {constraint.dims}")
            continue
        values = tuple(int(dims[dim]) for dim in constraint.dims)
        if constraint.kind == "equal" and len(set(values)) != 1:
            errors.append(f"{constraint.dims} must remain equal")
        elif constraint.kind == "divisible":
            divisor = int(constraint.value or 1)
            if values[0] % divisor != 0:
                errors.append(f"{constraint.dims[0]} must be divisible by {divisor}")
        elif constraint.kind == "divides" and values[1] > 0 and values[0] % values[1]:
            errors.append(f"{constraint.dims[1]} must divide {constraint.dims[0]}")
        elif constraint.kind == "le" and values[0] > values[1]:
            errors.append(f"{constraint.dims[0]} must be <= {constraint.dims[1]}")
    if policy.max_elements > 0 and _shape_volume(dims) > policy.max_elements:
        errors.append(
            f"shape volume {_shape_volume(dims)} exceeds cap {policy.max_elements}")
    return tuple(errors)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _target_value(
    value: int,
    transform: AtomicShapeTransform,
    regime: str,
    *,
    lane: str,
    seed: int,
    anchor_dims: Mapping[str, int],
) -> Optional[int]:
    multiple = max(1, transform.multiple)
    if value <= 0 or value % multiple:
        return None
    if value == 1 and transform.preserve_singleton:
        return None
    units = value // multiple
    minimum_units = max(1, (transform.minimum + multiple - 1) // multiple)
    bias = seed % 3
    if regime == "small":
        target_units = max(
            minimum_units, units // (2 if lane == "train" else 3))
    elif regime == "large":
        target_units = (
            units * 2 + bias
            if lane == "train"
            else units + max(2, units // 2) + bias
        )
    elif regime == "non_power_of_two":
        target_units = units + (3 + 2 * bias if lane == "train" else 9 + 2 * bias)
        while _is_power_of_two(target_units * multiple):
            target_units += 1
    else:  # operation-specific tail/misalignment regime
        offset = 1 + 2 * bias if lane == "train" else 7 + 2 * bias
        if multiple == 1:
            target = value + (offset if value % 2 == 0 else offset + 1)
            alignment = transform.misalign_to
            if transform.misalign_dim:
                alignment = int(anchor_dims.get(transform.misalign_dim, alignment))
            alignment = max(2, alignment)
            while target % alignment == 0 or _is_power_of_two(target):
                target += 2
            return target
        offset = 5 + 2 * bias if lane == "train" else 13 + 2 * bias
        target_units = units + offset
        while _is_power_of_two(target_units) or (
            target_units * multiple) % max(multiple + 1, transform.misalign_to) == 0:
            target_units += 1
    target = target_units * multiple
    return target if target > 1 and target != value else None


def _ordered_anchors(base: Sequence[Shape]) -> list[Shape]:
    primary = [shape for shape in base if shape.name == "primary"]
    return primary + [shape for shape in base if shape.name != "primary"]


def _candidate_shapes(
    base: Sequence[Shape],
    policy: ShapeAugmentationPolicy,
    *,
    lane: str,
    seed: int,
) -> Iterable[Shape]:
    if not policy.claim_eligible:
        return
    for anchor in _ordered_anchors(base):
        for transform in policy.effective_transforms:
            values = tuple(anchor.dims.get(dim) for dim in transform.dims)
            if not values or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in values
            ):
                continue
            if transform.relation == "equal" and len(set(values)) != 1:
                continue
            if transform.relation == "independent" and len(values) != 1:
                continue
            for regime in transform.regimes:
                target = _target_value(
                    int(values[0]),
                    transform,
                    regime,
                    lane=lane,
                    seed=seed,
                    anchor_dims=anchor.dims,
                )
                if target is None:
                    continue
                dims = dict(anchor.dims)
                for dim in transform.dims:
                    dims[dim] = target
                if constraint_errors(dims, policy):
                    continue
                yield Shape(
                    f"{anchor.name}__{lane}_{regime}_{transform.name}",
                    dims,
                )


def _validate_max_shapes(max_shapes: Optional[int]) -> None:
    if max_shapes is None:
        return
    if not isinstance(max_shapes, int) or isinstance(max_shapes, bool) or max_shapes < 0:
        raise ValueError("max_shapes must be a non-negative integer or None")


def augment_shapes(
    base_shapes: Iterable[Shape],
    *,
    task: Optional[Task] = None,
    policy: Optional[ShapeAugmentationPolicy] = None,
    max_shapes: Optional[int] = 6,
    seed: int = 0,
) -> list[Shape]:
    """Preserve declarations first, then append deterministic safe boundaries."""
    _validate_max_shapes(max_shapes)
    base = list(base_shapes)
    if not base:
        return []
    resolved = policy
    if resolved is None:
        resolved = augmentation_policy(task) if task is not None else \
            ShapeAugmentationPolicy.disabled(
                "task metadata or an explicit policy is required")
    out = list(base)
    limit = None if max_shapes is None else max(max_shapes, len(base))
    if not resolved.claim_eligible or (limit is not None and len(out) >= limit):
        return out
    seen = {shape_key(shape) for shape in base}
    for candidate in _candidate_shapes(
        base, resolved, lane="train", seed=int(seed)
    ):
        key = shape_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if limit is not None and len(out) >= limit:
            break
    return out


def boundary_regime(shape: Shape) -> Optional[str]:
    marker = "__train_" if "__train_" in shape.name else \
        ("__hidden_" if "__hidden_" in shape.name else None)
    if marker is None:
        return None
    suffix = shape.name.split(marker, 1)[1]
    for regime in (
        "non_power_of_two", "attention_tail", "sequence_tail", "spatial_tail",
        "reduction_tail", "chunk_tail", "page_tail", "block_tail", "row_tail",
        "small", "large", "tail",
    ):
        if suffix.startswith(regime + "_"):
            return regime
    return None


def generated_shape_error(
    base_shapes: Sequence[Shape],
    candidate: Shape,
    policy: ShapeAugmentationPolicy,
) -> Optional[str]:
    regime = boundary_regime(candidate)
    if regime is None:
        return "generated shape has no boundary-regime provenance"
    for anchor in base_shapes:
        if tuple(anchor.dims) != tuple(candidate.dims):
            continue
        changed = tuple(
            dim for dim in anchor.dims
            if anchor.dims[dim] != candidate.dims[dim]
        )
        for transform in policy.effective_transforms:
            if changed != transform.dims:
                continue
            values = tuple(candidate.dims[dim] for dim in transform.dims)
            if transform.relation == "equal" and len(set(values)) != 1:
                continue
            if constraint_errors(candidate.dims, policy):
                continue
            if regime == "small" and values[0] >= anchor.dims[transform.dims[0]]:
                continue
            if regime == "large" and values[0] <= anchor.dims[transform.dims[0]]:
                continue
            if regime == "non_power_of_two" and _is_power_of_two(values[0]):
                continue
            if regime.endswith("tail"):
                alignment = transform.misalign_to
                if transform.misalign_dim:
                    alignment = anchor.dims.get(transform.misalign_dim, alignment)
                if values[0] % max(2, alignment) == 0:
                    continue
            return None
    return "candidate violates its atomic transform, constraints, regime, or memory cap"


def _task_digest(task: Task) -> str:
    return _digest({
        "task_id": str(getattr(task, "task_id", "")),
        "operation": str(getattr(task, "operation", "")),
        "dtype": str(getattr(task, "dtype", "")),
        "backend": str(getattr(task, "backend", "")),
        "gpu_target": str(getattr(task, "gpu_target", "")),
        "shapes": [
            _shape_dict(shape) for shape in (getattr(task, "shapes", ()) or ())
        ],
    })


def _task_file_digest(task: Task, *, refresh: bool = False) -> str:
    loaded = str(getattr(task, "task_file_digest", "") or "")
    if loaded and not refresh:
        return loaded
    directory = getattr(task, "dir", None)
    if directory:
        path = Path(directory) / "task.yaml"
        if path.exists():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    return loaded or _digest(getattr(task, "raw", {}) or {})


@lru_cache(maxsize=1)
def _cached_engine_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _engine_digest(*, refresh: bool = False) -> str:
    if refresh:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return _cached_engine_digest()


def _read_repository_code_identity() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return f"engine:{_engine_digest()}"


@lru_cache(maxsize=1)
def _cached_repository_code_identity() -> str:
    return _read_repository_code_identity()


def _code_identity(task: Task, *, refresh: bool = False) -> str:
    explicit = os.environ.get("KORE_CODE_IDENTITY")
    if explicit:
        return explicit
    return (
        _read_repository_code_identity()
        if refresh else _cached_repository_code_identity()
    )


def freeze_shape_split(
    task: Task,
    *,
    prompt_shapes: Iterable[Shape] = (),
    hidden_max_shapes: int = 8,
    seed: int = 0,
    created_at: Optional[str] = None,
    code_identity: Optional[str] = None,
) -> FrozenShapeSplit:
    """Create the exact train/hidden artifact that training lineage persists."""
    _validate_max_shapes(hidden_max_shapes)
    policy = augmentation_policy(task)
    declared = _copy_shapes(getattr(task, "shapes", ()) or ())
    prompt = _copy_shapes((*declared, *tuple(prompt_shapes)))
    train = tuple(augment_shapes(
        declared,
        task=task,
        policy=policy,
        max_shapes=None,
        seed=seed,
    ))
    excluded = {shape_key(shape) for shape in (*prompt, *train)}

    hidden: list[Shape] = []
    if policy.authored_hidden_shapes:
        for shape in policy.authored_hidden_shapes:
            if shape_key(shape) in excluded:
                raise ValueError(f"authored hidden shape {shape.name!r} overlaps prompt/train")
            if constraint_errors(shape.dims, policy):
                raise ValueError(f"authored hidden shape {shape.name!r} violates policy")
            excluded.add(shape_key(shape))
            hidden.append(Shape(shape.name, dict(shape.dims)))
    elif policy.claim_eligible:
        for shape in _candidate_shapes(
            declared, policy, lane="hidden", seed=int(seed)
        ):
            if shape_key(shape) in excluded:
                continue
            excluded.add(shape_key(shape))
            hidden.append(shape)
            if len(hidden) >= hidden_max_shapes:
                break
    if policy.claim_eligible and hidden_max_shapes > 0 and not hidden:
        raise ValueError(
            f"claim-eligible task {getattr(task, 'task_id', '')!r} has no hidden shapes")

    manifest = FrozenShapeSplit(
        schema_version=MANIFEST_SCHEMA_VERSION,
        task_id=str(getattr(task, "task_id", "")),
        policy=policy,
        policy_digest=policy_digest(policy),
        task_digest=_task_digest(task),
        task_file_digest=_task_file_digest(task),
        engine_digest=_engine_digest(),
        code_identity=code_identity or _code_identity(task),
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        seed=int(seed),
        prompt_shapes=prompt,
        train_shapes=train,
        hidden_shapes=tuple(hidden),
        content_hash="",
    )
    return replace(manifest, content_hash=manifest.computed_hash())


def validate_frozen_split(
    task: Task,
    frozen_split: FrozenShapeSplit,
    *,
    code_identity: Optional[str] = None,
    refresh_task_file: bool = True,
    refresh_code: bool = True,
) -> None:
    """Reject stale, tampered, overlapping, or constraint-invalid artifacts."""
    if frozen_split.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported frozen shape manifest schema")
    if frozen_split.content_hash != frozen_split.computed_hash():
        raise ValueError("frozen shape manifest content hash mismatch")
    if frozen_split.policy_digest != policy_digest(frozen_split.policy):
        raise ValueError("frozen shape manifest internal policy digest mismatch")
    if frozen_split.task_id != str(getattr(task, "task_id", "")):
        raise ValueError("frozen shape manifest belongs to another task")
    current_policy = augmentation_policy(task)
    checks = (
        ("task", frozen_split.task_digest, _task_digest(task)),
        (
            "task file",
            frozen_split.task_file_digest,
            _task_file_digest(task, refresh=refresh_task_file),
        ),
        ("policy", frozen_split.policy_digest, policy_digest(current_policy)),
        (
            "policy engine",
            frozen_split.engine_digest,
            _engine_digest(refresh=refresh_code),
        ),
        (
            "code identity",
            frozen_split.code_identity,
            code_identity or _code_identity(task, refresh=refresh_code),
        ),
    )
    for label, frozen, current in checks:
        if frozen != current:
            raise ValueError(f"frozen shape manifest {label} digest changed")
    expected_train = tuple(augment_shapes(
        getattr(task, "shapes", ()) or (),
        task=task,
        policy=current_policy,
        max_shapes=None,
        seed=frozen_split.seed,
    ))
    if tuple(frozen_split.train_shapes) != expected_train:
        raise ValueError("frozen shape manifest train/augmentation universe changed")
    declared_keys = {
        shape_key(shape) for shape in (getattr(task, "shapes", ()) or ())
    }
    if not declared_keys <= frozen_split.prompt_keys:
        raise ValueError("frozen shape manifest omits declared prompt shapes")
    if not declared_keys <= frozen_split.train_keys:
        raise ValueError("frozen shape manifest omits declared train shapes")
    if len(frozen_split.hidden_keys) != len(frozen_split.hidden_shapes):
        raise ValueError("frozen shape manifest contains duplicate hidden shapes")
    overlap = frozen_split.hidden_keys & (
        frozen_split.prompt_keys | frozen_split.train_keys)
    if overlap:
        raise ValueError("hidden shapes overlap prompt/train/augmentation shapes")
    base = list(getattr(task, "shapes", ()) or ())
    for shape in frozen_split.train_shapes:
        errors = constraint_errors(shape.dims, current_policy)
        if errors:
            raise ValueError(
                f"train shape {shape.name!r} violates policy: {', '.join(errors)}")
    for shape in frozen_split.hidden_shapes:
        errors = constraint_errors(shape.dims, current_policy)
        if errors:
            raise ValueError(
                f"hidden shape {shape.name!r} violates policy: {', '.join(errors)}")
        if not current_policy.authored_hidden_shapes:
            error = generated_shape_error(base, shape, current_policy)
            if error:
                raise ValueError(f"hidden shape {shape.name!r} is invalid: {error}")


def generate_hidden_shapes(
    task: Task,
    frozen_split: FrozenShapeSplit,
    *,
    max_shapes: int = 8,
) -> list[Shape]:
    """Consume stored hidden shapes; never derive a post-training split."""
    _validate_max_shapes(max_shapes)
    validate_frozen_split(task, frozen_split)
    return list(_copy_shapes(frozen_split.hidden_shapes[:max_shapes]))


@dataclass(frozen=True)
class TaskShapeAudit:
    task_id: str
    explicit_status: bool
    claim_eligible: bool
    supported: bool
    policy_source: str
    policy_reason: str
    declared_count: int
    effective_count: int
    candidate_count: int
    odd_candidate_count: int
    hidden_count: int
    hidden_regimes: tuple[str, ...]
    originals_preserved: bool
    deterministic: bool
    hidden_train_overlap: int
    invariant_errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            self.explicit_status
            and self.originals_preserved
            and self.deterministic
            and self.hidden_train_overlap == 0
            and not self.invariant_errors
        )


@dataclass(frozen=True)
class RegistryShapeAudit:
    task_count: int
    explicit_status_tasks: int
    claim_eligible_tasks: int
    supported_tasks: int
    unsupported_tasks: int
    declared_shapes: int
    effective_shapes: int
    generated_candidates: int
    odd_candidates: int
    hidden_shapes: int
    eligible_family_hidden_coverage: dict[str, int]
    unsupported_families: dict[str, int]
    failures: tuple[str, ...]
    tasks: tuple[TaskShapeAudit, ...]

    @property
    def ok(self) -> bool:
        return not self.failures and self.explicit_status_tasks == self.task_count


def audit_task_shapes(
    task: Task,
    *,
    max_shapes: int = 6,
    hidden_max_shapes: int = 8,
) -> TaskShapeAudit:
    base = list(getattr(task, "shapes", ()) or ())
    policy = augmentation_policy(task)
    effective = augment_shapes(base, task=task, policy=policy, max_shapes=max_shapes)
    universe = augment_shapes(base, task=task, policy=policy, max_shapes=None)
    repeated = augment_shapes(base, task=task, policy=policy, max_shapes=None)
    generated = universe[len(base):]
    errors: list[str] = []

    for shape in base:
        for error in constraint_errors(shape.dims, policy):
            errors.append(f"{shape.name}: declared constraint violation: {error}")
    for shape in generated:
        error = generated_shape_error(base, shape, policy)
        if error:
            errors.append(f"{shape.name}: {error}")

    manifest: Optional[FrozenShapeSplit] = None
    hidden: tuple[Shape, ...] = ()
    try:
        manifest = freeze_shape_split(
            task,
            hidden_max_shapes=hidden_max_shapes,
            created_at="2000-01-01T00:00:00+00:00",
        )
        validate_frozen_split(
            task,
            manifest,
            refresh_task_file=False,
            refresh_code=False,
        )
        hidden = manifest.hidden_shapes
    except ValueError as exc:
        errors.append(str(exc))

    hidden_keys = {shape_key(shape) for shape in hidden}
    excluded = set()
    if manifest is not None:
        excluded = set(manifest.prompt_keys | manifest.train_keys)
    overlap = len(hidden_keys & excluded)
    regimes = tuple(sorted({
        regime for shape in hidden
        if (regime := boundary_regime(shape)) is not None
    }))
    if policy.claim_eligible:
        if not hidden:
            errors.append("claim-eligible task has no hidden coverage")
        if len(regimes) < 3 and not policy.authored_hidden_shapes:
            errors.append(f"hidden coverage has only {len(regimes)} boundary regimes")

    originals_preserved = effective[:len(base)] == base
    if not originals_preserved:
        errors.append("declared shapes were reordered, changed, or removed")
    deterministic = universe == repeated
    if not deterministic:
        errors.append("augmentation output is not deterministic")
    explicit = policy.status in ("eligible", "ineligible") and policy.source != "none"
    if not explicit:
        errors.append("task has no explicit shape-claim status")

    return TaskShapeAudit(
        task_id=str(getattr(task, "task_id", "")),
        explicit_status=explicit,
        claim_eligible=policy.claim_eligible,
        supported=policy.claim_eligible,
        policy_source=policy.source,
        policy_reason=policy.reason,
        declared_count=len(base),
        effective_count=len(effective),
        candidate_count=len(generated),
        odd_candidate_count=sum(
            boundary_regime(shape) in (
                "non_power_of_two", "attention_tail", "sequence_tail",
                "spatial_tail", "reduction_tail", "chunk_tail", "page_tail",
                "block_tail", "row_tail", "tail",
            )
            for shape in generated
        ),
        hidden_count=len(hidden),
        hidden_regimes=regimes,
        originals_preserved=originals_preserved,
        deterministic=deterministic,
        hidden_train_overlap=overlap,
        invariant_errors=tuple(errors),
    )


def _audit_family(task: Task) -> str:
    operation = str(getattr(task, "operation", "") or "").lower()
    raw_family = str((getattr(task, "raw", {}) or {}).get("op_family", "") or "")
    if raw_family.startswith("breadth_"):
        return raw_family[len("breadth_"):].split("_", 1)[0]
    if raw_family.startswith("vendor_"):
        return "vendor"
    if raw_family:
        return raw_family.split("_", 1)[0]
    if operation.startswith("flash_attn"):
        return "attention"
    if operation.startswith("gemm"):
        return "gemm"
    if operation.startswith("moe"):
        return "moe"
    return operation.split("_", 1)[0] or "unknown"


def audit_registry_shapes(
    tasks: Optional[Iterable[Task]] = None,
    *,
    max_shapes: int = 6,
    hidden_max_shapes: int = 8,
) -> RegistryShapeAudit:
    if tasks is None:
        from kore.tasks.registry import all_tasks
        task_list = all_tasks()
    else:
        task_list = list(tasks)
    reports = tuple(
        audit_task_shapes(
            task, max_shapes=max_shapes, hidden_max_shapes=hidden_max_shapes)
        for task in task_list
    )
    unsupported = Counter(
        _audit_family(task)
        for task, report in zip(task_list, reports)
        if not report.claim_eligible
    )
    family_coverage = Counter(
        _audit_family(task)
        for task, report in zip(task_list, reports)
        if report.claim_eligible and report.hidden_count > 0
    )
    failures = tuple(
        f"{report.task_id}: {error}"
        for report in reports
        for error in report.invariant_errors
    )
    return RegistryShapeAudit(
        task_count=len(task_list),
        explicit_status_tasks=sum(report.explicit_status for report in reports),
        claim_eligible_tasks=sum(report.claim_eligible for report in reports),
        supported_tasks=sum(report.supported for report in reports),
        unsupported_tasks=sum(not report.claim_eligible for report in reports),
        declared_shapes=sum(report.declared_count for report in reports),
        effective_shapes=sum(report.effective_count for report in reports),
        generated_candidates=sum(report.candidate_count for report in reports),
        odd_candidates=sum(report.odd_candidate_count for report in reports),
        hidden_shapes=sum(report.hidden_count for report in reports),
        eligible_family_hidden_coverage=dict(sorted(family_coverage.items())),
        unsupported_families=dict(sorted(unsupported.items())),
        failures=failures,
        tasks=reports,
    )


audit_registry_augmentation = audit_registry_shapes
