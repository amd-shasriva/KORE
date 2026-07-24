"""Semantics-preserving shape augmentation.

Declared ``minimal``, ``primary``, and validation shapes are part of a task's
correctness contract.  They are therefore always returned first and unchanged.
Generated shapes only perturb an independently variable extent selected by task
metadata or a conservative operation-family rule.  Categorical and coupled
dimensions (heads, groups, top-k, reduction widths, quantization groups, and
decode batch/query regimes) remain frozen.

Unknown operations deliberately receive no generated shapes.  This fail-closed
behaviour is preferable to silently changing an operation's semantics.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from kore.tasks.base import Shape, Task

ShapeKey = tuple[tuple[str, int], ...]

# Train and hidden candidates use separate deterministic lanes.  A frozen split
# reserves the complete train lane before hidden candidates are constructed.
_TRAIN_ORDINALS = tuple(range(8))
_HIDDEN_ORDINALS = tuple(range(16, 32))


@dataclass(frozen=True)
class ShapeAugmentationPolicy:
    """Dimensions that may vary independently without changing task semantics.

    Every dimension not listed in ``mutable_dims`` is frozen to a declared
    shape.  Inferred policies intentionally contain only one mutable dimension.
    A task may opt out with ``shape_augmentation: false`` or declare a policy in
    ``task.yaml`` as ``shape_augmentation: {mutable_dims: [M]}``.
    """

    mutable_dims: tuple[str, ...] = ()
    enabled: bool = False
    source: str = "none"
    reason: str = "no safe augmentation policy"

    @classmethod
    def disabled(cls, reason: str, *, source: str = "none") -> "ShapeAugmentationPolicy":
        return cls(enabled=False, source=source, reason=reason)


@dataclass(frozen=True)
class FrozenShapeSplit:
    """Immutable exclusion set created before hidden shapes are generated."""

    task_id: str
    declared_shapes: tuple[Shape, ...]
    train_shapes: tuple[Shape, ...]
    train_keys: frozenset[ShapeKey]
    policy: ShapeAugmentationPolicy


@dataclass(frozen=True)
class TaskShapeAudit:
    task_id: str
    supported: bool
    policy_source: str
    policy_reason: str
    declared_count: int
    effective_count: int
    candidate_count: int
    odd_candidate_count: int
    hidden_count: int
    originals_preserved: bool
    deterministic: bool
    hidden_train_overlap: int
    invariant_errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            self.originals_preserved
            and self.deterministic
            and self.hidden_train_overlap == 0
            and not self.invariant_errors
        )


@dataclass(frozen=True)
class RegistryShapeAudit:
    task_count: int
    supported_tasks: int
    unsupported_tasks: int
    declared_shapes: int
    effective_shapes: int
    generated_candidates: int
    odd_candidates: int
    hidden_shapes: int
    unsupported_families: dict[str, int]
    failures: tuple[str, ...]
    tasks: tuple[TaskShapeAudit, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def shape_key(shape: Shape | Mapping[str, int]) -> ShapeKey:
    """Canonical equality key used for deduplication, never for output ordering."""
    dims = shape.dims if isinstance(shape, Shape) else shape
    return tuple(sorted(dims.items()))


def _common_dims(shapes: Sequence[Shape]) -> tuple[str, ...]:
    if not shapes:
        return ()
    return tuple(
        dim for dim in shapes[0].dims
        if all(dim in shape.dims for shape in shapes)
    )


def _dims_are_positive_ints(shapes: Sequence[Shape], dims: Sequence[str]) -> bool:
    return all(
        isinstance(shape.dims.get(dim), int)
        and not isinstance(shape.dims.get(dim), bool)
        and shape.dims[dim] > 0
        for shape in shapes
        for dim in dims
    )


def _declared_policy(task: Task, common_dims: tuple[str, ...]) -> Optional[ShapeAugmentationPolicy]:
    raw = getattr(task, "raw", {}) or {}
    if "shape_augmentation" not in raw:
        return None
    spec = raw["shape_augmentation"]
    if spec is False or spec is None or spec in ("none", "disabled"):
        return ShapeAugmentationPolicy.disabled(
            "task metadata explicitly disables augmentation", source="metadata")
    if not isinstance(spec, Mapping):
        return ShapeAugmentationPolicy.disabled(
            "shape_augmentation metadata must declare mutable_dims", source="metadata")
    if not bool(spec.get("enabled", True)):
        return ShapeAugmentationPolicy.disabled(
            str(spec.get("reason", "task metadata explicitly disables augmentation")),
            source="metadata")

    declared = spec.get("mutable_dims", ())
    if isinstance(declared, str):
        declared = (declared,)
    if not isinstance(declared, Sequence) or not declared:
        return ShapeAugmentationPolicy.disabled(
            "shape_augmentation metadata has no mutable_dims", source="metadata")
    mutable_dims = tuple(str(dim) for dim in declared)
    missing = tuple(dim for dim in mutable_dims if dim not in common_dims)
    if missing:
        return ShapeAugmentationPolicy.disabled(
            f"declared mutable dims absent from shapes: {missing}", source="metadata")
    shapes = tuple(getattr(task, "shapes", ()) or ())
    if not _dims_are_positive_ints(shapes, mutable_dims):
        return ShapeAugmentationPolicy.disabled(
            "declared mutable dims must be positive integers", source="metadata")
    return ShapeAugmentationPolicy(
        mutable_dims=mutable_dims,
        enabled=True,
        source="metadata",
        reason=str(spec.get("reason", "explicit task metadata")),
    )


def _is_stateful_or_coupled_sequence_op(operation: str) -> bool:
    """Families whose length axes have chunk/state relations we do not infer."""
    markers = (
        "ssm", "scan", "cumsum", "cumprod", "retention", "retnet",
        "rwkv", "deltanet", "hgrn", "lru", "s4d",
    )
    return any(marker in operation for marker in markers)


def _generated_family(raw: Mapping[str, object]) -> str:
    family = str(raw.get("op_family", "") or "")
    if family.startswith("breadth_"):
        return family[len("breadth_"):].split("_", 1)[0]
    if family.startswith("vendor_"):
        return "vendor"
    return family


_OUTER_M_GENERATED_FAMILIES = frozenset({
    "argmax", "binary", "block", "cross", "fused", "fusion", "fx",
    "gemm", "gemm_fusion", "kl", "label", "moe", "mse", "norm", "qx",
    "red", "reduce", "sddmm", "smp", "sort", "sparse", "spmm", "topk",
    "topp", "tr", "unary", "vendor",
})


def _derived_mutable_dim(task: Task, common_dims: tuple[str, ...]) -> tuple[str, str] | None:
    """Return one independently safe extent and the operation-family rationale."""
    operation = str(getattr(task, "operation", "") or "").lower()
    raw = getattr(task, "raw", {}) or {}
    generated = bool(raw.get("generated", False))

    if _is_stateful_or_coupled_sequence_op(operation):
        return None

    # Reductions over dim 0 use M as the reduction axis; N is the independent
    # output extent.  Other row reductions keep N/V frozen and vary outer M.
    if "dim0" in operation and "N" in common_dims:
        return "N", "dim-0 reduction outer extent"

    # Attention changes only context length.  Batch/query decode regimes and all
    # head, group, window, page-layout, and head-dimension fields stay frozen.
    if any(marker in operation for marker in ("attn", "attention", "mla")):
        for dim in ("Skv", "SK", "SKctx", "SMAX", "S"):
            if dim in common_dims:
                query_dim = "SQ" if "SQ" in common_dims else \
                    ("Sq" if "Sq" in common_dims else None)
                shapes = tuple(getattr(task, "shapes", ()) or ())
                if query_dim is not None and all(
                    shape.dims[query_dim] == shape.dims[dim] for shape in shapes
                ):
                    # Equality encodes self-attention.  Changing only one side
                    # would silently turn it into rectangular/cross attention.
                    return None
                return dim, "attention context extent"

    # Spatial extent is independent for ordinary convolution/pooling drivers.
    # Winograd transforms have fixed tile contracts and fail closed.
    spatial = ("conv", "pool", "interpolate", "im2col", "col2im")
    if any(marker in operation for marker in spatial):
        if "winograd" in operation:
            return None
        if "W" in common_dims:
            return "W", "spatial width extent"
        if "L" in common_dims:
            return "L", "convolution sequence extent"
        return None

    if "rope" in operation and "S" in common_dims:
        return "S", "RoPE sequence extent"
    if "embed" in operation:
        if "T" in common_dims:
            return "T", "embedding token extent"
        if "M" in common_dims:
            return "M", "embedding token extent"

    # These families encode block divisibility in M itself; there is no safe odd
    # M variant without a task-specific declaration.
    if operation.startswith("qx_") and "block2d" in operation:
        return None
    if operation == "block_sparse_matmul":
        return None

    # Generated M-shaped families share an explicit generator contract: M is
    # the outer row/token extent.  N/V/K/E/topk and other dimensions are not
    # touched.  Hand-authored tasks are admitted only for known operation groups.
    hand_outer_m = (
        "gemm", "matmul", "moe", "softmax", "norm", "quant", "topk",
        "entropy", "loss", "gelu", "silu", "relu", "embedding",
    )
    if "M" in common_dims and (
        (generated and _generated_family(raw) in _OUTER_M_GENERATED_FAMILIES)
        or any(marker in operation for marker in hand_outer_m)
    ):
        return "M", "operation-family outer M extent"

    if "m" in common_dims and "moe" in operation:
        return "m", "MoE per-expert row extent"
    if "R" in common_dims and "softmax" in operation:
        return "R", "row-softmax outer extent"
    return None


def augmentation_policy(task: Task) -> ShapeAugmentationPolicy:
    """Resolve an explicit or conservative inferred policy for ``task``."""
    shapes = tuple(getattr(task, "shapes", ()) or ())
    if not shapes:
        return ShapeAugmentationPolicy.disabled("task has no declared shapes")
    common_dims = _common_dims(shapes)
    explicit = _declared_policy(task, common_dims)
    if explicit is not None:
        return explicit

    derived = _derived_mutable_dim(task, common_dims)
    if derived is None:
        return ShapeAugmentationPolicy.disabled(
            "no independently variable dimension is known for this operation family",
            source="operation-family")
    dim, reason = derived
    if not _dims_are_positive_ints(shapes, (dim,)):
        return ShapeAugmentationPolicy.disabled(
            f"inferred mutable dim {dim!r} is not always a positive integer",
            source="operation-family")
    return ShapeAugmentationPolicy(
        mutable_dims=(dim,), enabled=True, source="operation-family", reason=reason)


# More discoverable alias for callers that prefer the full name.
augmentation_policy_for_task = augmentation_policy


def _next_odd(value: int, ordinal: int) -> int:
    first = value + (1 if value % 2 == 0 else 2)
    return first + 2 * ordinal


def _ordered_anchors(base: Sequence[Shape]) -> list[Shape]:
    """Prefer primary while retaining declaration order for every other anchor."""
    primary = [shape for shape in base if shape.name == "primary"]
    return primary + [shape for shape in base if shape.name != "primary"]


def _candidate_shapes(
    base: Sequence[Shape],
    policy: ShapeAugmentationPolicy,
    *,
    lane: str,
) -> Iterable[Shape]:
    if not policy.enabled:
        return
    ordinals = _TRAIN_ORDINALS if lane == "train" else _HIDDEN_ORDINALS
    anchors = _ordered_anchors(base)
    for ordinal in ordinals:
        for dim in policy.mutable_dims:
            for anchor in anchors:
                value = anchor.dims.get(dim)
                if not isinstance(value, int) or isinstance(value, bool) or value <= 1:
                    # Singleton extents frequently encode decode/control regimes,
                    # not a scalable workload size.
                    continue
                dims = dict(anchor.dims)
                dims[dim] = _next_odd(value, ordinal)
                yield Shape(
                    f"{anchor.name}_{lane}_{dim.lower()}_odd{ordinal}",
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
) -> list[Shape]:
    """Return declarations unchanged, followed by safe generated additions.

    ``max_shapes`` is a soft total-cost cap: it limits additions but never
    truncates declarations.  Thus ``max_shapes=2`` with five declared shapes
    returns all five originals and no generated shape.  ``None`` materializes
    the complete deterministic train-candidate lane.

    Without task context or an explicit policy, augmentation fails closed and
    returns only the declared shapes.
    """
    _validate_max_shapes(max_shapes)
    base = list(base_shapes)
    if not base:
        return []
    resolved = policy
    if resolved is None:
        resolved = augmentation_policy(task) if task is not None else \
            ShapeAugmentationPolicy.disabled(
                "task metadata or an explicit policy is required")

    # Never deduplicate or reorder declarations: names can encode explicit
    # coverage roles even when two declarations happen to share dimensions.
    out = list(base)
    limit = None if max_shapes is None else max(max_shapes, len(base))
    if not resolved.enabled or (limit is not None and len(out) >= limit):
        return out

    seen = {shape_key(shape) for shape in base}
    for candidate in _candidate_shapes(base, resolved, lane="train"):
        key = shape_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if limit is not None and len(out) >= limit:
            break
    return out


def freeze_shape_split(task: Task) -> FrozenShapeSplit:
    """Freeze every declared/train-candidate key before hidden generation."""
    policy = augmentation_policy(task)
    declared = tuple(
        Shape(shape.name, dict(shape.dims))
        for shape in (getattr(task, "shapes", ()) or ())
    )
    train = tuple(augment_shapes(
        declared,
        task=task,
        policy=policy,
        max_shapes=None,
    ))
    return FrozenShapeSplit(
        task_id=str(getattr(task, "task_id", "")),
        declared_shapes=declared,
        train_shapes=train,
        train_keys=frozenset(shape_key(shape) for shape in train),
        policy=policy,
    )


def generate_hidden_shapes(
    task: Task,
    frozen_split: FrozenShapeSplit,
    *,
    max_shapes: int = 8,
) -> list[Shape]:
    """Generate hidden candidates after, and disjoint from, a frozen train lane."""
    _validate_max_shapes(max_shapes)
    task_id = str(getattr(task, "task_id", ""))
    if frozen_split.task_id != task_id:
        raise ValueError(
            f"frozen split belongs to {frozen_split.task_id!r}, not {task_id!r}")
    if max_shapes == 0 or not frozen_split.policy.enabled:
        return []

    base = list(frozen_split.declared_shapes)
    seen = set(frozen_split.train_keys)
    hidden: list[Shape] = []
    for candidate in _candidate_shapes(base, frozen_split.policy, lane="hidden"):
        key = shape_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        hidden.append(candidate)
        if len(hidden) >= max_shapes:
            break
    return hidden


def generated_shape_error(
    base_shapes: Sequence[Shape],
    candidate: Shape,
    policy: ShapeAugmentationPolicy,
) -> Optional[str]:
    """Explain why a generated shape violates its frozen-dimension policy."""
    candidate_keys = tuple(candidate.dims)
    for anchor in base_shapes:
        if tuple(anchor.dims) != candidate_keys:
            continue
        changed = tuple(
            dim for dim in anchor.dims
            if anchor.dims[dim] != candidate.dims[dim]
        )
        if len(changed) != 1 or changed[0] not in policy.mutable_dims:
            continue
        value = candidate.dims[changed[0]]
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 1
            and value % 2 == 1
            and value % 8 != 0
            and value & (value - 1) != 0
        ):
            return None
    return "candidate does not match one declared anchor with one safe odd extent change"


def audit_task_shapes(
    task: Task,
    *,
    max_shapes: int = 6,
    hidden_max_shapes: int = 8,
) -> TaskShapeAudit:
    """Audit preservation, invariants, determinism, and hidden disjointness."""
    base = list(getattr(task, "shapes", ()) or ())
    policy = augmentation_policy(task)
    effective = augment_shapes(base, task=task, policy=policy, max_shapes=max_shapes)
    universe = augment_shapes(base, task=task, policy=policy, max_shapes=None)
    repeated = augment_shapes(base, task=task, policy=policy, max_shapes=None)
    generated = universe[len(base):]
    split = freeze_shape_split(task)
    hidden = generate_hidden_shapes(task, split, max_shapes=hidden_max_shapes)

    errors: list[str] = []
    for shape in (*generated, *hidden):
        error = generated_shape_error(base, shape, policy)
        if error is not None:
            errors.append(f"{shape.name}: {error}")
    hidden_keys = {shape_key(shape) for shape in hidden}
    overlap = len(hidden_keys & split.train_keys)
    if overlap:
        errors.append(f"{overlap} hidden shapes overlap the frozen train lane")

    originals_preserved = effective[:len(base)] == base
    if not originals_preserved:
        errors.append("declared shapes were reordered, changed, or removed")
    deterministic = universe == repeated
    if not deterministic:
        errors.append("augmentation output is not deterministic")

    return TaskShapeAudit(
        task_id=str(getattr(task, "task_id", "")),
        supported=policy.enabled,
        policy_source=policy.source,
        policy_reason=policy.reason,
        declared_count=len(base),
        effective_count=len(effective),
        candidate_count=len(generated),
        odd_candidate_count=sum(
            generated_shape_error(base, shape, policy) is None for shape in generated),
        hidden_count=len(hidden),
        originals_preserved=originals_preserved,
        deterministic=deterministic,
        hidden_train_overlap=overlap,
        invariant_errors=tuple(errors),
    )


def _audit_family(task: Task) -> str:
    operation = str(getattr(task, "operation", "") or "").lower()
    raw_family = str((getattr(task, "raw", {}) or {}).get("op_family", "") or "")
    if _is_stateful_or_coupled_sequence_op(operation):
        return "stateful_sequence"
    if "winograd" in operation:
        return "winograd"
    if raw_family.startswith("breadth_"):
        return raw_family[len("breadth_"):].split("_", 1)[0]
    if raw_family:
        return raw_family.split("_", 1)[0]
    return operation.split("_", 1)[0] or "unknown"


def audit_registry_shapes(
    tasks: Optional[Iterable[Task]] = None,
    *,
    max_shapes: int = 6,
    hidden_max_shapes: int = 8,
) -> RegistryShapeAudit:
    """Run the semantics/preservation audit across the live task registry."""
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
        if not report.supported
    )
    failures = tuple(
        f"{report.task_id}: {error}"
        for report in reports
        for error in report.invariant_errors
    )
    return RegistryShapeAudit(
        task_count=len(task_list),
        supported_tasks=sum(report.supported for report in reports),
        unsupported_tasks=sum(not report.supported for report in reports),
        declared_shapes=sum(report.declared_count for report in reports),
        effective_shapes=sum(report.effective_count for report in reports),
        generated_candidates=sum(report.candidate_count for report in reports),
        odd_candidates=sum(report.odd_candidate_count for report in reports),
        hidden_shapes=sum(report.hidden_count for report in reports),
        unsupported_families=dict(sorted(unsupported.items())),
        failures=failures,
        tasks=reports,
    )


# Backward-friendly alternate name for audit/reporting callers.
audit_registry_augmentation = audit_registry_shapes
