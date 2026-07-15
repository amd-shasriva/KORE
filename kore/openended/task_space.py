"""A PARAMETRIC, verifiable task descriptor + space for open-ended co-evolution.

A :class:`TaskDescriptor` is a concrete, gradable KORE task drawn from the
existing parametric op registries:

  * ``kore.tasks._genops`` - the generated op space (``unary`` / ``binary`` /
    ``reduce`` / ``fusion`` / ``gemm_fusion`` families, torch-eager / hipBLASLt
    baselines).
  * ``kore.tasks.vendor_ops`` - the vendor-baselined ops (graded vs real AITER
    kernels), family ``vendor_<op>`` to match ``registry.operator_family`` /
    the generated ``op_family`` yaml field.

Both source registries are trainable-only: the held-out generalization families
(``mla`` / ``paged_attention``, see :mod:`kore.tasks.registry`) are absent from
this space by construction, so a descriptor can never name a held-out task and the
proposer built on top of it cannot leak one into training.

A descriptor = ``(source, family, op, dtype, shape_regime)``; its *difficulty
features* are derived (not stored) via :func:`descriptor_features`. Those
features double as the MAP-Elites behavior descriptor:

  * ``family``               - coarse operator family (categorical)
  * ``arithmetic_intensity`` - ``compute-bound`` (gemm) vs ``memory-bound``
  * ``fusion_depth``         - number of fused sub-ops (int)
  * ``dtype_precision``      - ``16b`` (bf16/fp16) vs ``32b`` (fp32)
  * ``shape_scale``          - ``small`` / ``medium`` / ``large`` by problem volume

:func:`descriptor_key` reduces those to the archive niche tuple.

Everything is deterministic and pure. ``torch`` is only imported lazily (the
underlying ``_genops._registry`` builds torch op specs), so importing this module
and reading op names/dtypes/shapes needs no GPU; only enumerating the ``fusion`` /
``gemm_fusion`` families touches torch (CPU is fine).
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, replace
from typing import Iterable, Optional

# --------------------------------------------------------------------------- #
# Lazy views onto the existing KORE registries (torch imported only on demand)
# --------------------------------------------------------------------------- #
# _genops families whose op specs are torch-free to inspect for fusion-depth.
_SIMPLE_GENOPS_FAMILIES = ("unary", "binary", "reduce")

# vendor op -> family (matches generate_vendor_ops.py `op_family: vendor_<op>`).
_VENDOR_OPS = ("rmsnorm", "layernorm", "silu_mul", "gelu_mul", "softmax", "gemm_a8w8",
               "fused_add_rmsnorm", "rope", "topk_softmax", "batched_gemm",
               "gemm_a8w8_blockscale")

# per-op fusion depth for vendor ops (reduction/affine/gated chains).
_VENDOR_FUSION_DEPTH = {"rmsnorm": 2, "layernorm": 3, "silu_mul": 2, "gelu_mul": 2,
                        "softmax": 2, "gemm_a8w8": 2, "fused_add_rmsnorm": 3, "rope": 1,
                        "topk_softmax": 3, "batched_gemm": 1, "gemm_a8w8_blockscale": 2}

# vendor ops that are matmul-class (compute-bound) rather than memory-bound.
_VENDOR_COMPUTE_BOUND = frozenset({"gemm_a8w8", "batched_gemm", "gemm_a8w8_blockscale"})


@functools.lru_cache(maxsize=1)
def _genops_registry() -> dict:
    """``op -> (family, spec)`` from ``kore.tasks._genops`` (imports torch)."""
    from kore.tasks import _genops
    return _genops._registry()


@functools.lru_cache(maxsize=1)
def _genops_family_dtypes() -> dict:
    from kore.tasks.generate_ops import FAMILY_DTYPES
    return {k: tuple(v) for k, v in FAMILY_DTYPES.items()}


@functools.lru_cache(maxsize=1)
def _genops_shape_tables() -> tuple:
    from kore.tasks.generate_ops import GEMM_SHAPES, SHAPES
    return SHAPES, GEMM_SHAPES


@functools.lru_cache(maxsize=1)
def _vendor_tables() -> tuple:
    from kore.tasks.vendor_ops import VENDOR_DTYPES, VENDOR_SHAPES
    return VENDOR_SHAPES, tuple(VENDOR_DTYPES)


@functools.lru_cache(maxsize=32)
def _vendor_op_dtypes(op: str) -> tuple:
    from kore.tasks.vendor_ops import vendor_op_dtypes
    return tuple(vendor_op_dtypes(op))


def _regime_dims(shapes: dict) -> dict:
    """Expand a family shape table into ``regime_name -> dims`` (validation split)."""
    out: dict[str, dict] = {"minimal": dict(shapes["minimal"]),
                            "primary": dict(shapes["primary"])}
    for i, dims in enumerate(shapes.get("validation", []) or []):
        out[f"validation_{i}"] = dict(dims)
    return out


def vendor_family(op: str) -> str:
    return f"vendor_{op}"


# --------------------------------------------------------------------------- #
# TaskDescriptor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaskDescriptor:
    """A concrete, verifiable KORE task drawn from the parametric op space.

    ``source`` is ``"genops"`` (torch-eager / hipBLASLt baseline) or ``"vendor"``
    (AITER baseline); ``family`` is the coarse operator family; ``op`` the op
    name; ``dtype`` in {bf16, fp16, fp32}; ``shape_regime`` one of
    ``minimal`` / ``primary`` / ``validation_<i>``.
    """

    source: str
    family: str
    op: str
    dtype: str
    shape_regime: str = "primary"

    @property
    def task_id(self) -> str:
        """The KORE task_id this descriptor corresponds to (`gen_`/`genv_`)."""
        prefix = "genv" if self.source == "vendor" else "gen"
        return f"{prefix}_{self.op}_{self.dtype}"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.task_id}[{self.shape_regime}]"


def _sort_key(desc: TaskDescriptor) -> tuple:
    """Deterministic total order over descriptors (stable tie-breaks)."""
    return (desc.source, desc.family, desc.op, desc.dtype, desc.shape_regime)


# --------------------------------------------------------------------------- #
# Enumeration / sampling
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4)
def _enumerate_cached(include_vendor: bool) -> tuple:
    descs: list[TaskDescriptor] = []
    reg = _genops_registry()
    fam_dtypes = _genops_family_dtypes()
    shapes, gemm_shapes = _genops_shape_tables()
    for op in sorted(reg):
        family = reg[op][0]
        regimes = _regime_dims(gemm_shapes if family == "gemm_fusion" else shapes)
        for dtype in fam_dtypes[family]:
            for regime in regimes:
                descs.append(TaskDescriptor("genops", family, op, dtype, regime))
    if include_vendor:
        vendor_shapes, _ = _vendor_tables()
        for op in _VENDOR_OPS:
            regimes = _regime_dims(vendor_shapes[op])
            for dtype in _vendor_op_dtypes(op):     # per-op dtype sweep (fp8 GEMM etc.)
                for regime in regimes:
                    descs.append(
                        TaskDescriptor("vendor", vendor_family(op), op, dtype, regime))
    descs.sort(key=_sort_key)
    return tuple(descs)


def enumerate_descriptors(include_vendor: bool = True) -> list[TaskDescriptor]:
    """All concrete descriptors in the parametric space (deterministic order)."""
    return list(_enumerate_cached(include_vendor))


def families(include_vendor: bool = True) -> list[str]:
    """Sorted list of operator families present in the space."""
    return sorted({d.family for d in _enumerate_cached(include_vendor)})


def sample_descriptor(rng: random.Random, include_vendor: bool = True) -> TaskDescriptor:
    return rng.choice(_enumerate_cached(include_vendor))


def sample_descriptors(n: int, seed: int = 0,
                       include_vendor: bool = True) -> list[TaskDescriptor]:
    """``n`` descriptors sampled uniformly from the space (seeded, deterministic)."""
    rng = random.Random(seed)
    pool = _enumerate_cached(include_vendor)
    return [rng.choice(pool) for _ in range(max(0, n))]


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #
def descriptor_shape(desc: TaskDescriptor) -> dict:
    """Concrete dim dict (e.g. ``{"M":.., "N":..}``) for this descriptor's regime."""
    if desc.source == "vendor":
        vendor_shapes, _ = _vendor_tables()
        regimes = _regime_dims(vendor_shapes[desc.op])
    else:
        shapes, gemm_shapes = _genops_shape_tables()
        regimes = _regime_dims(gemm_shapes if desc.family == "gemm_fusion" else shapes)
    if desc.shape_regime not in regimes:
        raise KeyError(f"unknown shape_regime {desc.shape_regime!r} for {desc.task_id}")
    return dict(regimes[desc.shape_regime])


def _problem_volume(desc: TaskDescriptor) -> int:
    dims = descriptor_shape(desc)
    vol = 1
    for v in dims.values():
        vol *= int(v)
    return vol


# --------------------------------------------------------------------------- #
# Behavior / difficulty features
# --------------------------------------------------------------------------- #
_COMPUTE_BOUND_FAMILIES = frozenset({"gemm_fusion"})
_PRECISION_CLASS = {"bf16": "16b", "fp16": "16b", "fp32": "32b", "fp8": "8b", "int8": "8b"}
# problem-volume thresholds (elements, or M*N*K work for gemm) -> scale class.
_SCALE_SMALL = 1_000_000
_SCALE_LARGE = 1_000_000_000


def arithmetic_intensity(desc: TaskDescriptor) -> str:
    """``compute-bound`` (matmul-class) vs ``memory-bound`` (elementwise/reduce)."""
    if desc.family in _COMPUTE_BOUND_FAMILIES or desc.op in _VENDOR_COMPUTE_BOUND:
        return "compute-bound"
    return "memory-bound"


def fusion_depth(desc: TaskDescriptor) -> int:
    """Number of fused sub-ops (higher = harder to keep fused in registers)."""
    fam = desc.family
    if fam in _SIMPLE_GENOPS_FAMILIES:
        return 1
    if fam.startswith("vendor_"):
        return _VENDOR_FUSION_DEPTH.get(desc.op, 2)
    # fusion / gemm_fusion need the torch-built spec (lazy).
    spec = _genops_registry()[desc.op][1]
    if fam == "fusion":
        return int(getattr(spec, "arity", 2))
    if fam == "gemm_fusion":
        depth = 1  # the matmul itself
        if getattr(spec, "has_bias", False):
            depth += 1
        if getattr(spec, "act", "none") != "none":
            depth += 1
        return depth
    return 1


def shape_scale(desc: TaskDescriptor) -> str:
    vol = _problem_volume(desc)
    if vol < _SCALE_SMALL:
        return "small"
    if vol < _SCALE_LARGE:
        return "medium"
    return "large"


def descriptor_features(desc: TaskDescriptor) -> dict:
    """MAP-Elites behavior dimensions + difficulty features for ``desc``."""
    return {
        "family": desc.family,
        "arithmetic_intensity": arithmetic_intensity(desc),
        "fusion_depth": fusion_depth(desc),
        "dtype_precision": _PRECISION_CLASS[desc.dtype],
        "dtype": desc.dtype,
        "shape_scale": shape_scale(desc),
    }


# order of the fields that make up the archive niche key.
NICHE_FIELDS = ("family", "arithmetic_intensity", "fusion_depth",
                "dtype_precision", "shape_scale")


def descriptor_key(desc: TaskDescriptor) -> tuple:
    """The MAP-Elites niche key (behavior descriptor) for the task archive."""
    feats = descriptor_features(desc)
    return tuple(feats[f] for f in NICHE_FIELDS)


def static_difficulty(desc: TaskDescriptor) -> float:
    """A cheap, GPU-free difficulty prior in ``[0, 1]`` (heuristic).

    Compute-bound + deeper fusion + lower precision + bigger shapes are harder.
    Used only as a prior/tie-break; the real signal is measured solve-rate/regret.
    """
    feats = descriptor_features(desc)
    score = 0.0
    if feats["arithmetic_intensity"] == "compute-bound":
        score += 0.35
    score += min(0.30, 0.10 * (feats["fusion_depth"] - 1))
    if feats["dtype_precision"] == "16b":
        score += 0.15
    score += {"small": 0.0, "medium": 0.10, "large": 0.20}[feats["shape_scale"]]
    return max(0.0, min(1.0, score))


# --------------------------------------------------------------------------- #
# Mutation (perturb shape / dtype / fusion-depth) - the frontier operator
# --------------------------------------------------------------------------- #
MUTATION_KINDS = ("shape", "dtype", "fusion")


def _valid_dtypes(desc: TaskDescriptor) -> list[str]:
    if desc.source == "vendor":
        return list(_vendor_op_dtypes(desc.op))
    return list(_genops_family_dtypes()[desc.family])


def _family_ops(desc: TaskDescriptor) -> list[str]:
    """Other ops in the same family (perturbs op / fusion-depth within family)."""
    if desc.source == "vendor":
        return [op for op in _VENDOR_OPS if vendor_family(op) == desc.family]
    reg = _genops_registry()
    return sorted(op for op, (fam, _) in reg.items() if fam == desc.family)


def _regime_names(desc: TaskDescriptor) -> list[str]:
    if desc.source == "vendor":
        vendor_shapes, _ = _vendor_tables()
        return list(_regime_dims(vendor_shapes[desc.op]))
    shapes, gemm_shapes = _genops_shape_tables()
    table = gemm_shapes if desc.family == "gemm_fusion" else shapes
    return list(_regime_dims(table))


def mutate(desc: TaskDescriptor, rng: random.Random,
           kinds: Iterable[str] = MUTATION_KINDS) -> TaskDescriptor:
    """Perturb one axis of ``desc`` (shape regime, dtype, or op/fusion-depth).

    Tries the requested mutation kinds in a seeded-random order and applies the
    first one that yields a *different* valid descriptor. Falls back to ``desc``
    unchanged only when no axis can move (degenerate family)."""
    order = list(kinds)
    rng.shuffle(order)
    for kind in order:
        if kind == "shape":
            choices = [r for r in _regime_names(desc) if r != desc.shape_regime]
            if choices:
                return replace(desc, shape_regime=rng.choice(sorted(choices)))
        elif kind == "dtype":
            choices = [d for d in _valid_dtypes(desc) if d != desc.dtype]
            if choices:
                return replace(desc, dtype=rng.choice(sorted(choices)))
        elif kind == "fusion":
            choices = [o for o in _family_ops(desc) if o != desc.op]
            if choices:
                return replace(desc, op=rng.choice(sorted(choices)))
        else:
            raise ValueError(f"unknown mutation kind {kind!r}")
    return desc


def describe(desc: TaskDescriptor) -> dict:
    """A compact JSON-friendly view (task_id + regime + behavior features)."""
    out = {"task_id": desc.task_id, "source": desc.source,
           "shape_regime": desc.shape_regime, "shape": descriptor_shape(desc)}
    out.update(descriptor_features(desc))
    out["static_difficulty"] = static_difficulty(desc)
    return out
