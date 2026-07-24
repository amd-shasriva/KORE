"""Versioned authority for KORE task families and train/eval assignment.

There are two related, deliberately different views:

* ``product_family`` is the leaf used by the immutable product split.  Core
  attention, MLA, and paged attention are distinct leaves.
* ``analysis_family`` is a parent rollup used for reports and leave-one-family-out
  analysis.  The three attention leaves all roll up to ``attention``.

Registry tasks are classified from exact task metadata and the generator source
that emitted that metadata.  Name rules exist only as a centralized adapter for
unregistered records/minted candidates; registry validation never relies on them.
Unknown identities are eval-only, rather than becoming a raw-string family.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pkgutil
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional


TAXONOMY_VERSION = "1.0.0"

# The product is native to gfx950 and accepts the immediately preceding CDNA
# generation for continuity.  Every other architecture is an explicit OOD slice.
PRIMARY_TRAIN_ARCHITECTURE = "gfx950"
TRAIN_ARCHITECTURES = frozenset({"gfx950", "gfx942"})

# These are the storage/contract dtype IDs present in the product task ABI.  A
# newly introduced dtype must be reviewed and added here before it can enter train.
TRAIN_DTYPES = frozenset({
    "bf16",
    "fp16",
    "fp32",
    "fp8",
    "fp8_e4m3fn",
    "int8",
    "int4_w4a16",
    "int4_w4a8",
    "mxfp4",
    "mxfp4_w4a16",
})


class TaxonomyError(ValueError):
    """A task identity cannot be classified under the versioned contract."""


@dataclass(frozen=True)
class FamilySpec:
    """One product leaf and its report/mutation parents."""

    product_id: str
    analysis_id: str
    mutation_id: str
    description: str


FAMILY_SPECS: tuple[FamilySpec, ...] = (
    FamilySpec("activation", "activation", "activation", "activations and gates"),
    FamilySpec("attention", "attention", "attention", "core dense/flash attention"),
    FamilySpec("convolution", "convolution", "gemm", "convolution and pooling"),
    FamilySpec("data_movement", "data_movement", "generic", "gather and layout movement"),
    FamilySpec("elementwise", "activation", "activation", "non-activation pointwise math"),
    FamilySpec("fusion", "activation", "activation", "multi-op pointwise/projection fusion"),
    FamilySpec("gemm", "gemm", "gemm", "matrix contractions and epilogues"),
    FamilySpec("mla", "attention", "attention", "multi-head latent attention"),
    FamilySpec("moe", "moe", "moe", "mixture-of-experts routing and compute"),
    FamilySpec("normalization", "norm", "norm", "normalization forward/backward/fusion"),
    FamilySpec("paged_attention", "attention", "attention", "paged-KV attention"),
    FamilySpec("positional", "positional", "activation", "rotary/position transforms"),
    FamilySpec("quantization", "quant", "quant", "quantize/dequantize/packing"),
    FamilySpec("reduction", "reduction", "generic", "reductions, softmax, and losses"),
    FamilySpec("sampling", "sampling", "generic", "sampling and logit processing"),
    FamilySpec("sequence", "sequence", "generic", "scan, SSM, and sequence operators"),
    FamilySpec("sparse", "sparse", "gemm", "sorting and sparse contractions"),
    FamilySpec("training", "training", "generic", "optimizers, gradients, and train losses"),
)

_FAMILY_BY_ID: Mapping[str, FamilySpec] = MappingProxyType(
    {spec.product_id: spec for spec in FAMILY_SPECS}
)
PRODUCT_FAMILIES: tuple[str, ...] = tuple(spec.product_id for spec in FAMILY_SPECS)
ANALYSIS_FAMILIES: tuple[str, ...] = tuple(
    dict.fromkeys(spec.analysis_id for spec in FAMILY_SPECS)
)
LEGACY_FAMILY_ALIASES: Mapping[str, str] = MappingProxyType({
    "layernorm": "normalization",
    "minted_elementwise": "activation",
    "minted_fusion": "fusion",
    "minted_gemm_fusion": "gemm",
    "minted_norm": "normalization",
    "minted_reduce": "reduction",
    "norm": "normalization",
    "quant": "quantization",
    "rmsnorm": "normalization",
    "rope": "positional",
})

# Whole product leaves withheld from training.  ``attention`` is intentionally
# absent: near probes inside core attention are task-level reservations only.
WHOLE_FAMILY_HOLDOUTS = frozenset({"mla", "paged_attention"})

# Deterministic, stratified near-generalization probes.  Their product families
# remain trainable; only these exact task identities (and descendants carrying
# the same provenance root) are eval-only.
NEAR_GENERALIZATION_TASK_IDS = frozenset({
    "genb_attn2_cross_gqa_step_fp16",
    "genb_attn2_decode_mqa_hd256_bf16",
    "genb_attn2_varlen_gqa_causal_bf16",
    "genb_attn_fp8_mha_hd128_causal_fp8",
    "genb_attn_mha_hd128_noncausal_fp16",
    "genb_attn_mqa_hd64_causal_bf16",
    "genb_cv_conv2d_1x1_s2_fp16",
    "genb_cv_conv2d_7x7_s1_d1_fp16",
    "genb_cv_conv2d_nhwc_5x5_s1_d1_bf16",
    "genb_cv_depthwise_conv2d_5x5_s1_bf16",
    "genb_fx_embed_scale_bf16",
    "genb_fx_reglu_act_fp16",
    "genb_fx_rope_qk_half_qknorm_bf16",
    "genb_gemm_bf16_residual_bf16",
    "genb_gemm_fp8_channelwise_fp8",
    "genb_gemm_int4_asym_group_fp16",
    "genb_gemm_int8_pertensor_int8",
    "genb_moe_block_silu_k8_e256_bf16",
    "genb_moe_fused_moe_silu_fp16",
    "genb_moe_grouped_mlp_gelu_bf16",
    "genb_moe_sigmoid_topk_norenorm_fp16",
    "genb_norm_groupnorm_bf16",
    "genb_norm_layernorm_bwd_fp32",
    "genb_norm_layernorm_quant_fp8_fp8",
    "genb_norm_rmsnorm_h16384_bf16",
    "genb_qx_int4_unpack_group_bf16",
    "genb_qx_quant_fp8_block2d_fp8",
    "genb_qx_quant_int8_pertoken_int8",
    "genb_red_entropy_bf16",
    "genb_red_log_softmax_bwd_fp32",
    "genb_red_rms_bf16",
    "genb_red_topk256_bf16",
    "genb_smp_repetition_penalty_bf16",
    "genb_smp_rope_yarn_bf16",
    "genb_smp_topk_sample_bf16",
    "genb_ssm_gated_retention_c128_bf16",
    "genb_ssm_lightning_attn_bf16",
    "genb_ssm_mamba2_ssd_c128_n128_bf16",
    "genb_ssm_retnet_c64_bf16",
    "genb_tr_adamw_8bit_bf16",
    "genb_tr_foreach_sgd_bf16",
    "genb_tr_ls_ce_bwd_bf16",
    "genb_tr_rmsprop_centered_momentum_fp32",
})


# Generator metadata -> canonical product leaf.  ``gen_*`` stores one of these
# source-family IDs in task.yaml; the raw source family is never exposed as the
# product taxonomy.
GENOPS_SOURCE_FAMILIES: Mapping[str, str] = MappingProxyType({
    "unary": "activation",
    "binary": "elementwise",
    "reduce": "reduction",
    "fusion": "fusion",
    "gemm_fusion": "gemm",
})

# Semantically stable operation IDs override a broader generator template family.
# The vendor generator emits the same operations, so both sources must agree.
GENOPS_OPERATION_OVERRIDES: Mapping[str, str] = MappingProxyType({
    "gelu_mul": "activation",
    "silu_mul": "activation",
})

# Breadth operation membership comes from each generator module's ``OPS`` tuple.
# This table maps the source module (not substrings in the operation) to a product
# leaf.  Adding a conformant module without assigning it here fails registry load.
BREADTH_MODULE_FAMILIES: Mapping[str, str] = MappingProxyType({
    "attn2_ext": "attention",
    "attn_ext": "attention",
    "conv": "convolution",
    "conv_ext": "convolution",
    "fused_ext": "fusion",
    "gemm_ext": "gemm",
    "moe_ext": "moe",
    "norm_ext": "normalization",
    "quant_ext": "quantization",
    "reduce_ext": "reduction",
    "sample_ext": "sampling",
    "seq": "sequence",
    "sort_sparse": "sparse",
    "ssm_ext": "sequence",
    "train_ext": "training",
    "train_ops": "training",
})

VENDOR_OPERATION_FAMILIES: Mapping[str, str] = MappingProxyType({
    "rmsnorm": "normalization",
    "layernorm": "normalization",
    "silu_mul": "activation",
    "gelu_mul": "activation",
    "softmax": "reduction",
    "gemm_a8w8": "gemm",
    "fused_add_rmsnorm": "normalization",
    "rope": "positional",
    "topk_softmax": "moe",
    "batched_gemm": "gemm",
    "gemm_a8w8_blockscale": "gemm",
    "rope_gptj": "positional",
    "rope_partial": "positional",
    "embedding_gather": "data_movement",
})


def _grouped_map(groups: Mapping[str, tuple[str, ...]]) -> Mapping[str, str]:
    out: dict[str, str] = {}
    for family, operations in groups.items():
        for operation in operations:
            previous = out.setdefault(operation, family)
            if previous != family:
                raise RuntimeError(
                    f"taxonomy source collision for {operation!r}: "
                    f"{previous!r} versus {family!r}"
                )
    return MappingProxyType(out)


# Hand-authored task metadata has no generator source-family field, so its exact
# operation IDs are curated here.  This is intentionally exhaustive and exact.
HAND_OPERATION_FAMILIES: Mapping[str, str] = _grouped_map({
    "attention": (
        "flash_attn_backward",
        "flash_attn_chunked_prefill",
        "flash_attn_decode",
        "flash_attn_decode_fp8",
        "flash_attn_fp8",
        "flash_attn_headdim_prefill",
        "flash_attn_mha_prefill",
        "flash_attn_mqa_decode",
        "flash_attn_mqa_prefill",
        "flash_attn_noncausal_fp8",
        "flash_attn_noncausal_prefill",
        "flash_attn_prefill",
        "flash_attn_sink_prefill",
        "flash_attn_sliding",
        "flash_attn_sliding_decode",
        "flash_attn_varlen",
        "flash_attn_varlen_noncausal",
    ),
    "mla": ("mla_decode",),
    "paged_attention": ("paged_attn_decode",),
    "normalization": (
        "fused_add_rmsnorm",
        "fused_rmsnorm_quant_fp8",
        "layernorm",
        "layernorm_backward",
        "rmsnorm",
        "rmsnorm_backward",
    ),
    "activation": (
        "fused_silu_mul_quant_fp8",
        "gelu_tanh",
        "silu_and_mul",
    ),
    "gemm": (
        "gemm",
        "gemm_backward",
        "gemm_fp8",
        "gemm_fp8_a8w8_blockscale",
        "gemm_fp8_a8w8_pertensor",
        "gemm_fp8_a8w8_pertoken",
        "gemm_fp8_requant_epilogue",
        "gemm_int8_a8w8",
        "gemm_mxfp4",
        "gemm_mxfp4_a4w4",
        "gemm_w4a16",
        "gemm_w4a16_g128",
        "gemm_w4a8_fp8",
    ),
    "moe": (
        "fused_moe_silu",
        "moe_batched_gemm",
        "moe_biased_grouped_topk",
        "moe_gelu",
        "moe_grouped_gemm",
        "moe_grouped_gemm_fp8",
        "moe_permute",
        "moe_sum_combine",
        "moe_topk_softmax_norenorm",
        "topk_softmax",
    ),
    "quantization": ("quant_fp8_pertoken",),
    "positional": ("rope",),
    "reduction": ("softmax", "softmax_backward"),
})


@lru_cache(maxsize=1)
def breadth_operation_families() -> Mapping[str, str]:
    """Exact breadth operation map, derived from generator module ``OPS`` tables."""

    import kore.tasks.breadth as breadth_pkg

    out: dict[str, str] = {}
    for info in sorted(pkgutil.iter_modules(breadth_pkg.__path__), key=lambda x: x.name):
        if info.name == "tests" or info.name.startswith("_"):
            continue
        module = importlib.import_module(f"kore.tasks.breadth.{info.name}")
        if not all(
            hasattr(module, attr)
            for attr in ("OPS", "SHAPES", "make_reference", "seed_source")
        ):
            continue
        family = BREADTH_MODULE_FAMILIES.get(info.name)
        if family is None:
            raise TaxonomyError(
                f"breadth generator {info.name!r} has no product-family assignment"
            )
        for operation in module.OPS:
            previous = out.setdefault(operation, family)
            if previous != family:
                raise TaxonomyError(
                    f"breadth operation {operation!r} collides between "
                    f"{previous!r} and {family!r}"
                )
    return MappingProxyType(out)


@lru_cache(maxsize=1)
def vendor_operation_families() -> Mapping[str, str]:
    """Validate the curated vendor map against the live generator inventory."""

    from kore.tasks.vendor_ops import VENDOR_OPS

    expected = set(VENDOR_OPS)
    actual = set(VENDOR_OPERATION_FAMILIES)
    if expected != actual:
        raise TaxonomyError(
            "vendor taxonomy/source drift: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return VENDOR_OPERATION_FAMILIES


# The only ordered name adapter in the repository.  It is used for external
# records and minted names, never for registry admission.  Reserved attention
# variants intentionally precede generic attention.
_NAME_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mla", ("mla", "latent_attn", "latent_attention")),
    ("paged_attention", ("paged_attn", "paged_attention", "paged_kv")),
    ("attention", ("attention", "attn", "flash", "mha", "mqa", "gqa", "sdpa")),
    ("moe", ("moe", "expert", "topk_softmax", "grouped_topk")),
    ("gemm", ("gemm", "matmul", "bmm", "matrix_mul")),
    ("normalization", (
        "rmsnorm", "layernorm", "rms_norm", "layer_norm", "groupnorm",
        "batchnorm", "instancenorm", "l2norm",
    )),
    ("positional", ("rope", "rotary")),
    ("quantization", (
        "quant", "dequant", "w8a8", "w4a", "mxfp", "fp8_scale", "qx_",
    )),
    ("convolution", ("conv", "pool", "interpolate")),
    ("sampling", ("smp_", "sample", "temperature", "repetition_penalty")),
    ("sequence", ("ssm_", "scan", "cumsum", "cumprod", "retention", "rwkv", "mamba")),
    ("training", ("tr_", "optimizer", "adam", "rmsprop", "sgd_", "grad_")),
    ("sparse", ("sparse", "spmm", "sddmm", "sort_lastdim")),
    ("reduction", (
        "softmax", "reduce", "row_", "argmax", "argmin", "logsumexp",
        "cross_entropy", "entropy", "variance", "welford",
    )),
    ("fusion", ("fx_", "fusion", "fused_")),
    ("activation", (
        "gelu", "silu", "relu", "sigmoid", "tanh", "swiglu", "geglu",
        "reglu", "activation",
    )),
    ("data_movement", ("embedding_gather", "gather", "scatter")),
)

_EXACT_NAME_FAMILIES: Mapping[str, str] = MappingProxyType({
    "abs": "activation",
    "act_fn": "activation",
    "add": "elementwise",
    "add_mul": "fusion",
    "div": "elementwise",
    "elu": "activation",
    "exp": "activation",
    "fma": "fusion",
    "hardsigmoid": "activation",
    "hardswish": "activation",
    "hardtanh": "activation",
    "log": "activation",
    "maximum": "elementwise",
    "minimum": "elementwise",
    "mish": "activation",
    "mul": "elementwise",
    "mul_sig": "elementwise",
    "neg": "activation",
    "reciprocal": "activation",
    "rsqrt": "activation",
    "sign": "activation",
    "softplus": "activation",
    "softsign": "activation",
    "sqrt": "activation",
    "square": "activation",
    "sub": "elementwise",
})


def canonical_product_family(family: str) -> str:
    """Validate and return a canonical product family ID."""

    value = str(family or "").strip().lower()
    value = LEGACY_FAMILY_ALIASES.get(value, value)
    if value not in _FAMILY_BY_ID:
        raise TaxonomyError(
            f"unknown product family {family!r}; known={list(PRODUCT_FAMILIES)}"
        )
    return value


def family_spec(product_family: str) -> FamilySpec:
    return _FAMILY_BY_ID[canonical_product_family(product_family)]


def analysis_family(product_family: str) -> str:
    return family_spec(product_family).analysis_id


def mutation_family(product_family: str) -> str:
    return family_spec(product_family).mutation_id


def product_family_for_name(name: str) -> Optional[str]:
    """Best-effort adapter for an unregistered operation/task name.

    The return value is a canonical finite family or ``None``.  It never returns
    the raw input, which prevents arbitrary operation names from becoming families.
    """

    value = str(name or "").strip().lower()
    if not value:
        return None
    if value in _EXACT_NAME_FAMILIES:
        return _EXACT_NAME_FAMILIES[value]
    if value in HAND_OPERATION_FAMILIES:
        return HAND_OPERATION_FAMILIES[value]
    if value in vendor_operation_families():
        return vendor_operation_families()[value]
    breadth = breadth_operation_families()
    if value in breadth:
        return breadth[value]
    for family, markers in _NAME_RULES:
        if any(marker in value for marker in markers):
            return family
    return None


def product_family_for_source(
    source: str,
    operation: str,
    source_family: Optional[str] = None,
) -> str:
    """Classify an operation from an exact generator source contract."""

    src = str(source or "").strip().lower()
    op = str(operation or "").strip().lower()
    sf = str(source_family or "").strip().lower()
    if src == "genops":
        family = GENOPS_SOURCE_FAMILIES.get(sf)
        if family is None:
            raise TaxonomyError(f"unknown genops source family {source_family!r}")
        return GENOPS_OPERATION_OVERRIDES.get(op, family)
    if src == "vendor":
        family = vendor_operation_families().get(op)
        if family is None:
            raise TaxonomyError(f"unknown vendor operation {operation!r}")
        expected = f"vendor_{op}"
        if sf and sf != expected:
            raise TaxonomyError(
                f"vendor operation {op!r} declares {sf!r}, expected {expected!r}"
            )
        return family
    if src == "breadth":
        family = breadth_operation_families().get(op)
        if family is None:
            raise TaxonomyError(f"unknown breadth operation {operation!r}")
        expected = f"breadth_{op}"
        if sf and sf != expected:
            raise TaxonomyError(
                f"breadth operation {op!r} declares {sf!r}, expected {expected!r}"
            )
        return family
    if src == "minted":
        family = canonical_product_family(sf)
        reserved = product_family_for_name(op)
        if reserved in WHOLE_FAMILY_HOLDOUTS and reserved != family:
            raise TaxonomyError(
                f"minted operation {op!r} aliases reserved family {reserved!r}, "
                f"not declared family {family!r}"
            )
        return family
    raise TaxonomyError(f"unknown task generator source {source!r}")


def product_family_for_task(task: Any, *, strict: bool = True) -> Optional[str]:
    """Classify a task from exact metadata/source, optionally adapting externals."""

    task_id = str(getattr(task, "task_id", "") or "").strip()
    operation = str(getattr(task, "operation", "") or "").strip().lower()
    raw = getattr(task, "raw", {}) or {}
    if not isinstance(raw, dict):
        raise TaxonomyError(f"task {task_id or '<unknown>'}: raw metadata is not a mapping")
    source_family = raw.get("op_family")

    if raw.get("minted"):
        family = raw.get("taxonomy_family") or source_family
        return product_family_for_source("minted", operation, family)
    if raw.get("generated"):
        if task_id.startswith("genb_"):
            return product_family_for_source("breadth", operation, source_family)
        if task_id.startswith("genv_"):
            return product_family_for_source("vendor", operation, source_family)
        if task_id.startswith("gen_"):
            return product_family_for_source("genops", operation, source_family)
        raise TaxonomyError(
            f"generated task {task_id!r} has no recognized generator prefix"
        )

    source = str(raw.get("source") or "").lower()
    if source == "kernelbench":
        declared = raw.get("taxonomy_family") or raw.get("family")
        if declared in _FAMILY_BY_ID:
            return canonical_product_family(declared)
        inferred = product_family_for_name(operation or task_id)
        if inferred is not None:
            return inferred
    family = HAND_OPERATION_FAMILIES.get(operation)
    if family is not None:
        return family
    if strict:
        raise TaxonomyError(
            f"task {task_id!r}: operation {operation!r} is absent from exact "
            "task metadata/generator taxonomy"
        )
    return product_family_for_name(operation or task_id)


def analysis_family_for_task(task: Any, *, strict: bool = True) -> str:
    product = product_family_for_task(task, strict=strict)
    return analysis_family(product) if product is not None else "other"


def analysis_family_for_name(name: str) -> str:
    product = product_family_for_name(name)
    return analysis_family(product) if product is not None else "other"


def provenance_root_for_task(task: Any) -> str:
    """Return the immutable lineage root used by split collision checks."""

    task_id = str(getattr(task, "task_id", "") or "").strip()
    root = getattr(task, "provenance_root", None)
    raw = getattr(task, "raw", {}) or {}
    if not root and isinstance(raw, dict):
        root = raw.get("provenance_root") or raw.get("lineage_root")
        provenance = raw.get("provenance")
        if not root and isinstance(provenance, dict):
            root = provenance.get("root")
    root = str(root or task_id).strip()
    if not root:
        raise TaxonomyError(f"task {task_id or '<unknown>'}: empty provenance root")
    return root


@dataclass(frozen=True)
class SplitDecision:
    task_id: str
    split: str
    reason: str
    product_family: Optional[str]
    analysis_family: str
    provenance_root: str

    @property
    def heldout(self) -> bool:
        return self.split == "eval"


def split_decision(task: Any, *, strict: bool = True) -> SplitDecision:
    """Assign a Task-like object under the fail-closed product split."""

    task_id = str(getattr(task, "task_id", "") or "").strip()
    if not task_id:
        raise TaxonomyError("task has an empty task_id")
    product = product_family_for_task(task, strict=strict)
    analysis = analysis_family(product) if product is not None else "other"
    root = provenance_root_for_task(task)

    if task_id in NEAR_GENERALIZATION_TASK_IDS:
        return SplitDecision(task_id, "eval", "near_probe", product, analysis, root)
    if root in NEAR_GENERALIZATION_TASK_IDS:
        return SplitDecision(task_id, "eval", "heldout_lineage", product, analysis, root)
    if product in WHOLE_FAMILY_HOLDOUTS:
        return SplitDecision(task_id, "eval", "whole_family", product, analysis, root)

    arch = str(getattr(task, "gpu_target", "") or "").strip()
    if arch not in TRAIN_ARCHITECTURES:
        return SplitDecision(task_id, "eval", "foreign_arch", product, analysis, root)
    dtype = str(getattr(task, "dtype", "") or "").strip()
    if dtype not in TRAIN_DTYPES:
        return SplitDecision(task_id, "eval", "foreign_dtype", product, analysis, root)
    if product is None:
        return SplitDecision(task_id, "eval", "unclassified_operation", None, "other", root)
    return SplitDecision(task_id, "train", "train", product, analysis, root)


def split_decision_for_identity(
    *,
    task_id: str,
    operation: str = "",
    product_family: Optional[str] = None,
    architecture: Optional[str] = None,
    dtype: Optional[str] = None,
    provenance_root: Optional[str] = None,
) -> SplitDecision:
    """Fail-closed split adapter for records and in-memory candidates."""

    tid = str(task_id or "").strip()
    if not tid:
        raise TaxonomyError("record has an empty task_id")
    inferred = product_family_for_name(operation or tid)
    declared = None
    if product_family:
        try:
            declared = canonical_product_family(product_family)
        except TaxonomyError:
            declared = None
    # A reserved name always wins over a generic/incorrect declaration.
    product = inferred if inferred in WHOLE_FAMILY_HOLDOUTS else (declared or inferred)
    analysis = analysis_family(product) if product is not None else "other"
    root = str(provenance_root or tid).strip()
    if not root:
        raise TaxonomyError(f"record {tid!r}: empty provenance root")

    if tid in NEAR_GENERALIZATION_TASK_IDS:
        return SplitDecision(tid, "eval", "near_probe", product, analysis, root)
    if root in NEAR_GENERALIZATION_TASK_IDS:
        return SplitDecision(tid, "eval", "heldout_lineage", product, analysis, root)
    if product in WHOLE_FAMILY_HOLDOUTS:
        return SplitDecision(tid, "eval", "whole_family", product, analysis, root)
    if architecture is not None and architecture not in TRAIN_ARCHITECTURES:
        return SplitDecision(tid, "eval", "foreign_arch", product, analysis, root)
    if dtype is not None and dtype not in TRAIN_DTYPES:
        return SplitDecision(tid, "eval", "foreign_dtype", product, analysis, root)
    if product is None:
        return SplitDecision(tid, "eval", "unclassified_operation", None, "other", root)
    return SplitDecision(tid, "train", "train", product, analysis, root)


def validate_task_assignments(tasks: Iterable[Any]) -> Mapping[str, str]:
    """Validate complete IDs and a collision-free operation -> family map."""

    by_id: dict[str, Any] = {}
    casefold_ids: dict[str, str] = {}
    operation_map: dict[str, str] = {}
    for task in tasks:
        task_id = str(getattr(task, "task_id", "") or "").strip()
        operation = str(getattr(task, "operation", "") or "").strip().lower()
        if not task_id or not operation:
            raise TaxonomyError("task_id and operation must both be non-empty")
        if task_id in by_id:
            raise TaxonomyError(f"duplicate task_id {task_id!r}")
        folded = task_id.casefold()
        if folded in casefold_ids:
            raise TaxonomyError(
                f"case-colliding task IDs {casefold_ids[folded]!r} and {task_id!r}"
            )
        by_id[task_id] = task
        casefold_ids[folded] = task_id
        family = product_family_for_task(task, strict=True)
        previous = operation_map.setdefault(operation, family)
        if previous != family:
            raise TaxonomyError(
                f"operation {operation!r} maps to both {previous!r} and {family!r}"
            )
        split_decision(task, strict=True)
    return MappingProxyType(operation_map)


def taxonomy_payload(tasks: Iterable[Any]) -> dict[str, Any]:
    """Canonical JSON-ready taxonomy + registry assignment payload."""

    task_list = list(tasks)
    operation_map = validate_task_assignments(task_list)
    assignments = []
    for task in sorted(task_list, key=lambda item: item.task_id):
        decision = split_decision(task)
        assignments.append({
            "task_id": decision.task_id,
            "operation": str(task.operation),
            "dtype": str(task.dtype),
            "architecture": str(task.gpu_target),
            "product_family": decision.product_family,
            "analysis_family": decision.analysis_family,
            "split": decision.split,
            "reason": decision.reason,
            "provenance_root": decision.provenance_root,
        })
    return {
        "version": TAXONOMY_VERSION,
        "families": [asdict(spec) for spec in FAMILY_SPECS],
        "legacy_family_aliases": dict(sorted(LEGACY_FAMILY_ALIASES.items())),
        "train_architectures": sorted(TRAIN_ARCHITECTURES),
        "train_dtypes": sorted(TRAIN_DTYPES),
        "whole_family_holdouts": sorted(WHOLE_FAMILY_HOLDOUTS),
        "near_generalization_tasks": sorted(NEAR_GENERALIZATION_TASK_IDS),
        "generator_sources": {
            "genops": dict(sorted(GENOPS_SOURCE_FAMILIES.items())),
            "genops_operation_overrides": dict(
                sorted(GENOPS_OPERATION_OVERRIDES.items())
            ),
            "vendor": dict(sorted(vendor_operation_families().items())),
            "breadth": dict(sorted(breadth_operation_families().items())),
            "hand": dict(sorted(HAND_OPERATION_FAMILIES.items())),
        },
        "fallback_rules": [
            {"family": family, "markers": list(markers)}
            for family, markers in _NAME_RULES
        ],
        "exact_name_families": dict(sorted(_EXACT_NAME_FAMILIES.items())),
        "operation_families": dict(sorted(operation_map.items())),
        "assignments": assignments,
    }


def taxonomy_digest(tasks: Iterable[Any]) -> str:
    """SHA-256 of the versioned rules and exact live task assignments."""

    encoded = json.dumps(
        taxonomy_payload(tasks),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def describe(tasks: Iterable[Any]) -> dict[str, Any]:
    """Machine-derived taxonomy/split description for docs and reports."""

    task_list = list(tasks)
    product = Counter()
    analysis = Counter()
    reasons = Counter()
    for task in task_list:
        decision = split_decision(task)
        product[decision.product_family] += 1
        analysis[decision.analysis_family] += 1
        reasons[decision.reason] += 1
    return {
        "version": TAXONOMY_VERSION,
        "digest": taxonomy_digest(task_list),
        "tasks": len(task_list),
        "train": reasons["train"],
        "eval": len(task_list) - reasons["train"],
        "product_families": dict(sorted(product.items())),
        "analysis_rollups": dict(sorted(analysis.items())),
        "split_reasons": dict(sorted(reasons.items())),
        "whole_family_holdouts": sorted(WHOLE_FAMILY_HOLDOUTS),
        "near_generalization_tasks": sorted(NEAR_GENERALIZATION_TASK_IDS),
        "train_architectures": sorted(TRAIN_ARCHITECTURES),
        "train_dtypes": sorted(TRAIN_DTYPES),
    }


__all__ = [
    "ANALYSIS_FAMILIES",
    "BREADTH_MODULE_FAMILIES",
    "FAMILY_SPECS",
    "GENOPS_SOURCE_FAMILIES",
    "NEAR_GENERALIZATION_TASK_IDS",
    "PRIMARY_TRAIN_ARCHITECTURE",
    "PRODUCT_FAMILIES",
    "SplitDecision",
    "TAXONOMY_VERSION",
    "TRAIN_ARCHITECTURES",
    "TRAIN_DTYPES",
    "TaxonomyError",
    "WHOLE_FAMILY_HOLDOUTS",
    "analysis_family",
    "analysis_family_for_name",
    "analysis_family_for_task",
    "breadth_operation_families",
    "canonical_product_family",
    "describe",
    "family_spec",
    "mutation_family",
    "product_family_for_name",
    "product_family_for_source",
    "product_family_for_task",
    "provenance_root_for_task",
    "split_decision",
    "split_decision_for_identity",
    "taxonomy_digest",
    "taxonomy_payload",
    "validate_task_assignments",
]
