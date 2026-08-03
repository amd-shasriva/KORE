"""External task pool: trainable KORE tasks mined from outside the registry.

The registry in :mod:`kore.tasks.registry` is the *product* task set: 1,334
hand-authored and generated tasks whose exact inventory is content-addressed by
:func:`kore.tasks.taxonomy.taxonomy_digest` and pinned by contract tests.  Its
size is also the binding constraint on datagen -- generating many episodes over
~1,052 trainable tasks yields near-duplicates.

Growing the registry itself is the wrong lever.  Every added directory moves the
taxonomy digest, which is what ``validate_split_manifest`` compares a serialized
split manifest against, so an in-flight campaign's manifest would start raising
``StaleSplitManifestError``.  It also makes ``registry._discover()`` (a ~2.5 s
pass over 1,334 ``task.yaml`` files today) scale linearly in a cost every
``parallel_datagen`` worker process pays.

So an external task is a *pool* task, not a registry task.  The pool is a
separate, versioned, content-addressed corpus that:

* reuses the same on-disk task ABI (``task.yaml`` / ``reference.py`` /
  ``seed_triton.py`` / ``driver.py``), so :class:`~kore.env.kore_env.KoreEnv`
  grades a pool task through the same trusted generic driver;
* is classified by the same authority (:mod:`kore.tasks.taxonomy`), so a pool
  task's family and train/eval decision mean exactly what a registry task's do;
* never appears in ``registry.all_tasks()``, so no pinned count, digest, or
  manifest moves.

This mirrors the precedent already in the tree: :mod:`kore.openended.materialize`
writes minted tasks into a scratch root and
``CoevolutionController.resolve_task`` shadows ``registry.get_task``.

Two properties are load-bearing and are enforced here rather than documented.

**The oracle is self-contained.**  The emitted ``reference.py`` carries a JSON
spec -- module source, constructor arguments, and a measured statistical
description of every input -- and rebuilds the oracle from it.  It never calls
the upstream module's own ``get_inputs()``, so input generation is deterministic
under a seed and independent of whatever the source repository happened to do.

**Admission is fail-closed.**  A module that cannot be safely imported, cannot be
classified into a canonical product family, aliases a reserved family, or whose
``split_decision`` is anything but ``train`` is dropped with a recorded reason.
Unknown identity resolves to "not a pool task", never to a trainable one.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

POOL_SCHEMA_VERSION = "1.0"
POOL_INDEX_NAME = "pool.jsonl"
POOL_MANIFEST_NAME = "manifest.json"
POOL_TASKS_DIRNAME = "tasks"

#: Pool task IDs carry a source prefix so a pool task is never mistaken for a
#: registry task by the many call sites that filter on ``gen_``/``genb_``/``genv_``.
SOURCE_PREFIXES: Mapping[str, str] = {
    "kernelbook": "kbk",
    "synthetic": "syn",
}

#: The architecture and dtype a pool task declares.  Both must stay inside the
#: taxonomy's train sets or ``split_decision`` sends the task to eval.
POOL_GPU_TARGET = "gfx950"
POOL_BACKEND = "triton"
POOL_SEED_KERNEL = "seed_triton.py"

#: fp32 oracle gates, matching the generated-task convention in
#: ``kore.tasks.generate_ops``.
SNR_BY_DTYPE: Mapping[str, float] = {"fp32": 40.0, "bf16": 30.0, "fp16": 30.0}

#: A task whose primary shape is this small is launch-overhead-bound, so a
#: speedup measured on it says nothing about the kernel.  Upstream corpora are
#: full of toy shapes (``torch.rand([4, 4, 4, 4])`` -- 256 elements -- is
#: KernelBook's default), so this gate is what separates an optimization target
#: from a smoke test.
MIN_PRIMARY_ELEMENTS = 1 << 16

#: What the primary shape aims for.  Large enough to be bandwidth- rather than
#: launch-bound, small enough that the CPU-side ingest probe stays cheap across
#: tens of thousands of modules.
TARGET_PRIMARY_ELEMENTS = 1 << 22

#: Upper bound on elements in any single generated input, so a scaled shape
#: cannot allocate an unreasonable tensor on the eval node.
MAX_INPUT_ELEMENTS = 1 << 27

#: Scale factors for the symbolic leading dimension, largest first.
SCALE_LADDER: tuple[int, ...] = (
    8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1,
)

_INIT_SEED = 1234
_INPUT_SEED_BASE = 777


class ExternalTaskError(RuntimeError):
    """A candidate module cannot become a pool task."""


# --------------------------------------------------------------------------- #
# Import/behaviour safety
# --------------------------------------------------------------------------- #
#: Modules a mined ``nn.Module`` may import.  Anything else is refused rather
#: than sandboxed: the reference oracle is executed in the verifier subprocess,
#: so "probably harmless" is not a standard this gate can use.
ALLOWED_IMPORT_ROOTS = frozenset({
    "torch", "math", "numpy", "typing", "collections", "functools", "itertools",
    "dataclasses", "abc", "enum", "warnings", "copy", "random", "string",
    "operator", "numbers", "einops", "__future__",
})

#: Call targets that reach outside the process.  Matched on the attribute path,
#: so ``os.system`` is refused while a local variable named ``system`` is not.
FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "compile", "__import__", "open", "input", "breakpoint",
    "system", "popen", "spawn", "fork", "execv", "execve", "remove", "unlink",
    "rmtree", "rmdir", "chmod", "chown", "urlopen", "urlretrieve", "request",
    "get", "post", "connect", "socket", "loads", "load", "dump", "dumps",
})

#: Attribute paths that are always refused regardless of how they are called.
FORBIDDEN_ATTRS = frozenset({
    "os.system", "os.popen", "os.remove", "os.unlink", "os.rmdir", "os.environ",
    "subprocess.run", "subprocess.call", "subprocess.Popen",
    "shutil.rmtree", "sys.exit", "sys.modules", "pickle.loads", "pickle.load",
    "torch.load", "torch.hub", "torch.jit.load",
})

#: Non-deterministic torch entry points.  An oracle that changes between two
#: calls cannot grade a kernel, and these are not disabled by ``Module.eval()``.
NONDETERMINISTIC_CALLS = frozenset({
    "torch.rand", "torch.randn", "torch.randint", "torch.randperm",
    "torch.bernoulli", "torch.multinomial", "torch.normal", "torch.poisson",
    "torch.rand_like", "torch.randn_like", "torch.randint_like",
    "dropout", "dropout2d", "dropout3d", "alpha_dropout",
    "feature_alpha_dropout",
})

#: Functions whose bodies are the module's *input constructors*, not its
#: computation.  KernelBench-format modules build their example inputs with
#: ``torch.rand``, which is both expected and irrelevant: the constructor is run
#: once at ingest time to observe shapes and is never shipped into the task.
INPUT_CONSTRUCTOR_FUNCTIONS = frozenset({"get_inputs", "get_init_inputs"})


def _attr_path(node: ast.AST) -> str:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def module_safety_reason(source: str) -> Optional[str]:
    """Return why ``source`` may not be executed as a pool oracle, else ``None``.

    Static and conservative.  It runs at ingest time *and* again inside the
    emitted ``reference.py``, so a pool directory edited after the fact still
    cannot smuggle an unreviewed import into the verifier subprocess.
    """
    text = source or ""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return f"syntax_error: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    return f"forbidden_import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                return "forbidden_import: relative"
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                return f"forbidden_import: {node.module}"
        elif isinstance(node, ast.Call):
            path = _attr_path(node.func)
            if path in FORBIDDEN_ATTRS:
                return f"forbidden_call: {path}"
            tail = path.rsplit(".", 1)[-1]
            # A bare builtin call is refused; the same name behind an object
            # (``self.get(...)``, ``d.load(...)``) is ordinary Python.
            if "." not in path and tail in FORBIDDEN_CALLS:
                return f"forbidden_call: {tail}"
        elif isinstance(node, ast.Attribute):
            path = _attr_path(node)
            if path in FORBIDDEN_ATTRS:
                return f"forbidden_attr: {path}"
        elif isinstance(node, ast.Name) and node.id in {
            "__builtins__", "__loader__", "__spec__", "globals", "locals",
            "vars", "getattr", "setattr", "delattr",
        }:
            return f"forbidden_name: {node.id}"
    return None


def nondeterminism_reason(source: str) -> Optional[str]:
    """Return the non-deterministic call in ``source``'s computation, if any.

    Scoped to exclude the input-constructor functions, whose whole job is to draw
    random example inputs.  This is a cheap pre-filter for the real gate, which
    is the measured bit-exact repeat in
    :func:`kore.data.task_mining.probe_module`: ``Module.eval()`` already
    neutralizes ``nn.Dropout``, so a static hit here is a hint, not a verdict.
    """
    try:
        tree = ast.parse(source or "")
    except SyntaxError as exc:
        return f"syntax_error: {exc.msg}"

    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name in INPUT_CONSTRUCTOR_FUNCTIONS:
            for inner in ast.walk(node):
                skip.add(id(inner))
    for node in ast.walk(tree):
        if id(node) in skip or not isinstance(node, ast.Call):
            continue
        path = _attr_path(node.func)
        tail = path.rsplit(".", 1)[-1]
        if path in NONDETERMINISTIC_CALLS or (
            tail in NONDETERMINISTIC_CALLS and path.split(".")[0] in
            {"torch", "F", "nn", "functional", "tf"}
        ):
            return f"nondeterministic_call: {path}"
    return None


# --------------------------------------------------------------------------- #
# Family classification from measured operator evidence
# --------------------------------------------------------------------------- #
#: torch/`nn` call names that the KORE taxonomy has no name rule for, mapped to a
#: canonical product family.  This is an adapter for an external vocabulary, not
#: a second taxonomy: every value is validated through
#: :func:`~kore.tasks.taxonomy.canonical_product_family`, and
#: :func:`classify_module` consults the taxonomy's own
#: :func:`~kore.tasks.taxonomy.product_family_for_name` first.
TORCH_OP_FAMILIES: Mapping[str, str] = {
    # gemm
    "Linear": "gemm", "linear": "gemm", "Bilinear": "gemm", "LazyLinear": "gemm",
    "matmul": "gemm", "mm": "gemm", "bmm": "gemm", "addmm": "gemm",
    "baddbmm": "gemm", "einsum": "gemm", "tensordot": "gemm", "outer": "gemm",
    "addbmm": "gemm", "chain_matmul": "gemm",
    # convolution / pooling
    "Conv1d": "convolution", "Conv2d": "convolution", "Conv3d": "convolution",
    "ConvTranspose1d": "convolution", "ConvTranspose2d": "convolution",
    "ConvTranspose3d": "convolution", "LazyConv2d": "convolution",
    "Unfold": "convolution", "Fold": "convolution", "unfold": "convolution",
    # normalization
    "LayerNorm": "normalization", "BatchNorm1d": "normalization",
    "BatchNorm2d": "normalization", "BatchNorm3d": "normalization",
    "GroupNorm": "normalization", "InstanceNorm1d": "normalization",
    "InstanceNorm2d": "normalization", "InstanceNorm3d": "normalization",
    "LocalResponseNorm": "normalization", "RMSNorm": "normalization",
    "normalize": "normalization",
    # attention
    "MultiheadAttention": "attention",
    "scaled_dot_product_attention": "attention",
    # reduction / losses
    "softmax": "reduction", "log_softmax": "reduction", "Softmax": "reduction",
    "LogSoftmax": "reduction", "logsumexp": "reduction", "sum": "reduction",
    "mean": "reduction", "var": "reduction", "std": "reduction",
    "prod": "reduction", "amax": "reduction", "amin": "reduction",
    "cross_entropy": "reduction", "nll_loss": "reduction", "kl_div": "reduction",
    "binary_cross_entropy": "reduction", "mse_loss": "reduction",
    "CrossEntropyLoss": "reduction", "MSELoss": "reduction", "BCELoss": "reduction",
    "BCEWithLogitsLoss": "reduction", "NLLLoss": "reduction", "L1Loss": "reduction",
    "SmoothL1Loss": "reduction", "KLDivLoss": "reduction", "norm": "reduction",
    "Softmin": "reduction",
    # sequence / scan
    "cumsum": "sequence", "cumprod": "sequence", "LSTM": "sequence",
    "GRU": "sequence", "RNN": "sequence", "LSTMCell": "sequence",
    "GRUCell": "sequence", "cummax": "sequence", "cummin": "sequence",
    # sparse / sort
    "sort": "sparse", "argsort": "sparse", "topk": "reduction",
    "Embedding": "data_movement", "embedding": "data_movement",
    "EmbeddingBag": "data_movement", "gather": "data_movement",
    "scatter": "data_movement", "index_select": "data_movement",
    "scatter_add": "data_movement", "take": "data_movement",
    # quantization
    "quantize_per_tensor": "quantization", "dequantize": "quantization",
    "fake_quantize_per_tensor_affine": "quantization",
    # activation
    "ReLU": "activation", "ReLU6": "activation", "LeakyReLU": "activation",
    "GELU": "activation", "SiLU": "activation", "Mish": "activation",
    "ELU": "activation", "SELU": "activation", "CELU": "activation",
    "PReLU": "activation", "RReLU": "activation", "Hardswish": "activation",
    "Hardsigmoid": "activation", "Hardtanh": "activation", "Softplus": "activation",
    "Softsign": "activation", "Tanhshrink": "activation", "Softshrink": "activation",
    "Hardshrink": "activation", "GLU": "activation", "glu": "activation",
    "relu": "activation", "gelu": "activation", "silu": "activation",
    "leaky_relu": "activation", "elu": "activation", "selu": "activation",
    "hardswish": "activation", "hardtanh": "activation", "softplus": "activation",
    "sigmoid": "activation", "tanh": "activation", "Sigmoid": "activation",
    "Tanh": "activation", "exp": "activation", "log": "activation",
    "sqrt": "activation", "rsqrt": "activation", "abs": "activation",
    "erf": "activation", "clamp": "activation", "pow": "activation",
    # positional
    "rotary_embedding": "positional",
    # data movement / pointwise
    "cat": "data_movement", "stack": "data_movement", "permute": "data_movement",
    "transpose": "data_movement", "reshape": "data_movement",
    "view": "data_movement", "repeat_interleave": "data_movement",
    "pad": "data_movement", "roll": "data_movement", "flip": "data_movement",
    "add": "elementwise", "mul": "elementwise", "sub": "elementwise",
    "div": "elementwise", "maximum": "elementwise", "minimum": "elementwise",
    "where": "elementwise",
    # pooling reads as convolution's analysis parent in the taxonomy
    "MaxPool1d": "convolution", "MaxPool2d": "convolution", "MaxPool3d": "convolution",
    "AvgPool1d": "convolution", "AvgPool2d": "convolution", "AvgPool3d": "convolution",
    "AdaptiveAvgPool1d": "convolution", "AdaptiveAvgPool2d": "convolution",
    "AdaptiveMaxPool2d": "convolution", "max_pool2d": "convolution",
    "avg_pool2d": "convolution", "interpolate": "convolution",
    "adaptive_avg_pool2d": "convolution", "PixelShuffle": "data_movement",
}

#: Which family wins when a module shows evidence for several.  Ordered from
#: most structurally distinctive to least, so a transformer block classifies as
#: ``attention`` rather than as the ``activation`` its GELU also implies.
FAMILY_PRECEDENCE: tuple[str, ...] = (
    "attention", "moe", "convolution", "gemm", "normalization", "sequence",
    "sampling", "quantization", "sparse", "reduction", "positional",
    "data_movement", "fusion", "activation", "elementwise",
)

#: Markers whose presence anywhere in a module means it belongs to a reserved
#: leaf.  Checked on raw source, because a reserved family must not depend on the
#: classifier noticing the right call.
RESERVED_SOURCE_MARKERS: tuple[str, ...] = (
    "mla", "multi_head_latent", "multihead_latent", "latent_attn",
    "latent_attention", "flashmla", "paged_attn", "paged_attention",
    "paged_kv", "pagedkv", "kv_cache_page", "block_table",
)


def detected_operators(source: str) -> tuple[str, ...]:
    """Distinct torch/`nn` call and attribute names used by ``source``."""
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return ()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            path = _attr_path(node.func)
            if path:
                names.add(path.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return tuple(sorted(names))


def reserved_family_marker(source: str, name: str = "") -> Optional[str]:
    """The reserved-leaf marker ``source``/``name`` matches, if any.

    Word-boundary matched so ``mla`` does not fire on ``mlanguage``; the point is
    to keep an MLA or paged-KV variant out of training by family, not to reject
    every identifier containing those letters.
    """
    haystack = f"{name}\n{source or ''}".lower()
    for marker in RESERVED_SOURCE_MARKERS:
        if re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", haystack):
            return marker
    return None


def classify_module(source: str, name: str = "") -> Optional[str]:
    """Canonical product family for a mined module, or ``None`` if unclassifiable.

    Evidence is the set of torch operators the module actually calls, not its
    class name: upstream class names are arbitrary (``Net``, ``Actor``,
    ``Block``), while the operators are the semantics.  Each detected operator is
    resolved through the taxonomy's own name adapter first and only then through
    :data:`TORCH_OP_FAMILIES`, so the two can never disagree about a name the
    taxonomy already knows.
    """
    from kore.tasks import taxonomy

    if reserved_family_marker(source, name):
        return None

    votes: set[str] = set()
    for op in detected_operators(source):
        family = TORCH_OP_FAMILIES.get(op)
        if family is None:
            inferred = taxonomy.product_family_for_name(op)
            family = inferred
        if family in taxonomy.WHOLE_FAMILY_HOLDOUTS:
            return None
        if family:
            votes.add(taxonomy.canonical_product_family(family))
    for family in FAMILY_PRECEDENCE:
        if family in votes:
            return family
    return None


def _slug(value: str) -> str:
    """A lowercase, taxonomy-safe operation stem."""
    text = re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", str(value or ""))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", text) or "module"


def make_identity(
    source_id: str,
    module_name: str,
    module_source: str,
    dtype: str,
) -> tuple[str, str]:
    """Return ``(operation, task_id)`` for one mined module.

    The identity is content-addressed on the module source, so re-running the
    ingest over the same upstream revision reproduces the same IDs and a
    structurally different module can never collide onto an existing one.
    """
    prefix = SOURCE_PREFIXES.get(source_id, "ext")
    digest = hashlib.sha256((module_source or "").encode("utf-8")).hexdigest()[:8]
    operation = f"{prefix}_{_slug(module_name)}_{digest}"
    return operation, f"{operation}_{dtype}"


# --------------------------------------------------------------------------- #
# The spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InputSpec:
    """A measured statistical description of one forward-pass input.

    The upstream ``get_inputs()`` is *not* shipped into the task.  It is run once
    at ingest time, and what survives is this description: shape, dtype, and the
    distribution the observed tensor came from.  Regenerating from the
    description keeps input synthesis deterministic under a seed and lets the
    leading dimension be scaled to a shape worth optimizing.
    """

    shape: tuple[int, ...]
    dtype: str
    kind: str = "uniform"      # uniform | normal | integer | bool
    low: float = 0.0
    high: float = 1.0
    mean: float = 0.0
    std: float = 1.0
    scalable: bool = True

    def sized(self, scale: int) -> tuple[int, ...]:
        if not self.scalable or not self.shape:
            return tuple(self.shape)
        return (max(1, int(self.shape[0]) * int(scale)),) + tuple(self.shape[1:])

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InputSpec":
        return cls(
            shape=tuple(int(x) for x in data.get("shape", ())),
            dtype=str(data.get("dtype", "float32")),
            kind=str(data.get("kind", "uniform")),
            low=float(data.get("low", 0.0)),
            high=float(data.get("high", 1.0)),
            mean=float(data.get("mean", 0.0)),
            std=float(data.get("std", 1.0)),
            scalable=bool(data.get("scalable", True)),
        )


@dataclass(frozen=True)
class ExternalTaskSpec:
    """Everything needed to rebuild one pool task's oracle from scratch."""

    task_id: str
    operation: str
    dtype: str
    family: str
    entry_class: str
    entry_name: str
    module_source: str
    init_args: tuple[Any, ...] = ()
    init_kwargs: Mapping[str, Any] = field(default_factory=dict)
    input_specs: tuple[InputSpec, ...] = ()
    primary_scale: int = 1
    validation_scales: tuple[int, ...] = ()
    snr_threshold: float = 40.0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        from kore.data.dedup import content_hash

        return content_hash(self.module_source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POOL_SCHEMA_VERSION,
            "task_id": self.task_id,
            "operation": self.operation,
            "dtype": self.dtype,
            "family": self.family,
            "entry_class": self.entry_class,
            "entry_name": self.entry_name,
            "module_source": self.module_source,
            "init_args": list(self.init_args),
            "init_kwargs": dict(self.init_kwargs),
            "input_specs": [spec.to_dict() for spec in self.input_specs],
            "primary_scale": int(self.primary_scale),
            "validation_scales": list(self.validation_scales),
            "snr_threshold": float(self.snr_threshold),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExternalTaskSpec":
        return cls(
            task_id=str(data["task_id"]),
            operation=str(data["operation"]),
            dtype=str(data["dtype"]),
            family=str(data["family"]),
            entry_class=str(data["entry_class"]),
            entry_name=str(data["entry_name"]),
            module_source=str(data["module_source"]),
            init_args=tuple(data.get("init_args", ())),
            init_kwargs=dict(data.get("init_kwargs", {})),
            input_specs=tuple(
                InputSpec.from_dict(item) for item in data.get("input_specs", ())
            ),
            primary_scale=int(data.get("primary_scale", 1)),
            validation_scales=tuple(int(x) for x in data.get("validation_scales", ())),
            snr_threshold=float(data.get("snr_threshold", 40.0)),
            provenance=dict(data.get("provenance", {})),
        )


def split_decision_for_spec(spec: ExternalTaskSpec):
    """The authoritative train/eval decision for a pool task.

    Delegates to the same classifier the registry uses, so "trainable" means the
    same thing for a pool task and a registry task.
    """
    from kore.tasks import taxonomy

    return taxonomy.split_decision_for_identity(
        task_id=spec.task_id,
        operation=spec.operation,
        product_family=spec.family,
        architecture=POOL_GPU_TARGET,
        dtype=spec.dtype,
        provenance_root=spec.task_id,
    )


# --------------------------------------------------------------------------- #
# Oracle reconstruction (runs inside the driver subprocess)
# --------------------------------------------------------------------------- #
_TORCH_DTYPES = {
    "fp32": "float32", "fp16": "float16", "bf16": "bfloat16",
    "float32": "float32", "float16": "float16", "bfloat16": "bfloat16",
}


def _torch_dtype(name: str):
    import torch

    return getattr(torch, _TORCH_DTYPES.get(str(name), str(name)))


def exec_module_source(source: str) -> dict:
    """Execute a safety-checked module source and return its namespace."""
    reason = module_safety_reason(source)
    if reason is not None:
        raise ExternalTaskError(f"refusing to execute pool module: {reason}")
    namespace: dict[str, Any] = {"__name__": "kore_pool_module"}
    exec(compile(source, "<kore-pool-module>", "exec"), namespace)  # noqa: S102
    return namespace


def build_inputs(
    input_specs: Sequence[InputSpec],
    scale: int,
    device: str = "cuda",
    seed: int = 0,
    dtype: str = "fp32",
):
    """Deterministically synthesize the forward-pass inputs at ``scale``."""
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + _INPUT_SEED_BASE)
    task_dtype = _torch_dtype(dtype)
    out = []
    for spec in input_specs:
        shape = spec.sized(scale)
        if spec.kind == "integer":
            low, high = int(spec.low), max(int(spec.low) + 1, int(spec.high) + 1)
            tensor = torch.randint(
                low, high, shape, generator=generator, dtype=torch.int64
            )
            tensor = tensor.to(getattr(torch, spec.dtype))
        elif spec.kind == "bool":
            tensor = torch.rand(shape, generator=generator) < max(
                0.0, min(1.0, spec.mean)
            )
        elif spec.kind == "normal":
            tensor = torch.randn(shape, generator=generator) * spec.std + spec.mean
            tensor = tensor.to(task_dtype)
        else:
            span = max(spec.high - spec.low, 1e-6)
            tensor = torch.rand(shape, generator=generator) * span + spec.low
            tensor = tensor.to(task_dtype)
        out.append(tensor.to(device))
    return tuple(out)


def reference_namespace_from_spec(spec: Mapping[str, Any]) -> dict:
    """Rebuild the ``_genops``-style reference namespace from a pool spec.

    Called by the emitted ``reference.py`` inside the driver subprocess, and by
    the materialize-time self-check in-process, so the two can never diverge.
    """
    import torch

    task = ExternalTaskSpec.from_dict(spec)
    namespace = exec_module_source(task.module_source)
    cls = namespace.get(task.entry_class)
    if cls is None:
        raise ExternalTaskError(f"module defines no class {task.entry_class!r}")

    task_dtype = _torch_dtype(task.dtype)
    models: dict[tuple[str, str], Any] = {}

    def model_for(device, dtype):
        key = (str(device), str(dtype))
        if key not in models:
            torch.manual_seed(_INIT_SEED)
            model = cls(*task.init_args, **dict(task.init_kwargs))
            model = model.to(device=device, dtype=dtype)
            # ``eval()`` is what makes the oracle a function: it disables dropout
            # and switches batch norm to its running statistics.  Without it the
            # same inputs would produce a different answer on every call.
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            models[key] = model
        return models[key]

    def parse_shape(shape_str):
        if not shape_str or shape_str == "default":
            return {"S": int(task.primary_scale)}
        out: dict[str, int] = {}
        for item in str(shape_str).split(","):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            out[key.strip()] = int(value)
        return out or {"S": int(task.primary_scale)}

    def get_inputs(shape, device="cuda", seed=0):
        scale = int(shape.get("S", task.primary_scale)) if isinstance(shape, dict) \
            else int(task.primary_scale)
        return build_inputs(
            task.input_specs, scale, device=device, seed=seed, dtype=task.dtype
        )

    def _up(value):
        if torch.is_tensor(value) and value.is_floating_point():
            return value.float()
        return value

    def _down(value):
        if torch.is_tensor(value) and value.is_floating_point():
            return value.to(task_dtype)
        if isinstance(value, (tuple, list)):
            return type(value)(_down(item) for item in value)
        return value

    def ref_fn(*inputs):
        device = next(
            (x.device for x in inputs if torch.is_tensor(x)), torch.device("cpu")
        )
        model = model_for(device, torch.float32)
        with torch.no_grad():
            return _down(model(*[_up(x) for x in inputs]))

    def baseline_fn(*inputs):
        device = next(
            (x.device for x in inputs if torch.is_tensor(x)), torch.device("cpu")
        )
        model = model_for(device, task_dtype)
        with torch.no_grad():
            return _down(model(*inputs))

    return {
        "parse_shape": parse_shape,
        "get_inputs": get_inputs,
        "ref_fn": ref_fn,
        "baseline_fn": baseline_fn,
        "arity": len(task.input_specs),
        "entry_name": task.entry_name,
        "dtype_name": task.dtype,
        # Declared for the publication paired-timing protocol; a mined module is
        # a pure function of its inputs, and the safety gate refuses in-place
        # torch entry points, so nothing here mutates an argument.
        "mutates_input": False,
        # Deliberately NOT one of ``_genops._GENERIC_ADV_FAMILIES``: the generic
        # adversarial fills assume plain float inputs, which a mined module with
        # index or mask inputs does not have.
        "family": f"external_{task.family}",
    }


# --------------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------------- #
_DRIVER_SHIM = '''"""GENERATED driver shim for an EXTERNAL POOL task.
See kore/tasks/external.py. Do not hand-edit."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import reference as ref  # noqa: E402
from kore.tasks._genops import driver_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(driver_main(ref, _here))
'''


def _embedded_spec(spec: ExternalTaskSpec) -> str:
    """The spec as a Python literal that reconstructs it exactly.

    Embedded as a JSON *string* decoded at import, not as a JSON object pasted
    into the file: JSON's ``true``/``false``/``null`` are not Python literals, so
    a directly-pasted object raises ``NameError`` the moment a spec grows a
    boolean field.
    """
    return f"_SPEC = json.loads({json.dumps(json.dumps(spec.to_dict(), sort_keys=True))})"


def reference_source(spec: ExternalTaskSpec) -> str:
    return (
        '"""GENERATED reference for an EXTERNAL POOL task.\n'
        'See kore/tasks/external.py. Do not hand-edit."""\n'
        "import json\n\n"
        "from kore.tasks.external import reference_namespace_from_spec\n\n"
        f"{_embedded_spec(spec)}\n"
        "globals().update(reference_namespace_from_spec(_SPEC))\n"
    )


def seed_source(spec: ExternalTaskSpec) -> str:
    """The seed is the torch baseline under the task's entry-point name.

    A mined module has no reference Triton kernel for this architecture, so a
    from-scratch prompt would be an unanswerable task.  Aliasing the correct
    eager implementation makes it the well-posed "make this fast" problem the
    edit-trained policy can attempt, starting at ~1x.
    """
    return (
        '"""GENERATED seed for an EXTERNAL POOL task: the correct torch baseline.\n'
        'See kore/tasks/external.py. Do not hand-edit."""\n'
        "import json\n\n"
        "import torch  # noqa: F401 (available in the eval env)\n"
        "from kore.tasks.external import reference_namespace_from_spec\n\n"
        f"{_embedded_spec(spec)}\n"
        "_NS = reference_namespace_from_spec(_SPEC)\n"
        f"{spec.entry_name} = _NS['baseline_fn']\n"
    )


def task_yaml(spec: ExternalTaskSpec) -> str:
    """The task metadata, as JSON (which ``yaml.safe_load`` reads directly)."""
    scales = [int(s) for s in spec.validation_scales]
    meta = {
        "task_id": spec.task_id,
        "operation": spec.operation,
        "dtype": spec.dtype,
        "backend": POOL_BACKEND,
        "gpu_target": POOL_GPU_TARGET,
        "seed_kernel_name": POOL_SEED_KERNEL,
        "snr_threshold": float(spec.snr_threshold),
        "op_family": spec.family,
        "taxonomy_family": spec.family,
        "provenance_root": spec.task_id,
        "baseline_tier": "external_pool",
        # ``minted`` routes classification through
        # ``taxonomy.product_family_for_source("minted", ...)``, which refuses to
        # let a declared family override a reserved one.
        "minted": True,
        "external_pool": True,
        "provenance": dict(spec.provenance),
        "shapes": {
            "minimal": {"S": 1},
            "primary": {"S": int(spec.primary_scale)},
            **({"validation": [{"S": s} for s in scales]} if scales else {}),
        },
        "targets": {
            "snr_db": float(spec.snr_threshold),
            "comparison_baseline": f"torch_{spec.operation}",
        },
    }
    return json.dumps(meta, indent=2)


def materialize_external_task(spec: ExternalTaskSpec, root: str | os.PathLike):
    """Write one pool task directory and return its :class:`Task`.

    Raises :class:`ExternalTaskError` rather than returning a partially written
    directory, so a caller cannot mistake a failed emission for a usable task.
    """
    from kore.tasks.base import Task

    task_dir = Path(root) / spec.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "reference.py").write_text(reference_source(spec), encoding="utf-8")
    (task_dir / "driver.py").write_text(_DRIVER_SHIM, encoding="utf-8")
    (task_dir / POOL_SEED_KERNEL).write_text(seed_source(spec), encoding="utf-8")
    (task_dir / "task.yaml").write_text(task_yaml(spec), encoding="utf-8")
    try:
        return Task.from_dir(task_dir)
    except Exception as exc:  # noqa: BLE001
        raise ExternalTaskError(f"{spec.task_id}: emitted task is unreadable: {exc}")


# --------------------------------------------------------------------------- #
# The pool on disk
# --------------------------------------------------------------------------- #
def pool_root(root: Optional[str | os.PathLike] = None) -> Path:
    """Where the pool lives: explicit argument, ``KORE_TASK_POOL``, or default."""
    if root:
        return Path(root)
    env = os.environ.get("KORE_TASK_POOL")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "task_pool"


def write_pool_index(specs: Iterable[ExternalTaskSpec], path: str | os.PathLike) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(spec.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def read_pool_index(path: str | os.PathLike) -> list[ExternalTaskSpec]:
    specs: list[ExternalTaskSpec] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                specs.append(ExternalTaskSpec.from_dict(json.loads(line)))
    return specs


def load_pool_specs(root: Optional[str | os.PathLike] = None) -> list[ExternalTaskSpec]:
    index = pool_root(root) / POOL_INDEX_NAME
    if not index.is_file():
        return []
    return read_pool_index(index)


def load_pool_manifest(root: Optional[str | os.PathLike] = None) -> dict:
    path = pool_root(root) / POOL_MANIFEST_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pool_tasks_dir(root: Optional[str | os.PathLike] = None) -> Path:
    return pool_root(root) / POOL_TASKS_DIRNAME


def materialize_pool(
    root: Optional[str | os.PathLike] = None,
    specs: Optional[Iterable[ExternalTaskSpec]] = None,
) -> list[str]:
    """Write every indexed spec into ``<root>/tasks/``; return the task IDs."""
    base = pool_root(root)
    items = list(specs) if specs is not None else load_pool_specs(base)
    target = pool_tasks_dir(base)
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for spec in items:
        materialize_external_task(spec, target)
        written.append(spec.task_id)
    return written


def load_pool(root: Optional[str | os.PathLike] = None) -> list:
    """Every materialized pool task, as :class:`~kore.tasks.base.Task` objects."""
    from kore.tasks.base import Task

    tasks = []
    directory = pool_tasks_dir(root)
    if not directory.is_dir():
        return tasks
    for yml in sorted(directory.glob("*/task.yaml")):
        tasks.append(Task.from_dir(yml.parent))
    return tasks


def pool_train_task_ids(root: Optional[str | os.PathLike] = None) -> tuple[str, ...]:
    """Pool task IDs the authoritative split marks trainable.

    The index is already decontaminated at build time; this re-derives the
    decision at read time so a hand-edited index cannot widen the train set.
    """
    return tuple(
        spec.task_id
        for spec in load_pool_specs(root)
        if split_decision_for_spec(spec).split == "train"
    )


def resolve_task(task_id: str, root: Optional[str | os.PathLike] = None):
    """Resolve a task ID against the pool first, then the registry.

    Mirrors ``kore.openended.controller.CoevolutionController.resolve_task`` so a
    caller can serve pool and registry tasks through one lookup.
    """
    directory = pool_tasks_dir(root) / str(task_id)
    if (directory / "task.yaml").is_file():
        from kore.tasks.base import Task

        return Task.from_dir(directory)
    from kore.tasks.registry import get_task

    return get_task(task_id)


__all__ = [
    "ALLOWED_IMPORT_ROOTS",
    "ExternalTaskError",
    "ExternalTaskSpec",
    "FAMILY_PRECEDENCE",
    "InputSpec",
    "INPUT_CONSTRUCTOR_FUNCTIONS",
    "MAX_INPUT_ELEMENTS",
    "MIN_PRIMARY_ELEMENTS",
    "NONDETERMINISTIC_CALLS",
    "POOL_GPU_TARGET",
    "TARGET_PRIMARY_ELEMENTS",
    "POOL_INDEX_NAME",
    "POOL_MANIFEST_NAME",
    "POOL_SCHEMA_VERSION",
    "SCALE_LADDER",
    "SNR_BY_DTYPE",
    "SOURCE_PREFIXES",
    "TORCH_OP_FAMILIES",
    "build_inputs",
    "classify_module",
    "detected_operators",
    "exec_module_source",
    "load_pool",
    "load_pool_manifest",
    "load_pool_specs",
    "make_identity",
    "materialize_external_task",
    "materialize_pool",
    "module_safety_reason",
    "nondeterminism_reason",
    "pool_root",
    "pool_tasks_dir",
    "pool_train_task_ids",
    "read_pool_index",
    "reference_namespace_from_spec",
    "reference_source",
    "reserved_family_marker",
    "resolve_task",
    "seed_source",
    "split_decision_for_spec",
    "task_yaml",
    "write_pool_index",
]
