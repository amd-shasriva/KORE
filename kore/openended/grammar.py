"""A TYPED composition grammar over verified torch primitives (the minter's oracle
builder).

The open-ended minter needs to synthesize *net-new*, correct-by-construction KORE
tasks. This module supplies the "correct-by-construction" half: a small typed IR
whose leaves are VERIFIED torch primitives (relu / matmul / rmsnorm / ...) and
whose composition rule is type-checked, so any well-typed :class:`Pipeline`
denotes a pure torch function that IS the task's reference oracle. There is no
separate spec to drift from - the composed torch function is the spec.

Design (mirrors ``kore.tasks._genops`` conventions):

  * A value flowing through a pipeline has a coarse tensor TYPE
    (:data:`MATRIX` ``[M,N]`` or :data:`ROWVEC` ``[M]``). Each :class:`Primitive`
    declares the type it consumes and the type it produces plus the auxiliary
    inputs it samples (a matmul weight, a bias, a residual, a norm scale). The
    :class:`Pipeline` type-checker rejects ill-formed chains (two sources, a
    reduction that is not terminal, a type mismatch), so composition is sound.

  * ``reference_fn`` folds the primitives in fp32 and casts to the task dtype -
    exactly the ``ref_fn`` convention in ``_genops.make_reference`` (compute in
    float, store in the task dtype), so a minted reference grades identically to a
    hand-written generated op.

  * ``input_sampler`` draws seeded inputs per aux role with the same 1/sqrt(K)
    GEMM scaling ``_genops`` uses, so magnitudes stay ~O(1) and the reference is
    well-conditioned (a precondition for the minter's non-degeneracy gate).

Everything is pure and CPU-only. ``torch`` is imported lazily (inside the
primitive tables / builders) so importing this module never needs torch or a GPU;
only building a reference / sampling touches torch (CPU is fine).

Grammar EVOLUTION (escaping the bounded encoding)
-------------------------------------------------
The primitive libraries above are a FIXED, bounded space; a minter that only
composes them with fixed-depth templates hits the classic POET/OMNI "bounded
encoding" ceiling. The :class:`Production` layer at the bottom of this module
lifts that ceiling *without* weakening correct-by-construction: a
:class:`Production` is an evolvable, well-typed composition operator whose body is
always a flat tuple of the SAME name-addressable primitives. Productions compose
from other productions (:func:`compose_productions`) - a self-referential grammar
- so the reachable pipeline space is unbounded, yet every emitted pipeline is
still a plain tuple of verified primitives that type-checks, behavioral-hashes,
niches, and materializes exactly like a hand-composed one. Correctness is enforced
per task by the minter's construction gate, so it is preserved by CONSTRUCTION.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Optional

# --------------------------------------------------------------------------- #
# Tensor types (the grammar's type system)
# --------------------------------------------------------------------------- #
MATRIX = "matrix"   # a [M, N] activation tensor (the main data flowing through)
ROWVEC = "rowvec"   # a [M] per-row reduction result (a terminal type)

# Aux-input ROLES a primitive can request from the sampler. The sampler maps each
# role to a concrete seeded tensor with a magnitude that keeps the op stable.
ROLE_MATRIX = "matrix"        # [M, N]  (primary input / residual / gate operand)
ROLE_MATRIX_MK = "matrix_mk"  # [M, K]  (matmul lhs)
ROLE_WEIGHT_KN = "weight_kn"  # [K, N]  (matmul rhs)
ROLE_BIAS_N = "bias_n"        # [N]     (broadcast bias)
ROLE_WEIGHT_N = "weight_n"    # [N]     (affine/norm scale, ~1)

_NORM_EPS = 1.0e-5


def _lazy():
    import torch
    import torch.nn.functional as F
    return torch, F


# --------------------------------------------------------------------------- #
# Primitive
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Primitive:
    """One verified torch op node in the grammar.

    ``in_type`` is ``None`` for a SOURCE (it produces a value from its aux inputs
    alone, e.g. an input tensor or a matmul); otherwise it consumes the current
    value of that type and produces ``out_type``. ``apply(main, auxs, torch, F)``
    computes the node in fp32 (``main`` is ``None`` for a source). ``aux_roles``
    lists the auxiliary tensors the sampler must provide, in call order.
    """

    name: str
    in_type: Optional[str]
    out_type: str
    apply: Callable                    # (main|None, auxs_tuple, torch, F) -> fp32 tensor
    aux_roles: tuple = ()
    flops_per_elem: float = 1.0        # ~fp32 FLOPs per OUTPUT element (roofline proxy)
    tag: str = "op"                    # coarse class: source|act|binary|bias|norm|reduce|gemm|scale


# --------------------------------------------------------------------------- #
# Primitive libraries (built lazily so importing this module needs no torch)
# --------------------------------------------------------------------------- #
def source_prims() -> dict[str, Primitive]:
    """Pipeline heads: an input matrix, or a GEMM (the compute-bound source)."""
    def _input(main, auxs, t, F):
        return auxs[0]

    def _matmul(main, auxs, t, F):
        return auxs[0] @ auxs[1]

    return {
        "input": Primitive("input", None, MATRIX, _input,
                           aux_roles=(ROLE_MATRIX,), flops_per_elem=0.0, tag="source"),
        "matmul": Primitive("matmul", None, MATRIX, _matmul,
                            aux_roles=(ROLE_MATRIX_MK, ROLE_WEIGHT_KN),
                            flops_per_elem=0.0, tag="gemm"),  # flops handled specially (needs K)
    }


def _act_table(t, F):
    """name -> (fn, ~flops/elem). Bounded on randn inputs (stay finite when composed)."""
    return {
        "relu": (t.relu, 1.0),
        "relu6": (F.relu6, 2.0),
        "leaky_relu": (lambda x: F.leaky_relu(x, 0.01), 2.0),
        "silu": (F.silu, 4.0),
        "sigmoid": (t.sigmoid, 4.0),
        "tanh": (t.tanh, 4.0),
        "gelu": (lambda x: F.gelu(x, approximate="tanh"), 8.0),
        "gelu_erf": (F.gelu, 9.0),
        "softsign": (F.softsign, 2.0),
        "elu": (F.elu, 3.0),
        "softplus": (F.softplus, 4.0),
        "mish": (F.mish, 9.0),
        "hardswish": (F.hardswish, 3.0),
        "hardsigmoid": (F.hardsigmoid, 2.0),
        "square": (t.square, 1.0),
        "abs": (t.abs, 1.0),
        "neg": (t.neg, 1.0),
    }


def act_names() -> tuple:
    t, F = _lazy()
    return tuple(_act_table(t, F))


def middle_prims() -> dict[str, Primitive]:
    """MATRIX -> MATRIX transforms: activations, binaries, bias, gates, norms, scale."""
    t, F = _lazy()
    prims: dict[str, Primitive] = {}

    # unary activations
    for name, (fn, cost) in _act_table(t, F).items():
        def _mk_act(fn):
            return lambda main, auxs, t, F: fn(main)
        prims[name] = Primitive(name, MATRIX, MATRIX, _mk_act(fn),
                                aux_roles=(), flops_per_elem=cost, tag="act")

    # elementwise binaries (2nd operand is a fresh sampled matrix -> real fusion)
    binaries = {
        "add": (lambda a, b: a + b, 1.0),
        "mul": (lambda a, b: a * b, 1.0),
        "sub": (lambda a, b: a - b, 1.0),
        "add_relu": (lambda a, b: t.relu(a + b), 2.0),
        "silu_mul": (lambda a, b: F.silu(a) * b, 5.0),   # SwiGLU-style gate
        "gelu_mul": (lambda a, b: F.gelu(a, approximate="tanh") * b, 9.0),
        "sigmoid_mul": (lambda a, b: t.sigmoid(a) * b, 5.0),
    }
    for name, (fn, cost) in binaries.items():
        def _mk_bin(fn):
            return lambda main, auxs, t, F: fn(main, auxs[0])
        prims[name] = Primitive(name, MATRIX, MATRIX, _mk_bin(fn),
                                aux_roles=(ROLE_MATRIX,), flops_per_elem=cost, tag="binary")

    # bias add ([N] broadcast) + scalar scales
    prims["add_bias"] = Primitive(
        "add_bias", MATRIX, MATRIX,
        lambda main, auxs, t, F: main + auxs[0], aux_roles=(ROLE_BIAS_N,),
        flops_per_elem=1.0, tag="bias")
    for c in (0.5, 2.0):
        def _mk_scale(c):
            return lambda main, auxs, t, F: main * c
        nm = f"scale{str(c).replace('.', '_')}"
        prims[nm] = Primitive(nm, MATRIX, MATRIX, _mk_scale(c),
                              aux_roles=(), flops_per_elem=1.0, tag="scale")

    # normalizations (last-dim), the canonical fused reduction+affine chains
    def _rmsnorm(main, auxs, t, F):
        w = auxs[0]
        return main * t.rsqrt(main.pow(2).mean(-1, keepdim=True) + _NORM_EPS) * w

    def _layernorm(main, auxs, t, F):
        w, b = auxs
        mu = main.mean(-1, keepdim=True)
        var = main.var(-1, unbiased=False, keepdim=True)
        return (main - mu) * t.rsqrt(var + _NORM_EPS) * w + b

    def _softmax(main, auxs, t, F):
        return t.softmax(main, dim=-1)

    def _l2norm(main, auxs, t, F):
        return main / (main.norm(p=2, dim=-1, keepdim=True) + _NORM_EPS)

    prims["rmsnorm"] = Primitive("rmsnorm", MATRIX, MATRIX, _rmsnorm,
                                 aux_roles=(ROLE_WEIGHT_N,), flops_per_elem=4.0, tag="norm")
    prims["layernorm"] = Primitive("layernorm", MATRIX, MATRIX, _layernorm,
                                   aux_roles=(ROLE_WEIGHT_N, ROLE_BIAS_N),
                                   flops_per_elem=6.0, tag="norm")
    prims["softmax"] = Primitive("softmax", MATRIX, MATRIX, _softmax,
                                 aux_roles=(), flops_per_elem=5.0, tag="norm")
    prims["l2norm"] = Primitive("l2norm", MATRIX, MATRIX, _l2norm,
                                aux_roles=(), flops_per_elem=3.0, tag="norm")
    return prims


def terminal_prims() -> dict[str, Primitive]:
    """MATRIX -> ROWVEC per-row reductions (terminal: nothing consumes ROWVEC)."""
    t, F = _lazy()
    reducers = {
        "row_sum": (lambda x: x.sum(-1), 1.0),
        "row_mean": (lambda x: x.mean(-1), 1.0),
        "row_max": (lambda x: x.amax(-1), 1.0),
        "row_l2": (lambda x: x.norm(p=2, dim=-1), 2.0),
        "row_rms": (lambda x: x.pow(2).mean(-1).sqrt(), 2.0),
        "row_l1": (lambda x: x.abs().sum(-1), 1.0),
    }
    prims: dict[str, Primitive] = {}
    for name, (fn, cost) in reducers.items():
        def _mk_red(fn):
            return lambda main, auxs, t, F: fn(main)
        prims[name] = Primitive(name, MATRIX, ROWVEC, _mk_red(fn),
                                aux_roles=(), flops_per_elem=cost, tag="reduce")
    return prims


def fused_primitive(name: str, torch_fn: Callable, arity: int) -> Primitive:
    """Wrap an existing multi-input torch fusion (``_genops`` FusionSpec.torch_fn)
    as a MATRIX primitive: the main value is the first operand, the remaining
    ``arity-1`` operands are fresh sampled matrices. Used by the crossover move to
    lift a registered fusion op into the grammar."""
    def _apply(main, auxs, t, F):
        return torch_fn(main, *auxs)
    return Primitive(name, MATRIX, MATRIX, _apply,
                     aux_roles=(ROLE_MATRIX,) * (arity - 1),
                     flops_per_elem=float(arity + 2), tag="binary")


def wrap_unary(name: str, torch_fn: Callable, cost: float = 4.0) -> Primitive:
    """Wrap a registered unary torch op as a MATRIX->MATRIX activation primitive."""
    def _apply(main, auxs, t, F):
        return torch_fn(main)
    return Primitive(name, MATRIX, MATRIX, _apply, aux_roles=(),
                     flops_per_elem=cost, tag="act")


def wrap_reduce(name: str, torch_fn: Callable, cost: float = 2.0) -> Primitive:
    """Wrap a registered reduction torch op as a terminal MATRIX->ROWVEC primitive."""
    def _apply(main, auxs, t, F):
        return torch_fn(main)
    return Primitive(name, MATRIX, ROWVEC, _apply, aux_roles=(),
                     flops_per_elem=cost, tag="reduce")


# --------------------------------------------------------------------------- #
# Pipeline (a type-checked chain of primitives = one minted op)
# --------------------------------------------------------------------------- #
class GrammarTypeError(TypeError):
    """Raised when a pipeline is not well-typed (the grammar's soundness gate)."""


@dataclass(frozen=True)
class Pipeline:
    """An ordered chain of primitives denoting a single composed torch op.

    Well-typed iff: exactly one leading SOURCE (``in_type is None``), every later
    stage consumes the previous stage's ``out_type``, and no stage follows a
    terminal (``ROWVEC``) producer. :meth:`typecheck` enforces this; the minter
    calls it before building a reference so no ill-formed op is ever emitted.
    """

    stages: tuple

    def typecheck(self) -> "Pipeline":
        if not self.stages:
            raise GrammarTypeError("empty pipeline")
        head, *rest = self.stages
        if head.in_type is not None:
            raise GrammarTypeError(f"pipeline must start with a source, got {head.name!r}")
        cur = head.out_type
        for i, st in enumerate(rest, start=1):
            if st.in_type is None:
                raise GrammarTypeError(f"stage {i} {st.name!r} is a second source")
            if st.in_type != cur:
                raise GrammarTypeError(
                    f"stage {i} {st.name!r} expects {st.in_type!r} but got {cur!r}")
            cur = st.out_type
        return self

    @property
    def out_type(self) -> str:
        return self.stages[-1].out_type

    @property
    def aux_roles(self) -> tuple:
        """Flattened aux roles across all stages (== reference/sampler arg order)."""
        roles: list[str] = []
        for st in self.stages:
            roles.extend(st.aux_roles)
        return tuple(roles)

    @property
    def arity(self) -> int:
        return len(self.aux_roles)

    @property
    def uses_matmul(self) -> bool:
        return any(st.tag == "gemm" for st in self.stages)

    def tags(self) -> tuple:
        return tuple(st.tag for st in self.stages)

    def signature(self) -> str:
        """A structural name, e.g. ``matmul->add_bias->gelu->rmsnorm``."""
        return "->".join(st.name for st in self.stages)


# --------------------------------------------------------------------------- #
# Reference oracle + input sampler (correct-by-construction)
# --------------------------------------------------------------------------- #
def _torch_dtype(t, dtype: str):
    return {"bf16": t.bfloat16, "fp16": t.float16, "fp32": t.float32}[dtype]


def apply_pipeline(pipeline: Pipeline, inputs, t, F):
    """Fold ``inputs`` (fp32) through the pipeline, returning the final fp32 value.

    Inputs are consumed in :pyattr:`Pipeline.aux_roles` order. This is the single
    place the composition semantics live (both ``reference_fn`` and the test's
    sequential re-derivation route through it)."""
    cur = None
    idx = 0
    for st in pipeline.stages:
        k = len(st.aux_roles)
        cur = st.apply(cur, tuple(inputs[idx:idx + k]), t, F)
        idx += k
    return cur


def build_reference(pipeline: Pipeline, dtype: str) -> Callable:
    """Return ``reference_fn(*inputs)`` - the correct-by-construction oracle.

    Mirrors ``_genops`` ``ref_fn``: cast inputs to fp32, compute the composition,
    cast the result back to the task dtype. Pure and deterministic."""
    pipeline.typecheck()

    def reference_fn(*inputs):
        t, F = _lazy()
        tdt = _torch_dtype(t, dtype)
        floats = [x.float() if t.is_tensor(x) else x for x in inputs]
        out = apply_pipeline(pipeline, floats, t, F)
        return out.to(tdt)

    return reference_fn


def _sample_role(role: str, dims: dict, gen, t, F):
    M, N = dims["M"], dims["N"]
    K = dims.get("K", N)
    if role == ROLE_MATRIX:
        return t.randn((M, N), generator=gen, dtype=t.float32)
    if role == ROLE_MATRIX_MK:
        return t.randn((M, K), generator=gen, dtype=t.float32) * (1.0 / math.sqrt(K))
    if role == ROLE_WEIGHT_KN:
        return t.randn((K, N), generator=gen, dtype=t.float32) * (1.0 / math.sqrt(K))
    if role == ROLE_BIAS_N:
        return t.randn((N,), generator=gen, dtype=t.float32) * 0.1
    if role == ROLE_WEIGHT_N:
        # norm/affine scale ~ N(1, 0.1): positive and O(1) so norms stay well-posed.
        return t.randn((N,), generator=gen, dtype=t.float32) * 0.1 + 1.0
    raise GrammarTypeError(f"unknown aux role {role!r}")


def build_sampler(pipeline: Pipeline, dims: dict, dtype: str) -> Callable:
    """Return ``input_sampler(seed=0, device="cpu")`` -> tuple of dtype tensors.

    One seeded generator per aux tensor (offset by position, like ``_genops``), so
    the inputs are deterministic and stable across processes."""
    roles = pipeline.aux_roles

    def input_sampler(seed: int = 0, device: str = "cpu"):
        t, F = _lazy()
        tdt = _torch_dtype(t, dtype)
        out = []
        for i, role in enumerate(roles):
            gen = t.Generator(device=device).manual_seed(int(seed) + i)
            x = _sample_role(role, dims, gen, t, F)
            out.append(x.to(device=device, dtype=tdt))
        return tuple(out)

    return input_sampler


# --------------------------------------------------------------------------- #
# Cost model (measured-on-CPU proxies for the MAP-Elites descriptor)
# --------------------------------------------------------------------------- #
_ELEM_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


def flops_and_bytes(pipeline: Pipeline, dims: dict, dtype: str) -> tuple:
    """Estimate (fused) FLOPs, HBM bytes, and arithmetic intensity for ``dims``.

    FLOPs sum each primitive's per-output-element cost (matmul handled with its K
    dependence). Bytes model the FUSED kernel's traffic - every distinct aux input
    read once + the output written once, intermediates staying in registers - so
    deeper fusions raise intensity (more FLOPs, same bytes), matching why fusion is
    the high-value class. AI = FLOPs / bytes (the roofline x-axis)."""
    M, N = dims["M"], dims["N"]
    K = dims.get("K", N)
    out_elems = M * N if pipeline.out_type == MATRIX else M
    flops = 0.0
    for st in pipeline.stages:
        if st.tag == "gemm":
            flops += 2.0 * M * N * K
        else:
            flops += st.flops_per_elem * out_elems

    b = _ELEM_BYTES[dtype]
    role_elems = {ROLE_MATRIX: M * N, ROLE_MATRIX_MK: M * K, ROLE_WEIGHT_KN: K * N,
                  ROLE_BIAS_N: N, ROLE_WEIGHT_N: N}
    in_bytes = sum(role_elems[r] for r in pipeline.aux_roles) * b
    total_bytes = in_bytes + out_elems * b
    ai = flops / total_bytes if total_bytes else 0.0
    return flops, total_bytes, ai


# --------------------------------------------------------------------------- #
# Behavioral hash (probe the reference on a fixed input set)
# --------------------------------------------------------------------------- #
# Canonical probe: a fixed small shape + seed so the hash is a shape-invariant
# fingerprint of the OPERATION (not a particular parametric instance).
PROBE_DIMS = {"M": 16, "N": 24, "K": 24}
PROBE_SEED = 8080
_HASH_GRID = 1.0e4   # round outputs to 4 decimals -> robust to benign fp jitter


def behavioral_hash(pipeline: Pipeline, *, dims: dict = None, seed: int = PROBE_SEED) -> str:
    """SHA1 of the reference's fp32 outputs on the canonical probe set.

    Two pipelines that compute the same function hash identically (behavioral
    dedup); structurally different ops (almost surely) differ. Computed in fp32 so
    it is the precision-independent math identity - the parametric axes (dtype /
    shape scale) are combined on top by the minter to form the full task key."""
    t, F = _lazy()
    d = dims or PROBE_DIMS
    ref = build_reference(pipeline, "fp32")
    sampler = build_sampler(pipeline, d, "fp32")
    out = ref(*sampler(seed=seed))
    q = t.round(out.double().flatten() * _HASH_GRID).to(t.int64).tolist()
    preimage = f"{tuple(out.shape)}|{q}"
    return hashlib.sha1(preimage.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Grammar evolution: self-referential productions (escape the bounded encoding)
# --------------------------------------------------------------------------- #
# The minter's fixed MOVES compose the primitive tables with bounded-depth
# templates, so the reachable task set is a bounded subset of all type-valid
# pipelines - the POET/OMNI "bounded encoding" ceiling. A :class:`Production`
# makes the *composition rules themselves* first-class and EVOLVABLE: it is a
# reusable, well-typed pipeline fragment whose body is a flat tuple of EXISTING
# named primitives. New productions are built by composing existing productions
# (:func:`compose_productions`), including previously-evolved ones, so this is a
# self-referential grammar over ``B : MATRIX -> MATRIX`` (and terminal
# ``T : MATRIX -> ROWVEC``) whose reachable depth/structure is unbounded.
#
# Why correctness-by-construction survives:
#   * Composition is type-safe by construction - concatenating a ``t0 -> t1``
#     fragment with a ``t1 -> t2`` fragment is exactly a ``t0 -> t2`` fragment, and
#     a ROWVEC (terminal) fragment can never be extended - so a production can only
#     ever denote a well-typed chain.
#   * A production's stages are always the ORIGINAL name-addressable primitives, so
#     :func:`pipeline_from_production` yields an ordinary flat :class:`Pipeline`.
#     It type-checks, ``behavioral_hash``-es and niches like any other pipeline,
#     and every stage name still resolves in ``materialize._prim_by_name`` (so the
#     materialize self-check can pass). Nothing here bypasses the construction
#     gate - the minter still gates every emitted task.
@dataclass(frozen=True)
class Production:
    """An evolvable, well-typed grammar production (a composition operator).

    Denotes a pipeline FRAGMENT that consumes ``in_type`` and produces
    ``out_type``. ``stages`` is a flat tuple of EXISTING :class:`Primitive`
    objects (a base production wraps one primitive; a composed production
    concatenates the stages of its parents), and ``depth`` is the composition
    depth (number of primitive stages), used for the growth budget / QD niche.

    Because the stages are the same named primitives the fixed grammar uses, any
    pipeline built from a production is a plain, verifiable, materialize-safe
    pipeline; the production layer only changes HOW pipelines are *generated*
    (an unbounded, self-referential search), never what a valid task is.
    """

    name: str
    in_type: str
    out_type: str
    stages: tuple
    depth: int = 1

    def typecheck(self) -> "Production":
        """Validate the fragment composes ``in_type -> out_type`` with no inner
        source and no stage after a terminal. Raises :class:`GrammarTypeError`."""
        if not self.stages:
            raise GrammarTypeError("empty production")
        cur = self.in_type
        for i, st in enumerate(self.stages):
            if st.in_type is None:
                raise GrammarTypeError(
                    f"production {self.name!r} stage {i} {st.name!r} is a source")
            if st.in_type != cur:
                raise GrammarTypeError(
                    f"production {self.name!r} stage {i} {st.name!r} expects "
                    f"{st.in_type!r} but got {cur!r}")
            cur = st.out_type
        if cur != self.out_type:
            raise GrammarTypeError(
                f"production {self.name!r} declares out_type {self.out_type!r} "
                f"but its stages yield {cur!r}")
        return self

    @property
    def aux_roles(self) -> tuple:
        roles: list[str] = []
        for st in self.stages:
            roles.extend(st.aux_roles)
        return tuple(roles)

    def signature(self) -> str:
        """Structural id over the FLAT primitive names (dedup key for productions)."""
        return "->".join(st.name for st in self.stages)


def base_productions() -> list[Production]:
    """The grammar's AXIOM productions: one depth-1 production per composable
    primitive. MATRIX->MATRIX middles become ``MATRIX->MATRIX`` block axioms and
    terminal reducers become ``MATRIX->ROWVEC`` axioms - the seeds the evolver
    composes into deeper, net-new productions. (Touches torch lazily via the
    primitive tables, exactly like the rest of the module.)"""
    prods: list[Production] = []
    for p in middle_prims().values():
        if p.in_type == MATRIX and p.out_type == MATRIX:
            prods.append(Production(p.name, MATRIX, MATRIX, (p,), depth=1))
    for p in terminal_prims().values():
        prods.append(Production(p.name, p.in_type, p.out_type, (p,), depth=1))
    return prods


def compose_productions(a: Production, b: Production) -> Optional[Production]:
    """Sequentially compose ``a`` then ``b`` - the self-referential grammar operator.

    Valid iff ``a`` produces what ``b`` consumes (``a.out_type == b.in_type``); a
    terminal (ROWVEC) fragment cannot be extended. Returns the composed production
    (its stages the concatenation of the parents', so it is well-typed BY
    construction) or ``None`` for an incompatible pair (fail-safe)."""
    if a.out_type == ROWVEC:            # nothing consumes a terminal
        return None
    if a.out_type != b.in_type:
        return None
    return Production(
        name=f"({a.name}.{b.name})",
        in_type=a.in_type,
        out_type=b.out_type,
        stages=a.stages + b.stages,
        depth=a.depth + b.depth,
    )


def pipeline_from_production(source: Primitive, body: Production,
                             terminal: Optional[Production] = None) -> Pipeline:
    """Instantiate a full, type-checked :class:`Pipeline` from a SOURCE primitive,
    a body production (``MATRIX->MATRIX``) and an optional terminal production
    (``MATRIX->ROWVEC``). Flattens everything into a single tuple of named
    primitives, so the result is an ordinary gate-ready, materialize-safe pipeline."""
    if source.in_type is not None:
        raise GrammarTypeError(f"{source.name!r} is not a source")
    stages = [source, *body.stages]
    if terminal is not None:
        stages.extend(terminal.stages)
    return Pipeline(tuple(stages)).typecheck()
