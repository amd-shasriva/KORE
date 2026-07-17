"""Verifiable, open-ended TASK MINTER for the KORE RL curriculum (P3 paradigm).

The proposer (:mod:`kore.openended.proposer`) selects tasks at the policy's
competence frontier from a FIXED parametric space (the 282 registered op x dtype
tasks). This module removes that ceiling: it MINTS *net-new*, correct-by-
construction tasks at train time by composing verified torch primitives
(:mod:`kore.openended.grammar`), so the curriculum can grow open-endedly instead
of only re-weighting a fixed menu.

This is NOT the offline datagen stage (``kore.tasks.generate_ops`` writes task
dirs to disk). A :class:`MintedTask` is an in-memory RL-curriculum task -
``(name, reference_fn, input_sampler, dtype, tol, family)`` - whose
``reference_fn`` IS the spec (a pure torch oracle built by composing primitives).

Four minting MOVES:

  a. **fusion / composition** - chain primitives into a new fused op
     (e.g. ``matmul -> bias -> gelu -> residual -> rmsnorm``).
  b. **parametric extrapolation** - re-cast a structure at a new dtype / shape
     scale (a new region of behavior space).
  c. **novel elementwise / reduction** - compose activations / reductions into a
     brand-new op defined purely by its torch reference.
  d. **mutation / crossover** - lift REGISTERED descriptors
     (:mod:`kore.openended.task_space`) into the grammar and perturb / recombine.

Every candidate passes a CONSTRUCTION GATE before it can enter the curriculum
(type-check, executes on CPU, deterministic, finite, non-degenerate - output
variance, sensitivity to every input, variation across axes; robust-kbench-style -
plus behavioral-hash dedup and held-out-family rejection). Survivors get a
MAP-Elites niche from measured CPU proxies (arithmetic intensity, fusion depth,
dtype precision, shape scale - the :func:`task_space.descriptor_features`
conventions), a learnability score ``4p(1-p)`` from a supplied rollout success
rate, and a proposer reward = measured learning-progress delta.

Pure and CPU-only; ``torch`` is used only for the CPU gate / probes (never a GPU).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from kore.openended import archive as arch_mod
from kore.openended import grammar as g
from kore.openended import task_space as ts
from kore.openended.proposer import DescriptorStats, clamp, learnability

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# dtypes we mint. fp8/int8 are intentionally excluded (they need quantization
# scales; the generic randn sampler degenerates under int8/fp8 - exactly why
# generate_ops.FAMILY_DTYPES also omits them for generated ops).
MINT_DTYPES = ("bf16", "fp16", "fp32")

# roofline ridge (FLOPs/byte): >= => compute-bound, else memory-bound. matmul
# chains sit far above; elementwise/reduction chains far below.
AI_RIDGE = 20.0

# non-degeneracy thresholds for the construction gate.
_VAR_EPS = 1.0e-6
_SENS_EPS = 1.0e-6

# per-dtype correctness tolerance carried on the task (allclose atol/rtol proxy).
_TOL = {"fp32": 1.0e-4, "fp16": 1.0e-2, "bf16": 2.0e-2}

# Shape regimes -> concrete dims. Chosen so problem VOLUME lands in the
# task_space shape_scale bands (small<1e6<=medium<1e9<=large) while every tensor
# stays cheap to allocate on CPU (matmul volume is M*N*K but its operands are only
# M*K + K*N, so gemm reaches 'large' scale with ~tens of MB of operands).
_REGIMES_2D = {
    "small": {"M": 64, "N": 512},        # vol 3.3e4
    "medium": {"M": 1024, "N": 2048},    # vol 2.1e6
}
_REGIMES_GEMM = {
    "small": {"M": 32, "N": 128, "K": 128},      # vol 5.2e5
    "medium": {"M": 128, "N": 512, "K": 512},    # vol 3.4e7
    "large": {"M": 256, "N": 1024, "K": 4096},   # vol 1.1e9
}

# held-out tokens (defense in depth on top of the grammar having NO attention/
# mla/paged primitives at all - so a held-out task is unrepresentable).
_HELDOUT_TOKENS = ("mla", "paged", "latent_attn", "latent_attention")


# --------------------------------------------------------------------------- #
# MintedTask
# --------------------------------------------------------------------------- #
@dataclass
class MintedTask:
    """A minted RL-curriculum task. The first six fields are the task ABI
    ``(name, reference_fn, input_sampler, dtype, tol, family)``; the rest is
    construction metadata + the MAP-Elites descriptor + frontier scores."""

    # --- task ABI (the spec) ---
    name: str
    reference_fn: Callable                 # pure torch oracle: reference IS the spec
    input_sampler: Callable                # (seed=0, device="cpu") -> tuple[Tensor]
    dtype: str
    tol: float
    family: str

    # --- construction / identity ---
    pipeline: g.Pipeline
    shape: dict
    shape_regime: str
    arity: int
    behavioral_hash: str
    features: dict                         # descriptor_features-style (ts conventions)
    niche_key: tuple                       # tuple(features[f] for f in ts.NICHE_FIELDS)
    arithmetic_intensity: float            # measured FLOPs/byte
    move: str

    # --- frontier scores (filled at mint time) ---
    solve_rate: float = 0.0
    learnability: float = 0.0
    novelty: float = 0.0
    difficulty: float = 0.0
    proposer_reward: float = 0.0

    @property
    def task_id(self) -> str:
        return f"gen_{self.name}_{self.dtype}"

    @property
    def dedup_key(self) -> tuple:
        """Full task identity: behavioral (math) hash x precision x shape scale.

        The behavioral hash is the shape/precision-independent math fingerprint;
        combining it with dtype + shape_scale means a genuine parametric-
        extrapolation variant (move b) is NOT a duplicate, while re-minting the
        exact same op is."""
        return (self.behavioral_hash, self.dtype, self.features["shape_scale"])

    def carrier(self) -> ts.TaskDescriptor:
        """A ``TaskDescriptor`` view for archive bookkeeping (``source='minted'``).

        Used only for its ``task_id`` / total-order fields; its niche key comes
        from :pyattr:`niche_key` (measured), not ``ts.descriptor_key`` (which is
        defined only over the registered space)."""
        return ts.TaskDescriptor("minted", self.family, self.name, self.dtype,
                                 self.shape_regime)

    def probe_inputs(self, seed: int = g.PROBE_SEED, device: str = "cpu"):
        """Cheap fixed-shape inputs for the CPU gate / smoke checks."""
        return g.build_sampler(self.pipeline, g.PROBE_DIMS, self.dtype)(seed, device)

    def describe(self) -> dict:
        return {"task_id": self.task_id, "family": self.family, "move": self.move,
                "arity": self.arity, "signature": self.pipeline.signature(),
                "niche": self.niche_key, "learnability": round(self.learnability, 4),
                "novelty": round(self.novelty, 4), "difficulty": round(self.difficulty, 4),
                "arithmetic_intensity": round(self.arithmetic_intensity, 2)}

    # -- integration: minted task -> runnable KORE reference.py namespace -------
    def to_reference_namespace(self, base_shape: Optional[dict] = None) -> dict:
        """Emit the SAME namespace ``_genops.make_reference`` returns, so a minted
        task drops into KORE's task ABI: write a ``reference.py`` that does
        ``globals().update(minted.to_reference_namespace())`` and the generic
        driver / verifier grades it exactly like a generated op. ``get_inputs``
        re-samples at any requested shape/device (GPU at train time)."""
        base_shape = dict(base_shape or self.shape)

        def parse_shape(shape_str: str) -> dict:
            if not shape_str or shape_str == "default":
                return dict(base_shape)
            out = {}
            for kv in shape_str.split(","):
                k, v = kv.split("=")
                out[k.strip()] = int(v)
            return out

        def get_inputs(shape, device="cuda", seed=0):
            return g.build_sampler(self.pipeline, shape, self.dtype)(seed, device)

        return {
            "parse_shape": parse_shape,
            "get_inputs": get_inputs,
            "ref_fn": self.reference_fn,
            "baseline_fn": self.reference_fn,   # torch-eager baseline (fusion headroom)
            "arity": self.arity,
            "entry_name": self.name,
            "dtype_name": self.dtype,
            "family": self.family,
        }


# --------------------------------------------------------------------------- #
# Behavior descriptor (measured CPU proxies -> ts.descriptor_features conventions)
# --------------------------------------------------------------------------- #
def _fusion_depth(pipeline: g.Pipeline) -> int:
    """Number of fused sub-ops (every non-source primitive, incl. the matmul)."""
    return sum(1 for st in pipeline.stages if st.tag != "source")


def family_of(pipeline: g.Pipeline) -> str:
    """Coarse minted family (the categorical MAP-Elites dimension)."""
    if pipeline.uses_matmul:
        return "minted_gemm_fusion"
    if pipeline.out_type == g.ROWVEC:
        return "minted_reduce"
    if any(st.tag == "norm" for st in pipeline.stages):
        return "minted_norm"
    if _fusion_depth(pipeline) >= 2:
        return "minted_fusion"
    return "minted_elementwise"


def _shape_scale(volume: int) -> str:
    if volume < ts._SCALE_SMALL:
        return "small"
    if volume < ts._SCALE_LARGE:
        return "medium"
    return "large"


def features_of(pipeline: g.Pipeline, dims: dict, dtype: str,
                ai_ridge: float = AI_RIDGE) -> tuple:
    """MAP-Elites behavior dims for a minted op, mirroring
    :func:`task_space.descriptor_features` (same keys/order) but from MEASURED
    CPU proxies (arithmetic intensity from a FLOPs/byte estimate, fusion depth
    from the pipeline, precision/scale from dtype/volume). Returns
    ``(features_dict, arithmetic_intensity_flops_per_byte)``."""
    _flops, _bytes, ai = g.flops_and_bytes(pipeline, dims, dtype)
    volume = 1
    for v in dims.values():
        volume *= int(v)
    feats = {
        "family": family_of(pipeline),
        "arithmetic_intensity": "compute-bound" if ai >= ai_ridge else "memory-bound",
        "fusion_depth": _fusion_depth(pipeline),
        "dtype_precision": ts._PRECISION_CLASS[dtype],
        "dtype": dtype,
        "shape_scale": _shape_scale(volume),
    }
    return feats, ai


def niche_key_of(features: dict) -> tuple:
    """Archive niche tuple in the SAME field order as ``task_space.NICHE_FIELDS``."""
    return tuple(features[f] for f in ts.NICHE_FIELDS)


def difficulty_of(features: dict) -> float:
    """GPU-free difficulty prior in [0,1] from measured features (mirrors
    ``task_space.static_difficulty``: compute-bound + deeper fusion + lower
    precision + bigger shapes are harder). A prior only; measured solve-rate rules."""
    score = 0.0
    if features["arithmetic_intensity"] == "compute-bound":
        score += 0.35
    score += min(0.30, 0.10 * (features["fusion_depth"] - 1))
    if features["dtype_precision"] == "16b":
        score += 0.15
    score += {"small": 0.0, "medium": 0.10, "large": 0.20}[features["shape_scale"]]
    return clamp(score)


# --------------------------------------------------------------------------- #
# Held-out rejection (by construction; enforced anyway)
# --------------------------------------------------------------------------- #
def is_heldout(name: str, family: str = "") -> bool:
    """True iff a task names a held-out generalization family (mla / paged_attn).

    Reuses the canonical registry classifier so the guard can never drift from the
    train/held-out split, then a token backstop. The grammar has no attention /
    mla / paged primitive, so this never fires for a real minted task - it is a
    correctness guarantee, not a runtime filter."""
    text = f"{name} {family}".lower()
    if any(tok in text for tok in _HELDOUT_TOKENS):
        return True
    try:
        from types import SimpleNamespace

        from kore.tasks import registry
        fam = registry.operator_family(SimpleNamespace(operation=name, task_id=name))
        return fam in registry.HELDOUT_FAMILIES
    except Exception:  # noqa: BLE001 - registry optional; token check already ran
        return False


# --------------------------------------------------------------------------- #
# Construction gate (type + execute + determinism + non-degeneracy + held-out)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GateResult:
    ok: bool
    reason: str = "ok"

    def __bool__(self) -> bool:
        return self.ok


def construction_gate(pipeline: g.Pipeline, dtype: str, name: str, family: str,
                      *, gate_dims: dict = None) -> GateResult:
    """Reject any candidate that is not a well-formed, non-degenerate task.

    Checks (robust-kbench-inspired), all on CPU torch:
      1. type-checks (sound composition);
      2. not a held-out family (by construction, enforced);
      3. executes on sample inputs;
      4. outputs are finite;
      5. deterministic (same seed -> identical outputs);
      6. output is non-constant (variance > eps);
      7. output varies along EVERY axis (no collapsed/degenerate structure);
      8. output is sensitive to EVERY input (resampling any input changes it).
    """
    gate_dims = gate_dims or g.PROBE_DIMS
    # 1. type soundness
    try:
        pipeline.typecheck()
    except g.GrammarTypeError as e:
        return GateResult(False, f"typecheck:{e}")
    # 2. held-out rejection
    if is_heldout(name, family):
        return GateResult(False, "heldout_family")

    import torch
    ref = g.build_reference(pipeline, dtype)
    sampler = g.build_sampler(pipeline, gate_dims, dtype)

    # 3. executes
    try:
        inp = sampler(seed=1)
        out = ref(*inp)
    except Exception as e:  # noqa: BLE001 - a candidate that raises is simply rejected
        return GateResult(False, f"execute:{type(e).__name__}")

    of = out.float()
    # 4. finite
    if not torch.isfinite(of).all():
        return GateResult(False, "nonfinite")
    # 5. determinism (seeded, stable)
    if not torch.equal(out, ref(*sampler(seed=1))):
        return GateResult(False, "nondeterministic")
    # 6. non-constant output
    if of.std().item() <= _VAR_EPS:
        return GateResult(False, "constant_output")
    # 7. varies across axes (rejects broadcast-of-a-scalar style collapse)
    if of.dim() == 2:
        if of.std(dim=1).max().item() <= _VAR_EPS:
            return GateResult(False, "constant_along_rows")
        if of.std(dim=0).max().item() <= _VAR_EPS:
            return GateResult(False, "constant_along_cols")
    # 8. sensitive to every input (resample input i -> output must change)
    alt = sampler(seed=424242)
    for i in range(len(inp)):
        pert = list(inp)
        pert[i] = alt[i]
        if torch.allclose(of, ref(*pert).float(), atol=_SENS_EPS, rtol=0.0):
            return GateResult(False, f"insensitive_input_{i}")

    return GateResult(True, "ok")


# --------------------------------------------------------------------------- #
# TaskMinter
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Candidate:
    pipeline: g.Pipeline
    dtype: str
    regime: str
    move: str


class TaskMinter:
    """Deterministic (seeded), LLM-free minter of verifiable curriculum tasks."""

    MOVES = ("fusion", "extrapolate", "novel", "mutate_crossover")

    def __init__(self, seed: int = 0, *, include_vendor: bool = True,
                 ai_ridge: float = AI_RIDGE):
        self.rng = random.Random(seed)
        self.include_vendor = include_vendor
        self.ai_ridge = ai_ridge
        self._seen: set = set()            # dedup keys of accepted tasks
        self._pool: list = []              # accepted pipelines (bases for extrapolation)
        self._libs_cache = None

    # -- primitive libraries (lazy: torch only touched here) ---------------- #
    def _libs(self):
        if self._libs_cache is None:
            self._libs_cache = (g.source_prims(), g.middle_prims(), g.terminal_prims())
        return self._libs_cache

    def _matrix_middles(self) -> list:
        _s, mid, _t = self._libs()
        return [p for p in mid.values() if p.out_type == g.MATRIX]

    def _acts(self) -> list:
        _s, mid, _t = self._libs()
        return [p for p in mid.values() if p.tag == "act"]

    def _scales(self) -> list:
        _s, mid, _t = self._libs()
        return [p for p in mid.values() if p.tag == "scale"]

    # -- shape regimes ------------------------------------------------------ #
    def _regime_table(self, pipeline: g.Pipeline) -> dict:
        return _REGIMES_GEMM if pipeline.uses_matmul else _REGIMES_2D

    def _choose_regime(self, pipeline: g.Pipeline, exclude: str = None) -> str:
        keys = [r for r in self._regime_table(pipeline) if r != exclude]
        return self.rng.choice(keys or list(self._regime_table(pipeline)))

    def _dims(self, pipeline: g.Pipeline, regime: str) -> tuple:
        table = self._regime_table(pipeline)
        if regime not in table:
            regime = list(table)[-1]           # clamp to the largest available
        return dict(table[regime]), regime

    def _ensure_nontrivial(self, stages: list) -> list:
        """Guarantee at least one compute stage (a bare ``input`` copy is trivial)."""
        if all(st.tag == "source" for st in stages):
            stages = stages + [self.rng.choice(self._acts())]
        return stages

    # -- MOVE (a): fusion / composition ------------------------------------ #
    def _move_fusion(self) -> _Candidate:
        src, _mid, term = self._libs()
        head = src["matmul"] if self.rng.random() < 0.4 else src["input"]
        stages = [head]
        for _ in range(self.rng.randint(1, 4)):
            stages.append(self.rng.choice(self._matrix_middles()))
        if self.rng.random() < 0.25:           # optional terminal reduction
            stages.append(self.rng.choice(list(term.values())))
        pipeline = g.Pipeline(tuple(self._ensure_nontrivial(stages)))
        dtype = self.rng.choice(MINT_DTYPES)
        return _Candidate(pipeline, dtype, self._choose_regime(pipeline), "fusion")

    # -- MOVE (b): parametric extrapolation -------------------------------- #
    def _move_extrapolate(self) -> _Candidate:
        if not self._pool:                     # nothing to extrapolate from yet
            return self._move_fusion()
        base = self.rng.choice(self._pool)
        pipeline = base.pipeline
        dtype = self.rng.choice([d for d in MINT_DTYPES if d != base.dtype] or MINT_DTYPES)
        regime = self._choose_regime(pipeline, exclude=base.regime)
        return _Candidate(pipeline, dtype, regime, "extrapolate")

    # -- MOVE (c): novel elementwise / reduction op ------------------------ #
    def _move_novel(self) -> _Candidate:
        src, _mid, term = self._libs()
        acts = self._acts()
        if self.rng.random() < 0.5:            # a novel activation (composed acts)
            stages = [src["input"]] + [self.rng.choice(acts)
                                       for _ in range(self.rng.randint(2, 3))]
            if self.rng.random() < 0.5:
                stages.append(self.rng.choice(self._scales()))
        else:                                  # a novel reduction (act then reduce)
            stages = [src["input"], self.rng.choice(acts),
                      self.rng.choice(list(term.values()))]
        pipeline = g.Pipeline(tuple(stages))
        return _Candidate(pipeline, self.rng.choice(MINT_DTYPES),
                          self._choose_regime(pipeline), "novel")

    # -- MOVE (d): mutation / crossover of REGISTERED descriptors ---------- #
    def _move_mutate_crossover(self) -> _Candidate:
        d1 = ts.sample_descriptor(self.rng, self.include_vendor)
        p1 = self._descriptor_to_pipeline(d1)
        if p1 is None:
            return self._move_fusion()
        if self.rng.random() < 0.5:
            pipeline = self._mutate_pipeline(p1)
        else:
            d2 = ts.sample_descriptor(self.rng, self.include_vendor)
            p2 = self._descriptor_to_pipeline(d2) or p1
            pipeline = self._crossover(p1, p2)
        pipeline = g.Pipeline(tuple(self._ensure_nontrivial(list(pipeline.stages))))
        dtype = d1.dtype if d1.dtype in MINT_DTYPES else self.rng.choice(MINT_DTYPES)
        return _Candidate(pipeline, dtype, self._choose_regime(pipeline), "mutate_crossover")

    def _descriptor_to_pipeline(self, desc: ts.TaskDescriptor):
        """Lift a registered descriptor into the grammar (None if unmappable)."""
        src, mid, _term = self._libs()
        if desc.source == "vendor":
            op = desc.op
            if op in ("rmsnorm", "fused_add_rmsnorm"):
                return g.Pipeline((src["input"], mid["rmsnorm"]))
            if op == "layernorm":
                return g.Pipeline((src["input"], mid["layernorm"]))
            if op in ("softmax", "topk_softmax"):
                return g.Pipeline((src["input"], mid["softmax"]))
            if op in ("silu_mul", "gelu_mul") and op in mid:
                return g.Pipeline((src["input"], mid[op]))
            if op in ("gemm_a8w8", "batched_gemm", "gemm_a8w8_blockscale"):
                return g.Pipeline((src["matmul"],))
            return None
        # genops
        try:
            family, spec = ts._genops_registry()[desc.op]
        except KeyError:
            return None
        if family == "unary":
            return g.Pipeline((src["input"], g.wrap_unary(desc.op, spec.torch_fn)))
        if family == "binary":
            return g.Pipeline((src["input"], g.fused_primitive(desc.op, spec.torch_fn, 2)))
        if family == "reduce":
            return g.Pipeline((src["input"], g.wrap_reduce(desc.op, spec.torch_fn)))
        if family == "fusion":
            return g.Pipeline((src["input"],
                               g.fused_primitive(desc.op, spec.torch_fn, spec.arity)))
        if family == "gemm_fusion":
            stages = [src["matmul"]]
            if getattr(spec, "has_bias", False):
                stages.append(mid["add_bias"])
            act = getattr(spec, "act", "none")
            if act in mid:
                stages.append(mid[act])
            return g.Pipeline(tuple(stages))
        return None

    def _mutate_pipeline(self, p: g.Pipeline) -> g.Pipeline:
        stages = list(p.stages)
        has_term = stages[-1].out_type == g.ROWVEC
        body = stages[1:-1] if has_term else stages[1:]
        op = self.rng.choice(("swap", "append", "scale"))
        if op == "swap" and body:
            body[self.rng.randrange(len(body))] = self.rng.choice(self._acts())
        elif op == "append":
            body.append(self.rng.choice(self._matrix_middles()))
        else:
            body.insert(self.rng.randrange(len(body) + 1), self.rng.choice(self._scales()))
        new = [stages[0]] + body + ([stages[-1]] if has_term else [])
        try:
            return g.Pipeline(tuple(new)).typecheck()
        except g.GrammarTypeError:
            return p

    def _crossover(self, p1: g.Pipeline, p2: g.Pipeline) -> g.Pipeline:
        def parts(p):
            s = list(p.stages)
            term = s[-1].out_type == g.ROWVEC
            return s[0], (s[1:-1] if term else s[1:]), (s[-1] if term else None)

        src1, body1, term1 = parts(p1)
        _src2, body2, term2 = parts(p2)
        body = body1[:len(body1) // 2] + body2[len(body2) // 2:]
        term = term2 or term1
        stages = [src1] + body + ([term] if term else [])
        try:
            return g.Pipeline(tuple(stages)).typecheck()
        except g.GrammarTypeError:
            return p1

    # -- candidate dispatch ------------------------------------------------- #
    def _make_candidate(self, move: str) -> Optional[_Candidate]:
        try:
            if move == "fusion":
                return self._move_fusion()
            if move == "extrapolate":
                return self._move_extrapolate()
            if move == "novel":
                return self._move_novel()
            if move == "mutate_crossover":
                return self._move_mutate_crossover()
        except Exception:  # noqa: BLE001 - a bad draw is just skipped
            return None
        return None

    # -- build + gate + score + place -------------------------------------- #
    def _build(self, cand: _Candidate) -> Optional[MintedTask]:
        pipeline = g.Pipeline(tuple(self._ensure_nontrivial(list(cand.pipeline.stages))))
        name = pipeline.signature().replace("->", "_")
        family = family_of(pipeline)
        gate = construction_gate(pipeline, cand.dtype, name, family)
        if not gate.ok:
            return None
        dims, regime = self._dims(pipeline, cand.regime)
        feats, ai = features_of(pipeline, dims, cand.dtype, self.ai_ridge)
        return MintedTask(
            name=name,
            reference_fn=g.build_reference(pipeline, cand.dtype),
            input_sampler=g.build_sampler(pipeline, dims, cand.dtype),
            dtype=cand.dtype,
            tol=_TOL[cand.dtype],
            family=family,
            pipeline=pipeline,
            shape=dims,
            shape_regime=regime,
            arity=pipeline.arity,
            behavioral_hash=g.behavioral_hash(pipeline),
            features=feats,
            niche_key=niche_key_of(feats),
            arithmetic_intensity=ai,
            move=cand.move,
        )

    def novelty(self, niche_key: tuple, archive) -> float:
        """Novelty of a niche vs the archive's occupied niches, in [0,1].

        Mirrors ``proposer.descriptor_novelty`` (Hamming over niche fields), but
        keyed on the MEASURED minted niche (``ts.descriptor_key`` is only defined
        over the registered space, so it is not reused for minted ops)."""
        if archive is None:
            return 1.0
        occupied = archive.occupied_keys()
        if not occupied:
            return 1.0
        if niche_key in occupied:
            return 0.0
        n = len(niche_key)
        return min(sum(1 for a, b in zip(niche_key, k) if a != b) / n for k in occupied)

    def _place(self, archive, mt: MintedTask, stats: DescriptorStats) -> bool:
        """Niche-place a minted task into a ``TaskArchive`` (fitness-gated, like
        ``TaskArchive.add`` but keyed on the measured minted niche)."""
        if archive is None:
            return False
        key = mt.niche_key
        cur = archive.cells.get(key)
        if cur is None:
            archive.cells[key] = arch_mod.TaskCell(descriptor=mt.carrier(),
                                                   stats=stats, key=key,
                                                   history=[mt.task_id])
            return True
        cur.history.append(mt.task_id)
        if arch_mod.informativeness(stats) > cur.fitness:
            cur.descriptor = mt.carrier()
            cur.stats = stats
            return True
        return False

    def register(self, cand: _Candidate, archive=None,
                 policy_p_fn: Optional[Callable] = None,
                 progress_fn: Optional[Callable] = None) -> Optional[MintedTask]:
        """Gate + dedup + score + niche-place one candidate. Returns the accepted
        :class:`MintedTask`, or ``None`` if it was rejected or a duplicate."""
        mt = self._build(cand)
        if mt is None:
            return None
        if mt.dedup_key in self._seen:         # behavioral-hash dedup
            return None
        self._seen.add(mt.dedup_key)

        p = clamp(policy_p_fn(mt)) if policy_p_fn is not None else 0.5
        mt.solve_rate = p
        mt.learnability = learnability(p)
        mt.novelty = self.novelty(mt.niche_key, archive)
        mt.difficulty = difficulty_of(mt.features)
        # proposer reward = measured learning-progress delta (injected), else the
        # learnability prior.
        mt.proposer_reward = clamp(progress_fn(mt)) if progress_fn is not None \
            else mt.learnability

        stats = DescriptorStats(solve_rate=p, headroom_regret=mt.difficulty,
                                attempts=0, novelty=mt.novelty)
        self._place(archive, mt, stats)
        self._pool.append(cand)
        return mt

    def mint_batch(self, archive, policy_p_fn, n: int, *,
                   progress_fn: Optional[Callable] = None,
                   max_attempts: Optional[int] = None) -> list:
        """Mint up to ``n`` net-new tasks: cycle the four moves, gate + dedup each,
        score + niche-place survivors. Deterministic given the minter seed."""
        if n <= 0:
            return []
        out: list = []
        budget = max_attempts if max_attempts is not None else max(64, 48 * n)
        attempt = 0
        while len(out) < n and attempt < budget:
            move = self.MOVES[attempt % len(self.MOVES)]
            attempt += 1
            cand = self._make_candidate(move)
            if cand is None:
                continue
            mt = self.register(cand, archive, policy_p_fn, progress_fn)
            if mt is not None:
                out.append(mt)
        return out


# --------------------------------------------------------------------------- #
# Module-level convenience API
# --------------------------------------------------------------------------- #
def mint_batch(archive, policy_p_fn, n: int, seed: int, *,
               progress_fn: Optional[Callable] = None,
               include_vendor: bool = True,
               max_attempts: Optional[int] = None) -> list:
    """Mint, gate, dedup, and niche-place ``n`` net-new tasks (seeded).

    Parameters
    ----------
    archive:
        A :class:`kore.openended.archive.TaskArchive` (or ``None``); survivors are
        niche-placed into it by their measured MAP-Elites key.
    policy_p_fn:
        ``(MintedTask) -> p`` rollout success-rate for learnability ``4p(1-p)``.
    n, seed:
        Batch size and deterministic seed.
    progress_fn:
        Optional ``(MintedTask) -> deltaP`` learning-progress callback; when given,
        it becomes the proposer reward (else the learnability prior is used).
    """
    minter = TaskMinter(seed=seed, include_vendor=include_vendor)
    return minter.mint_batch(archive, policy_p_fn, n, progress_fn=progress_fn,
                             max_attempts=max_attempts)
