"""Self-extending transform library: PROPOSE candidate new rewrite rules.

This is the frontier "the library extends itself" mechanism, and it is **OFF by
default** - importing this module changes nothing; the curated
:data:`kore.transform.library.LIBRARY` is untouched unless a caller explicitly
merges in proposals.

What it does
------------
Given a set of *base* transforms, :func:`discover_transforms` synthesizes NEW
:class:`~kore.transform.calculus.Transformation` objects by cheap, CPU-only,
source-level strategies:

  * **knob sweeps** - parameter-sweep variants of existing numeric knobs
    (``num_warps`` / ``num_stages`` / ``waves_per_eu`` / ``GROUP_M`` / block
    sizes / ``SPLIT_K`` / IO dtype) at values NOT already in the base grid;
  * **vectorization widths** - pinned ``tl.max_contiguous(tl.multiple_of(...))``
    contiguity annotations at candidate widths;
  * **elementwise fusion** - fuse two adjacent elementwise assignments by
    inlining the temporary.

Every proposal is:

  * a **pure source->source function** that RETURNS ``None`` (a no-op) when its
    precondition / side-condition does not match, and **never raises** (it reuses
    the base transform's guarded ``apply`` / ``side_conditions``, or a
    self-guarding pattern matcher);
  * typed **conservatively ``approx`` (≈_ε)** with a non-under-estimated ε (at
    least the base transform's ε for those params, floored), because a proposal
    is *unverified*;
  * namespaced with the :data:`DISCOVERED_PREFIX` so it can never be confused
    with, or silently shadow, a curated transform.

.. important::
   **These are PROPOSALS, not proofs.** Nothing here verifies semantic
   equivalence or the ε contract. Correctness is enforced **downstream by the
   env's SNR oracle**, which build/test/benches every rewritten kernel: an
   out-of-contract or wrong proposal fails the SNR gate and is rejected/pruned.
   The value of this module is *broadening the in-contract action space* with
   cheap, structurally-bounded candidates - never a correctness guarantee.

Opt-in wiring
-------------
The default calculus is unchanged. A caller opts in by building an extended
registry and passing it through the ``library=`` seam that already exists on
:func:`kore.transform.apply_sequence` / :func:`kore.transform.admissible_actions`
and :class:`kore.search.propose.TransformProposePolicy`::

    from kore.transform import LIBRARY, admissible_actions, apply_sequence
    from kore.transform.discover import extend_library

    ext = extend_library(source=kernel_src)          # base + relevant proposals
    actions = admissible_actions(kernel_src, budget, library=ext)
    new_src, applied, rejected, state = apply_sequence(
        kernel_src, [actions[i].as_step()], budget, library=ext)

Pure (stdlib + ``re`` only), CPU-only, deterministic.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from kore.transform.budget import RELATION_APPROX, RELATION_EXACT
from kore.transform.calculus import Transformation

# Namespace for every synthesized proposal, so a discovered transform is visually
# and programmatically distinct from a curated one (and merge de-dup is trivial).
DISCOVERED_PREFIX = "disc:"

# Conservative ε floor for proposals: an unverified rewrite is never typed cheaper
# than a real numeric contract. The effective ε = max(base ε for the params,
# floor), so we never UNDER-estimate drift (the SNR gate is still the authority).
_DISCOVER_EPS_FLOOR = 0.01
_FUSION_EPS = 0.02          # fused elementwise may enable FMA contraction / reassoc

# Per-knob candidate sweeps, keyed by a base transform's ``knob``. Values are
# hardware-plausible for MI350X / gfx950 and are chosen to AVOID the base grids
# (so a sweep is a genuinely new candidate, not a duplicate). Each entry is a
# full param dict for the base transform's ``apply`` / ``side_conditions``.
_KNOB_SWEEPS: dict[str, list[dict]] = {
    "num_warps":    [{"value": 1}, {"value": 2}, {"value": 16}],
    "num_stages":   [{"value": 1}, {"value": 5}, {"value": 6}],
    "waves_per_eu": [{"value": 1}, {"value": 3}, {"value": 4}],
    "group_m":      [{"value": 2}, {"value": 16}, {"value": 32}],
    "block":        [{"block_m": 128}, {"block_m": 256}, {"block_n": 128},
                     {"block_n": 256}, {"block_k": 128},
                     {"block_m": 128, "block_n": 128}],
    "split_k":      [{"value": 8}, {"value": 16}],
    "dtype":        [{"to": "fp16"}, {"to": "bf16"}, {"to": "fp8"}],
}

_DEFAULT_VECTORIZE_WIDTHS: tuple[int, ...] = (4, 8, 16)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def is_discovered(t: Any) -> bool:
    """True iff ``t`` is a synthesized proposal (name in the discovered namespace)."""
    return str(getattr(t, "name", "")).startswith(DISCOVERED_PREFIX)


def _params_repr(params: dict) -> str:
    return ",".join(f"{k}={params[k]}" for k in sorted(params))


def _base_eps(base: Transformation, params: dict) -> float:
    """The base transform's ε for ``params`` (0 for an exact base). Fail-safe."""
    try:
        if getattr(base, "relation", RELATION_EXACT) == RELATION_APPROX:
            return max(0.0, float(base.epsilon(**params)))
    except Exception:  # a broken base ε model must not break discovery
        return 0.0
    return 0.0


def _applies(t: Transformation, source: str) -> bool:
    """Does proposal ``t`` actually fire on ``source`` (passes conds + changes it)?

    Used only to prune IRRELEVANT proposals when a concrete source is supplied;
    the transforms themselves are always no-op-safe when inapplicable. Never raises.
    """
    try:
        if t.side_conditions(source):
            return False
        new = t.apply(source)
        return new is not None and new != source
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Strategy 1: numeric-knob parameter sweeps (reuse the base transform's rewrite)
# --------------------------------------------------------------------------- #
def _pinned_variant(base: Transformation, params: dict) -> Transformation:
    """A NEW transform that applies ``base`` with ``params`` pinned in.

    Reuses the base's guarded ``apply`` / ``side_conditions`` (so it inherits the
    base's no-op-when-inapplicable + never-raise behavior), but is typed
    conservatively ``approx`` with a floored ε - a proposal, not a proof.
    """
    eps = max(_base_eps(base, params), _DISCOVER_EPS_FLOOR)
    name = f"{DISCOVERED_PREFIX}{base.name}[{_params_repr(params)}]"

    def apply(src: str, **_) -> Optional[str]:
        return base.apply(src, **params)

    def cond(src: str, **_) -> list[str]:
        return base.side_conditions(src, **params)

    return Transformation(
        name=name, relation=RELATION_APPROX, knob=getattr(base, "knob", "unknown"),
        summary=(f"[proposed] parameter-sweep variant of {base.name} pinned "
                 f"{params} - conservatively approx, SNR-gated, NOT verified."),
        apply_fn=apply, cond_fn=cond, default_eps=eps, grid_fn=None,
    )


def _sweep_variants(base: Transformation) -> list[Transformation]:
    knob = getattr(base, "knob", None)
    sweeps = _KNOB_SWEEPS.get(knob or "")
    if not sweeps:
        return []
    out: list[Transformation] = []
    for params in sweeps:
        # Skip param values the BASE already rejects on their own merit (range /
        # power-of-two / multiple-of-64 ...), probed source-free so we isolate
        # param validity from source applicability.
        try:
            if base.side_conditions("", **params):
                continue
        except Exception:
            continue
        try:
            out.append(_pinned_variant(base, params))
        except Exception:
            continue
    return out


def propose_knob_sweeps(
    base_transforms: Sequence[Transformation], *, source: Optional[str] = None
) -> list[Transformation]:
    """Parameter-sweep proposals over the base transforms' numeric knobs."""
    return discover_transforms(
        base_transforms, source=source, sweep_knobs=True,
        enable_fusion=False, enable_vectorize_widths=False)


# --------------------------------------------------------------------------- #
# Strategy 2: vectorization-width contiguity annotations
# --------------------------------------------------------------------------- #
_ARANGE_RE = re.compile(
    r"([A-Za-z_]\w*)\s*=\s*tl\.arange\(\s*0\s*,\s*([A-Za-z_]\w*)\s*\)")


def _make_vectorize_width_apply(width: int):
    w = int(width)

    def apply(src: str, **_) -> Optional[str]:
        for m in _ARANGE_RE.finditer(src):
            pre = src[max(0, m.start() - 48):m.start()]
            if "multiple_of" in pre or "max_contiguous" in pre:
                continue  # already annotated
            lhs, block = m.group(1), m.group(2)
            repl = (f"{lhs} = tl.max_contiguous(tl.multiple_of("
                    f"tl.arange(0, {block}), {w}), {w})")
            return src[:m.start()] + repl + src[m.end():]
        return None

    return apply


def _vectorize_width_transform(width: int) -> Transformation:
    return Transformation(
        name=f"{DISCOVERED_PREFIX}vectorize_width[{int(width)}]",
        relation=RELATION_APPROX, knob="vectorize",
        summary=(f"[proposed] assert contiguity/divisibility width {int(width)} on "
                 f"an offset arange for wide loads - conservatively approx (the "
                 f"width promise is SNR-gated), NOT verified."),
        apply_fn=_make_vectorize_width_apply(width),
        default_eps=_DISCOVER_EPS_FLOOR,
    )


def propose_vectorize_widths(
    widths: Sequence[int] = _DEFAULT_VECTORIZE_WIDTHS, *, source: Optional[str] = None
) -> list[Transformation]:
    """Vectorization-width contiguity-annotation proposals."""
    return discover_transforms(
        [], source=source, sweep_knobs=False, enable_fusion=False,
        enable_vectorize_widths=True, vectorize_widths=widths)


# --------------------------------------------------------------------------- #
# Strategy 3: adjacent-elementwise fusion
# --------------------------------------------------------------------------- #
# Tokens that mark a line as NOT a pure elementwise data expression (loads/stores,
# reductions, and index/launch computation). A line whose RHS contains any of
# these is never inlined as the fused source.
_FUSE_BLOCK = (
    "tl.load", "tl.store", "tl.atomic", "tl.dot", "tl.zeros",
    "tl.reduce", "tl.sum", "tl.max", "tl.min", "tl.cumsum",
    "tl.program_id", "tl.arange", "tl.cdiv", "tl.num_programs",
    "tl.max_contiguous", "tl.multiple_of",
)
_ASSIGN_RE = re.compile(r"^(\s*)([A-Za-z_]\w*)\s*=\s*([^=].*)$")


def _apply_fuse_elementwise(src: str, **_) -> Optional[str]:
    """Fuse two adjacent elementwise assignments by inlining the temporary.

    ``t = <pure elementwise expr>`` immediately followed by a line that consumes
    ``t`` (and ``t`` is used NOWHERE else) becomes the second line with ``t``
    replaced by ``(<expr>)`` - eliminating the temp. Returns ``None`` (no-op) when
    no such safe pair exists. Conservative guards: the inlined RHS must be a pure
    elementwise expression (no loads/stores/reductions/index math), not
    self-referential, comment-free, and the temp must be consumed exactly on the
    next line. Approx: fusing can enable FMA contraction / reassociated rounding.
    """
    lines = src.split("\n")
    for i in range(len(lines) - 1):
        m1 = _ASSIGN_RE.match(lines[i])
        m2 = _ASSIGN_RE.match(lines[i + 1])
        if not (m1 and m2):
            continue
        indent1, name1, rhs1 = m1.group(1), m1.group(2), m1.group(3).rstrip()
        indent2, name2, rhs2 = m2.group(1), m2.group(2), m2.group(3).rstrip()
        if indent1 != indent2 or name1 == name2:
            continue
        if "#" in rhs1 or "#" in rhs2 or not rhs1:
            continue
        if any(tok in rhs1 for tok in _FUSE_BLOCK):
            continue
        if re.search(rf"\b{re.escape(name1)}\b", rhs1):
            continue  # self-referential temp -> unsafe to eliminate
        rest = "\n".join(lines[i + 1:])
        uses_rest = len(re.findall(rf"\b{re.escape(name1)}\b", rest))
        uses_next = len(re.findall(rf"\b{re.escape(name1)}\b", rhs2))
        if uses_next == 0 or uses_rest != uses_next:
            continue  # temp is read elsewhere (or not next) -> not safe to fold
        fused_rhs = re.sub(rf"\b{re.escape(name1)}\b", f"({rhs1})", rhs2)
        new_lines = lines[:i] + [f"{indent2}{name2} = {fused_rhs}"] + lines[i + 2:]
        new = "\n".join(new_lines)
        return new if new != src else None
    return None


def _fusion_transform() -> Transformation:
    return Transformation(
        name=f"{DISCOVERED_PREFIX}fuse_elementwise", relation=RELATION_APPROX,
        knob="fusion",
        summary=("[proposed] fuse two adjacent elementwise ops (inline the temp) - "
                 "conservatively approx (may enable FMA contraction), NOT verified."),
        apply_fn=_apply_fuse_elementwise, default_eps=_FUSION_EPS,
    )


def propose_fusions(*, source: Optional[str] = None) -> list[Transformation]:
    """Adjacent-elementwise-fusion proposal(s)."""
    return discover_transforms(
        [], source=source, sweep_knobs=False, enable_fusion=True,
        enable_vectorize_widths=False)


# --------------------------------------------------------------------------- #
# Orchestration + registry merge
# --------------------------------------------------------------------------- #
def discover_transforms(
    base_transforms: Sequence[Transformation],
    *,
    source: Optional[str] = None,
    sweep_knobs: bool = True,
    enable_fusion: bool = True,
    enable_vectorize_widths: bool = True,
    vectorize_widths: Sequence[int] = _DEFAULT_VECTORIZE_WIDTHS,
    max_proposals: Optional[int] = 64,
) -> list[Transformation]:
    """Synthesize candidate NEW transforms from ``base_transforms``.

    Strategies (each independently toggleable):
      * ``sweep_knobs`` - parameter-sweep variants of the base numeric knobs;
      * ``enable_vectorize_widths`` - pinned contiguity-width annotations;
      * ``enable_fusion`` - adjacent-elementwise fusion.

    Returns a list of well-typed, conservatively-``approx`` proposals, each named
    in the :data:`DISCOVERED_PREFIX` namespace and de-duplicated against the base
    names and each other. When ``source`` is given, proposals that do not actually
    fire on it are pruned (relevance filter); otherwise all well-typed proposals
    are returned (they remain no-op-safe on non-matching kernels). ``max_proposals``
    caps the result (``None`` = unlimited). Never raises; the base library and
    global ``LIBRARY`` are never mutated.

    NOTE: proposals are UNVERIFIED. Correctness is enforced downstream by the env
    SNR oracle, exactly as for the curated library.
    """
    base_list = list(base_transforms or [])
    base_names = {getattr(t, "name", None) for t in base_list}
    out: list[Transformation] = []
    seen: set[str] = set()

    def _add(t: Optional[Transformation]) -> None:
        if t is None:
            return
        name = getattr(t, "name", None)
        if not name or name in seen or name in base_names:
            return
        if source is not None and not _applies(t, source):
            return
        seen.add(name)
        out.append(t)

    if sweep_knobs:
        for base in base_list:
            for cand in _sweep_variants(base):
                _add(cand)
    if enable_vectorize_widths:
        for w in vectorize_widths:
            try:
                _add(_vectorize_width_transform(int(w)))
            except Exception:
                continue
    if enable_fusion:
        _add(_fusion_transform())

    if max_proposals is not None and len(out) > max_proposals:
        out = out[:max_proposals]
    return out


def merge_transforms(
    base_transforms: Sequence[Transformation],
    discovered: Sequence[Transformation],
    *,
    override: bool = False,
) -> list[Transformation]:
    """Registry-merge helper: return a NEW ``base + discovered`` list.

    De-duplicates by ``name``. By default the **base wins** a name collision (a
    proposal never silently shadows a curated transform); ``override=True`` lets a
    discovered transform replace a base entry in place. Base ordering is preserved
    (so the default action-enumeration order is unchanged) with new proposals
    appended. Neither input nor the global ``LIBRARY`` is mutated.
    """
    base = list(base_transforms or [])
    index: dict[Any, int] = {}
    out: list[Transformation] = []
    for t in base:
        index[getattr(t, "name", None)] = len(out)
        out.append(t)
    for t in discovered or []:
        name = getattr(t, "name", None)
        if name in index:
            if override:
                out[index[name]] = t
            continue
        index[name] = len(out)
        out.append(t)
    return out


def extend_library(
    *,
    source: Optional[str] = None,
    base: Optional[Sequence[Transformation]] = None,
    **discover_kwargs: Any,
) -> list[Transformation]:
    """Convenience: discover proposals and registry-merge them into a NEW library.

    ``base`` defaults to the curated :data:`kore.transform.library.LIBRARY`. The
    global default library is **never** mutated - discovery is opt-in: pass the
    returned list to ``admissible_actions(..., library=)`` /
    ``apply_sequence(..., library=)`` / ``TransformProposePolicy(library=)``.
    """
    if base is None:
        from kore.transform.library import LIBRARY  # lazy: no import-time coupling
        base = LIBRARY
    discovered = discover_transforms(base, source=source, **discover_kwargs)
    return merge_transforms(base, discovered)


def describe_proposals(transforms: Sequence[Transformation]) -> list[dict]:
    """Static menu rows (name/relation/knob/summary) for the discovered subset."""
    return [t.as_metadata() for t in transforms if is_discovered(t)]
