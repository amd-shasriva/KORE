"""The KORE transformation library: >=12 verified Triton source rewrites.

Each entry is a :class:`~kore.transform.calculus.Transformation` - a pure
source->source rewrite of a ``@triton.jit`` kernel (+ its Python wrapper),
tagged ``exact`` (≡ bit-preserving) or ``approx`` (≈_ε numeric contract). The
rewrites operate on the *tunable knobs* using the SAME token conventions the
rest of KORE already keys off, WITHOUT importing/modifying those modules:

  * numeric knobs mirror ``kore.policy.format._NUMERIC_KNOB_RE`` -
    ``num_warps``, ``num_stages``, ``BLOCK_M/N/K``, ``GROUP_M`` /
    ``GROUP_SIZE_M``, ``waves_per_eu`` (kwarg, ``: tl.constexpr`` default, and
    positional-tuple forms are all handled, as in ``kore.value.features``);
  * structural tokens mirror ``kore.value.features.extract_schedule_features``
    and ``kore.policy.format._CONSISTENCY_KEYWORDS`` - ``tl.dot`` fusion,
    ``mask=`` boundary guards, ``tl.float32`` accumulation, ``multiple_of`` /
    ``max_contiguous`` vectorization hints, and low-precision IO dtypes.

Exact vs approx is the whole point of the ε-typed calculus:

  EXACT   (≡)   set_num_warps, set_num_stages, set_waves_per_eu, swizzle_group_m,
                vectorize_loads, add_mask_boundary, reorder_loads,
                fp32_accumulator
  APPROX  (≈_ε) retile_block, split_k, downcast_dtype, reassociate_reduction,
                fast_math_recip

Everything is regex/AST-lite over the source string - PURE, deterministic, and
CPU-only. Rewrites return ``None`` when structurally inapplicable so the calculus
can distinguish "illegal params" (side conditions) from "pattern absent".
"""

from __future__ import annotations

import re
from typing import Optional

from kore.transform.budget import RELATION_APPROX, RELATION_EXACT
from kore.transform.calculus import Transformation

# --------------------------------------------------------------------------- #
# Low-level string / knob helpers (mirror the KORE knob token conventions)
# --------------------------------------------------------------------------- #
# Integer knobs that only ever appear as ``name=<int>`` launch kwargs (never as a
# tuple LHS), so a plain kwarg substitution is unambiguous for them.
_LAUNCH_KWARGS = {
    "num_warps", "num_stages", "num_ctas",
    "waves_per_eu", "matrix_instr_nonkdim", "kpack",
}

# A scalar / constexpr-default assignment whose LHS is exactly one name:
#   BLOCK_M = 128            |  BLOCK_M: tl.constexpr = 128  |  num_warps=4
_SCALAR_RE = re.compile(
    r"^(\s*)([A-Za-z_]\w*)((?:\s*:\s*tl\.constexpr)?\s*=\s*)(-?\d+)(.*)$")
# A positional tuple assignment: BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128,128,64,8
_TUPLE_RE = re.compile(
    r"^(\s*)([A-Za-z_][\w\s,]*?)(\s*=\s*)([-\d][\d\s,]*\d|\d)(\s*(?:#.*)?)$")


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _balanced_end(src: str, i: int) -> int:
    """Index of the ``)`` matching the ``(`` at ``src[i]`` (-1 if unbalanced)."""
    if i >= len(src) or src[i] != "(":
        return -1
    depth = 0
    for j in range(i, len(src)):
        c = src[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return j
    return -1


def _iter_calls(src: str, func: str):
    """Yield ``(name_start, open_paren_idx, close_paren_idx)`` for each ``func(...)``
    with balanced parentheses."""
    start = 0
    token = func + "("
    while True:
        i = src.find(token, start)
        if i == -1:
            return
        open_idx = i + len(func)
        close_idx = _balanced_end(src, open_idx)
        if close_idx == -1:
            return
        yield i, open_idx, close_idx
        start = close_idx + 1


def _set_scalar(line: str, name: str, value: int) -> Optional[str]:
    m = _SCALAR_RE.match(line)
    if not m or m.group(2) != name:
        return None
    return f"{m.group(1)}{m.group(2)}{m.group(3)}{value}{m.group(5)}"


def _set_tuple(line: str, name: str, value: int) -> Optional[str]:
    m = _TUPLE_RE.match(line)
    if not m:
        return None
    lhs = [t.strip() for t in m.group(2).split(",")]
    rhs = [t.strip() for t in m.group(4).split(",")]
    if name not in lhs or len(lhs) != len(rhs) or len(lhs) < 2:
        return None
    i = lhs.index(name)
    if not rhs[i].lstrip("-").isdigit():
        return None
    rhs[i] = str(value)
    return f"{m.group(1)}{m.group(2)}{m.group(3)}{', '.join(rhs)}{m.group(5)}"


def _set_knob_in_line(line: str, name: str, value: int) -> Optional[str]:
    if name in _LAUNCH_KWARGS:
        new = re.sub(rf"(\b{re.escape(name)}\s*=\s*)(\d+)", rf"\g<1>{value}", line)
        return new if new != line else None
    r = _set_scalar(line, name, value)
    if r is not None:
        return r
    return _set_tuple(line, name, value)


def _set_knob(src: str, name: str, value: int) -> Optional[str]:
    """Set numeric knob ``name`` to ``value`` across kwarg / constexpr / tuple forms.

    Returns the rewritten source, or ``None`` if the knob is absent OR already at
    ``value`` (a no-op is reported as inapplicable so the calculus never records a
    do-nothing step).
    """
    value = int(value)
    out: list[str] = []
    changed = False
    for line in src.split("\n"):
        nl = _set_knob_in_line(line, name, value)
        if nl is not None and nl != line:
            out.append(nl)
            changed = True
        else:
            out.append(line)
    return "\n".join(out) if changed else None


def _read_knob(src: str, name: str) -> Optional[int]:
    """Current value of numeric knob ``name`` (scalar/constexpr, tuple, or kwarg)."""
    for line in src.split("\n"):
        m = _SCALAR_RE.match(line)
        if m and m.group(2) == name:
            return int(m.group(4))
    for line in src.split("\n"):
        m = _TUPLE_RE.match(line)
        if not m:
            continue
        lhs = [t.strip() for t in m.group(2).split(",")]
        rhs = [t.strip() for t in m.group(4).split(",")]
        if name in lhs and len(lhs) == len(rhs) and len(lhs) >= 2:
            v = rhs[lhs.index(name)]
            if v.lstrip("-").isdigit():
                return int(v)
    m = re.search(rf"\b{re.escape(name)}\s*=\s*(\d+)", src)
    return int(m.group(1)) if m else None


def _add_launch_kwarg(src: str, key: str, value: int) -> Optional[str]:
    """Insert ``key=value`` next to ``num_warps=...`` in the kernel launch."""
    m = re.search(r"\bnum_warps\s*=\s*\d+", src)
    if not m:
        return None
    return src[:m.end()] + f", {key}={value}" + src[m.end():]


# --------------------------------------------------------------------------- #
# Reusable side-condition predicates
# --------------------------------------------------------------------------- #
def _cond_pow2(lo: int, hi: int, label: str):
    def cond(src, value=None, **_):
        if value is None:
            return [f"{label} value required"]
        if not _is_int(value):
            return [f"{label} must be an integer"]
        iv = int(value)
        if iv < lo or iv > hi:
            return [f"{label}={iv} out of range [{lo}, {hi}]"]
        if iv & (iv - 1) != 0:
            return [f"{label}={iv} must be a power of two"]
        return []
    return cond


def _cond_range(lo: int, hi: int, label: str):
    def cond(src, value=None, **_):
        if value is None:
            return [f"{label} value required"]
        if not _is_int(value):
            return [f"{label} must be an integer"]
        iv = int(value)
        if not (lo <= iv <= hi):
            return [f"{label}={iv} out of range [{lo}, {hi}]"]
        return []
    return cond


def _grid_value(name: str, options: tuple[int, ...]):
    """Candidate-``value`` grid for a set-knob action, excluding the current value."""
    def grid(src):
        cur = _read_knob(src, name)
        return [{"value": o} for o in options if o != cur] or [{}]
    return grid


# ===========================================================================
# EXACT transforms (≡ bit-preserving)
# ===========================================================================
def _apply_set_knob(name: str):
    def apply(src, value=None, **_):
        if value is None:
            return None
        return _set_knob(src, name, int(value))
    return apply


def _apply_set_waves(src, value=None, **_):
    # AMD gfx950 occupancy hint: set the existing knob or wire it into the launch.
    if value is None:
        return None
    r = _set_knob(src, "waves_per_eu", int(value))
    if r is not None:
        return r
    return _add_launch_kwarg(src, "waves_per_eu", int(value))


def _apply_swizzle_group_m(src, value=None, **_):
    if value is None:
        return None
    r = _set_knob(src, "GROUP_M", int(value))
    if r is not None:
        return r
    return _set_knob(src, "GROUP_SIZE_M", int(value))


def _grid_group_m(src):
    cur = _read_knob(src, "GROUP_M")
    if cur is None:
        cur = _read_knob(src, "GROUP_SIZE_M")
    return [{"value": o} for o in (1, 4, 8, 16) if o != cur] or [{}]


_ARANGE_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*tl\.arange\(\s*0\s*,\s*([A-Za-z_]\w*)\s*\)")


def _apply_vectorize(src, **_):
    """Assert contiguity on an offset ``tl.arange`` so Triton emits wide, coalesced
    loads (``tl.max_contiguous(tl.multiple_of(...))``). Compiler HINT only - it
    changes no numeric value, hence exact."""
    for m in _ARANGE_RE.finditer(src):
        pre = src[max(0, m.start() - 48):m.start()]
        if "multiple_of" in pre or "max_contiguous" in pre:
            continue  # this arange is already annotated
        lhs, block = m.group(1), m.group(2)
        repl = (f"{lhs} = tl.max_contiguous(tl.multiple_of("
                f"tl.arange(0, {block}), {block}), {block})")
        return src[:m.start()] + repl + src[m.end():]
    return None


def _boundary_guard(src: str) -> Optional[str]:
    """Find an existing boolean-mask expression to reuse as a boundary guard."""
    # (a) an explicit *mask* variable assigned from a comparison
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*([^=\n]*[<>][^=\n]*)$", src, re.M):
        if "mask" in m.group(1).lower():
            return m.group(1)
    # (b) reuse an existing ``mask=<ident>`` kwarg (not the ``mask = a < b`` assign)
    m = re.search(r"\bmask\s*=\s*([A-Za-z_]\w*)\b(?!\s*[<>])", src)
    if m:
        return m.group(1)
    # (c) synthesize the canonical elementwise guard ``offsets < n_elements``
    if re.search(r"\bn_elements\b", src):
        om = re.search(r"\b(offsets|offs|offset|idx|offs_\w+)\b", src)
        if om:
            return f"{om.group(1)} < n_elements"
    return None


def _apply_add_mask(src, **_):
    """Add a boundary ``mask=`` (and ``other=0.0`` for loads) to the first
    unmasked ``tl.store`` / ``tl.load``. For an in-bounds kernel this is exact
    (the masked-off lanes were already never the result)."""
    guard = _boundary_guard(src)
    if guard is None:
        return None
    for func, extra in (("tl.store", ""), ("tl.load", ", other=0.0")):
        for _i, o, c in _iter_calls(src, func):
            args = src[o + 1:c]
            if "mask=" in args or not args.strip():
                continue
            new_args = args.rstrip() + f", mask={guard}{extra}"
            return src[:o + 1] + new_args + src[c:]
    return None


_LOAD_LINE_RE = re.compile(r"^(\s*)([A-Za-z_]\w*)\s*=\s*tl\.load\(")


def _apply_reorder_loads(src, **_):
    """Swap two adjacent, DATA-INDEPENDENT ``= tl.load(...)`` statements (a
    non-reduction layout/loop reorder). Independent loads commute, so this is
    bit-exact; a dependent pair is left untouched."""
    lines = src.split("\n")
    for i in range(len(lines) - 1):
        m1 = _LOAD_LINE_RE.match(lines[i])
        m2 = _LOAD_LINE_RE.match(lines[i + 1])
        if not (m1 and m2) or m1.group(1) != m2.group(1):
            continue
        lhs1, lhs2 = m1.group(2), m2.group(2)
        if lhs1 == lhs2:
            continue
        rhs1 = lines[i].split("=", 1)[1]
        rhs2 = lines[i + 1].split("=", 1)[1]
        if re.search(rf"\b{lhs1}\b", rhs2) or re.search(rf"\b{lhs2}\b", rhs1):
            continue  # data-dependent: reorder would change semantics
        lines[i], lines[i + 1] = lines[i + 1], lines[i]
        return "\n".join(lines)
    return None


def _apply_fp32_acc(src, **_):
    """Force the reduction accumulator to fp32 (CDNA4 discipline). Raising acc
    precision only STRENGTHENS the numeric contract, so relative to the fp32
    reference oracle this is exact."""
    for _i, o, c in _iter_calls(src, "tl.zeros"):
        args = src[o + 1:c]
        m = re.search(r"(?:dtype\s*=\s*tl\.|,\s*tl\.)(bfloat16|float16|float8\w*)", args)
        if m and m.group(1) != "float32":
            new_args = args[:m.start(1)] + "float32" + args[m.end(1):]
            return src[:o + 1] + new_args + src[c:]
    return None


# ===========================================================================
# APPROX transforms (≈_ε numeric contract)
# ===========================================================================
_BLOCK_KNOB = {"block_m": "BLOCK_M", "block_n": "BLOCK_N", "block_k": "BLOCK_K"}


def _cond_retile(src, block_m=None, block_n=None, block_k=None, group_m=None, **_):
    v: list[str] = []
    if all(x is None for x in (block_m, block_n, block_k, group_m)):
        v.append("no BLOCK/GROUP dimension provided")
    for label, val in (("BLOCK_M", block_m), ("BLOCK_N", block_n), ("BLOCK_K", block_k)):
        if val is None:
            continue
        if not _is_int(val):
            v.append(f"{label} must be an integer")
            continue
        iv = int(val)
        if iv <= 0:
            v.append(f"{label} must be positive")
        elif iv % 64 != 0:
            v.append(f"{label}={iv} must be a multiple of 64 "
                     f"(gfx950 wavefront=64 / MFMA tiling)")
        elif iv > 512:
            v.append(f"{label}={iv} exceeds 512 (VGPR/LDS pressure)")
    if group_m is not None:
        if not _is_int(group_m):
            v.append("GROUP_M must be an integer")
        elif int(group_m) < 1:
            v.append("GROUP_M must be >= 1")
    return v


def _apply_retile(src, block_m=None, block_n=None, block_k=None, group_m=None, **_):
    new = src
    changed = False
    for pname, val in (("block_m", block_m), ("block_n", block_n), ("block_k", block_k)):
        if val is None:
            continue
        r = _set_knob(new, _BLOCK_KNOB[pname], int(val))
        if r is not None:
            new, changed = r, True
    if group_m is not None:
        r = _set_knob(new, "GROUP_M", int(group_m)) or _set_knob(new, "GROUP_SIZE_M", int(group_m))
        if r is not None:
            new, changed = r, True
    return new if (changed and new != src) else None


def _eps_retile(block_m=None, block_n=None, block_k=None, group_m=None, **_):
    # BLOCK_M/N only re-tile the parallel (M,N) grid -> no reassociation -> tiny ε.
    # BLOCK_K re-tiles the K reduction -> changes the summation order -> larger ε.
    eps = 0.0
    if block_m is not None or block_n is not None:
        eps = max(eps, 0.02)
    if block_k is not None:
        eps = max(eps, 0.06)
    return eps


def _grid_retile(src):
    out = []
    bm, bn, bk = _read_knob(src, "BLOCK_M"), _read_knob(src, "BLOCK_N"), _read_knob(src, "BLOCK_K")
    if bm is not None:
        out.append({"block_m": 128 if bm != 128 else 256})
    if bn is not None:
        out.append({"block_n": 256 if bn != 256 else 128})
    if bk is not None:
        out.append({"block_k": 128 if bk != 128 else 64})
    return out or [{}]


_KLOOP_CDIV_RE = re.compile(
    r"(for\s+(\w+)\s+in\s+range\(\s*0\s*,\s*tl\.cdiv\(\s*K\s*,\s*BLOCK_K\s*\))(\s*\)\s*:)")


def _store_to_atomic(src: str) -> str:
    """Convert the epilogue ``tl.store`` into an ``tl.atomic_add`` (split partials
    accumulate into the output)."""
    idx = src.rfind("tl.store(")
    if idx == -1:
        return src
    return src[:idx] + "tl.atomic_add(" + src[idx + len("tl.store("):]


def _apply_split_k(src, value=2, **_):
    """Split the K reduction ``value`` ways (a second grid axis) with atomic
    accumulation of the partials. Distinct partial sums re-order the reduction, so
    the result is ≈_ε rather than bit-exact."""
    v = int(value)
    if "SPLIT_K" in src:
        return None  # already split
    m = _KLOOP_CDIV_RE.search(src)
    if not m:
        return None
    loopvar = m.group(2)
    loop_repl = (f"for {loopvar} in range(pid_k, tl.cdiv(K, BLOCK_K), SPLIT_K)"
                 + m.group(3))
    new = src[:m.start()] + loop_repl + src[m.end():]

    # wire SPLIT_K into the kernel signature (after GROUP_M / BLOCK_K constexpr)
    if re.search(r"GROUP_M\s*:\s*tl\.constexpr\s*,", new):
        new = re.sub(r"(GROUP_M\s*:\s*tl\.constexpr\s*,)",
                     r"\1 SPLIT_K: tl.constexpr,", new, count=1)
    elif re.search(r"BLOCK_K\s*:\s*tl\.constexpr\s*,", new):
        new = re.sub(r"(BLOCK_K\s*:\s*tl\.constexpr\s*,)",
                     r"\1 SPLIT_K: tl.constexpr,", new, count=1)
    else:
        return None  # nowhere to declare the knob

    # second program axis for the split
    new = re.sub(
        r"(pid\s*=\s*tl\.program_id\(\s*0\s*\)\s*\n)(\s*)",
        lambda mm: f"{mm.group(1)}{mm.group(2)}pid_k = tl.program_id(axis=1)\n{mm.group(2)}",
        new, count=1)

    # pre-offset each K pointer by this split's start, then stride by SPLIT_K
    advances = list(re.finditer(r"(\w+)\s*\+=\s*BLOCK_K\s*\*\s*(\w+)", new))
    loop_anchor = re.search(r"\n(\s*)for\s+\w+\s+in\s+range\([^\n]*SPLIT_K[^\n]*:\n", new)
    if advances and loop_anchor:
        indent = loop_anchor.group(1)
        pre = "".join(f"{indent}{a.group(1)} += pid_k * BLOCK_K * {a.group(2)}\n"
                      for a in advances)
        li = loop_anchor.start() + 1
        new = new[:li] + pre + new[li:]
    new = re.sub(r"(\w+)\s*\+=\s*BLOCK_K\s*\*\s*(\w+)",
                 r"\1 += SPLIT_K * BLOCK_K * \2", new)

    # epilogue store -> atomic; host launch grid + SPLIT_K kwarg
    new = _store_to_atomic(new)
    new = re.sub(r"(grid\s*=\s*\([^\n]*?)\s*,\s*\)", r"\1, SPLIT_K)", new, count=1)
    new = re.sub(r"(\bnum_warps\s*=\s*\d+)", rf"SPLIT_K={v}, \1", new, count=1)
    return new if new != src else None


def _cond_split_k(src, value=2, **_):
    if not _is_int(value):
        return ["SPLIT_K must be an integer"]
    iv = int(value)
    if iv < 2:
        return [f"SPLIT_K={iv} must be >= 2"]
    if iv & (iv - 1) != 0:
        return [f"SPLIT_K={iv} must be a power of two"]
    if iv > 16:
        return [f"SPLIT_K={iv} exceeds 16 (atomic contention)"]
    return []


def _eps_split_k(value=2, **_):
    return {2: 0.05, 4: 0.10, 8: 0.16, 16: 0.24}.get(int(value), 0.08)


# low-precision IO dtype maps (Triton + torch); fp8 is CDNA4 OCP e4m3fn
_TL_DT = {"bf16": "tl.bfloat16", "fp16": "tl.float16", "fp8": "tl.float8e4m3fn"}
_TORCH_DT = {"bf16": "torch.bfloat16", "fp16": "torch.float16", "fp8": "torch.float8_e4m3fn"}
_LOWP = set(_TL_DT)


def _has_nonfp32_acc(src: str) -> bool:
    for _i, o, c in _iter_calls(src, "tl.zeros"):
        args = src[o + 1:c]
        if re.search(r"(?:dtype\s*=\s*tl\.|,\s*tl\.)(bfloat16|float16|float8\w*)", args):
            return True
    return False


def _cond_downcast(src, to=None, **_):
    v: list[str] = []
    if to is None:
        return ["downcast target dtype required (bf16|fp16|fp8)"]
    if to not in _LOWP:
        v.append(f"target {to!r} must be one of bf16/fp16/fp8 (downcast only)")
    # Downcasting IO is only safe while the reduction still accumulates in fp32;
    # compose with fp32_accumulator first if the acc is low precision.
    if _has_nonfp32_acc(src):
        v.append("accumulator is not fp32; apply fp32_accumulator before downcasting IO")
    return v


def _apply_downcast(src, to=None, **_):
    if to not in _TL_DT:
        return None
    tgt_tl, tgt_torch = _TL_DT[to], _TORCH_DT[to]
    new = src
    for tl_name in _TL_DT.values():          # store-side ``.to(tl.<lowp>)`` casts
        if tl_name != tgt_tl:
            new = new.replace(f".to({tl_name})", f".to({tgt_tl})")
    for tn in _TORCH_DT.values():            # host output allocation dtype
        if tn != tgt_torch:
            new = new.replace(f"dtype={tn}", f"dtype={tgt_torch}")
    return new if new != src else None


def _eps_downcast(to=None, **_):
    return {"fp16": 0.03, "bf16": 0.05, "fp8": 0.15}.get(to, 0.08)


def _apply_reassociate(src, **_):
    """Fold the K-accumulate into the MFMA (``acc += tl.dot(a,b)`` ->
    ``acc = tl.dot(a,b,acc)``). Fused vs separate rounding of the running sum is a
    genuine reassociation of the reduction, hence ≈_ε."""
    for pat in (r"^(\s*)(\w+)\s*\+=\s*tl\.dot\(",
                r"^(\s*)(\w+)\s*=\s*\2\s*\+\s*tl\.dot\("):
        m = re.search(pat, src, re.M)
        if not m:
            continue
        indent, acc = m.group(1), m.group(2)
        open_idx = m.end() - 1  # the '(' of tl.dot(
        close_idx = _balanced_end(src, open_idx)
        if close_idx == -1:
            continue
        args = src[open_idx + 1:close_idx].strip()
        repl = f"{indent}{acc} = tl.dot({args}, {acc})"
        return src[:m.start()] + repl + src[close_idx + 1:]
    return None


def _apply_fast_recip(src, **_):
    """Replace a true reciprocal ``1.0 / x`` with the hardware fast reciprocal
    ``tl.math.rcp(x)`` (lower-precision, ≈_ε)."""
    m = re.search(r"\b1(?:\.0)?\s*/\s*\(", src)
    if m:
        open_idx = src.index("(", m.end() - 1)
        close_idx = _balanced_end(src, open_idx)
        if close_idx != -1:
            inner = src[open_idx + 1:close_idx]
            return src[:m.start()] + f"tl.math.rcp({inner})" + src[close_idx + 1:]
    m = re.search(r"\b1(?:\.0)?\s*/\s*([A-Za-z_]\w*(?:\[[^\]]*\])?)", src)
    if m:
        return src[:m.start()] + f"tl.math.rcp({m.group(1)})" + src[m.end():]
    return None


# ===========================================================================
# The library
# ===========================================================================
LIBRARY: list[Transformation] = [
    # ---- EXACT (≡) --------------------------------------------------------
    Transformation(
        name="set_num_warps", relation=RELATION_EXACT, knob="num_warps",
        summary="Set num_warps (wavefront parallelism / occupancy).",
        apply_fn=_apply_set_knob("num_warps"),
        cond_fn=_cond_pow2(1, 16, "num_warps"),
        grid_fn=_grid_value("num_warps", (4, 8)),
    ),
    Transformation(
        name="set_num_stages", relation=RELATION_EXACT, knob="num_stages",
        summary="Set num_stages (software pipelining / LDS double-buffering).",
        apply_fn=_apply_set_knob("num_stages"),
        cond_fn=_cond_range(1, 6, "num_stages"),
        grid_fn=_grid_value("num_stages", (2, 3, 4)),
    ),
    Transformation(
        name="set_waves_per_eu", relation=RELATION_EXACT, knob="waves_per_eu",
        summary="Set waves_per_eu (gfx950 occupancy hint).",
        apply_fn=_apply_set_waves,
        cond_fn=_cond_range(1, 8, "waves_per_eu"),
        grid_fn=_grid_value("waves_per_eu", (1, 2)),
    ),
    Transformation(
        name="swizzle_group_m", relation=RELATION_EXACT, knob="group_m",
        summary="Set GROUP_M / GROUP_SIZE_M (L2 super-grouping / program swizzle).",
        apply_fn=_apply_swizzle_group_m,
        cond_fn=_cond_range(1, 32, "GROUP_M"),
        grid_fn=_grid_group_m,
    ),
    Transformation(
        name="vectorize_loads", relation=RELATION_EXACT, knob="vectorize",
        summary="Annotate offsets with tl.max_contiguous/tl.multiple_of for "
                "coalesced wide loads.",
        apply_fn=_apply_vectorize,
    ),
    Transformation(
        name="add_mask_boundary", relation=RELATION_EXACT, knob="mask",
        summary="Add a boundary mask= (and other=0.0) to an unmasked load/store.",
        apply_fn=_apply_add_mask,
    ),
    Transformation(
        name="reorder_loads", relation=RELATION_EXACT, knob="layout",
        summary="Swap two adjacent, independent tl.load statements "
                "(non-reduction layout reorder).",
        apply_fn=_apply_reorder_loads,
    ),
    Transformation(
        name="fp32_accumulator", relation=RELATION_EXACT, knob="accumulator",
        summary="Force the tl.zeros reduction accumulator to tl.float32.",
        apply_fn=_apply_fp32_acc,
    ),
    # ---- APPROX (≈_ε) -----------------------------------------------------
    Transformation(
        name="retile_block", relation=RELATION_APPROX, knob="block",
        summary="Re-tile BLOCK_M/N/K (incl. K-split); BLOCK_K change reassociates "
                "the reduction.",
        apply_fn=_apply_retile, cond_fn=_cond_retile, eps_fn=_eps_retile,
        grid_fn=_grid_retile, default_eps=0.04,
    ),
    Transformation(
        name="split_k", relation=RELATION_APPROX, knob="split_k",
        summary="Split the K reduction across a second grid axis with atomic "
                "accumulation of partials.",
        apply_fn=_apply_split_k, cond_fn=_cond_split_k, eps_fn=_eps_split_k,
        grid_fn=lambda src: [{"value": 2}, {"value": 4}], default_eps=0.05,
    ),
    Transformation(
        name="downcast_dtype", relation=RELATION_APPROX, knob="dtype",
        summary="Downcast IO to bf16/fp16/fp8 while keeping the fp32 accumulator.",
        apply_fn=_apply_downcast, cond_fn=_cond_downcast, eps_fn=_eps_downcast,
        grid_fn=lambda src: [{"to": "fp16"}, {"to": "bf16"}, {"to": "fp8"}],
        default_eps=0.08,
    ),
    Transformation(
        name="reassociate_reduction", relation=RELATION_APPROX, knob="reduction",
        summary="Fuse the accumulate into tl.dot (acc += tl.dot -> "
                "tl.dot(...,acc)); reassociates the sum.",
        apply_fn=_apply_reassociate, default_eps=0.01,
    ),
    Transformation(
        name="fast_math_recip", relation=RELATION_APPROX, knob="fast_math",
        summary="Replace 1.0/x with the fast hardware reciprocal tl.math.rcp(x).",
        apply_fn=_apply_fast_recip, default_eps=0.03,
    ),
]

BY_NAME: dict[str, Transformation] = {t.name: t for t in LIBRARY}
EXACT: list[Transformation] = [t for t in LIBRARY if t.is_exact()]
APPROX: list[Transformation] = [t for t in LIBRARY if t.is_approx()]


def get(name: str) -> Optional[Transformation]:
    """Look up a transformation by name."""
    return BY_NAME.get(name)


def default_library() -> list[Transformation]:
    """The full library list (stable order)."""
    return list(LIBRARY)
