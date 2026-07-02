"""KORE value-model featurization.

Turns a candidate-kernel *move* (a small dict of static metadata known BEFORE we
pay for a GPU benchmark) into a fixed-length numeric feature vector. This is the
input to the cheap surrogate ValueModel that ranks candidates so only the top-k
are benched.

Design analogue (KORE.pdf Sec 4.5):
  - Ansor's learned cost model featurizes a loop-nest schedule and predicts its
    throughput without running it, so autotuning explores far fewer real
    measurements. Here the "schedule" is a proposed kernel edit and its context
    (operation, shape, dtype, parent stats, PMC bottleneck).
  - Compiler / world-model work (Compiler-World-Models, TVM) shows a static,
    cheap predictor of a measurement is enough to prune the search and cut the
    number of expensive evaluations several-fold.

Two feature families are concatenated into one fixed-length vector:
  1. PROBLEM/CONTEXT features (operation, shape, dtype, parent stats, PMC
     bottleneck) — what the move is being applied *to*.
  2. CANDIDATE-SCHEDULE features extracted from the kernel SOURCE itself (BLOCK
     sizes, num_warps/num_stages, tiling area, vectorization width, tl.dot/MFMA
     presence, LDS/pipeline hints, loop structure). This is the Ansor/NLTSP move:
     the cost model is *action-conditioned* — it sees the actual schedule the
     candidate encodes, not only the problem it targets. When a move carries no
     source these features are all zero, so the vector layout (and every existing
     model) stays backward-compatible.

Everything here is PURE (numpy + re only) and deterministic, so it is trivially
testable and reproducible. Categoricals are one-hot encoded; sizes are
log-scaled (kernel cost scales multiplicatively with problem size, so log space
is the natural feature space, exactly as Ansor log-transforms tile extents).
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import numpy as np

# --- Categorical vocabularies (fixed so the feature layout is stable) --------
# Unknown / out-of-vocab values fall into a dedicated bucket so featurize never
# changes vector length.
OPERATIONS: tuple[str, ...] = (
    "gemm",
    "matmul",
    "conv",
    "attention",
    "reduction",
    "elementwise",
    "norm",
)
DTYPES: tuple[str, ...] = ("fp32", "fp16", "bf16", "fp8", "int8")
BOTTLENECKS: tuple[str, ...] = ("compute", "balanced", "memory", "unknown")

# Number of shape dims we log-scale. GEMM uses (M, N, K); generic ops pad/truncate
# their `dims` list to this many entries.
_MAX_DIMS = 4


def _norm_dtype(dtype: str) -> str:
    d = (dtype or "").lower()
    if "bf16" in d or "bfloat16" in d:
        return "bf16"
    if "fp16" in d or "float16" in d or "half" in d:
        return "fp16"
    if "fp8" in d or "float8" in d:
        return "fp8"
    if "int8" in d or "i8" in d:
        return "int8"
    if "fp32" in d or "float32" in d or "float" in d or d == "":
        return "fp32"
    return "fp32"


def _norm_operation(op: str) -> str:
    o = (op or "").lower()
    for known in OPERATIONS:
        if known in o:
            return known
    return ""  # out-of-vocab -> all-zero one-hot block


def _norm_bottleneck(b: str) -> str:
    v = (b or "").lower()
    if v in BOTTLENECKS:
        return v
    return "unknown"


def _extract_dims(meta: dict) -> list[float]:
    """Return up to _MAX_DIMS problem dimensions from a meta dict.

    Accepts either explicit M/N/K or a generic `dims`/`shape` list.
    """
    dims: list[float] = []
    if any(k in meta for k in ("M", "N", "K")):
        for k in ("M", "N", "K"):
            v = meta.get(k)
            if v is not None:
                dims.append(float(v))
    generic = meta.get("dims") or meta.get("shape")
    if generic is not None:
        try:
            dims.extend(float(x) for x in generic)
        except TypeError:
            dims.append(float(generic))
    # pad / truncate to fixed width
    dims = dims[:_MAX_DIMS]
    while len(dims) < _MAX_DIMS:
        dims.append(0.0)
    return dims


def _log1p_pos(x: float) -> float:
    """log1p of a non-negative quantity; negatives clamped to 0."""
    return math.log1p(max(0.0, float(x)))


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


# --------------------------------------------------------------------------- #
# Candidate-schedule extraction (Ansor / NLTSP: featurize the SCHEDULE)
#
# These parse the proposed kernel source for the knobs a cost model needs to be
# action-conditioned. All patterns are best-effort and side-effect-free; a knob
# that isn't found is simply absent (None / 0), so a partial or exotic kernel
# still featurizes cleanly.
# --------------------------------------------------------------------------- #
_BLOCK_NAMES = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M")

# Names whose schedule value we surface as an ordered numeric block. Keep this in
# sync with SCHEDULE_FEATURE_NAMES below.
SCHEDULE_FEATURE_NAMES: list[str] = [
    "sched_has_source",
    "sched_log_block_m",
    "sched_log_block_n",
    "sched_log_block_k",
    "sched_log_group_m",
    "sched_log_tile_area",
    "sched_blocks_mult64",
    "sched_blocks_pow2",
    "sched_log_num_warps",
    "sched_log_num_stages",
    "sched_has_tl_dot",
    "sched_has_mfma",
    "sched_has_fp32_acc",
    "sched_has_mask",
    "sched_has_reduction_loop",
    "sched_log_n_loads",
    "sched_log_n_stores",
    "sched_log_n_loops",
    "sched_log_vec_width",
]


def _scalar_assign(src: str, name: str):
    """First *simple* assignment / constexpr default of `name` (e.g. BLOCK_M=128).

    Only accepts a line whose left-hand side is exactly ``name`` (optionally with
    a ``: tl.constexpr`` annotation) so a tuple LHS like
    ``BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = ...`` never mis-binds via its tail name.
    """
    for line in src.splitlines():
        if "=" not in line:
            continue
        lhs = line.split("=", 1)[0]
        lhs_name = lhs.split(":", 1)[0].strip()
        if lhs_name == name:
            m = re.search(r"=\s*(\d+)", line)
            if m:
                return int(m.group(1))
    return None


def _tuple_assign(src: str, name: str):
    """Value of `name` when set positionally in a tuple assignment, e.g.
    ``BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8``."""
    for m in re.finditer(
        r"^[ \t]*([A-Za-z_][\w\s,]*?)\s*=\s*([-\d][\d\s,]*\d|\d)\s*$",
        src,
        re.MULTILINE,
    ):
        lhs = [t.strip() for t in m.group(1).split(",") if t.strip()]
        rhs = [t.strip() for t in m.group(2).split(",") if t.strip()]
        if name in lhs and len(lhs) == len(rhs):
            try:
                return int(rhs[lhs.index(name)])
            except ValueError:
                continue
    return None


def _block_value(src: str, name: str):
    v = _scalar_assign(src, name)
    if v is not None:
        return v
    return _tuple_assign(src, name)


def _kwarg_int(src: str, name: str):
    m = re.search(rf"\b{name}\b\s*=\s*(\d+)", src)
    return int(m.group(1)) if m else None


def extract_schedule_features(source: str) -> dict:
    """Parse a candidate kernel SOURCE into raw schedule knobs.

    Returns a plain dict of the schedule descriptors used by :func:`featurize`
    (and by the untrained heuristic ranker in ``rerank``). Missing knobs are
    ``None``/``0`` so a partial kernel still featurizes.
    """
    src = source or ""
    has_source = bool(src.strip())
    bm = _block_value(src, "BLOCK_M")
    bn = _block_value(src, "BLOCK_N")
    bk = _block_value(src, "BLOCK_K")
    gm = _block_value(src, "GROUP_M")
    blocks = [b for b in (bm, bn, bk) if b is not None]

    num_warps = _kwarg_int(src, "num_warps")
    num_stages = _kwarg_int(src, "num_stages")

    has_tl_dot = bool(re.search(r"\btl\.dot\s*\(", src))
    has_mfma = bool(re.search(r"mfma|matrix_core|v_mfma", src, re.IGNORECASE))
    has_fp32_acc = bool(
        re.search(r"tl\.zeros\([^)]*float32", src)
        or re.search(r"dtype\s*=\s*tl\.float32", src)
        or re.search(r"\.to\(tl\.float32\)", src)
    )
    has_mask = bool(re.search(r"\bmask\s*=", src))
    # a reduction loop over the K contraction dimension (software pipelining site)
    has_reduction_loop = bool(
        re.search(r"for\s+\w+\s+in\s+range\([^)]*\bK\b", src)
        or re.search(r"for\s+\w+\s+in\s+range\([^)]*BLOCK_K", src)
    )
    n_loads = len(re.findall(r"\btl\.load\s*\(", src))
    n_stores = len(re.findall(r"\btl\.store\s*\(", src))
    n_loops = len(re.findall(r"\bfor\s+\w+\s+in\s+", src))

    tile_area = (bm * bn) if (bm is not None and bn is not None) else None
    blocks_mult64 = 1.0 if (blocks and all(b % 64 == 0 for b in blocks)) else 0.0
    blocks_pow2 = 1.0 if (blocks and all(_is_pow2(b) for b in blocks)) else 0.0
    # contiguous vectorization width: the innermost (K) tile drives coalesced
    # loads; fall back to the N tile, else 0.
    vec_width = bk if bk is not None else (bn if bn is not None else 0)

    return {
        "has_source": has_source,
        "block_m": bm,
        "block_n": bn,
        "block_k": bk,
        "group_m": gm,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "has_tl_dot": has_tl_dot,
        "has_mfma": has_mfma,
        "has_fp32_acc": has_fp32_acc,
        "has_mask": has_mask,
        "has_reduction_loop": has_reduction_loop,
        "n_loads": n_loads,
        "n_stores": n_stores,
        "n_loops": n_loops,
        "tile_area": tile_area,
        "blocks_mult64": blocks_mult64,
        "blocks_pow2": blocks_pow2,
        "vec_width": vec_width,
    }


def _schedule_vector(source: str) -> list[float]:
    s = extract_schedule_features(source)
    return [
        1.0 if s["has_source"] else 0.0,
        _log1p_pos(s["block_m"] or 0),
        _log1p_pos(s["block_n"] or 0),
        _log1p_pos(s["block_k"] or 0),
        _log1p_pos(s["group_m"] or 0),
        _log1p_pos(s["tile_area"] or 0),
        float(s["blocks_mult64"]),
        float(s["blocks_pow2"]),
        _log1p_pos(s["num_warps"] or 0),
        _log1p_pos(s["num_stages"] or 0),
        1.0 if s["has_tl_dot"] else 0.0,
        1.0 if s["has_mfma"] else 0.0,
        1.0 if s["has_fp32_acc"] else 0.0,
        1.0 if s["has_mask"] else 0.0,
        1.0 if s["has_reduction_loop"] else 0.0,
        _log1p_pos(s["n_loads"]),
        _log1p_pos(s["n_stores"]),
        _log1p_pos(s["n_loops"]),
        _log1p_pos(s["vec_width"] or 0),
    ]


def _build_feature_names() -> list[str]:
    names: list[str] = []
    names += [f"op={o}" for o in OPERATIONS]
    names += [f"dtype={d}" for d in DTYPES]
    names += [f"bottleneck={b}" for b in BOTTLENECKS]
    names += [f"log_dim{i}" for i in range(_MAX_DIMS)]
    names += ["log_total_size", "dim_aspect_log"]
    names += [
        "log_diff_size",
        "parent_snr",
        "parent_snr_margin",
        "log_parent_wall_ms",
        "log_parent_vgpr",
        "has_parent",
    ]
    # candidate-schedule block (Ansor/NLTSP action-conditioned features)
    names += list(SCHEDULE_FEATURE_NAMES)
    return names


FEATURE_NAMES: list[str] = _build_feature_names()
N_FEATURES: int = len(FEATURE_NAMES)


def featurize(meta: dict) -> np.ndarray:
    """Featurize one candidate move into a fixed-length float32 vector.

    `meta` keys (all optional; sensible defaults applied):
        operation (str), M/N/K or dims/shape, dtype (str),
        diff_size (int, chars changed vs parent),
        parent_snr (float), parent_wall_ms (float), parent_vgpr (int),
        pmc_bottleneck (str in {compute, balanced, memory, unknown}),
        source (str): the candidate kernel source. When present, the
            candidate-schedule block (BLOCK sizes, num_warps/num_stages, tiling,
            vectorization width, tl.dot/MFMA, LDS/pipeline hints, loop structure)
            is extracted so the model is action/schedule-conditioned. When
            absent, that block is all-zero and the vector is unchanged from the
            problem-only layout (backward-compatible).
    """
    meta = meta or {}
    vec: list[float] = []

    # --- one-hot: operation ---
    op = _norm_operation(meta.get("operation", ""))
    vec += [1.0 if op == o else 0.0 for o in OPERATIONS]

    # --- one-hot: dtype ---
    dt = _norm_dtype(meta.get("dtype", ""))
    vec += [1.0 if dt == d else 0.0 for d in DTYPES]

    # --- one-hot: pmc bottleneck ---
    bn = _norm_bottleneck(meta.get("pmc_bottleneck", "unknown"))
    vec += [1.0 if bn == b else 0.0 for b in BOTTLENECKS]

    # --- log-scaled sizes ---
    dims = _extract_dims(meta)
    vec += [_log1p_pos(d) for d in dims]
    total = 1.0
    nonzero = [d for d in dims if d > 0]
    for d in nonzero:
        total *= d
    vec.append(math.log1p(total) if nonzero else 0.0)
    # aspect ratio (max/min) captures skewed shapes that stress memory vs compute
    if len(nonzero) >= 2:
        aspect = max(nonzero) / max(min(nonzero), 1e-9)
    else:
        aspect = 1.0
    vec.append(math.log1p(aspect))

    # --- edit / parent context ---
    vec.append(_log1p_pos(meta.get("diff_size", 0)))
    parent_snr = meta.get("parent_snr")
    has_parent = parent_snr is not None or meta.get("parent_wall_ms") is not None
    parent_snr_val = float(parent_snr) if parent_snr is not None else 0.0
    vec.append(parent_snr_val)
    # margin above a nominal 30 dB pass threshold (how much slack the parent had)
    vec.append(parent_snr_val - 30.0 if parent_snr is not None else 0.0)
    vec.append(_log1p_pos(meta.get("parent_wall_ms", 0.0)))
    vec.append(_log1p_pos(meta.get("parent_vgpr", 0)))
    vec.append(1.0 if has_parent else 0.0)

    # --- candidate-schedule block (from the kernel SOURCE, if provided) ---
    vec += _schedule_vector(meta.get("source", "") or "")

    arr = np.asarray(vec, dtype=np.float32)
    assert arr.shape[0] == N_FEATURES, (
        f"feature length {arr.shape[0]} != {N_FEATURES}; FEATURE_NAMES out of sync"
    )
    return arr


def featurize_many(metas: Sequence[dict] | Iterable[dict]) -> np.ndarray:
    """Featurize a batch of moves into an (n, N_FEATURES) float32 matrix."""
    rows = [featurize(m) for m in metas]
    if not rows:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    return np.stack(rows, axis=0)
