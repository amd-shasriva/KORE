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

Everything here is PURE (numpy only) and deterministic, so it is trivially
testable and reproducible. Categoricals are one-hot encoded; sizes are
log-scaled (kernel cost scales multiplicatively with problem size, so log space
is the natural feature space, exactly as Ansor log-transforms tile extents).
"""

from __future__ import annotations

import math
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
    return names


FEATURE_NAMES: list[str] = _build_feature_names()
N_FEATURES: int = len(FEATURE_NAMES)


def featurize(meta: dict) -> np.ndarray:
    """Featurize one candidate move into a fixed-length float32 vector.

    `meta` keys (all optional; sensible defaults applied):
        operation (str), M/N/K or dims/shape, dtype (str),
        diff_size (int, chars changed vs parent),
        parent_snr (float), parent_wall_ms (float), parent_vgpr (int),
        pmc_bottleneck (str in {compute, balanced, memory, unknown}).
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
