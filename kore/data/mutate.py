"""Pure source-mutation functions that BREAK a correct Triton kernel.

Used to manufacture abundant cold-start *repair* data (KORE Stage 1): take the
known-good seed kernel, apply one plausible-looking but wrong mutation, run it
through the verifier to confirm it fails, then ask the teacher to repair it.

All functions are pure string transforms (no GPU, no imports of torch/triton) so
they are trivially unit-testable. Each returns a NEW source string; if a pattern
can't be found a guaranteed fallback still produces a distinct, broken variant.

``failure_class_hint`` values match the reward module's buckets:
  - "compile_fail": the mutation should not build (e.g. non-power-of-2 tile).
  - "snr_fail": the mutation builds but is numerically wrong (low SNR).
"""

from __future__ import annotations

import random
import re

FailureHint = str  # "compile_fail" | "snr_fail"


def _first_sub(src: str, pattern: str, repl: str, flags: int = 0) -> tuple[str, bool]:
    """Substitute only the first match; report whether anything changed."""
    new, n = re.subn(pattern, repl, src, count=1, flags=flags)
    return new, (n > 0 and new != src)


def break_block_size(src: str) -> tuple[str, FailureHint]:
    """Make a BLOCK size a non-power-of-2 / non-64-multiple value.

    Triton's ``tl.arange`` requires a power-of-2 length, and MFMA tiles want
    multiples of 64, so a value like 96 typically fails to compile on gfx942.
    """
    # tuple assignment, e.g. `BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8`
    out, ok = _first_sub(
        src,
        r"(BLOCK_M\s*,[^\n=]*=\s*)(\d+)",
        lambda m: m.group(1) + "96",
    )
    if ok:
        return out, "compile_fail"
    # single assignment / kwarg, e.g. `BLOCK_M = 128`
    out, ok = _first_sub(src, r"(BLOCK_M\s*=\s*)(\d+)", lambda m: m.group(1) + "96")
    if ok:
        return out, "compile_fail"
    # generic fallback: shrink the first standalone 128 to 96
    out, ok = _first_sub(src, r"\b128\b", "96")
    if ok:
        return out, "compile_fail"
    # last resort: append a broken constexpr tile
    return src + "\n# BROKEN_TILE\nBLOCK_M = 96\n", "compile_fail"


def break_accumulator_dtype(src: str) -> tuple[str, FailureHint]:
    """Downcast the fp32 accumulator to bf16 (a low-precision dtype).

    Accumulating a long K-reduction in bf16/fp16 destroys precision, so the
    kernel still compiles but fails the SNR correctness gate (DATASET_SPEC N1).
    """
    # `tl.zeros((...), dtype=tl.float32)` -> bf16 accumulator
    out, ok = _first_sub(
        src,
        r"(tl\.zeros\([^)]*dtype\s*=\s*tl\.)float32",
        lambda m: m.group(1) + "bfloat16",
    )
    if ok:
        return out, "snr_fail"
    # any explicit fp32 accumulator dtype
    out, ok = _first_sub(src, r"tl\.float32", "tl.bfloat16")
    if ok:
        return out, "snr_fail"
    out, ok = _first_sub(src, r"dtype\s*=\s*tl\.float32", "dtype=tl.bfloat16")
    if ok:
        return out, "snr_fail"
    return src + "\n# BROKEN_ACC: accumulate in low precision\n", "snr_fail"


def break_index_offset(src: str) -> tuple[str, FailureHint]:
    """Introduce an off-by-one in a load index so the math is wrong.

    Shifting the K-offset by 1 reads the wrong elements: the kernel compiles and
    runs but produces an incorrect result (SNR failure)."""
    # shift the K contraction index: `offs_k = tl.arange(0, BLOCK_K)`
    out, ok = _first_sub(
        src,
        r"(offs_k\s*=\s*tl\.arange\(0,\s*BLOCK_K\))",
        lambda m: m.group(1) + " + 1",
    )
    if ok:
        return out, "snr_fail"
    # generic: offset the row index by 1
    out, ok = _first_sub(
        src,
        r"(offs_am\s*=\s*\([^\n]*tl\.arange\(0,\s*BLOCK_M\))",
        lambda m: m.group(1) + " + 1",
    )
    if ok:
        return out, "snr_fail"
    # generic fallback: perturb the first arange result by +1
    out, ok = _first_sub(
        src, r"(tl\.arange\(0,\s*BLOCK_\w+\))", lambda m: m.group(1) + " + 1"
    )
    if ok:
        return out, "snr_fail"
    return src + "\n# BROKEN_INDEX: off-by-one in load offset\n", "snr_fail"


# --------------------------------------------------------------------------- #
# OP-family-aware mutators (norm / softmax / activation / attention / moe ...)
#
# Unlike the GEMM-specific breakers above, these do NOT append a fallback
# comment when their pattern is absent -- they return the source *unchanged* so
# ``apply_random_breakage`` can move on to the next candidate. This is what lets
# a family's mutator list degrade gracefully to the generic ones.
# --------------------------------------------------------------------------- #
def break_reduction_axis(src: str) -> tuple[str, FailureHint]:
    """Reduce over the wrong axis or drop the ``/ N`` normalization.

    Flipping ``tl.sum(..., axis=0)`` to ``axis=1`` (or dropping the mean divide
    in a norm/softmax) still compiles but yields a numerically wrong result.
    """
    # flip an explicit reduction axis on tl.sum / tl.max
    out, ok = _first_sub(
        src, r"(tl\.(?:sum|max)\([^)]*axis\s*=\s*)0\b", lambda m: m.group(1) + "1"
    )
    if ok:
        return out, "snr_fail"
    out, ok = _first_sub(
        src, r"(tl\.(?:sum|max)\([^)]*axis\s*=\s*)1\b", lambda m: m.group(1) + "0"
    )
    if ok:
        return out, "snr_fail"
    # drop the mean/normalization divide right after a reduction: `... ) / N`
    out, ok = _first_sub(
        src, r"(tl\.sum\([^)]*\))\s*/\s*\w+", lambda m: m.group(1)
    )
    if ok:
        return out, "snr_fail"
    # softmax-style: drop division by the running denominator
    out, ok = _first_sub(
        src,
        r"(=\s*[\w\.\[\]]+)\s*/\s*(?:denom|denominator|Z|l_i|_sum|sum_exp)\b",
        lambda m: m.group(1),
    )
    if ok:
        return out, "snr_fail"
    return src, "snr_fail"


def break_mask(src: str) -> tuple[str, FailureHint]:
    """Invert or drop a bounds ``mask=`` in tl.load / tl.store.

    Inverting the comparison (``<`` -> ``>=``) makes the mask select the wrong
    (or no) elements, giving wrong / out-of-bounds reads: an SNR failure.
    """
    # invert the first comparison operator inside a mask expression
    for pat, repl in (
        (r"(\bmask\w*\s*=\s*[^\n]*?)<=", r"\1>"),
        (r"(\bmask\w*\s*=\s*[^\n]*?)>=", r"\1<"),
        (r"(\bmask\w*\s*=\s*[^\n]*?)<(?!=)", r"\1>="),
        (r"(\bmask\w*\s*=\s*[^\n]*?)>(?!=)", r"\1<="),
    ):
        out, ok = _first_sub(src, pat, repl)
        if ok:
            return out, "snr_fail"
    # otherwise remove a simple mask kwarg entirely: `, mask=<ident>`
    out, ok = _first_sub(src, r",\s*mask\s*=\s*\w+", "")
    if ok:
        return out, "snr_fail"
    return src, "snr_fail"


def break_eps(src: str) -> tuple[str, FailureHint]:
    """Drop the ``+ eps`` guard in an rsqrt / normalization.

    Without the epsilon a zero variance divides by zero -> inf/NaN -> SNR fail.
    """
    for pat in (r"\s*\+\s*eps\b", r"\s*\+\s*epsilon\b"):
        out, ok = _first_sub(src, pat, "")
        if ok:
            return out, "snr_fail"
    # drop a small additive epsilon literal inside a sqrt/rsqrt
    out, ok = _first_sub(
        src, r"(sqrt\([^)]*?)\s*\+\s*1e-?\d+", lambda m: m.group(1)
    )
    if ok:
        return out, "snr_fail"
    return src, "snr_fail"


def break_dtype_cast(src: str) -> tuple[str, FailureHint]:
    """Remove an fp32 upcast/accumulate cast or corrupt the output cast.

    Dropping ``.to(tl.float32)`` accumulates in low precision (SNR fail); flipping
    the output cast dtype can mismatch the buffer and fail to compile.
    """
    # remove an fp32 upcast (accumulate in the input's low precision)
    out, ok = _first_sub(src, r"\.to\(tl\.float32\)", "")
    if ok:
        return out, "snr_fail"
    # corrupt the output cast dtype (bf16 <-> fp16 mismatch with the buffer)
    out, ok = _first_sub(
        src, r"(\.to\(tl\.)bfloat16(\))", lambda m: m.group(1) + "float16" + m.group(2)
    )
    if ok:
        return out, "compile_fail"
    out, ok = _first_sub(
        src, r"(\.to\(tl\.)float16(\))", lambda m: m.group(1) + "bfloat16" + m.group(2)
    )
    if ok:
        return out, "snr_fail"
    return src, "snr_fail"


def break_scale(src: str) -> tuple[str, FailureHint]:
    """Drop (or duplicate) a scale multiply used in attention/quant kernels.

    Removing ``* sm_scale`` / ``* scale`` changes the magnitude of the result,
    a classic softmax/quant scaling bug that survives compilation."""
    for pat in (r"\s*\*\s*sm_scale\b", r"\s*\*\s*qk_scale\b", r"\s*\*\s*scale\b"):
        out, ok = _first_sub(src, pat, "")
        if ok:
            return out, "snr_fail"
    # duplicate a bare scale multiply if we couldn't cleanly drop one
    out, ok = _first_sub(
        src, r"(\*\s*(?:sm_scale|qk_scale|scale)\b)", lambda m: m.group(1) + " " + m.group(1)
    )
    if ok:
        return out, "snr_fail"
    return src, "snr_fail"


def break_missing_mask(src: str) -> tuple[str, FailureHint]:
    """Drop the tail bounds mask so the last (partial) tile is mishandled.

    DATASET_SPEC S5: when ``dim % BLOCK != 0`` the final tile needs a boundary
    mask. Widening the guard (``offs < N`` -> ``offs < N + BLOCK``) or removing
    the ``mask=`` kwarg makes the tail rows/cols read/write past the valid
    region, corrupting the result (SNR fail) rather than a clean compile error.
    """
    # widen the tail guard so out-of-range lanes are (wrongly) included
    out, ok = _first_sub(
        src,
        r"(\bmask\w*\s*=\s*offs\w*\s*<\s*)(N|M|K|n_cols|n_rows)\b",
        lambda m: m.group(1) + "(" + m.group(2) + " + BLOCK_N)",
    )
    if ok:
        return out, "snr_fail"
    # otherwise drop a mask kwarg entirely (unmasked tail load/store)
    out, ok = _first_sub(src, r",\s*mask\s*=\s*\w+", "")
    if ok:
        return out, "snr_fail"
    return src, "snr_fail"


def break_fp8_variant(src: str) -> tuple[str, FailureHint]:
    """Swap the AMD FNUZ fp8 encoding for the OCP ``fn`` variant (N8 / L-trap).

    gfx942 uses ``float8_e4m3fnuz`` (FNUZ). Emitting the OCP ``e4m3fn`` layout
    silently mismatches what AITER/hipBLASLt expect (``aiter_ref.py``), so the
    kernel compiles/runs but is numerically wrong vs the production baseline.
    """
    for pat, repl in (
        (r"float8_e4m3fnuz", "float8_e4m3fn"),
        (r"float8_e5m2fnuz", "float8_e5m2"),
        (r"\btl\.float8e4b8\b", "tl.float8e4nv"),   # AMD FNUZ -> NVIDIA/OCP
        (r"\btl\.float8e5b16\b", "tl.float8e5"),
        (r"e4m3fnuz", "e4m3fn"),
        (r"e5m2fnuz", "e5m2"),
        (r"fnuz", "fn"),
    ):
        out, ok = _first_sub(src, pat, repl)
        if ok:
            return out, "snr_fail"
    return src, "snr_fail"


def break_k_multiple_of_32(src: str) -> tuple[str, FailureHint]:
    """Make the K tile not a multiple of 32 (illegal for fp8/MX scale groups).

    DATASET_SPEC S6: fp8/MX MFMA requires ``BLOCK_K % 32 == 0`` (scale group).
    48 is neither a multiple of 32 nor a power of two, so ``tl.arange(0, 48)``
    fails to build -> compile_fail.
    """
    # tuple form: `BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 128, 64, 8`
    out, ok = _first_sub(
        src,
        r"(BLOCK_M\s*,\s*BLOCK_N\s*,\s*BLOCK_K[^\n=]*=\s*\d+\s*,\s*\d+\s*,\s*)\d+",
        lambda m: m.group(1) + "48",
    )
    if ok:
        return out, "compile_fail"
    for pat in (r"(BLOCK_K\s*:\s*tl\.constexpr\s*=\s*)\d+", r"(BLOCK_K\s*=\s*)\d+"):
        out, ok = _first_sub(src, pat, lambda m: m.group(1) + "48")
        if ok:
            return out, "compile_fail"
    return src, "compile_fail"


def break_transpose_operand(src: str) -> tuple[str, FailureHint]:
    """Transpose/permute an operand by swapping its two stride multipliers (L1).

    Swapping ``stride_am``<->``stride_ak`` (or B's) in a pointer expression
    reads the operand transposed: it still compiles and runs but the contraction
    is wrong (SNR fail). Falls back to swapping the broadcast axes.
    """
    for a, b in (
        ("stride_am", "stride_ak"),
        ("stride_bn", "stride_bk"),
        ("stride_xm", "stride_xn"),
    ):
        pat = re.escape(a) + r"(.*?)" + re.escape(b)
        out, ok = _first_sub(src, pat, lambda m, a=a, b=b: b + m.group(1) + a)
        if ok:
            return out, "snr_fail"
    # generic: swap the broadcast axes on the first 2D pointer index
    out, ok = _first_sub(src, r"\[:,\s*None\]", "[None, :]")
    if ok:
        return out, "snr_fail"
    return src, "snr_fail"


def break_missing_barrier(src: str) -> tuple[str, FailureHint]:
    """Remove a synchronization barrier -> cross-wavefront race (C1).

    Dropping a ``tl.debug_barrier()`` between a shared-memory write and the
    dependent read lets wavefronts read stale/partial data: a nondeterministic
    correctness failure (SNR fail, may only show up at scale).
    """
    for pat in (
        r"\n[ \t]*tl\.debug_barrier\(\)[ \t]*",
        r"\n[ \t]*tl\.barrier\(\)[ \t]*",
        r"\n[ \t]*__syncthreads\(\)[ \t]*",
    ):
        out, ok = _first_sub(src, pat, "")
        if ok:
            return out, "snr_fail"
    return src, "snr_fail"


def break_block_m_to_64(src: str) -> tuple[str, FailureHint]:
    """Shrink BLOCK_M from 128 to 64 (sparse-attn cross-WG corruption, C1).

    Skill ``sparse_block_m_128_guard``: BLOCK_M=64 causes *silent* cross-
    workgroup corruption in block-sparse/split-K kernels; 128 is required.
    """
    out, ok = _first_sub(src, r"(BLOCK_M\s*=\s*)128\b", lambda m: m.group(1) + "64")
    if ok:
        return out, "snr_fail"
    out, ok = _first_sub(
        src, r"(BLOCK_M\s*,[^\n=]*=\s*)128\b", lambda m: m.group(1) + "64"
    )
    if ok:
        return out, "snr_fail"
    return src, "snr_fail"


# Map each op family to the mutators that plausibly break its kernels. "generic"
# holds mutators that apply to essentially any Triton kernel.
OP_FAMILY_MUTATORS: dict[str, list] = {
    "gemm": [
        break_block_size,
        break_accumulator_dtype,
        break_index_offset,
        break_dtype_cast,
        break_transpose_operand,
        break_missing_mask,
        break_mask,
        break_fp8_variant,
        break_k_multiple_of_32,
        break_block_m_to_64,
        break_scale,
    ],
    "norm": [
        break_reduction_axis,
        break_eps,
        break_dtype_cast,
        break_missing_mask,
        break_mask,
        break_block_size,
        break_index_offset,
    ],
    "activation": [
        break_missing_mask,
        break_mask,
        break_dtype_cast,
        break_scale,
        break_index_offset,
        break_block_size,
    ],
    "attention": [
        break_scale,
        break_reduction_axis,
        break_missing_mask,
        break_mask,
        break_dtype_cast,
        break_index_offset,
        break_missing_barrier,
        break_block_m_to_64,
    ],
    "moe": [
        break_scale,
        break_missing_mask,
        break_mask,
        break_dtype_cast,
        break_index_offset,
        break_block_size,
        break_block_m_to_64,
        break_transpose_operand,
    ],
    "quant": [
        break_fp8_variant,
        break_k_multiple_of_32,
        break_accumulator_dtype,
        break_dtype_cast,
        break_scale,
        break_missing_mask,
    ],
    "generic": [
        break_missing_mask,
        break_mask,
        break_dtype_cast,
        break_index_offset,
    ],
}

# Retained for backward compatibility with any callers importing the old tuple.
_BREAKERS = (break_block_size, break_accumulator_dtype, break_index_offset)


def infer_family(operation_or_task_id: str) -> str:
    """Map an operation name or task id (e.g. "rmsnorm_bf16") to an op family."""
    s = (operation_or_task_id or "").lower()
    families = (
        ("gemm", ("gemm", "matmul")),
        ("norm", ("rmsnorm", "layernorm", "rms_norm", "layer_norm", "norm")),
        ("attention", ("attention", "attn", "mha", "mla", "mqa", "flash", "sdpa")),
        ("moe", ("moe", "expert")),
        ("activation", ("silu", "gelu", "relu", "swiglu", "geglu", "glu", "act")),
        ("quant", ("quant", "dequant", "w8a8", "scaled_mm", "fp8_scale")),
    )
    for family, keys in families:
        if any(k in s for k in keys):
            return family
    return "generic"


def apply_random_breakage(
    src: str,
    family_or_rng: "str | random.Random | None" = None,
    rng_or_family: "str | random.Random | None" = None,
    *,
    family: str | None = None,
    rng: random.Random | None = None,
) -> tuple[str, FailureHint, str]:
    """Apply one randomly-chosen, family-appropriate breakage.

    Signature is deliberately flexible so both the documented order
    ``apply_random_breakage(src, family, rng)`` and the legacy order
    ``apply_random_breakage(src, rng, family=...)`` (used by ``gen_repair`` and
    the existing test suite) work: the two positional slots are resolved by
    type (a ``random.Random`` is the rng, a ``str`` is the family).

    Picks from ``family``'s mutator list (always keeping the generic mutators as
    a fallback), trying candidates in random order until one *actually* changes
    the source. Guaranteed to return a changed source.

    Returns ``(broken_src, failure_class, mutator_name)`` — the first two are the
    ``(broken_src, failure_class)`` pair from the spec; the third names the
    mutator used, kept for provenance/back-compat with ``gen_repair``.
    """
    for a in (family_or_rng, rng_or_family):
        if isinstance(a, random.Random):
            if rng is None:
                rng = a
        elif isinstance(a, str):
            if family is None:
                family = a
    rng = rng or random.Random()
    family = family or "generic"

    mutators = list(OP_FAMILY_MUTATORS.get(family) or OP_FAMILY_MUTATORS["generic"])
    for fn in OP_FAMILY_MUTATORS["generic"]:
        if fn not in mutators:
            mutators.append(fn)
    rng.shuffle(mutators)
    for fn in mutators:
        broken, hint = fn(src)
        if broken != src:
            return broken, hint, fn.__name__
    # last resort: the GEMM block-size breaker always mutates (has an append
    # fallback), so this still guarantees a changed source.
    broken, hint = break_block_size(src)
    if broken != src:
        return broken, hint, "break_block_size"
    return src + "\n# BROKEN\n", "compile_fail", "fallback"
