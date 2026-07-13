"""CPU-only tests for EVIDENCE-BASED repair diagnosis (KORE Stage 1 quality gate).

The audited failure these lock down: the repair chain-of-thought used to be 100%
TEMPLATED — it emitted one of two fixed strings and never named the actual bug.
These tests assert the diagnosis is derived from the real broken->fixed diff, names
the concrete changed token for several synthetic bug classes (inverted mask, bf16
accumulator, off-by-one, ...), stays grounded (its cited tokens are actually in the
diff), is op-appropriate (no ``tl.dot`` / multiple-of-64 talk on pointwise ops),
and degrades to a minimal factual fallback (verifier error + one changed token)
when the diff is not a single recognizable pattern.

No GPU, no teacher model, no torch/triton. Pure string analysis + a StubTeacher.
"""

from __future__ import annotations

import re

from kore.data import mutate
from kore.data.gen_repair import (
    analyze_repair_diff,
    classify_repair_diff,
    make_repair_record,
    _op_appropriate_repair_prompt,
)
from kore.data.prompts import build_turn_prompt, extract_kernel
from kore.data.teacher import StubTeacher
from kore.reward.reward import Observation


# --------------------------------------------------------------------------- #
# Synthetic known-good seeds (one per op family).
# --------------------------------------------------------------------------- #
GOOD_NORM = """import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, N, eps, scale, BLOCK_N: tl.constexpr):
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = x * rstd * scale
    tl.store(y_ptr + offs, y.to(tl.bfloat16), mask=mask)
"""

GOOD_GEMM = """import triton
import triton.language as tl

@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    offs_am = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        acc += tl.dot(a, a)

def mm(a, b):
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8
    return a
"""

GOOD_POINTWISE = """import triton
import triton.language as tl

@triton.jit
def _relu(x_ptr, y_ptr, N, BLOCK_M: tl.constexpr):
    offs = tl.arange(0, BLOCK_M)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)

def relu(x):
    BLOCK_M = 128
    return x
"""

GOOD_QUANT = """import triton
import triton.language as tl

@triton.jit
def _q(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float8_e4m3fnuz)
    tl.store(y_ptr + offs, x)
"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


# --------------------------------------------------------------------------- #
# 1. The diagnosis names the concrete changed token, per bug class.
# --------------------------------------------------------------------------- #
def test_inverted_mask_names_predicate():
    broken, _ = mutate.break_mask(GOOD_NORM)
    f = classify_repair_diff(broken, GOOD_NORM, "snr_fail", "norm")
    assert f is not None and f.change_class == "mask_predicate_flip"
    analysis, proposed = analyze_repair_diff(broken, GOOD_NORM, "snr_fail",
                                             "worst SNR 3.0 < 25.0 dB", "norm")
    # names BOTH the wrong predicate and the corrected one
    assert "offs >= N" in analysis and "offs < N" in analysis
    assert "offs < N" in proposed
    # this is a REAL diagnosis, not the old templated boilerplate
    assert "restore fp32 accumulation / correct masking / indexing" not in analysis


def test_bf16_accumulator_named():
    broken, _ = mutate.break_accumulator_dtype(GOOD_GEMM)
    f = classify_repair_diff(broken, GOOD_GEMM, "snr_fail", "gemm")
    assert f is not None and f.change_class == "accumulator_dtype"
    assert f.before == "tl.bfloat16" and f.after == "tl.float32"
    analysis, _ = analyze_repair_diff(broken, GOOD_GEMM, "snr_fail", "err", "gemm")
    assert "tl.bfloat16" in analysis and "tl.float32" in analysis


def test_off_by_one_named():
    broken, _ = mutate.break_index_offset(GOOD_GEMM)
    f = classify_repair_diff(broken, GOOD_GEMM, "snr_fail", "gemm")
    assert f is not None and f.change_class == "off_by_one_offset"
    analysis, _ = analyze_repair_diff(broken, GOOD_GEMM, "snr_fail", "err", "gemm")
    assert "+ 1" in analysis
    assert "tl.arange(0, BLOCK_K)" in analysis


def test_reduction_axis_named():
    broken, _ = mutate.break_reduction_axis(GOOD_NORM)
    f = classify_repair_diff(broken, GOOD_NORM, "snr_fail", "norm")
    assert f is not None and f.change_class == "reduction_axis"
    analysis, _ = analyze_repair_diff(broken, GOOD_NORM, "snr_fail", "err", "norm")
    assert "axis=0" in analysis and "axis=1" in analysis


def test_fp8_variant_named():
    broken, _ = mutate.break_fp8_variant(GOOD_QUANT)
    f = classify_repair_diff(broken, GOOD_QUANT, "snr_fail", "quant")
    assert f is not None and f.change_class == "fp8_variant"
    analysis, _ = analyze_repair_diff(broken, GOOD_QUANT, "snr_fail", "err", "quant")
    assert "fnuz" in analysis.lower()


def test_block_size_multiple_named():
    broken, _ = mutate.break_block_size(GOOD_GEMM)
    f = classify_repair_diff(broken, GOOD_GEMM, "compile_fail", "gemm")
    assert f is not None and f.change_class == "block_size_multiple"
    analysis, _ = analyze_repair_diff(broken, GOOD_GEMM, "compile_fail", "err", "gemm")
    assert "96" in analysis and "128" in analysis


# --------------------------------------------------------------------------- #
# 2. Grounding: every cited token is actually present in the corresponding side.
# --------------------------------------------------------------------------- #
def test_diagnosis_is_grounded_in_the_diff():
    cases = [
        (mutate.break_mask(GOOD_NORM)[0], GOOD_NORM, "norm"),
        (mutate.break_accumulator_dtype(GOOD_GEMM)[0], GOOD_GEMM, "gemm"),
        (mutate.break_index_offset(GOOD_GEMM)[0], GOOD_GEMM, "gemm"),
        (mutate.break_reduction_axis(GOOD_NORM)[0], GOOD_NORM, "norm"),
        (mutate.break_fp8_variant(GOOD_QUANT)[0], GOOD_QUANT, "quant"),
    ]
    for broken, fixed, fam in cases:
        f = classify_repair_diff(broken, fixed, "snr_fail", fam)
        assert f is not None
        # a real "before" token comes from the broken source; "after" from the fix
        if not f.before.startswith("("):
            assert _norm(f.before) in _norm(broken)
        if not f.after.startswith("("):
            assert _norm(f.after) in _norm(fixed)


# --------------------------------------------------------------------------- #
# 3. Op-appropriateness: no MFMA / tl.dot / multiple-of-64 on pointwise ops.
# --------------------------------------------------------------------------- #
def test_pointwise_diagnosis_has_no_mfma_boilerplate():
    for breaker in (mutate.break_mask, mutate.break_block_size, mutate.break_dtype_cast):
        broken, _ = breaker(GOOD_POINTWISE)
        if broken == GOOD_POINTWISE:
            continue
        analysis, proposed = analyze_repair_diff(
            broken, GOOD_POINTWISE, "snr_fail", "err", "activation")
        text = (analysis + " " + proposed)
        assert "tl.dot" not in text
        assert "MFMA" not in text
        assert "multiple of 64" not in text


def test_gemm_blocksize_diagnosis_may_mention_mfma():
    broken, _ = mutate.break_block_size(GOOD_GEMM)
    analysis, _ = analyze_repair_diff(broken, GOOD_GEMM, "compile_fail", "err", "gemm")
    assert "MFMA" in analysis  # GEMM tile alignment legitimately mentions MFMA


# --------------------------------------------------------------------------- #
# 4. Distinct bugs produce DISTINCT diagnoses (not one templated string).
# --------------------------------------------------------------------------- #
def test_distinct_bugs_give_distinct_diagnoses():
    a1, _ = analyze_repair_diff(mutate.break_mask(GOOD_NORM)[0], GOOD_NORM,
                                "snr_fail", "err", "norm")
    a2, _ = analyze_repair_diff(mutate.break_accumulator_dtype(GOOD_GEMM)[0], GOOD_GEMM,
                                "snr_fail", "err", "gemm")
    a3, _ = analyze_repair_diff(mutate.break_index_offset(GOOD_GEMM)[0], GOOD_GEMM,
                                "snr_fail", "err", "gemm")
    assert a1 != a2 != a3 and a1 != a3


# --------------------------------------------------------------------------- #
# 5. Fallback: ambiguous rewrite -> verifier error + one concrete changed token.
# --------------------------------------------------------------------------- #
def test_fallback_names_error_and_one_token():
    broken = "def broken():\n    return 0"
    fixed = "def k():\n    return 1"
    assert classify_repair_diff(broken, fixed, "snr_fail", "generic") is None
    analysis, proposed = analyze_repair_diff(
        broken, fixed, "snr_fail", "worst SNR 5.0 < 25.0 dB", "generic")
    assert "worst SNR 5.0 < 25.0 dB" in analysis  # the verifier error is retained
    # a concrete changed token is cited (the one-token fallback)
    assert "`0`" in analysis or "`1`" in analysis or "broken" in analysis or "k" in proposed
    assert "tl.dot" not in analysis and "MFMA" not in analysis


def test_classify_none_for_identical_sources():
    assert classify_repair_diff(GOOD_NORM, GOOD_NORM, "snr_fail", "norm") is None


# --------------------------------------------------------------------------- #
# 6. Prompt sanitization: MFMA constraints stripped for non-GEMM families.
# --------------------------------------------------------------------------- #
def test_op_appropriate_prompt_strips_tldot_for_pointwise():
    p = build_turn_prompt("src", feedback="e", mode="repair")
    assert "use tl.dot" in p  # baseline prompt injects it for every op
    assert "use tl.dot" not in _op_appropriate_repair_prompt(p, "activation")
    assert "use tl.dot" not in _op_appropriate_repair_prompt(p, "quant")
    assert "use tl.dot" not in _op_appropriate_repair_prompt(p, "norm")
    # accumulate-in-fp32 (universally valid) is preserved
    assert "fp32" in _op_appropriate_repair_prompt(p, "activation")
    # GEMM / attention keep the MFMA discipline (it applies there)
    assert "use tl.dot" in _op_appropriate_repair_prompt(p, "gemm")
    assert "use tl.dot" in _op_appropriate_repair_prompt(p, "attention")


# --------------------------------------------------------------------------- #
# 7. End-to-end: make_repair_record stores a real, grounded, op-appropriate turn.
# --------------------------------------------------------------------------- #
class _Task:
    def __init__(self, op, dtype="fp16"):
        self.task_id = f"gen_{op}_{dtype}"
        self.operation = op
        self.dtype = dtype
        self.gpu_target = "gfx942"


class _FixEnv:
    """The fix always validates (the repair path only steps the fix)."""

    def step(self, source, full_validation=True, multi_shape=True):
        return Observation(compiled=True, validation_passed=True, snr_db=80.0,
                           snr_by_shape={"s": 80.0}, wall_ms=1.0, baseline_ms=2.0,
                           dtype="fp16")


def _snr_fail_obs():
    return Observation(compiled=True, validation_passed=False, snr_db=2.0,
                       snr_by_shape={"s": 2.0},
                       error_text="worst SNR 2.0 < 25.0 dB", dtype="fp16")


def _teacher_returning(src):
    return StubTeacher(fn=lambda m: "FULL_KERNEL:\n```python\n" + src + "\n```")


def test_make_repair_record_names_bug_end_to_end():
    broken, _ = mutate.break_mask(GOOD_NORM)
    rec = make_repair_record(_Task("rmsnorm"), _teacher_returning(GOOD_NORM),
                             _FixEnv(), broken_src=broken, broken_obs=_snr_fail_obs())
    assert rec is not None and rec.failure_class == "snr_fail"
    asst = rec.messages[-1]["content"]
    assert asst.startswith("ANALYSIS:")
    assert "PROPOSED_CHANGE:" in asst and "FULL_KERNEL:" in asst
    # the stored ANALYSIS names the concrete bug (not a templated string)
    assert "offs < N" in asst and "offs >= N" in asst
    # the verified fix is recoverable and is the corrected kernel
    assert extract_kernel(asst).strip() == GOOD_NORM.strip()
    # the verifier error is folded into the ANALYSIS grounding
    assert "worst SNR 2.0" in asst


def test_make_repair_record_pointwise_prompt_is_op_appropriate():
    broken, _ = mutate.break_mask(GOOD_POINTWISE)
    rec = make_repair_record(_Task("relu"), _teacher_returning(GOOD_POINTWISE),
                             _FixEnv(), broken_src=broken, broken_obs=_snr_fail_obs())
    assert rec is not None
    user_turn = rec.messages[1]["content"]
    assert "use tl.dot" not in user_turn  # sanitized for a pointwise op
    asst = rec.messages[-1]["content"]
    assert "tl.dot" not in asst and "MFMA" not in asst


def test_make_repair_record_diagnoses_off_by_one_end_to_end():
    broken, _ = mutate.break_index_offset(GOOD_GEMM)
    rec = make_repair_record(_Task("gemm", "bf16"), _teacher_returning(GOOD_GEMM),
                             _FixEnv(), broken_src=broken, broken_obs=_snr_fail_obs())
    assert rec is not None
    asst = rec.messages[-1]["content"]
    assert "+ 1" in asst and "off-by-one" in asst.lower()
