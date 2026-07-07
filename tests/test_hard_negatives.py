"""CPU-only tests for labeled reward-hack negatives + op-family mutate dispatch.

No GPU / teacher / torch execution: every assertion is either a static-scan
result or a by-construction property of the generated source.

For EACH of the nine DATASET_SPEC §2.6 negatives we assert it is caught by the
appropriate detection layer:
  - Layer A: ``kore.reward.reward.scan_for_hacks`` (owned by reward.py)
  - Layer B: ``hard_negatives.static_hack_scan`` (provided here; the patterns the
    parent must fold into ``scan_for_hacks`` -- see ``SCAN_FOR_HACKS_GAPS``)
  - Layer C: execution-time correctness/SNR gate (asserted by construction here)
"""

from __future__ import annotations

import random

import pytest

from kore.data import mutate
from kore.data import hard_negatives as hn
from kore.data.schemas import RankedGroupRecord
from kore.data.build_datasets import build_dpo
from kore.reward.reward import scan_for_hacks


# --------------------------------------------------------------------------- #
# representative correct sources
# --------------------------------------------------------------------------- #
GEMM_SRC = """
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                 stride_am, stride_ak, stride_bn, stride_bk,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_am = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask = offs_am < M
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    acc += tl.dot(a.to(tl.float32), a.to(tl.float32))
    tl.store(c_ptr + offs_am, acc.to(tl.bfloat16), mask=mask)


def gemm(a, b):
    import torch
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 128),)
    _gemm_kernel[grid](a, b, c, M, N, K,
                       a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                       BLOCK_M=128, BLOCK_N=128, BLOCK_K=64)
    return c
"""

NORM_SRC = """
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(y_ptr + offs, y.to(tl.bfloat16), mask=mask)


def rmsnorm(x, weight, eps=1e-6):
    return x
"""

FP8_SRC = """
import torch
import triton
import triton.language as tl


def quant(x):
    xq = x.to(torch.float8_e4m3fnuz)
    return xq
"""

BARRIER_SRC = """
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr):
    v = tl.load(x_ptr)
    tl.debug_barrier()
    tl.store(x_ptr, v)
"""

BLOCKM_SRC = """
def run(a, b):
    BLOCK_M = 128
    return a @ b
"""


# --------------------------------------------------------------------------- #
# hard negatives: catalogue
# --------------------------------------------------------------------------- #
def test_nine_kinds_unique():
    assert len(hn.HARD_NEGATIVE_KINDS) == 9
    assert len(set(hn.HARD_NEGATIVE_KINDS)) == 9


def test_all_hard_negatives_shape_and_labels():
    negs = hn.all_hard_negatives(GEMM_SRC, task=None)
    assert len(negs) == 9
    seen_kinds = set()
    for (src, label), kind in zip(negs, hn.HARD_NEGATIVE_KINDS):
        assert label == f"reward_hack:{kind}", (label, kind)
        assert src and src != GEMM_SRC          # a real, distinct variant
        seen_kinds.add(kind)
    assert seen_kinds == set(hn.HARD_NEGATIVE_KINDS)


def test_make_hard_negative_unknown_kind():
    with pytest.raises(ValueError):
        hn.make_hard_negative("not_a_hack", GEMM_SRC)


# --------------------------------------------------------------------------- #
# hard negatives: EACH of the nine is caught by its detection layer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", hn.HARD_NEGATIVE_KINDS)
def test_each_negative_caught_by_its_layer(kind):
    """Every negative must be rejected by scan_for_hacks (A), static_hack_scan
    (B), or (for wrong_dtype_accum) fail correctness by construction (C)."""
    src, label = hn.make_hard_negative(kind, GEMM_SRC, task=None)
    layer = hn.DETECTION_LAYER[kind]
    a = scan_for_hacks(src)
    b = hn.static_hack_scan(src)

    if layer == "A":
        assert a is not None, f"{kind}: expected scan_for_hacks (Layer A) to catch it"
    elif layer == "B":
        assert b is not None, f"{kind}: expected static_hack_scan (Layer B) to catch it"
    elif layer == "C":
        # execution-time only: neither static layer flags it, but it is wrong by
        # construction (fp32 accumulator downgraded to bf16 -> fails the SNR gate).
        assert a is None and b is None
        assert src.count("tl.float32") < GEMM_SRC.count("tl.float32") or "bfloat16" in src
    else:  # pragma: no cover - guard against a typo in the map
        raise AssertionError(f"unknown detection layer {layer!r}")

    # Regardless of layer, SOME gate must reject each hack pre- or at-execution.
    assert (a is not None) or (b is not None) or (layer == "C")


def test_layer_A_kinds_are_the_scan_for_hacks_ones():
    """Document exactly which hacks reward.scan_for_hacks catches today."""
    caught = {k for k in hn.HARD_NEGATIVE_KINDS
              if scan_for_hacks(hn.make_hard_negative(k, GEMM_SRC)[0]) is not None}
    # copy_reference is now folded into Layer A (parent closed the gap): a kernel
    # that returns the oracle output passes the SNR gate, so it must be rejected
    # statically. It is still additionally covered by Layer B static_hack_scan.
    assert caught == {"vendor_call", "torch_delegation", "try_except_fallback", "copy_reference"}


def test_static_hack_scan_catches_layer_B_kinds():
    layer_b = [k for k in hn.HARD_NEGATIVE_KINDS if hn.DETECTION_LAYER[k] == "B"]
    assert set(layer_b) == {
        "copy_reference", "reference_recycling", "partial_compute",
        "hardcoded_shape", "timing_hack",
    }
    for kind in layer_b:
        src, _ = hn.make_hard_negative(kind, GEMM_SRC)
        assert hn.static_hack_scan(src) is not None, kind


def test_static_hack_scan_clean_on_correct_source():
    # the honest correct kernels must NOT be flagged
    for src in (GEMM_SRC, NORM_SRC):
        assert hn.static_hack_scan(src) is None
        assert scan_for_hacks(src) is None


# --------------------------------------------------------------------------- #
# hard negatives: documented scan_for_hacks GAPS (xfail until parent closes)
# --------------------------------------------------------------------------- #
def test_scan_for_hacks_gaps_are_documented():
    """Every hack NOT caught by Layer A must have a documented gap entry so the
    parent knows exactly what to extend scan_for_hacks to catch."""
    not_layer_a = {k for k in hn.HARD_NEGATIVE_KINDS if hn.DETECTION_LAYER[k] != "A"}
    assert not_layer_a <= set(hn.SCAN_FOR_HACKS_GAPS)


def test_copy_reference_should_be_caught_by_scan_for_hacks():
    # GAP CLOSED by parent: reward._HACK_PATTERNS now flags reference/oracle calls.
    src, _ = hn.make_hard_negative("copy_reference", GEMM_SRC)
    assert scan_for_hacks(src) is not None


@pytest.mark.xfail(
    strict=False,
    reason="GAP: scan_for_hacks does not flag output-recycling (returns a "
    "pre-allocated buffer with no tl.store / no kernel launch). Primary gate is "
    "runtime (candidate-runs-first + NaN-poisoned out); a static smell is a bonus.",
)
def test_reference_recycling_should_be_caught_by_scan_for_hacks():
    src, _ = hn.make_hard_negative("reference_recycling", GEMM_SRC)
    assert scan_for_hacks(src) is not None


# --------------------------------------------------------------------------- #
# hard negatives: DPO pair construction (>= 8% of DPO data target)
# --------------------------------------------------------------------------- #
def test_build_hard_negative_pairs():
    pairs = hn.build_hard_negative_pairs(GEMM_SRC, task=None)
    assert len(pairs) == 9
    kinds = {p["kind"] for p in pairs}
    assert kinds == set(hn.HARD_NEGATIVE_KINDS)
    for p in pairs:
        assert p["chosen"] == GEMM_SRC
        assert p["rejected"] != p["chosen"]
        assert p["label"].startswith("reward_hack:")


def test_build_hard_negative_group_is_dpo_ready():
    grp = hn.build_hard_negative_group(GEMM_SRC, task=None)
    assert isinstance(grp, RankedGroupRecord)
    assert len(grp.candidates) == 10                 # correct + 9 hacks
    assert grp.candidates[0]["source"] == GEMM_SRC
    assert grp.preferences == [[0, i] for i in range(1, 10)]

    rows = build_dpo([grp])
    assert len(rows) == 9
    for row in rows:
        assert set(row) == {"prompt", "chosen", "rejected"}
        assert row["chosen"] != row["rejected"]
        # conversational trl.DPOTrainer shape: chosen/rejected are message lists
        assert isinstance(row["chosen"], list)
        assert "FULL_KERNEL" in row["chosen"][0]["content"]


def test_meets_hard_negative_target():
    assert hn.HARD_NEGATIVE_DPO_TARGET == 0.08
    assert hn.meets_hard_negative_target(8, 100) is True
    assert hn.meets_hard_negative_target(7, 100) is False
    assert hn.meets_hard_negative_target(0, 0) is False


def test_hard_negatives_work_for_all_real_task_families():
    from kore.tasks.registry import all_tasks

    tasks = all_tasks()
    assert tasks, "no tasks discovered"
    checked = 0
    for task in tasks:
        try:
            correct = task.seed_source
        except (FileNotFoundError, OSError):
            continue  # tolerate half-created tasks (seed not written yet)
        for src, label in hn.all_hard_negatives(correct, task):
            assert src and src != correct
            assert label.startswith("reward_hack:")
        checked += 1
    assert checked, "no task with a readable seed was exercised"


# --------------------------------------------------------------------------- #
# mutate: op-family dispatch changes source for EACH family
# --------------------------------------------------------------------------- #
RICH_SRC = GEMM_SRC + NORM_SRC + "\n" + FP8_SRC + "\n" + BARRIER_SRC


def test_mutate_dispatch_changes_source_for_each_family():
    for family in mutate.OP_FAMILY_MUTATORS:
        rng = random.Random(hash(family) & 0xFFFF)
        out, failure_class, name = mutate.apply_random_breakage(
            RICH_SRC, family=family, rng=rng
        )
        assert out != RICH_SRC, family
        assert failure_class in ("compile_fail", "snr_fail"), family
        assert name, family


def test_apply_random_breakage_new_positional_order():
    # documented signature: apply_random_breakage(src, family, rng)
    rng = random.Random(0)
    out, fc, name = mutate.apply_random_breakage(NORM_SRC, "norm", rng)
    assert out != NORM_SRC and fc in ("compile_fail", "snr_fail") and name


def test_apply_random_breakage_legacy_positional_order():
    # legacy call used by gen_repair + existing tests: (src, rng, family=...)
    rng = random.Random(0)
    out, fc, name = mutate.apply_random_breakage(NORM_SRC, rng, family="norm")
    assert out != NORM_SRC and fc in ("compile_fail", "snr_fail") and name


def test_op_family_mutators_include_quant_and_new_mutators():
    assert "quant" in mutate.OP_FAMILY_MUTATORS
    families = {"gemm", "norm", "activation", "attention", "moe", "quant", "generic"}
    assert families <= set(mutate.OP_FAMILY_MUTATORS)
    # the new mutators are wired somewhere
    all_muts = {fn.__name__ for muts in mutate.OP_FAMILY_MUTATORS.values() for fn in muts}
    for new in (
        "break_missing_mask", "break_fp8_variant", "break_k_multiple_of_32",
        "break_transpose_operand", "break_missing_barrier", "break_block_m_to_64",
    ):
        assert new in all_muts, new


def test_infer_family_quant_and_gemm_fp8():
    assert mutate.infer_family("fp8_quant") == "quant"
    assert mutate.infer_family("dequant_fp8") == "quant"
    # a fp8 GEMM is still the gemm family (matmul dominates)
    assert mutate.infer_family("gemm_fp8_a8w8") == "gemm"


# --------------------------------------------------------------------------- #
# mutate: the new individual mutators
# --------------------------------------------------------------------------- #
def test_break_fp8_variant_swaps_fnuz():
    out, fc = mutate.break_fp8_variant(FP8_SRC)
    assert out != FP8_SRC
    assert "float8_e4m3fn" in out and "float8_e4m3fnuz" not in out
    assert fc == "snr_fail"


def test_break_k_multiple_of_32_makes_block_k_illegal():
    out, fc = mutate.break_k_multiple_of_32(GEMM_SRC)
    assert out != GEMM_SRC
    # the K tile becomes 48 (non-pow2, non-mult-of-32)
    assert "BLOCK_K=48" in out
    assert fc == "compile_fail"


def test_break_transpose_operand_swaps_strides():
    out, fc = mutate.break_transpose_operand(GEMM_SRC)
    assert out != GEMM_SRC
    assert fc == "snr_fail"


def test_break_missing_mask_widens_or_drops_guard():
    out, fc = mutate.break_missing_mask(NORM_SRC)
    assert out != NORM_SRC
    assert fc == "snr_fail"


def test_break_missing_barrier_removes_barrier():
    out, fc = mutate.break_missing_barrier(BARRIER_SRC)
    assert out != BARRIER_SRC
    assert "debug_barrier" not in out
    assert fc == "snr_fail"


def test_break_block_m_to_64():
    out, fc = mutate.break_block_m_to_64(BLOCKM_SRC)
    assert out != BLOCKM_SRC
    assert "BLOCK_M = 64" in out
    assert fc == "snr_fail"


def test_new_family_aware_mutators_degrade_gracefully():
    # a mutator whose pattern is absent returns the source UNCHANGED (so the
    # dispatch can move to the next candidate) rather than appending junk.
    unchanged, _ = mutate.break_fp8_variant(BLOCKM_SRC)
    assert unchanged == BLOCKM_SRC
    unchanged, _ = mutate.break_missing_barrier(BLOCKM_SRC)
    assert unchanged == BLOCKM_SRC
    unchanged, _ = mutate.break_transpose_operand(NORM_SRC)
    assert unchanged == NORM_SRC
