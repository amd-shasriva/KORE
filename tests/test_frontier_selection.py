"""The frontier selector must rank by what a win is worth, not by availability.

The sweep it replaces mined 6,457 KernelBook modules whose median baseline was
17us -- launch-bound, so nothing about tiling, LDS staging or MFMA scheduling can
be expressed in them -- while 434 registry tasks in exactly the families the
arena scores hardest (170 attention, 115 MoE, 91 quant/fp8) sat unmined, one of
them reaching v5. These pin the ordering that stops that happening again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from select_frontier_tasks import (  # noqa: E402
    BASELINE_WEIGHT, DTYPE_WEIGHT, FAMILY_WEIGHT, MIN_ELEMENTS,
    classify_family, score_task)


def _meta(dtype="bf16", baseline="vendor", elements=MIN_ELEMENTS, **kw):
    m = {"dtype": dtype, "comparison_baseline": baseline,
         "provenance": {"primary_elements": elements}}
    m.update(kw)
    return m


# ---- family classification ------------------------------------------------

@pytest.mark.parametrize("name,expect", [
    ("flash_attn_varlen_bf16", "attention"),
    ("paged_attn_decode_bf16", "attention"),
    ("mla_decode_bf16", "attention"),
    ("genv_rope_bf16", "attention"),
    ("fused_moe_silu_bf16", "moe"),
    ("genv_topk_softmax_bf16", "moe"),
    ("genb_moe_align_block_offsets_bf16", "moe"),
    ("gemm_fp8_a8w8_blockscale", "quant"),
    ("genb_qx_mxfp4_pack_fp8", "quant"),
    ("gemm_w4a16_g128", "quant"),
    ("gemm_bf16", "gemm"),
    ("fused_rmsnorm_quant_fp8", "quant"),
])
def test_frontier_families_are_recognised(name, expect):
    assert classify_family(name) == expect


@pytest.mark.parametrize("name", [
    "kbk_elu_c33e2733_fp32", "kbk_clamp_13361270_fp32",
    "kbk_dice_loss_064fdb0b_fp32", "kbk_hard_swish_093d88bd_fp32",
])
def test_the_launch_bound_bulk_is_not_a_frontier_family(name):
    """These are the 86% under 100us. Scoring them at all was the old mistake."""
    assert classify_family(name) is None


def test_out_of_scope_tasks_score_zero_and_are_dropped():
    s, why = score_task("kbk_elu_c33e2733_fp32", _meta(), "pool")
    assert s == 0.0
    assert "family" in why["reason"]


# ---- the orderings that matter --------------------------------------------

def test_attention_outranks_gemm_at_equal_everything_else():
    a, _ = score_task("flash_attn_prefill_bf16", _meta(), "registry")
    g, _ = score_task("gemm_bf16", _meta(), "registry")
    assert a > g


def test_low_precision_outranks_fp32():
    """fp32 cannot reach the fp8/fp6/fp4 MFMA paths CDNA4 exists to offer."""
    lo, _ = score_task("gemm_fp8_a8w8", _meta(dtype="fp8_e4m3fn"), "registry")
    hi, _ = score_task("gemm_bf16", _meta(dtype="fp32"), "registry")
    assert lo > hi


def test_vendor_baseline_outranks_torch_baseline():
    """Beating hipBLASLt by 1.2x is a result; beating eager torch by 38x is not."""
    v, _ = score_task("gemm_bf16", _meta(baseline="aiter"), "registry")
    t, _ = score_task("gemm_bf16", _meta(baseline="torch"), "registry")
    assert v > t


def test_registry_outranks_pool_for_the_same_family():
    r, _ = score_task("gemm_bf16", _meta(baseline="torch"), "registry")
    p, _ = score_task("kbk_linear_gemm_x_fp32", _meta(baseline="external_pool"), "pool")
    assert r > p


def test_a_launch_bound_pool_task_is_heavily_penalised():
    """1M elements is the gate; below it the kernel cannot show optimisation."""
    big, _ = score_task("kbk_attn_x", _meta(elements=MIN_ELEMENTS), "pool")
    small, _ = score_task("kbk_attn_x", _meta(elements=1024), "pool")
    assert small < big / 3


def test_registry_is_not_penalised_for_unparsed_shapes():
    """A hand-authored task with no primary_elements must not rank below a
    17us KernelBook module purely because its shapes are declared differently."""
    reg, _ = score_task("flash_attn_prefill_bf16",
                        {"dtype": "bf16", "comparison_baseline": "torch"}, "registry")
    pool, _ = score_task("kbk_attn_y", _meta(dtype="fp32", baseline="external_pool"),
                         "pool")
    assert reg > pool


def test_the_single_highest_value_shape_is_vendor_fp8_quant():
    """What the arena pays most for: low precision, tuned baseline, hard family."""
    top, _ = score_task("gemm_fp8_a8w8_blockscale",
                        _meta(dtype="fp8_e4m3fn", baseline="aiter"), "registry")
    for other in ("gemm_bf16", "fused_rmsnorm_quant_fp8"):
        s, _ = score_task(other, _meta(dtype="fp32", baseline="torch"), "registry")
        assert top > s


# ---- weight-table sanity ---------------------------------------------------

def test_weight_tables_are_ordered_as_documented():
    assert FAMILY_WEIGHT["attention"] > FAMILY_WEIGHT["moe"] > FAMILY_WEIGHT["gemm"]
    assert DTYPE_WEIGHT["mxfp4"] > DTYPE_WEIGHT["fp8"] > DTYPE_WEIGHT["bf16"] > DTYPE_WEIGHT["fp32"]
    assert BASELINE_WEIGHT["vendor"] > BASELINE_WEIGHT["compile"] > BASELINE_WEIGHT["torch"]
    assert BASELINE_WEIGHT["torch"] > BASELINE_WEIGHT["external_pool"]


def test_every_family_weight_has_a_pattern_and_vice_versa():
    from select_frontier_tasks import FAMILY_PATTERNS
    assert {l for l, _ in FAMILY_PATTERNS} == set(FAMILY_WEIGHT)
