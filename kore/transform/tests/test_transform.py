"""CPU-only, pure-string tests for the KORE transformation calculus.

No GPU, no torch/triton import - every transform is exercised on small in-file
Triton *source* fixtures. Covers: the menu shape + relation tags, that each
transform edits its intended knob/token, side-condition rejection of illegal
params, ε-budget composition (weakest = max) + blocking when exhausted, and that
``admissible_actions`` (the RL action space) shrinks as budget is spent.
"""

from __future__ import annotations

from kore.transform import (
    APPROX,
    EXACT,
    LIBRARY,
    RELATION_APPROX,
    RELATION_EXACT,
    Action,
    ErrorBudget,
    action_menu,
    admissible_actions,
    apply_sequence,
    compose_eps,
    compose_relation,
    default_budget,
    get,
)
from kore.transform.budget import DEFAULT_BUDGET_TABLE

# --------------------------------------------------------------------------- #
# Fixtures - representative Triton kernel SOURCES (strings only).
# --------------------------------------------------------------------------- #
# A bf16 GEMM mirroring kore/tasks/gemm_int8_a8w8/seed_triton.py conventions:
# tuple BLOCK defn, GROUP_M swizzle, fp32 acc, tl.dot accumulate, masked IO,
# cdiv K-loop, num_warps/num_stages launch kwargs.
GEMM = '''\
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=kmask, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def gemm(a, b):
    M, K = a.shape
    N, K2 = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 128, 64, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=4, num_stages=2,
    )
    return c
'''

# A bf16 elementwise kernel (vector-add + logistic gate) with an UNMASKED store
# and a true reciprocal - the sites add_mask_boundary / fast_math_recip target.
ELEMENTWISE = '''\
import torch
import triton
import triton.language as tl


@triton.jit
def _act_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    denom = 1.0 + tl.exp(-x)
    inv_denom = 1.0 / denom
    out = (x + y) * inv_denom
    tl.store(out_ptr + offsets, out)


def activation(x, y):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _act_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out
'''

# A deliberately buggy GEMM whose accumulator is bf16 (not fp32) - exercises
# fp32_accumulator and the downcast<-fp32-acc side-condition composition.
BF16_ACC = '''\
import torch
import triton
import triton.language as tl


@triton.jit
def _bad_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_am = tl.arange(0, BLOCK_M)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=kmask, other=0.0)
        b = tl.load(b_ptrs, mask=kmask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = tl.arange(0, BLOCK_M)
    offs_cn = tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def gemm(a, b):
    M, K = a.shape
    N, K2 = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 128, 64, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _bad_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=4, num_stages=2,
    )
    return c
'''


def _fresh(total=10.0):
    """A generous budget so budget gating never interferes with knob-edit tests."""
    return ErrorBudget(total=total)


# --------------------------------------------------------------------------- #
# Menu / library shape
# --------------------------------------------------------------------------- #
def test_library_has_at_least_12_transforms():
    assert len(LIBRARY) >= 12
    names = [t.name for t in LIBRARY]
    assert len(names) == len(set(names)), "transform names must be unique"


def test_every_transform_is_tagged_exact_or_approx():
    for t in LIBRARY:
        assert t.relation in (RELATION_EXACT, RELATION_APPROX)
        assert t.knob and t.summary
    # both families are represented, and exact truly costs no ε
    assert len(EXACT) >= 5 and len(APPROX) >= 5
    for t in EXACT:
        assert t.epsilon(value=8) == 0.0
    for t in APPROX:
        assert t.default_eps >= 0.0


def test_action_menu_is_serializable():
    menu = action_menu()
    assert len(menu) == len(LIBRARY)
    for row in menu:
        assert set(row) == {"name", "relation", "knob", "summary"}


# --------------------------------------------------------------------------- #
# Each transform edits its intended knob / token
# --------------------------------------------------------------------------- #
def test_set_num_warps_changes_num_warps():
    out = get("set_num_warps").apply(GEMM, value=8)
    assert "num_warps=8" in out and "num_warps=4" not in out


def test_set_num_stages_changes_num_stages():
    out = get("set_num_stages").apply(GEMM, value=3)
    assert "num_stages=3" in out and "num_stages=2" not in out


def test_set_waves_per_eu_wires_the_amd_hint():
    out = get("set_waves_per_eu").apply(GEMM, value=2)
    assert "waves_per_eu=2" in out


def test_swizzle_group_m_changes_group_m():
    out = get("swizzle_group_m").apply(GEMM, value=16)
    # GROUP_M is the 4th entry of the positional tuple defn
    assert "BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 128, 64, 16" in out


def test_vectorize_loads_adds_contiguity_hints():
    out = get("vectorize_loads").apply(GEMM)
    assert "tl.multiple_of" in out and "tl.max_contiguous" in out


def test_add_mask_boundary_masks_unmasked_store():
    assert get("add_mask_boundary").apply(GEMM) is None  # GEMM is fully masked
    out = get("add_mask_boundary").apply(ELEMENTWISE)
    assert "tl.store(out_ptr + offsets, out, mask=mask)" in out
    assert out.count("mask=") > ELEMENTWISE.count("mask=")


def test_reorder_loads_swaps_independent_loads():
    out = get("reorder_loads").apply(GEMM)
    assert out is not None and out != GEMM
    # the b-load now precedes the a-load in the reduction body
    assert out.index("b = tl.load(") < out.index("a = tl.load(")


def test_fp32_accumulator_forces_fp32_acc():
    assert get("fp32_accumulator").apply(GEMM) is None  # already fp32
    out = get("fp32_accumulator").apply(BF16_ACC)
    assert "tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)" in out
    assert "dtype=tl.bfloat16)" not in out.split("acc = tl.zeros")[1][:60]


def test_retile_block_changes_block_sizes():
    out = get("retile_block").apply(GEMM, block_m=128, block_k=128)
    assert "BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 128, 8" in out


def test_split_k_introduces_split_knob_and_atomics():
    out = get("split_k").apply(GEMM, value=4)
    assert out is not None
    assert "SPLIT_K" in out
    assert "SPLIT_K=4" in out                       # wired into the launch
    assert "range(pid_k, tl.cdiv(K, BLOCK_K), SPLIT_K)" in out
    assert "tl.atomic_add(" in out and "pid_k = tl.program_id(axis=1)" in out


def test_split_k_is_inapplicable_without_k_loop():
    assert get("split_k").apply(ELEMENTWISE, value=2) is None


def test_downcast_dtype_changes_io_dtype():
    out = get("downcast_dtype").apply(GEMM, to="fp16")
    assert ".to(tl.float16)" in out and ".to(tl.bfloat16)" not in out
    assert "dtype=torch.float16" in out
    # the fp32 accumulator is preserved (downcast is IO-only)
    assert "tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)" in out


def test_reassociate_reduction_fuses_the_dot_accumulate():
    out = get("reassociate_reduction").apply(GEMM)
    assert "acc = tl.dot(a, b, acc)" in out and "acc += tl.dot" not in out


def test_fast_math_recip_uses_hardware_reciprocal():
    assert get("fast_math_recip").apply(GEMM) is None  # no division in GEMM
    out = get("fast_math_recip").apply(ELEMENTWISE)
    assert "tl.math.rcp(denom)" in out and "1.0 / denom" not in out


def test_exact_transforms_keep_the_kernel_wellformed():
    # An exact rewrite preserves the structural skeleton of the kernel.
    for name in ("set_num_warps", "swizzle_group_m", "vectorize_loads", "reorder_loads"):
        params = {"value": 8} if name == "set_num_warps" else (
            {"value": 16} if name == "swizzle_group_m" else {})
        out = get(name).apply(GEMM, **params)
        assert out and "@triton.jit" in out and "def gemm(" in out


# --------------------------------------------------------------------------- #
# Side conditions reject illegal params
# --------------------------------------------------------------------------- #
def test_retile_rejects_non_64_multiple_block():
    viol = get("retile_block").side_conditions(GEMM, block_m=100)
    assert viol and any("multiple of 64" in v for v in viol)
    # ... and apply_sequence refuses it without mutating the source
    budget = _fresh()
    new, applied, rejected, _ = apply_sequence(GEMM, [("retile_block", {"block_m": 100})], budget)
    assert new == GEMM and not applied
    assert rejected and rejected[0]["reason"] == "side_condition"


def test_num_warps_rejects_non_power_of_two():
    assert get("set_num_warps").side_conditions(GEMM, value=3)
    assert not get("set_num_warps").side_conditions(GEMM, value=8)


def test_split_k_rejects_non_power_of_two():
    assert get("split_k").side_conditions(GEMM, value=3)
    assert not get("split_k").side_conditions(GEMM, value=4)


def test_downcast_rejects_non_lowp_target_and_bf16_accumulator():
    assert get("downcast_dtype").side_conditions(GEMM, to="fp64")
    # downcasting IO on a bf16-accumulator kernel is inadmissible until fp32 acc
    assert get("downcast_dtype").side_conditions(BF16_ACC, to="fp16")
    assert not get("downcast_dtype").side_conditions(GEMM, to="fp16")


def test_downcast_composes_after_fp32_accumulator():
    # on the buggy kernel the downcast is rejected by its side condition ...
    _, applied, rejected, _ = apply_sequence(
        BF16_ACC, [("downcast_dtype", {"to": "fp16"})], _fresh())
    assert not applied and rejected[0]["reason"] == "side_condition"
    # ... but fp32_accumulator first makes it admissible (a 2-step trajectory)
    new, applied, rejected, _ = apply_sequence(
        BF16_ACC,
        [("fp32_accumulator", {}), ("downcast_dtype", {"to": "fp16"})],
        _fresh())
    assert [a["name"] for a in applied] == ["fp32_accumulator", "downcast_dtype"]
    assert "dtype=tl.float32" in new and ".to(tl.float16)" in new


# --------------------------------------------------------------------------- #
# Relation algebra + ε-budget composition
# --------------------------------------------------------------------------- #
def test_relation_algebra_helpers():
    assert compose_relation(RELATION_EXACT, RELATION_EXACT) == RELATION_EXACT
    assert compose_relation(RELATION_EXACT, RELATION_APPROX) == RELATION_APPROX
    assert compose_relation(RELATION_APPROX, RELATION_EXACT) == RELATION_APPROX
    assert compose_eps(0.02, 0.06) == 0.06  # weakest = max


def test_exact_only_trajectory_stays_exact_and_spends_nothing():
    budget = _fresh(total=0.5)
    new, applied, rejected, state = apply_sequence(
        GEMM,
        [("set_num_warps", {"value": 8}), ("swizzle_group_m", {"value": 16})],
        budget)
    assert len(applied) == 2 and not rejected
    assert state["relation"] == RELATION_EXACT
    assert state["spent"] == 0.0 and budget.remaining() == 0.5
    assert new != GEMM


def test_approx_trajectory_composes_to_weakest_max_eps():
    budget = _fresh(total=0.5)
    _, applied, _, state = apply_sequence(
        GEMM,
        [("retile_block", {"block_k": 128}),      # ε = 0.06 (reassociates K)
         ("reassociate_reduction", {})],           # ε = 0.01
        budget)
    assert len(applied) == 2
    assert state["relation"] == RELATION_APPROX
    # carried contract is the WEAKEST (max) step ε ...
    assert abs(state["weakest_eps"] - 0.06) < 1e-9
    # ... while the additive meter spent the sum
    assert abs(state["cumulative_eps"] - 0.07) < 1e-9


def test_budget_blocks_approx_when_exhausted_but_not_exact():
    budget = ErrorBudget(total=0.04)
    new, applied, rejected, state = apply_sequence(
        GEMM,
        [("downcast_dtype", {"to": "fp16"}),   # ε = 0.03  -> fits (spends)
         ("retile_block", {"block_k": 128}),   # ε = 0.06  -> exceeds remaining
         ("set_num_stages", {"value": 4})],    # exact     -> still allowed
        budget)
    names_applied = [a["name"] for a in applied]
    assert "downcast_dtype" in names_applied
    assert "set_num_stages" in names_applied          # exact unaffected by budget
    blocked = [r for r in rejected if r["name"] == "retile_block"]
    assert blocked and blocked[0]["reason"] == "budget_exhausted"
    assert budget.remaining() < 0.03


def test_apply_sequence_return_contract():
    budget = _fresh()
    result = apply_sequence(GEMM, [("set_num_warps", {"value": 8})], budget)
    assert isinstance(result, tuple) and len(result) == 4
    new, applied, rejected, state = result
    assert isinstance(new, str) and isinstance(applied, list)
    assert isinstance(rejected, list) and isinstance(state, dict)
    for key in ("total", "spent", "remaining", "relation", "weakest_eps",
                "cumulative_eps", "exhausted", "steps"):
        assert key in state


def test_unknown_transform_is_reported_not_raised():
    _, applied, rejected, _ = apply_sequence(GEMM, [("no_such_move", {})], _fresh())
    assert not applied and rejected[0]["reason"] == "unknown_transform"


# --------------------------------------------------------------------------- #
# admissible_actions == the (shrinking) RL action space
# --------------------------------------------------------------------------- #
def _split(actions):
    approx = [a for a in actions if a.relation == RELATION_APPROX]
    exact = [a for a in actions if a.relation == RELATION_EXACT]
    return exact, approx


def test_admissible_actions_are_actionable():
    actions = admissible_actions(GEMM, _fresh())
    assert actions and all(isinstance(a, Action) for a in actions)
    # every returned action really is applicable and (for approx) affordable
    budget = _fresh()
    for a in actions[:5]:
        new, applied, rejected, _ = apply_sequence(GEMM, [a.as_step()], _fresh())
        assert applied and not rejected and new != GEMM


def test_admissible_actions_shrink_as_budget_is_spent():
    full = admissible_actions(GEMM, ErrorBudget(total=default_budget("gemm", "bf16")))
    exact_full, approx_full = _split(full)
    assert approx_full, "expected some approx moves at full budget"

    spent = ErrorBudget(total=default_budget("gemm", "bf16"))
    spent.spend(spent.total)  # exhaust the ε budget
    after = admissible_actions(GEMM, spent)
    exact_after, approx_after = _split(after)

    assert not approx_after, "approx moves must vanish once the budget is spent"
    assert len(after) < len(full)
    # exact moves are budget-independent, so they are unchanged
    assert {a.name for a in exact_after} == {a.name for a in exact_full}


def test_partial_budget_prunes_only_unaffordable_approx():
    # a tiny budget keeps cheap approx (ε<=0.02) and drops expensive ones (K-split)
    tiny = ErrorBudget(total=0.02)
    actions = admissible_actions(GEMM, tiny)
    names = {a.name for a in actions}
    # a BLOCK_K retile (ε=0.06) and split_k (ε>=0.05) are unaffordable at 0.02,
    # so only the cheap M/N retiles (ε=0.02) survive.
    for a in actions:
        if a.name == "retile_block":
            assert "block_k" not in a.params
    assert "split_k" not in names               # ε>=0.05, pruned
    assert "reassociate_reduction" in names      # ε=0.01, still affordable


# --------------------------------------------------------------------------- #
# Per-(op, dtype) default budget table
# --------------------------------------------------------------------------- #
def test_default_budget_scales_with_precision():
    # low precision tolerates more numerical drift -> larger ε budget
    assert default_budget("gemm", "fp8") > default_budget("gemm", "bf16")
    assert default_budget("gemm", "bf16") > default_budget("gemm", "fp32")
    # shallow elementwise reductions get a larger budget than a deep GEMM
    assert default_budget("elementwise", "bf16") > default_budget("gemm", "bf16")


def test_default_budget_table_is_populated_and_positive():
    assert ("gemm", "fp32") in DEFAULT_BUDGET_TABLE
    assert all(v > 0 for v in DEFAULT_BUDGET_TABLE.values())
    assert DEFAULT_BUDGET_TABLE[("gemm", "fp32")] == default_budget("gemm", "fp32")


def test_budget_for_op_uses_the_table():
    b = ErrorBudget.for_op("gemm", "bf16")
    assert b.total == default_budget("gemm", "bf16")
    b2 = ErrorBudget.for_op("gemm", "bf16", total=0.5)
    assert b2.total == 0.5


# --------------------------------------------------------------------------- #
# Robustness: no transform ever raises on odd input
# --------------------------------------------------------------------------- #
def test_transforms_never_raise_on_degenerate_source():
    for t in LIBRARY:
        for src in ("", "   ", "not a kernel", GEMM[:40]):
            assert t.apply(src) is None or isinstance(t.apply(src), str)
            assert isinstance(t.side_conditions(src), list)
