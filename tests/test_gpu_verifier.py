"""Real-hardware verifier tests: the execution boundary, not a mock of it.

Everything the project claims rests on ``KoreEnv`` actually compiling, running
and timing a kernel on a gfx950 card, yet the CPU suite reaches those verdicts by
monkeypatching ``_exec``/``_env``/``_bench_all`` away (see
``tests/test_phase0_verifier_fixes.py``). These tests drive the same code with no
stubs at all: the driver really runs in a subprocess pinned to one physical GPU,
the candidate really compiles through Triton, and the verdicts come from measured
output.

Covered here:

* the happy path end to end - compile, correctness, finite positive timing, and
  the reward tier the observation lands in;
* a deliberately wrong kernel rejected by hardware rather than by a stub;
* the post-timing re-verification catching a stateful kernel that is correct when
  checked and wrong while timed (the invocation-count timing hack);
* the AITER vendor baseline - whether the runtime loads at all on this node, and
  whether the ``KORE_BASELINE_IMPL:`` sentinel honestly reports ``aiter_vendor``
  instead of a silent ``framework`` fallback;
* ``HIP_VISIBLE_DEVICES`` mapping and detection of a partial GPU allocation.

Run with ``python -m pytest -m gpu -q``. Pinning is deliberate: see
``conftest.DEFAULT_TEST_GPU``.
"""

from __future__ import annotations

import math
import re

import pytest

from kore.config import CONFIG
from kore.data.schemas import BASELINE_KIND_VENDOR, classify_baseline_kind
from kore.reward.reward import compute_reward, scan_for_hacks

pytestmark = pytest.mark.gpu

#: Tags ``kore.tasks.aiter_ref._mark_baseline`` may emit, and which of them are a
#: real production kernel rather than the torch framework fallback.
VENDOR_SENTINELS = ("aiter_vendor", "hipblaslt_vendor")
KNOWN_SENTINELS = VENDOR_SENTINELS + ("framework",)

#: Tasks whose declared ``comparison_baseline`` is a production vendor kernel.
#: The verifier never puts the runtime sentinel on its ``Observation`` (see
#: ``kore.data.schemas.resolve_baseline_identity``), so "declared vendor" is an
#: unconfirmed claim until something runs the bench process and reads the
#: sentinel - which is exactly what these tests do.
#:
#: This is a verified subset, not every vendor-declared task. ``moe_gelu_bf16``
#: is deliberately absent: it declares ``aiter_fused_moe`` while its
#: ``baseline_output`` times per-expert ``torch.matmul`` (hipBLASLt) because
#: AITER's fused MoE does not JIT-build here, and it emits no sentinel at all, so
#: nothing at runtime can say which vendor library produced its baseline. Adding
#: it here is the reproducer for that provenance gap; it belongs in the list once
#: the task marks its baseline.
VENDOR_BASELINE_TASKS = ("rmsnorm_aiter", "silu_mul_bf16", "gemm_bf16")

#: A task that deliberately declares the torch framework path as its baseline
#: (AITER ships no standalone plain GELU-tanh kernel), so its honest sentinel is
#: ``framework``. It guards the other direction: a framework bar must never be
#: labelled as vendor.
FRAMEWORK_BASELINE_TASK = "gelu_tanh_bf16"


def _wrong_kernel(seed_source: str) -> str:
    """The seed with its arithmetic broken, so it compiles but computes garbage.

    Halving one addend keeps the kernel structurally honest - it still launches
    and still stores - so nothing but a real numeric comparison against the
    oracle can reject it.
    """
    broken = seed_source.replace("o = x + y", "o = x + y * 0.5")
    assert broken != seed_source, "seed kernel no longer contains the patched line"
    return broken


#: Correct for the first ``_HONEST`` invocations, wrong afterwards. The driver
#: caches the candidate MODULE, so the counter survives from the timed loop into
#: the post-timing correctness re-verification: correctness runs in their own
#: process and see ~5 calls, while the timed loop burns >100, so the re-check
#: lands in the dishonest regime. This is the invocation-count timing hack the
#: environment claims to catch, and it has never been tested against hardware.
STATEFUL_KERNEL = '''"""Candidate that is correct while checked and wrong while timed."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_CALLS = 0
_HONEST = 24


@triton.jit
def _add_kernel(a_ptr, b_ptr, o_ptr, stride_am, stride_bm, stride_om, N, gain,
                BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(a_ptr + row * stride_am + offs, mask=mask, other=1.0).to(tl.float32)
    y = tl.load(b_ptr + row * stride_bm + offs, mask=mask, other=1.0).to(tl.float32)
    o = x + y * gain
    tl.store(o_ptr + row * stride_om + offs, o.to(tl.float32), mask=mask)


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    global _CALLS
    _CALLS += 1
    gain = 1.0 if _CALLS <= _HONEST else 0.0
    M, N = a.shape
    o = torch.empty_like(a)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _add_kernel[grid](a, b, o, a.stride(0), b.stride(0), o.stride(0), N, gain,
                      BLOCK_N=BLOCK_N, num_warps=4)
    return o
'''


# --------------------------------------------------------------------------- #
# hardware sanity
# --------------------------------------------------------------------------- #
def test_child_sees_the_requested_gpu_and_target_arch(gpu_harness, gpu_id) -> None:
    probe = gpu_harness.probe
    assert probe["device_count"] >= 1
    assert probe["hip_version"], "torch is not a ROCm build"
    assert probe["visibility"]["HIP_VISIBLE_DEVICES"] == gpu_id
    assert probe["gpu_target"] == CONFIG.gpu_target
    arch = probe["devices"][0]["arch"]
    assert arch.startswith(CONFIG.gpu_target), (
        f"device reports {arch!r} but KORE targets {CONFIG.gpu_target!r}; every "
        "timing claim and every fp8 encoding choice is arch-specific")


# --------------------------------------------------------------------------- #
# 1. the full evaluate path
# --------------------------------------------------------------------------- #
def test_seed_kernel_compiles_and_passes_correctness(seed_observation,
                                                     gpu_harness) -> None:
    obs = seed_observation
    assert obs.infra_error is False, f"infra failure: {obs.error_text}"
    assert obs.compiled is True
    assert obs.flagged_hack is False
    assert obs.validation_passed is True, obs.error_text
    shape = gpu_harness.shape("minimal")
    assert set(obs.snr_by_shape) == {shape.name}
    threshold = gpu_harness.task.snr_threshold
    assert obs.snr_by_shape[shape.name] >= threshold


def test_seed_kernel_produces_finite_positive_timing(seed_observation,
                                                     gpu_harness) -> None:
    obs = seed_observation
    name = gpu_harness.shape("minimal").name
    assert obs.timing_requested is True
    for label, table in (("candidate", obs.wall_by_shape),
                         ("baseline", obs.baseline_by_shape)):
        assert set(table) == {name}, f"{label} timing keys {sorted(table)}"
        value = table[name]
        assert math.isfinite(value) and value > 0.0, f"{label} timing {value!r}"


def test_seed_kernel_lands_in_the_correct_reward_tier(seed_observation,
                                                      gpu_harness) -> None:
    """Correct + timed must reach the correct tier, and the grade must match it.

    A busy node can legitimately fail the CV/CI admission gates, which
    ``KoreEnv._run`` now demotes to ``timing_grade="screening"`` instead of
    calling it an infra error. Both outcomes are asserted precisely rather than
    accepting either one loosely: publication grade must survive the reward's
    independent recompute and score in ``correct_timed``, and screening must be
    justified by a stated measurement-noise reason and score in
    ``correct_screening``.
    """
    from kore.reward.reward import _publication_timing_error

    obs = seed_observation
    result = compute_reward(obs, gpu_harness.task.seed_source,
                            dtype=gpu_harness.task.dtype)
    assert result.correct is True, result.detail
    assert result.reward >= CONFIG.correctness_weight
    assert "hack" not in result.flags and "infra" not in result.flags

    if obs.timing_grade == "publication":
        assert obs.performance_eligible is True
        assert _publication_timing_error(obs, CONFIG) is None
        assert result.tier == "correct_timed"
        assert result.speedup is not None and result.speedup > 0.0
        assert math.isfinite(result.speedup)
    else:
        assert obs.timing_grade == "screening", (
            f"unexpected timing grade {obs.timing_grade!r}: {obs.error_text}")
        assert obs.performance_eligible is False
        assert "measurement noise" in (obs.error_text or "")
        assert result.tier == "correct_screening"


# --------------------------------------------------------------------------- #
# 2. a deliberately wrong kernel is rejected by hardware
# --------------------------------------------------------------------------- #
def test_wrong_kernel_is_rejected_by_measured_output(gpu_harness) -> None:
    source = _wrong_kernel(gpu_harness.task.seed_source)
    assert scan_for_hacks(source) is None, (
        "the wrong kernel must be rejected by the numeric gate, not by the "
        "static scanner - otherwise this test proves nothing about hardware")

    obs = gpu_harness.evaluate(source, [gpu_harness.shape("minimal")])

    assert obs.infra_error is False, f"infra failure: {obs.error_text}"
    assert obs.compiled is True, "the wrong kernel must still compile"
    assert obs.validation_passed is False
    name = gpu_harness.shape("minimal").name
    assert obs.snr_by_shape[name] < gpu_harness.task.snr_threshold
    assert not obs.wall_by_shape, "an incorrect kernel must never be timed"

    result = compute_reward(obs, source, dtype=gpu_harness.task.dtype)
    assert result.correct is False
    assert result.tier == "incorrect"
    assert result.reward < CONFIG.correctness_weight


# --------------------------------------------------------------------------- #
# 4. post-timing re-verification catches a stateful kernel
# --------------------------------------------------------------------------- #
def test_post_timing_reverification_catches_a_stateful_kernel(gpu_harness) -> None:
    """A kernel correct when checked and wrong when timed must be flagged.

    The guarantee is structural: the driver caches the candidate module so state
    persists from the timed loop into the re-check, the timed window is
    randomized so the kernel cannot count its way around it, and the environment
    turns a failed post-timing verdict into the anti-hack floor. None of that had
    ever been exercised against a real GPU.
    """
    assert scan_for_hacks(STATEFUL_KERNEL) is None, (
        "the stateful kernel must survive the static scanner so the runtime "
        "post-timing re-verification is what rejects it")

    obs = gpu_harness.evaluate(STATEFUL_KERNEL, [gpu_harness.shape("minimal")])

    assert obs.infra_error is False, f"infra failure: {obs.error_text}"
    assert obs.flagged_hack is True, (
        "the stateful kernel passed timing undetected: post-timing "
        f"re-verification did not fire (grade={obs.timing_grade!r}, "
        f"timing={obs.wall_by_shape})")
    assert obs.hack_reason == "bench-time output mismatch"
    assert obs.validation_passed is False
    assert not obs.wall_by_shape

    result = compute_reward(obs, STATEFUL_KERNEL, dtype=gpu_harness.task.dtype)
    assert result.tier == "hack"
    assert result.reward == pytest.approx(CONFIG.reward_hack)
    assert result.reward < CONFIG.reward_compile_fail


def test_stateful_kernel_passes_correctness_without_timing(gpu_harness) -> None:
    """The hack really is invisible to the correctness gate alone.

    Without this, the test above could be passing because the kernel is simply
    broken. With ``do_bench=False`` there is no timed loop, the invocation count
    stays in the honest regime, and the candidate is accepted - which is what
    makes the post-timing re-check the only thing standing in its way.
    """
    obs = gpu_harness.evaluate(STATEFUL_KERNEL, [gpu_harness.shape("minimal")],
                               do_bench=False)

    assert obs.infra_error is False, f"infra failure: {obs.error_text}"
    assert obs.compiled is True
    assert obs.validation_passed is True, obs.error_text
    assert obs.flagged_hack is False
    assert obs.timing_requested is False


# --------------------------------------------------------------------------- #
# 6. the AITER vendor baseline
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def aiter_report(gpu_harness):
    """AITER availability plus the sentinel each wrapper emits, from a child."""
    report, raw = gpu_harness.aiter_probe()
    if report is None:
        pytest.skip(
            "the AITER probe produced no report, so vendor-baseline behavior is "
            f"unknown on this node; child output tail: {raw.strip()[-400:]!r}")
    return report


def test_aiter_runtime_loads_on_this_node(aiter_report) -> None:
    """Whether the production baseline runtime is actually present here.

    A skip is the honest verdict when AITER is missing: every ``aiter_*`` task
    would then be measuring the torch framework path while its ``task.yaml``
    still claims a vendor bar, and a passing test would hide that.
    """
    if aiter_report["import_error"]:
        pytest.skip(
            "AITER is not importable on this node "
            f"({aiter_report['import_error']}); every task declaring an aiter_* "
            "baseline silently measures the torch framework path instead")
    resolved = {op: case["resolved"] for op, case in aiter_report["cases"].items()}
    assert any(resolved.values()), (
        "AITER imports but exposes none of the KORE baseline ops "
        f"(resolution: {aiter_report['cases']})")


def test_aiter_wrapper_sentinels_are_honest(aiter_report) -> None:
    """``aiter_vendor`` is claimed only when the AITER op really resolved.

    The wrapper degrades to torch on any failure, so the sentinel is the only
    signal separating "we beat the vendor kernel" from "we beat eager torch".
    Both directions are checked: a vendor claim requires a resolved AITER
    callable, and a failed resolution must be reported as ``framework``. The
    fallback must also still be numerically correct - a wrong baseline is worse
    than a slow one.
    """
    if aiter_report["import_error"]:
        pytest.skip(f"AITER unavailable: {aiter_report['import_error']}")
    for op, case in aiter_report["cases"].items():
        sentinels = case["sentinels"]
        assert len(sentinels) == 1, f"{op}: expected one sentinel, got {sentinels}"
        tag = sentinels[0]
        assert tag in KNOWN_SENTINELS, f"{op}: unknown baseline tag {tag!r}"
        if tag == "aiter_vendor":
            assert case["resolved"] is True, (
                f"{op}: claims aiter_vendor while the AITER op did not resolve "
                f"({case['resolve_error']})")
            assert case["call_error"] is None
        if not case["resolved"]:
            assert tag == "framework", (
                f"{op}: AITER op unresolved ({case['resolve_error']}) but the "
                f"baseline was labelled {tag!r}")
        assert case["matches_oracle"] is True, (
            f"{op}: baseline {tag!r} disagrees with the torch oracle")


@pytest.mark.parametrize("task_id", VENDOR_BASELINE_TASKS)
def test_declared_vendor_baseline_is_confirmed_at_runtime(gpu_harness,
                                                          task_id) -> None:
    """A task claiming a vendor bar must emit a vendor sentinel when benched.

    ``resolve_baseline_identity`` classifies these tasks as ``vendor`` purely
    from the declared string and documents that it "never claims runtime
    confirmation". This is that confirmation: run the driver's real
    ``--impl reference`` bench and read the sentinel it prints.
    """
    task = gpu_harness.task_by_id(task_id)
    assert classify_baseline_kind(task.comparison_baseline) == BASELINE_KIND_VENDOR

    workdir = gpu_harness.stage(task.seed_source, task=task)
    shape = task.shape("minimal") or task.shapes[0]
    rc, out, timed_out = gpu_harness.run_driver(
        workdir, "--bench-mode", "--impl", "reference",
        "--warmup", "3", "--iters", "5", *shape.as_args(), task=task)

    assert timed_out is False and rc == 0, f"reference bench failed: {out[-800:]}"
    medians = re.findall(r"median_ms:\s*([-\d.eE]+)", out)
    assert medians, f"reference bench printed no median_ms: {out[-800:]}"
    assert float(medians[-1]) > 0.0

    tags = re.findall(rf"{gpu_harness.BASELINE_SENTINEL}(\w+)", out)
    assert tags, (
        f"{task_id} declares vendor baseline {task.comparison_baseline!r} but the "
        "bench process emitted no KORE_BASELINE_IMPL sentinel, so the vendor "
        "claim is unverifiable")
    assert tags[-1] in VENDOR_SENTINELS, (
        f"{task_id} declares vendor baseline {task.comparison_baseline!r} but "
        f"measured against {tags[-1]!r} - the vendor kernel silently fell back, "
        "so any reported speedup is against the framework path, not the "
        "production bar")


def test_framework_baseline_task_is_not_labelled_vendor(gpu_harness) -> None:
    """A declared framework bar must report ``framework``, never a vendor tag."""
    task = gpu_harness.task_by_id(FRAMEWORK_BASELINE_TASK)
    assert classify_baseline_kind(task.comparison_baseline) != BASELINE_KIND_VENDOR

    workdir = gpu_harness.stage(task.seed_source, task=task)
    shape = task.shape("minimal") or task.shapes[0]
    rc, out, timed_out = gpu_harness.run_driver(
        workdir, "--bench-mode", "--impl", "reference",
        "--warmup", "3", "--iters", "5", *shape.as_args(), task=task)

    assert timed_out is False and rc == 0, f"reference bench failed: {out[-800:]}"
    tags = re.findall(rf"{gpu_harness.BASELINE_SENTINEL}(\w+)", out)
    assert tags == ["framework"], (
        f"{FRAMEWORK_BASELINE_TASK} declares {task.comparison_baseline!r} "
        f"(framework) but reported {tags!r}")


# --------------------------------------------------------------------------- #
# 7. GPU count and visibility handling
# --------------------------------------------------------------------------- #
def test_explicit_gpu_pin_selects_that_physical_card(gpu_harness, gpu_id,
                                                     alt_gpu_id) -> None:
    """``HIP_VISIBLE_DEVICES`` must resolve to the card the caller asked for.

    Under distributed GRPO every rank benches on its own card, so a mapping bug
    silently puts several ranks on one GPU - they contend, the measurement
    inflates, and the cross-rank all_gather stalls. Device UUIDs make the mapping
    checkable without trusting the ordinal: the same physical card must answer
    for the same id, and two different ids must be two different cards.
    """
    primary = gpu_harness.device_probe(gpu=gpu_id)
    secondary = gpu_harness.device_probe(gpu=alt_gpu_id)
    assert primary and secondary

    for probe, requested in ((primary, gpu_id), (secondary, alt_gpu_id)):
        assert probe["device_count"] == 1, (
            f"HIP_VISIBLE_DEVICES={requested} exposed "
            f"{probe['device_count']} devices; the verifier subprocess must see "
            "exactly the one card it was pinned to")
        assert probe["visibility"]["HIP_VISIBLE_DEVICES"] == requested
        assert probe["visibility"]["CUDA_VISIBLE_DEVICES"] == requested
        assert probe["visibility"]["ROCR_VISIBLE_DEVICES"] is None, (
            "an inherited ROCR mask must be dropped: intersecting it with KORE's "
            "HIP mask can leave the child with no device at all")

    uuids = (primary["devices"][0]["uuid"], secondary["devices"][0]["uuid"])
    if not all(uuids):
        pytest.skip("this ROCm build reports no device uuid to compare against")
    assert uuids[0] != uuids[1], (
        f"HIP_VISIBLE_DEVICES={gpu_id} and ={alt_gpu_id} resolved to the same "
        f"physical card ({uuids[0]}) - the pin is not selecting a card")


def test_partial_allocation_is_visible_to_the_child(gpu_harness, gpu_id,
                                                    alt_gpu_id) -> None:
    """A restricted allocation must be observable, not silently full-node.

    ``scripts/spur_gpu_smoke.sbatch`` hard-fails unless torch reports all 8
    cards, so the difference between a full node and a partial allocation has to
    be detectable from inside the child. It is: the masked child counts exactly
    the cards in its mask, and the ordinals map positionally onto the physical
    ids, so a two-card allocation is not mistaken for the whole node.
    """
    pair = f"{gpu_id},{alt_gpu_id}"
    both = gpu_harness.device_probe(gpu=pair)
    assert both and both["device_count"] == 2, (
        f"HIP_VISIBLE_DEVICES={pair} exposed {both and both['device_count']} "
        "devices instead of 2")
    assert both["visibility"]["HIP_VISIBLE_DEVICES"] == pair

    single = gpu_harness.device_probe(gpu=gpu_id)
    assert single["device_count"] < both["device_count"], (
        "a narrower mask did not narrow what the child sees, so a partial "
        "allocation is indistinguishable from a full one")

    uuids = [device["uuid"] for device in both["devices"]]
    if not all(uuids) or not single["devices"][0]["uuid"]:
        pytest.skip("this ROCm build reports no device uuid to compare against")
    alt = gpu_harness.device_probe(gpu=alt_gpu_id)
    assert uuids == [single["devices"][0]["uuid"], alt["devices"][0]["uuid"]], (
        f"HIP_VISIBLE_DEVICES={pair} did not enumerate the cards in the order "
        "given, so a rank-to-card assignment cannot be trusted")


def test_gpu_selection_record_matches_what_the_child_receives(gpu_harness,
                                                             gpu_id) -> None:
    """The provenance the contract records must be the environment actually used.

    ``_gpu_selection`` is what lands in the evaluation contract (and therefore in
    replay identity), so a drift between it and the child's real environment
    would let a cached observation claim a card it never ran on.
    """
    kore_env = gpu_harness.kore_env(gpu=gpu_id)
    selection = kore_env._gpu_selection(gpu_harness.task)
    assert selection["mode"] == "explicit-physical"
    assert selection["selected_gpu"] == gpu_id
    assert selection["effective_gpu_target"] == gpu_harness.task.gpu_target

    probe = gpu_harness.device_probe(gpu=gpu_id)
    child = selection["child_visibility"]
    assert probe["visibility"]["HIP_VISIBLE_DEVICES"] == child["HIP_VISIBLE_DEVICES"]
    assert probe["visibility"]["CUDA_VISIBLE_DEVICES"] == child["CUDA_VISIBLE_DEVICES"]
    assert probe["gpu_target"] == selection["effective_gpu_target"]
