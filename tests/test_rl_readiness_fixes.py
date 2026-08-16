"""Regressions for three defects found auditing the RL stage before launch.

Each of these was silent in the sense that matters: nothing crashed, no test
failed, and the number that came out was wrong or absent.

1. A HIP candidate was re-scanned for hacks in PYTHON mode at reward time, so
   its ``//`` comments survived into the scan and a comment naming a vendor
   library charged the -1.5 hack floor to an honest kernel. Measured over the 89
   real HIP kernels in AgentKernelArena, 7 hit exactly this.
2. ``_grpo_step_mismatch_stat`` read sample field 7, samples are built with
   seven fields (0..6), and nothing ever wrote an eighth -- so it returned
   ``(None, None)`` for every possible input. It was also never called. Both the
   30B recipe and docs/GRPO_READINESS.md justify ``mismatch_correction: false``
   on the grounds that this diagnostic runs "unconditionally".
3. The sharded training loop logged no advantage, KL, ratio-drift, grad-norm or
   LR diagnostic, so on the production path there was no way to confirm from the
   step log that TRLOO had taken effect.

CPU only.
"""

from __future__ import annotations

import re
from pathlib import Path

from kore.reward.reward import (
    detect_source_language,
    scan_for_hacks,
)

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# 1. language-aware hack scanning
# --------------------------------------------------------------------------- #
#: A HIP kernel that mentions a vendor library in a comment and calls none.
_HONEST_HIP = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Tiled 128x128 GEMM. Measured slightly faster than hipblaslt on this shape,
// and rocblas_sgemm was slower still, so neither is used here.
__global__ void gemm_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { c[i] = a[i] * b[i]; }
}

torch::Tensor run(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    hipLaunchKernelGGL(gemm_kernel, dim3((n + 255) / 256), dim3(256), 0, 0,
                       a.data_ptr<float>(), b.data_ptr<float>(),
                       c.data_ptr<float>(), n);
    return c;
}
"""


def test_a_vendor_name_in_a_cpp_comment_is_not_a_hack():
    assert scan_for_hacks(_HONEST_HIP, "cpp") is None, (
        "an honest HIP kernel that only NAMES a vendor library in a comment must "
        "not be charged the hack floor"
    )


def test_python_mode_is_what_produced_the_false_positive():
    """Pin the mechanism, so the fix cannot be reverted as cosmetic."""
    reason = scan_for_hacks(_HONEST_HIP, "python")
    assert reason is not None and "vendor library" in reason.lower(), (
        "scanning C++ as Python is expected to false-positive here; if this ever "
        "stops being true the language argument has stopped mattering"
    )


def test_detection_calls_hip_cpp_and_leaves_triton_alone():
    assert detect_source_language(_HONEST_HIP) == "cpp"
    triton = (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "@triton.jit\n"
        "def k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    off = pid * BLOCK + tl.arange(0, BLOCK)\n"
        "    tl.store(y_ptr + off, tl.load(x_ptr + off, mask=off < n), mask=off < n)\n"
    )
    assert detect_source_language(triton) == "python"
    # Detection is one-directional: no marker means Python, never a guess.
    assert detect_source_language("") == "python"
    assert detect_source_language("x = 1") == "python"


def test_a_real_vendor_link_is_still_a_hack_in_cpp_mode():
    """The fix must not have turned the scan off for C++."""
    cheating = _HONEST_HIP.replace(
        "#include <torch/extension.h>",
        "#include <torch/extension.h>\n#include <hipblaslt/hipblaslt.h>",
    )
    assert scan_for_hacks(cheating, "cpp") is not None
    assert detect_source_language(cheating) == "cpp"


def test_python_mode_also_HID_a_real_vendor_link():
    """The mirror-image defect, and the reason this is not just a false-positive fix.

    Python comment stripping deletes everything after a ``#``, and a C++ vendor
    dependency is spelled ``#include <hipblaslt/hipblaslt.h>``. So scanning a HIP
    candidate as Python did not merely charge honest kernels the hack floor -- it
    also erased the single strongest piece of evidence that a kernel really did
    delegate to a vendor library, and let that one through clean.
    """
    cheating = _HONEST_HIP.replace(
        "#include <torch/extension.h>",
        "#include <torch/extension.h>\n#include <hipblaslt/hipblaslt.h>",
    )
    reason = scan_for_hacks(cheating, "python")
    # It is "caught", but for the wrong reason: the comment, not the include.
    # Strip the comments a C++ compiler would and the Python scan sees nothing.
    no_comments = re.sub(r"//[^\n]*", "", cheating)
    assert scan_for_hacks(no_comments, "python") is None, (
        "expected the Python scan to MISS a real vendor include once the "
        "comments are gone, because '#include ...' looks like a Python comment"
    )
    assert scan_for_hacks(no_comments, "cpp") is not None, (
        "cpp mode must still catch the real vendor include"
    )
    assert reason is not None  # the comment false-positive, documented above


def test_the_reward_path_scans_hip_as_cpp():
    """``compute_reward`` must not re-scan a HIP candidate in Python mode.

    This is the actual defect: ``KoreEnv`` scanned correctly with the task's
    backend, found nothing, and left ``hack_reason`` unset -- and then the reward
    re-scanned the same source with the default language.
    """
    from kore.env.kore_env import Observation
    from kore.reward.reward import compute_reward

    obs = Observation(compiled=True, validation_passed=True,
                      snr_by_shape={"s": 99.0}, wall_by_shape={"s": 0.5},
                      baseline_by_shape={"s": 1.0})
    # Detected, no explicit language: this is the path every caller took.
    rr = compute_reward(obs, _HONEST_HIP, dtype="bf16")
    assert rr.tier != "hack", f"honest HIP kernel scored the hack tier: {rr.detail}"
    assert rr.correct
    # And an explicit language is honored.
    assert compute_reward(obs, _HONEST_HIP, dtype="bf16",
                          language="cpp").tier != "hack"


# --------------------------------------------------------------------------- #
# 2. the rollout-vs-training divergence diagnostic
# --------------------------------------------------------------------------- #
def test_mismatch_stat_was_unreachable_without_a_recorded_training_logp():
    """Field 7 is absent on a freshly built sample, so the stat must be null.

    Kept as a test rather than deleted: it documents that the diagnostic reports
    ``None`` honestly instead of a misleading 1.0 when the step has not recorded
    a training-side log-prob yet.
    """
    from kore.policy.grpo import _grpo_step_mismatch_stat

    # The real shape: (ret, gen_inputs, ref_logp, old_logp, n_tokens, sc_w, key).
    sample = [1.0, [("p", "g")], None, -0.5, 8, None, (0, 0)]
    assert len(sample) == 7
    assert _grpo_step_mismatch_stat([[sample]]) == (None, None)


def test_recording_the_training_logp_makes_the_diagnostic_report():
    from kore.policy.grpo import _grpo_step_mismatch_stat, _record_train_logp

    sample = [1.0, [("p", "g")], None, -0.5, 8, None, (0, 0)]
    _record_train_logp(sample, -0.5)  # on-policy: identical to old_logp
    mean, mx = _grpo_step_mismatch_stat([[sample]])
    assert mean is not None and mx is not None
    # ppo_epochs=1 recomputes old_logp on the weights about to be trained, so a
    # healthy step's sequence importance weight is exactly 1.0.
    assert abs(mean - 1.0) < 1e-9 and abs(mx - 1.0) < 1e-9


def test_the_diagnostic_sees_divergence_and_respects_the_cap():
    from kore.policy.grpo import _grpo_step_mismatch_stat, _record_train_logp

    drifted = [1.0, [("p", "g")], None, -0.5, 8, None, (0, 0)]
    _record_train_logp(drifted, 5.0)  # exp(5.5) is far past any cap
    mean, mx = _grpo_step_mismatch_stat([[drifted]], 2.0)
    assert mx is not None and mx <= 2.0 + 1e-9, (
        "the weight must be capped, so a single diverged sample cannot own the "
        "whole diagnostic"
    )
    assert mean is not None and mean > 1.0


def test_recording_tolerates_an_immutable_sample():
    """Unit-test fixtures pass tuples; the trainer must not raise on them."""
    from kore.policy.grpo import _record_train_logp

    _record_train_logp((1.0, None, None, -0.5, 8, None, (0, 0)), -0.5)


# --------------------------------------------------------------------------- #
# 3. the sharded step log
# --------------------------------------------------------------------------- #
def test_the_sharded_loop_logs_the_diagnostics_the_recipe_relies_on():
    """The production path is the sharded one, and it logged none of these.

    Read as text rather than by running a distributed step: this pins the
    contract cheaply, and a real 8-rank run is what proves it emits.
    """
    source = (REPO / "kore" / "policy" / "grpo.py").read_text()
    event = re.search(r'log\.event\(\s*"grpo_step_dist".*?\)\n', source, re.S)
    assert event, "could not find the sharded step-log event"
    body = event.group(0)
    for field in ("adv_absmean", "kl", "mismatch_mean", "mismatch_max",
                  "grad_norm", "lr"):
        assert f"{field}=" in body, (
            f"the sharded step log omits {field!r}; on the production path a "
            f"reader has no way to confirm the estimator, the anchor, or the "
            f"rollout/training agreement"
        )


def test_advantages_are_logged_from_the_values_the_step_used():
    """Not recomputed locally, which would be wrong under sharding.

    ``distributed_group_advantages`` builds each baseline from returns gathered
    across every rank, so re-deriving advantages from one rank's groups would
    report a number the trainer never saw.
    """
    source = (REPO / "kore" / "policy" / "grpo.py").read_text()
    at = source.index('"grpo_step_dist"')
    window = source[max(0, at - 2000):at]
    assert "for a, _ in local_terms" in window, (
        "adv_absmean must come from local_terms (the advantages actually used)"
    )
    assert "_grpo_step_adv_stats(keep" not in window, (
        "recomputing advantages from local groups misreports them under sharding"
    )
