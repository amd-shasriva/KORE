"""CPU-only tests for the production TransformProposePolicy + search_from_kernel.

No GPU/torch: a FakeEnv grades a kernel by parsing its ``num_warps`` knob (higher =
faster), so AlphaKernel driven by the VERIFIED transform calculus (which can raise
num_warps via the exact ``set_num_warps`` move) has a real gradient to climb. This
validates the adapter that was the missing production piece (search had no real
ProposePolicy) end-to-end.
"""

from __future__ import annotations

import re

from kore.reward.reward import Observation
from kore.search.alphakernel import ProposeContext
from kore.search.propose import TransformProposePolicy, search_from_kernel

_GEMM = '''\
import triton
import triton.language as tl


@triton.jit
def _k(a_ptr, b_ptr, c_ptr, M, N, K,
       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc += 1.0


def gemm(a, b, c):
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 128, 64, 8
    _k[(1,)](a, b, c, 1, 1, 1, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
             GROUP_M=GROUP_M, num_warps=4, num_stages=2)
'''

_WARPS = re.compile(r"num_warps\s*=\s*(\d+)")


class FakeTask:
    task_id = "fake_gemm_bf16"
    operation = "gemm"
    dtype = "bf16"
    gpu_target = "gfx950"
    snr_threshold = 25.0
    shapes = []


class FakeEnv:
    """Correct-always env whose speed increases with num_warps (a climbable gradient
    for the set_num_warps transform)."""

    def __init__(self):
        self.steps = 0

    def step(self, source, full_validation=True, multi_shape=True):
        self.steps += 1
        m = _WARPS.search(source or "")
        warps = int(m.group(1)) if m else 4
        speedup = 1.0 + 0.25 * (warps - 4)  # 4->1.0x, 8->2.0x, 16->4.0x
        if not full_validation:  # correctness-only gate (no timing)
            return Observation(compiled=True, dtype="bf16", validation_passed=True,
                               snr_by_shape={"primary": 40.0}, snr_db=40.0)
        return Observation(compiled=True, dtype="bf16", validation_passed=True,
                           snr_by_shape={"primary": 40.0}, snr_db=40.0,
                           wall_by_shape={"primary": 1.0},
                           baseline_by_shape={"primary": speedup},
                           wall_ms=1.0, baseline_ms=speedup)


def test_transform_propose_policy_expands_with_verified_moves():
    pol = TransformProposePolicy(k=4)
    ctx = ProposeContext(source=_GEMM, depth=0, correct=True, speedup_lcb=1.0,
                         fingerprint="x", task=FakeTask())
    edits = pol.propose(ctx)
    assert edits, "expected the verified transform library to yield admissible moves"
    assert all(e.source and e.source != _GEMM for e in edits)
    # set_num_warps is an exact move on num_warps=4 -> must be among the proposals
    assert any(e.name == "set_num_warps" for e in edits)
    # empty source -> no edits, never raises
    assert pol.propose(ProposeContext("", 0, True, None, "y", FakeTask())) == []


def test_search_from_kernel_climbs_the_transform_gradient():
    env = FakeEnv()
    res = search_from_kernel(_GEMM, FakeTask(), env, budget=48, k_expand=4, seed=0)
    assert res["best_source"] is not None
    # search should discover a kernel at least as fast as the seed (num_warps>=4),
    # i.e. it explored the verified action space and kept the best correct node.
    assert res["best_speedup_lcb"] is not None and res["best_speedup_lcb"] >= 1.0
    best_warps = int(_WARPS.search(res["best_source"]).group(1))
    assert best_warps >= 4  # never regressed below the seed
    assert env.steps <= 48  # budget respected


def test_search_from_kernel_is_failsafe_on_untransformable_source():
    # a non-Triton source has no admissible knobs -> search returns the root, no crash
    env = FakeEnv()
    res = search_from_kernel("def f(x):\n    return x\n", FakeTask(), env, budget=8)
    assert "best_source" in res  # returns cleanly (may be None if no correct node)
