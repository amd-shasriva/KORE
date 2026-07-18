"""CPU-only tests for the self-extending transform library wired into search.

No GPU/torch. Verifies the ``transform_discover`` lever on the AlphaKernel search
action space (``TransformProposePolicy`` / ``search_from_kernel``):

  * ``discover=True`` exposes a STRICT SUPERSET of the curated action space -- the
    curated moves are unchanged (byte-identical prefix) and SNR-gated discovered
    proposals are added on top;
  * a discovered rewrite APPLIES end-to-end through the policy (the enumerated
    action really rewrites the seed);
  * ``discover=False`` / ``library=None`` (the default) is byte-identical to the
    curated action space -- no discovered move ever leaks in;
  * the whole path is fail-safe (any discovery error -> curated library, no raise).

Discovered transforms are conservatively-typed PROPOSALS, not proofs: correctness
is enforced downstream by the env SNR oracle. These tests assert the action-space
plumbing (enumeration + apply), never that a rewrite is verified.
"""

from __future__ import annotations

import re

from kore.reward.reward import Observation
from kore.search.alphakernel import ProposeContext
from kore.search.propose import (
    TransformProposePolicy,
    _resolve_search_library,
    search_from_kernel,
)
from kore.transform import ErrorBudget, admissible_actions
from kore.transform.discover import DISCOVERED_PREFIX

# A minimal Triton GEMM with tunable knobs (num_warps/num_stages + tuple BLOCK defn)
# so both curated moves (e.g. set_num_warps) and discovered sweeps are admissible.
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
    """Correct-always env whose speed increases with num_warps (a climbable
    gradient for the set_num_warps family of moves)."""

    def __init__(self):
        self.steps = 0

    def step(self, source, full_validation=True, multi_shape=True):
        self.steps += 1
        m = _WARPS.search(source or "")
        warps = int(m.group(1)) if m else 4
        speedup = 1.0 + 0.25 * (warps - 4)  # 4->1.0x, 8->2.0x, 16->4.0x
        if not full_validation:
            return Observation(compiled=True, dtype="bf16", validation_passed=True,
                               snr_by_shape={"primary": 40.0}, snr_db=40.0)
        return Observation(compiled=True, dtype="bf16", validation_passed=True,
                           snr_by_shape={"primary": 40.0}, snr_db=40.0,
                           wall_by_shape={"primary": 1.0},
                           baseline_by_shape={"primary": speedup},
                           wall_ms=1.0, baseline_ms=speedup)


def _ctx(src=_GEMM):
    return ProposeContext(source=src, depth=0, correct=True, speedup_lcb=1.0,
                          fingerprint="x", task=FakeTask())


def _edit_names(policy):
    return {e.name for e in policy.propose(_ctx())}


# --------------------------------------------------------------------------- #
# discover=True exposes a STRICT SUPERSET of the curated action space
# --------------------------------------------------------------------------- #
def test_discover_true_is_strict_superset_of_curated_actions():
    off = _edit_names(TransformProposePolicy(k=100, discover=False))
    on = _edit_names(TransformProposePolicy(k=100, discover=True))
    assert off, "curated policy should yield admissible moves on the GEMM"
    # curated moves are all still present (superset) ...
    assert off <= on
    # ... and discovery strictly widens the action space with disc: proposals.
    assert on - off
    assert any(n.startswith(DISCOVERED_PREFIX) for n in on)
    # the curated (default-off) set never contains a discovered proposal.
    assert not any(n.startswith(DISCOVERED_PREFIX) for n in off)


def test_discover_true_matches_extended_library_action_space():
    # The policy's discovered action space equals enumerating the curated+discovered
    # library directly (the policy just reuses extend_library under the hood).
    ext = _resolve_search_library(_GEMM, discover=True, library=None)
    assert ext is not None
    lib_disc = {a.name for a in admissible_actions(_GEMM, ErrorBudget.for_op("gemm", "bf16"), ext)
                if a.name.startswith(DISCOVERED_PREFIX)}
    on = _edit_names(TransformProposePolicy(k=100, discover=True))
    assert lib_disc, "expected discovered actions once discovery is on"
    assert lib_disc <= on  # every enumerated discovered move is a proposable edit


# --------------------------------------------------------------------------- #
# A discovered rewrite APPLIES end-to-end through the policy
# --------------------------------------------------------------------------- #
def test_discover_applies_a_discovered_transform_end_to_end():
    pol = TransformProposePolicy(k=100, discover=True)
    edits = pol.propose(_ctx())
    disc = [e for e in edits if e.name.startswith(DISCOVERED_PREFIX)]
    assert disc, "expected at least one applied discovered rewrite"
    # every discovered edit really rewrote the seed into a distinct kernel ...
    assert all(e.source and e.source != _GEMM for e in disc)
    assert all(e.meta.get("relation") == "approx" for e in disc)  # conservatively typed
    # ... and the concrete num_warps=16 sweep (outside the curated (4,8) grid) fired.
    warps16 = [e for e in disc if e.name == f"{DISCOVERED_PREFIX}set_num_warps[value=16]"]
    assert warps16 and "num_warps=16" in warps16[0].source


# --------------------------------------------------------------------------- #
# default-OFF is byte-identical to the curated action space
# --------------------------------------------------------------------------- #
def test_default_off_policy_is_byte_identical_to_curated():
    # The default policy (no discover, no library) reproduces the curated action
    # space EXACTLY: same edit names AND same rewritten sources as an explicit
    # discover=False policy, with zero discovered proposals.
    default = TransformProposePolicy(k=100)
    explicit_off = TransformProposePolicy(k=100, discover=False)
    d = {(e.name, e.source) for e in default.propose(_ctx())}
    x = {(e.name, e.source) for e in explicit_off.propose(_ctx())}
    assert d == x
    assert d and not any(n.startswith(DISCOVERED_PREFIX) for n, _ in d)
    # and the default matches enumerating the curated LIBRARY directly (library=None).
    curated = {a.name for a in admissible_actions(_GEMM, ErrorBudget.for_op("gemm", "bf16"))}
    assert {n for n, _ in d} == curated


def test_explicit_library_overrides_discover_flag():
    # An explicit library ALWAYS wins: even with discover=True, passing the curated
    # library (via library=[...]) yields no discovered proposals.
    from kore.transform import LIBRARY
    pol = TransformProposePolicy(k=100, discover=True, library=list(LIBRARY))
    names = {e.name for e in pol.propose(_ctx())}
    assert names and not any(n.startswith(DISCOVERED_PREFIX) for n in names)


# --------------------------------------------------------------------------- #
# search_from_kernel: the orchestrator-facing wiring
# --------------------------------------------------------------------------- #
def test_resolve_search_library_precedence_and_failsafe():
    # default: curated (None)
    assert _resolve_search_library(_GEMM, discover=False, library=None) is None
    # explicit library wins over discover
    sentinel = ["not-a-real-library"]
    assert _resolve_search_library(_GEMM, discover=True, library=sentinel) is sentinel
    # discover -> curated+discovered superset of the curated LIBRARY
    from kore.transform import LIBRARY
    ext = _resolve_search_library(_GEMM, discover=True, library=None)
    assert ext is not None and len(ext) > len(LIBRARY)
    assert list(ext)[:len(LIBRARY)] == list(LIBRARY)  # curated prefix preserved


def test_search_from_kernel_discover_off_is_curated_default():
    # discover defaults to False -> resolver returns the curated library (None).
    assert _resolve_search_library(_GEMM, discover=False, library=None) is None
    env = FakeEnv()
    res = search_from_kernel(_GEMM, FakeTask(), env, budget=32, k_expand=4, seed=0)
    assert res["best_source"] is not None
    assert res["best_speedup_lcb"] is not None and res["best_speedup_lcb"] >= 1.0


def test_search_from_kernel_discover_true_runs_and_climbs():
    env = FakeEnv()
    res = search_from_kernel(_GEMM, FakeTask(), env, budget=64, k_expand=6,
                             discover=True, seed=0)
    assert res["best_source"] is not None
    # the discovered num_warps sweeps reach higher warp counts than the curated
    # (4,8) grid, so the search can only do at least as well as the seed.
    assert res["best_speedup_lcb"] is not None and res["best_speedup_lcb"] >= 1.0
    best_warps = int(_WARPS.search(res["best_source"]).group(1))
    assert best_warps >= 4  # never regressed below the seed
    assert env.steps <= 64  # budget respected


def test_search_from_kernel_discover_is_failsafe_on_untransformable_source():
    # A non-Triton source: discovery finds nothing extra, search returns cleanly.
    env = FakeEnv()
    res = search_from_kernel("def f(x):\n    return x\n", FakeTask(), env,
                             budget=8, discover=True)
    assert "best_source" in res  # no crash; may be None if no correct node
