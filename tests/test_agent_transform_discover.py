"""CPU-only tests for the ``transform_discover`` lever on the agentic tool layer.

No GPU, no torch/vllm. The ToolExecutor's verified-transform tools
(``list_transforms`` / ``apply_transform``) normally expose the curated
:data:`kore.transform.library.LIBRARY`. With ``KORE_TRANSFORM_DISCOVER=1`` set
(env-gated exactly like ``KORE_ROOFLINE_GATE`` -- the ToolExecutor has no config
handle) they instead expose the curated library PLUS the self-extending library's
SNR-gated discovered proposals, seeded from the current kernel.

Verifies:
  * default (env unset) is byte-identical to the curated tools -- no discovered
    move ever leaks into ``list_transforms``;
  * with the lever ON, ``list_transforms`` is a STRICT SUPERSET (adds ``disc:``
    proposals) and ``apply_transform`` can apply a discovered rewrite end-to-end;
  * the whole path is fail-safe (never raises into the tool loop).

Discovered rewrites are conservatively-typed PROPOSALS, not proofs: correctness is
enforced downstream by the env SNR oracle when the model build/test/benches the
rewritten kernel.
"""

from __future__ import annotations

from kore.agent.tools import ToolExecutor
from kore.transform import ErrorBudget, admissible_actions
from kore.transform.discover import DISCOVERED_PREFIX

# A minimal Triton GEMM with tunable launch kwargs (num_warps=4/num_stages=2) + a
# tuple BLOCK defn, so curated knob moves AND discovered sweeps are admissible.
_XFORM_GEMM = '''\
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


class FakeTask:
    task_id = "fake_gemm_bf16"
    operation = "gemm"
    dtype = "bf16"
    gpu_target = "gfx950"


class FakeEnv:
    """The transform tools are pure/no-GPU (source rewrites only) and never touch
    the env, so a trivial stub suffices."""

    def step(self, source, full_validation=True, multi_shape=True):  # pragma: no cover
        raise AssertionError("transform tools must not call env.step")


def _curated_action_names():
    return {a.name for a in admissible_actions(
        _XFORM_GEMM, ErrorBudget.for_op("gemm", "bf16"))}


# --------------------------------------------------------------------------- #
# Default (lever OFF) is byte-identical to the curated transform tools
# --------------------------------------------------------------------------- #
def test_list_transforms_off_is_curated_default(monkeypatch):
    monkeypatch.delenv("KORE_TRANSFORM_DISCOVER", raising=False)
    ex = ToolExecutor(FakeEnv(), FakeTask(), seed_src=_XFORM_GEMM)
    listed = ex.dispatch({"name": "list_transforms", "arguments": {}}, turn=0)
    assert listed["ok"] is True
    names = {a["name"] for a in listed["actions"]}
    # exactly the curated action space -- no discovered proposal leaks in.
    assert names == _curated_action_names()
    assert not any(n.startswith(DISCOVERED_PREFIX) for n in names)


def test_apply_transform_off_rejects_discovered_name(monkeypatch):
    # With the lever OFF, a discovered transform name is unknown to the curated
    # library -> rejected, source UNCHANGED, never raises.
    monkeypatch.delenv("KORE_TRANSFORM_DISCOVER", raising=False)
    ex = ToolExecutor(FakeEnv(), FakeTask(), seed_src=_XFORM_GEMM)
    r = ex.dispatch({"name": "apply_transform", "arguments": {
        "name": f"{DISCOVERED_PREFIX}set_num_warps[value=16]", "params": {}}}, turn=0)
    assert r["ok"] is False
    assert r["kernel_src"] == _XFORM_GEMM  # unchanged
    assert r["rejected"]


# --------------------------------------------------------------------------- #
# Lever ON: strict superset + apply a discovered rewrite end-to-end
# --------------------------------------------------------------------------- #
def test_list_transforms_on_is_strict_superset(monkeypatch):
    monkeypatch.setenv("KORE_TRANSFORM_DISCOVER", "1")
    ex = ToolExecutor(FakeEnv(), FakeTask(), seed_src=_XFORM_GEMM)
    listed = ex.dispatch({"name": "list_transforms", "arguments": {}}, turn=0)
    assert listed["ok"] is True
    names = {a["name"] for a in listed["actions"]}
    curated = _curated_action_names()
    # curated moves preserved + discovered proposals added on top.
    assert curated <= names
    assert names - curated
    assert any(n.startswith(DISCOVERED_PREFIX) for n in names)
    # every discovered action is conservatively typed approx.
    disc = [a for a in listed["actions"] if a["name"].startswith(DISCOVERED_PREFIX)]
    assert disc and all(a["relation"] == "approx" for a in disc)


def test_apply_transform_on_applies_discovered_rewrite(monkeypatch):
    monkeypatch.setenv("KORE_TRANSFORM_DISCOVER", "1")
    ex = ToolExecutor(FakeEnv(), FakeTask(), seed_src=_XFORM_GEMM)
    # a num_warps=16 sweep is OUTSIDE the curated (4,8) grid -> only reachable via
    # discovery. It is listed ...
    listed = ex.dispatch({"name": "list_transforms", "arguments": {}}, turn=0)
    disc_name = f"{DISCOVERED_PREFIX}set_num_warps[value=16]"
    assert disc_name in {a["name"] for a in listed["actions"]}
    # ... and applies end-to-end, rewriting num_warps 4 -> 16, budget-accounted.
    applied = ex.dispatch({"name": "apply_transform", "arguments": {
        "name": disc_name, "params": {}}}, turn=0)
    assert applied["ok"] is True
    assert applied["applied"] and not applied["rejected"]
    assert "num_warps=16" in applied["kernel_src"]
    assert applied["kernel_src"] != _XFORM_GEMM


def test_apply_any_listed_discovered_move_is_applicable(monkeypatch):
    # Generic guarantee: EVERY discovered move the tool advertises actually applies
    # (the list + apply contract holds for the whole discovered action space). A
    # FRESH executor per move gives each a fresh epsilon budget -- within ONE episode
    # the budget is shared, so a chain of approx moves correctly drains it (that
    # exhaustion path is covered by test_apply_transform_drains_shared_budget).
    monkeypatch.setenv("KORE_TRANSFORM_DISCOVER", "1")
    listed = ToolExecutor(FakeEnv(), FakeTask(), seed_src=_XFORM_GEMM).dispatch(
        {"name": "list_transforms", "arguments": {}}, turn=0)
    disc = [a for a in listed["actions"] if a["name"].startswith(DISCOVERED_PREFIX)]
    assert disc
    for a in disc:
        ex = ToolExecutor(FakeEnv(), FakeTask(), seed_src=_XFORM_GEMM)
        r = ex.dispatch({"name": "apply_transform", "arguments": {
            "name": a["name"], "params": {}}}, turn=0)
        assert r["ok"] is True and r["applied"] and not r["rejected"], a["name"]
        assert r["kernel_src"] != _XFORM_GEMM


def test_apply_transform_drains_shared_budget(monkeypatch):
    # The episode-shared epsilon budget accumulates across apply_transform calls, so
    # repeatedly spending an approx (discovered) move eventually exhausts it and the
    # calculus REFUSES further approx moves -- the discovered action space is still
    # gated by the numerical contract, never unbounded.
    monkeypatch.setenv("KORE_TRANSFORM_DISCOVER", "1")
    ex = ToolExecutor(FakeEnv(), FakeTask(), seed_src=_XFORM_GEMM)
    disc_name = f"{DISCOVERED_PREFIX}set_num_warps[value=16]"
    outcomes = [ex.dispatch({"name": "apply_transform", "arguments": {
        "name": disc_name, "params": {}}}, turn=0)["ok"] for _ in range(200)]
    assert outcomes[0] is True         # first approx move fits the budget
    assert outcomes[-1] is False       # ... but the shared budget is finite


# --------------------------------------------------------------------------- #
# Fail-safe
# --------------------------------------------------------------------------- #
def test_discover_on_is_failsafe_on_non_kernel_source(monkeypatch):
    # Lever ON but the working kernel is not a Triton source: discovery finds
    # nothing to add and the tool degrades gracefully (no crash, no exception).
    monkeypatch.setenv("KORE_TRANSFORM_DISCOVER", "1")
    ex = ToolExecutor(FakeEnv(), FakeTask(), seed_src="def f(x):\n    return x\n")
    listed = ex.dispatch({"name": "list_transforms", "arguments": {}}, turn=0)
    assert listed["ok"] is True  # runs cleanly
    assert not any(a["name"].startswith(DISCOVERED_PREFIX) for a in listed["actions"])


def test_transform_library_helper_gating(monkeypatch):
    # Direct unit check of the env gate + fail-safe caching on the executor.
    ex = ToolExecutor(FakeEnv(), FakeTask(), seed_src=_XFORM_GEMM)
    monkeypatch.delenv("KORE_TRANSFORM_DISCOVER", raising=False)
    assert ex._transform_library(_XFORM_GEMM) is None  # OFF -> curated (None)
    monkeypatch.setenv("KORE_TRANSFORM_DISCOVER", "1")
    lib = ex._transform_library(_XFORM_GEMM)
    from kore.transform import LIBRARY
    assert lib is not None and len(lib) > len(LIBRARY)  # curated + discovered
    assert list(lib)[:len(LIBRARY)] == list(LIBRARY)    # curated prefix preserved
    # memoized for the same source (no rebuild).
    assert ex._transform_library(_XFORM_GEMM) is lib
