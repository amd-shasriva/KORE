"""The measured speedup must be a speedup over something a practitioner runs.

KORE reports every generated task's speedup against that task's ``baseline_fn``.
Two ways that number stops meaning anything, both closed here and both guarded
against silent regression:

DEFECT 1 - the sequence / state-space families used the SAME eager
``for t in range(L)`` recurrence for the perf baseline as for the fp32 oracle.
At the declared L (2048 primary, up to 8192 in validation) that bar is a Python
interpreter loop dispatching thousands of tiny kernels, so any correct fused
kernel beats it by orders of magnitude - a measurement of the harness, not of the
kernel.  The oracle must stay the eager recurrence (it is the correctness
authority), so the invariant is asymmetric and is tested as such:

  * every ``baseline_fn`` is a VECTORIZED formulation - its dispatched-op count
    must not scale with the sequence length;
  * every ``ref_fn`` still IS the eager per-timestep recurrence (positive control:
    without it, "does not scale with L" could pass vacuously);
  * and the fast baseline still equals the oracle, so a faster baseline can never
    quietly become a wrong baseline.

DEFECT 2 - the compiler-fused bar for the ``fusion`` / ``gemm_fusion`` families
was opt-in and defaulted OFF, so every default-path evaluation graded a fused
Triton kernel against UNFUSED eager torch, measuring the absence of
``torch.compile``.  It is now the default, and dropping back to eager has to be a
deliberate, recorded act.

CPU-only: torch is used, no GPU or triton runtime is required.
"""

from __future__ import annotations

import os
import warnings

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from kore.tasks import _genops as G
from kore.tasks.breadth import seq as SQ
from kore.tasks.breadth import ssm_ext as S

# Below this ratio a baseline is vectorized; a per-timestep Python loop grows
# with L and lands near 4.0 when the length is quadrupled.
_SCALING_TOLERANCE = 2.5
_SHORT_L, _LONG_L = 64, 256


class _DispatchCounter(TorchDispatchMode):
    """Counts ATen dispatches, i.e. the kernel launches a GPU run would make."""

    def __init__(self) -> None:
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += 1
        return func(*args, **(kwargs or {}))


def _dispatches(fn, inputs) -> int:
    with _DispatchCounter() as counter:
        fn(*inputs)
    return counter.count


# --------------------------------------------------------------------------- #
# The op catalog under test: every sequence/SSM op of BOTH breadth engines, with
# a tiny CPU shape whose only free parameter is the sequence length.
# --------------------------------------------------------------------------- #
def _ssm_shape(op: str, length: int) -> dict:
    fam, cfg = S.OP_FAMILY[op], S.OP_CONFIG[op]
    if fam in ("gla", "gated_retention", "delta", "linattn", "hgrn2"):
        shape = {"B": 1, "H": 2, "L": length, "Dh": 4}
    elif fam == "retention":
        shape = {"B": 1, "H": cfg["H"], "L": length, "Dh": 4}
    elif fam == "mamba2_ssd":
        shape = {"B": 1, "L": length, "H": 2, "P": 3, "N": 4}
    elif fam == "selective":
        shape = {"B": 1, "L": length, "D": 5, "N": 4}
    elif fam == "rwkv":
        shape = {"B": 1, "L": length, "C": 5}
    elif fam == "s4d":
        shape = {"B": 1, "D": 3, "L": length, "N": 4}
    elif fam == "lru":
        shape = {"B": 1, "D": 3, "L": length}
    elif fam in ("conv_ssd", "conv_selective"):
        shape = {"B": 1, "L": length, "D": 4, "N": 4, "K": 4}
    else:                       # the last-dim scan primitives (+ hgrn / gilr)
        shape = {"B": 1, "D": 3, "L": length}
    if "chunk" in cfg:
        shape["chunk"] = cfg["chunk"]
    return shape


_SEQ_SHAPE = {
    "cumsum": {"B": 1, "D": 3},
    "cumprod": {"B": 1, "D": 3},
    "assoc_scan_segmented": {"B": 1, "D": 3},
    "selective_scan": {"B": 1, "D": 4, "N": 8},
    "ssd_chunk_scan": {"B": 1, "D": 4, "N": 8},
    "linear_attention": {"B": 1, "H": 2, "Dh": 4},
    "causal_conv1d": {"B": 1, "D": 3, "K": 4},
}


def _cases():
    """(label, make_reference, shape_at_length) for every sequence/SSM op."""
    for op in S.OPS:
        yield f"ssm_ext:{op}", (lambda o=op: S.make_reference(o, "fp32")), \
            (lambda length, o=op: _ssm_shape(o, length))
    for op in SQ.OPS:
        yield f"seq:{op}", (lambda o=op: SQ.make_reference(o, "fp32")), \
            (lambda length, o=op: dict(_SEQ_SHAPE[o], L=length))


_CASES = list(_cases())
_IDS = [label for label, _, _ in _CASES]


def test_the_catalog_under_test_is_the_whole_sequence_corpus():
    """94 generated tasks live behind these ops (40 ssm_ + 7 seq, x bf16/fp16)."""
    assert len(S.OPS) == 40 and len(SQ.OPS) == 7
    assert len(_CASES) == 47
    generated = 2 * len(_CASES)          # every op ships a bf16 and an fp16 task
    assert generated == 94


# --------------------------------------------------------------------------- #
# DEFECT 1 - the sequence baselines are no longer a Python interpreter loop
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,make_ref,shape_at", _CASES, ids=_IDS)
def test_baseline_does_not_dispatch_per_timestep(label, make_ref, shape_at):
    """Quadrupling L must not (anywhere near) quadruple the baseline's work.

    This is the structural form of the defect: an eager ``for t in range(L)``
    baseline issues O(L) kernel launches, which is what made a fused kernel look
    hundreds of times faster than "torch"."""
    ns = make_ref()
    short = _dispatches(ns["baseline_fn"],
                        ns["get_inputs"](shape_at(_SHORT_L), device="cpu", seed=0))
    long = _dispatches(ns["baseline_fn"],
                       ns["get_inputs"](shape_at(_LONG_L), device="cpu", seed=0))
    assert long < _SCALING_TOLERANCE * short, (
        f"{label}: baseline dispatches {short} ops at L={_SHORT_L} and {long} at "
        f"L={_LONG_L} ({long / short:.2f}x for a 4x longer sequence) - it is "
        f"stepping the recurrence in Python instead of a chunked/parallel scan")


@pytest.mark.parametrize("label,make_ref,shape_at", _CASES, ids=_IDS)
def test_oracle_is_still_the_eager_per_timestep_recurrence(label, make_ref, shape_at):
    """Positive control for the test above, and a guard on the oracle itself.

    ``ref_fn`` is the correctness authority behind every stored correctness label,
    so it must NOT be swapped for the fast formulation. Ops whose oracle is a
    single native torch call (cumsum, cumprod, causal_conv1d) have nothing to
    step and are exempt."""
    ns = make_ref()
    short = _dispatches(ns["ref_fn"],
                        ns["get_inputs"](shape_at(_SHORT_L), device="cpu", seed=0))
    if short <= 8:                        # native one-shot oracle, no scan to step
        return
    long = _dispatches(ns["ref_fn"],
                       ns["get_inputs"](shape_at(_LONG_L), device="cpu", seed=0))
    assert long > 3.0 * short, (
        f"{label}: the fp32 oracle no longer scales with L ({short} -> {long} "
        f"ops); it must stay the exact eager recurrence")


@pytest.mark.parametrize("label,make_ref,shape_at", _CASES, ids=_IDS)
def test_fast_baseline_still_equals_the_oracle(label, make_ref, shape_at):
    """A faster baseline must not be a wrong baseline.

    L = 71 spans several chunks at every configured chunk size and divides none of
    them, so the chunked baselines' tail padding, inter-chunk carry and boundary
    scan are all live."""
    ns = make_ref()
    inputs = ns["get_inputs"](shape_at(71), device="cpu", seed=2)
    out, ref = ns["baseline_fn"](*inputs), ns["ref_fn"](*inputs)
    assert out.shape == ref.shape, label
    assert torch.allclose(out.double(), ref.double(), atol=2e-4, rtol=2e-3), (
        f"{label}: baseline deviates from the oracle by "
        f"{(out.double() - ref.double()).abs().max().item():.3e}")


def test_no_sequence_baseline_aliases_the_oracle():
    """The two must stay distinct callables: the oracle is exact-and-slow, the
    baseline is the production path."""
    for label, make_ref, _ in _CASES:
        ns = make_ref()
        assert ns["baseline_fn"] is not ns["ref_fn"], label


# --------------------------------------------------------------------------- #
# DEFECT 2 - the compiler-fused bar is the DEFAULT, opting out is deliberate
# --------------------------------------------------------------------------- #
_FUSION_OP, _GEMM_OP = "mul_tanh", "gemm_bias_relu"


def test_compile_baseline_defaults_to_the_fused_bar(monkeypatch):
    """Unset (and empty) means the honest bar, not the inflated one."""
    for value in (None, "", "   "):
        if value is None:
            monkeypatch.delenv(G.COMPILE_BASELINE_ENV, raising=False)
        else:
            monkeypatch.setenv(G.COMPILE_BASELINE_ENV, value)
        status = G.compile_baseline_status()
        assert status["enabled"] is True
        assert status["source"] == "default"
        assert G._vendor_baseline_kind(_FUSION_OP, "fusion", "bf16") == "torch_compile"


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_explicit_truthy_values_keep_the_fused_bar(monkeypatch, value):
    monkeypatch.setenv(G.COMPILE_BASELINE_ENV, value)
    assert G.compile_baseline_status() == {
        "enabled": True, "declared": value, "source": "env"}


@pytest.mark.parametrize("value", ["0", "false", "NO", "off"])
def test_only_an_explicit_falsey_value_opts_out(monkeypatch, value):
    monkeypatch.setenv(G.COMPILE_BASELINE_ENV, value)
    status = G.compile_baseline_status()
    assert status["enabled"] is False
    assert status["source"] == "env_opt_out"
    assert G._vendor_baseline_kind(_FUSION_OP, "fusion", "bf16") == "eager"


@pytest.mark.parametrize("value", ["maybe", "2", "of"])
def test_an_unrecognized_value_fails_closed_onto_the_honest_bar(monkeypatch, value):
    """A typo must not silently restore the inflated bar."""
    monkeypatch.setenv(G.COMPILE_BASELINE_ENV, value)
    status = G.compile_baseline_status()
    assert status["enabled"] is True
    assert status["source"] == "unrecognized_value"


def test_opt_out_is_announced_and_recorded_on_the_reference(monkeypatch):
    """Disabling the fused bar is loud (a warning) and durable (it rides on the
    reference namespace next to the baseline it downgraded)."""
    monkeypatch.setenv(G.COMPILE_BASELINE_ENV, "0")
    monkeypatch.setattr(G, "_OPT_OUT_ANNOUNCED", set())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert G._compile_baseline_enabled() is False
        G._compile_baseline_enabled()                       # announced ONCE
    assert len(caught) == 1
    assert issubclass(caught[0].category, RuntimeWarning)
    assert G.COMPILE_BASELINE_ENV in str(caught[0].message)

    for op, family in ((_FUSION_OP, "fusion"), (_GEMM_OP, "gemm_fusion")):
        ns = G.make_reference(op, family, "fp32")
        assert ns["baseline_compile_opt_out"] is True

    monkeypatch.setenv(G.COMPILE_BASELINE_ENV, "1")
    for op, family in ((_FUSION_OP, "fusion"), (_GEMM_OP, "gemm_fusion")):
        assert G.make_reference(op, family, "fp32")["baseline_compile_opt_out"] is False


def test_non_fusion_families_are_never_marked_as_an_opt_out(monkeypatch):
    """Only fusion/gemm_fusion have a compiler-fused bar to lose."""
    monkeypatch.setenv(G.COMPILE_BASELINE_ENV, "0")
    for op, family in (("relu", "unary"), ("add", "binary"), ("row_sum", "reduce")):
        ns = G.make_reference(op, family, "fp32")
        assert ns["baseline_compile_opt_out"] is False
        assert ns["baseline_kind"] == "eager"


def test_config_import_publishes_the_default_into_the_environment():
    """The default lives in the ENVIRONMENT, not only in Python, because the
    verifier benches in a subprocess that inherits os.environ and because the
    provenance recorders read the raw variable - a Python-only default would let
    a run's recorded identity disagree with the bar it actually used."""
    import kore.config as config

    assert os.environ.get(G.COMPILE_BASELINE_ENV) is not None
    assert G.compile_baseline_status()["enabled"] is True
    assert config.CONFIG.compile_baseline is True


def test_an_operator_opt_out_survives_config_import(monkeypatch):
    """``setdefault`` publishes a default; it never overrides an explicit choice."""
    import importlib

    import kore.config as config

    monkeypatch.setenv(G.COMPILE_BASELINE_ENV, "0")
    importlib.reload(config)
    try:
        assert os.environ[G.COMPILE_BASELINE_ENV] == "0"
        assert config.KoreConfig().compile_baseline is False
    finally:
        monkeypatch.setenv(G.COMPILE_BASELINE_ENV, "1")
        importlib.reload(config)


def test_datagen_rigor_still_names_the_compile_baseline():
    """The datagen rigor set is what production exports; it must keep agreeing
    with the new default rather than silently diverging from it."""
    from kore.data.verify_rigor import RIGOR_ENV

    assert RIGOR_ENV[G.COMPILE_BASELINE_ENV] == "1"


def test_the_fused_bar_is_applied_exactly_once(monkeypatch):
    """Regression: the hipBLASLt epilogue path used to hand ``torch.compile`` the
    already-gated ``baseline_fn`` instead of the pure eager epilogue. Dynamo
    traces whatever it is given, so it tried to trace the ``os.environ`` lookup
    inside and every gemm_fusion bench died the moment the fused bar was on -
    which making it the default would have triggered everywhere."""
    monkeypatch.setenv(G.COMPILE_BASELINE_ENV, "1")
    monkeypatch.setenv("KORE_USE_VENDOR_BASELINE", "1")
    ns = G.make_reference(_GEMM_OP, "gemm_fusion", "fp32")

    wrapped: list = []

    def _record(fn, key):
        wrapped.append((fn, key))
        return fn

    monkeypatch.setattr(G, "_fused_baseline", _record)
    a, b, bias = torch.randn(4, 8), torch.randn(8, 6), torch.randn(6)
    out = ns["baseline_fn"](a, b, bias)

    assert len(wrapped) == 1, (
        f"the fused bar was applied {len(wrapped)} times "
        f"({[k for _, k in wrapped]}); compile must wrap the pure epilogue once")
    assert out.shape == (4, 6)
    assert torch.allclose(out, torch.relu(a @ b + bias), atol=1e-5)
