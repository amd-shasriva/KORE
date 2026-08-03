"""CPU-only tests for instruction-residual (chat-vector) transfer.

The transfer is worth about 20 hours of SFT compute, so the arithmetic has to
be right for reasons no smoke test would catch: a sign error still produces a
plausible-looking model, and a bf16 round-trip still loads. The identity test
below is the load-bearing one -- applying the residual to the base must return
the instruct checkpoint exactly.
"""

from __future__ import annotations

import pytest

from kore.policy import residual as res


def _sd(torch, **kw):
    return {k: v for k, v in kw.items()}


def _triplet(torch, dtype=None):
    dtype = dtype or torch.float32
    g = torch.Generator().manual_seed(0)
    base = {
        "w": torch.randn(4, 3, generator=g).to(dtype),
        "b": torch.randn(4, generator=g).to(dtype),
        "n": torch.tensor([7, 8], dtype=torch.int64),
    }
    instruct = {
        "w": base["w"] + torch.randn(4, 3, generator=g).to(dtype) * 0.1,
        "b": base["b"] + torch.randn(4, generator=g).to(dtype) * 0.1,
        "n": torch.tensor([7, 8], dtype=torch.int64),
    }
    target = {
        "w": base["w"] + torch.randn(4, 3, generator=g).to(dtype) * 0.2,
        "b": base["b"] + torch.randn(4, generator=g).to(dtype) * 0.2,
        "n": torch.tensor([9, 9], dtype=torch.int64),
    }
    return base, instruct, target


def test_module_imports_without_heavy_stack():
    # Mirrors the contract in tests/test_policy.py: policy submodules must be
    # importable without torch, so torch stays inside the functions.
    import importlib

    assert importlib.import_module("kore.policy.residual") is res


def test_identity_applying_residual_to_base_recovers_instruct():
    torch = pytest.importorskip("torch")
    base, instruct, _ = _triplet(torch)
    report = res.verify_identity(base, instruct)
    assert report["exact_fraction"] == 1.0, report
    assert report["max_abs_diff"] == 0.0, report


def test_identity_holds_in_bfloat16():
    # The real checkpoints are bf16; FP32 intermediate math must not drift when
    # cast back, or every weight in the model picks up rounding error.
    torch = pytest.importorskip("torch")
    base, instruct, _ = _triplet(torch, dtype=torch.bfloat16)
    report = res.verify_identity(base, instruct)
    assert report["exact_fraction"] == 1.0, report


def test_result_equals_instruct_plus_domain_delta():
    # theta_out = theta_instruct + (theta_target - theta_base): the rearranged
    # form is what makes the transfer intuitive, so assert it directly.
    torch = pytest.importorskip("torch")
    base, instruct, target = _triplet(torch)
    out = res.apply_residual(base, instruct, target)
    for key in ("w", "b"):
        expected = instruct[key] + (target[key] - base[key])
        assert torch.allclose(out[key], expected, atol=1e-6), key


def test_scale_zero_leaves_target_untouched():
    torch = pytest.importorskip("torch")
    base, instruct, target = _triplet(torch)
    out = res.apply_residual(base, instruct, target, scale=0.0)
    for key in ("w", "b"):
        assert torch.allclose(out[key], target[key], atol=1e-7), key


def test_integer_buffers_come_from_target_not_the_delta():
    torch = pytest.importorskip("torch")
    base, instruct, target = _triplet(torch)
    out = res.apply_residual(base, instruct, target)
    assert torch.equal(out["n"], target["n"])


def test_rejects_shape_mismatch():
    torch = pytest.importorskip("torch")
    base, instruct, target = _triplet(torch)
    target["w"] = torch.randn(5, 3)
    with pytest.raises(res.ResidualError, match="shape mismatch"):
        res.apply_residual(base, instruct, target)


def test_rejects_key_mismatch():
    torch = pytest.importorskip("torch")
    base, instruct, target = _triplet(torch)
    del instruct["b"]
    with pytest.raises(res.ResidualError, match="key mismatch"):
        res.apply_residual(base, instruct, target)


def test_rejects_non_finite_scale():
    torch = pytest.importorskip("torch")
    base, instruct, target = _triplet(torch)
    with pytest.raises(res.ResidualError, match="finite"):
        res.apply_residual(base, instruct, target, scale=float("nan"))


def test_report_flags_a_mispaired_checkpoint():
    # Passing the same checkpoint twice yields an all-zero delta, which would
    # silently produce a no-op transfer rather than an error.
    torch = pytest.importorskip("torch")
    base, _, _ = _triplet(torch)
    report = res.residual_report(base, base)
    assert report["n_zero_delta"] == report["n_float"]
    assert report["rel_delta_max"] == 0.0


def test_report_shows_a_small_nonzero_delta_for_a_real_pair():
    torch = pytest.importorskip("torch")
    base, instruct, _ = _triplet(torch)
    report = res.residual_report(base, instruct)
    assert report["n_zero_delta"] == 0
    assert 0.0 < report["rel_delta_median"] < 1.0
