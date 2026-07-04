"""CPU-only tests for the vendor-baselined op authoring engine."""

from __future__ import annotations

import ast

import pytest

from kore.tasks import vendor_ops as V


@pytest.mark.parametrize("op", V.VENDOR_OPS)
def test_vendor_seed_parses_and_defines_entry(op):
    dtype = V.vendor_op_dtypes(op)[0]
    src = V.vendor_seed_source(op, dtype)
    tree = ast.parse(src)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert op in funcs


@pytest.mark.parametrize("op", V.VENDOR_OPS)
def test_vendor_reference_namespace(op):
    dtype = V.vendor_op_dtypes(op)[0]
    ns = V.make_vendor_reference(op, dtype)
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity", "entry_name"):
        assert k in ns
    assert ns["entry_name"] == op
    assert ns["arity"] in (1, 2, 3, 4)


def test_vendor_op_dtypes_override():
    # fp8 GEMM is fp8-only; norm/act ops use the default bf16/fp16 sweep.
    assert V.vendor_op_dtypes("gemm_a8w8") == ("fp8",)
    assert V.vendor_op_dtypes("softmax") == V.VENDOR_DTYPES
    assert V.vendor_op_dtypes("rmsnorm") == V.VENDOR_DTYPES


def test_vendor_softmax_and_gemm_oracle_cpu():
    """softmax + fp8-gemm oracle sanity on CPU vs a direct torch compute."""
    import torch

    ns = V.make_vendor_reference("softmax", "fp32")
    (x,) = ns["get_inputs"]({"M": 4, "N": 32}, device="cpu", seed=0)
    y = ns["ref_fn"](x)
    assert torch.allclose(y, torch.softmax(x.float(), dim=-1).to(x.dtype), atol=1e-5)
    assert torch.allclose(y.sum(-1), torch.ones(4), atol=1e-4)

    ns2 = V.make_vendor_reference("gemm_a8w8", "fp8")
    xq, wq, xs, ws = ns2["get_inputs"]({"M": 8, "N": 16, "K": 32}, device="cpu", seed=0)
    out = ns2["ref_fn"](xq, wq, xs, ws)
    assert out.shape == (8, 16) and out.dtype == torch.bfloat16
    # oracle == explicit dequantized matmul
    exp = (xq.float() * xs.float()) @ (wq.float() * ws.float().reshape(-1, 1)).t()
    assert torch.allclose(out.float(), exp, atol=1e-2, rtol=1e-2)


def test_vendor_reference_numerics_cpu():
    """rmsnorm/silu_mul oracle sanity on CPU (fp32) — vs a direct torch compute."""
    import torch
    import torch.nn.functional as F

    ns = V.make_vendor_reference("rmsnorm", "fp32")
    x, w = ns["get_inputs"]({"M": 4, "N": 16}, device="cpu", seed=0)
    y = ns["ref_fn"](x, w)
    exp = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + V.EPS) * w.float()
    assert torch.allclose(y, exp.to(x.dtype), atol=1e-5)

    ns2 = V.make_vendor_reference("silu_mul", "fp32")
    (xx,) = ns2["get_inputs"]({"M": 4, "N": 16}, device="cpu", seed=0)
    yy = ns2["ref_fn"](xx)
    inter = xx.shape[-1] // 2
    exp2 = F.silu(xx[:, :inter].float()) * xx[:, inter:].float()
    assert torch.allclose(yy, exp2.to(xx.dtype), atol=1e-5)
    assert yy.shape == (4, inter)


def test_vendor_tasks_registered():
    from kore.tasks.registry import all_tasks
    ids = {t.task_id for t in all_tasks()}
    assert any(t.startswith("genv_") for t in ids)
    # vendor tasks carry the real-baseline tier
    from kore.tasks.registry import get_task
    t = get_task("genv_rmsnorm_bf16")
    assert t.raw.get("baseline_tier") == "vendor"
