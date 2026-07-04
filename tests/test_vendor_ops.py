"""CPU-only tests for the vendor-baselined op authoring engine."""

from __future__ import annotations

import ast

import pytest

from kore.tasks import vendor_ops as V


@pytest.mark.parametrize("op", V.VENDOR_OPS)
def test_vendor_seed_parses_and_defines_entry(op):
    src = V.vendor_seed_source(op, "bf16")
    tree = ast.parse(src)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert op in funcs


@pytest.mark.parametrize("op", V.VENDOR_OPS)
def test_vendor_reference_namespace(op):
    ns = V.make_vendor_reference(op, "fp16")
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity", "entry_name"):
        assert k in ns
    assert ns["entry_name"] == op
    assert ns["arity"] in (1, 2, 3)


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
