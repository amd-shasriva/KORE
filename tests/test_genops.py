"""CPU-only tests for the operator-generation engine (data-scale harness)."""

from __future__ import annotations

import ast

import pytest

from kore.tasks import _genops
from kore.reward.reward import scan_for_hacks


def test_registry_has_wide_op_coverage():
    names = _genops.op_names()
    assert len(names) >= 30
    # families represented
    fams = {_genops._registry()[n][0] for n in names}
    assert {"unary", "binary", "reduce"} <= fams


@pytest.mark.parametrize("op", _genops.op_names())
def test_seed_source_parses_and_defines_entry(op):
    family = _genops._registry()[op][0]
    src = _genops.seed_source(op, family, "bf16")
    tree = ast.parse(src)                      # compiles as valid Python
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert op in funcs                         # exposes the entry function
    # a generated seed is a real Triton kernel, never a reward-hack
    assert scan_for_hacks(src) is None


@pytest.mark.parametrize("op", _genops.op_names())
def test_make_reference_namespace(op):
    family = _genops._registry()[op][0]
    ns = _genops.make_reference(op, family, "fp32")
    for key in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity", "entry_name"):
        assert key in ns
    assert ns["entry_name"] == op
    assert ns["arity"] in (1, 2)


def test_reference_numerics_match_torch_cpu():
    """Oracle vs a direct torch compute agree on CPU (fp32) for a few ops."""
    import torch

    for op in ("relu", "silu", "square", "add", "row_sum"):
        family = _genops._registry()[op][0]
        ns = _genops.make_reference(op, family, "fp32")
        shape = {"M": 8, "N": 16}
        inputs = ns["get_inputs"](shape, device="cpu", seed=0)
        out = ns["ref_fn"](*inputs)
        assert torch.isfinite(out).all()
        if op == "relu":
            assert torch.allclose(out, torch.relu(inputs[0]))
        if op == "add":
            assert torch.allclose(out, inputs[0] + inputs[1])
        if op == "row_sum":
            assert out.shape == (8,)
            assert torch.allclose(out, inputs[0].sum(-1), atol=1e-4)


def test_positive_domain_inputs_are_positive():
    import torch
    ns = _genops.make_reference("sqrt", "unary", "fp32")
    (x,) = ns["get_inputs"]({"M": 4, "N": 8}, device="cpu", seed=0)
    assert (x > 0).all()   # sqrt/log/rsqrt/reciprocal get a positive domain


def test_generated_tasks_registered_and_wide():
    from kore.tasks.registry import all_tasks, train_tasks
    tasks = {t.task_id for t in all_tasks()}
    # generation is checked in; the suite should be wide (>= 100 operators)
    assert len(tasks) >= 100
    assert any(t.startswith("gen_") for t in tasks)
    # generated ops are all training ops (held-out stays the attention family)
    assert len(train_tasks()) >= 90
