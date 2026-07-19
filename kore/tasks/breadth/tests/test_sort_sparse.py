"""CPU-only tests for the breadth SORT/SELECT + SPARSE task engine.

Asserts (all on CPU, fp32 unless noted): the fp32 oracle (``ref_fn``) matches an
INDEPENDENT torch computation for every op, arity is exposed correctly, every seed
parses + compiles + defines its entry function, shapes round-trip through
``parse_shape``, and ref_fn preserves the input dtype. No GPU / triton runtime is
required (the seed is only static-checked; the teacher compiles it on the MI350X).
"""

from __future__ import annotations

import ast

import pytest

from kore.tasks.breadth import sort_sparse as S

_EXPECTED_ARITY = {
    "topk_values": 1, "argmax_lastdim": 1, "sort_lastdim": 1, "topp_mask": 1,
    "sparse_2to4_apply": 2, "block_sparse_matmul": 3, "spmm_csr": 3, "sddmm": 3,
}


def _all_shape_dicts(op):
    """Every concrete shape dict declared for an op (minimal + primary + validation)."""
    spec = S.SHAPES[op]
    out = [spec["minimal"], spec["primary"], *spec["validation"]]
    return out


def _spec_str(shape):
    return ",".join(f"{k}={v}" for k, v in shape.items())


# --------------------------------------------------------------------------- #
# metadata / ABI surface
# --------------------------------------------------------------------------- #
def test_ops_and_metadata_cover_every_op():
    assert len(S.OPS) == 8
    assert set(S.OPS) == set(_EXPECTED_ARITY)
    for op in S.OPS:
        assert op in S.OP_DTYPES and op in S.SHAPES
        assert S.op_dtypes(op) == S.OP_DTYPES[op]
        assert all(dt in ("bf16", "fp16", "fp32") for dt in S.op_dtypes(op))
    assert S.op_dtypes("topk_values") == S.DEFAULT_DTYPES


@pytest.mark.parametrize("op", S.OPS)
def test_reference_namespace(op):
    ns = S.make_reference(op, S.op_dtypes(op)[0])
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
              "entry_name", "dtype_name", "family"):
        assert k in ns, k
    assert ns["entry_name"] == op
    assert ns["family"] == f"breadth_{op}"
    assert ns["arity"] == _EXPECTED_ARITY[op]
    assert callable(ns["ref_fn"]) and callable(ns["baseline_fn"])
    assert ns[f"{op}_ref"] is ns["ref_fn"]


@pytest.mark.parametrize("op", S.OPS)
def test_seed_parses_compiles_and_defines_entry(op):
    for dtype in S.op_dtypes(op):
        src = S.seed_source(op, dtype)
        tree = ast.parse(src)                       # valid Python
        compile(src, f"<seed:{op}:{dtype}>", "exec")  # and compiles to bytecode
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert op in funcs, (op, dtype, funcs)
        # the public entry must be an importable top-level def
        assert any(isinstance(n, ast.FunctionDef) and n.name == op
                   for n in tree.body)


@pytest.mark.parametrize("op", S.OPS)
def test_shapes_parse_roundtrip(op):
    ns = S.make_reference(op, S.op_dtypes(op)[0])
    parse_shape = ns["parse_shape"]
    for shape in _all_shape_dicts(op):
        parsed = parse_shape(_spec_str(shape))
        assert parsed == shape, (op, parsed, shape)


@pytest.mark.parametrize("op", S.OPS)
def test_arity_matches_get_inputs(op):
    ns = S.make_reference(op, "fp32")
    inputs = ns["get_inputs"](S.SHAPES[op]["minimal"], device="cpu", seed=0)
    assert isinstance(inputs, tuple)
    assert len(inputs) == ns["arity"] == _EXPECTED_ARITY[op]


@pytest.mark.parametrize("op", S.OPS)
@pytest.mark.parametrize("dtype", ("bf16", "fp16"))
def test_ref_preserves_input_dtype(op, dtype):
    import torch
    ns = S.make_reference(op, dtype)
    inputs = ns["get_inputs"](S.SHAPES[op]["minimal"], device="cpu", seed=1)
    out = ns["ref_fn"](*inputs)
    tdt = getattr(torch, {"bf16": "bfloat16", "fp16": "float16"}[dtype])
    outs = out if isinstance(out, (tuple, list)) else (out,)
    assert all(o.dtype == tdt for o in outs)


# --------------------------------------------------------------------------- #
# fp32 oracle correctness vs an INDEPENDENT torch compute
# --------------------------------------------------------------------------- #
def _close(a, b, atol=1e-4, rtol=1e-4):
    import torch
    return torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)


def test_topk_values_matches_torch():
    import torch
    ns = S.make_reference("topk_values", "fp32")
    (x,) = ns["get_inputs"]({"M": 6, "N": 41}, device="cpu", seed=0)
    out = ns["ref_fn"](x)
    exp = torch.topk(x.float(), S.TOPK_K, dim=-1).values
    assert out.shape == (6, S.TOPK_K)
    assert _close(out, exp)
    # descending, and equals the k largest of the row (order-independent set check)
    assert torch.all(out[:, 1:] <= out[:, :-1] + 1e-6)


def test_argmax_lastdim_matches_torch():
    import torch
    ns = S.make_reference("argmax_lastdim", "fp32")
    (x,) = ns["get_inputs"]({"M": 6, "N": 41}, device="cpu", seed=0)
    out = ns["ref_fn"](x)
    # SNR-safe: value gathered at argmax == row max
    gathered = torch.gather(x.float(), -1, x.float().argmax(-1, keepdim=True)).squeeze(-1)
    assert out.shape == (6,)
    assert _close(out, x.float().amax(-1)) and _close(out, gathered)


def test_sort_lastdim_matches_torch():
    import torch
    ns = S.make_reference("sort_lastdim", "fp32")
    (x,) = ns["get_inputs"]({"M": 6, "N": 41}, device="cpu", seed=0)
    out = ns["ref_fn"](x)
    exp = torch.sort(x.float(), dim=-1).values
    assert out.shape == (6, 41)
    assert _close(out, exp)
    assert torch.all(out[:, 1:] >= out[:, :-1])  # ascending


def test_topp_mask_matches_torch_and_renormalizes():
    import torch
    ns = S.make_reference("topp_mask", "fp32")
    (x,) = ns["get_inputs"]({"M": 7, "N": 53}, device="cpu", seed=0)
    out = ns["ref_fn"](x)
    # independent HF-style nucleus mask (shift-right of cumsum > p)
    probs = torch.softmax(x.float(), dim=-1)
    sp, si = torch.sort(probs, dim=-1, descending=True)
    remove = sp.cumsum(-1) > S.TOPP_P
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, si, ~remove)
    masked = torch.where(keep, probs, torch.zeros_like(probs))
    exp = masked / masked.sum(-1, keepdim=True)
    assert out.shape == (7, 53)
    assert _close(out, exp, atol=1e-5, rtol=1e-5)
    assert _close(out.sum(-1), torch.ones(7))          # renormalized
    assert torch.all(out >= 0)                          # a valid distribution


def test_sparse_2to4_apply_matches_torch():
    import torch
    ns = S.make_reference("sparse_2to4_apply", "fp32")
    x, w = ns["get_inputs"]({"M": 5, "K": 12, "N": 16}, device="cpu", seed=0)
    out = ns["ref_fn"](x, w)
    K, N = w.shape
    g = w.float().reshape(K, N // 4, 4)
    idx = g.abs().topk(2, dim=-1).indices
    keep = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx, True)
    ws = torch.where(keep, g, torch.zeros_like(g)).reshape(K, N)
    # exactly 2 of every 4 kept along the last dim
    nnz = (ws.reshape(K, N // 4, 4) != 0).sum(-1)
    assert torch.all(nnz == 2)
    assert out.shape == (5, N)
    assert _close(out, x.float() @ ws, atol=1e-3, rtol=1e-3)


def test_block_sparse_matmul_matches_torch():
    import torch
    ns = S.make_reference("block_sparse_matmul", "fp32")
    x, w, mask = ns["get_inputs"]({"M": 5, "K": 64, "N": 96}, device="cpu", seed=0)
    assert mask.shape == (64 // S.BLK_K, 96 // S.BLK_N)
    out = ns["ref_fn"](x, w, mask)
    K, N = w.shape
    Kb, Nb = mask.shape
    bk, bn = K // Kb, N // Nb
    wm = (w.float().reshape(Kb, bk, Nb, bn)
          * mask.float().reshape(Kb, 1, Nb, 1)).reshape(K, N)
    assert out.shape == (5, N)
    assert _close(out, x.float() @ wm, atol=1e-3, rtol=1e-3)


def test_spmm_csr_matches_torch():
    import torch
    ns = S.make_reference("spmm_csr", "fp32")
    a, b, mask = ns["get_inputs"]({"M": 5, "K": 20, "N": 9}, device="cpu", seed=0)
    assert mask.shape == (5, 20)
    out = ns["ref_fn"](a, b, mask)
    exp = (a.float() * mask.float()) @ b.float()
    assert out.shape == (5, 9)
    assert _close(out, exp, atol=1e-3, rtol=1e-3)


def test_sddmm_matches_torch():
    import torch
    ns = S.make_reference("sddmm", "fp32")
    a, b, mask = ns["get_inputs"]({"M": 5, "K": 20, "N": 9}, device="cpu", seed=0)
    assert mask.shape == (5, 9)
    out = ns["ref_fn"](a, b, mask)
    exp = (a.float() @ b.float()) * mask.float()
    assert out.shape == (5, 9)
    assert _close(out, exp, atol=1e-3, rtol=1e-3)


def test_baseline_matches_reference_shapes():
    """The torch baseline (native dtype) must agree in shape with the fp32 oracle
    on every op (they compute the same math; baseline is the timed 'production' bar)."""
    for op in S.OPS:
        ns = S.make_reference(op, "fp32")
        inputs = ns["get_inputs"](S.SHAPES[op]["minimal"], device="cpu", seed=3)
        r = ns["ref_fn"](*inputs)
        b = ns["baseline_fn"](*inputs)
        rs = r if isinstance(r, (tuple, list)) else (r,)
        bs = b if isinstance(b, (tuple, list)) else (b,)
        assert len(rs) == len(bs)
        for ro, bo in zip(rs, bs):
            assert ro.shape == bo.shape, (op, ro.shape, bo.shape)
