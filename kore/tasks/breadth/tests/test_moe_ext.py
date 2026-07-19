"""CPU-only tests for the breadth MIXTURE-OF-EXPERTS authoring engine.

Every ``ref_fn`` is checked against an INDEPENDENT torch computation on a DIFFERENT
code path than the (vectorized, per-expert) oracle it wraps: the routers are
cross-checked against an argsort/python-loop selection; the grouped/batched GEMMs
against a per-TOKEN (or einsum) matmul; the fused MoE MLP against a per-token /
per-slot dense loop; permute/unpermute against explicit index loops; histogram /
offsets against python Counter / cumulative loops. So a wrong oracle is caught with
certainty. Also asserts the ABI surface, arity, that each seed compiles + defines
its entry, the shape catalog round-trips through ``parse_shape``, output dtype
preservation, and the MoE routing invariants (top-k weight renorm/sum, top-k
selection count, expert-choice capacity, permutation bijection, offsets == scan of
the histogram). All fp32/fp64 on CPU (no GPU / triton execution - the seed is only
static-checked; fp8 uses shared e4m3 quant so the oracle match is tight)."""

from __future__ import annotations

import ast

import pytest
import torch

from kore.tasks._genops import DTYPES
from kore.tasks.breadth import moe_ext as M

DTYPE_NAMES = ("bf16", "fp16", "fp32", "fp8")

# --------------------------------------------------------------------------- #
# tiny CPU shapes + expected arity, keyed by kind
# --------------------------------------------------------------------------- #
SMALL = {
    "route_softmax": {"M": 6, "E": 8, "topk": 3},
    "route_sigmoid": {"M": 6, "E": 8, "topk": 3},
    "route_topk_then_softmax": {"M": 6, "E": 8, "topk": 3},
    "route_grouped": {"M": 6, "E": 8, "topk": 3, "n_groups": 2, "topk_group": 1},
    "route_biased_grouped": {"M": 6, "E": 8, "topk": 3, "n_groups": 2, "topk_group": 1},
    "route_expert_choice": {"M": 10, "E": 4, "cap": 3},
    "permute": {"M": 10, "E": 4, "D": 6},
    "unpermute": {"M": 10, "E": 4, "D": 6},
    "permute_probs": {"M": 10, "E": 4, "D": 6},
    "histogram": {"M": 10, "E": 4},
    "offsets": {"M": 10, "E": 4},
    "align_offsets": {"M": 10, "E": 4},
    "grouped_gemm": {"M": 10, "E": 4, "N": 6, "K": 5},
    "grouped_gemm_fp8": {"M": 10, "E": 4, "N": 6, "K": 5},
    "batched_gemm": {"E": 3, "m": 4, "N": 6, "K": 5},
    "batched_gemm_fp8": {"E": 3, "m": 4, "N": 6, "K": 5},
    "grouped_gate_up": {"M": 10, "E": 4, "D": 6, "I": 5},
    "grouped_down": {"M": 10, "E": 4, "D": 6, "I": 5},
    "grouped_swiglu": {"M": 10, "E": 4, "D": 6, "I": 5},
    "grouped_geglu": {"M": 10, "E": 4, "D": 6, "I": 5},
    "grouped_mlp": {"M": 10, "E": 4, "D": 6, "I": 5},
    "grouped_mlp_fp8": {"M": 10, "E": 4, "D": 6, "I": 5},
    "sum_combine": {"M": 8, "topk": 3, "D": 6},
    "finalize": {"M": 8, "topk": 3, "D": 6},
    "shared_expert": {"M": 8, "D": 6, "I": 5},
    "fused_moe": {"M": 8, "E": 4, "topk": 2, "D": 6, "I": 5},
    "moe_block": {"M": 8, "E": 4, "topk": 2, "D": 6, "I": 5},
}
ARITY_BY_KIND = {
    "route_softmax": 2, "route_sigmoid": 2, "route_topk_then_softmax": 2,
    "route_grouped": 4, "route_biased_grouped": 5, "route_expert_choice": 2,
    "permute": 2, "unpermute": 2, "permute_probs": 3,
    "histogram": 2, "offsets": 2, "align_offsets": 3,
    "grouped_gemm": 3, "grouped_gemm_fp8": 5, "batched_gemm": 2, "batched_gemm_fp8": 4,
    "grouped_gate_up": 3, "grouped_down": 3, "grouped_swiglu": 3, "grouped_geglu": 3,
    "grouped_mlp": 4, "grouped_mlp_fp8": 7,
    "sum_combine": 2, "finalize": 3, "shared_expert": 4, "fused_moe": 5, "moe_block": 5,
}


def _small(op):
    return dict(SMALL[M.KIND[op]])


def _oracle_dtype(op):
    return "fp8" if M.KIND[op] in M.FP8_KINDS else "fp32"


# --------------------------------------------------------------------------- #
# independent fp64 oracles (distinct code paths from the vectorized ref_fn)
# --------------------------------------------------------------------------- #
def _act_d(x, act):
    xf = x.double()
    if act == "silu":
        return xf * torch.sigmoid(xf)
    return torch.nn.functional.gelu(xf, approximate="tanh")


def _softmax_d(g):
    ex = torch.exp(g - g.max(dim=-1, keepdim=True).values)
    return ex / ex.sum(dim=-1, keepdim=True)


def _ind_route_score(gate, topk, mode, renorm):
    Mn, E = gate.shape
    sc = _softmax_d(gate.double()) if mode == "softmax" else torch.sigmoid(gate.double())
    dense = torch.zeros(Mn, E, dtype=torch.float64)
    for m in range(Mn):
        order = torch.argsort(sc[m], descending=True)[:topk]
        w = sc[m][order].clone()
        if renorm:
            w = w / w.sum()
        for j, e in enumerate(order.tolist()):
            dense[m, e] = w[j]
    return dense


def _ind_route_tts(gate, topk):
    gf = gate.double()
    Mn, E = gate.shape
    dense = torch.zeros(Mn, E, dtype=torch.float64)
    for m in range(Mn):
        order = torch.argsort(gf[m], descending=True)[:topk]
        w = torch.softmax(gf[m][order], dim=0)
        for j, e in enumerate(order.tolist()):
            dense[m, e] = w[j]
    return dense


def _ind_route_grouped(gate, topk, n_groups, topk_group, renorm=True):
    Mn, E = gate.shape
    grp = E // n_groups
    sm = _softmax_d(gate.double())
    dense = torch.zeros(Mn, E, dtype=torch.float64)
    for m in range(Mn):
        gscore = [float(sm[m][g * grp:(g + 1) * grp].max()) for g in range(n_groups)]
        kept = sorted(range(n_groups), key=lambda g: -gscore[g])[:topk_group]
        cand = [e for g in kept for e in range(g * grp, (g + 1) * grp)]
        cand = sorted(cand, key=lambda e: -float(sm[m][e]))[:topk]
        w = torch.tensor([float(sm[m][e]) for e in cand], dtype=torch.float64)
        if renorm:
            w = w / w.sum()
        for j, e in enumerate(cand):
            dense[m, e] = w[j]
    return dense


def _ind_route_biased(gate, bias, topk, n_groups, topk_group, renorm=True):
    Mn, E = gate.shape
    grp = E // n_groups
    scores = torch.sigmoid(gate.double())
    sb = scores + bias.double().view(1, E)
    dense = torch.zeros(Mn, E, dtype=torch.float64)
    for m in range(Mn):
        gscore = []
        for g in range(n_groups):
            seg = sb[m][g * grp:(g + 1) * grp]
            gscore.append(float(torch.sort(seg, descending=True).values[:min(2, grp)].sum()))
        kept = sorted(range(n_groups), key=lambda g: -gscore[g])[:topk_group]
        cand = [e for g in kept for e in range(g * grp, (g + 1) * grp)]
        cand = sorted(cand, key=lambda e: -float(sb[m][e]))[:topk]
        w = torch.tensor([float(scores[m][e]) for e in cand], dtype=torch.float64)
        if renorm:
            w = w / w.sum().clamp(min=1e-12)
        for j, e in enumerate(cand):
            dense[m, e] = w[j]
    return dense


def _ind_route_ec(gate, cap):
    Mn, E = gate.shape
    cap = min(int(cap), Mn)
    sm = _softmax_d(gate.double())
    dense = torch.zeros(Mn, E, dtype=torch.float64)
    for e in range(E):
        order = torch.argsort(sm[:, e], descending=True)[:cap]
        for t in order.tolist():
            dense[t, e] = sm[t, e]
    return dense


def _ind_permute(hidden, sort_idx):
    Mn, D = hidden.shape
    out = torch.zeros(Mn, D, dtype=torch.float64)
    for i in range(Mn):
        out[i] = hidden[int(sort_idx[i])].double()
    return out


def _ind_unpermute(permuted, sort_idx):
    Mn, D = permuted.shape
    out = torch.zeros(Mn, D, dtype=torch.float64)
    for i in range(Mn):
        out[int(sort_idx[i])] = permuted[i].double()
    return out


def _ind_permute_probs(hidden, probs, sort_idx):
    Mn, D = hidden.shape
    oh = torch.zeros(Mn, D, dtype=torch.float64)
    op = torch.zeros(Mn, dtype=torch.float64)
    for i in range(Mn):
        oh[i] = hidden[int(sort_idx[i])].double()
        op[i] = probs[int(sort_idx[i])].double()
    return (oh, op)


def _ind_histogram(expert_ids, E):
    counts = [0] * E
    for e in expert_ids.tolist():
        counts[e] += 1
    return torch.tensor(counts, dtype=torch.int64)


def _ind_offsets(expert_ids, E):
    counts = [0] * E
    for e in expert_ids.tolist():
        counts[e] += 1
    off = [0]
    for e in range(E):
        off.append(off[-1] + counts[e])
    return torch.tensor(off, dtype=torch.int64)


def _ind_align(expert_ids, E, block):
    counts = [0] * E
    for e in expert_ids.tolist():
        counts[e] += 1
    off = [0]
    for e in range(E):
        off.append(off[-1] + ((counts[e] + block - 1) // block) * block)
    return torch.tensor(off, dtype=torch.int64)


def _ind_grouped(x, w, expert_ids):
    Mn = x.shape[0]
    N = w.shape[1]
    out = torch.zeros(Mn, N, dtype=torch.float64)
    for m in range(Mn):
        e = int(expert_ids[m])
        out[m] = x[m].double() @ w[e].double().t()
    return out


def _ind_grouped_fp8(xq, wq, xs, ws, expert_ids):
    x = xq.double() * xs.double()
    w = wq.double() * ws.double()
    Mn, N = x.shape[0], w.shape[1]
    out = torch.zeros(Mn, N, dtype=torch.float64)
    for m in range(Mn):
        e = int(expert_ids[m])
        out[m] = x[m] @ w[e].t()
    return out


def _ind_batched(a, b):
    return torch.einsum("emk,enk->emn", a.double(), b.double())


def _ind_batched_fp8(aq, bq, as_, bs):
    a = aq.double() * as_.double()
    b = bq.double() * bs.double()
    return torch.einsum("emk,enk->emn", a, b)


def _ind_grouped_act(hidden, w13, expert_ids, act):
    Mn = hidden.shape[0]
    I = w13.shape[1] // 2
    out = torch.zeros(Mn, I, dtype=torch.float64)
    for m in range(Mn):
        e = int(expert_ids[m])
        gu = hidden[m].double() @ w13[e].double().t()
        out[m] = _act_d(gu[:I], act) * gu[I:]
    return out


def _ind_mlp(hidden, w13, w2, expert_ids, act):
    Mn, D = hidden.shape
    I = w2.shape[2]
    out = torch.zeros(Mn, D, dtype=torch.float64)
    for m in range(Mn):
        e = int(expert_ids[m])
        gu = hidden[m].double() @ w13[e].double().t()
        h = _act_d(gu[:I], act) * gu[I:]
        out[m] = h @ w2[e].double().t()
    return out


def _ind_mlp_fp8(xq, w13q, w2q, xs, w13s, w2s, expert_ids, act):
    x = xq.double() * xs.double()
    w13 = w13q.double() * w13s.double()
    w2 = w2q.double() * w2s.double()
    Mn, D = x.shape
    I = w2.shape[2]
    out = torch.zeros(Mn, D, dtype=torch.float64)
    for m in range(Mn):
        e = int(expert_ids[m])
        gu = x[m] @ w13[e].t()
        h = _act_d(gu[:I], act) * gu[I:]
        out[m] = h @ w2[e].t()
    return out


def _ind_sum_combine(y, tw):
    Mn, topk, D = y.shape
    out = torch.zeros(Mn, D, dtype=torch.float64)
    for m in range(Mn):
        for k in range(topk):
            out[m] += float(tw[m, k]) * y[m, k].double()
    return out


def _ind_finalize(y_perm, row_map, tw):
    Mn, topk = row_map.shape
    D = y_perm.shape[1]
    out = torch.zeros(Mn, D, dtype=torch.float64)
    for m in range(Mn):
        for k in range(topk):
            out[m] += float(tw[m, k]) * y_perm[int(row_map[m, k])].double()
    return out


def _ind_shared(hidden, ws1, ws2, routed, act):
    Mn, D = hidden.shape
    I = ws2.shape[1]
    out = torch.zeros(Mn, D, dtype=torch.float64)
    for m in range(Mn):
        gu = hidden[m].double() @ ws1.double().t()
        h = _act_d(gu[:I], act) * gu[I:]
        out[m] = routed[m].double() + h @ ws2.double().t()
    return out


def _ind_fused(hidden, w1, w2, tw, ti, act):
    Mn, D = hidden.shape
    I = w2.shape[2]
    out = torch.zeros(Mn, D, dtype=torch.float64)
    for m in range(Mn):
        for j in range(ti.shape[1]):
            e = int(ti[m, j])
            gu = hidden[m].double() @ w1[e].double().t()
            h = _act_d(gu[:I], act) * gu[I:]
            out[m] += float(tw[m, j]) * (h @ w2[e].double().t())
    return out


def _ind_block(hidden, gate, w1, w2, topk, act, router):
    sc = torch.sigmoid(gate.double()) if router == "sigmoid" else _softmax_d(gate.double())
    Mn, D = hidden.shape
    I = w2.shape[2]
    out = torch.zeros(Mn, D, dtype=torch.float64)
    for m in range(Mn):
        order = torch.argsort(sc[m], descending=True)[:topk]
        w = sc[m][order].clone()
        w = w / w.sum()
        for j, e in enumerate(order.tolist()):
            gu = hidden[m].double() @ w1[e].double().t()
            h = _act_d(gu[:I], act) * gu[I:]
            out[m] += float(w[j]) * (h @ w2[e].double().t())
    return out


def _independent(op, inputs):
    kind = M.KIND[op]
    spec = M._SPECS[op]
    act = spec.get("act", "silu")
    renorm = spec.get("renorm", True)
    router = spec.get("router", "softmax")
    if kind == "route_softmax":
        return _ind_route_score(inputs[0], inputs[1], "softmax", renorm)
    if kind == "route_sigmoid":
        return _ind_route_score(inputs[0], inputs[1], "sigmoid", renorm)
    if kind == "route_topk_then_softmax":
        return _ind_route_tts(inputs[0], inputs[1])
    if kind == "route_grouped":
        return _ind_route_grouped(*inputs, renorm=renorm)
    if kind == "route_biased_grouped":
        return _ind_route_biased(*inputs, renorm=renorm)
    if kind == "route_expert_choice":
        return _ind_route_ec(inputs[0], inputs[1])
    if kind == "permute":
        return _ind_permute(*inputs)
    if kind == "unpermute":
        return _ind_unpermute(*inputs)
    if kind == "permute_probs":
        return _ind_permute_probs(*inputs)
    if kind == "histogram":
        return _ind_histogram(*inputs)
    if kind == "offsets":
        return _ind_offsets(*inputs)
    if kind == "align_offsets":
        return _ind_align(*inputs)
    if kind in ("grouped_gemm", "grouped_gate_up", "grouped_down"):
        return _ind_grouped(*inputs)
    if kind == "grouped_gemm_fp8":
        return _ind_grouped_fp8(*inputs)
    if kind == "batched_gemm":
        return _ind_batched(*inputs)
    if kind == "batched_gemm_fp8":
        return _ind_batched_fp8(*inputs)
    if kind in ("grouped_swiglu", "grouped_geglu"):
        return _ind_grouped_act(*inputs, act=act)
    if kind == "grouped_mlp":
        return _ind_mlp(*inputs, act=act)
    if kind == "grouped_mlp_fp8":
        return _ind_mlp_fp8(*inputs, act=act)
    if kind == "sum_combine":
        return _ind_sum_combine(*inputs)
    if kind == "finalize":
        return _ind_finalize(*inputs)
    if kind == "shared_expert":
        return _ind_shared(*inputs, act=act)
    if kind == "fused_moe":
        return _ind_fused(*inputs, act=act)
    if kind == "moe_block":
        return _ind_block(*inputs, act=act, router=router)
    raise AssertionError(f"no independent oracle for {op!r}")


def _relerr(a, b):
    a, b = a.double(), b.double()
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


# --------------------------------------------------------------------------- #
# metadata / ABI surface
# --------------------------------------------------------------------------- #
def test_abi_present():
    assert isinstance(M.OPS, list) and len(M.OPS) == 40
    assert len(set(M.OPS)) == len(M.OPS)
    assert all(op.startswith("moe_") for op in M.OPS)
    assert callable(M.make_reference) and callable(M.seed_source)
    assert set(M.OP_DTYPES) == set(M.OPS)
    assert set(M.SHAPES) == set(M.OPS)
    assert set(M.KIND) == set(M.OPS)
    assert {M.KIND[op] for op in M.OPS} == set(SMALL), "SMALL must cover every op kind"


def test_ops_dtypes_shapes_consistent():
    for op in M.OPS:
        assert M.OP_DTYPES[op], f"empty dtype sweep for {op}"
        assert M.op_dtypes(op) == M.OP_DTYPES[op]
        for d in M.OP_DTYPES[op]:
            assert d in DTYPE_NAMES, f"unknown dtype {d} for {op}"
        sh = M.SHAPES[op]
        assert "minimal" in sh and "primary" in sh and "validation" in sh
        assert isinstance(sh["validation"], list) and sh["validation"]
    # fp8 only for the quantized expert-GEMM variants
    for op in M.OPS:
        if M.KIND[op] in M.FP8_KINDS:
            assert M.OP_DTYPES[op] == ["fp8"]
        else:
            assert M.OP_DTYPES[op] == ["bf16", "fp16"]


@pytest.mark.parametrize("op", M.OPS)
def test_namespace_contract(op):
    d = M.op_dtypes(op)[0]
    ns = M.make_reference(op, d)
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
              "entry_name", "dtype_name", "family", "mutates_input"):
        assert k in ns, k
    assert ns["entry_name"] == op
    assert ns["dtype_name"] == d
    assert ns["family"] == f"breadth_{op}"
    assert ns["mutates_input"] is False
    assert callable(ns["ref_fn"]) and callable(ns["baseline_fn"])
    assert ns[f"{op}_ref"] is ns["ref_fn"]


@pytest.mark.parametrize("op", M.OPS)
def test_arity(op):
    ns = M.make_reference(op, _oracle_dtype(op))
    assert ns["arity"] == ARITY_BY_KIND[M.KIND[op]]
    inputs = ns["get_inputs"](_small(op), device="cpu", seed=0)
    assert isinstance(inputs, tuple)
    assert len(inputs) == ns["arity"]


# --------------------------------------------------------------------------- #
# fp32 oracle correctness vs an INDEPENDENT torch compute
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", M.OPS)
def test_ref_matches_independent(op):
    ns = M.make_reference(op, _oracle_dtype(op))
    inputs = ns["get_inputs"](_small(op), device="cpu", seed=0)
    ref = ns["ref_fn"](*inputs)
    ind = _independent(op, inputs)
    if M.KIND[op] in M.INT_KINDS:
        assert torch.equal(ref.long(), ind.long()), (op, ref, ind)
    elif isinstance(ref, tuple):
        assert len(ref) == len(ind)
        for r, i in zip(ref, ind):
            assert _relerr(r, i) < 2e-3, (op, _relerr(r, i))
    else:
        tol = 3e-2 if M.KIND[op] in M.FP8_KINDS else 2e-3
        assert ref.shape == ind.shape, (op, tuple(ref.shape), tuple(ind.shape))
        assert _relerr(ref, ind) < tol, (op, _relerr(ref, ind))


@pytest.mark.parametrize("op", M.OPS)
def test_baseline_matches_ref(op):
    """The torch eager baseline agrees with the fp32/quant oracle (same math)."""
    ns = M.make_reference(op, _oracle_dtype(op))
    inputs = ns["get_inputs"](_small(op), device="cpu", seed=1)
    out = ns["baseline_fn"](*inputs)
    ref = ns["ref_fn"](*inputs)
    outs = out if isinstance(out, tuple) else (out,)
    refs = ref if isinstance(ref, tuple) else (ref,)
    for o, r in zip(outs, refs):
        assert o.shape == r.shape
        if o.is_floating_point():
            assert _relerr(o, r) < 1e-6
        else:
            assert torch.equal(o, r)


@pytest.mark.parametrize("op", M.OPS)
def test_ref_preserves_output_dtype(op):
    for d in M.op_dtypes(op):
        ns = M.make_reference(op, d)
        inputs = ns["get_inputs"](_small(op), device="cpu", seed=2)
        out = ns["ref_fn"](*inputs)
        outs = out if isinstance(out, (tuple, list)) else (out,)
        exp = [getattr(torch, n) for n in M.out_dtype_names(op, d)]
        assert [o.dtype for o in outs] == exp, (op, d, [o.dtype for o in outs], exp)


# --------------------------------------------------------------------------- #
# seed static checks (compiles + defines a top-level entry fn)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", M.OPS)
def test_seed_compiles_and_defines_entry(op):
    for d in M.op_dtypes(op):
        src = M.seed_source(op, d)
        compile(src, f"<{op}_{d}_seed>", "exec")          # valid Python (COMPILING seed)
        tree = ast.parse(src)
        assert any(isinstance(n, ast.FunctionDef) and n.name == op for n in tree.body), (
            f"{op} entry must be a top-level def")


@pytest.mark.parametrize("op", M.OPS)
def test_shapes_parse_roundtrip(op):
    ns = M.make_reference(op, M.op_dtypes(op)[0])
    parse = ns["parse_shape"]
    sh = M.SHAPES[op]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        s = ",".join(f"{k}={v}" for k, v in spec.items())
        assert parse(s) == spec, (op, parse(s), spec)


# --------------------------------------------------------------------------- #
# MoE routing / permute invariants
# --------------------------------------------------------------------------- #
_RENORM_SUM1 = [op for op in M.OPS if M.KIND[op] in
                ("route_topk_then_softmax", "route_grouped", "route_biased_grouped")
                or (M.KIND[op] in ("route_softmax", "route_sigmoid")
                    and M._SPECS[op].get("renorm"))]
_NORENORM = [op for op in M.OPS if M.KIND[op] in ("route_softmax", "route_sigmoid")
             and not M._SPECS[op].get("renorm")]
_TOPK_ROUTERS = [op for op in M.OPS if M.KIND[op] in
                 ("route_softmax", "route_sigmoid", "route_topk_then_softmax",
                  "route_grouped", "route_biased_grouped")]


@pytest.mark.parametrize("op", _RENORM_SUM1)
def test_router_renorm_sums_to_one(op):
    ns = M.make_reference(op, "fp32")
    inputs = ns["get_inputs"](_small(op), device="cpu", seed=0)
    dense = ns["ref_fn"](*inputs)
    rows = dense.sum(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5), (op, rows)


@pytest.mark.parametrize("op", _NORENORM)
def test_router_norenorm_not_normalized(op):
    ns = M.make_reference(op, "fp32")
    inputs = ns["get_inputs"](_small(op), device="cpu", seed=0)
    dense = ns["ref_fn"](*inputs)
    rows = dense.sum(dim=-1)
    assert (rows - 1.0).abs().max() > 1e-4, (op, rows)


@pytest.mark.parametrize("op", _TOPK_ROUTERS)
def test_router_selects_exactly_topk(op):
    ns = M.make_reference(op, "fp32")
    sh = _small(op)
    inputs = ns["get_inputs"](sh, device="cpu", seed=0)
    dense = ns["ref_fn"](*inputs)
    nnz = (dense > 0).sum(dim=-1)
    assert torch.equal(nnz, torch.full_like(nnz, sh["topk"])), (op, nnz)


def test_expert_choice_capacity():
    ns = M.make_reference("moe_expert_choice", "fp32")
    sh = _small("moe_expert_choice")
    inputs = ns["get_inputs"](sh, device="cpu", seed=0)
    dense = ns["ref_fn"](*inputs)
    per_expert = (dense > 0).sum(dim=0)               # tokens picked per expert (columns)
    assert torch.equal(per_expert, torch.full_like(per_expert, sh["cap"]))
    assert int((dense > 0).sum()) == sh["E"] * sh["cap"]


def test_permute_is_bijection_and_roundtrips():
    nsp = M.make_reference("moe_permute", "fp32")
    nsu = M.make_reference("moe_unpermute", "fp32")
    hidden, sort_idx = nsp["get_inputs"](_small("moe_permute"), device="cpu", seed=0)
    assert sorted(sort_idx.tolist()) == list(range(hidden.shape[0]))   # a permutation
    perm = nsp["ref_fn"](hidden, sort_idx)
    back = nsu["ref_fn"](perm, sort_idx)
    assert torch.equal(back, hidden)                                   # unpermute∘permute = id


def test_offsets_are_exclusive_scan_of_histogram():
    nsh = M.make_reference("moe_expert_histogram", "fp32")
    nso = M.make_reference("moe_expert_offsets", "fp32")
    eids, E = nsh["get_inputs"](_small("moe_expert_histogram"), device="cpu", seed=0)
    hist = nsh["ref_fn"](eids, E).long()
    off = nso["ref_fn"](eids, E).long()
    assert off.numel() == E + 1
    assert int(off[0]) == 0
    assert int(off[-1]) == eids.shape[0]                              # covers every token
    assert torch.equal(off[1:] - off[:-1], hist)


def test_align_offsets_block_multiple_and_cover():
    nsa = M.make_reference("moe_align_block_offsets", "fp32")
    eids, E, block = nsa["get_inputs"](_small("moe_align_block_offsets"), device="cpu", seed=0)
    nsh = M.make_reference("moe_expert_histogram", "fp32")
    hist = nsh["ref_fn"](eids, E).long()
    off = nsa["ref_fn"](eids, E, block).long()
    seg = off[1:] - off[:-1]
    assert torch.equal(seg % block, torch.zeros_like(seg))            # each block aligned
    assert bool((seg >= hist).all())                                  # room for every token


def test_grouped_gemm_matches_per_token_dense():
    ns = M.make_reference("moe_grouped_gemm", "fp32")
    hidden, w, eids = ns["get_inputs"](_small("moe_grouped_gemm"), device="cpu", seed=3)
    ref = ns["ref_fn"](hidden, w, eids).double()
    dense = torch.stack([hidden[m].double() @ w[int(eids[m])].double().t()
                         for m in range(hidden.shape[0])])
    assert _relerr(ref, dense) < 2e-3


def test_fused_moe_matches_dense_reference():
    ns = M.make_reference("moe_fused_moe_silu", "fp32")
    hidden, w1, w2, tw, ti = ns["get_inputs"](_small("moe_fused_moe_silu"), device="cpu", seed=2)
    ref = ns["ref_fn"](hidden, w1, w2, tw, ti).double()
    ind = _ind_fused(hidden, w1, w2, tw, ti, "silu")
    assert _relerr(ref, ind) < 2e-3
