"""Breadth MIXTURE-OF-EXPERTS frontier task-authoring engine (torch-baselined).

Widens the KORE suite with the HARD sparse-expert machinery that dominates modern
MoE LLMs (Mixtral 8x7B/8x22B, DeepSeek-V2/V3, Qwen1.5/2-MoE, Llama-4, gpt-oss):
dynamic routing, token permutation, grouped/batched GEMM with variable group
sizes, load imbalance (a giant expert + a dead expert), fused gate/up SwiGLU/GeGLU
expert MLPs, and fp8-quantized expert GEMMs. These are the archetypal "hard for a
GPU" kernels - a naive impl launches one GEMM per expert (or scatters tokens with a
memory-bound gather) and is FAR from optimal, so a fused/grouped Triton kernel has
genuine headroom over the torch-eager multi-kernel baseline. No trivial ops live
here: every task carries the routing/permute/grouped-matmul/combine structure.

Unlike the vendor tasks (graded against AITER), these grade against the honest
torch reference: the correctness ORACLE is an exact fp32 reference (``ref_fn``,
which casts back to the task dtype) and the perf BASELINE is the eager torch
computation (``baseline_fn``) - the naive per-expert path a fused Triton kernel
must beat. The routing / grouped-GEMM / permute / combine conventions REUSE the
verified ``kore.tasks._moe_common`` math (jagged unbalanced router with a giant
expert + a guaranteed 0-token last expert; aiter-native weight layouts
w1 ``[E,2I,D]`` / w2 ``[E,D,I]``; per-token/per-channel fp8 a8w8 quant) but are
re-implemented SELF-CONTAINED here (import only from ``_genops``): we deliberately
do NOT import ``aiter_ref`` (it touches ``torch.cuda`` at import), so the fp8
constant (OCP e4m3fn max = 448) is defined locally and every reference is a pure
CPU-importable torch computation.

Contract mirrors ``kore/tasks/breadth/seq.py`` (and ``vendor_ops.py``) so the shared
``_genops`` driver + generator machinery consume it unchanged:

    OPS / OP_DTYPES / SHAPES              module-level task catalog (every op name is
        prefixed ``moe_``; ~40 hard MoE tasks). Config variants (router type,
        renorm, k/E, activation, fp8) are DISTINCT op names.
    make_reference(op, dtype) -> dict     reference.py namespace (parse_shape,
        get_inputs, ref_fn EXACT fp32 oracle - casts back, may return TUPLES;
        baseline_fn torch; arity; entry_name=op; dtype_name; family=f"breadth_{op}";
        mutates_input=False).
    seed_source(op, dtype) -> str         a naive, COMPILING, CORRECT Triton seed
        defining ``def <op>(*inputs)`` (host-side routing selection + a Triton
        grouped-GEMM / gather / scatter is an honest naive starting point).

CORRECTNESS is paramount: every ``ref_fn`` computes the router / permute /
grouped-matmul / combine math in fp32 and casts back to the task dtype (fp8 GEMMs
dequantize the shared fp8 operands and accumulate in fp32 -> bf16). Every oracle is
validated on CPU against an INDEPENDENT torch implementation at tight fp32 tolerance
plus routing invariants (top-k weights renorm/sum, permutation bijection) - see
tests/test_moe_ext.py. torch/triton are imported lazily so registry discovery never
needs a GPU.
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# --------------------------------------------------------------------------- #
# Local constants (self-contained; do NOT import aiter_ref -> torch.cuda touch)
# --------------------------------------------------------------------------- #
FP8_MAX = 448.0        # OCP e4m3fn max finite (gfx950/CDNA4 native fp8) - quant clamp
ALIGN_BLOCK = 64       # moe_align_block_size padding granularity (aiter moe_sorting)
EPS = 1e-12

# Realistic default MoE dimensions (Mixtral/DeepSeek/Qwen-MoE scale).
DEF_E, DEF_TOPK, DEF_D, DEF_I = 64, 8, 4096, 14336

# Representative jagged per-expert weighting (DATASET_SPEC 1.6 unbalanced 32-expert
# trace): one giant expert, several mid, a long tail, a final DEAD expert. Used as a
# router bias so token->expert counts are unbalanced and expert E-1 gets 0 tokens
# (the mandatory MoE load-imbalance edge). Mirrors _moe_common._TRACE32.
_TRACE32 = [
    16053, 105, 1843, 2724, 327, 88, 4102, 51, 9210, 61, 3020, 44, 1502, 990,
    233, 77, 6740, 120, 410, 58, 2210, 39, 812, 175, 5030, 66, 1360, 92, 3550,
    47, 1180, 0,
]


# --------------------------------------------------------------------------- #
# Task catalog: op -> spec (kind + optional pins E/topk/act/router).
# Every op name is prefixed ``moe_``. Config variants are DISTINCT op names.
# --------------------------------------------------------------------------- #
_SPECS: dict[str, dict] = {
    # -- ROUTING: which experts + weights (order-independent dense [M,E] fp32) ----
    "moe_topk_softmax_renorm":     {"kind": "route_softmax", "renorm": True},
    "moe_topk_softmax_norenorm":   {"kind": "route_softmax", "renorm": False},
    "moe_topk_softmax_k2_e8":      {"kind": "route_softmax", "renorm": True, "E": 8, "topk": 2},
    "moe_topk_softmax_k8_e256":    {"kind": "route_softmax", "renorm": True, "E": 256, "topk": 8},
    "moe_sigmoid_topk_renorm":     {"kind": "route_sigmoid", "renorm": True},
    "moe_sigmoid_topk_norenorm":   {"kind": "route_sigmoid", "renorm": False},
    "moe_topk_then_softmax":       {"kind": "route_topk_then_softmax"},
    "moe_grouped_topk":            {"kind": "route_grouped", "renorm": True},
    "moe_biased_grouped_topk":     {"kind": "route_biased_grouped", "renorm": True},
    "moe_expert_choice":           {"kind": "route_expert_choice"},
    # -- PERMUTE / GATHER / HISTOGRAM: the MoE dispatch machinery ----------------
    "moe_permute":                 {"kind": "permute"},
    "moe_unpermute":               {"kind": "unpermute"},
    "moe_permute_with_probs":      {"kind": "permute_probs"},
    "moe_expert_histogram":        {"kind": "histogram"},
    "moe_expert_offsets":          {"kind": "offsets"},
    "moe_align_block_offsets":     {"kind": "align_offsets"},
    # -- EXPERT GEMM: grouped (variable M) / batched (padded) / fp8 --------------
    "moe_grouped_gemm":            {"kind": "grouped_gemm"},
    "moe_grouped_gemm_fp8":        {"kind": "grouped_gemm_fp8"},
    "moe_batched_gemm":            {"kind": "batched_gemm"},
    "moe_batched_gemm_fp8":        {"kind": "batched_gemm_fp8"},
    "moe_grouped_gemm_gate_up":    {"kind": "grouped_gate_up"},
    "moe_grouped_gemm_down":       {"kind": "grouped_down"},
    # -- FUSED gate/up ACTIVATION expert MLP (SwiGLU/GeGLU) ----------------------
    "moe_grouped_swiglu":          {"kind": "grouped_swiglu", "act": "silu"},
    "moe_grouped_geglu":           {"kind": "grouped_geglu", "act": "gelu"},
    "moe_grouped_mlp_silu":        {"kind": "grouped_mlp", "act": "silu"},
    "moe_grouped_mlp_gelu":        {"kind": "grouped_mlp", "act": "gelu"},
    "moe_grouped_mlp_fp8":         {"kind": "grouped_mlp_fp8", "act": "silu"},
    # -- COMBINE: weighted top-k sum / finalize / shared-expert add --------------
    "moe_sum_combine":             {"kind": "sum_combine"},
    "moe_finalize_moe":            {"kind": "finalize"},
    "moe_shared_expert_mlp":       {"kind": "shared_expert", "act": "silu"},
    # -- FUSED top-k MoE MLP (route weights+ids given -> permute/GG/act/combine) -
    "moe_fused_moe_silu":          {"kind": "fused_moe", "act": "silu"},
    "moe_fused_moe_gelu":          {"kind": "fused_moe", "act": "gelu"},
    "moe_fused_moe_silu_k2_e8":    {"kind": "fused_moe", "act": "silu", "E": 8, "topk": 2},
    "moe_fused_moe_silu_k8_e256":  {"kind": "fused_moe", "act": "silu", "E": 256, "topk": 8},
    "moe_fused_moe_gelu_k4_e64":   {"kind": "fused_moe", "act": "gelu", "E": 64, "topk": 4},
    # -- END-TO-END block: router logits -> route -> MLP -> combine -> [M,D] -----
    "moe_block_silu":              {"kind": "moe_block", "act": "silu", "router": "softmax"},
    "moe_block_gelu":              {"kind": "moe_block", "act": "gelu", "router": "softmax"},
    "moe_block_sigmoid_silu":      {"kind": "moe_block", "act": "silu", "router": "sigmoid"},
    "moe_block_silu_k2_e8":        {"kind": "moe_block", "act": "silu", "router": "softmax", "E": 8, "topk": 2},
    "moe_block_silu_k8_e256":      {"kind": "moe_block", "act": "silu", "router": "softmax", "E": 256, "topk": 8},
}

OPS: list[str] = list(_SPECS)
KIND: dict[str, str] = {op: s["kind"] for op, s in _SPECS.items()}

# Kind groupings (used for dtype sweeps + the test's expected-output-dtype logic).
ROUTER_KINDS = frozenset({
    "route_softmax", "route_sigmoid", "route_topk_then_softmax",
    "route_grouped", "route_biased_grouped", "route_expert_choice",
})
INT_KINDS = frozenset({"histogram", "offsets", "align_offsets"})
FP8_KINDS = frozenset({"grouped_gemm_fp8", "batched_gemm_fp8", "grouped_mlp_fp8"})
TUPLE_KINDS = frozenset({"permute_probs"})

# fp8 only for the quantized expert-GEMM variants; everything else sweeps bf16/fp16
# (the fp32 oracle casts back; routers emit fp32 dense weights regardless of gate
# dtype; histogram/offsets emit int32).
DEFAULT_DTYPES: list[str] = ["bf16", "fp16"]
OP_DTYPES: dict[str, list[str]] = {
    op: (["fp8"] if KIND[op] in FP8_KINDS else list(DEFAULT_DTYPES)) for op in OPS
}


def op_dtypes(op: str) -> list[str]:
    """The dtype sweep for an op (fp8 for the quantized GEMMs, else bf16/fp16)."""
    return OP_DTYPES.get(op, DEFAULT_DTYPES)


def out_dtype_names(op: str, dtype: str) -> list[str]:
    """Expected torch dtype attr-name(s) of ``ref_fn``'s output(s) for this op+dtype.

    Routers emit fp32 dense weights; histogram/offsets emit int32; fp8 GEMMs
    accumulate to bf16; permute_with_probs returns (task-dtype hidden, fp32 probs);
    every other op preserves the task float dtype."""
    kind = KIND[op]
    if kind in ROUTER_KINDS:
        return ["float32"]
    if kind in INT_KINDS:
        return ["int32"]
    if kind in FP8_KINDS:
        return ["bfloat16"]
    if kind == "permute_probs":
        return [DTYPES[dtype][0], "float32"]
    return [DTYPES[dtype][0]]


# --------------------------------------------------------------------------- #
# Shape catalog (realistic MoE shapes): n_tokens M in {4096, 16384}, hidden D=4096,
# n_experts E in {8,16,64,128,256}, topk in {1,2,4,8}, expert ffn I=14336. A
# non-power-of-2 M/K tail stresses masking. Op-name pins (E/topk) fix the router
# width. Minimal shapes are tiny (fast dry-runs); primary/validation are realistic.
# --------------------------------------------------------------------------- #
def _make_shapes(op: str) -> dict:
    spec = _SPECS[op]
    kind = spec["kind"]
    pinE, pinK = ("E" in spec), ("topk" in spec)

    if kind in ("route_softmax", "route_sigmoid", "route_topk_then_softmax"):
        if pinE:
            E, tk = spec["E"], spec["topk"]
            return {"minimal": {"M": 64, "E": E, "topk": tk},
                    "primary": {"M": 4096, "E": E, "topk": tk},
                    "validation": [{"M": 16384, "E": E, "topk": tk},
                                   {"M": 8193, "E": E, "topk": tk}]}
        return {"minimal": {"M": 64, "E": 8, "topk": 2},
                "primary": {"M": 4096, "E": 64, "topk": 8},
                "validation": [{"M": 16384, "E": 128, "topk": 8},
                               {"M": 4096, "E": 256, "topk": 8},
                               {"M": 8193, "E": 64, "topk": 8}]}
    if kind in ("route_grouped", "route_biased_grouped"):
        return {"minimal": {"M": 64, "E": 8, "topk": 2, "n_groups": 2, "topk_group": 1},
                "primary": {"M": 4096, "E": 64, "topk": 8, "n_groups": 8, "topk_group": 4},
                "validation": [{"M": 16384, "E": 128, "topk": 8, "n_groups": 8, "topk_group": 4},
                               {"M": 4096, "E": 256, "topk": 8, "n_groups": 8, "topk_group": 4},
                               {"M": 8193, "E": 64, "topk": 8, "n_groups": 8, "topk_group": 4}]}
    if kind == "route_expert_choice":
        return {"minimal": {"M": 16, "E": 4, "cap": 8},
                "primary": {"M": 4096, "E": 64, "cap": 512},
                "validation": [{"M": 16384, "E": 128, "cap": 1024},
                               {"M": 8192, "E": 64, "cap": 1024},
                               {"M": 4097, "E": 64, "cap": 512}]}
    if kind in ("permute", "unpermute", "permute_probs"):
        return {"minimal": {"M": 64, "E": 8, "D": 128},
                "primary": {"M": 4096, "E": 64, "D": 4096},
                "validation": [{"M": 16384, "E": 128, "D": 4096},
                               {"M": 8192, "E": 256, "D": 4096},
                               {"M": 4095, "E": 64, "D": 4096}]}
    if kind in ("histogram", "offsets", "align_offsets"):
        return {"minimal": {"M": 64, "E": 8},
                "primary": {"M": 4096, "E": 64},
                "validation": [{"M": 16384, "E": 128},
                               {"M": 8192, "E": 256},
                               {"M": 4095, "E": 64}]}
    if kind in ("grouped_gemm", "grouped_gemm_fp8"):
        return {"minimal": {"M": 64, "E": 8, "N": 256, "K": 256},
                "primary": {"M": 4096, "E": 64, "N": 4096, "K": 4096},
                "validation": [{"M": 16384, "E": 128, "N": 4096, "K": 4096},
                               {"M": 4096, "E": 256, "N": 2048, "K": 4096},
                               {"M": 8192, "E": 64, "N": 4096, "K": 4095}]}
    if kind in ("batched_gemm", "batched_gemm_fp8"):
        return {"minimal": {"E": 4, "m": 16, "N": 64, "K": 64},
                "primary": {"E": 8, "m": 512, "N": 4096, "K": 4096},
                "validation": [{"E": 16, "m": 256, "N": 4096, "K": 4096},
                               {"E": 8, "m": 512, "N": 14336, "K": 4096},
                               {"E": 8, "m": 511, "N": 512, "K": 512}]}
    if kind in ("grouped_gate_up", "grouped_down", "grouped_swiglu",
                "grouped_geglu", "grouped_mlp", "grouped_mlp_fp8"):
        return {"minimal": {"M": 64, "E": 8, "D": 128, "I": 256},
                "primary": {"M": 4096, "E": 64, "D": 4096, "I": 14336},
                "validation": [{"M": 16384, "E": 128, "D": 4096, "I": 14336},
                               {"M": 4096, "E": 256, "D": 4096, "I": 14336},
                               {"M": 8192, "E": 64, "D": 4096, "I": 14335}]}
    if kind in ("sum_combine", "finalize"):
        return {"minimal": {"M": 64, "topk": 2, "D": 128},
                "primary": {"M": 4096, "topk": 8, "D": 4096},
                "validation": [{"M": 16384, "topk": 8, "D": 4096},
                               {"M": 8192, "topk": 4, "D": 4096},
                               {"M": 4095, "topk": 8, "D": 4096}]}
    if kind == "shared_expert":
        return {"minimal": {"M": 64, "D": 128, "I": 256},
                "primary": {"M": 4096, "D": 4096, "I": 14336},
                "validation": [{"M": 16384, "D": 4096, "I": 14336},
                               {"M": 8192, "D": 4096, "I": 14336},
                               {"M": 4095, "D": 4096, "I": 14335}]}
    if kind in ("fused_moe", "moe_block"):
        if pinE:
            E, tk = spec["E"], spec["topk"]
            return {"minimal": {"M": 64, "E": E, "topk": tk, "D": 128, "I": 256},
                    "primary": {"M": 4096, "E": E, "topk": tk, "D": 4096, "I": 14336},
                    "validation": [{"M": 16384, "E": E, "topk": tk, "D": 4096, "I": 14336},
                                   {"M": 8193, "E": E, "topk": tk, "D": 4096, "I": 14336}]}
        return {"minimal": {"M": 64, "E": 8, "topk": 2, "D": 128, "I": 256},
                "primary": {"M": 4096, "E": 64, "topk": 8, "D": 4096, "I": 14336},
                "validation": [{"M": 16384, "E": 128, "topk": 8, "D": 4096, "I": 14336},
                               {"M": 4096, "E": 256, "topk": 8, "D": 4096, "I": 14336},
                               {"M": 8193, "E": 64, "topk": 8, "D": 4096, "I": 14336}]}
    raise ValueError(f"no shape template for kind {kind!r}")


SHAPES: dict[str, dict] = {op: _make_shapes(op) for op in OPS}


# --------------------------------------------------------------------------- #
# Shared fp32 oracle helpers (self-contained; REUSE the _moe_common conventions).
# All accumulate in fp32; callers cast to the task dtype.
# --------------------------------------------------------------------------- #
def _act_fp32(x, act: str):
    import torch

    xf = x.float()
    if act == "silu":
        return xf * torch.sigmoid(xf)
    if act == "gelu":
        return torch.nn.functional.gelu(xf, approximate="tanh")
    raise ValueError(f"unknown activation {act!r}")


def _jagged_counts(E: int) -> list[int]:
    """Reproducible jagged per-expert count pattern of length E, last entry 0
    (the giant-expert + dead-last-expert imbalance). Mirrors _moe_common."""
    if E <= len(_TRACE32):
        c = list(_TRACE32[:E])
    else:
        tile = [x for x in _TRACE32 if x > 0]
        c = [tile[i % len(tile)] for i in range(E)]
    c[-1] = 0
    return c


def _make_routing(M, E, topk, device, g, renorm=True):
    """Unbalanced router assignment with a guaranteed 0-token last expert.

    Softmax over an unbalanced log-count bias + noise, top-k select, optional
    renorm. Returns (topk_weight [M,topk] fp32, topk_ids [M,topk] int32)."""
    import torch

    counts = torch.tensor([float(x) for x in _jagged_counts(E)],
                          dtype=torch.float32, device=device)
    bias = torch.log(counts + 1e-6)
    bias[counts == 0] = float("-inf")               # never select the dead expert
    gate = torch.randn((M, E), generator=g, device=device, dtype=torch.float32) + bias
    probs = torch.softmax(gate, dim=-1)
    tw, ti = torch.topk(probs, topk, dim=-1)
    if renorm:
        tw = tw / tw.sum(dim=-1, keepdim=True)
    return tw.to(torch.float32), ti.to(torch.int32)


def _scatter_dense(tw, ti, E):
    """Scatter (weights, ids) to a dense [M, E] fp32 map (order-independent)."""
    import torch

    dense = torch.zeros((tw.shape[0], E), device=tw.device, dtype=torch.float32)
    dense.scatter_(1, ti.long(), tw.float())
    return dense


def _quant_lastdim_fp8(x):
    """Symmetric per-last-dim-row fp8 (e4m3fn) quant. x[...,K] -> (xq fp8, s[...,1] fp32)."""
    import torch

    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=EPS)
    scale = (amax / FP8_MAX).to(torch.float32)
    xq = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return xq, scale


# ---- routers (fp32 dense [M,E]) -------------------------------------------- #
def _route_softmax_fp32(gate, topk, renorm):
    import torch

    sm = torch.softmax(gate.float(), dim=-1)
    tw, ti = torch.topk(sm, topk, dim=-1)
    if renorm:
        tw = tw / tw.sum(dim=-1, keepdim=True)
    return _scatter_dense(tw, ti, gate.shape[1])


def _route_sigmoid_fp32(gate, topk, renorm):
    import torch

    sc = torch.sigmoid(gate.float())
    tw, ti = torch.topk(sc, topk, dim=-1)
    if renorm:
        tw = tw / tw.sum(dim=-1, keepdim=True)
    return _scatter_dense(tw, ti, gate.shape[1])


def _route_topk_then_softmax_fp32(gate, topk):
    import torch

    tv, ti = torch.topk(gate.float(), topk, dim=-1)
    w = torch.softmax(tv, dim=-1)                          # softmax over ONLY the k
    return _scatter_dense(w, ti, gate.shape[1])


def _route_grouped_fp32(gate, topk, n_groups, topk_group, renorm):
    import torch

    M, E = gate.shape
    grp = E // n_groups
    sm = torch.softmax(gate.float(), dim=-1)
    gscore = sm.view(M, n_groups, grp).max(dim=-1).values          # [M,n_groups]
    keep = gscore.topk(topk_group, dim=-1).indices
    gmask = torch.zeros((M, n_groups), device=gate.device, dtype=torch.bool)
    gmask.scatter_(1, keep, True)
    emask = gmask.view(M, n_groups, 1).expand(M, n_groups, grp).reshape(M, E)
    masked = torch.where(emask, sm, torch.full_like(sm, float("-inf")))
    tw, ti = masked.topk(topk, dim=-1)
    if renorm:
        tw = tw / tw.sum(dim=-1, keepdim=True)
    return _scatter_dense(tw, ti, E)


def _route_biased_grouped_fp32(gate, bias, topk, n_groups, topk_group, renorm, scale=1.0):
    import torch

    M, E = gate.shape
    grp = E // n_groups
    scores = torch.sigmoid(gate.float())
    sb = scores + bias.float().view(1, E)
    gview = sb.view(M, n_groups, grp)
    top2 = gview.topk(min(2, grp), dim=-1).values.sum(dim=-1)      # [M,n_groups]
    keep = top2.topk(topk_group, dim=-1).indices
    gmask = torch.zeros((M, n_groups), device=gate.device, dtype=torch.bool)
    gmask.scatter_(1, keep, True)
    emask = gmask.view(M, n_groups, 1).expand(M, n_groups, grp).reshape(M, E)
    masked = torch.where(emask, sb, torch.full_like(sb, float("-inf")))
    ti = masked.topk(topk, dim=-1).indices
    tw = torch.gather(scores, 1, ti)                              # ORIGINAL sigmoid
    if renorm:
        tw = tw / tw.sum(dim=-1, keepdim=True).clamp(min=EPS)
    tw = tw * scale
    return _scatter_dense(tw, ti, E)


def _route_expert_choice_fp32(gate, cap):
    import torch

    M, E = gate.shape
    cap = min(int(cap), M)
    sm = torch.softmax(gate.float(), dim=-1)                       # [M,E]
    tv, ti = torch.topk(sm.t().contiguous(), cap, dim=-1)         # per expert over tokens
    dense = torch.zeros((M, E), device=gate.device, dtype=torch.float32)
    eidx = torch.arange(E, device=gate.device).view(E, 1).expand(E, cap)
    dense[ti.reshape(-1), eidx.reshape(-1)] = tv.reshape(-1)
    return dense


# ---- grouped / batched matmul (fp32) --------------------------------------- #
def _grouped_matmul_fp32(x, w, expert_ids):
    """Segmented per-expert GEMM: out[m] = x[m] @ w[expert_ids[m]].T (fp32).

    x [M,K] fp32, w [E,N,K] fp32, expert_ids [M] int -> out [M,N] fp32. Experts
    with no assigned tokens are simply never visited (the 0-token edge)."""
    import torch

    M, K = x.shape
    E, N, _ = w.shape
    out = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    eids = expert_ids.long()
    for e in range(E):
        idx = (eids == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        out[idx] = x[idx] @ w[e].t()
    return out


def _fused_moe_fp32(hidden, w1, w2, tw, ti, act):
    """Exact fp32 top-k fused-MoE MLP oracle -> fp32 [M, model_dim].

    Per token, for each selected expert e: gate_up = x @ w1[e].T -> split
    (gate,up) -> h = act(gate)*up -> y_e = h @ w2[e].T; y = sum_k tw*y_{e_k}.
    Weight layout (aiter): w1 [E,2I,D], w2 [E,D,I]. 0-token experts skipped."""
    import torch

    x = hidden.float()
    M, D = x.shape
    E = w1.shape[0]
    I = w2.shape[2]
    w1f, w2f = w1.float(), w2.float()
    out = torch.zeros((M, D), device=x.device, dtype=torch.float32)
    ids = ti.long()
    tw_f = tw.float()
    for e in range(E):
        mask = ids == e                              # [M, topk]
        tok = mask.any(dim=1)
        if not bool(tok.any()):
            continue
        idx = tok.nonzero(as_tuple=True)[0]
        xe = x[idx]
        gate_up = xe @ w1f[e].t()
        gate, up = gate_up[:, :I], gate_up[:, I:]
        h = _act_fp32(gate, act) * up
        ye = h @ w2f[e].t()
        w_e = (tw_f * mask.float()).sum(dim=1)[idx]
        out[idx] += ye * w_e[:, None]
    return out


# --------------------------------------------------------------------------- #
# reference.py namespace (exact fp32 oracle + torch eager baseline)
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import torch

    if op not in _SPECS:
        raise ValueError(f"unknown breadth MoE op {op!r}")
    spec = _SPECS[op]
    kind = spec["kind"]
    act = spec.get("act", "silu")
    renorm = spec.get("renorm", True)
    router = spec.get("router", "softmax")
    tdt = getattr(torch, DTYPES[dtype][0])

    def _randn(shape, device, seed, scale=1.0, dt=None):
        g = torch.Generator(device=device).manual_seed(seed)
        t = torch.randn(shape, generator=g, device=device, dtype=torch.float32) * scale
        return t.to(tdt if dt is None else dt)

    def _routing(M, E, topk, device, seed, rn=False):
        g = torch.Generator(device=device).manual_seed(seed)
        return _make_routing(M, E, topk, device, g, renorm=rn)

    # ===================================================== ROUTERS (dense [M,E]) =
    if kind == "route_softmax":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["E"]), device, seed), int(shape["topk"]))

        def ref_fn(gate, topk):
            return _route_softmax_fp32(gate, topk, renorm)
        baseline_fn, arity = ref_fn, 2

    elif kind == "route_sigmoid":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["E"]), device, seed), int(shape["topk"]))

        def ref_fn(gate, topk):
            return _route_sigmoid_fp32(gate, topk, renorm)
        baseline_fn, arity = ref_fn, 2

    elif kind == "route_topk_then_softmax":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["E"]), device, seed), int(shape["topk"]))

        def ref_fn(gate, topk):
            return _route_topk_then_softmax_fp32(gate, topk)
        baseline_fn, arity = ref_fn, 2

    elif kind == "route_grouped":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["E"]), device, seed),
                    int(shape["topk"]), int(shape["n_groups"]), int(shape["topk_group"]))

        def ref_fn(gate, topk, n_groups, topk_group):
            return _route_grouped_fp32(gate, topk, n_groups, topk_group, renorm)
        baseline_fn, arity = ref_fn, 4

    elif kind == "route_biased_grouped":
        def get_inputs(shape, device="cuda", seed=0):
            E = shape["E"]
            gate = _randn((shape["M"], E), device, seed)
            bias = _randn((E,), device, seed + 1, scale=0.1, dt=torch.float32)
            return (gate, bias, int(shape["topk"]),
                    int(shape["n_groups"]), int(shape["topk_group"]))

        def ref_fn(gate, bias, topk, n_groups, topk_group):
            return _route_biased_grouped_fp32(gate, bias, topk, n_groups, topk_group, renorm)
        baseline_fn, arity = ref_fn, 5

    elif kind == "route_expert_choice":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["E"]), device, seed), int(shape["cap"]))

        def ref_fn(gate, cap):
            return _route_expert_choice_fp32(gate, cap)
        baseline_fn, arity = ref_fn, 2

    # ============================================ PERMUTE / GATHER / HISTOGRAM ==
    elif kind == "permute":
        def get_inputs(shape, device="cuda", seed=0):
            M, E, D = shape["M"], shape["E"], shape["D"]
            hidden = _randn((M, D), device, seed)
            _, ti = _routing(M, E, 1, device, seed + 1)
            sort_idx = torch.argsort(ti[:, 0], stable=True).to(torch.int32)
            return (hidden, sort_idx)

        def ref_fn(hidden, sort_idx):
            return hidden[sort_idx.long()]
        baseline_fn, arity = ref_fn, 2

    elif kind == "unpermute":
        def get_inputs(shape, device="cuda", seed=0):
            M, E, D = shape["M"], shape["E"], shape["D"]
            hidden = _randn((M, D), device, seed)
            _, ti = _routing(M, E, 1, device, seed + 1)
            sort_idx = torch.argsort(ti[:, 0], stable=True).to(torch.int32)
            permuted = hidden[sort_idx.long()]
            return (permuted, sort_idx)

        def ref_fn(permuted, sort_idx):
            out = torch.empty_like(permuted)
            out[sort_idx.long()] = permuted
            return out
        baseline_fn, arity = ref_fn, 2

    elif kind == "permute_probs":
        def get_inputs(shape, device="cuda", seed=0):
            M, E, D = shape["M"], shape["E"], shape["D"]
            hidden = _randn((M, D), device, seed)
            probs = _randn((M,), device, seed + 2, dt=torch.float32).abs()
            _, ti = _routing(M, E, 1, device, seed + 1)
            sort_idx = torch.argsort(ti[:, 0], stable=True).to(torch.int32)
            return (hidden, probs, sort_idx)

        def ref_fn(hidden, probs, sort_idx):
            si = sort_idx.long()
            return hidden[si], probs.float()[si]
        baseline_fn, arity = ref_fn, 3

    elif kind == "histogram":
        def get_inputs(shape, device="cuda", seed=0):
            M, E = shape["M"], shape["E"]
            _, ti = _routing(M, E, 1, device, seed + 1)
            return (ti[:, 0].contiguous().to(torch.int32), int(E))

        def ref_fn(expert_ids, E):
            return torch.bincount(expert_ids.long(), minlength=E).to(torch.int32)
        baseline_fn, arity = ref_fn, 2

    elif kind == "offsets":
        def get_inputs(shape, device="cuda", seed=0):
            M, E = shape["M"], shape["E"]
            _, ti = _routing(M, E, 1, device, seed + 1)
            return (ti[:, 0].contiguous().to(torch.int32), int(E))

        def ref_fn(expert_ids, E):
            counts = torch.bincount(expert_ids.long(), minlength=E)
            off = torch.zeros(E + 1, dtype=torch.int64, device=expert_ids.device)
            off[1:] = torch.cumsum(counts, 0)
            return off.to(torch.int32)
        baseline_fn, arity = ref_fn, 2

    elif kind == "align_offsets":
        def get_inputs(shape, device="cuda", seed=0):
            M, E = shape["M"], shape["E"]
            _, ti = _routing(M, E, 1, device, seed + 1)
            return (ti[:, 0].contiguous().to(torch.int32), int(E), int(ALIGN_BLOCK))

        def ref_fn(expert_ids, E, block):
            counts = torch.bincount(expert_ids.long(), minlength=E)
            padded = ((counts + block - 1) // block) * block
            off = torch.zeros(E + 1, dtype=torch.int64, device=expert_ids.device)
            off[1:] = torch.cumsum(padded, 0)
            return off.to(torch.int32)
        baseline_fn, arity = ref_fn, 3

    # ================================================ EXPERT GEMM (grouped/batched)
    elif kind == "grouped_gemm":
        def get_inputs(shape, device="cuda", seed=0):
            M, E, N, K = shape["M"], shape["E"], shape["N"], shape["K"]
            sc = 1.0 / (K ** 0.5)
            hidden = _randn((M, K), device, seed, scale=sc)
            w = _randn((E, N, K), device, seed + 1, scale=sc)
            _, ti = _routing(M, E, 1, device, seed + 2)
            return (hidden, w, ti[:, 0].contiguous().to(torch.int32))

        def ref_fn(hidden, w, expert_ids):
            return _grouped_matmul_fp32(hidden.float(), w.float(), expert_ids).to(tdt)
        baseline_fn, arity = ref_fn, 3

    elif kind == "grouped_gemm_fp8":
        def get_inputs(shape, device="cuda", seed=0):
            M, E, N, K = shape["M"], shape["E"], shape["N"], shape["K"]
            xf = _randn((M, K), device, seed, dt=torch.float32)
            wf = _randn((E, N, K), device, seed + 1, scale=1.0 / (K ** 0.5), dt=torch.float32)
            xq, xs = _quant_lastdim_fp8(xf)
            wq, ws = _quant_lastdim_fp8(wf)
            _, ti = _routing(M, E, 1, device, seed + 2)
            return (xq, wq, xs, ws, ti[:, 0].contiguous().to(torch.int32))

        def ref_fn(xq, wq, xs, ws, expert_ids):
            x = xq.float() * xs.float()
            w = wq.float() * ws.float()
            return _grouped_matmul_fp32(x, w, expert_ids).to(torch.bfloat16)
        baseline_fn, arity = ref_fn, 5

    elif kind == "batched_gemm":
        def get_inputs(shape, device="cuda", seed=0):
            E, m, N, K = shape["E"], shape["m"], shape["N"], shape["K"]
            sc = 1.0 / (K ** 0.5)
            a = _randn((E, m, K), device, seed, scale=sc)
            b = _randn((E, N, K), device, seed + 1, scale=sc)
            return (a, b)

        def ref_fn(a, b):
            return torch.bmm(a.float(), b.float().transpose(1, 2)).to(tdt)
        baseline_fn, arity = ref_fn, 2

    elif kind == "batched_gemm_fp8":
        def get_inputs(shape, device="cuda", seed=0):
            E, m, N, K = shape["E"], shape["m"], shape["N"], shape["K"]
            af = _randn((E, m, K), device, seed, dt=torch.float32)
            bf = _randn((E, N, K), device, seed + 1, scale=1.0 / (K ** 0.5), dt=torch.float32)
            aq, as_ = _quant_lastdim_fp8(af)
            bq, bs = _quant_lastdim_fp8(bf)
            return (aq, bq, as_, bs)

        def ref_fn(aq, bq, as_, bs):
            a = aq.float() * as_.float()
            b = bq.float() * bs.float()
            return torch.bmm(a, b.transpose(1, 2)).to(torch.bfloat16)
        baseline_fn, arity = ref_fn, 4

    elif kind == "grouped_gate_up":
        def get_inputs(shape, device="cuda", seed=0):
            M, E, D, I = shape["M"], shape["E"], shape["D"], shape["I"]
            sc = 1.0 / (D ** 0.5)
            hidden = _randn((M, D), device, seed, scale=sc)
            w13 = _randn((E, 2 * I, D), device, seed + 1, scale=sc)
            _, ti = _routing(M, E, 1, device, seed + 2)
            return (hidden, w13, ti[:, 0].contiguous().to(torch.int32))

        def ref_fn(hidden, w13, expert_ids):
            return _grouped_matmul_fp32(hidden.float(), w13.float(), expert_ids).to(tdt)
        baseline_fn, arity = ref_fn, 3

    elif kind == "grouped_down":
        def get_inputs(shape, device="cuda", seed=0):
            M, E, D, I = shape["M"], shape["E"], shape["D"], shape["I"]
            sc = 1.0 / (I ** 0.5)
            hidden = _randn((M, I), device, seed, scale=sc)
            w2 = _randn((E, D, I), device, seed + 1, scale=sc)
            _, ti = _routing(M, E, 1, device, seed + 2)
            return (hidden, w2, ti[:, 0].contiguous().to(torch.int32))

        def ref_fn(hidden, w2, expert_ids):
            return _grouped_matmul_fp32(hidden.float(), w2.float(), expert_ids).to(tdt)
        baseline_fn, arity = ref_fn, 3

    elif kind in ("grouped_swiglu", "grouped_geglu"):
        def get_inputs(shape, device="cuda", seed=0):
            M, E, D, I = shape["M"], shape["E"], shape["D"], shape["I"]
            sc = 1.0 / (D ** 0.5)
            hidden = _randn((M, D), device, seed, scale=sc)
            w13 = _randn((E, 2 * I, D), device, seed + 1, scale=sc)
            _, ti = _routing(M, E, 1, device, seed + 2)
            return (hidden, w13, ti[:, 0].contiguous().to(torch.int32))

        def ref_fn(hidden, w13, expert_ids):
            gu = _grouped_matmul_fp32(hidden.float(), w13.float(), expert_ids)
            I = gu.shape[1] // 2
            h = _act_fp32(gu[:, :I], act) * gu[:, I:]
            return h.to(tdt)
        baseline_fn, arity = ref_fn, 3

    elif kind == "grouped_mlp":
        def get_inputs(shape, device="cuda", seed=0):
            M, E, D, I = shape["M"], shape["E"], shape["D"], shape["I"]
            sc = 1.0 / (D ** 0.5)
            hidden = _randn((M, D), device, seed, scale=sc)
            w13 = _randn((E, 2 * I, D), device, seed + 1, scale=sc)
            w2 = _randn((E, D, I), device, seed + 2, scale=1.0 / (I ** 0.5))
            _, ti = _routing(M, E, 1, device, seed + 3)
            return (hidden, w13, w2, ti[:, 0].contiguous().to(torch.int32))

        def ref_fn(hidden, w13, w2, expert_ids):
            gu = _grouped_matmul_fp32(hidden.float(), w13.float(), expert_ids)
            I = gu.shape[1] // 2
            h = _act_fp32(gu[:, :I], act) * gu[:, I:]
            y = _grouped_matmul_fp32(h, w2.float(), expert_ids)
            return y.to(tdt)
        baseline_fn, arity = ref_fn, 4

    elif kind == "grouped_mlp_fp8":
        def get_inputs(shape, device="cuda", seed=0):
            M, E, D, I = shape["M"], shape["E"], shape["D"], shape["I"]
            sc = 1.0 / (D ** 0.5)
            xf = _randn((M, D), device, seed, scale=sc, dt=torch.float32)
            w13f = _randn((E, 2 * I, D), device, seed + 1, scale=sc, dt=torch.float32)
            w2f = _randn((E, D, I), device, seed + 2, scale=1.0 / (I ** 0.5), dt=torch.float32)
            xq, xs = _quant_lastdim_fp8(xf)
            w13q, w13s = _quant_lastdim_fp8(w13f)
            w2q, w2s = _quant_lastdim_fp8(w2f)
            _, ti = _routing(M, E, 1, device, seed + 3)
            return (xq, w13q, w2q, xs, w13s, w2s, ti[:, 0].contiguous().to(torch.int32))

        def ref_fn(xq, w13q, w2q, xs, w13s, w2s, expert_ids):
            x = xq.float() * xs.float()
            w13 = w13q.float() * w13s.float()
            w2 = w2q.float() * w2s.float()
            gu = _grouped_matmul_fp32(x, w13, expert_ids)
            I = gu.shape[1] // 2
            h = _act_fp32(gu[:, :I], act) * gu[:, I:]
            y = _grouped_matmul_fp32(h, w2, expert_ids)
            return y.to(torch.bfloat16)
        baseline_fn, arity = ref_fn, 7

    # ================================================================= COMBINE ==
    elif kind == "sum_combine":
        def get_inputs(shape, device="cuda", seed=0):
            M, topk, D = shape["M"], shape["topk"], shape["D"]
            y = _randn((M, topk, D), device, seed, scale=1.0 / (D ** 0.5))
            g = torch.Generator(device=device).manual_seed(seed + 1)
            w = torch.rand((M, topk), generator=g, device=device, dtype=torch.float32) + 1e-3
            w = (w / w.sum(dim=-1, keepdim=True)).to(torch.float32)
            return (y, w)

        def ref_fn(y, tw):
            return (y.float() * tw.float().unsqueeze(-1)).sum(dim=1).to(tdt)
        baseline_fn, arity = ref_fn, 2

    elif kind == "finalize":
        def get_inputs(shape, device="cuda", seed=0):
            M, topk, D = shape["M"], shape["topk"], shape["D"]
            P = M * topk
            y_perm = _randn((P, D), device, seed, scale=1.0 / (D ** 0.5))
            g = torch.Generator(device=device).manual_seed(seed + 1)
            row_map = torch.randperm(P, generator=g, device=device).reshape(M, topk).to(torch.int32)
            w = torch.rand((M, topk), generator=g, device=device, dtype=torch.float32) + 1e-3
            w = (w / w.sum(dim=-1, keepdim=True)).to(torch.float32)
            return (y_perm, row_map, w)

        def ref_fn(y_perm, row_map, tw):
            yg = y_perm.float()[row_map.long()]              # [M,topk,D]
            return (yg * tw.float().unsqueeze(-1)).sum(dim=1).to(tdt)
        baseline_fn, arity = ref_fn, 3

    elif kind == "shared_expert":
        def get_inputs(shape, device="cuda", seed=0):
            M, D, I = shape["M"], shape["D"], shape["I"]
            sc = 1.0 / (D ** 0.5)
            hidden = _randn((M, D), device, seed, scale=sc)
            ws1 = _randn((2 * I, D), device, seed + 1, scale=sc)
            ws2 = _randn((D, I), device, seed + 2, scale=1.0 / (I ** 0.5))
            routed = _randn((M, D), device, seed + 3, scale=sc)
            return (hidden, ws1, ws2, routed)

        def ref_fn(hidden, ws1, ws2, routed):
            x = hidden.float()
            gu = x @ ws1.float().t()
            I = gu.shape[1] // 2
            h = _act_fp32(gu[:, :I], act) * gu[:, I:]
            y = h @ ws2.float().t()
            return (routed.float() + y).to(tdt)
        baseline_fn, arity = ref_fn, 4

    # ============================================ FUSED top-k MoE / end-to-end ==
    elif kind == "fused_moe":
        def get_inputs(shape, device="cuda", seed=0):
            M, E, topk = shape["M"], shape["E"], shape["topk"]
            D, I = shape["D"], shape["I"]
            hidden = _randn((M, D), device, seed)
            w1 = _randn((E, 2 * I, D), device, seed + 1, scale=0.05)
            w2 = _randn((E, D, I), device, seed + 2, scale=0.05)
            tw, ti = _routing(M, E, topk, device, seed + 3, rn=True)
            return (hidden, w1, w2, tw, ti)

        def ref_fn(hidden, w1, w2, tw, ti):
            return _fused_moe_fp32(hidden, w1, w2, tw, ti, act).to(tdt)
        baseline_fn, arity = ref_fn, 5

    elif kind == "moe_block":
        def get_inputs(shape, device="cuda", seed=0):
            M, E, topk = shape["M"], shape["E"], shape["topk"]
            D, I = shape["D"], shape["I"]
            hidden = _randn((M, D), device, seed)
            gate = _randn((M, E), device, seed + 4)
            w1 = _randn((E, 2 * I, D), device, seed + 1, scale=0.05)
            w2 = _randn((E, D, I), device, seed + 2, scale=0.05)
            return (hidden, gate, w1, w2, int(topk))

        def ref_fn(hidden, gate, w1, w2, topk):
            if router == "sigmoid":
                sc = torch.sigmoid(gate.float())
            else:
                sc = torch.softmax(gate.float(), dim=-1)
            tw, ti = torch.topk(sc, topk, dim=-1)
            tw = tw / tw.sum(dim=-1, keepdim=True)
            return _fused_moe_fp32(hidden, w1, w2, tw, ti.to(torch.int32), act).to(tdt)
        baseline_fn, arity = ref_fn, 5

    else:
        raise ValueError(f"unknown MoE kind {kind!r} for op {op!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op,
          "dtype_name": dtype, "family": f"breadth_{op}", "mutates_input": False}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Naive (correct, compiling) Triton seeds - the policy's starting point.
# Each seed does host-side routing/permute selection (torch) with a Triton kernel
# for the dominant primitive (grouped/batched GEMM, dense scatter, gather, weighted
# combine, histogram/scan) - an honest naive bar the policy fuses into one kernel.
# All seeds are valid Python that define ``def <op>(*inputs)`` (static-checked on
# CPU); the Triton kernels are correct for the real (GPU) shapes.
# --------------------------------------------------------------------------- #
_SEED_HEADER = '''"""GENERATED breadth MoE seed: {op} ({dtype}).

{desc}. Naive, COMPILING, CORRECT starting point: host-side routing/permute
selection (torch) with a Triton kernel for the dominant primitive. The policy is
expected to fuse the routing + grouped GEMM + activation + combine into one kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

'''

# ---- Triton matmul (C = A @ B^T, B stored [N,K]) + grouped host driver ----- #
_MM_BLOCK = '''

@triton.jit
def _mm_nt_kernel(a_ptr, b_ptr, c_ptr, Mr, N, K,
                  sam, sak, sbn, sbk, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)
    a_ptrs = a_ptr + offm[:, None] * sam + offk[None, :] * sak
    b_ptrs = b_ptr + offn[:, None] * sbn + offk[None, :] * sbk
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        km = offk[None, :] < (K - k0)
        a = tl.load(a_ptrs, mask=(offm[:, None] < Mr) & km, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(offn[:, None] < N) & km, other=0.0).to(tl.float32)
        acc += tl.dot(a, tl.trans(b))
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    cmask = (offm[:, None] < Mr) & (offn[None, :] < N)
    tl.store(c_ptr + offm[:, None] * scm + offn[None, :] * scn,
             acc.to(c_ptr.dtype.element_ty), mask=cmask)


def _mm_nt(a, b):
    """a [m,K], b [N,K] -> a @ b.T (fp32 accumulate); out dtype = a.dtype."""
    m, K = a.shape
    N = b.shape[0]
    c = torch.empty((m, N), device=a.device, dtype=a.dtype)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(m, BM), triton.cdiv(N, BN))
    _mm_nt_kernel[grid](a, b, c, m, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK)
    return c


def _grouped_mm(x, w, expert_ids):
    """Per-expert grouped GEMM: out[m] = x[m] @ w[expert_ids[m]].T (naive: one GEMM
    launch per non-empty expert -- the bar a fused variable-M grouped kernel beats)."""
    M, K = x.shape
    E, N, _ = w.shape
    out = torch.zeros((M, N), device=x.device, dtype=x.dtype)
    eids = expert_ids.to(torch.long)
    for e in range(E):
        idx = (eids == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        ye = _mm_nt(x.index_select(0, idx).contiguous(), w[e].contiguous())
        out.index_copy_(0, idx, ye)
    return out
'''

_SWIGLU_BLOCK = '''

@triton.jit
def _gated_act_kernel(gu_ptr, out_ptr, I, sgm, sgi, som, soi,
                      GELU: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < I
    gate = tl.load(gu_ptr + row * sgm + offs * sgi,
                   mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gu_ptr + row * sgm + (I + offs) * sgi,
                 mask=mask, other=0.0).to(tl.float32)
    if GELU:
        z = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        act = 0.5 * gate * (1.0 + (2.0 * tl.sigmoid(2.0 * z) - 1.0))
    else:
        act = gate * tl.sigmoid(gate)
    tl.store(out_ptr + row * som + offs * soi,
             (act * up).to(out_ptr.dtype.element_ty), mask=mask)


def _swiglu(gu, gelu):
    """Triton gated activation on [M, 2I] -> [M, I]."""
    M = gu.shape[0]
    I = gu.shape[1] // 2
    out = torch.empty((M, I), device=gu.device, dtype=gu.dtype)
    BLOCK = 256
    _gated_act_kernel[(M, triton.cdiv(I, BLOCK))](
        gu, out, I, gu.stride(0), gu.stride(1), out.stride(0), out.stride(1),
        GELU=gelu, BLOCK=BLOCK)
    return out
'''

_FUSED_RUN_BLOCK = '''

def _fused_run(hidden, w1, w2, tw, ti):
    """Naive top-k fused MoE MLP: per non-empty expert gather its tokens, run the
    gate/up GEMM -> gated activation -> down GEMM, then weighted-combine over top-k."""
    M, D = hidden.shape
    E = w1.shape[0]
    ids = ti.to(torch.long)
    twf = tw.float()
    out = torch.zeros((M, D), device=hidden.device, dtype=torch.float32)
    for e in range(E):
        mask = ids == e
        tok = mask.any(dim=1)
        if not bool(tok.any()):
            continue
        idx = tok.nonzero(as_tuple=True)[0]
        gu = _mm_nt(hidden.index_select(0, idx).contiguous(), w1[e].contiguous())
        h = _swiglu(gu, {gelu}).to(hidden.dtype)
        ye = _mm_nt(h, w2[e].contiguous()).float()
        we = (twf * mask.float()).sum(dim=1)[idx]
        out.index_add_(0, idx, ye * we[:, None])
    return out.to(hidden.dtype)
'''

_SCATTER_BLOCK = '''

@triton.jit
def _scatter_dense_kernel(w_ptr, id_ptr, out_ptr, topk, sw, sid, so, TB: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, TB)
    km = k < topk
    ids = tl.load(id_ptr + row * sid + k, mask=km, other=0).to(tl.int64)
    ws = tl.load(w_ptr + row * sw + k, mask=km, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * so + ids, ws, mask=km)


def _scatter(tw, ti, M, E):
    """Scatter top-k (weights, ids) into a dense [M, E] fp32 routing map."""
    out = torch.zeros((M, E), device=tw.device, dtype=torch.float32)
    tw = tw.contiguous().float()
    ti = ti.contiguous().to(torch.int32)
    topk = tw.shape[1]
    _scatter_dense_kernel[(M,)](tw, ti, out, topk, tw.stride(0), ti.stride(0),
                                out.stride(0), TB=triton.next_power_of_2(topk))
    return out
'''

_ROUTE_BLOCK = '''

@triton.jit
def _route_topk_kernel(gate_ptr, dense_ptr, tw_ptr, ti_ptr, E, sgm, sge,
                       TOPK: tl.constexpr, SOFTMAX: tl.constexpr,
                       SIGMOID: tl.constexpr, TOPK_SOFTMAX: tl.constexpr,
                       RENORM: tl.constexpr, EB: tl.constexpr):
    row = tl.program_id(0)
    e = tl.arange(0, EB)
    mask = e < E
    raw = tl.load(gate_ptr + row * sgm + e * sge,
                  mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(raw, axis=0)
    if SOFTMAX:
        ex = tl.exp(raw - row_max)
        scores = ex / tl.sum(tl.where(mask, ex, 0.0), axis=0)
    elif SIGMOID:
        scores = tl.sigmoid(raw)
    else:
        scores = raw
    candidates = tl.where(mask, scores, -float("inf"))
    tl.store(dense_ptr + row * E + e, 0.0, mask=mask)
    total = 0.0
    for j in range(0, TOPK):
        pick = tl.argmax(candidates, axis=0)
        picked = tl.max(candidates, axis=0)
        if TOPK_SOFTMAX:
            value = tl.exp(picked - row_max)
        else:
            value = picked
        total += value
        tl.store(dense_ptr + row * E + pick, value)
        tl.store(tw_ptr + row * TOPK + j, value)
        tl.store(ti_ptr + row * TOPK + j, pick)
        candidates = tl.where(e == pick, -float("inf"), candidates)
    if RENORM or TOPK_SOFTMAX:
        vals = tl.load(dense_ptr + row * E + e, mask=mask, other=0.0)
        vals = tl.where(vals != 0.0, vals / total, 0.0)
        tl.store(dense_ptr + row * E + e, vals, mask=mask)
        for j in range(0, TOPK):
            value = tl.load(tw_ptr + row * TOPK + j)
            tl.store(tw_ptr + row * TOPK + j, value / total)


def _route_topk(gate, topk, mode, renorm):
    gate = gate.contiguous()
    M, E = gate.shape
    dense = torch.zeros((M, E), device=gate.device, dtype=torch.float32)
    tw = torch.empty((M, topk), device=gate.device, dtype=torch.float32)
    ti = torch.empty((M, topk), device=gate.device, dtype=torch.int32)
    EB = triton.next_power_of_2(E)
    _route_topk_kernel[(M,)](
        gate, dense, tw, ti, E, gate.stride(0), gate.stride(1),
        TOPK=topk, SOFTMAX=mode == "softmax", SIGMOID=mode == "sigmoid",
        TOPK_SOFTMAX=mode == "topk_softmax", RENORM=renorm, EB=EB)
    return dense, tw, ti
'''

_GROUP_ROUTE_BLOCK = '''

@triton.jit
def _group_score_kernel(gate_ptr, bias_ptr, score_ptr, E, n_groups, group_size,
                        sgm, sge, BIASED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    k = tl.arange(0, BLOCK)
    mask = k < group_size
    e = group * group_size + k
    raw = tl.load(gate_ptr + row * sgm + e * sge,
                  mask=mask, other=-float("inf")).to(tl.float32)
    if BIASED:
        bias = tl.load(bias_ptr + e, mask=mask, other=0.0).to(tl.float32)
        scores = tl.sigmoid(raw) + bias
        first_idx = tl.argmax(scores, axis=0)
        first = tl.max(scores, axis=0)
        second = tl.max(tl.where(k == first_idx, -float("inf"), scores), axis=0)
        group_score = first + tl.where(group_size > 1, second, 0.0)
    else:
        group_score = tl.max(raw, axis=0)
    tl.store(score_ptr + row * n_groups + group, group_score)


@triton.jit
def _route_grouped_kernel(gate_ptr, bias_ptr, group_ptr, dense_ptr,
                          E, n_groups, group_size, sgm, sge,
                          TOPK: tl.constexpr, TOPK_GROUP: tl.constexpr,
                          BIASED: tl.constexpr, RENORM: tl.constexpr,
                          EB: tl.constexpr, GB: tl.constexpr):
    row = tl.program_id(0)
    e = tl.arange(0, EB)
    emask = e < E
    raw = tl.load(gate_ptr + row * sgm + e * sge,
                  mask=emask, other=-float("inf")).to(tl.float32)
    if BIASED:
        bias = tl.load(bias_ptr + e, mask=emask, other=0.0).to(tl.float32)
        weights = tl.sigmoid(raw)
        select_scores = weights + bias
    else:
        row_max = tl.max(raw, axis=0)
        ex = tl.exp(raw - row_max)
        weights = ex / tl.sum(tl.where(emask, ex, 0.0), axis=0)
        select_scores = weights

    groups = e // group_size
    goffs = tl.arange(0, GB)
    gmask = goffs < n_groups
    group_scores = tl.load(group_ptr + row * n_groups + goffs,
                           mask=gmask, other=-float("inf"))
    allowed = e < 0
    for j in range(0, TOPK_GROUP):
        picked_group = tl.argmax(group_scores, axis=0)
        allowed = allowed | (groups == picked_group)
        group_scores = tl.where(goffs == picked_group, -float("inf"), group_scores)

    candidates = tl.where(emask & allowed, select_scores, -float("inf"))
    tl.store(dense_ptr + row * E + e, 0.0, mask=emask)
    total = 0.0
    for j in range(0, TOPK):
        pick = tl.argmax(candidates, axis=0)
        value = tl.sum(tl.where(e == pick, weights, 0.0), axis=0)
        total += value
        tl.store(dense_ptr + row * E + pick, value)
        candidates = tl.where(e == pick, -float("inf"), candidates)
    if RENORM:
        vals = tl.load(dense_ptr + row * E + e, mask=emask, other=0.0)
        tl.store(dense_ptr + row * E + e,
                 tl.where(vals != 0.0, vals / tl.maximum(total, 1.0e-12), 0.0),
                 mask=emask)


def _route_grouped(gate, bias, topk, n_groups, topk_group, biased, renorm):
    gate = gate.contiguous()
    if biased:
        bias = bias.contiguous()
    M, E = gate.shape
    group_size = E // n_groups
    group_scores = torch.empty((M, n_groups), device=gate.device, dtype=torch.float32)
    dense = torch.zeros((M, E), device=gate.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(group_size)
    _group_score_kernel[(M, n_groups)](
        gate, bias if biased else gate, group_scores, E, n_groups, group_size,
        gate.stride(0), gate.stride(1), BIASED=biased, BLOCK=BLOCK)
    _route_grouped_kernel[(M,)](
        gate, bias if biased else gate, group_scores, dense,
        E, n_groups, group_size, gate.stride(0), gate.stride(1),
        TOPK=topk, TOPK_GROUP=topk_group, BIASED=biased, RENORM=renorm,
        EB=triton.next_power_of_2(E), GB=triton.next_power_of_2(n_groups))
    return dense
'''

_EC_BLOCK = '''

@triton.jit
def _row_softmax_kernel(gate_ptr, work_ptr, E, sgm, sge, EB: tl.constexpr):
    row = tl.program_id(0)
    e = tl.arange(0, EB)
    mask = e < E
    raw = tl.load(gate_ptr + row * sgm + e * sge,
                  mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(raw, axis=0)
    ex = tl.exp(raw - row_max)
    probs = ex / tl.sum(tl.where(mask, ex, 0.0), axis=0)
    tl.store(work_ptr + row * E + e, probs, mask=mask)


@triton.jit
def _expert_choice_kernel(work_ptr, out_ptr, M, E, cap, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    for c in range(0, cap):
        best = -float("inf")
        best_token = 0
        for start in range(0, M, BLOCK):
            tok = start + tl.arange(0, BLOCK)
            mask = tok < M
            values = tl.load(work_ptr + tok * E + expert,
                             mask=mask, other=-float("inf"))
            block_best = tl.max(values, axis=0)
            block_idx = tl.argmax(values, axis=0) + start
            take = block_best > best
            best = tl.where(take, block_best, best)
            best_token = tl.where(take, block_idx, best_token)
        tl.store(out_ptr + best_token * E + expert, best)
        tl.store(work_ptr + best_token * E + expert, -float("inf"))


def _expert_choice(gate, cap):
    gate = gate.contiguous()
    M, E = gate.shape
    cap = min(int(cap), M)
    work = torch.empty((M, E), device=gate.device, dtype=torch.float32)
    out = torch.zeros((M, E), device=gate.device, dtype=torch.float32)
    _row_softmax_kernel[(M,)](
        gate, work, E, gate.stride(0), gate.stride(1),
        EB=triton.next_power_of_2(E))
    _expert_choice_kernel[(E,)](work, out, M, E, cap, BLOCK=256)
    return out
'''

_GATHER_BLOCK = '''

@triton.jit
def _gather_kernel(src_ptr, idx_ptr, dst_ptr, D, ss, sd, BD: tl.constexpr):
    row = tl.program_id(0)
    src_row = tl.load(idx_ptr + row).to(tl.int64)
    for d0 in range(0, D, BD):
        off = d0 + tl.arange(0, BD)
        m = off < D
        v = tl.load(src_ptr + src_row * ss + off, mask=m)
        tl.store(dst_ptr + row * sd + off, v, mask=m)
'''

_HIST_BLOCK = '''

@triton.jit
def _hist_kernel(ids_ptr, cnt_ptr, M, BM: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BM + tl.arange(0, BM)
    m = off < M
    e = tl.load(ids_ptr + off, mask=m, other=0).to(tl.int32)
    tl.atomic_add(cnt_ptr + e, tl.where(m, 1, 0), mask=m)
'''

_CUMSUM_BLOCK = '''

@triton.jit
def _excl_cumsum_kernel(cnt_ptr, off_ptr, E):
    acc = tl.zeros([], dtype=tl.int32)
    tl.store(off_ptr + 0, acc)
    for e in range(0, E):
        c = tl.load(cnt_ptr + e).to(tl.int32)
        acc = acc + c
        tl.store(off_ptr + e + 1, acc)
'''

_COMBINE_BLOCK = '''

@triton.jit
def _combine_kernel(y_ptr, w_ptr, out_ptr, topk, D,
                    sy0, sy1, sy2, sw0, so0, so1, BD: tl.constexpr):
    row = tl.program_id(0)
    for d0 in range(0, D, BD):
        off = d0 + tl.arange(0, BD)
        m = off < D
        acc = tl.zeros([BD], dtype=tl.float32)
        for k in range(0, topk):
            wv = tl.load(w_ptr + row * sw0 + k).to(tl.float32)
            yv = tl.load(y_ptr + row * sy0 + k * sy1 + off * sy2, mask=m, other=0.0).to(tl.float32)
            acc += wv * yv
        tl.store(out_ptr + row * so0 + off * so1, acc.to(out_ptr.dtype.element_ty), mask=m)
'''

_DESC = {
    "route_softmax": "top-k softmax MoE router -> dense [M,E] routing weights",
    "route_sigmoid": "sigmoid top-k MoE router -> dense [M,E] routing weights",
    "route_topk_then_softmax": "top-k-then-softmax MoE router -> dense [M,E]",
    "route_grouped": "grouped top-k MoE router (group-limited) -> dense [M,E]",
    "route_biased_grouped": "DeepSeek-V3 biased grouped top-k router -> dense [M,E]",
    "route_expert_choice": "expert-choice router (each expert picks top-C tokens)",
    "permute": "MoE dispatch permute (gather tokens into expert-sorted order)",
    "unpermute": "MoE un-permute (scatter tokens back to original order)",
    "permute_probs": "MoE permute of tokens AND their router probs together",
    "histogram": "per-expert token histogram (routing counts)",
    "offsets": "per-expert exclusive-scan offsets into the sorted token buffer",
    "align_offsets": "block-aligned per-expert offsets (moe_align_block_size)",
    "grouped_gemm": "variable-M grouped (segmented) expert GEMM",
    "grouped_gemm_fp8": "fp8 a8w8 variable-M grouped expert GEMM",
    "batched_gemm": "balanced batched expert GEMM C[e]=A[e]@B[e]^T",
    "batched_gemm_fp8": "fp8 a8w8 batched expert GEMM",
    "grouped_gate_up": "grouped gate/up projection (w1 stage) of the expert MLP",
    "grouped_down": "grouped down projection (w2 stage) of the expert MLP",
    "grouped_swiglu": "grouped gate/up GEMM fused with a SwiGLU activation",
    "grouped_geglu": "grouped gate/up GEMM fused with a GeGLU activation",
    "grouped_mlp": "full grouped expert MLP (gate/up -> gated act -> down)",
    "grouped_mlp_fp8": "fp8 full grouped expert MLP",
    "sum_combine": "top-k weighted combine (moe_sum reduce)",
    "finalize": "MoE finalize: gather from the permuted buffer + weighted combine",
    "shared_expert": "shared-expert MLP added to the routed-expert output",
    "fused_moe": "end-to-end fused top-k MoE MLP (route ids/weights given)",
    "moe_block": "end-to-end MoE block from router logits (route -> MLP -> combine)",
}

# Compile-time activation selector used by the Triton gated-activation kernel.
_ACT_IS_GELU = {"silu": "False", "gelu": "True"}


def seed_source(op: str, dtype: str) -> str:
    if op not in _SPECS:
        raise ValueError(f"unknown breadth MoE op {op!r}")
    kind = KIND[op]
    spec = _SPECS[op]
    act = spec.get("act", "silu")
    router = spec.get("router", "softmax")
    renorm = spec.get("renorm", True)
    header = _SEED_HEADER.format(op=op, dtype=dtype, desc=_DESC.get(kind, "MoE op"))
    swiglu = _SWIGLU_BLOCK
    fused_run = _FUSED_RUN_BLOCK.format(gelu=_ACT_IS_GELU[act])

    # ------------------------------------------------------------- ROUTERS ----
    if kind in ("route_softmax", "route_sigmoid"):
        mode = "softmax" if kind == "route_softmax" else "sigmoid"
        entry = (f"def {op}(gate, topk):\n"
                 f"    dense, _, _ = _route_topk(gate, topk, {mode!r}, {renorm!r})\n"
                 f"    return dense\n")
        return header + _ROUTE_BLOCK + "\n" + entry
    if kind == "route_topk_then_softmax":
        entry = (f"def {op}(gate, topk):\n"
                 f"    dense, _, _ = _route_topk(gate, topk, 'topk_softmax', True)\n"
                 f"    return dense\n")
        return header + _ROUTE_BLOCK + "\n" + entry
    if kind == "route_grouped":
        entry = (f"def {op}(gate, topk, n_groups, topk_group):\n"
                 f"    return _route_grouped(gate, gate, topk, n_groups, topk_group,\n"
                 f"                          False, {renorm!r})\n")
        return header + _GROUP_ROUTE_BLOCK + "\n" + entry
    if kind == "route_biased_grouped":
        entry = (f"def {op}(gate, bias, topk, n_groups, topk_group):\n"
                 f"    return _route_grouped(gate, bias, topk, n_groups, topk_group,\n"
                 f"                          True, {renorm!r})\n")
        return header + _GROUP_ROUTE_BLOCK + "\n" + entry
    if kind == "route_expert_choice":
        entry = (f"def {op}(gate, cap):\n"
                 f"    return _expert_choice(gate, cap)\n")
        return header + _EC_BLOCK + "\n" + entry

    # ------------------------------------------------ PERMUTE / GATHER --------
    if kind == "permute":
        entry = (f"def {op}(hidden, sort_idx):\n"
                 f"    M, D = hidden.shape\n"
                 f"    idx = sort_idx.to(torch.int64).contiguous()\n"
                 f"    out = torch.empty_like(hidden)\n"
                 f"    _gather_kernel[(M,)](hidden, idx, out, D, hidden.stride(0), out.stride(0), BD=256)\n"
                 f"    return out\n")
        return header + _GATHER_BLOCK + "\n" + entry
    if kind == "unpermute":
        entry = (f"def {op}(permuted, sort_idx):\n"
                 f"    M, D = permuted.shape\n"
                 f"    inv = torch.argsort(sort_idx.to(torch.long)).to(torch.int64).contiguous()\n"
                 f"    out = torch.empty_like(permuted)\n"
                 f"    _gather_kernel[(M,)](permuted, inv, out, D, permuted.stride(0), out.stride(0), BD=256)\n"
                 f"    return out\n")
        return header + _GATHER_BLOCK + "\n" + entry
    if kind == "permute_probs":
        entry = (f"def {op}(hidden, probs, sort_idx):\n"
                 f"    M, D = hidden.shape\n"
                 f"    idx = sort_idx.to(torch.int64).contiguous()\n"
                 f"    out = torch.empty_like(hidden)\n"
                 f"    _gather_kernel[(M,)](hidden, idx, out, D, hidden.stride(0), out.stride(0), BD=256)\n"
                 f"    probs_out = probs.float().index_select(0, idx)\n"
                 f"    return out, probs_out\n")
        return header + _GATHER_BLOCK + "\n" + entry

    # --------------------------------------------- HISTOGRAM / OFFSETS --------
    if kind == "histogram":
        entry = (f"def {op}(expert_ids, E):\n"
                 f"    M = expert_ids.shape[0]\n"
                 f"    cnt = torch.zeros((E,), device=expert_ids.device, dtype=torch.int32)\n"
                 f"    ids = expert_ids.to(torch.int32).contiguous()\n"
                 f"    BM = 256\n"
                 f"    _hist_kernel[(triton.cdiv(M, BM),)](ids, cnt, M, BM=BM)\n"
                 f"    return cnt\n")
        return header + _HIST_BLOCK + "\n" + entry
    if kind == "offsets":
        entry = (f"def {op}(expert_ids, E):\n"
                 f"    cnt = torch.bincount(expert_ids.to(torch.long), minlength=E).to(torch.int32)\n"
                 f"    off = torch.zeros((E + 1,), device=expert_ids.device, dtype=torch.int32)\n"
                 f"    _excl_cumsum_kernel[(1,)](cnt, off, E)\n"
                 f"    return off\n")
        return header + _CUMSUM_BLOCK + "\n" + entry
    if kind == "align_offsets":
        entry = (f"def {op}(expert_ids, E, block):\n"
                 f"    cnt = torch.bincount(expert_ids.to(torch.long), minlength=E)\n"
                 f"    padded = (((cnt + block - 1) // block) * block).to(torch.int32)\n"
                 f"    off = torch.zeros((E + 1,), device=expert_ids.device, dtype=torch.int32)\n"
                 f"    _excl_cumsum_kernel[(1,)](padded, off, E)\n"
                 f"    return off\n")
        return header + _CUMSUM_BLOCK + "\n" + entry

    # ----------------------------------------------------- EXPERT GEMM --------
    if kind in ("grouped_gemm", "grouped_gate_up", "grouped_down"):
        entry = (f"def {op}(hidden, w, expert_ids):\n"
                 f"    return _grouped_mm(hidden, w, expert_ids)\n")
        return header + _MM_BLOCK + "\n" + entry
    if kind == "grouped_gemm_fp8":
        entry = (f"def {op}(xq, wq, xs, ws, expert_ids):\n"
                 f"    x = (xq.float() * xs.float()).to(torch.bfloat16)\n"
                 f"    w = (wq.float() * ws.float()).to(torch.bfloat16)\n"
                 f"    return _grouped_mm(x, w, expert_ids)\n")
        return header + _MM_BLOCK + "\n" + entry
    if kind == "batched_gemm":
        entry = (f"def {op}(a, b):\n"
                 f"    E, m, K = a.shape\n"
                 f"    N = b.shape[1]\n"
                 f"    out = torch.empty((E, m, N), device=a.device, dtype=a.dtype)\n"
                 f"    for e in range(E):\n"
                 f"        out[e] = _mm_nt(a[e].contiguous(), b[e].contiguous())\n"
                 f"    return out\n")
        return header + _MM_BLOCK + "\n" + entry
    if kind == "batched_gemm_fp8":
        entry = (f"def {op}(aq, bq, as_, bs):\n"
                 f"    E, m, K = aq.shape\n"
                 f"    N = bq.shape[1]\n"
                 f"    out = torch.empty((E, m, N), device=aq.device, dtype=torch.bfloat16)\n"
                 f"    for e in range(E):\n"
                 f"        a = (aq[e].float() * as_[e].float()).to(torch.bfloat16)\n"
                 f"        b = (bq[e].float() * bs[e].float()).to(torch.bfloat16)\n"
                 f"        out[e] = _mm_nt(a, b)\n"
                 f"    return out\n")
        return header + _MM_BLOCK + "\n" + entry

    # ------------------------------------------- FUSED-ACTIVATION MLP ---------
    if kind in ("grouped_swiglu", "grouped_geglu"):
        entry = (f"def {op}(hidden, w13, expert_ids):\n"
                 f"    gu = _grouped_mm(hidden, w13, expert_ids)\n"
                 f"    return _swiglu(gu, {_ACT_IS_GELU[act]}).to(hidden.dtype)\n")
        return header + _MM_BLOCK + swiglu + "\n" + entry
    if kind == "grouped_mlp":
        entry = (f"def {op}(hidden, w13, w2, expert_ids):\n"
                 f"    gu = _grouped_mm(hidden, w13, expert_ids)\n"
                 f"    h = _swiglu(gu, {_ACT_IS_GELU[act]}).to(hidden.dtype)\n"
                 f"    return _grouped_mm(h, w2, expert_ids)\n")
        return header + _MM_BLOCK + swiglu + "\n" + entry
    if kind == "grouped_mlp_fp8":
        entry = (f"def {op}(xq, w13q, w2q, xs, w13s, w2s, expert_ids):\n"
                 f"    x = (xq.float() * xs.float()).to(torch.bfloat16)\n"
                 f"    w13 = (w13q.float() * w13s.float()).to(torch.bfloat16)\n"
                 f"    w2 = (w2q.float() * w2s.float()).to(torch.bfloat16)\n"
                 f"    gu = _grouped_mm(x, w13, expert_ids)\n"
                 f"    h = _swiglu(gu, {_ACT_IS_GELU[act]}).to(torch.bfloat16)\n"
                 f"    return _grouped_mm(h, w2, expert_ids)\n")
        return header + _MM_BLOCK + swiglu + "\n" + entry

    # ------------------------------------------------------- COMBINE ----------
    if kind == "sum_combine":
        entry = (f"def {op}(y, tw):\n"
                 f"    M, topk, D = y.shape\n"
                 f"    y = y.contiguous()\n"
                 f"    tw = tw.contiguous()\n"
                 f"    out = torch.empty((M, D), device=y.device, dtype=y.dtype)\n"
                 f"    _combine_kernel[(M,)](y, tw, out, topk, D, y.stride(0), y.stride(1), y.stride(2),\n"
                 f"                          tw.stride(0), out.stride(0), out.stride(1), BD=256)\n"
                 f"    return out\n")
        return header + _COMBINE_BLOCK + "\n" + entry
    if kind == "finalize":
        entry = (f"def {op}(y_perm, row_map, tw):\n"
                 f"    M, topk = row_map.shape\n"
                 f"    D = y_perm.shape[1]\n"
                 f"    yg = y_perm.index_select(0, row_map.reshape(-1).to(torch.long)).reshape(M, topk, D).contiguous()\n"
                 f"    tw = tw.contiguous()\n"
                 f"    out = torch.empty((M, D), device=y_perm.device, dtype=y_perm.dtype)\n"
                 f"    _combine_kernel[(M,)](yg, tw, out, topk, D, yg.stride(0), yg.stride(1), yg.stride(2),\n"
                 f"                          tw.stride(0), out.stride(0), out.stride(1), BD=256)\n"
                 f"    return out\n")
        return header + _COMBINE_BLOCK + "\n" + entry
    if kind == "shared_expert":
        entry = (f"def {op}(hidden, ws1, ws2, routed):\n"
                 f"    gu = _mm_nt(hidden, ws1)\n"
                 f"    h = _swiglu(gu, {_ACT_IS_GELU[act]}).to(hidden.dtype)\n"
                 f"    y = _mm_nt(h, ws2)\n"
                 f"    return (routed.float() + y.float()).to(hidden.dtype)\n")
        return header + _MM_BLOCK + swiglu + "\n" + entry

    # ------------------------------------------ FUSED / END-TO-END BLOCK ------
    if kind == "fused_moe":
        entry = (f"def {op}(hidden, w1, w2, tw, ti):\n"
                 f"    return _fused_run(hidden, w1, w2, tw, ti)\n")
        return header + _MM_BLOCK + swiglu + fused_run + "\n" + entry
    if kind == "moe_block":
        mode = "softmax" if router == "softmax" else "sigmoid"
        entry = (f"def {op}(hidden, gate, w1, w2, topk):\n"
                 f"    _, tw, ti = _route_topk(gate, topk, {mode!r}, True)\n"
                 f"    return _fused_run(hidden, w1, w2, tw, ti)\n")
        return header + _ROUTE_BLOCK + _MM_BLOCK + swiglu + fused_run + "\n" + entry

    raise ValueError(f"no seed template for kind {kind!r}")


def op_names() -> list[str]:
    return list(OPS)
