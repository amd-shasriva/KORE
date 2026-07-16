"""CPU sanity check for the DRAFT MoE oracles (reference-correctness only).

Runs each task's ``reference.reference_output`` on CPU against a SMALL random
input and compares it to a fully INDEPENDENT torch implementation computed here
(a DIFFERENT code path than the ``_moe_common`` oracle the references use):
per-token gather + ``bmm`` instead of the reference's per-expert group loop,
manual softmax + argsort instead of ``torch.softmax`` + ``torch.topk``, and an
explicit per-token python loop for the DeepSeek-V3 biased grouped router. If the
two agree to bf16 precision, the oracle math (gated MLP + activation, grouped /
batched GEMM, router selection + weights, fp8 dequant, dispatch permute, weighted
combine) is corroborated.

MoE routing + expert dispatch is error-prone, so this ALSO verifies two things
EXPLICITLY (the user-requested integrity checks):
  * token->expert assignment: the permute ``sort_idx`` really groups every
    expert's tokens into one contiguous ascending-expert block, and the gather
    is exact (:func:`check_assignment`).
  * the weighted combine: ``moe_sum`` equals an explicit per-slot weighted sum
    (:func:`check_combine`).

This proves REFERENCE CORRECTNESS ON CPU ONLY. It does NOT compile the Triton
seeds, run the AITER vendor baselines, or measure anything on gfx950 -- see
VERIFICATION_CHECKLIST.md. Run from the repo root:

    python kore/tasks/_drafts/moe/_cpu_sanity_check.py
"""

from __future__ import annotations

import importlib.util
import math
import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_ref(task_id):
    path = os.path.join(HERE, task_id, "reference.py")
    spec = importlib.util.spec_from_file_location(f"draft_moe_ref_{task_id}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _snr_db(o, r):
    o, r = o.float(), r.float()
    noise = (o - r).norm().item()
    signal = r.norm().item()
    if noise == 0:
        return 999.0
    return 20.0 * math.log10(signal / noise) if signal > 0 else -999.0


# --------------------------------------------------------------------------- #
# Independent reimplementations (different code path than _moe_common)
# --------------------------------------------------------------------------- #
def _indep_gated_mlp(inputs, act):
    """Per-(token,slot) gather + bmm gated MLP (vs reference per-expert loop)."""
    hidden, w1, w2, tw, ti = inputs
    M, D = hidden.shape
    I = w2.shape[2]
    topk = ti.shape[1]
    x = hidden.float()
    ids = ti.long().reshape(-1)                          # [M*topk]
    w1g = w1.float()[ids]                                # [M*topk, 2I, D]
    w2g = w2.float()[ids]                                # [M*topk, D, I]
    xr = x.repeat_interleave(topk, dim=0)                # [M*topk, D]
    gate_up = torch.bmm(w1g, xr.unsqueeze(-1)).squeeze(-1)   # [M*topk, 2I]
    gate, up = gate_up[:, :I], gate_up[:, I:]
    if act == "gelu":
        h = F.gelu(gate, approximate="tanh") * up
    else:
        h = F.silu(gate) * up
    ys = torch.bmm(w2g, h.unsqueeze(-1)).squeeze(-1)     # [M*topk, D]
    ys = ys.reshape(M, topk, D)
    out = (ys * tw.float().unsqueeze(-1)).sum(dim=1)     # weighted combine
    return out.to(torch.bfloat16)


def _indep_batched_gemm(inputs):
    a, b = inputs
    E = a.shape[0]
    out = torch.stack([a[e].float() @ b[e].float().t() for e in range(E)])
    return out.to(torch.bfloat16)


def _indep_grouped_gemm(inputs):
    hidden, w, expert_ids = inputs
    wg = w.float()[expert_ids.long()]                    # [M,N,K]
    out = torch.bmm(wg, hidden.float().unsqueeze(-1)).squeeze(-1)  # [M,N]
    return out.to(torch.bfloat16)


def _indep_grouped_gemm_fp8(inputs):
    xq, wq, xs, ws, expert_ids = inputs
    xd = xq.float() * xs.float()                         # [M,K]
    wd = wq.float() * ws.float()                         # [E,N,K]
    wdg = wd[expert_ids.long()]                          # [M,N,K]
    out = torch.bmm(wdg, xd.unsqueeze(-1)).squeeze(-1)   # [M,N]
    return out.to(torch.bfloat16)


def _indep_topk_softmax_norenorm(inputs, topk, E):
    (gate,) = inputs
    gf = gate.float()
    mx = gf.max(dim=-1, keepdim=True).values
    ex = torch.exp(gf - mx)
    sm = ex / ex.sum(dim=-1, keepdim=True)               # manual softmax
    order = torch.argsort(sm, dim=-1, descending=True)[:, :topk]
    dense = torch.zeros((gf.shape[0], E), dtype=torch.float32)
    dense.scatter_(1, order, torch.gather(sm, 1, order))  # raw probs (no renorm)
    return dense


def _indep_biased_grouped_topk(inputs, topk, n_groups, topk_group, E):
    """Explicit per-token python DeepSeek-V3 router (vs vectorized reference)."""
    gate, bias = inputs
    gf = gate.float()
    b = bias.float()
    M = gf.shape[0]
    grp = E // n_groups
    dense = torch.zeros((M, E), dtype=torch.float32)
    for m in range(M):
        scores = torch.sigmoid(gf[m])                    # [E]
        sb = scores + b
        gscore = []
        for g in range(n_groups):
            seg = sb[g * grp:(g + 1) * grp]
            top2 = torch.sort(seg, descending=True).values[:min(2, grp)]
            gscore.append(top2.sum())
        gscore = torch.stack(gscore)
        keep = set(torch.argsort(gscore, descending=True)[:topk_group].tolist())
        masked = sb.clone()
        for g in range(n_groups):
            if g not in keep:
                masked[g * grp:(g + 1) * grp] = float("-inf")
        sel = torch.argsort(masked, descending=True)[:topk]
        w = scores[sel]
        w = w / w.sum().clamp(min=1e-12)
        for j, e in enumerate(sel.tolist()):
            dense[m, e] = w[j]
    return dense


def _indep_permute(inputs):
    hidden, sort_idx = inputs
    return hidden[sort_idx.long()]


def _indep_moe_sum(inputs):
    y, w = inputs
    M, topk, D = y.shape
    out = torch.zeros((M, D), dtype=torch.float32)
    for k in range(topk):
        out += w[:, k:k + 1].float() * y[:, k, :].float()
    return out.to(torch.bfloat16)


def _independent(task_id, shape, inputs):
    if task_id == "moe_gelu_bf16":
        return _indep_gated_mlp(inputs, act="gelu")
    if task_id == "moe_batched_gemm_bf16":
        return _indep_batched_gemm(inputs)
    if task_id == "moe_grouped_gemm_bf16":
        return _indep_grouped_gemm(inputs)
    if task_id == "moe_grouped_gemm_fp8":
        return _indep_grouped_gemm_fp8(inputs)
    if task_id == "moe_topk_softmax_norenorm_bf16":
        return _indep_topk_softmax_norenorm(inputs, shape["topk"], shape["E"])
    if task_id == "moe_biased_grouped_topk_bf16":
        return _indep_biased_grouped_topk(inputs, shape["topk"], shape["n_groups"],
                                          shape["topk_group"], shape["E"])
    if task_id == "moe_permute_bf16":
        return _indep_permute(inputs)
    if task_id == "moe_sum_combine_bf16":
        return _indep_moe_sum(inputs)
    raise KeyError(task_id)


TINY = {
    "moe_gelu_bf16": {"M": 8, "E": 4, "topk": 2, "D": 16, "I": 12},
    "moe_batched_gemm_bf16": {"E": 3, "m": 5, "N": 7, "K": 8},
    "moe_grouped_gemm_bf16": {"M": 12, "E": 4, "N": 10, "K": 8},
    "moe_grouped_gemm_fp8": {"M": 12, "E": 4, "N": 10, "K": 16},
    "moe_topk_softmax_norenorm_bf16": {"M": 6, "E": 5, "topk": 2},
    "moe_biased_grouped_topk_bf16": {"M": 6, "E": 8, "topk": 2, "n_groups": 4, "topk_group": 2},
    "moe_permute_bf16": {"M": 12, "E": 4, "D": 16},
    "moe_sum_combine_bf16": {"M": 8, "topk": 3, "D": 16},
}

# Edge shapes: non-pow2, higher topk, 0-token expert, single-token decode.
EXTRA = {
    "moe_gelu_bf16": {"M": 7, "E": 5, "topk": 3, "D": 24, "I": 9},
    "moe_grouped_gemm_bf16": {"M": 9, "E": 6, "N": 5, "K": 6},
    "moe_grouped_gemm_fp8": {"M": 8, "E": 5, "N": 6, "K": 16},
    "moe_topk_softmax_norenorm_bf16": {"M": 3, "E": 9, "topk": 1},
    "moe_biased_grouped_topk_bf16": {"M": 4, "E": 12, "topk": 3, "n_groups": 3, "topk_group": 2},
    "moe_sum_combine_bf16": {"M": 5, "topk": 1, "D": 20},
    "moe_permute_bf16": {"M": 1, "E": 4, "D": 20},
}


def _run_one(task_id, shape):
    ref = _load_ref(task_id)
    inputs = ref.get_inputs(shape, device="cpu", seed=0)
    got = ref.reference_output(shape, inputs)
    exp = _independent(task_id, shape, inputs)
    assert got.shape == exp.shape, f"{task_id}: shape {tuple(got.shape)} != {tuple(exp.shape)}"
    assert got.dtype == exp.dtype, f"{task_id}: dtype {got.dtype} != {exp.dtype}"
    snr = _snr_db(got, exp)
    md = (got.float() - exp.float()).abs().max().item()
    ac = torch.allclose(got.float(), exp.float(), atol=3e-2, rtol=3e-2)
    return snr, md, ac


# --------------------------------------------------------------------------- #
# Explicit integrity checks (token->expert assignment + weighted combine)
# --------------------------------------------------------------------------- #
def check_assignment():
    """Verify the permute sort_idx groups tokens by expert + gather is exact.

    Replays get_inputs' generator (hidden draw, then the top-1 router) to recover
    the token->expert assignment, then asserts (a) sort_idx is a permutation,
    (b) the experts along permuted order are non-decreasing (each expert's tokens
    form one contiguous block), and (c) permuted[i] == hidden[sort_idx[i]]."""
    from _moe_common import make_routing

    shape = {"M": 40, "E": 8, "D": 16}
    ref = _load_ref("moe_permute_bf16")
    hidden, sort_idx = ref.get_inputs(shape, device="cpu", seed=0)
    M, E, D = shape["M"], shape["E"], shape["D"]
    # replay the exact generator sequence get_inputs used
    g = torch.Generator(device="cpu").manual_seed(0)
    _ = torch.randn((M, D), generator=g, device="cpu", dtype=torch.float32)
    _, ti = make_routing(M, E, 1, "cpu", g, renorm=False)
    expert_ids = ti[:, 0]
    si = sort_idx.long()
    perm_ok = torch.equal(torch.sort(si).values, torch.arange(M))
    experts_sorted = expert_ids[si]
    grouped_ok = bool((experts_sorted[1:] >= experts_sorted[:-1]).all())
    gather_ok = torch.equal(hidden[si], ref.reference_output(shape, (hidden, sort_idx)))
    dead_ok = bool((expert_ids != (E - 1)).all())        # last expert 0-token
    ok = perm_ok and grouped_ok and gather_ok and dead_ok
    detail = (f"perm={perm_ok} grouped_by_expert={grouped_ok} gather_exact={gather_ok} "
              f"last_expert_0tok={dead_ok}")
    return ok, detail


def check_combine():
    """Verify moe_sum equals an explicit per-slot weighted sum."""
    ref = _load_ref("moe_sum_combine_bf16")
    shape = {"M": 10, "topk": 4, "D": 16}
    inputs = ref.get_inputs(shape, device="cpu", seed=0)
    y, w = inputs
    out = ref.reference_output(shape, inputs).float()
    expl = torch.zeros((shape["M"], shape["D"]), dtype=torch.float32)
    for k in range(shape["topk"]):
        expl += w[:, k:k + 1].float() * y[:, k, :].float()
    md = (out - expl).abs().max().item()
    ok = torch.allclose(out, expl, atol=3e-2, rtol=3e-2)
    return ok, f"max_diff={md:.6f}"


def main():
    tasks = sorted(TINY)
    all_ok = True
    print(f"{'task_id':<34} {'shape':<7} {'SNR(dB)':>9} {'max_diff':>10} {'allclose':>9}")
    print("-" * 76)
    for tid in tasks:
        for label, table in (("tiny", TINY), ("extra", EXTRA)):
            if tid not in table:
                continue
            snr, md, ac = _run_one(tid, dict(table[tid]))
            ok = (snr > 40.0) and ac
            all_ok = all_ok and ok
            flag = "OK" if ok else "FAIL"
            print(f"{tid:<34} {label:<7} {snr:>9.2f} {md:>10.5f} {str(ac):>9}  {flag}")
    print("-" * 76)
    print("Explicit integrity checks:")
    for name, fn in (("token->expert assignment (permute)", check_assignment),
                     ("weighted combine (moe_sum)", check_combine)):
        ok, detail = fn()
        all_ok = all_ok and ok
        print(f"  {name:<40} {'OK' if ok else 'FAIL'}   [{detail}]")
    print("-" * 76)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
