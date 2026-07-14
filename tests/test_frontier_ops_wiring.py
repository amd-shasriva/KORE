"""Regression coverage for the v3 frontier additions:

  * new ops registered + CPU-verifiable oracles (embedding_gather, fp8 fusions,
    W4A16 int4 GEMM, rmsnorm backward);
  * headroom rebalance of the curation mix (WS-C3);
  * frontier failure taxonomy for the new op classes (WS-D3).

All CPU-only (no GPU / no Triton compile) so it runs in CI.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch

from kore.tasks.registry import get_task, is_heldout


def _snr(a: torch.Tensor, b: torch.Tensor) -> float:
    noise = (a.float() - b.float()).norm().item()
    sig = b.float().norm().item()
    return 999.0 if noise == 0 else (20.0 * math.log10(sig / noise) if sig else -999.0)


# --------------------------------------------------------------------------- #
# New ops are registered (and land in the right train/heldout split).
# --------------------------------------------------------------------------- #
def test_new_ops_registered():
    expect = {
        "genv_embedding_gather_bf16": ("other", False),  # family via op name; not heldout
        "fused_rmsnorm_quant_fp8": ("rmsnorm", False),
        "fused_silu_mul_quant_fp8": ("quant", False),
        "gemm_w4a16": ("gemm", False),
        "rmsnorm_backward": ("rmsnorm", False),
    }
    for tid, (_fam, heldout) in expect.items():
        t = get_task(tid)
        assert t is not None
        assert is_heldout(t) is heldout, f"{tid} heldout={is_heldout(t)} want {heldout}"


# --------------------------------------------------------------------------- #
# Oracles are numerically correct on CPU (the poison gate).
# --------------------------------------------------------------------------- #
def test_embedding_gather_oracle_exact():
    from kore.tasks.vendor_ops import make_vendor_reference
    ns = make_vendor_reference("embedding_gather", "bf16")
    w, ids = ns["get_inputs"]({"V": 500, "Dim": 32, "T": 64}, device="cpu", seed=0)
    got = ns["ref_fn"](w, ids).float()
    exp = torch.stack([w[int(i)] for i in ids]).float()
    assert (got - exp).abs().max().item() < 1e-6


def test_fp8_fusion_oracles_dequant_fidelity():
    import kore.tasks.fused_rmsnorm_quant_fp8.reference as r1
    import kore.tasks.fused_silu_mul_quant_fp8.reference as r2
    x, w = r1.get_inputs({"M": 128, "N": 512}, device="cpu", seed=0)
    y = r1.rmsnorm_ref(x, w)
    xq, sc = r1.fused_ref(x, w)
    assert _snr(r1.dequant(xq, sc), y) > 22.0            # fp8 gate
    (x2,) = r2.get_inputs({"M": 128, "N": 1024}, device="cpu", seed=0)
    y2 = r2.silu_mul_ref(x2)
    xq2, sc2 = r2.fused_ref(x2)
    assert _snr(r2.dequant(xq2, sc2), y2) > 22.0
    assert xq2.shape == (128, 512)  # [M, inter]


def test_w4a16_oracle_and_seed_algo():
    import kore.tasks.gemm_w4a16.reference as r
    # pack/unpack lossless
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    packed, scale = r._quant_pack_int4(w)
    assert packed.dtype == torch.uint8 and packed.shape == (64, 64)
    # oracle == the even/odd-K dual-dot the seed implements (exact modulo bf16 out)
    a, wp, sc = r.get_inputs({"M": 40, "N": 96, "K": 256}, device="cpu", seed=3)
    oracle = r.matmul_ref(a, wp, sc)
    lo = (wp & 0xF).int() - 8
    hi = ((wp >> 4) & 0xF).int() - 8
    sim = ((a.float()[:, 0::2]) @ (lo.float() * sc.float()).t()) + \
          ((a.float()[:, 1::2]) @ (hi.float() * sc.float()).t())
    assert (sim.to(torch.bfloat16).float() - oracle.float()).abs().max().item() < 1e-2


def test_rmsnorm_backward_formula_matches_autograd():
    import kore.tasks.rmsnorm_backward.reference as r
    x, w, dy = r.get_inputs({"M": 64, "N": 512}, device="cpu", seed=0)
    dx_ref, dw_ref = r.backward_ref(x, w, dy)
    xf, wf, g = x.float(), w.float(), dy.float()
    N = xf.shape[1]
    rr = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + r.EPS)
    c = (g * wf * xf).sum(-1, keepdim=True)
    dx_an = rr * wf * g - (rr ** 3) * xf * c / N
    dw_an = (g * xf * rr).sum(0)
    assert _snr(dx_an, dx_ref) > 60.0 and _snr(dw_an, dw_ref) > 60.0


# --------------------------------------------------------------------------- #
# WS-D3: frontier failure taxonomy for the new op classes.
# --------------------------------------------------------------------------- #
def test_frontier_breakers_break_new_op_seeds():
    from kore.data import mutate as m
    root = Path(__file__).resolve().parents[1] / "kore" / "tasks"
    w4a16 = (root / "gemm_w4a16" / "seed_triton.py").read_text()
    qf = (root / "fused_rmsnorm_quant_fp8" / "seed_triton.py").read_text()
    bw = (root / "rmsnorm_backward" / "seed_triton.py").read_text()

    b, h = m.break_nibble_unpack(w4a16)
    assert b != w4a16 and h == "snr_fail"
    b, h = m.break_amax_abs(qf)
    assert b != qf and h == "snr_fail" and "tl.abs" not in b
    b, h = m.break_atomic_to_store(bw)
    assert b != bw and h == "snr_fail" and "tl.atomic_add" not in b


def test_headroom_rebalance_concentrates_compute_bound():
    from kore.data.curate import op_class, rebalance_by_headroom

    def krow(op, n_extra=0):
        return {"_provenance": {"operation": op, "verified": True, "speedup": 2.0},
                "_source": "kernel_repair_opt",
                "messages": [{"role": "assistant", "content": "x" * (100 + n_extra)}]}

    # op_class taxonomy
    assert op_class(krow("gemm_bf16")) == "compute_bound"
    assert op_class(krow("flash_attn_prefill")) == "compute_bound"
    assert op_class(krow("gen_add_bf16")) == "trivial"          # bare elementwise
    assert op_class(krow("rmsnorm_bf16")) == "memory_bound"     # structured fusion
    assert op_class({"messages": []}) == "retention"

    # rich pool: 10 compute + 90 trivial -> rebalance to ~50% compute
    pool = [krow("gemm_bf16") for _ in range(10)] + [krow("gen_add_bf16") for _ in range(90)]
    out, st = rebalance_by_headroom(pool, target_compute_frac=0.5)
    assert st["compute_frac"] >= 0.5 and st["capped"] == 80 and st["low_kept"] == 10

    # thin pool (only 2 compute): degrade gracefully, never below available
    thin = [krow("gemm_bf16") for _ in range(2)] + [krow("gen_add_bf16") for _ in range(5)]
    out2, st2 = rebalance_by_headroom(thin, target_compute_frac=0.5)
    assert st2["compute_bound"] == 2 and st2["low_kept"] == 2  # 2*(1-.5)/.5 = 2
    # degenerate: no compute-bound -> unchanged, no crash
    only_low = [krow("gen_add_bf16") for _ in range(4)]
    out3, st3 = rebalance_by_headroom(only_low, target_compute_frac=0.5)
    assert len(out3) == 4 and st3["capped"] == 0


def test_frontier_breakers_family_routed():
    from kore.data import mutate as m
    root = Path(__file__).resolve().parents[1] / "kore" / "tasks"
    cases = [("gemm", root / "gemm_w4a16" / "seed_triton.py", "break_nibble_unpack"),
             ("quant", root / "fused_silu_mul_quant_fp8" / "seed_triton.py", "break_amax_abs"),
             ("norm", root / "rmsnorm_backward" / "seed_triton.py", "break_atomic_to_store")]
    for fam, path, want in cases:
        src = path.read_text()
        names = {m.apply_random_breakage(src, fam, random.Random(i))[2] for i in range(200)}
        assert want in names, f"{want} unreachable for {fam}: {sorted(names)}"
