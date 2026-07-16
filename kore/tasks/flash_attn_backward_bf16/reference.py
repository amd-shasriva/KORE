"""Reference + inputs for the causal MHA FLASH-ATTENTION BACKWARD kernel (dQ/dK/dV).

This is the training-step backward of causal multi-head attention (H == KV, no
grouping). Forward (per (b, h), scale = 1/sqrt(D)):
    S_ij = scale * q_i . k_j        (causal: keep j <= i, else -inf)
    P    = softmax(S)  row-wise      O_i = sum_j P_ij v_j
Given dO, the FlashAttention-2 backward is (with delta_i = sum_d O_id dO_id):
    dP_ij = dO_i . v_j
    dS_ij = P_ij * (dP_ij - delta_i)          (softmax backward per row)
    dV_j  = sum_i P_ij dO_i
    dQ_i  = scale * sum_j dS_ij k_j
    dK_j  = scale * sum_i dS_ij q_i
These closed forms live in the SEED; the CORRECTNESS ORACLE here is torch AUTOGRAD
on the fp32 forward (ground truth), so a wrong hand-derived dS can never leak into
the oracle. The seed's analytic dS is separately cross-checked against this oracle
(and against finite differences) in ``_cpu_sanity_check.py``.

Candidate contract:
    ``flash_attn_backward(q, k, v, o, do, lse, causal=True) -> (dq, dk, dv)``
The candidate is GIVEN the saved forward output ``o`` and the row log-sum-exp
``lse`` (both saved by a real flash forward) plus ``do``; it recomputes P on the
fly (flash-style) rather than materializing the S x S matrix.

Layout matches the forward attention drafts / AITER ``flash_attn_func``:
q,k,v,o,do are [B, S, H, D] bf16; lse is [B, H, S] fp32; grads are [B, S, H, D].

Baseline (driver --impl reference): autograd through
``F.scaled_dot_product_attention`` at the native dtype -- on ROCm this dispatches
to the fused flash-attention backward (AOTriton / CK), a REAL fused kernel and a
strong perf bar (there is no standalone AITER "mha backward" python symbol we rely
on; see VERIFICATION_CHECKLIST.md). NEVER weaken the oracle to match a baseline.
"""

from __future__ import annotations

ENTRY = "flash_attn_backward"
GRAD_NAMES = ("dq", "dk", "dv")
# All three reduce over the sequence; keep a uniform bf16 gradient bar. The SNR
# gate (25 dB) is the real correctness gate; allclose is a coarse guard.
TOL = {"dq": (5e-2, 2e-2), "dk": (5e-2, 2e-2), "dv": (5e-2, 2e-2)}
CAUSAL = True


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 2, "H": 16, "S": 2048, "D": 128}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def _fwd_o_lse(q, k, v, scale, causal):
    """fp32 forward -> (o [B,S,H,D] fp32, lse [B,H,S] fp32). Saved activations."""
    import torch

    B, S, H, D = q.shape
    qh = q.float().transpose(1, 2)          # [B,H,S,D]
    kh = k.float().transpose(1, 2)
    vh = v.float().transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale   # [B,H,S,S]
    if causal:
        i = torch.arange(S, device=q.device)[:, None]
        j = torch.arange(S, device=q.device)[None, :]
        scores = scores + torch.where(j <= i, 0.0, float("-inf")).to(torch.float32)
    m = scores.amax(dim=-1, keepdim=True)                     # [B,H,S,1]
    m = torch.nan_to_num(m, neginf=0.0)                       # no fully-masked row in causal
    e = torch.exp(scores - m)
    denom = e.sum(dim=-1, keepdim=True)
    lse = (m.squeeze(-1) + torch.log(denom.squeeze(-1)))      # [B,H,S]
    p = e / denom
    o = torch.matmul(p, vh).transpose(1, 2).contiguous()     # [B,S,H,D]
    return o, lse.contiguous()


def get_inputs(shape: dict, device="cuda", seed: int = 0, dtype=None):
    """Returns (q, k, v, o, do, lse). q,k,v,o,do [B,S,H,D] ``dtype``; lse [B,H,S] fp32."""
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    B, H, S, D = shape["B"], shape["H"], shape["S"], shape["D"]
    scale = 1.0 / (D ** 0.5)
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((B, S, H, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    k = torch.randn((B, S, H, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    v = torch.randn((B, S, H, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    gd = torch.Generator(device=device).manual_seed(seed + 7)
    do = torch.randn((B, S, H, D), generator=gd, device=device, dtype=torch.float32).to(dtype)
    o, lse = _fwd_o_lse(q, k, v, scale, CAUSAL)
    return (q, k, v, o.to(dtype), do, lse)


def reference_forward(shape, inputs):
    """fp32 attention forward output (for the CPU independent-forward check)."""
    q, k, v = inputs[0], inputs[1], inputs[2]
    D = q.shape[-1]
    o, _lse = _fwd_o_lse(q, k, v, 1.0 / (D ** 0.5), CAUSAL)
    return o


def reference_grads(shape, inputs):
    """ORACLE: autograd grads of causal MHA attention on the fp32 forward.
    Returns (dq, dk, dv) fp32, each [B,S,H,D]."""
    import torch

    q, k, v, o, do, lse = inputs
    B, S, H, D = q.shape
    scale = 1.0 / (D ** 0.5)
    qf = q.float().detach().requires_grad_(True)       # [B,S,H,D]
    kf = k.float().detach().requires_grad_(True)
    vf = v.float().detach().requires_grad_(True)
    qh = qf.transpose(1, 2)                             # [B,H,S,D]
    kh = kf.transpose(1, 2)
    vh = vf.transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    if CAUSAL:
        i = torch.arange(S, device=q.device)[:, None]
        j = torch.arange(S, device=q.device)[None, :]
        scores = scores + torch.where(j <= i, 0.0, float("-inf")).to(torch.float32)
    p = torch.softmax(scores, dim=-1)
    oo = torch.matmul(p, vh).transpose(1, 2)           # [B,S,H,D]
    oo.backward(do.float())
    return (qf.grad.detach(), kf.grad.detach(), vf.grad.detach())


def candidate_grads(fn, shape, inputs):
    """Invoke candidate ``flash_attn_backward(q,k,v,o,do,lse,causal) -> (dq,dk,dv)``."""
    q, k, v, o, do, lse = inputs
    dq, dk, dv = fn(q, k, v, o, do, lse, causal=CAUSAL)
    return (dq, dk, dv)


def baseline_grads(shape, inputs):
    """Perf bar: autograd through F.scaled_dot_product_attention at the native dtype
    (fused flash-attention backward on ROCm). Returns (dq, dk, dv) [B,S,H,D]."""
    import torch
    import torch.nn.functional as F

    q, k, v, o, do, lse = inputs
    qh = q.transpose(1, 2).detach().clone().requires_grad_(True)   # [B,H,S,D]
    kh = k.transpose(1, 2).detach().clone().requires_grad_(True)
    vh = v.transpose(1, 2).detach().clone().requires_grad_(True)
    out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=CAUSAL)
    out.backward(do.transpose(1, 2))
    return (qh.grad.transpose(1, 2), kh.grad.transpose(1, 2), vh.grad.transpose(1, 2))
