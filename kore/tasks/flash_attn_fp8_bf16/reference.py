"""Reference + inputs for fp8 causal (GQA) flash attention (fp8 QKV -> bf16 out).

fp8 attention for high-throughput serving: q/k/v are fp8 (arch ``FP8_DTYPE``)
with per-tensor fp32 scales; the kernel dequantizes in-register and runs
online-softmax flash,
moving ~half the QKV bytes of bf16. Correctness oracle: exact fp32 SDPA on the
DEQUANTIZED fp8 q/k/v (the fp8 rounding is shared by candidate + reference, so the
SNR gate measures the kernel's online-softmax fidelity, not the fp8 quantization).

Layout: q ``[B,S,H,D]`` fp8, k/v ``[B,S,KV,D]`` fp8, per-tensor scales
``sq/sk/sv`` (fp32 scalars). fp8 e4m3 is arch-selected: OCP ``e4m3fn`` on
gfx950/CDNA4 (MI350X/MI355X), FNUZ ``e4m3fnuz`` on gfx942/CDNA3.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from kore.tasks.aiter_ref import FP8_DTYPE, FP8_MAX


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 1, "H": 32, "KV": 8, "S": 2048, "D": 128}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def _q8(x: torch.Tensor):
    amax = x.abs().amax().clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    xq = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return xq, scale


def get_inputs(shape: dict, dtype=None, device="cuda", seed: int = 0):
    """Returns (q, k, v, sq, sk, sv): q:[B,S,H,D] fp8, k/v:[B,S,KV,D] fp8, scales fp32."""
    g = torch.Generator(device=device).manual_seed(seed)
    B, H, KV, S, D = shape["B"], shape["H"], shape["KV"], shape["S"], shape["D"]
    qf = torch.randn((B, S, H, D), generator=g, device=device, dtype=torch.float32)
    kf = torch.randn((B, S, KV, D), generator=g, device=device, dtype=torch.float32)
    vf = torch.randn((B, S, KV, D), generator=g, device=device, dtype=torch.float32)
    q, sq = _q8(qf)
    k, sk = _q8(kf)
    v, sv = _q8(vf)
    return q, k, v, sq, sk, sv


def attn_ref(q, k, v, sq, sk, sv, causal: bool = True) -> torch.Tensor:
    """Exact fp32 causal GQA attention on the dequantized fp8 q/k/v -> bf16 [B,S,H,D]."""
    B, S, H, D = q.shape
    KV = k.shape[2]
    scale = 1.0 / math.sqrt(D)
    qf = (q.float() * float(sq)).transpose(1, 2)     # [B,H,S,D]
    kf = (k.float() * float(sk)).transpose(1, 2)     # [B,KV,S,D]
    vf = (v.float() * float(sv)).transpose(1, 2)
    rep = H // KV
    if rep > 1:
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal, scale=scale)
    return out.transpose(1, 2).to(torch.bfloat16)    # [B,S,H,D]
