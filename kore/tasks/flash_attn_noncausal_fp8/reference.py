"""Reference + inputs for fp8 NON-causal (GQA) flash attention (fp8 QKV -> bf16 out).

fp8 bidirectional attention for high-throughput serving: q/k/v are fp8 (arch
``FP8_DTYPE``) with per-tensor fp32 scales; the kernel dequantizes in-register and runs
online-softmax flash, moving ~half the QKV bytes of bf16. Complements the existing
causal fp8 prefill task with the bidirectional (encoder / prefix / non-causal) case.
Correctness oracle: exact fp32 NON-causal SDPA on the DEQUANTIZED fp8 q/k/v (the fp8
rounding is shared by candidate + reference, so the SNR gate measures the kernel's
online-softmax fidelity, not the fp8 quantization error).

Layout: q ``[B,S,H,D]`` fp8, k/v ``[B,S,KV,D]`` fp8, per-tensor scales sq/sk/sv (fp32
scalars). fp8 e4m3 is arch-selected: OCP ``e4m3fn`` on gfx950/CDNA4, FNUZ ``e4m3fnuz``
on gfx942/CDNA3 (see kore.tasks.aiter_ref).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._attn_common import expand_kv, sdpa_fp32  # noqa: E402

ENTRY = "flash_attn"
ATOL = 3e-2
RTOL = 3e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 1, "H": 32, "KV": 8, "S": 2048, "D": 128}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def _q8(x):
    import torch

    from kore.tasks.aiter_ref import FP8_DTYPE, FP8_MAX
    amax = x.abs().amax().clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    xq = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return xq, scale


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (q, k, v, sq, sk, sv): q ``[B,S,H,D]`` fp8, k/v ``[B,S,KV,D]`` fp8, scales fp32."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    B, H, KV, S, D = shape["B"], shape["H"], shape["KV"], shape["S"], shape["D"]
    qf = torch.randn((B, S, H, D), generator=g, device=device, dtype=torch.float32)
    kf = torch.randn((B, S, KV, D), generator=g, device=device, dtype=torch.float32)
    vf = torch.randn((B, S, KV, D), generator=g, device=device, dtype=torch.float32)
    q, sq = _q8(qf)
    k, sk = _q8(kf)
    v, sv = _q8(vf)
    return (q, k, v, sq, sk, sv)


def reference_output(shape, inputs):
    """Exact fp32 NON-causal GQA attention on the dequantized fp8 q/k/v -> bf16 ``[B,S,H,D]``."""
    import torch

    q, k, v, sq, sk, sv = inputs
    B, S, H, D = q.shape
    scale = 1.0 / (D ** 0.5)
    qf = (q.float() * float(sq)).transpose(1, 2)          # [B,H,S,D]
    kf = expand_kv((k.float() * float(sk)).transpose(1, 2), H)
    vf = expand_kv((v.float() * float(sv)).transpose(1, 2), H)
    out = sdpa_fp32(qf, kf, vf, scale, attn_mask=None)    # bidirectional
    return out.transpose(1, 2).to(torch.bfloat16)         # [B,S,H,D]


def candidate_output(fn, shape, inputs):
    q, k, v, sq, sk, sv = inputs
    return fn(q, k, v, sq, sk, sv, causal=False)


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER bf16 FMHA on the dequantized-to-bf16 q/k/v (non-causal).

    The fp8 kernel beats this on QKV bandwidth (fp8 moves ~half the bytes of bf16)."""
    import torch

    from kore.tasks.aiter_ref_attn import aiter_flash_attn

    q, k, v, sq, sk, sv = inputs
    qb = (q.float() * float(sq)).to(torch.bfloat16)
    kb = (k.float() * float(sk)).to(torch.bfloat16)
    vb = (v.float() * float(sv)).to(torch.bfloat16)
    return aiter_flash_attn(qb, kb, vb, causal=False)
