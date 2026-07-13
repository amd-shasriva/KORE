"""Reference + inputs for the W4A16 (int4 weight, bf16 activation) GEMM.

Weight-only int4 GEMM for memory-bound LLM decode: the activation A[M,K] stays
bf16 (16-bit), the weight W[N,K] is symmetric per-output-channel int4 (codes
0..15 <-> values -8..7), packed 2 nibbles per byte along K into W_packed[N, K//2]
uint8, with a per-row fp32 scale[N,1]. Computes
    W_deq[n,k] = (nibble(n,k) - 8) * scale[n]
    Y = A @ W_deq^T            (bf16 out)

Correctness oracle: exact fp32 matmul of the DEQUANTIZED int4 weight. The int4
rounding is SHARED by candidate + reference, so the gate measures the kernel's
matmul fidelity (bf16 MFMA accumulation), not the quantization. Baseline
(driver --impl reference): materialize the weight to bf16 + hipBLASLt matmul --
the bar an int4 kernel beats by moving ~4x less weight through HBM.
"""

from __future__ import annotations

import torch

INT4_MIN, INT4_MAX = -8, 7


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 4096, "K": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def _quant_pack_int4(w: torch.Tensor):
    """w[N,K] fp32 -> (packed[N,K//2] uint8, scale[N,1] fp32). Symmetric per row."""
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (amax / INT4_MAX).to(torch.float32)
    q = torch.round(w / scale).clamp(INT4_MIN, INT4_MAX).to(torch.int32)  # -8..7
    code = (q + 8).to(torch.uint8)                                        # 0..15
    lo = code[:, 0::2]                          # even-K nibble
    hi = code[:, 1::2]                          # odd-K nibble
    packed = (lo | (hi << 4)).contiguous()     # [N, K//2] uint8
    return packed, scale


def unpack_dequant(packed: torch.Tensor, scale: torch.Tensor, K: int) -> torch.Tensor:
    """packed[N,K//2] uint8 -> W_deq[N,K] fp32."""
    lo = (packed & 0xF).to(torch.int32) - 8
    hi = ((packed >> 4) & 0xF).to(torch.int32) - 8
    N = packed.shape[0]
    q = torch.empty((N, K), dtype=torch.int32, device=packed.device)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    return q.float() * scale.float()


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (a[M,K] bf16, w_packed[N,K//2] uint8, scale[N,1] fp32)."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N, K = shape["M"], shape["N"], shape["K"]
    assert K % 2 == 0, "K must be even for int4 nibble packing"
    a = torch.randn((M, K), generator=g, device=device, dtype=torch.float32).to(dtype)
    w = torch.randn((N, K), generator=g, device=device, dtype=torch.float32)
    packed, scale = _quant_pack_int4(w)
    return a, packed, scale


def matmul_ref(a: torch.Tensor, w_packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Exact fp32 oracle on the dequantized int4 weight -> bf16."""
    K = a.shape[1]
    w_deq = unpack_dequant(w_packed, scale, K)          # [N,K] fp32
    return (a.float() @ w_deq.t()).to(torch.bfloat16)
