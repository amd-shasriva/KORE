"""Reference + inputs for the MXFP4 weight-only GEMM (MI350X / gfx950 / CDNA4).

MI350's headline low-precision format: OCP **Microscaling FP4 (MXFP4)**. The weight
W[N,K] is quantized to 4-bit **E2M1** codes with a shared **E8M0** (power-of-two)
scale per block of 32 consecutive K elements (the OCP MX spec). The activation
A[M,K] stays bf16. Computes:
    W_deq[n,k] = e2m1(code[n,k]) * 2^(e8m0[n, k//32] - 127)
    Y = A @ W_deq^T                (bf16 out)

E2M1 (1 sign, 2 exp, 1 mantissa) magnitudes: {0, .5, 1, 1.5, 2, 3, 4, 6} (max 6).
Codes are packed 2 nibbles/byte along K -> W_packed[N, K//2] uint8; the per-block
shared exponent is E8M0 (biased-127 uint8) -> scale[N, K//32].

Correctness oracle: exact fp32 matmul of the DEQUANTIZED mxfp4 weight. The mxfp4
rounding is SHARED by candidate + reference, so the SNR gate measures the kernel's
bf16 MFMA accumulation fidelity, not the quantization. Baseline (driver --impl
reference): materialize the weight to bf16 + hipBLASLt matmul -- the bar an mxfp4
kernel beats by moving ~4x less weight through HBM (and, on CDNA4, by the native
MXFP4 matrix path a tuned kernel can reach).
"""

from __future__ import annotations

import torch

MX_BLOCK = 32            # OCP microscaling block size
E2M1_MAX = 6.0           # max representable |value| in e2m1
E2M1_EMAX = 2            # exponent of the max normal (6.0 = 1.5 * 2^2)

# e2m1 magnitude for the 3 low bits (idx 0..7); sign is bit 3.
_E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# midpoints between adjacent levels -> round-to-nearest bucket boundaries.
_E2M1_MIDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 4096, "K": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def _levels(device):
    return torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=device)


def _quant_e2m1_nearest(v: torch.Tensor) -> torch.Tensor:
    """Scaled values v -> e2m1 codes 0..15 (round-to-nearest magnitude + sign)."""
    sign = (v < 0).to(torch.uint8)
    a = v.abs()
    mids = torch.tensor(_E2M1_MIDS, dtype=torch.float32, device=v.device)
    idx = torch.bucketize(a, mids).to(torch.uint8)   # 0..7
    return (sign << 3) | idx


def _e2m1_decode(code: torch.Tensor) -> torch.Tensor:
    """e2m1 codes 0..15 -> fp32 values."""
    idx = (code & 0x7).long()
    mag = _levels(code.device)[idx]
    sign = torch.where((code & 0x8) != 0, -1.0, 1.0)
    return sign * mag


def quant_pack_mxfp4(w: torch.Tensor):
    """w[N,K] fp32 -> (packed[N,K//2] uint8, scale_e8m0[N,K//32] uint8).

    Per 32-wide K block: shared exponent = floor(log2(amax)) - E2M1_EMAX, stored
    biased (+127) as E8M0; codes = round(w / 2^exp) in e2m1 (clamped to +/-6).
    """
    N, K = w.shape
    assert K % MX_BLOCK == 0, "K must be a multiple of 32 for MXFP4 blocks"
    wb = w.reshape(N, K // MX_BLOCK, MX_BLOCK)
    amax = wb.abs().amax(dim=2, keepdim=True).clamp(min=1e-20)
    exp = torch.floor(torch.log2(amax)) - float(E2M1_EMAX)     # shared exponent
    exp = exp.clamp(-127.0, 127.0)
    e8m0 = (exp + 127.0).to(torch.uint8)                       # biased
    scale = torch.exp2(exp)
    wq = (wb / scale).clamp(-E2M1_MAX, E2M1_MAX).reshape(N, K)
    codes = _quant_e2m1_nearest(wq)
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    packed = (lo | (hi << 4)).contiguous()                    # [N, K//2]
    scale_e8m0 = e8m0.reshape(N, K // MX_BLOCK).contiguous()   # [N, K//32]
    return packed, scale_e8m0


def unpack_dequant(packed: torch.Tensor, scale_e8m0: torch.Tensor, K: int) -> torch.Tensor:
    """(packed[N,K//2], scale_e8m0[N,K//32]) -> W_deq[N,K] fp32."""
    N = packed.shape[0]
    codes = torch.empty((N, K), dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = (packed >> 4) & 0xF
    vals = _e2m1_decode(codes)                                 # [N,K] fp32
    exp = scale_e8m0.to(torch.float32) - 127.0                # [N, K//32]
    scale = torch.exp2(exp).repeat_interleave(MX_BLOCK, dim=1)  # [N,K]
    return vals * scale


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (a[M,K] bf16, w_packed[N,K//2] uint8, scale_e8m0[N,K//32] uint8)."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N, K = shape["M"], shape["N"], shape["K"]
    assert K % MX_BLOCK == 0, "K must be a multiple of 32"
    a = torch.randn((M, K), generator=g, device=device, dtype=torch.float32).to(dtype)
    w = torch.randn((N, K), generator=g, device=device, dtype=torch.float32)
    packed, scale = quant_pack_mxfp4(w)
    return a, packed, scale


def matmul_ref(a: torch.Tensor, w_packed: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Exact fp32 oracle on the dequantized mxfp4 weight -> bf16."""
    K = a.shape[1]
    w_deq = unpack_dequant(w_packed, scale_e8m0, K)
    return (a.float() @ w_deq.t()).to(torch.bfloat16)
