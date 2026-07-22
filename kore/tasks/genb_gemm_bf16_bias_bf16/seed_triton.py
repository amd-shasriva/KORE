"""GENERATED breadth GEMM seed: gemm_bf16_bias (bf16). Naive host dequant + a
tiled tl.dot GEMM (fp32 accumulate) + epilogue - a correct, COMPILING
starting point the KORE policy fuses into one quantized-GEMM kernel."""
from __future__ import annotations
import torch
import triton
import triton.language as tl


FP8_MAX = 448.0
_E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _dq_a8(codes, s, gran):
    c = codes.float()
    if gran == "block128":
        return c * s.float().repeat_interleave(128, dim=1)
    return c * s.float()


def _dq_w8(codes, s, gran):
    c = codes.float()
    if gran == "block128":
        return c * s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    return c * s.float()


def _dq_int4c(packed, scale):
    N, K = packed.shape[0], packed.shape[1] * 2
    q = torch.empty((N, K), dtype=torch.int32, device=packed.device)
    q[:, 0::2] = (packed & 0xF).to(torch.int32) - 8
    q[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32) - 8
    return q.float() * scale.float()


def _dq_int4gs(packed, scale, group):
    N, K = packed.shape[0], packed.shape[1] * 2
    q = torch.empty((N, K), dtype=torch.int32, device=packed.device)
    q[:, 0::2] = (packed & 0xF).to(torch.int32) - 8
    q[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32) - 8
    return q.float() * scale.float().repeat_interleave(group, dim=1)


def _dq_int4ga(packed, scale, zero, group):
    N, K = packed.shape[0], packed.shape[1] * 2
    codes = torch.empty((N, K), dtype=torch.int32, device=packed.device)
    codes[:, 0::2] = (packed & 0xF).to(torch.int32)
    codes[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32)
    z = zero.to(torch.int32).repeat_interleave(group, dim=1)
    s = scale.float().repeat_interleave(group, dim=1)
    return (codes.float() - z.float()) * s


def _dq_mxfp4(packed, e8m0):
    R, K = packed.shape[0], packed.shape[1] * 2
    codes = torch.empty((R, K), dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = (packed >> 4) & 0xF
    levels = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=packed.device)
    mag = levels[(codes & 0x7).long()]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    scale = torch.exp2(e8m0.float() - 127.0).repeat_interleave(32, dim=1)
    return (sign * mag) * scale


def _dq_mxfp8(codes, e8m0):
    scale = torch.exp2(e8m0.float() - 127.0).repeat_interleave(32, dim=1)
    return codes.float() * scale


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
    """a [m,K], b [N,K] -> a @ b.T (fp32 accumulate via tl.dot); out dtype = a.dtype."""
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


def gemm_bf16_bias(a, w, bias):
    a = a
    w = w
    c = _mm_nt(a, w)
    y = c.float()
    y = y + bias.float().reshape(1, -1)
    return y.to(a.dtype)
