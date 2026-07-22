"""GENERATED breadth quant seed: qx_kvcache_dequant_fp8 (bf16). Naive host amax /
nibble-pack + a tiled elementwise quantize/dequantize kernel - a correct,
COMPILING starting point the KORE policy fuses into one fused quant kernel."""
from __future__ import annotations
import torch
import triton
import triton.language as tl


FP8_MAX = 448.0
INT8_MAX = 127.0
BLK = 128
MX_BLOCK = 32
E2M1_MAX = 6.0
E2M1_EMAX = 2
_E2M1_MIDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@triton.jit
def _qx_q_kernel(x_ptr, inv_ptr, o_ptr, n, LO, HI, ROUND: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + offs, mask=m, other=0.0).to(tl.float32)
    v = x * inv
    if ROUND:
        v = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))
    v = tl.minimum(tl.maximum(v, LO), HI)
    tl.store(o_ptr + offs, v.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _qx_dq_kernel(c_ptr, s_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    c = tl.load(c_ptr + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs, (c * s).to(o_ptr.dtype.element_ty), mask=m)


def _q_map(xf, inv, lo, hi, do_round, out_dtype):
    xf, inv = xf.contiguous(), inv.contiguous()
    o = torch.empty(xf.shape, device=xf.device, dtype=out_dtype)
    n = xf.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _qx_q_kernel[grid](xf, inv, o, n, lo, hi, ROUND=(1 if do_round else 0), BLOCK=BLOCK)
    return o


def _dq_map(codes, scale_full, out_dtype):
    codes, scale_full = codes.contiguous(), scale_full.contiguous()
    o = torch.empty(codes.shape, device=codes.device, dtype=out_dtype)
    n = codes.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _qx_dq_kernel[grid](codes, scale_full, o, n, BLOCK=BLOCK)
    return o


def _e2m1_codes(v):
    sign = (v < 0).to(torch.uint8)
    mids = torch.tensor(_E2M1_MIDS, dtype=torch.float32, device=v.device)
    idx = torch.bucketize(v.abs(), mids).to(torch.uint8)
    return (sign << 3) | idx


def _e2m1_levels_lut(device):
    return torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=device)


def _fp8_levels(device):
    b = torch.arange(256, dtype=torch.uint8, device=device)
    lv = b.view(torch.float8_e4m3fn).float()
    return torch.unique(lv[torch.isfinite(lv)])


def qx_kvcache_dequant_fp8(kq, ksc, vq, vsc):
    ksf = ksc.float().expand(kq.shape[0], kq.shape[1])
    vsf = vsc.float().expand(vq.shape[0], vq.shape[1])
    return (_dq_map(kq, ksf, torch.bfloat16), _dq_map(vq, vsf, torch.bfloat16))
