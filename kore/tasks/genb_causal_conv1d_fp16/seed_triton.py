"""GENERATED breadth causal_conv1d seed (fp16). Depthwise causal 1D conv (Mamba short conv).
x[B,D,L], weight[D,K], bias[D] -> y[B,D,L]. One program per (b, d); for each output time t,
y[t] = bias + sum_k weight[k] * x[t-(K-1)+k] (left-causal, x=0 for t<0). fp32 accumulate,
tl.float16 store. Naive per-time loop; the policy vectorizes over the time axis."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _causal_conv1d_kernel(x_ptr, w_ptr, b_ptr, y_ptr, D, L, K,
                          sx_b, sx_d, sx_l, sw_d, sw_k, sb_d, KB: tl.constexpr):
    pid = tl.program_id(0)
    bb = pid // D
    d = pid % D
    kk = tl.arange(0, KB)
    kmask = kk < K
    wrow = tl.load(w_ptr + d * sw_d + kk * sw_k, mask=kmask, other=0.0).to(tl.float32)
    bias = tl.load(b_ptr + d * sb_d).to(tl.float32)
    base = bb * sx_b + d * sx_d
    for t in range(0, L):
        idx = t - (K - 1) + kk                                   # [KB] input positions
        vmask = kmask & (idx >= 0)
        xv = tl.load(x_ptr + base + idx * sx_l, mask=vmask, other=0.0).to(tl.float32)
        acc = tl.sum(tl.where(vmask, wrow * xv, 0.0), axis=0) + bias
        tl.store(y_ptr + base + t * sx_l, acc.to(tl.float16))


def causal_conv1d(x, weight, bias):
    Bsz, D, L = x.shape
    K = weight.shape[1]
    x = x.contiguous(); weight = weight.contiguous(); bias = bias.contiguous()
    y = torch.empty_like(x)
    KB = triton.next_power_of_2(K)
    _causal_conv1d_kernel[(Bsz * D,)](
        x, weight, bias, y, D, L, K,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1), bias.stride(0), KB=KB, num_warps=1)
    return y
