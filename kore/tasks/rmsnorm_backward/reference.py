"""Reference + inputs for the RMSNorm BACKWARD kernel (training-time op).

Given the forward ``y = x * rsqrt(mean(x^2)+eps) * w`` and an upstream gradient
dy, compute the input/weight gradients
    r    = rsqrt(mean(x^2)+eps)                     (per row)
    c    = sum_j (dy_j * w_j * x_j)                 (per row)
    dx_j = r*w_j*dy_j - (r^3 * x_j * c) / N
    dw_j = sum_m (dy_{m,j} * x_{m,j} * r_m)         (reduce over tokens)

The dw reduction over the token (M) axis is the interesting part - a good kernel
fuses the per-row dx with a blocked/atomic dw accumulation instead of PyTorch's
generic autograd graph. Backward/training kernels were a whole missing category
in the suite (it was inference-only).

Correctness oracle: torch AUTOGRAD gradients (ground truth). Baseline
(driver --impl reference): the torch autograd backward pass (the bar to beat).
"""

from __future__ import annotations

import torch

EPS = 1e-6


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (x[M,N] bf16, w[N] bf16, dy[M,N] bf16 upstream grad)."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N = shape["M"], shape["N"]
    x = torch.randn((M, N), generator=g, device=device, dtype=torch.float32).to(dtype)
    gw = torch.Generator(device=device).manual_seed(seed + 1)
    w = (torch.randn((N,), generator=gw, device=device, dtype=torch.float32) * 0.1 + 1.0).to(dtype)
    gd = torch.Generator(device=device).manual_seed(seed + 2)
    dy = torch.randn((M, N), generator=gd, device=device, dtype=torch.float32).to(dtype)
    return (x, w, dy)


def backward_ref(x: torch.Tensor, w: torch.Tensor, dy: torch.Tensor):
    """Ground-truth gradients via torch autograd. Returns (dx[M,N], dw[N]) fp32."""
    xf = x.float().detach().requires_grad_(True)
    wf = w.float().detach().requires_grad_(True)
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    y = xf * rms * wf
    y.backward(dy.float())
    return xf.grad.detach(), wf.grad.detach()
