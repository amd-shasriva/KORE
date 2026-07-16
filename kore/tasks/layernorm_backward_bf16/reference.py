"""Reference + inputs for the LayerNorm BACKWARD kernel (training-time op).

Forward (per row over the last dim N):
    mean = mean_j(x)                 var = mean_j((x-mean)^2)
    rstd = 1/sqrt(var + eps)         xhat = (x - mean) * rstd
    y = xhat * gamma + beta
Given the upstream gradient dy, the gradients are
    dbeta_j  = sum_m dy_{m,j}                          (reduce over tokens M)
    dgamma_j = sum_m dy_{m,j} * xhat_{m,j}             (reduce over tokens M)
    let g = dy * gamma  (dxhat); then per row (over N):
    dx = rstd * ( g - mean_j(g) - xhat * mean_j(g * xhat) )

The dgamma/dbeta reductions over the token (M) axis are the interesting part - a
good kernel fuses the per-row dx with a blocked/atomic (dgamma, dbeta) accumulation.

Candidate contract: ``layernorm_backward(x, gamma, dy) -> (dx, dgamma, dbeta)``.
The kernel recomputes mean/rstd from x (beta does not affect any gradient). The
task constant eps = 1e-5 (PyTorch LayerNorm default) MUST match the seed default.

Correctness ORACLE (ground truth): torch AUTOGRAD gradients of ``F.layer_norm`` on
the fp32 forward. Baseline (driver --impl reference): the framework fused LayerNorm
backward at the native dtype (aten ``native_layer_norm_backward``) -- a real fused
kernel and the honest perf bar (AITER exposes no standalone LayerNorm backward;
perf-only, see VERIFICATION_CHECKLIST.md). NEVER weaken the oracle to match it.
"""

from __future__ import annotations

EPS = 1e-5

ENTRY = "layernorm_backward"
GRAD_NAMES = ("dx", "dgamma", "dbeta")
# dgamma/dbeta reduce over up to 32k tokens; give them the looser (atol) bar used
# for reduction gradients (cf. rmsnorm_backward dw), dx keeps the tight per-row bar.
TOL = {"dx": (2e-2, 2e-2), "dgamma": (2e-1, 2e-2), "dbeta": (2e-1, 2e-2)}


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0, dtype=None):
    """Returns (x[M,N], gamma[N], beta[N], dy[M,N]) at ``dtype`` (default bf16)."""
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    M, N = shape["M"], shape["N"]
    gx = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((M, N), generator=gx, device=device, dtype=torch.float32).to(dtype)
    gg = torch.Generator(device=device).manual_seed(seed + 1)
    gamma = (torch.randn((N,), generator=gg, device=device, dtype=torch.float32) * 0.1 + 1.0).to(dtype)
    gb = torch.Generator(device=device).manual_seed(seed + 2)
    beta = (torch.randn((N,), generator=gb, device=device, dtype=torch.float32) * 0.1).to(dtype)
    gd = torch.Generator(device=device).manual_seed(seed + 3)
    dy = torch.randn((M, N), generator=gd, device=device, dtype=torch.float32).to(dtype)
    return (x, gamma, beta, dy)


def reference_forward(shape, inputs):
    """fp32 LayerNorm forward (for the CPU independent-forward check)."""
    import torch
    import torch.nn.functional as F

    x, gamma, beta, _dy = inputs
    N = x.shape[-1]
    return F.layer_norm(x.float(), (N,), gamma.float(), beta.float(), EPS)


def reference_grads(shape, inputs):
    """ORACLE: autograd grads of F.layer_norm on the fp32 forward.
    Returns (dx[M,N], dgamma[N], dbeta[N]) fp32."""
    import torch
    import torch.nn.functional as F

    x, gamma, beta, dy = inputs
    N = x.shape[-1]
    xf = x.float().detach().requires_grad_(True)
    gf = gamma.float().detach().requires_grad_(True)
    bf = beta.float().detach().requires_grad_(True)
    y = F.layer_norm(xf, (N,), gf, bf, EPS)
    y.backward(dy.float())
    return (xf.grad.detach(), gf.grad.detach(), bf.grad.detach())


def candidate_grads(fn, shape, inputs):
    """Invoke candidate ``layernorm_backward(x, gamma, dy) -> (dx, dgamma, dbeta)``."""
    x, gamma, beta, dy = inputs
    dx, dgamma, dbeta = fn(x, gamma, dy)
    return (dx, dgamma, dbeta)


def baseline_grads(shape, inputs):
    """Perf-only bar: framework fused LayerNorm backward at the native dtype
    (aten native_layer_norm_backward). NO AITER standalone LayerNorm backward."""
    import torch
    import torch.nn.functional as F

    x, gamma, beta, dy = inputs
    N = x.shape[-1]
    xf = x.detach().clone().requires_grad_(True)
    gf = gamma.detach().clone().requires_grad_(True)
    bf = beta.detach().clone().requires_grad_(True)
    y = F.layer_norm(xf, (N,), gf, bf, EPS)
    y.backward(dy)
    return (xf.grad.detach(), gf.grad.detach(), bf.grad.detach())
