"""Reference + inputs for the GEMM BACKWARD kernel (training-time op: dgrad + wgrad).

Forward is the linear layer ``Y = X @ W^T`` with X[M,K] (M tokens, K in-features),
W[N,K] (N out-features), Y[M,N] (matching the repo's a8w8 GEMM convention). Given
the upstream gradient dY[M,N], the two gradients a training step needs are
    dX = dY @ W       [M,K]     (dgrad / data gradient,   reduces over N)
    dW = dY^T @ X     [N,K]     (wgrad / weight gradient, reduces over M tokens)

Candidate contract: ``gemm_backward(x, w, dy) -> (dx, dw)``. Both are plain GEMMs;
a good kernel fuses/tunes them for CDNA4 matrix cores with fp32 accumulation.

Correctness ORACLE (ground truth): torch AUTOGRAD gradients of ``X @ W^T`` on the
fp32 forward. Baseline (driver --impl reference): the two backward GEMMs at the
native dtype via ``torch.matmul`` -- on ROCm this dispatches to hipBLASLt (the
vendor tuned GEMM library the serving/training stack uses), so it is a REAL vendor
GEMM bar, not a weak eager fallback (there is simply no single "gemm backward"
AITER symbol; the bar is two hipBLASLt GEMMs). NEVER weaken the oracle to match it.
"""

from __future__ import annotations

ENTRY = "gemm_backward"
GRAD_NAMES = ("dx", "dw")
# dx reduces over N, dw reduces over up to 32k tokens (M) -> looser (atol) bar for
# dw (cf. rmsnorm_backward dw); both are dominated by the SNR gate regardless.
TOL = {"dx": (5e-2, 2e-2), "dw": (2e-1, 2e-2)}


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 8192, "N": 4096, "K": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0, dtype=None):
    """Returns (x[M,K], w[N,K], dy[M,N]) at ``dtype`` (default bf16). x, w scaled by
    1/sqrt(K) so the forward/gradient magnitudes stay O(1) (SNR is scale-free, but
    this keeps the tiny CPU finite-difference case well-conditioned)."""
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    M, N, K = shape["M"], shape["N"], shape["K"]
    sc = 1.0 / (K ** 0.5)
    gx = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn((M, K), generator=gx, device=device, dtype=torch.float32) * sc).to(dtype)
    gw = torch.Generator(device=device).manual_seed(seed + 1)
    w = (torch.randn((N, K), generator=gw, device=device, dtype=torch.float32) * sc).to(dtype)
    gd = torch.Generator(device=device).manual_seed(seed + 2)
    dy = torch.randn((M, N), generator=gd, device=device, dtype=torch.float32).to(dtype)
    return (x, w, dy)


def reference_forward(shape, inputs):
    """fp32 linear forward Y = X @ W^T (for the CPU independent-forward check)."""
    x, w, _dy = inputs
    return x.float() @ w.float().t()


def reference_grads(shape, inputs):
    """ORACLE: autograd grads of Y = X @ W^T on the fp32 forward.
    Returns (dx[M,K], dw[N,K]) fp32."""
    x, w, dy = inputs
    xf = x.float().detach().requires_grad_(True)
    wf = w.float().detach().requires_grad_(True)
    y = xf @ wf.t()
    y.backward(dy.float())
    return (xf.grad.detach(), wf.grad.detach())


def candidate_grads(fn, shape, inputs):
    """Invoke candidate ``gemm_backward(x, w, dy) -> (dx, dw)``."""
    x, w, dy = inputs
    dx, dw = fn(x, w, dy)
    return (dx, dw)


def baseline_grads(shape, inputs):
    """Real vendor GEMM bar: dX = dY @ W and dW = dY^T @ X via torch.matmul
    (hipBLASLt on ROCm), at the native dtype. Perf bar to beat (see checklist)."""
    x, w, dy = inputs
    dx = dy @ w                 # [M,N] @ [N,K] -> [M,K]
    dw = dy.transpose(0, 1) @ x  # [N,M] @ [M,K] -> [N,K]
    return (dx, dw)
