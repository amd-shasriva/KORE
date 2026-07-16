"""Reference + inputs for the row-SOFTMAX BACKWARD kernel (training-time op).

Forward (per row, over the last dim N): ``y = softmax(x)``. Given the saved
forward output ``y`` and the upstream gradient ``dy``, the input gradient is the
softmax Jacobian-vector product
    dx_j = y_j * (dy_j - sum_k dy_k * y_k)                 (per row)
i.e. subtract the probability-weighted mean of dy, then scale by y. This is the
``_softmax_backward_data`` that sits inside every attention-prob and (log-)softmax
backward in a training step.

Candidate contract: ``softmax_backward(y, dy) -> dx``. The candidate is GIVEN the
saved forward output ``y`` (a real training kernel saves it) plus ``dy``; it does
NOT recompute the softmax.

Correctness ORACLE (ground truth): torch AUTOGRAD gradient of ``torch.softmax`` on
the fp32 forward. The task scores the candidate dx against this. Baseline
(driver --impl reference): the framework fused softmax backward at the native
dtype (aten ``_softmax_backward_data``) -- a real fused kernel and the honest perf
bar, since AITER exposes no standalone dense softmax backward (perf-only; see
VERIFICATION_CHECKLIST.md). NEVER weaken the oracle to match a baseline.
"""

from __future__ import annotations

ENTRY = "softmax_backward"
GRAD_NAMES = ("dx",)
TOL = {"dx": (2e-2, 2e-2)}


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 8192, "N": 8192}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0, dtype=None):
    """Returns (x[M,N], dy[M,N]) at ``dtype`` (default bf16). x is logit-scaled."""
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    M, N = shape["M"], shape["N"]
    gx = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn((M, N), generator=gx, device=device, dtype=torch.float32) * 2.0).to(dtype)
    gd = torch.Generator(device=device).manual_seed(seed + 1)
    dy = torch.randn((M, N), generator=gd, device=device, dtype=torch.float32).to(dtype)
    return (x, dy)


def reference_forward(shape, inputs):
    """fp32 softmax forward (for the CPU independent-forward check)."""
    import torch

    x, _dy = inputs
    return torch.softmax(x.float(), dim=-1)


def reference_grads(shape, inputs):
    """ORACLE: autograd gradient of softmax on the fp32 forward. Returns (dx,) fp32."""
    import torch

    x, dy = inputs
    xf = x.float().detach().requires_grad_(True)
    y = torch.softmax(xf, dim=-1)
    y.backward(dy.float())
    return (xf.grad.detach(),)


def candidate_grads(fn, shape, inputs):
    """Invoke candidate ``softmax_backward(y, dy) -> dx``. y = saved forward output."""
    import torch

    x, dy = inputs
    y = torch.softmax(x.float(), dim=-1).to(x.dtype)   # saved forward activation
    return (fn(y, dy),)


def baseline_grads(shape, inputs):
    """Perf-only bar: framework fused softmax backward at the native dtype
    (aten _softmax_backward_data). NO AITER standalone softmax backward exists."""
    import torch

    x, dy = inputs
    xf = x.detach().clone().requires_grad_(True)
    y = torch.softmax(xf, dim=-1)
    y.backward(dy)
    return (xf.grad.detach(),)
