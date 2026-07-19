"""Training-critical task authoring engine: loss + optimizer-step Triton tasks.

The vendor-baselined suite (``kore.tasks.vendor_ops``) covers the transformer
FORWARD block interior (norms / activations / GEMMs / RoPE / MoE routing). This
engine covers the TRAINING-CRITICAL families that dominate a real training step
but were absent: the loss head (cross-entropy and friends, incl. the Liger-style
fused-linear-CE that never materializes the [M, V] logits) and the FUSED OPTIMIZER
step (AdamW / Lion / Muon / global-norm grad clipping) that updates the parameters
in place.

Unlike the vendor suite (graded vs real AITER kernels), these ops have NO vendor
kernel, so the honest bar is torch: the correctness ORACLE is a torch fp32
reference (``make_reference(op, dtype)["ref_fn"]``), and the perf BASELINE is the
eager torch computation (``baseline_fn``) - the multi-kernel path a fused Triton
kernel must beat. CORRECTNESS IS PARAMOUNT: every ``ref_fn`` reproduces the exact
torch math (AdamW == ``torch.optim.AdamW``; grad clip == ``clip_grad_norm_``;
cross-entropy == ``F.cross_entropy``), cast back to the task dtype.

Contract (mirrors ``vendor_ops`` so the generic ``kore.tasks._genops`` driver +
generator machinery consume it unchanged): ``make_reference(op, dtype)`` returns
the reference.py namespace (parse_shape / get_inputs / ref_fn oracle / baseline_fn
/ arity / entry_name / dtype_name / family / mutates_input); ``seed_source(op,
dtype)`` returns a naive-but-correct COMPILING Triton starter kernel.

The optimizer-step ops set ``mutates_input=True`` (they update the param + moment
buffers in place), so the bench loop feeds a fresh clone each timed call (see
``_genops._build_bench_fn`` mutates_input path).

torch imported lazily (registry discovery never needs a GPU/torch).
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# op -> family metadata; each op has a bespoke oracle/baseline/seed (below).
OPS: tuple[str, ...] = (
    # LOSSES (torch reference oracle; torch baseline)
    "cross_entropy", "fused_linear_cross_entropy", "kl_div",
    "label_smoothing_ce", "mse_loss",
    # OPTIMIZERS (single fused step; mutates param(s) in place)
    "fused_adamw", "fused_lion", "fused_muon", "grad_clip_global_norm",
)

# Optimizer-step ops whose kernel UPDATES its param/moment tensors IN PLACE (so the
# bench loop must feed a fresh clone each timed call - see _genops mutates_input).
TRAIN_MUTATES_INPUT: frozenset[str] = frozenset({
    "fused_adamw", "fused_lion", "fused_muon", "grad_clip_global_norm",
})

# bf16 is the training dtype; fp16 (loss-scaled mixed precision) is also swept.
DEFAULT_DTYPES: tuple[str, ...] = ("bf16", "fp16")
# Complete per-op dtype sweep (explicit so a generator can iterate OPS x dtypes).
OP_DTYPES: dict[str, tuple[str, ...]] = {op: DEFAULT_DTYPES for op in OPS}


def op_dtypes(op: str) -> tuple[str, ...]:
    """The dtype sweep for a breadth op (per-op override or the global default)."""
    return OP_DTYPES.get(op, DEFAULT_DTYPES)


# --------------------------------------------------------------------------- #
# Task constants (must match the seed kernels)
# --------------------------------------------------------------------------- #
LS_EPS = 0.1        # label-smoothing epsilon
NS_STEPS = 5        # Muon Newton-Schulz orthogonalization iterations
CLIP_EPS = 1e-6     # global-norm grad-clip denominator epsilon (torch default)
# Muon Newton-Schulz quintic coefficients (Keller Jordan; tuned, NOT convergent
# to exact orthogonal - the iteration keeps singular values bounded away from 0).
NS_COEFFS = (3.4445, -4.7750, 2.0315)


# --------------------------------------------------------------------------- #
# Real training shapes per op class (M = tokens, V = vocab, H = hidden, N = width)
# --------------------------------------------------------------------------- #
_CE_SHAPES = {  # logits[M, V] + targets[M] ; softmax/CE over the vocab V
    "minimal": {"M": 64, "V": 2048},
    "primary": {"M": 4096, "V": 32000},                    # Llama-2 vocab
    "validation": [{"M": 8192, "V": 32000}, {"M": 4096, "V": 128256},
                   {"M": 16384, "V": 32000}],              # wide batch, Llama-3 vocab, huge batch
}
_FLCE_SHAPES = {  # x[M, H], W[V, H], targets[M] ; fused (x @ W^T) -> CE, no logits materialized
    "minimal": {"M": 64, "H": 512, "V": 2048},
    "primary": {"M": 4096, "H": 4096, "V": 32000},
    "validation": [{"M": 8192, "H": 4096, "V": 32000}, {"M": 4096, "H": 4096, "V": 128256},
                   {"M": 2048, "H": 4096, "V": 128256}],   # Llama-3 LM head
}
_KL_SHAPES = {  # log_p[M, V], q[M, V] ; distillation KL over the vocab V
    "minimal": {"M": 64, "V": 2048},
    "primary": {"M": 4096, "V": 32000},
    "validation": [{"M": 8192, "V": 32000}, {"M": 4096, "V": 128256},
                   {"M": 16384, "V": 32000}],
}
_MSE_SHAPES = {  # input[M, N], target[M, N] ; per-element squared error, mean
    "minimal": {"M": 64, "N": 2048},
    "primary": {"M": 4096, "N": 4096},
    "validation": [{"M": 8192, "N": 4096}, {"M": 16384, "N": 4096},
                   {"M": 4096, "N": 8192}],
}
_OPT_SHAPES = {  # param[M, N] (+ grad + moments) ; a 2D weight matrix (Muon needs 2D)
    "minimal": {"M": 64, "N": 256},
    "primary": {"M": 4096, "N": 4096},                     # H x H hidden weight
    "validation": [{"M": 8192, "N": 4096}, {"M": 4096, "N": 14336},
                   {"M": 16384, "N": 4096}],               # tall, MLP up-proj, huge
}
_CLIP_SHAPES = {  # grads[G, N] ; G stacked grad tensors clipped by their GLOBAL L2 norm
    "minimal": {"G": 4, "N": 1024},
    "primary": {"G": 16, "N": 4096},
    "validation": [{"G": 8, "N": 8192}, {"G": 32, "N": 4096},
                   {"G": 16, "N": 14336}],
}

SHAPES: dict[str, dict] = {
    "cross_entropy": _CE_SHAPES, "label_smoothing_ce": _CE_SHAPES,
    "fused_linear_cross_entropy": _FLCE_SHAPES, "kl_div": _KL_SHAPES,
    "mse_loss": _MSE_SHAPES,
    "fused_adamw": _OPT_SHAPES, "fused_lion": _OPT_SHAPES, "fused_muon": _OPT_SHAPES,
    "grad_clip_global_norm": _CLIP_SHAPES,
}


# --------------------------------------------------------------------------- #
# reference.py namespace (torch fp32 oracle + torch eager perf baseline)
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import torch
    import torch.nn.functional as F

    tdt = getattr(torch, DTYPES[dtype][0])

    def _randn(shape, device, seed, scale=1.0):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape, generator=g, device=device, dtype=torch.float32) * scale).to(tdt)

    def _randn_sq(shape, device, seed, scale=1.0):
        """Non-negative fill (randn**2 * scale) for the AdamW 2nd-moment buffer."""
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape, generator=g, device=device, dtype=torch.float32) ** 2 * scale).to(tdt)

    def _targets(M, V, device, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.randint(0, V, (M,), generator=g, device=device, dtype=torch.int64)

    # ------------------------------------------------------------------ LOSSES
    if op == "cross_entropy":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _targets(M, V, device, seed + 1))

        def ref_fn(logits, targets):
            return F.cross_entropy(logits.float(), targets.long()).to(logits.dtype)

        def baseline_fn(logits, targets):
            return F.cross_entropy(logits, targets.long())

        arity = 2

    elif op == "fused_linear_cross_entropy":
        def get_inputs(shape, device="cuda", seed=0):
            M, H, V = shape["M"], shape["H"], shape["V"]
            x = _randn((M, H), device, seed, scale=1.0)
            w = _randn((V, H), device, seed + 1, scale=1.0 / (H ** 0.5))  # -> logits ~ N(0,1)
            return (x, w, _targets(M, V, device, seed + 2))

        def ref_fn(x, w, targets):
            logits = x.float() @ w.float().t()
            return F.cross_entropy(logits, targets.long()).to(x.dtype)

        def baseline_fn(x, w, targets):
            return F.cross_entropy(x @ w.t(), targets.long())

        arity = 3

    elif op == "kl_div":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            log_p = F.log_softmax(_randn((M, V), device, seed, scale=2.0).float(), dim=-1).to(tdt)
            q = F.softmax(_randn((M, V), device, seed + 1, scale=2.0).float(), dim=-1).to(tdt)
            return (log_p, q)

        def ref_fn(log_p, q):
            return F.kl_div(log_p.float(), q.float(), reduction="batchmean").to(log_p.dtype)

        def baseline_fn(log_p, q):
            return F.kl_div(log_p, q, reduction="batchmean")

        arity = 2

    elif op == "label_smoothing_ce":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _targets(M, V, device, seed + 1))

        def ref_fn(logits, targets):
            return F.cross_entropy(logits.float(), targets.long(),
                                   label_smoothing=LS_EPS).to(logits.dtype)

        def baseline_fn(logits, targets):
            return F.cross_entropy(logits, targets.long(), label_smoothing=LS_EPS)

        arity = 2

    elif op == "mse_loss":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), device, seed), _randn((M, N), device, seed + 1))

        def ref_fn(inp, target):
            return F.mse_loss(inp.float(), target.float()).to(inp.dtype)

        def baseline_fn(inp, target):
            return F.mse_loss(inp, target)

        arity = 2

    # -------------------------------------------------------------- OPTIMIZERS
    elif op == "fused_adamw":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            param = _randn((M, N), device, seed, scale=0.02)
            grad = _randn((M, N), device, seed + 1, scale=0.01)
            exp_avg = _randn((M, N), device, seed + 2, scale=0.01)
            exp_avg_sq = _randn_sq((M, N), device, seed + 3, scale=0.001)
            # (lr, beta1, beta2, eps, weight_decay, step) - realistic AdamW hyperparams
            return (param, grad, exp_avg, exp_avg_sq, 1e-3, 0.9, 0.999, 1e-8, 0.01, 10)

        def ref_fn(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, wd, step):
            # EXACT torch.optim.AdamW update math (decoupled weight decay), fp32.
            p = param.float() * (1.0 - lr * wd)
            m = beta1 * exp_avg.float() + (1.0 - beta1) * grad.float()
            v = beta2 * exp_avg_sq.float() + (1.0 - beta2) * grad.float() * grad.float()
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            denom = v.sqrt() / (bc2 ** 0.5) + eps
            p = p - (lr / bc1) * m / denom
            dt = param.dtype
            return (p.to(dt), m.to(dt), v.to(dt))

        def baseline_fn(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, wd, step):
            # Eager multi-kernel torch step at the native dtype (perf bar to beat).
            p = param * (1.0 - lr * wd)
            m = beta1 * exp_avg + (1.0 - beta1) * grad
            v = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            denom = v.sqrt() / (bc2 ** 0.5) + eps
            p = p - (lr / bc1) * m / denom
            return (p, m, v)

        arity = 10

    elif op == "fused_lion":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            param = _randn((M, N), device, seed, scale=0.02)
            grad = _randn((M, N), device, seed + 1, scale=0.01)
            exp_avg = _randn((M, N), device, seed + 2, scale=0.01)
            # (lr, beta1, beta2, weight_decay) - Lion hyperparams
            return (param, grad, exp_avg, 1e-4, 0.9, 0.99, 0.01)

        def ref_fn(param, grad, exp_avg, lr, beta1, beta2, wd):
            # Lion (Chen et al. 2023): sign of a beta1-interpolated update; decoupled
            # weight decay; the momentum EMA is updated with beta2 (both use the OLD m).
            p = param.float()
            g = grad.float()
            m = exp_avg.float()
            update = torch.sign(beta1 * m + (1.0 - beta1) * g)
            p = p - lr * (update + wd * p)
            m = beta2 * m + (1.0 - beta2) * g
            dt = param.dtype
            return (p.to(dt), m.to(dt))

        def baseline_fn(param, grad, exp_avg, lr, beta1, beta2, wd):
            update = torch.sign(beta1 * exp_avg + (1.0 - beta1) * grad)
            p = param - lr * (update + wd * param)
            m = beta2 * exp_avg + (1.0 - beta2) * grad
            return (p, m)

        arity = 7

    elif op == "fused_muon":
        def _newton_schulz5(gm):
            # Orthogonalize gm via the quintic Newton-Schulz iteration (NS_STEPS iters):
            # X_{k+1} = a*X + (b*A + c*A@A) @ X, A = X@X^T, on the norm-normalized matrix.
            # Preserves gm's singular VECTORS, maps its singular VALUES toward ~1.
            a, b, c = NS_COEFFS
            x = gm.float()
            transposed = False
            if x.shape[-2] > x.shape[-1]:
                x = x.mT
                transposed = True
            x = x / (x.norm() + 1e-7)
            for _ in range(NS_STEPS):
                aa = x @ x.mT
                bb = b * aa + c * (aa @ aa)
                x = a * x + bb @ x
            if transposed:
                x = x.mT
            return x

        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            param = _randn((M, N), device, seed, scale=0.02)
            grad = _randn((M, N), device, seed + 1, scale=1.0)
            momentum_buffer = _randn((M, N), device, seed + 2, scale=1.0)
            # (lr, momentum) - Muon hyperparams
            return (param, grad, momentum_buffer, 2e-2, 0.95)

        def ref_fn(param, grad, momentum_buffer, lr, momentum):
            p = param.float()
            g = grad.float()
            buf = momentum * momentum_buffer.float() + (1.0 - momentum) * g   # buf.lerp_(g, 1-mu)
            g_eff = (1.0 - momentum) * g + momentum * buf                     # nesterov g.lerp_(buf, mu)
            o = _newton_schulz5(g_eff)
            scale = max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5                # aspect-ratio scale
            p = p - lr * scale * o
            dt = param.dtype
            return (p.to(dt), buf.to(dt))

        def baseline_fn(param, grad, momentum_buffer, lr, momentum):
            buf = momentum * momentum_buffer + (1.0 - momentum) * grad
            g_eff = (1.0 - momentum) * grad + momentum * buf
            o = _newton_schulz5(g_eff).to(param.dtype)
            scale = max(1.0, param.shape[-2] / param.shape[-1]) ** 0.5
            p = param - lr * scale * o
            return (p, buf)

        arity = 5

    elif op == "grad_clip_global_norm":
        def get_inputs(shape, device="cuda", seed=0):
            G, N = shape["G"], shape["N"]
            grads = _randn((G, N), device, seed, scale=1.0)
            # (max_norm,) - clip threshold
            return (grads, 1.0)

        def ref_fn(grads, max_norm):
            # torch.nn.utils.clip_grad_norm_ math: global L2 over ALL elements, then
            # scale by min(max_norm / (total_norm + eps), 1.0). (Frobenius of the
            # stack == the global L2 norm of the concatenated grads.)
            gf = grads.float()
            total_norm = gf.norm()
            coef = torch.clamp(max_norm / (total_norm + CLIP_EPS), max=1.0)
            return (gf * coef).to(grads.dtype)

        def baseline_fn(grads, max_norm):
            total_norm = grads.float().norm()
            coef = torch.clamp(max_norm / (total_norm + CLIP_EPS), max=1.0)
            return grads * coef.to(grads.dtype)

        arity = 2

    else:
        raise ValueError(f"unknown breadth op {op!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op, "dtype_name": dtype,
          "family": f"breadth_{op}", "mutates_input": op in TRAIN_MUTATES_INPUT}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Naive compiling+correct Triton starter seeds (the policy optimizes these)
# --------------------------------------------------------------------------- #
_CROSS_ENTROPY_SEED = '''"""GENERATED breadth cross_entropy seed ({dtype}). logits[M,V] + targets[M] -> mean CE.
One program per row: streaming (online) fp32 logsumexp so any vocab width V fits;
loss[m] = logsumexp(logits[m]) - logits[m, target[m]]; the row losses are then
mean-reduced. Naive starting point (per-row + a torch mean); the policy fuses it."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cross_entropy_kernel(logits_ptr, tgt_ptr, loss_ptr, sm, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    m = -float("inf")
    s = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m
    lse = m + tl.log(s)
    tgt = tl.load(tgt_ptr + row)
    xt = tl.load(logits_ptr + base + tgt).to(tl.float32)
    tl.store(loss_ptr + row, lse - xt)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    loss = torch.empty((M,), device=logits.device, dtype=torch.float32)
    _cross_entropy_kernel[(M,)](logits, targets, loss, logits.stride(0), V,
                                BLOCK=1024, num_warps=8)
    return loss.mean().to(logits.dtype)
'''


_FLCE_SEED = '''"""GENERATED breadth fused_linear_cross_entropy seed ({dtype}).
x[M,H], W[V,H], targets[M] -> mean CE of logits = x @ W^T. Naive TWO-pass seed:
a tiled fp32 GEMM materializes logits[M,V], then a streaming logsumexp computes the
row CE. The Liger-style FUSION (never materialize [M,V]) is the optimization target."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _flce_gemm_kernel(a_ptr, b_ptr, c_ptr, M, V, H,
                      sam, sah, sbv, sbh, scm, scv,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + (offs_m[:, None] * sam + offs_k[None, :] * sah)
    b_ptrs = b_ptr + (offs_n[None, :] * sbv + offs_k[:, None] * sbh)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(H, BK)):
        kmask = offs_k < H - k * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < V) & kmask[:, None], other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
        a_ptrs += BK * sah
        b_ptrs += BK * sbh
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scv
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < V))


@triton.jit
def _flce_ce_kernel(logits_ptr, tgt_ptr, loss_ptr, sm, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    m = -float("inf")
    s = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m
    lse = m + tl.log(s)
    tgt = tl.load(tgt_ptr + row)
    xt = tl.load(logits_ptr + base + tgt).to(tl.float32)
    tl.store(loss_ptr + row, lse - xt)


def fused_linear_cross_entropy(x: torch.Tensor, weight: torch.Tensor,
                               targets: torch.Tensor) -> torch.Tensor:
    M, H = x.shape
    V = weight.shape[0]
    logits = torch.empty((M, V), device=x.device, dtype=torch.float32)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(V, BN))
    _flce_gemm_kernel[grid](x, weight, logits, M, V, H,
                            x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
                            logits.stride(0), logits.stride(1),
                            BM=BM, BN=BN, BK=BK, num_warps=4)
    loss = torch.empty((M,), device=x.device, dtype=torch.float32)
    _flce_ce_kernel[(M,)](logits, targets, loss, logits.stride(0), V, BLOCK=1024, num_warps=8)
    return loss.mean().to(x.dtype)
'''


_KL_DIV_SEED = '''"""GENERATED breadth kl_div seed ({dtype}). log_p[M,V], q[M,V] -> KL(q || p) batchmean.
Per-row fp32 sum of q*(log q - log_p) (0*log0 -> 0), then sum over rows / M (batch).
Matches F.kl_div(log_p, q, reduction='batchmean')."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _kl_div_kernel(logp_ptr, q_ptr, out_ptr, sm, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    acc = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        lp = tl.load(logp_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        q = tl.load(q_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        term = tl.where(q > 0.0, q * (tl.log(q) - lp), 0.0)
        acc += tl.sum(term, axis=0)
    tl.store(out_ptr + row, acc)


def kl_div(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    M, V = log_p.shape
    rows = torch.empty((M,), device=log_p.device, dtype=torch.float32)
    _kl_div_kernel[(M,)](log_p, q, rows, log_p.stride(0), V, BLOCK=1024, num_warps=8)
    return (rows.sum() / M).to(log_p.dtype)
'''


_LABEL_SMOOTHING_CE_SEED = '''"""GENERATED breadth label_smoothing_ce seed ({dtype}). logits[M,V] + targets[M].
Per-row streaming fp32 pass tracks logsumexp AND the row sum of logits, so
loss[m] = (1-eps)*(lse - logit[target]) + eps*(lse - mean_v logit); eps={ls_eps}.
Matches F.cross_entropy(..., label_smoothing=eps). Then mean over rows."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _label_smoothing_ce_kernel(logits_ptr, tgt_ptr, loss_ptr, sm, V, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    m = -float("inf")
    s = 0.0
    ssum = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m
        xs = tl.load(logits_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        ssum += tl.sum(xs, axis=0)
    lse = m + tl.log(s)
    tgt = tl.load(tgt_ptr + row)
    xt = tl.load(logits_ptr + base + tgt).to(tl.float32)
    nll = lse - xt
    smooth = lse - ssum / V
    tl.store(loss_ptr + row, (1.0 - eps) * nll + eps * smooth)


def label_smoothing_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    loss = torch.empty((M,), device=logits.device, dtype=torch.float32)
    _label_smoothing_ce_kernel[(M,)](logits, targets, loss, logits.stride(0), V,
                                     {ls_eps}, BLOCK=1024, num_warps=8)
    return loss.mean().to(logits.dtype)
'''


_MSE_LOSS_SEED = '''"""GENERATED breadth mse_loss seed ({dtype}). input[M,N], target[M,N] -> mean((a-b)^2).
Per-row fp32 sum of squared error, then sum over rows / (M*N). Matches F.mse_loss."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _mse_loss_kernel(a_ptr, b_ptr, out_ptr, sm, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    acc = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        d = a - b
        acc += tl.sum(d * d, axis=0)
    tl.store(out_ptr + row, acc)


def mse_loss(inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    M, N = inp.shape
    rows = torch.empty((M,), device=inp.device, dtype=torch.float32)
    _mse_loss_kernel[(M,)](inp, target, rows, inp.stride(0), N, BLOCK=1024, num_warps=8)
    return (rows.sum() / (M * N)).to(inp.dtype)
'''


_FUSED_ADAMW_SEED = '''"""GENERATED breadth fused_adamw seed ({dtype}). One decoupled AdamW step, fused,
UPDATING param + exp_avg + exp_avg_sq IN PLACE. Elementwise fp32 math; the bias
corrections (1-beta**step) are precomputed host-side and passed as step_size /
bc2_sqrt. Matches torch.optim.AdamW. Regenerate/optimize from this naive seed."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fused_adamw_kernel(p_ptr, g_ptr, m_ptr, v_ptr, numel,
                        lr, wd, beta1, beta2, eps, step_size, bc2_sqrt, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=mask).to(tl.float32)
    p = p * (1.0 - lr * wd)
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    denom = tl.sqrt(v) / bc2_sqrt + eps
    p = p - step_size * m / denom
    tl.store(p_ptr + offs, p.to({tldt}), mask=mask)
    tl.store(m_ptr + offs, m.to({tldt}), mask=mask)
    tl.store(v_ptr + offs, v.to({tldt}), mask=mask)


def fused_adamw(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, wd, step):
    numel = param.numel()
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    step_size = lr / bc1
    bc2_sqrt = bc2 ** 0.5
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _fused_adamw_kernel[grid](param, grad, exp_avg, exp_avg_sq, numel,
                              lr, wd, beta1, beta2, eps, step_size, bc2_sqrt,
                              BLOCK=BLOCK, num_warps=4)
    return param, exp_avg, exp_avg_sq
'''


_FUSED_LION_SEED = '''"""GENERATED breadth fused_lion seed ({dtype}). One Lion step, fused, UPDATING
param + exp_avg IN PLACE. update = sign(beta1*m + (1-beta1)*g); param -= lr*(update
+ wd*param); then m = beta2*m + (1-beta2)*g (both use the OLD m). Elementwise fp32."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fused_lion_kernel(p_ptr, g_ptr, m_ptr, numel, lr, beta1, beta2, wd, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask).to(tl.float32)
    c = beta1 * m + (1.0 - beta1) * g
    upd = tl.where(c > 0.0, 1.0, tl.where(c < 0.0, -1.0, 0.0))
    p = p - lr * (upd + wd * p)
    m = beta2 * m + (1.0 - beta2) * g
    tl.store(p_ptr + offs, p.to({tldt}), mask=mask)
    tl.store(m_ptr + offs, m.to({tldt}), mask=mask)


def fused_lion(param, grad, exp_avg, lr, beta1, beta2, wd):
    numel = param.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _fused_lion_kernel[grid](param, grad, exp_avg, numel, lr, beta1, beta2, wd,
                             BLOCK=BLOCK, num_warps=4)
    return param, exp_avg
'''


_FUSED_MUON_SEED = '''"""GENERATED breadth fused_muon seed ({dtype}). One Muon step on a 2D param, UPDATING
param + momentum_buffer IN PLACE. Triton elementwise kernels do the (nesterov)
momentum accumulation and the final scaled update; the {ns_steps}-iter Newton-Schulz
orthogonalization (the quintic X = a*X + (b*A + c*A@A)@X) runs as torch matmuls in
fp32 - FUSING those matmuls into Triton is the optimization target."""
from __future__ import annotations
import torch, triton, triton.language as tl

_NS_A, _NS_B, _NS_C = {ns_a}, {ns_b}, {ns_c}
_NS_STEPS = {ns_steps}


@triton.jit
def _muon_momentum_kernel(g_ptr, buf_ptr, geff_ptr, numel, momentum, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    buf = tl.load(buf_ptr + offs, mask=mask).to(tl.float32)
    buf = momentum * buf + (1.0 - momentum) * g
    geff = (1.0 - momentum) * g + momentum * buf
    tl.store(buf_ptr + offs, buf.to({tldt}), mask=mask)
    tl.store(geff_ptr + offs, geff, mask=mask)


@triton.jit
def _muon_update_kernel(p_ptr, o_ptr, numel, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    o = tl.load(o_ptr + offs, mask=mask).to(tl.float32)
    p = p - alpha * o
    tl.store(p_ptr + offs, p.to({tldt}), mask=mask)


def _newton_schulz5(gm):
    x = gm.float()
    transposed = False
    if x.shape[-2] > x.shape[-1]:
        x = x.mT
        transposed = True
    x = x / (x.norm() + 1e-7)
    for _ in range(_NS_STEPS):
        a = x @ x.mT
        b = _NS_B * a + _NS_C * (a @ a)
        x = _NS_A * x + b @ x
    if transposed:
        x = x.mT
    return x


def fused_muon(param, grad, momentum_buffer, lr, momentum):
    M, N = param.shape
    numel = param.numel()
    geff = torch.empty((M, N), device=param.device, dtype=torch.float32)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _muon_momentum_kernel[grid](grad, momentum_buffer, geff, numel, momentum,
                                BLOCK=BLOCK, num_warps=4)
    o = _newton_schulz5(geff).contiguous()
    scale = max(1.0, M / N) ** 0.5
    _muon_update_kernel[grid](param, o, numel, lr * scale, BLOCK=BLOCK, num_warps=4)
    return param, momentum_buffer
'''


_GRAD_CLIP_SEED = '''"""GENERATED breadth grad_clip_global_norm seed ({dtype}). grads[G,N] clipped by their
GLOBAL L2 norm IN PLACE. Kernel 1: per-row fp32 sum-of-squares -> [G]; host: total
norm + coef = min(max_norm/(total+{clip_eps}), 1). Kernel 2: scale grads by coef.
Matches torch.nn.utils.clip_grad_norm_ over the stacked grads."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _gc_sumsq_kernel(g_ptr, part_ptr, sm, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    acc = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(g_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    tl.store(part_ptr + row, acc)


@triton.jit
def _gc_scale_kernel(g_ptr, numel, coef, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    tl.store(g_ptr + offs, (x * coef).to({tldt}), mask=mask)


def grad_clip_global_norm(grads: torch.Tensor, max_norm) -> torch.Tensor:
    G, N = grads.shape
    part = torch.empty((G,), device=grads.device, dtype=torch.float32)
    _gc_sumsq_kernel[(G,)](grads, part, grads.stride(0), N, BLOCK=1024, num_warps=8)
    total_norm = torch.sqrt(part.sum())
    coef = min(max_norm / (total_norm.item() + {clip_eps}), 1.0)
    numel = grads.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _gc_scale_kernel[grid](grads, numel, coef, BLOCK=BLOCK, num_warps=4)
    return grads
'''


def seed_source(op: str, dtype: str) -> str:
    tldt = DTYPES[dtype][1]
    if op == "cross_entropy":
        return _CROSS_ENTROPY_SEED.format(dtype=dtype, tldt=tldt)
    if op == "fused_linear_cross_entropy":
        return _FLCE_SEED.format(dtype=dtype, tldt=tldt)
    if op == "kl_div":
        return _KL_DIV_SEED.format(dtype=dtype, tldt=tldt)
    if op == "label_smoothing_ce":
        return _LABEL_SMOOTHING_CE_SEED.format(dtype=dtype, tldt=tldt, ls_eps=LS_EPS)
    if op == "mse_loss":
        return _MSE_LOSS_SEED.format(dtype=dtype, tldt=tldt)
    if op == "fused_adamw":
        return _FUSED_ADAMW_SEED.format(dtype=dtype, tldt=tldt)
    if op == "fused_lion":
        return _FUSED_LION_SEED.format(dtype=dtype, tldt=tldt)
    if op == "fused_muon":
        return _FUSED_MUON_SEED.format(dtype=dtype, tldt=tldt, ns_steps=NS_STEPS,
                                       ns_a=NS_COEFFS[0], ns_b=NS_COEFFS[1], ns_c=NS_COEFFS[2])
    if op == "grad_clip_global_norm":
        return _GRAD_CLIP_SEED.format(dtype=dtype, tldt=tldt, clip_eps=CLIP_EPS)
    raise ValueError(f"unknown breadth op {op!r}")
