"""Breadth REDUCTION / NORMALIZATION task-authoring engine (torch-baselined).

Widens the KORE suite with the HARD frontier of row-reductions: the numerically
unstable / streaming / multi-pass-fused reductions that dominate a transformer's
loss head, normalization layers and sampling tail but where a NAIVE implementation
is either wrong (overflow / catastrophic cancellation) or slow (many HBM passes).
These are the honest "real headroom" reduction kernels - no trivial single-op
sums: every op here is a stable max-subtracted softmax/log-sum-exp, a Welford /
centered-variance statistic, a stable cross-entropy over a large vocab (+ its
backward), a distribution divergence, a parallel top-k / top-p, an associative
cumulative scan, or a normalized Lp reduction.

Op families (every name is prefixed ``red_``)
---------------------------------------------
  * softmax / log_softmax over the last dim (+ temperature, online/flash, over
    dim=0), and their BACKWARD given the saved forward output + upstream grad.
  * streaming max-subtracted log-sum-exp (last dim & dim=0), Shannon entropy,
    stable log-cumsum-exp scan, Gumbel-softmax.
  * Welford / centered variance (biased & unbiased), std, RMS, RMS-norm,
    layer-norm, batchnorm-style running mean/var (reduce over dim=0).
  * stable cross-entropy from logits over a large vocab (+ backward), label
    smoothing, z-loss (log-Z^2 penalty), CE+z-loss, focal loss, soft-label CE,
    binary-cross-entropy-with-logits.
  * KL / Jensen-Shannon divergence over softmax distributions.
  * parallel top-k (k in {2,8,50,256}), top-p (nucleus) mask+renormalize,
    argmax / argmin (first-occurrence tie rule), cumulative max / min scans.
  * row-wise Lp norms (p=1,2,inf), L2-normalize, pairwise Euclidean distance,
    cosine similarity.

Contract (mirrors ``kore.tasks.vendor_ops`` and the sibling breadth engines so the
generic ``kore.tasks._genops`` driver + generator consume it unchanged):

    OPS / OP_DTYPES / SHAPES              module-level task catalog
    make_reference(op, dtype) -> dict     reference.py namespace (parse_shape,
        get_inputs, ref_fn fp32 STABLE oracle [may return a tuple], baseline_fn
        torch path, arity, entry_name, dtype_name, family=f"breadth_{op}",
        mutates_input)
    seed_source(op, dtype) -> str         a naive, COMPILING, CORRECT Triton seed
        (fp32 accumulate, block-wise / streaming reduction) - the policy's start.

CORRECTNESS is paramount and EXACT: every ``ref_fn`` computes the numerically
STABLE formula in fp32 (max-subtracted softmax/lse, centered/Welford variance,
log1p-stable BCE, ...) and casts back to the task dtype. tests/test_reduce_ext.py
cross-checks every ``ref_fn`` against an INDEPENDENT torch computation
(torch.log_softmax / F.cross_entropy / torch.var / torch.logsumexp / autograd /
manual scans) at a tight fp32 tolerance, INCLUDING an extreme-magnitude input case
that a naive (non-max-subtracted / E[x^2]-E[x]^2) implementation would fail. The
naive Triton seeds stream the reduction in fp32 blocks (numerically stable) - a
correct-but-slow starting point the policy learns to fuse / parallelize.

torch/triton are imported lazily (registry discovery never needs a GPU).
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# --------------------------------------------------------------------------- #
# task catalog (every op is prefixed ``red_``)
# --------------------------------------------------------------------------- #
OPS: tuple[str, ...] = (
    # softmax / normalization over the last dim (+ temperature, online, dim=0, gumbel)
    "red_softmax", "red_log_softmax", "red_softmax_temp", "red_online_softmax",
    "red_softmax_dim0", "red_gumbel_softmax",
    # softmax / log_softmax BACKWARD (given saved forward output + upstream dy)
    "red_softmax_bwd", "red_log_softmax_bwd",
    # log-sum-exp / entropy / stable cumulative log-sum-exp (streaming, max-subtracted)
    "red_logsumexp", "red_logsumexp_dim0", "red_entropy", "red_logcumsumexp",
    # Welford variance / normalization statistics
    "red_var", "red_var_unbiased", "red_std", "red_welford", "red_rms",
    "red_rmsnorm", "red_layernorm", "red_running_stats",
    # cross-entropy / losses over a large vocab (+ backward)
    "red_cross_entropy", "red_cross_entropy_bwd", "red_label_smoothing_ce",
    "red_z_loss", "red_cross_entropy_zloss", "red_focal_loss",
    "red_soft_cross_entropy", "red_bce_with_logits",
    # divergences over distributions
    "red_kl_div", "red_js_div",
    # parallel top-k / top-p / arg-reduce / cumulative max-min
    "red_topk2", "red_topk8", "red_topk50", "red_topk256", "red_topp_renorm",
    "red_argmax", "red_argmin", "red_cummax", "red_cummin",
    # row-wise Lp norms / normalize / pairwise reductions
    "red_norm_l1", "red_norm_l2", "red_norm_linf", "red_l2_normalize",
    "red_pairwise_dist", "red_cosine_sim",
)

# Swept over the two serving activation dtypes plus fp32 (the fp32 oracle casts
# back). Materialized dict so a generator can iterate OPS x dtypes directly.
DEFAULT_DTYPES: tuple[str, ...] = ("bf16", "fp16", "fp32")
OP_DTYPES: dict[str, tuple[str, ...]] = {op: DEFAULT_DTYPES for op in OPS}


def op_dtypes(op: str) -> tuple[str, ...]:
    """The dtype sweep for a breadth op (per-op override or the global default)."""
    return OP_DTYPES.get(op, DEFAULT_DTYPES)


# --------------------------------------------------------------------------- #
# task hyper-params (baked into BOTH the fp32 oracle and the seed defaults)
# --------------------------------------------------------------------------- #
TEMP = 2.0            # softmax temperature (logits divided by TEMP)
GUMBEL_TAU = 0.5      # Gumbel-softmax temperature
LS_EPS = 0.1          # label-smoothing epsilon
FOCAL_GAMMA = 2.0     # focal-loss focusing parameter (integer 2 -> (1-pt)^2)
ZLOSS_COEF = 1e-4     # PaLM-style z-loss coefficient (coef * logsumexp^2)
LN_EPS = 1e-5         # layer-norm epsilon
RMS_EPS = 1e-6        # RMS-norm epsilon
NORM_EPS = 1e-12      # L2-normalize denominator floor (matches F.normalize)
COS_EPS = 1e-8        # cosine-similarity denominator floor (matches F.cosine_similarity)
TOPP_P = 0.9          # nucleus (top-p) mass threshold
TOPK_SIZES: dict[str, int] = {
    "red_topk2": 2, "red_topk8": 8, "red_topk50": 50, "red_topk256": 256,
}

# --------------------------------------------------------------------------- #
# Realistic shapes: rows M in {4096, 16384}, wide N in {8192, 32768, 131072}
# incl. large-vocab, plus a non-power-of-2 tail. Row ops reduce over the last
# dim; dim=0 ops reduce over the M (batch) axis.
# --------------------------------------------------------------------------- #
_ROW = {  # x[M, N] reduce over the last dim N
    "minimal": {"M": 64, "N": 256},
    "primary": {"M": 4096, "N": 8192},
    "validation": [
        {"M": 16384, "N": 8192},
        {"M": 4096, "N": 32768},
        {"M": 4096, "N": 131072},   # very wide reduction (large-vocab class)
        {"M": 4096, "N": 8191},     # non-pow2 tail
    ],
}
_VOCAB = {  # logits[M, V] reduce over the vocab V (cross-entropy / divergence class)
    "minimal": {"M": 64, "V": 2048},
    "primary": {"M": 4096, "V": 32000},                       # Llama-2 vocab
    "validation": [
        {"M": 16384, "V": 32000},                             # huge batch
        {"M": 4096, "V": 128256},                             # Llama-3 vocab
        {"M": 4096, "V": 32001},                              # non-pow2 tail
    ],
}
_TOPK_SHAPES = {  # top-k needs N >= k; keep a non-pow2 tail
    "minimal": {"M": 64, "N": 512},
    "primary": {"M": 4096, "N": 8192},
    "validation": [
        {"M": 16384, "N": 8192},
        {"M": 4096, "N": 32768},
        {"M": 4096, "N": 8191},
    ],
}

_LASTDIM_OPS = frozenset({
    "red_softmax", "red_log_softmax", "red_softmax_temp", "red_online_softmax",
    "red_softmax_dim0", "red_gumbel_softmax", "red_softmax_bwd", "red_log_softmax_bwd",
    "red_logsumexp", "red_logsumexp_dim0", "red_entropy", "red_logcumsumexp",
    "red_var", "red_var_unbiased", "red_std", "red_welford", "red_rms",
    "red_rmsnorm", "red_layernorm", "red_running_stats",
    "red_bce_with_logits",
    "red_topp_renorm", "red_argmax", "red_argmin", "red_cummax", "red_cummin",
    "red_norm_l1", "red_norm_l2", "red_norm_linf", "red_l2_normalize",
    "red_pairwise_dist", "red_cosine_sim",
})
_VOCAB_OPS = frozenset({
    "red_cross_entropy", "red_cross_entropy_bwd", "red_label_smoothing_ce",
    "red_z_loss", "red_cross_entropy_zloss", "red_focal_loss",
    "red_soft_cross_entropy", "red_kl_div", "red_js_div",
})

SHAPES: dict[str, dict] = {}
for _op in OPS:
    if _op in TOPK_SIZES:
        SHAPES[_op] = _TOPK_SHAPES
    elif _op in _VOCAB_OPS:
        SHAPES[_op] = _VOCAB
    else:
        SHAPES[_op] = _ROW


# --------------------------------------------------------------------------- #
# reference.py namespace (fp32 STABLE oracle + torch eager perf baseline)
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import torch
    import torch.nn.functional as F

    tdt = getattr(torch, DTYPES[dtype][0])

    # -------- input generators (fp32 -> task dtype) ------------------------- #
    def _randn(shape, device, seed, scale=1.0):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape, generator=g, device=device, dtype=torch.float32) * scale).to(tdt)

    def _rand01(shape, device, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.rand(shape, generator=g, device=device, dtype=torch.float32).to(tdt)

    def _targets(M, V, device, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.randint(0, V, (M,), generator=g, device=device, dtype=torch.int64)

    def _gumbel(shape, device, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        u = torch.rand(shape, generator=g, device=device, dtype=torch.float32).clamp_(1e-9, 1.0)
        return (-torch.log(-torch.log(u))).to(tdt)

    # -------- stable fp32 primitives (max-subtracted / centered) ----------- #
    def _sm(t, dim=-1):
        m = t.amax(dim=dim, keepdim=True)
        e = torch.exp(t - m)
        return e / e.sum(dim=dim, keepdim=True)

    def _lsm(t, dim=-1):
        m = t.amax(dim=dim, keepdim=True)
        z = t - m
        return z - torch.log(torch.exp(z).sum(dim=dim, keepdim=True))

    def _lse(t, dim=-1):
        m = t.amax(dim=dim, keepdim=True)
        return (m + torch.log(torch.exp(t - m).sum(dim=dim, keepdim=True))).squeeze(dim)

    def _gather(xf, targets):
        return xf.gather(-1, targets.long().view(-1, 1)).squeeze(-1)

    def _nucleus(probs, p):
        sp, si = torch.sort(probs, dim=-1, descending=True)
        excl = sp.cumsum(dim=-1) - sp                 # exclusive prefix mass
        keep_sorted = excl <= p                        # always keeps the crossing token
        keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, si, keep_sorted)
        masked = torch.where(keep, probs, torch.zeros_like(probs))
        return masked / masked.sum(dim=-1, keepdim=True)

    def _argfirst(t, largest):
        ext = t.amax(-1, keepdim=True) if largest else t.amin(-1, keepdim=True)
        is_ext = t == ext
        n = t.shape[-1]
        idxs = torch.arange(n, device=t.device)
        big = torch.where(is_ext, idxs, torch.full_like(idxs, n))
        return big.amin(-1)                            # first-occurrence index (int64)

    def _rn(shape):
        return shape["N"] if "N" in shape else shape["V"]

    # ===================================================================== #
    # SOFTMAX / NORMALIZATION over the last dim
    # ===================================================================== #
    if op in ("red_softmax", "red_online_softmax"):
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            return _sm(x.float()).to(x.dtype)

        def baseline_fn(x):
            return torch.softmax(x, dim=-1)

        arity = 1

    elif op == "red_log_softmax":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            return _lsm(x.float()).to(x.dtype)

        def baseline_fn(x):
            return torch.log_softmax(x, dim=-1)

        arity = 1

    elif op == "red_softmax_temp":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            return _sm(x.float() / TEMP).to(x.dtype)

        def baseline_fn(x):
            return torch.softmax(x / TEMP, dim=-1)

        arity = 1

    elif op == "red_softmax_dim0":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            return _sm(x.float(), dim=0).to(x.dtype)

        def baseline_fn(x):
            return torch.softmax(x, dim=0)

        arity = 1

    elif op == "red_gumbel_softmax":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], _rn(shape)
            return (_randn((M, N), device, seed, scale=2.0), _gumbel((M, N), device, seed + 1))

        def ref_fn(logits, gumbel):
            return _sm((logits.float() + gumbel.float()) / GUMBEL_TAU).to(logits.dtype)

        def baseline_fn(logits, gumbel):
            return torch.softmax((logits + gumbel) / GUMBEL_TAU, dim=-1)

        arity = 2

    # ---------------------- softmax / log_softmax BACKWARD ---------------- #
    elif op == "red_softmax_bwd":
        # Given the saved forward y = softmax(x) and upstream dy -> dx (the softmax
        # Jacobian-vector product). dx_j = y_j * (dy_j - sum_k y_k dy_k).
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], _rn(shape)
            y = _sm(_randn((M, N), device, seed, scale=2.0).float()).to(tdt)
            dy = _randn((M, N), device, seed + 1)
            return (y, dy)

        def ref_fn(y, dy):
            yf, dyf = y.float(), dy.float()
            return (yf * (dyf - (yf * dyf).sum(-1, keepdim=True))).to(y.dtype)

        def baseline_fn(y, dy):
            return y * (dy - (y * dy).sum(-1, keepdim=True))

        arity = 2

    elif op == "red_log_softmax_bwd":
        # Given the saved forward y = log_softmax(x) and upstream dy -> dx.
        # dx_j = dy_j - exp(y_j) * sum_k dy_k.
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], _rn(shape)
            y = _lsm(_randn((M, N), device, seed, scale=2.0).float()).to(tdt)
            dy = _randn((M, N), device, seed + 1)
            return (y, dy)

        def ref_fn(y, dy):
            yf, dyf = y.float(), dy.float()
            return (dyf - torch.exp(yf) * dyf.sum(-1, keepdim=True)).to(y.dtype)

        def baseline_fn(y, dy):
            return dy - torch.exp(y) * dy.sum(-1, keepdim=True)

        arity = 2

    # ===================================================================== #
    # LOG-SUM-EXP / ENTROPY / STABLE CUMULATIVE LOG-SUM-EXP
    # ===================================================================== #
    elif op == "red_logsumexp":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            return _lse(x.float()).to(x.dtype)

        def baseline_fn(x):
            return torch.logsumexp(x, dim=-1)

        arity = 1

    elif op == "red_logsumexp_dim0":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            return _lse(x.float(), dim=0).to(x.dtype)

        def baseline_fn(x):
            return torch.logsumexp(x, dim=0)

        arity = 1

    elif op == "red_entropy":
        # Shannon entropy of softmax(logits): H = -sum_j p_j log p_j = lse - sum_j p_j x_j.
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            xf = x.float()
            logp = _lsm(xf)
            p = torch.exp(logp)
            return (-(p * logp).sum(-1)).to(x.dtype)

        def baseline_fn(x):
            p = torch.softmax(x, dim=-1)
            return -(p * torch.log_softmax(x, dim=-1)).sum(-1)

        arity = 1

    elif op == "red_logcumsumexp":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            return torch.logcumsumexp(x.float(), dim=-1).to(x.dtype)

        def baseline_fn(x):
            return torch.logcumsumexp(x, dim=-1)

        arity = 1

    # ===================================================================== #
    # WELFORD VARIANCE / NORMALIZATION STATISTICS
    # ===================================================================== #
    elif op in ("red_var", "red_var_unbiased", "red_std"):
        _unbiased = op != "red_var"

        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            xf = x.float()
            mean = xf.mean(-1, keepdim=True)
            sq = ((xf - mean) ** 2).sum(-1)             # centered (stable), no cancellation
            denom = (xf.shape[-1] - 1) if _unbiased else xf.shape[-1]
            var = sq / denom
            out = torch.sqrt(var) if op == "red_std" else var
            return out.to(x.dtype)

        def baseline_fn(x):
            if op == "red_std":
                return torch.std(x, dim=-1, unbiased=True)
            return torch.var(x, dim=-1, unbiased=_unbiased)

        arity = 1

    elif op == "red_welford":
        # One-pass Welford (returns BOTH mean and biased variance per row).
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            xf = x.float()
            mean = xf.mean(-1)
            var = ((xf - mean.unsqueeze(-1)) ** 2).mean(-1)
            return (mean.to(x.dtype), var.to(x.dtype))

        def baseline_fn(x):
            return (x.mean(-1), x.var(-1, unbiased=False))

        arity = 1

    elif op == "red_rms":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            xf = x.float()
            return torch.sqrt((xf * xf).mean(-1)).to(x.dtype)

        def baseline_fn(x):
            return x.pow(2).mean(-1).sqrt()

        arity = 1

    elif op == "red_rmsnorm":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            xf = x.float()
            ms = (xf * xf).mean(-1, keepdim=True)
            return (xf * torch.rsqrt(ms + RMS_EPS)).to(x.dtype)

        def baseline_fn(x):
            xf = x.float()
            return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)).to(x.dtype)

        arity = 1

    elif op == "red_layernorm":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            xf = x.float()
            mean = xf.mean(-1, keepdim=True)
            var = ((xf - mean) ** 2).mean(-1, keepdim=True)
            return ((xf - mean) * torch.rsqrt(var + LN_EPS)).to(x.dtype)

        def baseline_fn(x):
            return F.layer_norm(x, (x.shape[-1],), eps=LN_EPS)

        arity = 1

    elif op == "red_running_stats":
        # Batchnorm-style feature statistics: mean/var over dim=0 (the batch axis).
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            xf = x.float()
            mean = xf.mean(0)
            var = ((xf - mean) ** 2).mean(0)
            return (mean.to(x.dtype), var.to(x.dtype))

        def baseline_fn(x):
            return (x.mean(0), x.var(0, unbiased=False))

        arity = 1

    # ===================================================================== #
    # CROSS-ENTROPY / LOSSES over a large vocab (+ backward)
    # ===================================================================== #
    elif op == "red_cross_entropy":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _targets(M, V, device, seed + 1))

        def ref_fn(logits, targets):
            lf = logits.float()
            return (_lse(lf) - _gather(lf, targets)).to(logits.dtype)

        def baseline_fn(logits, targets):
            return F.cross_entropy(logits, targets.long(), reduction="none")

        arity = 2

    elif op == "red_cross_entropy_bwd":
        # dlogits of sum-reduced CE = softmax(logits) - onehot(target)  (per row).
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _targets(M, V, device, seed + 1))

        def ref_fn(logits, targets):
            grad = _sm(logits.float())
            M = logits.shape[0]
            grad[torch.arange(M, device=logits.device), targets.long()] -= 1.0
            return grad.to(logits.dtype)

        def baseline_fn(logits, targets):
            grad = torch.softmax(logits, dim=-1).clone()
            M = logits.shape[0]
            grad[torch.arange(M, device=logits.device), targets.long()] -= 1.0
            return grad

        arity = 2

    elif op == "red_label_smoothing_ce":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _targets(M, V, device, seed + 1))

        def ref_fn(logits, targets):
            lf = logits.float()
            lse = _lse(lf)
            nll = lse - _gather(lf, targets)
            smooth = lse - lf.mean(-1)
            return ((1.0 - LS_EPS) * nll + LS_EPS * smooth).to(logits.dtype)

        def baseline_fn(logits, targets):
            return F.cross_entropy(logits, targets.long(), reduction="none",
                                   label_smoothing=LS_EPS)

        arity = 2

    elif op == "red_z_loss":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["V"]), device, seed, scale=2.0),)

        def ref_fn(logits):
            lse = _lse(logits.float())
            return (ZLOSS_COEF * lse * lse).to(logits.dtype)

        def baseline_fn(logits):
            lse = torch.logsumexp(logits, dim=-1)
            return ZLOSS_COEF * lse * lse

        arity = 1

    elif op == "red_cross_entropy_zloss":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _targets(M, V, device, seed + 1))

        def ref_fn(logits, targets):
            lf = logits.float()
            lse = _lse(lf)
            return ((lse - _gather(lf, targets)) + ZLOSS_COEF * lse * lse).to(logits.dtype)

        def baseline_fn(logits, targets):
            lse = torch.logsumexp(logits, dim=-1)
            return F.cross_entropy(logits, targets.long(), reduction="none") + ZLOSS_COEF * lse * lse

        arity = 2

    elif op == "red_focal_loss":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _targets(M, V, device, seed + 1))

        def ref_fn(logits, targets):
            lf = logits.float()
            logpt = _gather(lf, targets) - _lse(lf)
            pt = torch.exp(logpt)
            return (-((1.0 - pt) ** FOCAL_GAMMA) * logpt).to(logits.dtype)

        def baseline_fn(logits, targets):
            logpt = -F.cross_entropy(logits, targets.long(), reduction="none")
            pt = torch.exp(logpt)
            return -((1.0 - pt) ** FOCAL_GAMMA) * logpt

        arity = 2

    elif op == "red_soft_cross_entropy":
        # Soft-label cross-entropy: -sum_j q_j log_softmax(logits)_j, q a distribution.
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            logits = _randn((M, V), device, seed, scale=2.0)
            q = _sm(_randn((M, V), device, seed + 1, scale=2.0).float()).to(tdt)
            return (logits, q)

        def ref_fn(logits, q):
            return (-(q.float() * _lsm(logits.float())).sum(-1)).to(logits.dtype)

        def baseline_fn(logits, q):
            return -(q * F.log_softmax(logits, dim=-1)).sum(-1)

        arity = 2

    elif op == "red_bce_with_logits":
        # Stable binary cross-entropy with logits (mean over the last dim):
        # loss = max(x,0) - x*z + log1p(exp(-|x|)).
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], _rn(shape)
            return (_randn((M, N), device, seed, scale=2.0), _rand01((M, N), device, seed + 1))

        def ref_fn(logits, targets):
            x, z = logits.float(), targets.float()
            elem = x.clamp_min(0.0) - x * z + torch.log1p(torch.exp(-x.abs()))
            return elem.mean(-1).to(logits.dtype)

        def baseline_fn(logits, targets):
            return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(-1)

        arity = 2

    # ===================================================================== #
    # DIVERGENCES over softmax distributions
    # ===================================================================== #
    elif op == "red_kl_div":
        # KL(p || q) with p = softmax(logits_p), q = softmax(logits_q). Per row.
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0),
                    _randn((M, V), device, seed + 1, scale=2.0))

        def ref_fn(lp, lq):
            logp = _lsm(lp.float())
            logq = _lsm(lq.float())
            p = torch.exp(logp)
            return (p * (logp - logq)).sum(-1).to(lp.dtype)

        def baseline_fn(lp, lq):
            return F.kl_div(F.log_softmax(lq, dim=-1), F.softmax(lp, dim=-1),
                            reduction="none").sum(-1)

        arity = 2

    elif op == "red_js_div":
        # Jensen-Shannon divergence: 0.5 KL(p||m) + 0.5 KL(q||m), m = (p+q)/2.
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0),
                    _randn((M, V), device, seed + 1, scale=2.0))

        def ref_fn(lp, lq):
            # log-space (stable): log p / log q via log_softmax stay FINITE even for
            # huge logits; the p==0/q==0 lanes are guarded so log(0) never poisons the
            # sum (0*log0 -> 0). log m is finite wherever the guarded term is kept.
            logp = _lsm(lp.float())
            logq = _lsm(lq.float())
            p = torch.exp(logp)
            q = torch.exp(logq)
            logm = torch.log(0.5 * (p + q))
            z = torch.zeros_like(p)
            kl_pm = torch.where(p > 0, p * (logp - logm), z).sum(-1)
            kl_qm = torch.where(q > 0, q * (logq - logm), z).sum(-1)
            return (0.5 * kl_pm + 0.5 * kl_qm).to(lp.dtype)

        def baseline_fn(lp, lq):
            logp = F.log_softmax(lp, dim=-1)
            logq = F.log_softmax(lq, dim=-1)
            p = torch.exp(logp)
            q = torch.exp(logq)
            logm = torch.log(0.5 * (p + q))
            z = torch.zeros_like(p)
            kl_pm = torch.where(p > 0, p * (logp - logm), z).sum(-1)
            kl_qm = torch.where(q > 0, q * (logq - logm), z).sum(-1)
            return 0.5 * kl_pm + 0.5 * kl_qm

        arity = 2

    # ===================================================================== #
    # TOP-K / TOP-P / ARG-REDUCE / CUMULATIVE MAX-MIN
    # ===================================================================== #
    elif op in TOPK_SIZES:
        K = TOPK_SIZES[op]

        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            return torch.sort(x.float(), dim=-1, descending=True).values[..., :K].to(x.dtype)

        def baseline_fn(x):
            return torch.topk(x, K, dim=-1).values

        arity = 1

    elif op == "red_topp_renorm":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(logits):
            probs = _sm(logits.float())
            return _nucleus(probs, TOPP_P).to(logits.dtype)

        def baseline_fn(logits):
            probs = torch.softmax(logits.float(), dim=-1)
            return _nucleus(probs, TOPP_P).to(logits.dtype)

        arity = 1

    elif op in ("red_argmax", "red_argmin"):
        _largest = op == "red_argmax"

        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            return _argfirst(x.float(), _largest)          # int64 first-occurrence index

        def baseline_fn(x):
            return x.argmax(-1) if _largest else x.argmin(-1)

        arity = 1

    elif op in ("red_cummax", "red_cummin"):
        _ismax = op == "red_cummax"

        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            xf = x.float()
            out = torch.cummax(xf, dim=-1).values if _ismax else torch.cummin(xf, dim=-1).values
            return out.to(x.dtype)

        def baseline_fn(x):
            return (torch.cummax(x, dim=-1).values if _ismax
                    else torch.cummin(x, dim=-1).values)

        arity = 1

    # ===================================================================== #
    # ROW-WISE Lp NORMS / NORMALIZE / PAIRWISE REDUCTIONS
    # ===================================================================== #
    elif op == "red_norm_l1":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            return x.float().abs().sum(-1).to(x.dtype)

        def baseline_fn(x):
            return torch.linalg.vector_norm(x, ord=1, dim=-1)

        arity = 1

    elif op == "red_norm_l2":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            xf = x.float()
            return torch.sqrt((xf * xf).sum(-1)).to(x.dtype)

        def baseline_fn(x):
            return torch.linalg.vector_norm(x, ord=2, dim=-1)

        arity = 1

    elif op == "red_norm_linf":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            return x.float().abs().amax(-1).to(x.dtype)

        def baseline_fn(x):
            return torch.linalg.vector_norm(x, ord=float("inf"), dim=-1)

        arity = 1

    elif op == "red_l2_normalize":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], _rn(shape)), device, seed, scale=2.0),)

        def ref_fn(x):
            xf = x.float()
            n = torch.sqrt((xf * xf).sum(-1, keepdim=True))
            return (xf / n.clamp_min(NORM_EPS)).to(x.dtype)

        def baseline_fn(x):
            return F.normalize(x, p=2.0, dim=-1, eps=NORM_EPS)

        arity = 1

    elif op == "red_pairwise_dist":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], _rn(shape)
            return (_randn((M, N), device, seed), _randn((M, N), device, seed + 1))

        def ref_fn(a, b):
            d = a.float() - b.float()
            return torch.sqrt((d * d).sum(-1)).to(a.dtype)

        def baseline_fn(a, b):
            return torch.linalg.vector_norm(a - b, ord=2, dim=-1)

        arity = 2

    elif op == "red_cosine_sim":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], _rn(shape)
            return (_randn((M, N), device, seed), _randn((M, N), device, seed + 1))

        def ref_fn(a, b):
            af, bf = a.float(), b.float()
            dot = (af * bf).sum(-1)
            na = torch.sqrt((af * af).sum(-1))
            nb = torch.sqrt((bf * bf).sum(-1))
            return (dot / (na * nb).clamp_min(COS_EPS)).to(a.dtype)

        def baseline_fn(a, b):
            return F.cosine_similarity(a, b, dim=-1, eps=COS_EPS)

        arity = 2

    else:
        raise ValueError(f"unknown breadth op {op!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op,
          "dtype_name": dtype, "family": f"breadth_{op}", "mutates_input": False}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Naive (correct, COMPILING) Triton seeds - the policy's starting point.
# Every seed streams the reduction in fp32 blocks (numerically stable: online
# max-subtraction for the softmax/lse family, centered/Welford variance) - a
# correct-but-slow start the policy learns to fuse / parallelize.
# {dtype} lands only in the docstring; {tldt} is the tl store dtype literal.
# --------------------------------------------------------------------------- #
_SOFTMAX_ROW_TMPL = '''"""GENERATED breadth {op} seed ({dtype}). x[M,N] -> per-row softmax family
over the last dim. Numerically-stable TWO-pass row kernel: pass 1 an online
(flash-style) running max + rescaled exp-sum in fp32 (no overflow for large
logits); pass 2 reloads x and writes the normalized output. INV_T folds in the
temperature. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, o_ptr, sx, so, N, INV_T, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    m = -float("inf")
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32) * INV_T
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    logs = tl.log(s)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32) * INV_T
        z = x - m
        out = {store_expr}
        tl.store(o_ptr + row * so + offs, out.to({tldt}), mask=mask)


def {op}(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)
    _{op}_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, {inv_t},
                       BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


_GUMBEL_TMPL = '''"""GENERATED breadth red_gumbel_softmax seed ({dtype}). logits[M,N] + gumbel[M,N]
-> softmax((logits + gumbel)/tau) over the last dim, tau={tau}. Stable streaming
max + rescaled exp-sum (fp32), then a normalized write. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_gumbel_softmax_kernel(x_ptr, g_ptr, o_ptr, sx, sg, so, N, INV_T, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    m = -float("inf")
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + row * sg + offs, mask=mask, other=0.0).to(tl.float32)
        z = tl.where(mask, (x + g) * INV_T, -float("inf"))
        blk = tl.max(z, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(z - new_m), 0.0), axis=0)
        m = new_m
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + row * sg + offs, mask=mask, other=0.0).to(tl.float32)
        z = (x + g) * INV_T - m
        tl.store(o_ptr + row * so + offs, (tl.exp(z) / s).to({tldt}), mask=mask)


def red_gumbel_softmax(logits: torch.Tensor, gumbel: torch.Tensor) -> torch.Tensor:
    M, N = logits.shape
    o = torch.empty_like(logits)
    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)
    _red_gumbel_softmax_kernel[(M,)](logits, gumbel, o, logits.stride(0), gumbel.stride(0),
                                     o.stride(0), N, {inv_t}, BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


_SOFTMAX_DIM0_TMPL = '''"""GENERATED breadth red_softmax_dim0 seed ({dtype}). x[M,N] -> softmax over dim 0
(the ROW axis). One program per column-block; a streaming running max + rescaled
exp-sum over the M rows (fp32, stable), then a second pass writes the normalized
column. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_softmax_dim0_kernel(x_ptr, o_ptr, sr, sc, M, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = cols < N
    m = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    s = tl.zeros([BLOCK_N], dtype=tl.float32)
    for r in range(0, M):
        x = tl.load(x_ptr + r * sr + cols * sc, mask=cmask, other=-float("inf")).to(tl.float32)
        new_m = tl.maximum(m, x)
        s = s * tl.exp(m - new_m) + tl.exp(x - new_m)
        m = new_m
    for r in range(0, M):
        x = tl.load(x_ptr + r * sr + cols * sc, mask=cmask, other=0.0).to(tl.float32)
        tl.store(o_ptr + r * sr + cols * sc, (tl.exp(x - m) / s).to({tldt}), mask=cmask)


def red_softmax_dim0(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    BLOCK_N = 256
    grid = (triton.cdiv(N, BLOCK_N),)
    _red_softmax_dim0_kernel[grid](x, o, x.stride(0), x.stride(1), M, N,
                                   BLOCK_N=BLOCK_N, num_warps=4)
    return o
'''


_LSE_ROW_TMPL = '''"""GENERATED breadth {op} seed ({dtype}). x[M,N] -> a per-row log-sum-exp
reduction. Streaming max-subtracted log-sum-exp in fp32 (numerically stable for
large inputs): one online pass tracks the running max m and rescaled sum s, then
lse = m + log(s). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    m = -float("inf")
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    lse = m + tl.log(s)
    v = {post}
    tl.store(o_ptr + row, v.to({tldt}))


def {op}(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _{op}_kernel[(M,)](x, o, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


_LSE_DIM0_TMPL = '''"""GENERATED breadth red_logsumexp_dim0 seed ({dtype}). x[M,N] -> log-sum-exp over
dim 0 -> [N]. One program per column-block; streaming max-subtracted lse over the
M rows (fp32, stable). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_logsumexp_dim0_kernel(x_ptr, o_ptr, sr, sc, M, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = cols < N
    m = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    s = tl.zeros([BLOCK_N], dtype=tl.float32)
    for r in range(0, M):
        x = tl.load(x_ptr + r * sr + cols * sc, mask=cmask, other=-float("inf")).to(tl.float32)
        new_m = tl.maximum(m, x)
        s = s * tl.exp(m - new_m) + tl.exp(x - new_m)
        m = new_m
    out = m + tl.log(s)
    tl.store(o_ptr + cols, out.to({tldt}), mask=cmask)


def red_logsumexp_dim0(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((N,), device=x.device, dtype=x.dtype)
    BLOCK_N = 256
    grid = (triton.cdiv(N, BLOCK_N),)
    _red_logsumexp_dim0_kernel[grid](x, o, x.stride(0), x.stride(1), M, N,
                                     BLOCK_N=BLOCK_N, num_warps=4)
    return o
'''


_ENTROPY_TMPL = '''"""GENERATED breadth red_entropy seed ({dtype}). logits[M,N] -> Shannon entropy of
softmax(logits) per row: H = lse - sum_j p_j x_j (fp32, max-subtracted, stable).
Pass 1 the streaming lse; pass 2 the probability-weighted logit sum. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_entropy_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    m = -float("inf")
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    lse = m + tl.log(s)
    dot = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.exp(x - m) / s
        dot += tl.sum(tl.where(mask, p * x, 0.0), axis=0)
    tl.store(o_ptr + row, (lse - dot).to({tldt}))


def red_entropy(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _red_entropy_kernel[(M,)](x, o, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


_LOGCUMSUMEXP_TMPL = '''"""GENERATED breadth red_logcumsumexp seed ({dtype}). x[M,N] -> cumulative
log-sum-exp over the last dim. One program per row; a sequential fp32 running
(max, rescaled-sum) scan (numerically stable, naive but correct; the policy
replaces the serial loop with a parallel prefix scan). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_logcumsumexp_kernel(x_ptr, o_ptr, sx, so, N):
    row = tl.program_id(0)
    run_m = -float("inf")
    run_s = 0.0
    for i in range(0, N):
        v = tl.load(x_ptr + row * sx + i).to(tl.float32)
        new_m = tl.maximum(run_m, v)
        run_s = run_s * tl.exp(run_m - new_m) + tl.exp(v - new_m)
        run_m = new_m
        tl.store(o_ptr + row * so + i, (run_m + tl.log(run_s)).to({tldt}))


def red_logcumsumexp(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    _red_logcumsumexp_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, num_warps=1)
    return o
'''


_VAR_TMPL = '''"""GENERATED breadth {op} seed ({dtype}). x[M,N] -> a per-row variance/std.
Numerically-stable TWO-pass (Welford-equivalent) reduction: pass 1 the fp32 mean,
pass 2 the fp32 sum of CENTERED squares (avoids the catastrophic cancellation of
E[x^2]-E[x]^2 for large means). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = s / N
    ss = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        d = x - mean
        ss += tl.sum(tl.where(mask, d * d, 0.0), axis=0)
    v = {post}
    tl.store(o_ptr + row, v.to({tldt}))


def {op}(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _{op}_kernel[(M,)](x, o, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


_WELFORD_TMPL = '''"""GENERATED breadth red_welford seed ({dtype}). x[M,N] -> (mean, biased var) per
row via a single-pass chunked Welford merge in fp32 (numerically stable running
mean + M2). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_welford_kernel(x_ptr, mean_ptr, var_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    count = 0.0
    mean = 0.0
    m2 = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        cnt = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
        bsum = tl.sum(tl.where(mask, x, 0.0), axis=0)
        bmean = bsum / cnt
        bm2 = tl.sum(tl.where(mask, (x - bmean) * (x - bmean), 0.0), axis=0)
        new_count = count + cnt
        delta = bmean - mean
        mean = mean + delta * cnt / new_count
        m2 = m2 + bm2 + delta * delta * count * cnt / new_count
        count = new_count
    tl.store(mean_ptr + row, mean.to({tldt}))
    tl.store(var_ptr + row, (m2 / count).to({tldt}))


def red_welford(x: torch.Tensor):
    M, N = x.shape
    mean = torch.empty((M,), device=x.device, dtype=x.dtype)
    var = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _red_welford_kernel[(M,)](x, mean, var, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return mean, var
'''


_RUNNING_STATS_TMPL = '''"""GENERATED breadth red_running_stats seed ({dtype}). x[M,N] -> (mean, biased var)
over dim 0 (batch) -> [N], [N] (batchnorm-style feature statistics). One program
per column-block; TWO fp32 passes over the rows (stable centered variance). {tldt}."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_running_stats_kernel(x_ptr, mean_ptr, var_ptr, sr, sc, M, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = cols < N
    s = tl.zeros([BLOCK_N], dtype=tl.float32)
    for r in range(0, M):
        x = tl.load(x_ptr + r * sr + cols * sc, mask=cmask, other=0.0).to(tl.float32)
        s += x
    mean = s / M
    ss = tl.zeros([BLOCK_N], dtype=tl.float32)
    for r in range(0, M):
        x = tl.load(x_ptr + r * sr + cols * sc, mask=cmask, other=0.0).to(tl.float32)
        d = x - mean
        ss += d * d
    tl.store(mean_ptr + cols, mean.to({tldt}), mask=cmask)
    tl.store(var_ptr + cols, (ss / M).to({tldt}), mask=cmask)


def red_running_stats(x: torch.Tensor):
    M, N = x.shape
    mean = torch.empty((N,), device=x.device, dtype=x.dtype)
    var = torch.empty((N,), device=x.device, dtype=x.dtype)
    BLOCK_N = 256
    grid = (triton.cdiv(N, BLOCK_N),)
    _red_running_stats_kernel[grid](x, mean, var, x.stride(0), x.stride(1), M, N,
                                    BLOCK_N=BLOCK_N, num_warps=4)
    return mean, var
'''


_SUMRED_TMPL = '''"""GENERATED breadth {op} seed ({dtype}). x[M,N] -> a per-row additive reduction
(fp32 accumulate) with a scalar post-op. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(mask, {elem}, 0.0), axis=0)
    v = {post}
    tl.store(o_ptr + row, v.to({tldt}))


def {op}(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _{op}_kernel[(M,)](x, o, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


_MAXRED_TMPL = '''"""GENERATED breadth red_norm_linf seed ({dtype}). x[M,N] -> max_j |x_j| per row
(the L-infinity norm). Streaming fp32 running max of |x|. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_norm_linf_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.max(tl.where(mask, tl.abs(x), 0.0), axis=0))
    tl.store(o_ptr + row, acc.to({tldt}))


def red_norm_linf(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _red_norm_linf_kernel[(M,)](x, o, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


_NORMALIZE_TMPL = '''"""GENERATED breadth {op} seed ({dtype}). x[M,N] -> a per-row rescaled output.
Two fp32 passes: pass 1 sums squares, pass 2 rescales x by the (rms/l2) factor. {tldt}."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, o_ptr, sx, so, N, EPS, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    scale = {scale_expr}
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(o_ptr + row * so + offs, (x * scale).to({tldt}), mask=mask)


def {op}(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    BLOCK_N = 1024
    _{op}_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, {eps}, BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


_LAYERNORM_TMPL = '''"""GENERATED breadth red_layernorm seed ({dtype}). x[M,N] -> (x-mean)/sqrt(var+eps)
over the last dim (no affine), eps={eps}. Three fp32 passes (mean, centered var,
write); the centered variance is numerically stable. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_layernorm_kernel(x_ptr, o_ptr, sx, so, N, EPS, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = s / N
    ss = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        d = x - mean
        ss += tl.sum(tl.where(mask, d * d, 0.0), axis=0)
    rstd = 1.0 / tl.sqrt(ss / N + EPS)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(o_ptr + row * so + offs, ((x - mean) * rstd).to({tldt}), mask=mask)


def red_layernorm(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    BLOCK_N = 1024
    _red_layernorm_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, {eps},
                                BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


_SOFTMAX_BWD_TMPL = '''"""GENERATED breadth red_softmax_bwd seed ({dtype}). Given the saved forward
y=softmax(x) [M,N] and upstream dy [M,N] -> dx = y*(dy - sum_j y_j dy_j) per row.
Two fp32 passes (the row dot y.dy, then the elementwise combine). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_softmax_bwd_kernel(y_ptr, dy_ptr, dx_ptr, sy, sd, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    dot = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        y = tl.load(y_ptr + row * sy + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row * sd + offs, mask=mask, other=0.0).to(tl.float32)
        dot += tl.sum(tl.where(mask, y * dy, 0.0), axis=0)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        y = tl.load(y_ptr + row * sy + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row * sd + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(dx_ptr + row * so + offs, (y * (dy - dot)).to({tldt}), mask=mask)


def red_softmax_bwd(y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    M, N = y.shape
    dx = torch.empty_like(y)
    BLOCK_N = 1024
    _red_softmax_bwd_kernel[(M,)](y, dy, dx, y.stride(0), dy.stride(0), dx.stride(0), N,
                                  BLOCK_N=BLOCK_N, num_warps=8)
    return dx
'''


_LOG_SOFTMAX_BWD_TMPL = '''"""GENERATED breadth red_log_softmax_bwd seed ({dtype}). Given the saved forward
y=log_softmax(x) [M,N] and upstream dy [M,N] -> dx = dy - exp(y)*sum_j dy_j per row.
Two fp32 passes (the row sum of dy, then the elementwise combine). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_log_softmax_bwd_kernel(y_ptr, dy_ptr, dx_ptr, sy, sd, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    sdy = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        dy = tl.load(dy_ptr + row * sd + offs, mask=mask, other=0.0).to(tl.float32)
        sdy += tl.sum(tl.where(mask, dy, 0.0), axis=0)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        y = tl.load(y_ptr + row * sy + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row * sd + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(dx_ptr + row * so + offs, (dy - tl.exp(y) * sdy).to({tldt}), mask=mask)


def red_log_softmax_bwd(y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    M, N = y.shape
    dx = torch.empty_like(y)
    BLOCK_N = 1024
    _red_log_softmax_bwd_kernel[(M,)](y, dy, dx, y.stride(0), dy.stride(0), dx.stride(0), N,
                                      BLOCK_N=BLOCK_N, num_warps=8)
    return dx
'''


_CE_TMPL = '''"""GENERATED breadth red_cross_entropy seed ({dtype}). logits[M,V] + targets[M] ->
per-row NLL loss = logsumexp(logits) - logits[target] (streaming, max-subtracted,
stable for any vocab width). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_cross_entropy_kernel(x_ptr, t_ptr, o_ptr, sx, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sx
    m = -float("inf")
    s = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    lse = m + tl.log(s)
    tgt = tl.load(t_ptr + row)
    xt = tl.load(x_ptr + base + tgt).to(tl.float32)
    tl.store(o_ptr + row, (lse - xt).to({tldt}))


def red_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    o = torch.empty((M,), device=logits.device, dtype=logits.dtype)
    _red_cross_entropy_kernel[(M,)](logits, targets, o, logits.stride(0), V,
                                    BLOCK=1024, num_warps=8)
    return o
'''


_CE_ZLOSS_TMPL = '''"""GENERATED breadth red_cross_entropy_zloss seed ({dtype}). logits[M,V]+targets[M]
-> per-row (logsumexp - logit[target]) + {coef} * logsumexp^2 (the PaLM log-Z^2
regularizer). Streaming max-subtracted lse (stable). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_cross_entropy_zloss_kernel(x_ptr, t_ptr, o_ptr, sx, V, COEF, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sx
    m = -float("inf")
    s = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    lse = m + tl.log(s)
    tgt = tl.load(t_ptr + row)
    xt = tl.load(x_ptr + base + tgt).to(tl.float32)
    tl.store(o_ptr + row, ((lse - xt) + COEF * lse * lse).to({tldt}))


def red_cross_entropy_zloss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    o = torch.empty((M,), device=logits.device, dtype=logits.dtype)
    _red_cross_entropy_zloss_kernel[(M,)](logits, targets, o, logits.stride(0), V,
                                          {coef}, BLOCK=1024, num_warps=8)
    return o
'''


_LABEL_SMOOTH_TMPL = '''"""GENERATED breadth red_label_smoothing_ce seed ({dtype}). logits[M,V]+targets[M].
Per-row streaming pass tracks logsumexp AND the row sum of logits, so
loss = (1-eps)*(lse - logit[target]) + eps*(lse - mean_v logit_v), eps={eps}.
Matches F.cross_entropy(..., label_smoothing=eps). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_label_smoothing_ce_kernel(x_ptr, t_ptr, o_ptr, sx, V, EPS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sx
    m = -float("inf")
    s = 0.0
    ssum = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
        xs = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        ssum += tl.sum(xs, axis=0)
    lse = m + tl.log(s)
    tgt = tl.load(t_ptr + row)
    xt = tl.load(x_ptr + base + tgt).to(tl.float32)
    nll = lse - xt
    smooth = lse - ssum / V
    tl.store(o_ptr + row, ((1.0 - EPS) * nll + EPS * smooth).to({tldt}))


def red_label_smoothing_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    o = torch.empty((M,), device=logits.device, dtype=logits.dtype)
    _red_label_smoothing_ce_kernel[(M,)](logits, targets, o, logits.stride(0), V,
                                         {eps}, BLOCK=1024, num_warps=8)
    return o
'''


_FOCAL_TMPL = '''"""GENERATED breadth red_focal_loss seed ({dtype}). logits[M,V]+targets[M] -> the
multiclass focal loss -(1-pt)^2 * log(pt), pt = softmax(logits)[target] (gamma=2).
Streaming max-subtracted lse; log(pt) = logit[target] - lse (stable). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_focal_loss_kernel(x_ptr, t_ptr, o_ptr, sx, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sx
    m = -float("inf")
    s = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    lse = m + tl.log(s)
    tgt = tl.load(t_ptr + row)
    xt = tl.load(x_ptr + base + tgt).to(tl.float32)
    logpt = xt - lse
    pt = tl.exp(logpt)
    omp = 1.0 - pt
    tl.store(o_ptr + row, (-(omp * omp) * logpt).to({tldt}))


def red_focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    o = torch.empty((M,), device=logits.device, dtype=logits.dtype)
    _red_focal_loss_kernel[(M,)](logits, targets, o, logits.stride(0), V, BLOCK=1024, num_warps=8)
    return o
'''


_SOFT_CE_TMPL = '''"""GENERATED breadth red_soft_cross_entropy seed ({dtype}). logits[M,V], q[M,V] (a
distribution) -> -sum_j q_j log_softmax(logits)_j = lse*sum_j q_j - sum_j q_j x_j.
Pass 1 the streaming lse; pass 2 the weighted sums. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_soft_cross_entropy_kernel(x_ptr, q_ptr, o_ptr, sx, sq, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    bx = row * sx
    bq = row * sq
    m = -float("inf")
    s = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + bx + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    lse = m + tl.log(s)
    wsum = 0.0
    dot = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + bx + offs, mask=mask, other=0.0).to(tl.float32)
        q = tl.load(q_ptr + bq + offs, mask=mask, other=0.0).to(tl.float32)
        wsum += tl.sum(tl.where(mask, q, 0.0), axis=0)
        dot += tl.sum(tl.where(mask, q * x, 0.0), axis=0)
    tl.store(o_ptr + row, (lse * wsum - dot).to({tldt}))


def red_soft_cross_entropy(logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    o = torch.empty((M,), device=logits.device, dtype=logits.dtype)
    _red_soft_cross_entropy_kernel[(M,)](logits, q, o, logits.stride(0), q.stride(0), V,
                                         BLOCK=1024, num_warps=8)
    return o
'''


_CE_BWD_TMPL = '''"""GENERATED breadth red_cross_entropy_bwd seed ({dtype}). logits[M,V]+targets[M] ->
dlogits = softmax(logits) - onehot(target) [M,V] (the gradient of the sum-reduced
CE). Pass 1 the streaming max-subtracted lse; pass 2 writes exp(logit-lse), minus 1
at the target column. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_cross_entropy_bwd_kernel(x_ptr, t_ptr, o_ptr, sx, so, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    bx = row * sx
    m = -float("inf")
    s = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + bx + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    lse = m + tl.log(s)
    tgt = tl.load(t_ptr + row)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + bx + offs, mask=mask, other=-float("inf")).to(tl.float32)
        grad = tl.exp(x - lse)
        grad = grad - tl.where(offs == tgt, 1.0, 0.0)
        tl.store(o_ptr + row * so + offs, grad.to({tldt}), mask=mask)


def red_cross_entropy_bwd(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    o = torch.empty_like(logits)
    _red_cross_entropy_bwd_kernel[(M,)](logits, targets, o, logits.stride(0), o.stride(0), V,
                                        BLOCK=1024, num_warps=8)
    return o
'''


_BCE_TMPL = '''"""GENERATED breadth red_bce_with_logits seed ({dtype}). logits[M,N], targets[M,N]
-> per-row mean binary-cross-entropy-with-logits, elementwise
max(x,0) - x*z + log1p(exp(-|x|)) (the numerically stable form). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_bce_with_logits_kernel(x_ptr, z_ptr, o_ptr, sx, sz, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + row * sz + offs, mask=mask, other=0.0).to(tl.float32)
        elem = tl.maximum(x, 0.0) - x * z + tl.log(1.0 + tl.exp(-tl.abs(x)))
        acc += tl.sum(tl.where(mask, elem, 0.0), axis=0)
    tl.store(o_ptr + row, (acc / N).to({tldt}))


def red_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, N = logits.shape
    o = torch.empty((M,), device=logits.device, dtype=logits.dtype)
    BLOCK_N = 1024
    _red_bce_with_logits_kernel[(M,)](logits, targets, o, logits.stride(0), targets.stride(0), N,
                                      BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


_KL_TMPL = '''"""GENERATED breadth red_kl_div seed ({dtype}). logits_p[M,V], logits_q[M,V] ->
KL(softmax(p) || softmax(q)) per row. Streaming max-subtracted lse for both p and
q, then sum_j p_j*((x_p - lse_p) - (x_q - lse_q)) (stable). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_kl_div_kernel(p_ptr, q_ptr, o_ptr, sp, sq, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    bp = row * sp
    bq = row * sq
    mp = -float("inf")
    sps = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(p_ptr + bp + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(mp, blk)
        sps = sps * tl.exp(mp - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        mp = new_m
    lse_p = mp + tl.log(sps)
    mq = -float("inf")
    sqs = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(q_ptr + bq + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(mq, blk)
        sqs = sqs * tl.exp(mq - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        mq = new_m
    lse_q = mq + tl.log(sqs)
    acc = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        xp = tl.load(p_ptr + bp + offs, mask=mask, other=0.0).to(tl.float32)
        xq = tl.load(q_ptr + bq + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.exp(xp - lse_p)
        term = p * ((xp - lse_p) - (xq - lse_q))
        acc += tl.sum(tl.where(mask, term, 0.0), axis=0)
    tl.store(o_ptr + row, acc.to({tldt}))


def red_kl_div(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    M, V = logits_p.shape
    o = torch.empty((M,), device=logits_p.device, dtype=logits_p.dtype)
    _red_kl_div_kernel[(M,)](logits_p, logits_q, o, logits_p.stride(0), logits_q.stride(0), V,
                             BLOCK=1024, num_warps=8)
    return o
'''


_JS_TMPL = '''"""GENERATED breadth red_js_div seed ({dtype}). logits_p[M,V], logits_q[M,V] ->
Jensen-Shannon divergence between softmax(p) and softmax(q) per row:
0.5*sum p*log(p/m) + 0.5*sum q*log(q/m), m=(p+q)/2. Stable max-subtracted lse for
p and q, then the combine pass. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_js_div_kernel(p_ptr, q_ptr, o_ptr, sp, sq, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    bp = row * sp
    bq = row * sq
    mp = -float("inf")
    sps = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(p_ptr + bp + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(mp, blk)
        sps = sps * tl.exp(mp - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        mp = new_m
    lse_p = mp + tl.log(sps)
    mq = -float("inf")
    sqs = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(q_ptr + bq + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(mq, blk)
        sqs = sqs * tl.exp(mq - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        mq = new_m
    lse_q = mq + tl.log(sqs)
    acc = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        xp = tl.load(p_ptr + bp + offs, mask=mask, other=0.0).to(tl.float32)
        xq = tl.load(q_ptr + bq + offs, mask=mask, other=0.0).to(tl.float32)
        lp_i = xp - lse_p
        lq_i = xq - lse_q
        p = tl.exp(lp_i)
        q = tl.exp(lq_i)
        logmm = tl.log(0.5 * (p + q))
        tp = tl.where(p > 0.0, p * (lp_i - logmm), 0.0)
        tq = tl.where(q > 0.0, q * (lq_i - logmm), 0.0)
        term = 0.5 * (tp + tq)
        acc += tl.sum(tl.where(mask, term, 0.0), axis=0)
    tl.store(o_ptr + row, acc.to({tldt}))


def red_js_div(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    M, V = logits_p.shape
    o = torch.empty((M,), device=logits_p.device, dtype=logits_p.dtype)
    _red_js_div_kernel[(M,)](logits_p, logits_q, o, logits_p.stride(0), logits_q.stride(0), V,
                             BLOCK=1024, num_warps=8)
    return o
'''


_TOPK_TMPL = '''"""GENERATED breadth {op} seed ({dtype}). x[M,N] -> the top-{k} values per row,
descending. Naive but correct STREAMING threshold selection: {k} passes, each
pulling the running max strictly below the previous winner (O(k*N), the policy
replaces it with a real partial/bitonic top-k). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, o_ptr, sx, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    prev = float("inf")
    for i in range(0, {k}):
        cur = -float("inf")
        for start in range(0, N, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            mask = offs < N
            x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
            cand = tl.where(x < prev, x, -float("inf"))
            cur = tl.maximum(cur, tl.max(cand, axis=0))
        tl.store(o_ptr + row * so + i, cur.to({tldt}))
        prev = cur


def {op}(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M, {k}), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _{op}_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


_TOPP_TMPL = '''"""GENERATED breadth red_topp_renorm seed ({dtype}). logits[M,N] -> top-p (nucleus)
renormalized probabilities, p={p}. Partial-fusion starting point: a Triton kernel
computes the numerically-stable softmax; the data-dependent nucleus selection (sort
+ cumulative-mass threshold) and renormalization run host-side in torch. Fusing the
selection into the kernel (a streaming threshold search) is the optimization target.
{tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_topp_softmax_kernel(x_ptr, o_ptr, sx, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    m = -float("inf")
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(o_ptr + row * so + offs, (tl.exp(x - m) / s).to({tldt}), mask=mask)


def red_topp_renorm(logits: torch.Tensor, p: float = {p}) -> torch.Tensor:
    M, N = logits.shape
    probs = torch.empty_like(logits)
    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)
    _red_topp_softmax_kernel[(M,)](logits, probs, logits.stride(0), probs.stride(0), N,
                                   BLOCK_N=BLOCK_N, num_warps=8)
    pf = probs.float()
    sp, si = torch.sort(pf, dim=-1, descending=True)
    excl = sp.cumsum(-1) - sp
    keep_sorted = excl <= p
    keep = torch.zeros_like(pf, dtype=torch.bool).scatter_(-1, si, keep_sorted)
    masked = torch.where(keep, pf, torch.zeros_like(pf))
    return (masked / masked.sum(-1, keepdim=True)).to(logits.dtype)
'''


_ARG_TMPL = '''"""GENERATED breadth {op} seed ({dtype}). x[M,N] -> the {which} INDEX (int64),
first-occurrence on ties. Streaming fp32 running {which} + its index across blocks
(strict comparison keeps the earliest winner). int64 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    best = {init}
    best_idx = 0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other={other}).to(tl.float32)
        blk = {blk_reduce}
        blk_idx = start + {blk_arg}
        take = {cmp}
        best_idx = tl.where(take, blk_idx, best_idx)
        best = tl.where(take, blk, best)
    tl.store(o_ptr + row, best_idx.to(tl.int64))


def {op}(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=torch.int64)
    BLOCK_N = 1024
    _{op}_kernel[(M,)](x, o, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


_CUMSCAN_TMPL = '''"""GENERATED breadth {op} seed ({dtype}). x[M,N] -> cumulative {which} over the
last dim. One program per row; a sequential fp32 running-{which} scan (naive but
correct; the policy replaces the serial loop with a parallel prefix scan). {tldt}."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, o_ptr, sx, so, N):
    row = tl.program_id(0)
    run = {init}
    for i in range(0, N):
        v = tl.load(x_ptr + row * sx + i).to(tl.float32)
        run = {comb}
        tl.store(o_ptr + row * so + i, run.to({tldt}))


def {op}(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    _{op}_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, num_warps=1)
    return o
'''


_PAIRDIST_TMPL = '''"""GENERATED breadth red_pairwise_dist seed ({dtype}). a[M,N], b[M,N] -> the per-row
Euclidean distance ||a-b||_2. Single streaming fp32 sum of squared differences,
then sqrt. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_pairwise_dist_kernel(a_ptr, b_ptr, o_ptr, sa, sb, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        a = tl.load(a_ptr + row * sa + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + row * sb + offs, mask=mask, other=0.0).to(tl.float32)
        d = a - b
        acc += tl.sum(tl.where(mask, d * d, 0.0), axis=0)
    tl.store(o_ptr + row, tl.sqrt(acc).to({tldt}))


def red_pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, N = a.shape
    o = torch.empty((M,), device=a.device, dtype=a.dtype)
    BLOCK_N = 1024
    _red_pairwise_dist_kernel[(M,)](a, b, o, a.stride(0), b.stride(0), N,
                                    BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


_COSINE_TMPL = '''"""GENERATED breadth red_cosine_sim seed ({dtype}). a[M,N], b[M,N] -> per-row cosine
similarity <a,b>/(||a|| ||b||). Single streaming fp32 pass accumulates the dot and
both squared norms; denominator floored at {eps}. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_cosine_sim_kernel(a_ptr, b_ptr, o_ptr, sa, sb, N, EPS, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    dot = 0.0
    na = 0.0
    nb = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        a = tl.load(a_ptr + row * sa + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + row * sb + offs, mask=mask, other=0.0).to(tl.float32)
        dot += tl.sum(tl.where(mask, a * b, 0.0), axis=0)
        na += tl.sum(tl.where(mask, a * a, 0.0), axis=0)
        nb += tl.sum(tl.where(mask, b * b, 0.0), axis=0)
    denom = tl.maximum(tl.sqrt(na) * tl.sqrt(nb), EPS)
    tl.store(o_ptr + row, (dot / denom).to({tldt}))


def red_cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, N = a.shape
    o = torch.empty((M,), device=a.device, dtype=a.dtype)
    BLOCK_N = 1024
    _red_cosine_sim_kernel[(M,)](a, b, o, a.stride(0), b.stride(0), N, {eps},
                                 BLOCK_N=BLOCK_N, num_warps=8)
    return o
'''


def seed_source(op: str, dtype: str) -> str:
    tldt = DTYPES[dtype][1]

    # softmax family (last dim) ------------------------------------------- #
    if op in ("red_softmax", "red_online_softmax"):
        return _SOFTMAX_ROW_TMPL.format(op=op, dtype=dtype, tldt=tldt,
                                        store_expr="tl.exp(z) / s", inv_t="1.0")
    if op == "red_log_softmax":
        return _SOFTMAX_ROW_TMPL.format(op=op, dtype=dtype, tldt=tldt,
                                        store_expr="z - logs", inv_t="1.0")
    if op == "red_softmax_temp":
        return _SOFTMAX_ROW_TMPL.format(op=op, dtype=dtype, tldt=tldt,
                                        store_expr="tl.exp(z) / s", inv_t=repr(1.0 / TEMP))
    if op == "red_gumbel_softmax":
        return _GUMBEL_TMPL.format(dtype=dtype, tldt=tldt, tau=GUMBEL_TAU,
                                   inv_t=repr(1.0 / GUMBEL_TAU))
    if op == "red_softmax_dim0":
        return _SOFTMAX_DIM0_TMPL.format(dtype=dtype, tldt=tldt)

    # backward ------------------------------------------------------------- #
    if op == "red_softmax_bwd":
        return _SOFTMAX_BWD_TMPL.format(dtype=dtype, tldt=tldt)
    if op == "red_log_softmax_bwd":
        return _LOG_SOFTMAX_BWD_TMPL.format(dtype=dtype, tldt=tldt)

    # log-sum-exp / entropy / logcumsumexp -------------------------------- #
    if op == "red_logsumexp":
        return _LSE_ROW_TMPL.format(op=op, dtype=dtype, tldt=tldt, post="lse")
    if op == "red_z_loss":
        return _LSE_ROW_TMPL.format(op=op, dtype=dtype, tldt=tldt,
                                    post=f"{ZLOSS_COEF!r} * lse * lse")
    if op == "red_logsumexp_dim0":
        return _LSE_DIM0_TMPL.format(dtype=dtype, tldt=tldt)
    if op == "red_entropy":
        return _ENTROPY_TMPL.format(dtype=dtype, tldt=tldt)
    if op == "red_logcumsumexp":
        return _LOGCUMSUMEXP_TMPL.format(dtype=dtype, tldt=tldt)

    # variance / welford / normalization stats ---------------------------- #
    if op == "red_var":
        return _VAR_TMPL.format(op=op, dtype=dtype, tldt=tldt, post="ss / N")
    if op == "red_var_unbiased":
        return _VAR_TMPL.format(op=op, dtype=dtype, tldt=tldt, post="ss / (N - 1)")
    if op == "red_std":
        return _VAR_TMPL.format(op=op, dtype=dtype, tldt=tldt, post="tl.sqrt(ss / (N - 1))")
    if op == "red_welford":
        return _WELFORD_TMPL.format(dtype=dtype, tldt=tldt)
    if op == "red_running_stats":
        return _RUNNING_STATS_TMPL.format(dtype=dtype, tldt=tldt)
    if op == "red_rms":
        return _SUMRED_TMPL.format(op=op, dtype=dtype, tldt=tldt,
                                   elem="x * x", post="tl.sqrt(acc / N)")
    if op == "red_rmsnorm":
        return _NORMALIZE_TMPL.format(op=op, dtype=dtype, tldt=tldt,
                                      scale_expr="1.0 / tl.sqrt(acc / N + EPS)", eps=repr(RMS_EPS))
    if op == "red_layernorm":
        return _LAYERNORM_TMPL.format(dtype=dtype, tldt=tldt, eps=repr(LN_EPS))

    # cross-entropy / losses ---------------------------------------------- #
    if op == "red_cross_entropy":
        return _CE_TMPL.format(dtype=dtype, tldt=tldt)
    if op == "red_cross_entropy_bwd":
        return _CE_BWD_TMPL.format(dtype=dtype, tldt=tldt)
    if op == "red_label_smoothing_ce":
        return _LABEL_SMOOTH_TMPL.format(dtype=dtype, tldt=tldt, eps=repr(LS_EPS))
    if op == "red_cross_entropy_zloss":
        return _CE_ZLOSS_TMPL.format(dtype=dtype, tldt=tldt, coef=repr(ZLOSS_COEF))
    if op == "red_focal_loss":
        return _FOCAL_TMPL.format(dtype=dtype, tldt=tldt)
    if op == "red_soft_cross_entropy":
        return _SOFT_CE_TMPL.format(dtype=dtype, tldt=tldt)
    if op == "red_bce_with_logits":
        return _BCE_TMPL.format(dtype=dtype, tldt=tldt)

    # divergences ---------------------------------------------------------- #
    if op == "red_kl_div":
        return _KL_TMPL.format(dtype=dtype, tldt=tldt)
    if op == "red_js_div":
        return _JS_TMPL.format(dtype=dtype, tldt=tldt)

    # top-k / top-p / arg / cumulative ------------------------------------ #
    if op in TOPK_SIZES:
        return _TOPK_TMPL.format(op=op, dtype=dtype, tldt=tldt, k=TOPK_SIZES[op])
    if op == "red_topp_renorm":
        return _TOPP_TMPL.format(dtype=dtype, tldt=tldt, p=TOPP_P)
    if op == "red_argmax":
        return _ARG_TMPL.format(op=op, dtype=dtype, which="max", init='-float("inf")',
                                other='-float("inf")', blk_reduce="tl.max(x, axis=0)",
                                blk_arg="tl.argmax(x, axis=0)", cmp="blk > best")
    if op == "red_argmin":
        return _ARG_TMPL.format(op=op, dtype=dtype, which="min", init='float("inf")',
                                other='float("inf")', blk_reduce="tl.min(x, axis=0)",
                                blk_arg="tl.argmin(x, axis=0)", cmp="blk < best")
    if op == "red_cummax":
        return _CUMSCAN_TMPL.format(op=op, dtype=dtype, tldt=tldt, which="max",
                                    init='-float("inf")', comb="tl.maximum(run, v)")
    if op == "red_cummin":
        return _CUMSCAN_TMPL.format(op=op, dtype=dtype, tldt=tldt, which="min",
                                    init='float("inf")', comb="tl.minimum(run, v)")

    # Lp norms / normalize / pairwise ------------------------------------- #
    if op == "red_norm_l1":
        return _SUMRED_TMPL.format(op=op, dtype=dtype, tldt=tldt, elem="tl.abs(x)", post="acc")
    if op == "red_norm_l2":
        return _SUMRED_TMPL.format(op=op, dtype=dtype, tldt=tldt, elem="x * x", post="tl.sqrt(acc)")
    if op == "red_norm_linf":
        return _MAXRED_TMPL.format(dtype=dtype, tldt=tldt)
    if op == "red_l2_normalize":
        return _NORMALIZE_TMPL.format(op=op, dtype=dtype, tldt=tldt,
                                      scale_expr="1.0 / tl.maximum(tl.sqrt(acc), EPS)",
                                      eps=repr(NORM_EPS))
    if op == "red_pairwise_dist":
        return _PAIRDIST_TMPL.format(dtype=dtype, tldt=tldt)
    if op == "red_cosine_sim":
        return _COSINE_TMPL.format(dtype=dtype, tldt=tldt, eps=repr(COS_EPS))

    raise ValueError(f"unknown breadth op {op!r}")


def op_names() -> list[str]:
    return list(OPS)
