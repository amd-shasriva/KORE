"""Breadth SAMPLING / LOGIT-PROCESSOR / SPECULATIVE-DECODE / ROPE task-authoring engine.

Widens the KORE suite with the HARD, DATA-DEPENDENT, numerically-careful frontier of
the LLM *inference tail* - the logit processors, the (deterministic) samplers, the
speculative-decoding accept/verify math and the advanced long-context RoPE kernels
that dominate a serving decode step but where a naive implementation is either wrong
(overflow / bad tie rule / invalid renormalized distribution / wrong frequency scaling)
or slow (many HBM passes over a 128k-wide vocab). No trivial argmax: every op is a
sorted/thresholded/renormalized reduction over a large vocab, an inverse-CDF or
rejection-sampling decision, a tree/parent-table construction, or an orthogonal RoPE
rotation with the exact NTK / YaRN / dynamic-NTK / PI / Llama-3 frequency formula.

Op families (every name is prefixed ``smp_``)
---------------------------------------------
  * Logit processors over a large vocab V: temperature-softmax, top-k mask (k=20/50),
    top-p/nucleus renorm, min-p mask, locally-typical mask, repetition penalty,
    presence penalty, frequency penalty, additive logit-bias, no-repeat-ngram block,
    combined top-k+top-p (12).
  * Deterministic sampling (a SUPPLIED uniform/gumbel tensor makes the oracle exactly
    reproducible): categorical inverse-CDF, Gumbel-max, top-p-then-sample,
    top-k-then-sample (4).
  * Speculative decoding: rejection-sampling accept decision, corrected residual
    distribution normalize(relu(p_target - p_draft)), bonus-token sampling from the
    target dist, tree-attention mask from a parent table, verify-and-accept longest
    matching prefix (5).
  * Advanced RoPE / positional (real long-context kernels): linear position
    interpolation, static NTK-aware, dynamic-NTK, YaRN (ramped interp + mscale),
    partial rotary, 2D RoPE, Llama-3 frequency smoothing (7).

Contract (mirrors ``kore.tasks.breadth.reduce_ext`` so the generic ``kore.tasks._genops``
driver + generator consume it unchanged):

    OPS / OP_DTYPES / SHAPES              module-level task catalog
    make_reference(op, dtype) -> dict     reference.py namespace (parse_shape,
        get_inputs, ref_fn EXACT fp32 oracle [may return an index/mask], baseline_fn
        torch path, arity, entry_name, dtype_name, family=f"breadth_{op}", mutates_input)
    seed_source(op, dtype) -> str         a naive, COMPILING, CORRECT Triton seed (the
        policy's start; the data-dependent selection runs host-side in torch).

CORRECTNESS is paramount and EXACT: every ``ref_fn`` computes the numerically stable
formula in fp32 (max-subtracted softmax/lse for every distribution op, the min(1,
p/q) rejection rule, the exact NTK/YaRN frequency scaling) and casts back to the task
dtype. Any RANDOM op is made DETERMINISTIC by taking a supplied uniform / gumbel /
draft-token tensor, so the fp32 oracle is bit-reproducible. tests/test_sample_ext.py
cross-checks every ``ref_fn`` against an INDEPENDENT torch computation (torch.softmax /
searchsorted inverse-CDF / complex-number RoPE rotation / explicit ancestor walk /
per-row loops) at a tight fp32 tolerance, plus an EXTREME-magnitude case.

torch/triton are imported lazily (registry discovery never needs a GPU).
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# --------------------------------------------------------------------------- #
# task catalog (every op is prefixed ``smp_``)
# --------------------------------------------------------------------------- #
OPS: tuple[str, ...] = (
    # logit processors over a large vocab V (12)
    "smp_temperature", "smp_topk_mask_k20", "smp_topk_mask_k50", "smp_topp_renorm",
    "smp_minp_mask", "smp_typical_mask", "smp_repetition_penalty", "smp_presence_penalty",
    "smp_frequency_penalty", "smp_logit_bias", "smp_no_repeat_ngram", "smp_topk_topp",
    # deterministic sampling via supplied uniforms / gumbel noise (4)
    "smp_categorical_sample", "smp_gumbel_max", "smp_topp_sample", "smp_topk_sample",
    # speculative decoding (5)
    "smp_spec_accept", "smp_spec_residual", "smp_spec_bonus_token", "smp_tree_attn_mask",
    "smp_verify_prefix",
    # advanced RoPE / positional (7)
    "smp_rope_linear_pi", "smp_rope_ntk", "smp_rope_dynamic_ntk", "smp_rope_yarn",
    "smp_rope_partial", "smp_rope_2d", "smp_rope_llama3",
)

# Swept over the two serving activation dtypes plus fp32 (the fp32 oracle casts back).
DEFAULT_DTYPES: tuple[str, ...] = ("bf16", "fp16", "fp32")
OP_DTYPES: dict[str, tuple[str, ...]] = {op: DEFAULT_DTYPES for op in OPS}


def op_dtypes(op: str) -> tuple[str, ...]:
    """The dtype sweep for a breadth op (per-op override or the global default)."""
    return OP_DTYPES.get(op, DEFAULT_DTYPES)


# --------------------------------------------------------------------------- #
# task hyper-params (baked into BOTH the fp32 oracle and the seed defaults)
# --------------------------------------------------------------------------- #
TEMP = 0.7                 # sampling temperature (logits divided by TEMP before softmax)
TOPK_MASK_SIZES: dict[str, int] = {"smp_topk_mask_k20": 20, "smp_topk_mask_k50": 50}
TOPK_SAMPLE_K = 50         # k for the combined top-k+top-p and the top-k sampler
TOPP_P = 0.9               # nucleus (top-p) mass threshold
MIN_P = 0.05               # min-p threshold (fraction of the peak probability)
TYPICAL_MASS = 0.9         # locally-typical sampling mass
REP_PENALTY = 1.2          # CTRL repetition penalty (>1 discourages seen tokens)
PRESENCE_PENALTY = 0.5     # OpenAI-style presence penalty (subtract if token present)
FREQ_PENALTY = 0.5         # OpenAI-style frequency penalty (subtract freq * count)
NGRAM_N = 3                # no-repeat n-gram size (block repeats of the (n-1)-gram)

ROPE_THETA = 10000.0       # RoPE base frequency
ROPE_SCALE = 4.0           # long-context scale factor (PI / NTK / YaRN)
ROPE_ORIG_MAX = 4096       # original (pre-scaling) max position
ROPE_DYN_SEQ_LEN = 16384   # current sequence length for dynamic-NTK
YARN_BETA_FAST = 32.0      # YaRN correction range (fast/high-frequency rotations)
YARN_BETA_SLOW = 1.0       # YaRN correction range (slow/low-frequency rotations)
ROT_PCT = 0.5              # partial-rotary fraction of head_dim that is rotated
LLAMA3_FACTOR = 8.0        # Llama-3.1 rope scaling factor
LLAMA3_LOW_FREQ = 1.0      # Llama-3.1 low-frequency factor
LLAMA3_HIGH_FREQ = 4.0     # Llama-3.1 high-frequency factor
LLAMA3_OLD_CTX = 8192.0    # Llama-3.1 original context length

# --------------------------------------------------------------------------- #
# Realistic shapes: decode batch M in {1, 64, 256}, vocab V in {32000, 128256}
# (Llama-2 / Llama-3) plus a non-power-of-2 tail (GPT-2 50257 / 32001). RoPE ops
# use head_dim D in {64, 128} (+ a non-pow2 96 tail, divisible by 4 for 2D). The
# tree / verify / n-gram ops carry their own small axes.
# --------------------------------------------------------------------------- #
_VOCAB = {
    "minimal": {"M": 4, "V": 128},
    "primary": {"M": 64, "V": 32000},
    "validation": [
        {"M": 1, "V": 32000},
        {"M": 256, "V": 128256},
        {"M": 64, "V": 50257},          # GPT-2 vocab (non-pow2 tail)
        {"M": 64, "V": 32001},          # explicit non-pow2 tail
    ],
}
_NGRAM = {
    "minimal": {"M": 4, "V": 128, "L": 8},
    "primary": {"M": 64, "V": 32000, "L": 16},
    "validation": [
        {"M": 256, "V": 128256, "L": 32},
        {"M": 1, "V": 32000, "L": 8},
        {"M": 64, "V": 50257, "L": 17}, # non-pow2
    ],
}
_VERIFY = {
    "minimal": {"M": 4, "K": 4},
    "primary": {"M": 64, "K": 5},
    "validation": [
        {"M": 256, "K": 8},
        {"M": 1, "K": 4},
        {"M": 64, "K": 7},              # non-pow2
    ],
}
_TREE = {
    "minimal": {"M": 2, "T": 8},
    "primary": {"M": 64, "T": 32},
    "validation": [
        {"M": 256, "T": 64},
        {"M": 1, "T": 16},
        {"M": 64, "T": 63},            # non-pow2 tail
    ],
}
_ROPE = {
    "minimal": {"M": 4, "D": 64},
    "primary": {"M": 64, "D": 128},
    "validation": [
        {"M": 256, "D": 128},
        {"M": 1, "D": 64},
        {"M": 64, "D": 96},            # non-pow2 head_dim (divisible by 4 for 2D)
    ],
}

_NGRAM_OPS = frozenset({"smp_no_repeat_ngram"})
_VERIFY_OPS = frozenset({"smp_verify_prefix"})
_TREE_OPS = frozenset({"smp_tree_attn_mask"})
_ROPE_OPS = frozenset({
    "smp_rope_linear_pi", "smp_rope_ntk", "smp_rope_dynamic_ntk", "smp_rope_yarn",
    "smp_rope_partial", "smp_rope_2d", "smp_rope_llama3",
})
# everything else lives over the vocab axis (logits / probs / sampling / spec)
_VOCAB_OPS = frozenset(OPS) - _NGRAM_OPS - _VERIFY_OPS - _TREE_OPS - _ROPE_OPS

SHAPES: dict[str, dict] = {}
for _op in OPS:
    if _op in _NGRAM_OPS:
        SHAPES[_op] = _NGRAM
    elif _op in _VERIFY_OPS:
        SHAPES[_op] = _VERIFY
    elif _op in _TREE_OPS:
        SHAPES[_op] = _TREE
    elif _op in _ROPE_OPS:
        SHAPES[_op] = _ROPE
    else:
        SHAPES[_op] = _VOCAB


# --------------------------------------------------------------------------- #
# reference.py namespace (fp32 EXACT oracle + torch baseline)
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import math

    import torch
    import torch.nn.functional as F

    tdt = getattr(torch, DTYPES[dtype][0])

    # -------- input generators (deterministic; supplied noise -> reproducible) -- #
    def _randn(shape, device, seed, scale=1.0):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape, generator=g, device=device, dtype=torch.float32) * scale).to(tdt)

    def _uniform(shape, device, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.rand(shape, generator=g, device=device, dtype=torch.float32)

    def _probs(shape, device, seed, scale=1.0):
        return _sm(_randn(shape, device, seed, scale).float()).to(tdt)

    def _counts(shape, device, seed, hi=5):
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.randint(0, hi, shape, generator=g, device=device, dtype=torch.int64)

    def _seen(shape, device, seed, prob=0.5):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.rand(shape, generator=g, device=device, dtype=torch.float32) < prob).to(torch.float32)

    def _ids(shape, hi, device, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.randint(0, hi, shape, generator=g, device=device, dtype=torch.int64)

    def _positions(M, device, seed, hi=ROPE_ORIG_MAX):
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.randint(0, hi, (M,), generator=g, device=device, dtype=torch.int64).to(torch.float32)

    def _gumbel(shape, device, seed):
        u = _uniform(shape, device, seed).clamp_(1e-9, 1.0)
        return -torch.log(-torch.log(u))

    def _parents(M, T, device, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        par = torch.full((M, T), -1, dtype=torch.int64, device=device)
        for j in range(1, T):
            par[:, j] = torch.randint(0, j, (M,), generator=g, device=device, dtype=torch.int64)
        return par

    # -------- stable fp32 primitives --------------------------------------- #
    def _sm(t, dim=-1):
        m = t.amax(dim=dim, keepdim=True)
        e = torch.exp(t - m)
        return e / e.sum(dim=dim, keepdim=True)

    def _lsm(t, dim=-1):
        m = t.amax(dim=dim, keepdim=True)
        z = t - m
        return z - torch.log(torch.exp(z).sum(dim=dim, keepdim=True))

    def _nucleus(probs, p):
        sp, si = torch.sort(probs, dim=-1, descending=True)
        excl = sp.cumsum(-1) - sp                      # exclusive prefix mass
        keep_sorted = excl <= p                        # keeps the crossing token
        keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, si, keep_sorted)
        masked = torch.where(keep, probs, torch.zeros_like(probs))
        return masked / masked.sum(-1, keepdim=True)

    def _topk_mask_logits(xf, k):
        kth = torch.topk(xf, k, dim=-1).values[..., -1:]
        neg = torch.full_like(xf, float("-inf"))
        return torch.where(xf >= kth, xf, neg)

    def _inv_cdf(probs, u):
        cdf = probs.cumsum(-1)
        idx = (cdf <= u.float().view(-1, 1)).sum(-1)
        return idx.clamp_(max=probs.shape[-1] - 1).to(torch.int64)

    # -------- RoPE helpers (NeoX rotate-half; orthogonal rotation) --------- #
    def _inv_freq(dim, theta, device):
        half = dim // 2
        i = torch.arange(half, device=device, dtype=torch.float32)
        return theta ** (-(2.0 * i) / dim)

    def _apply_rope(x, inv_freq, pos, mscale=1.0):
        half = x.shape[-1] // 2
        ang = pos[:, None] * inv_freq[None, :]
        c = torch.cos(ang) * mscale
        s = torch.sin(ang) * mscale
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    # ===================================================================== #
    # LOGIT PROCESSORS over a large vocab V
    # ===================================================================== #
    if op == "smp_temperature":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["V"]), device, seed, scale=2.0),)

        def ref_fn(x):
            return _sm(x.float() / TEMP).to(x.dtype)

        def baseline_fn(x):
            return torch.softmax(x / TEMP, dim=-1)

        arity = 1

    elif op in TOPK_MASK_SIZES:
        K = TOPK_MASK_SIZES[op]

        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["V"]), device, seed, scale=2.0),)

        def ref_fn(x):
            return _topk_mask_logits(x.float(), K).to(x.dtype)

        def baseline_fn(x):
            kth = torch.topk(x, K, dim=-1).values[..., -1:]
            return x.masked_fill(x < kth, float("-inf"))

        arity = 1

    elif op == "smp_topp_renorm":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["V"]), device, seed, scale=2.0),)

        def ref_fn(x):
            return _nucleus(_sm(x.float()), TOPP_P).to(x.dtype)

        def baseline_fn(x):
            return _nucleus(torch.softmax(x.float(), dim=-1), TOPP_P).to(x.dtype)

        arity = 1

    elif op == "smp_minp_mask":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["V"]), device, seed, scale=2.0),)

        def ref_fn(x):
            p = _sm(x.float())
            keep = p >= MIN_P * p.amax(-1, keepdim=True)
            masked = torch.where(keep, p, torch.zeros_like(p))
            return (masked / masked.sum(-1, keepdim=True)).to(x.dtype)

        def baseline_fn(x):
            p = torch.softmax(x.float(), dim=-1)
            keep = p >= MIN_P * p.amax(-1, keepdim=True)
            masked = torch.where(keep, p, torch.zeros_like(p))
            return (masked / masked.sum(-1, keepdim=True)).to(x.dtype)

        arity = 1

    elif op == "smp_typical_mask":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["V"]), device, seed, scale=2.0),)

        def ref_fn(x):
            logp = _lsm(x.float())
            p = torch.exp(logp)
            H = -(p * logp).sum(-1, keepdim=True)          # per-row entropy
            dev = ((-logp) - H).abs()                      # surprisal deviation
            sd, si = torch.sort(dev, dim=-1, descending=False)
            p_sorted = p.gather(-1, si)
            excl = p_sorted.cumsum(-1) - p_sorted
            keep_sorted = excl <= TYPICAL_MASS
            keep = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, si, keep_sorted)
            masked = torch.where(keep, p, torch.zeros_like(p))
            return (masked / masked.sum(-1, keepdim=True)).to(x.dtype)

        baseline_fn = None  # set below to ref (torch path is identical)
        arity = 1

    elif op == "smp_repetition_penalty":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _seen((M, V), device, seed + 1))

        def ref_fn(x, seen):
            xf = x.float()
            pen = torch.where(xf > 0, xf / REP_PENALTY, xf * REP_PENALTY)
            return torch.where(seen > 0, pen, xf).to(x.dtype)

        def baseline_fn(x, seen):
            pen = torch.where(x > 0, x / REP_PENALTY, x * REP_PENALTY)
            return torch.where(seen > 0, pen, x)

        arity = 2

    elif op == "smp_presence_penalty":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _counts((M, V), device, seed + 1))

        def ref_fn(x, counts):
            return (x.float() - PRESENCE_PENALTY * (counts > 0).float()).to(x.dtype)

        def baseline_fn(x, counts):
            return x - PRESENCE_PENALTY * (counts > 0).to(x.dtype)

        arity = 2

    elif op == "smp_frequency_penalty":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _counts((M, V), device, seed + 1))

        def ref_fn(x, counts):
            return (x.float() - FREQ_PENALTY * counts.float()).to(x.dtype)

        def baseline_fn(x, counts):
            return x - FREQ_PENALTY * counts.to(x.dtype)

        arity = 2

    elif op == "smp_logit_bias":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _randn((M, V), device, seed + 1, scale=1.0))

        def ref_fn(x, bias):
            return (x.float() + bias.float()).to(x.dtype)

        def baseline_fn(x, bias):
            return x + bias

        arity = 2

    elif op == "smp_no_repeat_ngram":
        n = NGRAM_N

        def get_inputs(shape, device="cuda", seed=0):
            M, V, L = shape["M"], shape["V"], shape["L"]
            return (_randn((M, V), device, seed, scale=2.0), _ids((M, L), V, device, seed + 1))

        def _block(logits, prev_ids, fill):
            out = logits.clone()
            M, V = out.shape
            L = prev_ids.shape[1]
            if L >= n:
                rows = prev_ids.tolist()
                for i in range(M):
                    row = rows[i]
                    suffix = row[L - n + 1:L]
                    for j in range(0, L - n + 1):
                        if row[j:j + n - 1] == suffix:
                            out[i, row[j + n - 1]] = fill
            return out

        def ref_fn(x, prev_ids):
            return _block(x.float(), prev_ids, float("-inf")).to(x.dtype)

        def baseline_fn(x, prev_ids):
            return _block(x, prev_ids, float("-inf"))

        arity = 2

    elif op == "smp_topk_topp":
        K = TOPK_SAMPLE_K

        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["V"]), device, seed, scale=2.0),)

        def ref_fn(x):
            p = _sm(_topk_mask_logits(x.float(), K))
            return _nucleus(p, TOPP_P).to(x.dtype)

        def baseline_fn(x):
            p = torch.softmax(_topk_mask_logits(x.float(), K), dim=-1)
            return _nucleus(p, TOPP_P).to(x.dtype)

        arity = 1

    # ===================================================================== #
    # DETERMINISTIC SAMPLING (supplied uniform / gumbel -> reproducible index)
    # ===================================================================== #
    elif op == "smp_categorical_sample":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _uniform((M,), device, seed + 1))

        def ref_fn(x, u):
            return _inv_cdf(_sm(x.float()), u)

        def baseline_fn(x, u):
            cdf = torch.softmax(x.float(), dim=-1).cumsum(-1)
            idx = torch.searchsorted(cdf, u.float().view(-1, 1), right=True).squeeze(-1)
            return idx.clamp_(max=x.shape[-1] - 1).to(torch.int64)

        arity = 2

    elif op == "smp_gumbel_max":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _gumbel((M, V), device, seed + 1))

        def ref_fn(x, gumbel):
            return (x.float() + gumbel.float()).argmax(-1).to(torch.int64)

        def baseline_fn(x, gumbel):
            return (x + gumbel).argmax(-1).to(torch.int64)

        arity = 2

    elif op == "smp_topp_sample":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _uniform((M,), device, seed + 1))

        def ref_fn(x, u):
            return _inv_cdf(_nucleus(_sm(x.float()), TOPP_P), u)

        baseline_fn = None
        arity = 2

    elif op == "smp_topk_sample":
        K = TOPK_SAMPLE_K

        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_randn((M, V), device, seed, scale=2.0), _uniform((M,), device, seed + 1))

        def ref_fn(x, u):
            return _inv_cdf(_sm(_topk_mask_logits(x.float(), K)), u)

        baseline_fn = None
        arity = 2

    # ===================================================================== #
    # SPECULATIVE DECODING
    # ===================================================================== #
    elif op == "smp_spec_accept":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            q = _probs((M, V), device, seed, scale=2.0)
            p = _probs((M, V), device, seed + 1, scale=2.0)
            d = _ids((M,), V, device, seed + 2)
            u = _uniform((M,), device, seed + 3)
            return (q, p, d, u)

        def ref_fn(q, p, d, u):
            di = d.long().view(-1, 1)
            qd = q.float().gather(-1, di).squeeze(-1)
            pd = p.float().gather(-1, di).squeeze(-1)
            accept = torch.clamp(pd / qd, max=1.0)
            return (u.float() <= accept).to(q.dtype)

        def baseline_fn(q, p, d, u):
            di = d.long().view(-1, 1)
            accept = torch.clamp(p.gather(-1, di).squeeze(-1) / q.gather(-1, di).squeeze(-1), max=1.0)
            return (u <= accept).to(q.dtype)

        arity = 4

    elif op == "smp_spec_residual":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_probs((M, V), device, seed, scale=2.0), _probs((M, V), device, seed + 1, scale=2.0))

        def ref_fn(q, p):
            resid = torch.clamp(p.float() - q.float(), min=0.0)
            denom = resid.sum(-1, keepdim=True).clamp_min(1e-20)
            return (resid / denom).to(q.dtype)

        def baseline_fn(q, p):
            resid = torch.clamp(p.float() - q.float(), min=0.0)
            return (resid / resid.sum(-1, keepdim=True).clamp_min(1e-20)).to(q.dtype)

        arity = 2

    elif op == "smp_spec_bonus_token":
        def get_inputs(shape, device="cuda", seed=0):
            M, V = shape["M"], shape["V"]
            return (_probs((M, V), device, seed, scale=2.0), _uniform((M,), device, seed + 1))

        def ref_fn(p, u):
            return _inv_cdf(p.float(), u)

        def baseline_fn(p, u):
            cdf = p.float().cumsum(-1)
            idx = torch.searchsorted(cdf, u.float().view(-1, 1), right=True).squeeze(-1)
            return idx.clamp_(max=p.shape[-1] - 1).to(torch.int64)

        arity = 2

    elif op == "smp_tree_attn_mask":
        def get_inputs(shape, device="cuda", seed=0):
            return (_parents(shape["M"], shape["T"], device, seed),)

        def _mask(parent):
            M, T = parent.shape
            out = torch.zeros((M, T, T), dtype=torch.float32, device=parent.device)
            par = parent.tolist()
            for m in range(M):
                pm = par[m]
                for i in range(T):
                    out[m, i, i] = 1.0
                    a = pm[i]
                    while a >= 0:
                        out[m, i, a] = 1.0
                        a = pm[a]
            return out

        def ref_fn(parent):
            return _mask(parent).to(tdt)

        def baseline_fn(parent):
            return _mask(parent).to(tdt)

        arity = 1

    elif op == "smp_verify_prefix":
        def get_inputs(shape, device="cuda", seed=0):
            M, K = shape["M"], shape["K"]
            g = torch.Generator(device=device).manual_seed(seed)
            draft = torch.randint(0, 1000, (M, K), generator=g, device=device, dtype=torch.int64)
            # target agrees on a random prefix, then diverges (deterministic mix).
            same = torch.rand((M, K), generator=g, device=device) < 0.6
            other = torch.randint(0, 1000, (M, K), generator=g, device=device, dtype=torch.int64)
            target = torch.where(same, draft, other)
            return (draft, target)

        def ref_fn(draft, target):
            eq = (draft == target).to(torch.int64)
            return eq.cumprod(dim=1).sum(dim=1).to(torch.int64)

        def baseline_fn(draft, target):
            eq = (draft == target).to(torch.int64)
            return eq.cumprod(dim=1).sum(dim=1).to(torch.int64)

        arity = 2

    # ===================================================================== #
    # ADVANCED RoPE / POSITIONAL
    # ===================================================================== #
    elif op in _ROPE_OPS and op == "smp_rope_2d":
        def get_inputs(shape, device="cuda", seed=0):
            M, D = shape["M"], shape["D"]
            return (_randn((M, D), device, seed), _positions(M, device, seed + 1),
                    _positions(M, device, seed + 2))

        def ref_fn(x, pos_h, pos_w):
            xf = x.float()
            D = xf.shape[-1]
            Dh = D // 2
            inv = _inv_freq(Dh, ROPE_THETA, xf.device)
            oh = _apply_rope(xf[..., :Dh], inv, pos_h.float())
            ow = _apply_rope(xf[..., Dh:], inv, pos_w.float())
            return torch.cat([oh, ow], dim=-1).to(x.dtype)

        baseline_fn = None
        arity = 3

    elif op in _ROPE_OPS:
        def get_inputs(shape, device="cuda", seed=0):
            M, D = shape["M"], shape["D"]
            return (_randn((M, D), device, seed), _positions(M, device, seed + 1))

        def _inv_for(D, device):
            if op == "smp_rope_linear_pi":
                return _inv_freq(D, ROPE_THETA, device) / ROPE_SCALE, 1.0
            if op == "smp_rope_ntk":
                theta = ROPE_THETA * (ROPE_SCALE ** (D / (D - 2)))
                return _inv_freq(D, theta, device), 1.0
            if op == "smp_rope_dynamic_ntk":
                bf = (ROPE_SCALE * ROPE_DYN_SEQ_LEN / ROPE_ORIG_MAX) - (ROPE_SCALE - 1)
                theta = ROPE_THETA * (bf ** (D / (D - 2)))
                return _inv_freq(D, theta, device), 1.0
            if op == "smp_rope_llama3":
                inv0 = _inv_freq(D, ROPE_THETA, device)
                wl = 2.0 * math.pi / inv0
                low_wl = LLAMA3_OLD_CTX / LLAMA3_LOW_FREQ
                high_wl = LLAMA3_OLD_CTX / LLAMA3_HIGH_FREQ
                inv_low = inv0 / LLAMA3_FACTOR
                smooth = (LLAMA3_OLD_CTX / wl - LLAMA3_LOW_FREQ) / (LLAMA3_HIGH_FREQ - LLAMA3_LOW_FREQ)
                inv_sm = (1.0 - smooth) * inv0 / LLAMA3_FACTOR + smooth * inv0
                inv = torch.where(wl > low_wl, inv_low, torch.where(wl < high_wl, inv0, inv_sm))
                return inv, 1.0
            if op == "smp_rope_yarn":
                half = D // 2
                i = torch.arange(half, device=device, dtype=torch.float32)
                freq = ROPE_THETA ** ((2.0 * i) / D)
                inv_extra = 1.0 / freq
                inv_inter = 1.0 / (ROPE_SCALE * freq)

                def corr(nr):
                    return (D * math.log(ROPE_ORIG_MAX / (nr * 2.0 * math.pi))) / (2.0 * math.log(ROPE_THETA))

                low = max(math.floor(corr(YARN_BETA_FAST)), 0)
                high = min(math.ceil(corr(YARN_BETA_SLOW)), half - 1)
                if high == low:
                    high = low + 0.001
                ramp = torch.clamp((torch.arange(half, device=device, dtype=torch.float32) - low) / (high - low), 0.0, 1.0)
                inv_mask = 1.0 - ramp
                inv = inv_inter * (1.0 - inv_mask) + inv_extra * inv_mask
                mscale = 0.1 * math.log(ROPE_SCALE) + 1.0
                return inv, mscale
            raise ValueError(op)

        def ref_fn(x, pos):
            xf = x.float()
            D = xf.shape[-1]
            if op == "smp_rope_partial":
                rot = int(D * ROT_PCT)
                rot -= rot % 2
                inv = _inv_freq(rot, ROPE_THETA, xf.device)
                o_rot = _apply_rope(xf[..., :rot], inv, pos.float())
                return torch.cat([o_rot, xf[..., rot:]], dim=-1).to(x.dtype)
            inv, mscale = _inv_for(D, xf.device)
            return _apply_rope(xf, inv, pos.float(), mscale).to(x.dtype)

        baseline_fn = None
        arity = 2

    else:
        raise ValueError(f"unknown breadth op {op!r}")

    if baseline_fn is None:
        baseline_fn = ref_fn

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op,
          "dtype_name": dtype, "family": f"breadth_{op}", "mutates_input": False}
    ns[f"{op}_ref"] = ref_fn
    return ns


def seed_source(op: str, dtype: str) -> str:
    """A naive, COMPILING, CORRECT Triton seed - the policy's starting point. The
    numerically-stable softmax / rotation / elementwise math runs in Triton (fp32
    accumulate); the data-dependent selection (sort / nucleus / inverse-CDF / n-gram
    / tree walk) runs host-side in torch - exactly the fusion the policy learns."""
    tldt = DTYPES[dtype][1]
    tdt_name = DTYPES[dtype][0]

    def hdr(desc):
        return (f'"""GENERATED breadth {op} seed ({dtype}). {desc} Naive but correct; the\n'
                'data-dependent selection runs host-side in torch (the policy fuses it)."""\n'
                'from __future__ import annotations\n'
                'import torch, triton, triton.language as tl\n\n\n')

    def elem_kernel(expr):
        return (
            "@triton.jit\n"
            "def _elem_kernel(x_ptr, a_ptr, o_ptr, sx, sa, so, N, BLOCK_N: tl.constexpr):\n"
            "    row = tl.program_id(0)\n"
            "    col = tl.program_id(1)\n"
            "    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)\n"
            "    mask = offs < N\n"
            "    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)\n"
            "    a = tl.load(a_ptr + row * sa + offs, mask=mask, other=0.0).to(tl.float32)\n"
            f"    o = {expr}\n"
            f"    tl.store(o_ptr + row * so + offs, o.to({tldt}), mask=mask)\n\n\n")

    def elem_entry(arg):
        return (
            f"def {op}(x: torch.Tensor, {arg}: torch.Tensor) -> torch.Tensor:\n"
            "    M, N = x.shape\n"
            "    o = torch.empty_like(x)\n"
            "    BLOCK_N = 1024\n"
            "    grid = (M, triton.cdiv(N, BLOCK_N))\n"
            f"    _elem_kernel[grid](x, {arg}, o, x.stride(0), {arg}.stride(0), o.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)\n"
            "    return o\n")

    SM = (
        "@triton.jit\n"
        "def _sm_kernel(x_ptr, o_ptr, sx, so, N, INV_T, BLOCK_N: tl.constexpr):\n"
        "    row = tl.program_id(0)\n"
        "    m = -float('inf')\n"
        "    s = 0.0\n"
        "    for start in range(0, N, BLOCK_N):\n"
        "        offs = start + tl.arange(0, BLOCK_N)\n"
        "        mask = offs < N\n"
        "        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float('inf')).to(tl.float32) * INV_T\n"
        "        blk = tl.max(x, axis=0)\n"
        "        new_m = tl.maximum(m, blk)\n"
        "        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)\n"
        "        m = new_m\n"
        "    for start in range(0, N, BLOCK_N):\n"
        "        offs = start + tl.arange(0, BLOCK_N)\n"
        "        mask = offs < N\n"
        "        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32) * INV_T\n"
        "        tl.store(o_ptr + row * so + offs, tl.exp(x - m) / s, mask=mask)\n\n\n")

    def sm_run(var, invt="1.0"):
        return (
            f"    M, N = {var}.shape\n"
            f"    probs = torch.empty((M, N), device={var}.device, dtype=torch.float32)\n"
            "    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)\n"
            f"    _sm_kernel[(M,)]({var}, probs, {var}.stride(0), probs.stride(0), N, {invt}, BLOCK_N=BLOCK_N, num_warps=8)\n")

    ROPE = (
        "@triton.jit\n"
        "def _rope_kernel(x_ptr, c_ptr, s_ptr, o_ptr, sx, sc, ss, so, HALF, BLOCK_H: tl.constexpr):\n"
        "    row = tl.program_id(0)\n"
        "    offs = tl.arange(0, BLOCK_H)\n"
        "    mask = offs < HALF\n"
        "    x1 = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)\n"
        "    x2 = tl.load(x_ptr + row * sx + HALF + offs, mask=mask, other=0.0).to(tl.float32)\n"
        "    c = tl.load(c_ptr + row * sc + offs, mask=mask, other=0.0).to(tl.float32)\n"
        "    s = tl.load(s_ptr + row * ss + offs, mask=mask, other=0.0).to(tl.float32)\n"
        f"    tl.store(o_ptr + row * so + offs, (x1 * c - x2 * s).to({tldt}), mask=mask)\n"
        f"    tl.store(o_ptr + row * so + HALF + offs, (x2 * c + x1 * s).to({tldt}), mask=mask)\n\n\n")

    # ---- elementwise logit processors ----------------------------------- #
    if op in ("smp_logit_bias", "smp_presence_penalty", "smp_frequency_penalty",
              "smp_repetition_penalty"):
        if op == "smp_logit_bias":
            expr, arg, desc = "x + a", "bias", "logits[M,V] + additive per-token bias."
        elif op == "smp_presence_penalty":
            expr, arg, desc = (f"x - {PRESENCE_PENALTY!r} * tl.where(a > 0.0, 1.0, 0.0)",
                               "counts", "OpenAI presence penalty (subtract if seen).")
        elif op == "smp_frequency_penalty":
            expr, arg, desc = (f"x - {FREQ_PENALTY!r} * a", "counts",
                               "OpenAI frequency penalty (subtract coef * count).")
        else:
            expr, arg, desc = (f"tl.where(a > 0.0, tl.where(x > 0.0, x / {REP_PENALTY!r}, "
                               f"x * {REP_PENALTY!r}), x)", "seen",
                               "CTRL repetition penalty (sign-aware, on seen tokens).")
        return hdr(desc) + elem_kernel(expr) + elem_entry(arg)

    if op == "smp_gumbel_max":
        return (hdr("argmax(logits + supplied gumbel noise) - the Gumbel-max sampler.")
                + elem_kernel("x + a")
                + f"def {op}(x: torch.Tensor, gumbel: torch.Tensor) -> torch.Tensor:\n"
                "    M, N = x.shape\n"
                "    y = torch.empty_like(x)\n"
                "    BLOCK_N = 1024\n"
                "    grid = (M, triton.cdiv(N, BLOCK_N))\n"
                "    _elem_kernel[grid](x, gumbel, y, x.stride(0), gumbel.stride(0), y.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)\n"
                "    return y.argmax(-1).to(torch.int64)\n")

    if op == "smp_no_repeat_ngram":
        return (hdr("block tokens that would repeat a previously-seen n-gram.")
                + elem_kernel("tl.where(a > 0.0, -float('inf'), x)")
                + f"def {op}(x: torch.Tensor, prev_ids: torch.Tensor) -> torch.Tensor:\n"
                "    M, V = x.shape\n"
                "    L = prev_ids.shape[1]\n"
                f"    n = {NGRAM_N}\n"
                "    ban = torch.zeros((M, V), device=x.device, dtype=torch.float32)\n"
                "    if L >= n:\n"
                "        rows = prev_ids.tolist()\n"
                "        for i in range(M):\n"
                "            row = rows[i]\n"
                "            suffix = row[L - n + 1:L]\n"
                "            for j in range(0, L - n + 1):\n"
                "                if row[j:j + n - 1] == suffix:\n"
                "                    ban[i, row[j + n - 1]] = 1.0\n"
                "    o = torch.empty_like(x)\n"
                "    BLOCK_N = 1024\n"
                "    grid = (M, triton.cdiv(V, BLOCK_N))\n"
                "    _elem_kernel[grid](x, ban, o, x.stride(0), ban.stride(0), o.stride(0), V, BLOCK_N=BLOCK_N, num_warps=4)\n"
                "    return o\n")

    # ---- softmax-based renorm / mask ------------------------------------ #
    if op == "smp_temperature":
        return (hdr("temperature-scaled softmax over the vocab (stable, max-subtracted).")
                + SM + f"def {op}(x: torch.Tensor) -> torch.Tensor:\n"
                + sm_run("x", repr(1.0 / TEMP)) + "    return probs.to(x.dtype)\n")

    if op in TOPK_MASK_SIZES:
        K = TOPK_MASK_SIZES[op]
        return (hdr(f"keep the top-{K} logits per row, mask the rest to -inf.")
                + "@triton.jit\n"
                "def _mask_kernel(x_ptr, thr_ptr, o_ptr, sx, so, N, BLOCK_N: tl.constexpr):\n"
                "    row = tl.program_id(0)\n"
                "    col = tl.program_id(1)\n"
                "    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)\n"
                "    mask = offs < N\n"
                "    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float('inf')).to(tl.float32)\n"
                "    thr = tl.load(thr_ptr + row).to(tl.float32)\n"
                "    o = tl.where(x >= thr, x, -float('inf'))\n"
                f"    tl.store(o_ptr + row * so + offs, o.to({tldt}), mask=mask)\n\n\n"
                f"def {op}(x: torch.Tensor) -> torch.Tensor:\n"
                "    M, N = x.shape\n"
                f"    K = {K}\n"
                "    thr = torch.topk(x, K, dim=-1).values[:, -1].contiguous()\n"
                "    o = torch.empty_like(x)\n"
                "    BLOCK_N = 1024\n"
                "    grid = (M, triton.cdiv(N, BLOCK_N))\n"
                "    _mask_kernel[grid](x, thr, o, x.stride(0), o.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)\n"
                "    return o\n")

    if op == "smp_topp_renorm":
        return (hdr("top-p (nucleus) renormalized probabilities.")
                + SM + f"def {op}(x: torch.Tensor) -> torch.Tensor:\n" + sm_run("x")
                + "    sp, si = torch.sort(probs, dim=-1, descending=True)\n"
                "    excl = sp.cumsum(-1) - sp\n"
                f"    keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, si, excl <= {TOPP_P!r})\n"
                "    masked = torch.where(keep, probs, torch.zeros_like(probs))\n"
                "    return (masked / masked.sum(-1, keepdim=True)).to(x.dtype)\n")

    if op == "smp_minp_mask":
        return (hdr("min-p mask: keep probs >= min_p * peak, then renormalize.")
                + SM + f"def {op}(x: torch.Tensor) -> torch.Tensor:\n" + sm_run("x")
                + f"    keep = probs >= {MIN_P!r} * probs.amax(-1, keepdim=True)\n"
                "    masked = torch.where(keep, probs, torch.zeros_like(probs))\n"
                "    return (masked / masked.sum(-1, keepdim=True)).to(x.dtype)\n")

    if op == "smp_typical_mask":
        return (hdr("locally-typical mask: keep the lowest-surprisal-deviation tokens.")
                + SM + f"def {op}(x: torch.Tensor) -> torch.Tensor:\n" + sm_run("x")
                + "    logp = torch.log(probs.clamp_min(1e-30))\n"
                "    H = -(probs * logp).sum(-1, keepdim=True)\n"
                "    dev = ((-logp) - H).abs()\n"
                "    sd, si = torch.sort(dev, dim=-1, descending=False)\n"
                "    p_sorted = probs.gather(-1, si)\n"
                "    excl = p_sorted.cumsum(-1) - p_sorted\n"
                f"    keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, si, excl <= {TYPICAL_MASS!r})\n"
                "    masked = torch.where(keep, probs, torch.zeros_like(probs))\n"
                "    return (masked / masked.sum(-1, keepdim=True)).to(x.dtype)\n")

    if op == "smp_topk_topp":
        return (hdr("combined top-k then top-p renormalized probabilities.")
                + SM + f"def {op}(x: torch.Tensor) -> torch.Tensor:\n"
                f"    K = {TOPK_SAMPLE_K}\n"
                "    thr = torch.topk(x, K, dim=-1).values[:, -1:]\n"
                "    xk = torch.where(x >= thr, x, torch.full_like(x, float('-inf'))).contiguous()\n"
                + sm_run("xk")
                + "    sp, si = torch.sort(probs, dim=-1, descending=True)\n"
                "    excl = sp.cumsum(-1) - sp\n"
                f"    keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, si, excl <= {TOPP_P!r})\n"
                "    masked = torch.where(keep, probs, torch.zeros_like(probs))\n"
                "    return (masked / masked.sum(-1, keepdim=True)).to(x.dtype)\n")

    # ---- deterministic sampling (inverse-CDF from a supplied uniform) ---- #
    if op == "smp_categorical_sample":
        return (hdr("categorical inverse-CDF sample from softmax(logits) via supplied u.")
                + SM + f"def {op}(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:\n"
                + sm_run("x")
                + "    cdf = probs.cumsum(-1)\n"
                "    idx = torch.searchsorted(cdf, u.float().view(-1, 1), right=True).squeeze(-1)\n"
                "    return idx.clamp_(max=N - 1).to(torch.int64)\n")

    if op == "smp_topp_sample":
        return (hdr("top-p renormalize then inverse-CDF sample via supplied u.")
                + SM + f"def {op}(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:\n"
                + sm_run("x")
                + "    sp, si = torch.sort(probs, dim=-1, descending=True)\n"
                "    excl = sp.cumsum(-1) - sp\n"
                f"    keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, si, excl <= {TOPP_P!r})\n"
                "    masked = torch.where(keep, probs, torch.zeros_like(probs))\n"
                "    pp = masked / masked.sum(-1, keepdim=True)\n"
                "    cdf = pp.cumsum(-1)\n"
                "    idx = torch.searchsorted(cdf, u.float().view(-1, 1), right=True).squeeze(-1)\n"
                "    return idx.clamp_(max=N - 1).to(torch.int64)\n")

    if op == "smp_topk_sample":
        return (hdr("top-k mask + renormalize then inverse-CDF sample via supplied u.")
                + SM + f"def {op}(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:\n"
                f"    K = {TOPK_SAMPLE_K}\n"
                "    thr = torch.topk(x, K, dim=-1).values[:, -1:]\n"
                "    xk = torch.where(x >= thr, x, torch.full_like(x, float('-inf'))).contiguous()\n"
                + sm_run("xk")
                + "    cdf = probs.cumsum(-1)\n"
                "    idx = torch.searchsorted(cdf, u.float().view(-1, 1), right=True).squeeze(-1)\n"
                "    return idx.clamp_(max=N - 1).to(torch.int64)\n")

    # ---- speculative decoding ------------------------------------------- #
    if op == "smp_spec_accept":
        return (hdr("rejection-sampling accept: u <= min(1, p_target[d]/q_draft[d]).")
                + "@triton.jit\n"
                "def _acc_kernel(a_ptr, u_ptr, o_ptr, M, BLOCK: tl.constexpr):\n"
                "    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
                "    mask = offs < M\n"
                "    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)\n"
                "    u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)\n"
                "    o = tl.where(u <= a, 1.0, 0.0)\n"
                f"    tl.store(o_ptr + offs, o.to({tldt}), mask=mask)\n\n\n"
                f"def {op}(q: torch.Tensor, p: torch.Tensor, d: torch.Tensor, u: torch.Tensor) -> torch.Tensor:\n"
                "    M, V = q.shape\n"
                "    di = d.long().view(-1, 1)\n"
                "    ratio = p.float().gather(-1, di).squeeze(-1) / q.float().gather(-1, di).squeeze(-1)\n"
                "    accept = torch.clamp(ratio, max=1.0).contiguous()\n"
                "    uu = u.float().contiguous()\n"
                f"    o = torch.empty((M,), device=q.device, dtype=torch.{tdt_name})\n"
                "    BLOCK = 256\n"
                "    _acc_kernel[(triton.cdiv(M, BLOCK),)](accept, uu, o, M, BLOCK=BLOCK, num_warps=4)\n"
                "    return o\n")

    if op == "smp_spec_residual":
        return (hdr("corrected residual distribution normalize(relu(p_target - q_draft)).")
                + "@triton.jit\n"
                "def _resid_kernel(p_ptr, q_ptr, o_ptr, sp, sq, so, N, BLOCK_N: tl.constexpr):\n"
                "    row = tl.program_id(0)\n"
                "    col = tl.program_id(1)\n"
                "    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)\n"
                "    mask = offs < N\n"
                "    p = tl.load(p_ptr + row * sp + offs, mask=mask, other=0.0).to(tl.float32)\n"
                "    q = tl.load(q_ptr + row * sq + offs, mask=mask, other=0.0).to(tl.float32)\n"
                "    tl.store(o_ptr + row * so + offs, tl.maximum(p - q, 0.0), mask=mask)\n\n\n"
                f"def {op}(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:\n"
                "    M, N = q.shape\n"
                "    resid = torch.empty((M, N), device=q.device, dtype=torch.float32)\n"
                "    BLOCK_N = 1024\n"
                "    grid = (M, triton.cdiv(N, BLOCK_N))\n"
                "    _resid_kernel[grid](p, q, resid, p.stride(0), q.stride(0), resid.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)\n"
                "    denom = resid.sum(-1, keepdim=True).clamp_min(1e-20)\n"
                "    return (resid / denom).to(q.dtype)\n")

    if op == "smp_spec_bonus_token":
        return (hdr("bonus token: inverse-CDF sample from the target distribution.")
                + "@triton.jit\n"
                "def _cumsum_kernel(p_ptr, o_ptr, sp, so, N):\n"
                "    row = tl.program_id(0)\n"
                "    acc = 0.0\n"
                "    for i in range(0, N):\n"
                "        v = tl.load(p_ptr + row * sp + i).to(tl.float32)\n"
                "        acc += v\n"
                "        tl.store(o_ptr + row * so + i, acc)\n\n\n"
                f"def {op}(p: torch.Tensor, u: torch.Tensor) -> torch.Tensor:\n"
                "    M, N = p.shape\n"
                "    cdf = torch.empty((M, N), device=p.device, dtype=torch.float32)\n"
                "    _cumsum_kernel[(M,)](p, cdf, p.stride(0), cdf.stride(0), N, num_warps=1)\n"
                "    idx = torch.searchsorted(cdf, u.float().view(-1, 1), right=True).squeeze(-1)\n"
                "    return idx.clamp_(max=N - 1).to(torch.int64)\n")

    if op == "smp_tree_attn_mask":
        return (hdr("tree-attention mask from a parent table (node attends to its ancestors).")
                + "@triton.jit\n"
                "def _copy_kernel(x_ptr, o_ptr, sx, so, N, BLOCK_N: tl.constexpr):\n"
                "    row = tl.program_id(0)\n"
                "    offs = tl.arange(0, BLOCK_N)\n"
                "    mask = offs < N\n"
                "    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)\n"
                f"    tl.store(o_ptr + row * so + offs, x.to({tldt}), mask=mask)\n\n\n"
                f"def {op}(parent: torch.Tensor) -> torch.Tensor:\n"
                "    M, T = parent.shape\n"
                "    mask = torch.zeros((M, T, T), device=parent.device, dtype=torch.float32)\n"
                "    par = parent.tolist()\n"
                "    for m in range(M):\n"
                "        pm = par[m]\n"
                "        for it in range(T):\n"
                "            mask[m, it, it] = 1.0\n"
                "            a = pm[it]\n"
                "            while a >= 0:\n"
                "                mask[m, it, a] = 1.0\n"
                "                a = pm[a]\n"
                "    flat = mask.reshape(M * T, T)\n"
                f"    o = torch.empty((M * T, T), device=parent.device, dtype=torch.{tdt_name})\n"
                "    BLOCK_N = triton.next_power_of_2(T)\n"
                "    _copy_kernel[(M * T,)](flat, o, flat.stride(0), o.stride(0), T, BLOCK_N=BLOCK_N, num_warps=1)\n"
                "    return o.reshape(M, T, T)\n")

    if op == "smp_verify_prefix":
        return (hdr("verify-and-accept the longest matching draft/target prefix per row.")
                + f"def {op}(draft: torch.Tensor, target: torch.Tensor) -> torch.Tensor:\n"
                "    eq = (draft == target).to(torch.int64)\n"
                "    return eq.cumprod(dim=1).sum(dim=1).to(torch.int64)\n")

    # ---- advanced RoPE / positional ------------------------------------- #
    if op == "smp_rope_2d":
        return (hdr("2D RoPE: two head-dim halves rotated by two position coordinates.")
                + ROPE
                + f"def {op}(x: torch.Tensor, pos_h: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:\n"
                "    xf = x.float().contiguous()\n"
                "    M, D = xf.shape\n"
                "    device = xf.device\n"
                "    Dh = D // 2\n"
                "    half = Dh // 2\n"
                "    i = torch.arange(half, device=device, dtype=torch.float32)\n"
                f"    inv = {ROPE_THETA!r} ** (-(2.0 * i) / Dh)\n"
                "    ah = pos_h.float()[:, None] * inv[None, :]\n"
                "    aw = pos_w.float()[:, None] * inv[None, :]\n"
                "    ch = torch.cos(ah).contiguous(); sh = torch.sin(ah).contiguous()\n"
                "    cw = torch.cos(aw).contiguous(); sw = torch.sin(aw).contiguous()\n"
                "    xh = xf[:, :Dh].contiguous(); xw = xf[:, Dh:].contiguous()\n"
                f"    oh = torch.empty((M, Dh), device=device, dtype=torch.{tdt_name})\n"
                f"    ow = torch.empty((M, Dh), device=device, dtype=torch.{tdt_name})\n"
                "    BLOCK_H = triton.next_power_of_2(half)\n"
                "    _rope_kernel[(M,)](xh, ch, sh, oh, xh.stride(0), ch.stride(0), sh.stride(0), oh.stride(0), half, BLOCK_H=BLOCK_H, num_warps=4)\n"
                "    _rope_kernel[(M,)](xw, cw, sw, ow, xw.stride(0), cw.stride(0), sw.stride(0), ow.stride(0), half, BLOCK_H=BLOCK_H, num_warps=4)\n"
                "    return torch.cat([oh, ow], dim=-1)\n")

    if op == "smp_rope_partial":
        return (hdr("partial rotary: rotate the first rot_pct of head_dim, pass the rest.")
                + ROPE
                + f"def {op}(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:\n"
                "    xf = x.float().contiguous()\n"
                "    M, D = xf.shape\n"
                "    device = xf.device\n"
                f"    rot = int(D * {ROT_PCT!r})\n"
                "    rot -= rot % 2\n"
                "    half = rot // 2\n"
                "    i = torch.arange(half, device=device, dtype=torch.float32)\n"
                f"    inv = {ROPE_THETA!r} ** (-(2.0 * i) / rot)\n"
                "    ang = pos.float()[:, None] * inv[None, :]\n"
                "    c = torch.cos(ang).contiguous()\n"
                "    s = torch.sin(ang).contiguous()\n"
                f"    o = xf.to(torch.{tdt_name}).clone()\n"
                "    xr = xf[:, :rot].contiguous()\n"
                f"    orot = torch.empty((M, rot), device=device, dtype=torch.{tdt_name})\n"
                "    BLOCK_H = triton.next_power_of_2(half)\n"
                "    _rope_kernel[(M,)](xr, c, s, orot, xr.stride(0), c.stride(0), s.stride(0), orot.stride(0), half, BLOCK_H=BLOCK_H, num_warps=4)\n"
                "    o[:, :rot] = orot\n"
                "    return o\n")

    if op in _ROPE_OPS:
        if op == "smp_rope_linear_pi":
            inv_src = (f"    inv = ({ROPE_THETA!r} ** (-(2.0 * i) / D)) / {ROPE_SCALE!r}\n"
                       "    mscale = 1.0\n")
            desc = "linear position-interpolation RoPE."
        elif op == "smp_rope_ntk":
            inv_src = (f"    theta = {ROPE_THETA!r} * ({ROPE_SCALE!r} ** (D / (D - 2)))\n"
                       "    inv = theta ** (-(2.0 * i) / D)\n    mscale = 1.0\n")
            desc = "static NTK-aware scaled RoPE."
        elif op == "smp_rope_dynamic_ntk":
            inv_src = (f"    bf = ({ROPE_SCALE!r} * {ROPE_DYN_SEQ_LEN!r} / {ROPE_ORIG_MAX!r}) - ({ROPE_SCALE!r} - 1)\n"
                       f"    theta = {ROPE_THETA!r} * (bf ** (D / (D - 2)))\n"
                       "    inv = theta ** (-(2.0 * i) / D)\n    mscale = 1.0\n")
            desc = "dynamic-NTK RoPE (base grows with sequence length)."
        elif op == "smp_rope_llama3":
            inv_src = (f"    inv0 = {ROPE_THETA!r} ** (-(2.0 * i) / D)\n"
                       "    wl = 6.283185307179586 / inv0\n"
                       f"    low_wl = {LLAMA3_OLD_CTX!r} / {LLAMA3_LOW_FREQ!r}\n"
                       f"    high_wl = {LLAMA3_OLD_CTX!r} / {LLAMA3_HIGH_FREQ!r}\n"
                       f"    inv_low = inv0 / {LLAMA3_FACTOR!r}\n"
                       f"    smooth = ({LLAMA3_OLD_CTX!r} / wl - {LLAMA3_LOW_FREQ!r}) / ({LLAMA3_HIGH_FREQ!r} - {LLAMA3_LOW_FREQ!r})\n"
                       f"    inv_sm = (1.0 - smooth) * inv0 / {LLAMA3_FACTOR!r} + smooth * inv0\n"
                       "    inv = torch.where(wl > low_wl, inv_low, torch.where(wl < high_wl, inv0, inv_sm))\n"
                       "    mscale = 1.0\n")
            desc = "Llama-3 frequency-smoothing RoPE."
        else:  # smp_rope_yarn
            inv_src = ("    import math\n"
                       f"    freq = {ROPE_THETA!r} ** ((2.0 * i) / D)\n"
                       "    inv_extra = 1.0 / freq\n"
                       f"    inv_inter = 1.0 / ({ROPE_SCALE!r} * freq)\n"
                       "    def corr(nr):\n"
                       f"        return (D * math.log({ROPE_ORIG_MAX!r} / (nr * 2.0 * math.pi))) / (2.0 * math.log({ROPE_THETA!r}))\n"
                       f"    low = max(math.floor(corr({YARN_BETA_FAST!r})), 0)\n"
                       f"    high = min(math.ceil(corr({YARN_BETA_SLOW!r})), half - 1)\n"
                       "    if high == low:\n        high = low + 0.001\n"
                       "    ramp = torch.clamp((i - low) / (high - low), 0.0, 1.0)\n"
                       "    inv_mask = 1.0 - ramp\n"
                       "    inv = inv_inter * (1.0 - inv_mask) + inv_extra * inv_mask\n"
                       f"    mscale = 0.1 * math.log({ROPE_SCALE!r}) + 1.0\n")
            desc = "YaRN RoPE (ramped interp/extrapolation + attention mscale)."
        return (hdr(desc) + ROPE
                + f"def {op}(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:\n"
                "    xf = x.float().contiguous()\n"
                "    M, D = xf.shape\n"
                "    half = D // 2\n"
                "    device = xf.device\n"
                "    i = torch.arange(half, device=device, dtype=torch.float32)\n"
                + inv_src
                + "    ang = pos.float()[:, None] * inv[None, :]\n"
                "    c = (torch.cos(ang) * mscale).contiguous()\n"
                "    s = (torch.sin(ang) * mscale).contiguous()\n"
                f"    o = torch.empty((M, D), device=device, dtype=torch.{tdt_name})\n"
                "    BLOCK_H = triton.next_power_of_2(half)\n"
                "    _rope_kernel[(M,)](xf, c, s, o, xf.stride(0), c.stride(0), s.stride(0), o.stride(0), half, BLOCK_H=BLOCK_H, num_warps=4)\n"
                "    return o\n")

    raise ValueError(f"unknown breadth op {op!r}")


def op_names() -> list[str]:
    return list(OPS)
