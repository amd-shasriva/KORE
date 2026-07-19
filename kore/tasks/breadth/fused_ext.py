"""Breadth fused-transformer-block task-authoring engine (torch-baselined).

Widens the KORE suite with the HARD *fused* transformer-block kernels - the real
production fusions where the win is a collapsed kernel-launch chain plus saved
memory traffic (keep the intermediate in registers/LDS instead of round-tripping
HBM). No trivial single ops live here: every task fuses a genuine multi-stage
block (a GEMM + split, a 2-GEMM gated MLP, RoPE mixing, the bias/dropout/residual
/norm block "glue", a norm+quantize-out, an attention-logits epilogue, ...).

Contract mirrors ``kore/tasks/breadth/norm_ext.py`` so the shared ``_genops``
driver + generator machinery consume it unchanged:

    OPS / OP_DTYPES / SHAPES              module-level task catalog (every op name
        is prefixed ``fx_``; 32 tasks).
    make_reference(op, dtype) -> dict     reference.py namespace (parse_shape,
        get_inputs, ref_fn EXACT fp32 oracle of the FULL fused computation - casts
        back, may return TUPLES for the split / residual-passthrough / quant ops;
        baseline_fn torch; arity; entry_name=op; dtype_name; family=f"breadth_{op}";
        mutates_input where the op writes an input in place - the kv-cache writes).
    seed_source(op, dtype) -> str         a naive, COMPILING, correct Triton seed
        (the policy's starting point) defining ``def <op>(*inputs)``.

CORRECTNESS is paramount: every ``ref_fn`` recomputes the entire fused block in
fp32 and casts to the task dtype. Every oracle is validated on CPU against an
INDEPENDENT torch computation that composes the sub-ops separately (F.linear +
F.silu + mul + F.linear for the SwiGLU MLP, an apply-rope reference for RoPE, a
stable manual softmax for the logits epilogue, ...) at tight fp32 tolerance - see
tests/test_fused_ext.py. torch is imported lazily inside make_reference so
registry discovery never needs a GPU.
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# --------------------------------------------------------------------------- #
# Task constants (MUST match the seed kernels)
# --------------------------------------------------------------------------- #
EPS = 1e-6              # rms/layernorm epsilon (added inside the variance)
DROP_P = 0.1           # dropout probability for the dropout-fused blocks
INV_KEEP = 1.0 / (1.0 - DROP_P)   # 1/(1-p) inverted-dropout scale
FP8_MAX = 448.0        # OCP e4m3fn max finite (gfx950/CDNA4 native fp8) - quant clamp
INT8_MAX = 127.0       # int8 symmetric quant clamp
ROPE_BASE = 10000.0    # RoPE inverse-frequency base (theta)
SOFTCAP = 50.0         # attention-logit soft-cap (Gemma-style tanh cap)


# --------------------------------------------------------------------------- #
# Task catalog: op -> spec (kind + optional variant knobs).
# Every op name is prefixed ``fx_``. 32 hard fused-transformer-block tasks.
# (No op name contains the reserved substrings.)
# --------------------------------------------------------------------------- #
_SPECS: dict[str, dict] = {
    # -- fused RoPE on q,k : interleaved + half-rotation (+ optional QK-RMSNorm) --
    "fx_rope_qk_half":               {"kind": "rope_half"},
    "fx_rope_qk_interleaved":        {"kind": "rope_interleaved"},
    "fx_rope_qk_half_qknorm":        {"kind": "rope_half_qknorm"},
    "fx_rope_qk_interleaved_qknorm": {"kind": "rope_interleaved_qknorm"},
    # -- fused QKV projection (one GEMM) then split into q,k,v ------------------
    "fx_qkv_proj_split":             {"kind": "qkv_split", "has_bias": False},
    "fx_qkv_proj_split_bias":        {"kind": "qkv_split", "has_bias": True},
    # -- fused 2-GEMM gated MLP (gate,up GEMMs -> act(gate)*up -> down GEMM) -----
    "fx_swiglu_mlp":                 {"kind": "glu_mlp", "act": "silu"},
    "fx_geglu_mlp":                  {"kind": "glu_mlp", "act": "gelu"},
    "fx_reglu_mlp":                  {"kind": "glu_mlp", "act": "relu"},
    "fx_swiglu_mlp_gateup":          {"kind": "glu_mlp_gateup", "act": "silu"},
    # -- fused bias + dropout + residual-add + (RMS/Layer)Norm (block glue) -----
    "fx_bias_dropout_add_rmsnorm":   {"kind": "bias_drop_add_norm", "norm": "rms",   "has_bias": True},
    "fx_bias_dropout_add_layernorm": {"kind": "bias_drop_add_norm", "norm": "layer", "has_bias": True},
    "fx_dropout_add_rmsnorm":        {"kind": "bias_drop_add_norm", "norm": "rms",   "has_bias": False},
    "fx_dropout_add_layernorm":      {"kind": "bias_drop_add_norm", "norm": "layer", "has_bias": False},
    # -- fused add-residual + Norm + fp8/int8 quantize-out ---------------------
    "fx_add_rmsnorm_quant_fp8":      {"kind": "add_rmsnorm_quant"},
    "fx_add_rmsnorm_quant_int8":     {"kind": "add_rmsnorm_quant"},
    "fx_add_layernorm_quant_fp8":    {"kind": "add_layernorm_quant"},
    # -- fused attention-output projection + residual-add ----------------------
    "fx_attn_out_proj_add":          {"kind": "out_proj_add", "has_bias": False},
    "fx_attn_out_proj_add_bias":     {"kind": "out_proj_add", "has_bias": True},
    # -- fused embedding lookup + scale (+ positional add) ---------------------
    "fx_embed_scale":                {"kind": "embed", "pos": False},
    "fx_embed_scale_pos":            {"kind": "embed", "pos": True},
    # -- fused GLU activation-only (gate*act) at large hidden ------------------
    "fx_swiglu_act":                 {"kind": "glu_act", "act": "silu"},
    "fx_geglu_act":                  {"kind": "glu_act", "act": "gelu"},
    "fx_reglu_act":                  {"kind": "glu_act", "act": "relu"},
    # -- fused softcap + (mask) + softmax (attention-logits epilogue) ----------
    "fx_softcap_softmax":            {"kind": "softcap_softmax", "masked": False},
    "fx_softcap_mask_softmax":       {"kind": "softcap_softmax", "masked": True},
    # -- fused rotary + kv-cache write (apply rope to k then store into cache) --
    "fx_rope_kvcache_half":          {"kind": "rope_kvcache", "mode": "half"},
    "fx_rope_kvcache_interleaved":   {"kind": "rope_kvcache", "mode": "interleaved"},
    # -- fused (RMS/Layer)Norm + Linear (norm then x@W) ------------------------
    "fx_rmsnorm_linear":             {"kind": "norm_linear", "norm": "rms"},
    "fx_layernorm_linear":           {"kind": "norm_linear", "norm": "layer"},
    # -- fused residual + dropout + scale (stochastic depth / LayerScale) ------
    "fx_resid_dropout_scale":            {"kind": "resid_drop_scale", "layerscale": False},
    "fx_resid_dropout_scale_layerscale": {"kind": "resid_drop_scale", "layerscale": True},
}

OPS: list[str] = list(_SPECS)
KIND: dict[str, str] = {op: s["kind"] for op, s in _SPECS.items()}

# The only ops that write an input tensor in place are the rotary + kv-cache
# writes (rope is applied to k and the result is stored INTO the supplied cache).
FX_MUTATES_INPUT: frozenset[str] = frozenset(
    {op for op, s in _SPECS.items() if s["kind"] == "rope_kvcache"})

_QUANT_KINDS = ("add_rmsnorm_quant", "add_layernorm_quant")


# --------------------------------------------------------------------------- #
# Shape catalog (realistic transformer activation / weight shapes with a
# non-pow2 tail in every validation set).
# --------------------------------------------------------------------------- #
_ROPE = {  # q,k[B,S,H,D]  (D even, per-head rotary over the head dim)
    "minimal": {"B": 1, "S": 16, "H": 4, "D": 64},
    "primary": {"B": 1, "S": 4096, "H": 32, "D": 128},   # Llama-3 8B attention
    "validation": [{"B": 2, "S": 2048, "H": 32, "D": 128},
                   {"B": 1, "S": 8192, "H": 40, "D": 128},
                   {"B": 1, "S": 4096, "H": 16, "D": 96}],   # batched, wide, non-pow2 D
}
_KV = {  # k[B,S,H,D] roped and written into cache[B,S,H,D]
    "minimal": {"B": 1, "S": 16, "H": 4, "D": 64},
    "primary": {"B": 1, "S": 4096, "H": 8, "D": 128},    # GQA kv-heads
    "validation": [{"B": 2, "S": 2048, "H": 8, "D": 128},
                   {"B": 1, "S": 8192, "H": 8, "D": 128},
                   {"B": 1, "S": 4096, "H": 8, "D": 96}],
}
_MM = {  # x[M,K] @ W[K,N]  (attention proj / out-proj / norm-linear)
    "minimal": {"M": 64, "K": 256, "N": 256},
    "primary": {"M": 4096, "K": 4096, "N": 4096},
    "validation": [{"M": 16384, "K": 4096, "N": 4096},
                   {"M": 4096, "K": 8192, "N": 8192},
                   {"M": 4096, "K": 4096, "N": 4095}],   # non-pow2 output tail
}
_MLP = {  # x[M,K]; Wg,Wu[K,N]; Wd[N,K]  (N = the intermediate/hidden dim)
    "minimal": {"M": 64, "K": 256, "N": 512},
    "primary": {"M": 4096, "K": 4096, "N": 14336},       # Llama-2-13B-style MLP
    "validation": [{"M": 16384, "K": 4096, "N": 11008},
                   {"M": 4096, "K": 8192, "N": 14336},
                   {"M": 2048, "K": 4096, "N": 11007}],   # non-pow2 intermediate
}
_GLU2D = {  # x[M, 2H] -> [M, H]  (activation-only GLU, input width even)
    "minimal": {"M": 64, "N": 512},
    "primary": {"M": 4096, "N": 28672},                  # 2 * 14336
    "validation": [{"M": 16384, "N": 22016},
                   {"M": 4096, "N": 28672},
                   {"M": 8192, "N": 22014}],              # even non-pow2 (H = 11007)
}
_NORM2D = {  # x[M,N]  (block-glue / residual / quant activations)
    "minimal": {"M": 64, "N": 2048},
    "primary": {"M": 4096, "N": 8192},
    "validation": [{"M": 16384, "N": 4096}, {"M": 4096, "N": 16384},
                   {"M": 8192, "N": 8191}],               # non-pow2 hidden tail
}
_EMB = {  # ids[M] in [0,V); weight[V,D]
    "minimal": {"V": 128, "D": 64, "M": 32},
    "primary": {"V": 32000, "D": 4096, "M": 4096},        # Llama vocab
    "validation": [{"V": 128256, "D": 4096, "M": 8192},   # Llama-3 vocab
                   {"V": 32000, "D": 8192, "M": 4096},
                   {"V": 50257, "D": 4096, "M": 4095}],    # GPT-2 vocab, non-pow2 M
}
_ATTN = {  # scores[R, Ncol]  (R = B*H*Sq flattened query rows, Ncol = Skv)
    "minimal": {"R": 32, "Ncol": 40},
    "primary": {"R": 131072, "Ncol": 4096},               # 32 heads * 4096 q
    "validation": [{"R": 65536, "Ncol": 2048},
                   {"R": 40960, "Ncol": 8192},
                   {"R": 8192, "Ncol": 8191}],             # non-pow2 kv tail
}

_SHAPE_TMPL = {
    "rope_half": _ROPE, "rope_interleaved": _ROPE,
    "rope_half_qknorm": _ROPE, "rope_interleaved_qknorm": _ROPE,
    "rope_kvcache": _KV,
    "qkv_split": _MM, "out_proj_add": _MM, "norm_linear": _MM,
    "glu_mlp": _MLP, "glu_mlp_gateup": _MLP,
    "glu_act": _GLU2D,
    "bias_drop_add_norm": _NORM2D, "add_rmsnorm_quant": _NORM2D,
    "add_layernorm_quant": _NORM2D, "resid_drop_scale": _NORM2D,
    "embed": _EMB, "softcap_softmax": _ATTN,
}

SHAPES: dict[str, dict] = {op: _SHAPE_TMPL[KIND[op]] for op in OPS}


# --------------------------------------------------------------------------- #
# dtype sweep: bf16/fp16 default; the norm+quant-out variants sweep their single
# baked-in quant dtype (fp8 / int8).
# --------------------------------------------------------------------------- #
def op_dtypes(op: str) -> list[str]:
    if KIND[op] in _QUANT_KINDS:
        return ["fp8"] if op.endswith("_fp8") else ["int8"]
    return ["bf16", "fp16"]


OP_DTYPES: dict[str, list[str]] = {op: op_dtypes(op) for op in OPS}


def input_dtype(op: str, dtype: str) -> str:
    """torch dtype attr name of the FLOAT inputs for (op, dtype).

    The quant-out variants take bf16 activations (dtype selects the fp8/int8
    OUTPUT), so their float inputs are always bfloat16; everything else uses the
    task dtype. (Integer index tensors - embedding ids - are not covered here.)"""
    if KIND[op] in _QUANT_KINDS:
        return "bfloat16"
    return DTYPES[dtype][0]


# --------------------------------------------------------------------------- #
# reference.py namespace (fp32 oracle of the FULL fused block + torch baseline).
# torch imported lazily so registry discovery never needs a GPU.
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import torch

    spec = _SPECS[op]
    kind = spec["kind"]
    is_quant = kind in _QUANT_KINDS

    if is_quant:
        act_tdt = torch.bfloat16                     # activations are bf16
        q_torch = getattr(torch, DTYPES[dtype][0])   # float8_e4m3fn | int8
        qmax = FP8_MAX if dtype == "fp8" else INT8_MAX
    else:
        act_tdt = getattr(torch, DTYPES[dtype][0])
        q_torch, qmax = None, None

    # -- input fills -------------------------------------------------------
    def _randn(shape, seed, device, scale=1.0, dt=None):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape, generator=g, device=device,
                            dtype=torch.float32) * scale).to(dt or act_tdt)

    def _w2d(shape, seed, device, fan_in):    # GEMM weight ~ N(0, 1/fan_in)
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape, generator=g, device=device,
                            dtype=torch.float32) * (fan_in ** -0.5)).to(act_tdt)

    def _wt(n, seed, device):     # affine / norm weight ~ N(1, 0.1)
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn((n,), generator=g, device=device,
                            dtype=torch.float32) * 0.1 + 1.0).to(act_tdt)

    def _bs(n, seed, device):     # affine bias ~ N(0, 0.1)
        return _randn((n,), seed, device, scale=0.1)

    def _mask(shape, seed, device):    # deterministic Bernoulli(keep) dropout mask
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.rand(shape, generator=g, device=device) > DROP_P).to(act_tdt)

    # -- fp32 math primitives ---------------------------------------------
    def _rms(xf, w, eps=EPS):
        r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return xf * r * w.float()

    def _ln(xf, w, b, eps=EPS):
        mean = xf.mean(-1, keepdim=True)
        var = (xf - mean).pow(2).mean(-1, keepdim=True)
        y = (xf - mean) * torch.rsqrt(var + eps) * w.float()
        if b is not None:
            y = y + b.float()
        return y

    def _act(name, t):
        if name == "silu":
            return t * torch.sigmoid(t)
        if name == "gelu":   # tanh approximation (matches F.gelu(approximate='tanh'))
            return 0.5 * t * (1.0 + torch.tanh(
                0.7978845608028654 * (t + 0.044715 * t.pow(3))))
        return torch.clamp(t, min=0.0)   # relu

    def _quant(normed):
        amax = normed.abs().amax(-1, keepdim=True)
        scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
        q = normed / scale
        if dtype == "int8":
            q = q.round().clamp(-qmax, qmax)
        return q.to(q_torch), scale.squeeze(-1).to(torch.float32)

    # -- RoPE tables + application ----------------------------------------
    def _rope_tables(S, D, device, interleaved):
        half = D // 2
        inv_freq = ROPE_BASE ** (-(torch.arange(0, half, dtype=torch.float32,
                                                device=device) * 2.0 / D))
        pos = torch.arange(S, dtype=torch.float32, device=device)
        freqs = torch.outer(pos, inv_freq)                 # [S, half]
        if interleaved:
            return freqs.cos().to(act_tdt), freqs.sin().to(act_tdt)   # [S, half]
        emb = torch.cat((freqs, freqs), dim=-1)            # [S, D]
        return emb.cos().to(act_tdt), emb.sin().to(act_tdt)

    def _apply_half(xf, cos, sin):                         # cos/sin [S, D]
        c = cos.float()[None, :, None, :]
        s = sin.float()[None, :, None, :]
        h = xf.shape[-1] // 2
        x1, x2 = xf[..., :h], xf[..., h:]
        rot = torch.cat((-x2, x1), dim=-1)
        return xf * c + rot * s

    def _apply_inter(xf, cos, sin):                        # cos/sin [S, half]
        B, S, H, D = xf.shape
        h = D // 2
        c = cos.float()[None, :, None, :]
        s = sin.float()[None, :, None, :]
        xr = xf.reshape(B, S, H, h, 2)
        xe, xo = xr[..., 0], xr[..., 1]
        oe = xe * c - xo * s
        oo = xe * s + xo * c
        return torch.stack((oe, oo), dim=-1).reshape(B, S, H, D)

    # ====================================================================== #
    # FUSED RoPE on q,k (half-rotation / interleaved, optionally QK-RMSNorm'd)
    # ====================================================================== #
    if kind in ("rope_half", "rope_interleaved"):
        inter = kind == "rope_interleaved"
        apply = _apply_inter if inter else _apply_half

        def get_inputs(shape, device="cuda", seed=0):
            B, S, H, D = shape["B"], shape["S"], shape["H"], shape["D"]
            cos, sin = _rope_tables(S, D, device, inter)
            return (_randn((B, S, H, D), seed, device),
                    _randn((B, S, H, D), seed + 1, device), cos, sin)

        def ref_fn(q, k, cos, sin):
            return (apply(q.float(), cos, sin).to(act_tdt),
                    apply(k.float(), cos, sin).to(act_tdt))

        baseline_fn = ref_fn
        arity = 4

    elif kind in ("rope_half_qknorm", "rope_interleaved_qknorm"):
        inter = kind == "rope_interleaved_qknorm"
        apply = _apply_inter if inter else _apply_half

        def get_inputs(shape, device="cuda", seed=0):
            B, S, H, D = shape["B"], shape["S"], shape["H"], shape["D"]
            cos, sin = _rope_tables(S, D, device, inter)
            return (_randn((B, S, H, D), seed, device),
                    _randn((B, S, H, D), seed + 1, device),
                    _wt(D, seed + 2, device), _wt(D, seed + 3, device), cos, sin)

        def ref_fn(q, k, wq, wk, cos, sin):
            qn = _rms(q.float(), wq)
            kn = _rms(k.float(), wk)
            return (apply(qn, cos, sin).to(act_tdt),
                    apply(kn, cos, sin).to(act_tdt))

        baseline_fn = ref_fn
        arity = 6

    # ====================================================================== #
    # FUSED rotary + kv-cache write (rope applied to k, stored INTO the cache)
    # ====================================================================== #
    elif kind == "rope_kvcache":
        inter = spec["mode"] == "interleaved"
        apply = _apply_inter if inter else _apply_half

        def get_inputs(shape, device="cuda", seed=0):
            B, S, H, D = shape["B"], shape["S"], shape["H"], shape["D"]
            cos, sin = _rope_tables(S, D, device, inter)
            k = _randn((B, S, H, D), seed, device)
            cache = torch.zeros((B, S, H, D), dtype=act_tdt, device=device)
            return (k, cos, sin, cache)

        def ref_fn(k, cos, sin, cache):
            kr = apply(k.float(), cos, sin)
            cache.copy_(kr.to(cache.dtype))               # in-place cache write
            return cache

        baseline_fn = ref_fn
        arity = 4

    # ====================================================================== #
    # FUSED QKV projection (one GEMM) then split into q,k,v
    # ====================================================================== #
    elif kind == "qkv_split":
        has_bias = spec["has_bias"]

        def get_inputs(shape, device="cuda", seed=0):
            M, K, N = shape["M"], shape["K"], shape["N"]
            xs = [_randn((M, K), seed, device), _w2d((K, 3 * N), seed + 1, device, K)]
            if has_bias:
                xs.append(_randn((3 * N,), seed + 2, device, scale=0.1))
            return tuple(xs)

        def ref_fn(*a):
            x, w = a[0], a[1]
            y = x.float() @ w.float()
            if has_bias:
                y = y + a[2].float()
            n = y.shape[-1] // 3
            return (y[:, :n].to(act_tdt), y[:, n:2 * n].to(act_tdt),
                    y[:, 2 * n:3 * n].to(act_tdt))

        baseline_fn = ref_fn
        arity = 3 if has_bias else 2

    # ====================================================================== #
    # FUSED 2-GEMM gated MLP: gate=x@Wg, up=x@Wu, h=act(gate)*up, out=h@Wd
    # ====================================================================== #
    elif kind == "glu_mlp":
        act = spec["act"]

        def get_inputs(shape, device="cuda", seed=0):
            M, K, N = shape["M"], shape["K"], shape["N"]
            return (_randn((M, K), seed, device),
                    _w2d((K, N), seed + 1, device, K),
                    _w2d((K, N), seed + 2, device, K),
                    _w2d((N, K), seed + 3, device, N))

        def ref_fn(x, wg, wu, wd):
            xf = x.float()
            h = _act(act, xf @ wg.float()) * (xf @ wu.float())
            return (h @ wd.float()).to(act_tdt)

        baseline_fn = ref_fn
        arity = 4

    elif kind == "glu_mlp_gateup":
        act = spec["act"]

        def get_inputs(shape, device="cuda", seed=0):
            M, K, N = shape["M"], shape["K"], shape["N"]
            return (_randn((M, K), seed, device),
                    _w2d((K, 2 * N), seed + 1, device, K),
                    _w2d((N, K), seed + 2, device, N))

        def ref_fn(x, wgu, wd):
            gu = x.float() @ wgu.float()
            n = gu.shape[-1] // 2
            h = _act(act, gu[:, :n]) * gu[:, n:]
            return (h @ wd.float()).to(act_tdt)

        baseline_fn = ref_fn
        arity = 3

    # ====================================================================== #
    # FUSED GLU activation-only (gate * act) at large hidden
    # ====================================================================== #
    elif kind == "glu_act":
        act = spec["act"]

        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device),)

        def ref_fn(x):
            xf = x.float()
            h = xf.shape[-1] // 2
            return (_act(act, xf[:, :h]) * xf[:, h:]).to(act_tdt)

        baseline_fn = ref_fn
        arity = 1

    # ====================================================================== #
    # FUSED bias + dropout + residual-add + (RMS/Layer)Norm  (block glue)
    # returns (normed_out, new_residual)
    # ====================================================================== #
    elif kind == "bias_drop_add_norm":
        has_bias = spec["has_bias"]
        norm = spec["norm"]

        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            xs = [_randn((M, N), seed, device)]
            if has_bias:
                xs.append(_bs(N, seed + 1, device))
            xs.append(_randn((M, N), seed + 2, device))     # residual
            xs.append(_mask((M, N), seed + 3, device))       # dropout mask
            xs.append(_wt(N, seed + 4, device))              # norm weight
            if norm == "layer":
                xs.append(_bs(N, seed + 5, device))          # layernorm bias
            return tuple(xs)

        def ref_fn(*a):
            x = a[0]
            i = 1
            bias = a[i].float() if has_bias else 0.0
            i += 1 if has_bias else 0
            residual, mask, weight = a[i], a[i + 1], a[i + 2]
            i += 3
            lnbias = a[i] if norm == "layer" else None
            added = residual.float() + (x.float() + bias) * mask.float() * INV_KEEP
            out = _rms(added, weight) if norm == "rms" else _ln(added, weight, lnbias)
            return out.to(act_tdt), added.to(act_tdt)

        baseline_fn = ref_fn
        arity = 1 + (1 if has_bias else 0) + 3 + (1 if norm == "layer" else 0)

    # ====================================================================== #
    # FUSED add-residual + Norm + per-token fp8/int8 quantize-out
    # returns (q, scale, new_residual)
    # ====================================================================== #
    elif kind == "add_rmsnorm_quant":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _randn((M, N), seed + 1, device),
                    _wt(N, seed + 2, device))

        def ref_fn(x, residual, w):
            added = x.float() + residual.float()
            q, scale = _quant(_rms(added, w))
            return q, scale, added.to(act_tdt)

        baseline_fn = ref_fn
        arity = 3

    elif kind == "add_layernorm_quant":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _randn((M, N), seed + 1, device),
                    _wt(N, seed + 2, device), _bs(N, seed + 3, device))

        def ref_fn(x, residual, w, b):
            added = x.float() + residual.float()
            q, scale = _quant(_ln(added, w, b))
            return q, scale, added.to(act_tdt)

        baseline_fn = ref_fn
        arity = 4

    # ====================================================================== #
    # FUSED attention-output projection + residual-add
    # ====================================================================== #
    elif kind == "out_proj_add":
        has_bias = spec["has_bias"]

        def get_inputs(shape, device="cuda", seed=0):
            M, K, N = shape["M"], shape["K"], shape["N"]
            xs = [_randn((M, K), seed, device), _w2d((K, N), seed + 1, device, K)]
            if has_bias:
                xs.append(_bs(N, seed + 2, device))
            xs.append(_randn((M, N), seed + 3, device))       # residual
            return tuple(xs)

        def ref_fn(*a):
            attn, wo = a[0], a[1]
            y = attn.float() @ wo.float()
            if has_bias:
                y = y + a[2].float()
                residual = a[3]
            else:
                residual = a[2]
            return (y + residual.float()).to(act_tdt)

        baseline_fn = ref_fn
        arity = 4 if has_bias else 3

    # ====================================================================== #
    # FUSED embedding lookup + scale (+ positional add)
    # ====================================================================== #
    elif kind == "embed":
        pos = spec["pos"]

        def get_inputs(shape, device="cuda", seed=0):
            V, D, M = shape["V"], shape["D"], shape["M"]
            g = torch.Generator(device=device).manual_seed(seed)
            ids = torch.randint(0, V, (M,), generator=g, device=device)
            w = _randn((V, D), seed + 1, device)
            if pos:
                return (ids, w, _randn((M, D), seed + 2, device))
            return (ids, w)

        def ref_fn(*a):
            ids, w = a[0], a[1]
            y = w.float()[ids.long()] * (float(w.shape[-1]) ** 0.5)
            if pos:
                y = y + a[2].float()
            return y.to(act_tdt)

        baseline_fn = ref_fn
        arity = 3 if pos else 2

    # ====================================================================== #
    # FUSED softcap + (mask) + softmax  (attention-logits epilogue)
    # ====================================================================== #
    elif kind == "softcap_softmax":
        masked = spec["masked"]

        def get_inputs(shape, device="cuda", seed=0):
            R, Ncol = shape["R"], shape["Ncol"]
            scores = _randn((R, Ncol), seed, device, scale=8.0)
            if masked:
                g = torch.Generator(device=device).manual_seed(seed + 1)
                keep = torch.rand((R, Ncol), generator=g, device=device) > 0.3
                keep[:, 0] = True                            # never fully-mask a row
                add = torch.where(keep,
                                  torch.zeros((), dtype=torch.float32, device=device),
                                  torch.full((), -1e4, dtype=torch.float32, device=device))
                return (scores, add.to(act_tdt))
            return (scores,)

        def ref_fn(*a):
            s = SOFTCAP * torch.tanh(a[0].float() / SOFTCAP)
            if masked:
                s = s + a[1].float()
            s = s - s.amax(-1, keepdim=True)
            e = torch.exp(s)
            return (e / e.sum(-1, keepdim=True)).to(act_tdt)

        baseline_fn = ref_fn
        arity = 2 if masked else 1

    # ====================================================================== #
    # FUSED (RMS/Layer)Norm + Linear  (norm over K, then normed @ W)
    # ====================================================================== #
    elif kind == "norm_linear":
        norm = spec["norm"]

        def get_inputs(shape, device="cuda", seed=0):
            M, K, N = shape["M"], shape["K"], shape["N"]
            xs = [_randn((M, K), seed, device), _wt(K, seed + 1, device)]
            if norm == "layer":
                xs.append(_bs(K, seed + 2, device))
            xs.append(_w2d((K, N), seed + 3, device, K))
            return tuple(xs)

        def ref_fn(*a):
            x, weight = a[0], a[1]
            if norm == "layer":
                normed = _ln(x.float(), weight, a[2])
                W = a[3]
            else:
                normed = _rms(x.float(), weight)
                W = a[2]
            return (normed @ W.float()).to(act_tdt)

        baseline_fn = ref_fn
        arity = 4 if norm == "layer" else 3

    # ====================================================================== #
    # FUSED residual + dropout + scale  (stochastic depth / LayerScale)
    # ====================================================================== #
    elif kind == "resid_drop_scale":
        layerscale = spec["layerscale"]

        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            xs = [_randn((M, N), seed, device), _randn((M, N), seed + 1, device),
                  _mask((M,), seed + 2, device)]              # per-row (per-sample) mask
            if layerscale:
                xs.append(_randn((N,), seed + 3, device, scale=0.1))   # LayerScale gamma
            return tuple(xs)

        def ref_fn(*a):
            x, residual, mask = a[0], a[1], a[2]
            sc = x.float() * mask.float().unsqueeze(-1) * INV_KEEP
            if layerscale:
                sc = sc * a[3].float()
            return (residual.float() + sc).to(act_tdt)

        baseline_fn = ref_fn
        arity = 4 if layerscale else 3

    else:
        raise ValueError(f"unknown fused kind {kind!r} for op {op!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op,
          "dtype_name": dtype, "family": f"breadth_{op}",
          "mutates_input": op in FX_MUTATES_INPUT}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Naive COMPILING + correct Triton seeds (the policy's starting point). Each
# defines ``def <op>(*inputs)``; the fused math is inlined (fp32 reductions /
# a naive blocked GEMM), not a shim, so the policy has genuine code to optimize.
# --------------------------------------------------------------------------- #
_HDR = "from __future__ import annotations\nimport torch, triton, triton.language as tl\n"

# Naive blocked GEMM kernel (fp32 accumulate) embedded by every GEMM-fused seed.
_MMK = '''
@triton.jit
def _{op}_mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + (rm[:, None] * sam + rk[None, :] * sak)
    b_ptrs = b_ptr + (rk[:, None] * sbk + rn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c_ptrs = c_ptr + (rm[:, None] * scm + rn[None, :] * scn)
    tl.store(c_ptrs, acc.to({tldt}), mask=(rm[:, None] < M) & (rn[None, :] < N))
'''

# Elementwise gate kernel: h = act(gate) * up.
_GATEK = '''
@triton.jit
def _{op}_gate(g_ptr, u_ptr, h_ptr, Ntot, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < Ntot
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(h_ptr + offs, (({act_expr}) * u).to({tldt}), mask=mask)
'''

_ACT_TL = {
    "silu": "g * tl.sigmoid(g)",
    "gelu": "0.5 * g * (1.0 + tl.math.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))",
    "relu": "tl.maximum(g, 0.0)",
}

# ---- RoPE (half-rotation / interleaved) on q,k ----------------------------- #
_ROPE_HALF = _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, S, H, D, HALF, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = (row // H) % S
    base = row * D
    cb = pos * D
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    x1 = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (x1 * c - x2 * s).to({tldt}), mask=mask)
    tl.store(y_ptr + base + HALF + offs, (x2 * c + x1 * s).to({tldt}), mask=mask)


def {op}(q, k, cos, sin):
    B, S, H, D = q.shape
    HALF = D // 2
    qc, kc = q.contiguous(), k.contiguous()
    qn, kn = torch.empty_like(qc), torch.empty_like(kc)
    BLK = triton.next_power_of_2(HALF)
    grid = (B * S * H,)
    _{op}_kernel[grid](qc, cos, sin, qn, S, H, D, HALF, BLOCK=BLK, num_warps=4)
    _{op}_kernel[grid](kc, cos, sin, kn, S, H, D, HALF, BLOCK=BLK, num_warps=4)
    return qn, kn
'''

_ROPE_INTER = _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, S, H, D, HALF, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = (row // H) % S
    base = row * D
    cb = pos * HALF
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    xe = tl.load(x_ptr + base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    xo = tl.load(x_ptr + base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + 2 * offs, (xe * c - xo * s).to({tldt}), mask=mask)
    tl.store(y_ptr + base + 2 * offs + 1, (xe * s + xo * c).to({tldt}), mask=mask)


def {op}(q, k, cos, sin):
    B, S, H, D = q.shape
    HALF = D // 2
    qc, kc = q.contiguous(), k.contiguous()
    qn, kn = torch.empty_like(qc), torch.empty_like(kc)
    BLK = triton.next_power_of_2(HALF)
    grid = (B * S * H,)
    _{op}_kernel[grid](qc, cos, sin, qn, S, H, D, HALF, BLOCK=BLK, num_warps=4)
    _{op}_kernel[grid](kc, cos, sin, kn, S, H, D, HALF, BLOCK=BLK, num_warps=4)
    return qn, kn
'''

_ROPE_HALF_QK = _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, S, H, D, HALF, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = (row // H) % S
    base = row * D
    cb = pos * D
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    x1 = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)
    rstd = 1.0 / tl.sqrt(ss / D + eps)
    w1 = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    n1 = x1 * rstd * w1
    n2 = x2 * rstd * w2
    c = tl.load(cos_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (n1 * c - n2 * s).to({tldt}), mask=mask)
    tl.store(y_ptr + base + HALF + offs, (n2 * c + n1 * s).to({tldt}), mask=mask)


def {op}(q, k, wq, wk, cos, sin, eps: float = {eps}):
    B, S, H, D = q.shape
    HALF = D // 2
    qc, kc = q.contiguous(), k.contiguous()
    qn, kn = torch.empty_like(qc), torch.empty_like(kc)
    BLK = triton.next_power_of_2(HALF)
    grid = (B * S * H,)
    _{op}_kernel[grid](qc, wq, cos, sin, qn, S, H, D, HALF, eps, BLOCK=BLK, num_warps=4)
    _{op}_kernel[grid](kc, wk, cos, sin, kn, S, H, D, HALF, eps, BLOCK=BLK, num_warps=4)
    return qn, kn
'''

_ROPE_INTER_QK = _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, S, H, D, HALF, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = (row // H) % S
    base = row * D
    cb = pos * HALF
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    xe = tl.load(x_ptr + base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    xo = tl.load(x_ptr + base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(xe * xe, axis=0) + tl.sum(xo * xo, axis=0)
    rstd = 1.0 / tl.sqrt(ss / D + eps)
    we = tl.load(w_ptr + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    wo = tl.load(w_ptr + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
    ne = xe * rstd * we
    no = xo * rstd * wo
    c = tl.load(cos_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + 2 * offs, (ne * c - no * s).to({tldt}), mask=mask)
    tl.store(y_ptr + base + 2 * offs + 1, (ne * s + no * c).to({tldt}), mask=mask)


def {op}(q, k, wq, wk, cos, sin, eps: float = {eps}):
    B, S, H, D = q.shape
    HALF = D // 2
    qc, kc = q.contiguous(), k.contiguous()
    qn, kn = torch.empty_like(qc), torch.empty_like(kc)
    BLK = triton.next_power_of_2(HALF)
    grid = (B * S * H,)
    _{op}_kernel[grid](qc, wq, cos, sin, qn, S, H, D, HALF, eps, BLOCK=BLK, num_warps=4)
    _{op}_kernel[grid](kc, wk, cos, sin, kn, S, H, D, HALF, eps, BLOCK=BLK, num_warps=4)
    return qn, kn
'''

# ---- rotary + kv-cache write (rope applied to k, stored INTO the cache) ---- #
_ROPE_KV_HALF = _HDR + '''

@triton.jit
def _{op}_kernel(k_ptr, cos_ptr, sin_ptr, cache_ptr, S, H, D, HALF, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = (row // H) % S
    base = row * D
    cb = pos * D
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    x1 = tl.load(k_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(k_ptr + base + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(cache_ptr + base + offs, (x1 * c - x2 * s).to({tldt}), mask=mask)
    tl.store(cache_ptr + base + HALF + offs, (x2 * c + x1 * s).to({tldt}), mask=mask)


def {op}(k, cos, sin, cache):
    B, S, H, D = k.shape
    HALF = D // 2
    kc = k.contiguous()
    _{op}_kernel[(B * S * H,)](kc, cos, sin, cache, S, H, D, HALF,
                               BLOCK=triton.next_power_of_2(HALF), num_warps=4)
    return cache
'''

_ROPE_KV_INTER = _HDR + '''

@triton.jit
def _{op}_kernel(k_ptr, cos_ptr, sin_ptr, cache_ptr, S, H, D, HALF, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = (row // H) % S
    base = row * D
    cb = pos * HALF
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    xe = tl.load(k_ptr + base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    xo = tl.load(k_ptr + base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(cache_ptr + base + 2 * offs, (xe * c - xo * s).to({tldt}), mask=mask)
    tl.store(cache_ptr + base + 2 * offs + 1, (xe * s + xo * c).to({tldt}), mask=mask)


def {op}(k, cos, sin, cache):
    B, S, H, D = k.shape
    HALF = D // 2
    kc = k.contiguous()
    _{op}_kernel[(B * S * H,)](kc, cos, sin, cache, S, H, D, HALF,
                               BLOCK=triton.next_power_of_2(HALF), num_warps=4)
    return cache
'''

# ---- QKV projection (one GEMM) then split --------------------------------- #
_QKV = _HDR + _MMK + '''

def {op}({sig}):
    M, K = x.shape
    N3 = weight.shape[1]
    BM, BN, BK = 64, 64, 32
    c = torch.empty((M, N3), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N3, BN))
    _{op}_mm[grid](x, weight, c, M, N3, K, x.stride(0), x.stride(1), weight.stride(0), weight.stride(1), c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK)
{bias_apply}    N = N3 // 3
    return c[:, 0:N].contiguous(), c[:, N:2 * N].contiguous(), c[:, 2 * N:3 * N].contiguous()
'''

# ---- 2-GEMM gated MLP ------------------------------------------------------ #
_GLU_MLP = _HDR + _MMK + _GATEK + '''

def {op}(x, wg, wu, wd):
    M, K = x.shape
    N = wg.shape[1]
    BM, BN, BK = 64, 64, 32
    g = torch.empty((M, N), device=x.device, dtype=x.dtype)
    u = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid1 = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _{op}_mm[grid1](x, wg, g, M, N, K, x.stride(0), x.stride(1), wg.stride(0), wg.stride(1), g.stride(0), g.stride(1), BM=BM, BN=BN, BK=BK)
    _{op}_mm[grid1](x, wu, u, M, N, K, x.stride(0), x.stride(1), wu.stride(0), wu.stride(1), u.stride(0), u.stride(1), BM=BM, BN=BN, BK=BK)
    h = torch.empty((M, N), device=x.device, dtype=x.dtype)
    ntot = M * N
    _{op}_gate[(triton.cdiv(ntot, 1024),)](g, u, h, ntot, BLOCK=1024)
    out = torch.empty((M, K), device=x.device, dtype=x.dtype)
    grid2 = (triton.cdiv(M, BM), triton.cdiv(K, BN))
    _{op}_mm[grid2](h, wd, out, M, K, N, h.stride(0), h.stride(1), wd.stride(0), wd.stride(1), out.stride(0), out.stride(1), BM=BM, BN=BN, BK=BK)
    return out
'''

_GLU_MLP_GATEUP = _HDR + _MMK + _GATEK + '''

def {op}(x, wgu, wd):
    M, K = x.shape
    N2 = wgu.shape[1]
    N = N2 // 2
    BM, BN, BK = 64, 64, 32
    gu = torch.empty((M, N2), device=x.device, dtype=x.dtype)
    grid1 = (triton.cdiv(M, BM), triton.cdiv(N2, BN))
    _{op}_mm[grid1](x, wgu, gu, M, N2, K, x.stride(0), x.stride(1), wgu.stride(0), wgu.stride(1), gu.stride(0), gu.stride(1), BM=BM, BN=BN, BK=BK)
    g = gu[:, 0:N].contiguous()
    u = gu[:, N:2 * N].contiguous()
    h = torch.empty((M, N), device=x.device, dtype=x.dtype)
    ntot = M * N
    _{op}_gate[(triton.cdiv(ntot, 1024),)](g, u, h, ntot, BLOCK=1024)
    out = torch.empty((M, K), device=x.device, dtype=x.dtype)
    grid2 = (triton.cdiv(M, BM), triton.cdiv(K, BN))
    _{op}_mm[grid2](h, wd, out, M, K, N, h.stride(0), h.stride(1), wd.stride(0), wd.stride(1), out.stride(0), out.stride(1), BM=BM, BN=BN, BK=BK)
    return out
'''

# ---- GLU activation-only (gate * act) ------------------------------------- #
_GLU_ACT = _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, y_ptr, sm, sy, H, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    g = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * sm + H + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sy + offs, (({act_expr}) * u).to({tldt}), mask=mask)


def {op}(x):
    M, W = x.shape
    H = W // 2
    y = torch.empty((M, H), device=x.device, dtype=x.dtype)
    _{op}_kernel[(M,)](x, y, x.stride(0), y.stride(0), H, BLOCK=triton.next_power_of_2(H), num_warps=8)
    return y
'''

# ---- bias + dropout + residual-add + norm (block glue) -------------------- #
_BDAN = _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, {bias_ptr}res_ptr, msk_ptr, w_ptr, {lnb_ptr}y_ptr, added_ptr, sm, N, eps, inv_keep, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
{load_bias}    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(msk_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    added = r + (x{bias_term}) * d * inv_keep
    tl.store(added_ptr + base + offs, added.to({tldt}), mask=mask)
{mean_block}    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
{load_lnbias}    tl.store(y_ptr + base + offs, ({norm_expr}).to({tldt}), mask=mask)


def {op}({sig}, eps: float = {eps}, inv_keep: float = {inv_keep}):
    M, N = x.shape
    y = torch.empty_like(x)
    added = torch.empty_like(x)
    _{op}_kernel[(M,)]({call_args}, y, added, x.stride(0), N, eps, inv_keep,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y, added
'''

# ---- add-residual + norm + per-token quantize-out ------------------------- #
_ADD_RMS_QUANT = _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, res_ptr, w_ptr, q_ptr, s_ptr, added_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    added = x + r
    tl.store(added_ptr + base + offs, added.to(tl.bfloat16), mask=mask)
    var = tl.sum(added * added, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = added * rstd * w
    amax = tl.max(tl.abs(normed), axis=0)
    scale = tl.where(amax > 0.0, amax / {qmax}, 1.0)
    qv = normed / scale
    tl.store(q_ptr + base + offs, {quant_expr}, mask=mask)
    tl.store(s_ptr + row, scale)


def {op}(x, residual, weight, eps: float = {eps}):
    M, N = x.shape
    q = torch.empty((M, N), device=x.device, dtype={q_dt})
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    added = torch.empty_like(x)
    _{op}_kernel[(M,)](x, residual, weight, q, s, added, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return q, s, added
'''

_ADD_LN_QUANT = _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, res_ptr, w_ptr, b_ptr, q_ptr, s_ptr, added_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    added = x + r
    tl.store(added_ptr + base + offs, added.to(tl.bfloat16), mask=mask)
    mean = tl.sum(added, axis=0) / N
    xc = tl.where(mask, added - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = xc * rstd * w + b
    amax = tl.max(tl.abs(normed), axis=0)
    scale = tl.where(amax > 0.0, amax / {qmax}, 1.0)
    qv = normed / scale
    tl.store(q_ptr + base + offs, {quant_expr}, mask=mask)
    tl.store(s_ptr + row, scale)


def {op}(x, residual, weight, bias, eps: float = {eps}):
    M, N = x.shape
    q = torch.empty((M, N), device=x.device, dtype={q_dt})
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    added = torch.empty_like(x)
    _{op}_kernel[(M,)](x, residual, weight, bias, q, s, added, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return q, s, added
'''

# ---- attention-output projection + residual-add --------------------------- #
_OUT_PROJ = _HDR + _MMK + '''

def {op}({sig}):
    M, K = attn.shape
    N = wo.shape[1]
    BM, BN, BK = 64, 64, 32
    c = torch.empty((M, N), device=attn.device, dtype=attn.dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _{op}_mm[grid](attn, wo, c, M, N, K, attn.stride(0), attn.stride(1), wo.stride(0), wo.stride(1), c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK)
{bias_apply}    return c + residual
'''

# ---- embedding lookup + scale (+ positional add) -------------------------- #
_EMB_T = _HDR + '''

@triton.jit
def _{op}_kernel(ids_ptr, w_ptr, {pos_ptr}y_ptr, D, scale, BLOCK: tl.constexpr):
    m = tl.program_id(0)
    idx = tl.load(ids_ptr + m)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    w = tl.load(w_ptr + idx * D + offs, mask=mask, other=0.0).to(tl.float32)
    v = w * scale
{pos_add}    tl.store(y_ptr + m * D + offs, v.to({tldt}), mask=mask)


def {op}({sig}):
    M = ids.shape[0]
    D = weight.shape[1]
    scale = float(D) ** 0.5
    y = torch.empty((M, D), device=weight.device, dtype=weight.dtype)
    _{op}_kernel[(M,)]({call_args}, D, scale, BLOCK=triton.next_power_of_2(D), num_warps=4)
    return y
'''

# ---- softcap + (mask) + softmax (attention-logits epilogue) --------------- #
_SOFTCAP_T = _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, {mask_ptr}y_ptr, Ncol, cap, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < Ncol
    x = tl.load(x_ptr + row * Ncol + offs, mask=mask, other=0.0).to(tl.float32)
    s = cap * tl.math.tanh(x / cap)
{mask_add}    s = tl.where(mask, s, -1e30)
    mx = tl.max(s, axis=0)
    e = tl.exp(s - mx)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    tl.store(y_ptr + row * Ncol + offs, (e / denom).to({tldt}), mask=mask)


def {op}({sig}, cap: float = {cap}):
    R, Ncol = scores.shape
    y = torch.empty_like(scores)
    _{op}_kernel[(R,)]({call_args}, Ncol, cap, BLOCK=triton.next_power_of_2(Ncol), num_warps=8)
    return y
'''

# ---- (RMS/Layer)Norm + Linear --------------------------------------------- #
_RMS_LINEAR = _HDR + _MMK + '''

@triton.jit
def _{op}_norm(x_ptr, w_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (x * rstd * w).to({tldt}), mask=mask)


def {op}(x, weight, W, eps: float = {eps}):
    M, K = x.shape
    N = W.shape[1]
    normed = torch.empty_like(x)
    _{op}_norm[(M,)](x, weight, normed, x.stride(0), K, eps, BLOCK_N=triton.next_power_of_2(K), num_warps=8)
    BM, BN, BK = 64, 64, 32
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _{op}_mm[grid](normed, W, out, M, N, K, normed.stride(0), normed.stride(1), W.stride(0), W.stride(1), out.stride(0), out.stride(1), BM=BM, BN=BN, BK=BK)
    return out
'''

_LN_LINEAR = _HDR + _MMK + '''

@triton.jit
def _{op}_norm(x_ptr, w_ptr, b_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (xc * rstd * w + b).to({tldt}), mask=mask)


def {op}(x, weight, bias, W, eps: float = {eps}):
    M, K = x.shape
    N = W.shape[1]
    normed = torch.empty_like(x)
    _{op}_norm[(M,)](x, weight, bias, normed, x.stride(0), K, eps, BLOCK_N=triton.next_power_of_2(K), num_warps=8)
    BM, BN, BK = 64, 64, 32
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _{op}_mm[grid](normed, W, out, M, N, K, normed.stride(0), normed.stride(1), W.stride(0), W.stride(1), out.stride(0), out.stride(1), BM=BM, BN=BN, BK=BK)
    return out
'''

# ---- residual + dropout + scale (stochastic depth / LayerScale) ----------- #
_RESID_T = _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, r_ptr, msk_ptr, {g_ptr}y_ptr, sm, N, inv_keep, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(msk_ptr + row).to(tl.float32)
    sc = x * d * inv_keep
{g_apply}    tl.store(y_ptr + base + offs, (r + sc).to({tldt}), mask=mask)


def {op}({sig}, inv_keep: float = {inv_keep}):
    M, N = x.shape
    y = torch.empty_like(x)
    _{op}_kernel[(M,)]({call_args}, x.stride(0), N, inv_keep, BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
'''


def _quant_bits(dtype: str):
    """(triton store expression on ``qv``, torch output dtype literal, qmax literal)."""
    if dtype == "fp8":
        return "qv.to(tl.float8e4nv)", "torch.float8_e4m3fn", "448.0"
    expr = ("(tl.minimum(tl.maximum(qv + tl.where(qv >= 0.0, 0.5, -0.5), -127.0), "
            "127.0)).to(tl.int8)")
    return expr, "torch.int8", "127.0"


def _bdan_seed(op: str, spec: dict, tldt: str) -> str:
    has_bias = spec["has_bias"]
    norm = spec["norm"]
    bias_ptr = "b_ptr, " if has_bias else ""
    load_bias = ("    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)\n"
                 if has_bias else "")
    bias_term = " + b" if has_bias else ""
    if norm == "rms":
        lnb_ptr, load_lnbias = "", ""
        mean_block = ("    var = tl.sum(added * added, axis=0) / N\n"
                      "    rstd = 1.0 / tl.sqrt(var + eps)\n")
        norm_expr = "added * rstd * w"
    else:
        lnb_ptr = "lnb_ptr, "
        load_lnbias = "    lb = tl.load(lnb_ptr + offs, mask=mask, other=0.0).to(tl.float32)\n"
        mean_block = ("    mean = tl.sum(added, axis=0) / N\n"
                      "    xc = tl.where(mask, added - mean, 0.0)\n"
                      "    var = tl.sum(xc * xc, axis=0) / N\n"
                      "    rstd = 1.0 / tl.sqrt(var + eps)\n")
        norm_expr = "xc * rstd * w + lb"
    parts, call = ["x"], ["x"]
    if has_bias:
        parts.append("bias"); call.append("bias")
    parts += ["residual", "mask", "weight"]
    call += ["residual", "mask", "weight"]
    if norm == "layer":
        parts.append("lnbias"); call.append("lnbias")
    return _BDAN.format(op=op, tldt=tldt, eps=EPS, inv_keep=INV_KEEP, bias_ptr=bias_ptr,
                        load_bias=load_bias, bias_term=bias_term, lnb_ptr=lnb_ptr,
                        load_lnbias=load_lnbias, mean_block=mean_block, norm_expr=norm_expr,
                        sig=", ".join(parts), call_args=", ".join(call))


def seed_source(op: str, dtype: str) -> str:
    kind = KIND[op]
    spec = _SPECS[op]
    tldt = "tl.bfloat16" if kind in _QUANT_KINDS else DTYPES[dtype][1]
    eps = EPS

    if kind == "rope_half":
        return _ROPE_HALF.format(op=op, tldt=tldt)
    if kind == "rope_interleaved":
        return _ROPE_INTER.format(op=op, tldt=tldt)
    if kind == "rope_half_qknorm":
        return _ROPE_HALF_QK.format(op=op, tldt=tldt, eps=eps)
    if kind == "rope_interleaved_qknorm":
        return _ROPE_INTER_QK.format(op=op, tldt=tldt, eps=eps)
    if kind == "rope_kvcache":
        tmpl = _ROPE_KV_HALF if spec["mode"] == "half" else _ROPE_KV_INTER
        return tmpl.format(op=op, tldt=tldt)
    if kind == "qkv_split":
        if spec["has_bias"]:
            return _QKV.format(op=op, tldt=tldt, sig="x, weight, bias",
                               bias_apply="    c = c + bias.reshape(1, N3)\n")
        return _QKV.format(op=op, tldt=tldt, sig="x, weight", bias_apply="")
    if kind == "glu_mlp":
        return _GLU_MLP.format(op=op, tldt=tldt, act_expr=_ACT_TL[spec["act"]])
    if kind == "glu_mlp_gateup":
        return _GLU_MLP_GATEUP.format(op=op, tldt=tldt, act_expr=_ACT_TL[spec["act"]])
    if kind == "glu_act":
        return _GLU_ACT.format(op=op, tldt=tldt, act_expr=_ACT_TL[spec["act"]])
    if kind == "bias_drop_add_norm":
        return _bdan_seed(op, spec, tldt)
    if kind == "add_rmsnorm_quant":
        qexpr, q_dt, qmax = _quant_bits(dtype)
        return _ADD_RMS_QUANT.format(op=op, eps=eps, qmax=qmax, quant_expr=qexpr, q_dt=q_dt)
    if kind == "add_layernorm_quant":
        qexpr, q_dt, qmax = _quant_bits(dtype)
        return _ADD_LN_QUANT.format(op=op, eps=eps, qmax=qmax, quant_expr=qexpr, q_dt=q_dt)
    if kind == "out_proj_add":
        if spec["has_bias"]:
            return _OUT_PROJ.format(op=op, tldt=tldt, sig="attn, wo, bias, residual",
                                    bias_apply="    c = c + bias.reshape(1, N)\n")
        return _OUT_PROJ.format(op=op, tldt=tldt, sig="attn, wo, residual", bias_apply="")
    if kind == "embed":
        if spec["pos"]:
            return _EMB_T.format(op=op, tldt=tldt, pos_ptr="p_ptr, ",
                                 pos_add=("    p = tl.load(p_ptr + m * D + offs, mask=mask, "
                                          "other=0.0).to(tl.float32)\n    v = v + p\n"),
                                 sig="ids, weight, pos", call_args="ids, weight, pos, y")
        return _EMB_T.format(op=op, tldt=tldt, pos_ptr="", pos_add="",
                             sig="ids, weight", call_args="ids, weight, y")
    if kind == "softcap_softmax":
        if spec["masked"]:
            return _SOFTCAP_T.format(op=op, tldt=tldt, mask_ptr="am_ptr, ",
                                     mask_add=("    am = tl.load(am_ptr + row * Ncol + offs, "
                                               "mask=mask, other=0.0).to(tl.float32)\n    s = s + am\n"),
                                     sig="scores, addmask", call_args="scores, addmask, y",
                                     cap=repr(SOFTCAP))
        return _SOFTCAP_T.format(op=op, tldt=tldt, mask_ptr="", mask_add="",
                                 sig="scores", call_args="scores, y", cap=repr(SOFTCAP))
    if kind == "norm_linear":
        tmpl = _RMS_LINEAR if spec["norm"] == "rms" else _LN_LINEAR
        return tmpl.format(op=op, tldt=tldt, eps=eps)
    if kind == "resid_drop_scale":
        if spec["layerscale"]:
            return _RESID_T.format(op=op, tldt=tldt, inv_keep=INV_KEEP, g_ptr="g_ptr, ",
                                   g_apply=("    g = tl.load(g_ptr + offs, mask=mask, "
                                            "other=0.0).to(tl.float32)\n    sc = sc * g\n"),
                                   sig="x, residual, mask, gamma",
                                   call_args="x, residual, mask, gamma, y")
        return _RESID_T.format(op=op, tldt=tldt, inv_keep=INV_KEEP, g_ptr="", g_apply="",
                               sig="x, residual, mask", call_args="x, residual, mask, y")
    raise ValueError(f"unknown fused kind {kind!r} for op {op!r}")

