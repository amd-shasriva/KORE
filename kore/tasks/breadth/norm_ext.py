"""Breadth normalization-frontier task-authoring engine (torch-baselined).

Widens the KORE suite with the HARD fused-normalization kernels that are the real
transformer-/vision-block "glue" - the ops where a single-pass numerically-stable
reduction, fusion (residual add / activation gate / output quantization) and the
cross-batch backward reductions actually decide the kernel's speed and accuracy.
No trivial elementwise ops live here: every task carries genuine headroom (a
Welford/two-pass stable reduction, a fused residual/quant/gate write-back, or a
dgamma/dbeta/dx reduction across the token axis).

Contract mirrors ``kore/tasks/breadth/conv.py`` / ``kore/tasks/vendor_ops.py`` so the
shared ``_genops`` driver + generator machinery consume it unchanged:

    OPS / OP_DTYPES / SHAPES              module-level task catalog (every op name
        is prefixed ``norm_``; ~45 tasks).
    make_reference(op, dtype) -> dict     reference.py namespace (parse_shape,
        get_inputs, ref_fn EXACT fp32 oracle - casts back, may return TUPLES for
        fwd(+stats) and bwd (dx, dweight, dbias); baseline_fn torch; arity;
        entry_name=op; dtype_name; family=f"breadth_{op}"; mutates_input).
    seed_source(op, dtype) -> str         a naive, COMPILING, correct Triton seed
        (one row/group per program, fp32 reduction) defining ``def <op>(*inputs)``.

CORRECTNESS is paramount: every ``ref_fn`` computes mean/var/rms in fp32, normalizes
+ affine + fuses in fp32, and casts back to the task dtype; the backward oracles are
torch AUTOGRAD on the fp32 forward (ground truth). Every oracle is validated on CPU
against an INDEPENDENT torch computation (F.layer_norm / F.group_norm / hand-derived
analytic backward) at tight fp32 tolerance - see tests/test_norm_ext.py. torch is
imported lazily inside make_reference so registry discovery never needs a GPU.
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# --------------------------------------------------------------------------- #
# Task constants (MUST match the seed kernels)
# --------------------------------------------------------------------------- #
EPS = 1e-6              # rms/layer/group/instance/batch-norm epsilon (in var/ms)
NUM_GROUPS = 32        # GroupNorm groups (all catalog hidden dims divisible by 32)
DROP_P = 0.1           # dropout probability for the dropout-fused norms
INV_KEEP = 1.0 / (1.0 - DROP_P)   # 1/(1-p) inverted-dropout scale
L2_EPS = 1e-12         # F.normalize denominator floor (||x|| clamp)
FP8_MAX = 448.0        # OCP e4m3fn max finite (gfx950/CDNA4 native fp8) - quant clamp
INT8_MAX = 127.0       # int8 symmetric quant clamp


# --------------------------------------------------------------------------- #
# Task catalog: op -> spec (kind + optional has_bias / hidden pin).
# Every op name is prefixed ``norm_``. ~45 hard fused-normalization tasks.
# --------------------------------------------------------------------------- #
_SPECS: dict[str, dict] = {
    # -- forward core -------------------------------------------------------
    "norm_rmsnorm":            {"kind": "rmsnorm"},
    "norm_layernorm":          {"kind": "layernorm", "has_bias": True},
    "norm_layernorm_nobias":   {"kind": "layernorm", "has_bias": False},
    "norm_groupnorm":          {"kind": "groupnorm"},
    "norm_instancenorm":       {"kind": "instancenorm"},
    "norm_batchnorm":          {"kind": "batchnorm"},
    "norm_l2norm":             {"kind": "l2norm"},
    "norm_weightnorm":         {"kind": "weightnorm"},
    # -- forward + saved stats (training fwd returns mean/rstd) -------------
    "norm_rmsnorm_stats":      {"kind": "rmsnorm_stats"},
    "norm_layernorm_stats":    {"kind": "layernorm_stats"},
    "norm_groupnorm_stats":    {"kind": "groupnorm_stats"},
    "norm_batchnorm_stats":    {"kind": "batchnorm_stats"},
    "norm_groupnorm_silu":     {"kind": "groupnorm_silu"},
    # -- fused: residual add / gate / swiglu -------------------------------
    "norm_add_rmsnorm":        {"kind": "add_rmsnorm"},
    "norm_add_layernorm":      {"kind": "add_layernorm"},
    "norm_rmsnorm_swiglu":     {"kind": "rmsnorm_swiglu"},
    "norm_rmsnorm_gated":      {"kind": "rmsnorm_gated"},
    # -- fused: norm + output quantization (fp8 / int8) --------------------
    "norm_rmsnorm_quant_fp8":       {"kind": "rmsnorm_quant"},
    "norm_rmsnorm_quant_int8":      {"kind": "rmsnorm_quant"},
    "norm_layernorm_quant_fp8":     {"kind": "layernorm_quant"},
    "norm_layernorm_quant_int8":    {"kind": "layernorm_quant"},
    "norm_add_rmsnorm_quant_fp8":   {"kind": "add_rmsnorm_quant"},
    "norm_add_rmsnorm_quant_int8":  {"kind": "add_rmsnorm_quant"},
    # -- QK-norm (per-head norm on q,k) / dropout-fused --------------------
    "norm_qk_rmsnorm":         {"kind": "qk_rmsnorm"},
    "norm_qk_layernorm":       {"kind": "qk_layernorm"},
    "norm_dropout_rmsnorm":    {"kind": "dropout_rmsnorm"},
    "norm_dropout_layernorm":  {"kind": "dropout_layernorm"},
    # -- BACKWARD (dx + cross-token dweight/dbias reductions) --------------
    "norm_rmsnorm_bwd":            {"kind": "rmsnorm_bwd"},
    "norm_layernorm_bwd":          {"kind": "layernorm_bwd"},
    "norm_layernorm_nobias_bwd":   {"kind": "layernorm_nobias_bwd"},
    "norm_groupnorm_bwd":          {"kind": "groupnorm_bwd"},
    "norm_l2norm_bwd":             {"kind": "l2norm_bwd"},
    "norm_rmsnorm_bwd_h4096":      {"kind": "rmsnorm_bwd", "hidden": 4096},
    "norm_layernorm_bwd_h4096":    {"kind": "layernorm_bwd", "hidden": 4096},
    # -- hidden-size variants of the core norms (distinct kernels) ---------
    "norm_rmsnorm_h2048":      {"kind": "rmsnorm", "hidden": 2048},
    "norm_rmsnorm_h4096":      {"kind": "rmsnorm", "hidden": 4096},
    "norm_rmsnorm_h8192":      {"kind": "rmsnorm", "hidden": 8192},
    "norm_rmsnorm_h16384":     {"kind": "rmsnorm", "hidden": 16384},
    "norm_layernorm_h2048":    {"kind": "layernorm", "has_bias": True, "hidden": 2048},
    "norm_layernorm_h4096":    {"kind": "layernorm", "has_bias": True, "hidden": 4096},
    "norm_layernorm_h8192":    {"kind": "layernorm", "has_bias": True, "hidden": 8192},
    "norm_layernorm_h16384":   {"kind": "layernorm", "has_bias": True, "hidden": 16384},
    "norm_add_rmsnorm_h4096":  {"kind": "add_rmsnorm", "hidden": 4096},
    "norm_add_layernorm_h4096": {"kind": "add_layernorm", "hidden": 4096},
    "norm_rmsnorm_swiglu_h8192": {"kind": "rmsnorm_swiglu", "hidden": 8192},
}

OPS: list[str] = list(_SPECS)
KIND: dict[str, str] = {op: s["kind"] for op, s in _SPECS.items()}
HIDDEN: dict[str, int] = {op: s["hidden"] for op, s in _SPECS.items() if "hidden" in s}

# None of these ops mutate their inputs in place: the fused-residual variants RETURN
# the new residual as a fresh tensor and the quant variants return fresh (q, scale)
# tensors (so the bench loop never needs a per-call clone). Kept explicit for ABI
# parity with vendor_ops.VENDOR_MUTATES_INPUT / train_ops.TRAIN_MUTATES_INPUT.
NORM_MUTATES_INPUT: frozenset[str] = frozenset()

_QUANT_KINDS = ("rmsnorm_quant", "layernorm_quant", "add_rmsnorm_quant")


# --------------------------------------------------------------------------- #
# Shape catalog (realistic transformer/vision activation shapes).
# rows M in {4096, 16384}, hidden N in {2048, 4096, 8192, 16384} + a non-pow2 tail.
# --------------------------------------------------------------------------- #
_NORM2D = {  # x[M, N], reduce over the hidden N per row
    "minimal": {"M": 64, "N": 2048},
    "primary": {"M": 4096, "N": 8192},
    "validation": [{"M": 16384, "N": 4096}, {"M": 4096, "N": 16384},
                   {"M": 8192, "N": 8191}],   # non-pow2 hidden tail
}
_GATE2D = {  # x[M, 2*H] (SwiGLU input width even) -> [M, H]
    "minimal": {"M": 64, "N": 2048},
    "primary": {"M": 4096, "N": 8192},
    "validation": [{"M": 16384, "N": 4096}, {"M": 4096, "N": 16384},
                   {"M": 8192, "N": 8190}],   # even non-pow2 tail (H = 4095)
}
_GROUP = {  # x[M, C], GroupNorm over C//NUM_GROUPS per group (C divisible by 32)
    "minimal": {"M": 64, "N": 2048},
    "primary": {"M": 4096, "N": 8192},
    "validation": [{"M": 16384, "N": 4096}, {"M": 4096, "N": 16384},
                   {"M": 8192, "N": 6144}],   # 6144 = 32*192, non-pow2 group width
}
_CHAN = {  # x[N, C, L]: InstanceNorm reduces L per (N,C); BatchNorm reduces (N,L) per C
    "minimal": {"N": 2, "C": 32, "L": 64},
    "primary": {"N": 8, "C": 256, "L": 1024},
    "validation": [{"N": 16, "C": 128, "L": 512}, {"N": 4, "C": 512, "L": 256},
                   {"N": 8, "C": 256, "L": 1023}],   # non-pow2 spatial tail
}
_QK = {  # q,k[B, S, H, D]: per-head norm over the head-dim D per (B,S,H)
    "minimal": {"B": 1, "S": 128, "H": 8, "D": 128},
    "primary": {"B": 1, "S": 4096, "H": 32, "D": 128},   # Llama-3 8B attention
    "validation": [{"B": 2, "S": 2048, "H": 32, "D": 128},
                   {"B": 1, "S": 8192, "H": 40, "D": 128},
                   {"B": 1, "S": 4096, "H": 16, "D": 96}],   # batched, wide, non-pow2 D
}
_WN = {  # v[M, N] weight matrix (Cout=M rows), gain g[M,1]; norm over Cin=N per row
    "minimal": {"M": 64, "N": 2048},
    "primary": {"M": 4096, "N": 8192},
    "validation": [{"M": 16384, "N": 4096}, {"M": 8192, "N": 4096},
                   {"M": 4096, "N": 8191}],
}

_SHAPE_TMPL = {
    "rmsnorm": _NORM2D, "layernorm": _NORM2D, "l2norm": _NORM2D, "weightnorm": _WN,
    "groupnorm": _GROUP, "instancenorm": _CHAN, "batchnorm": _CHAN,
    "rmsnorm_stats": _NORM2D, "layernorm_stats": _NORM2D, "groupnorm_stats": _GROUP,
    "batchnorm_stats": _CHAN, "groupnorm_silu": _GROUP,
    "add_rmsnorm": _NORM2D, "add_layernorm": _NORM2D, "rmsnorm_swiglu": _GATE2D,
    "rmsnorm_gated": _NORM2D, "rmsnorm_quant": _NORM2D, "layernorm_quant": _NORM2D,
    "add_rmsnorm_quant": _NORM2D, "qk_rmsnorm": _QK, "qk_layernorm": _QK,
    "dropout_rmsnorm": _NORM2D, "dropout_layernorm": _NORM2D,
    "rmsnorm_bwd": _NORM2D, "layernorm_bwd": _NORM2D, "layernorm_nobias_bwd": _NORM2D,
    "groupnorm_bwd": _GROUP, "l2norm_bwd": _NORM2D,
}


def _pin_hidden(n: int) -> dict:
    """Hidden-pinned [M,N] catalog (N fixed = n) with a non-pow2 M tail."""
    return {"minimal": {"M": 64, "N": n}, "primary": {"M": 4096, "N": n},
            "validation": [{"M": 16384, "N": n}, {"M": 8192, "N": n},
                           {"M": 4095, "N": n}]}


def _shapes_for(op: str) -> dict:
    if op in HIDDEN:
        return _pin_hidden(HIDDEN[op])
    return _SHAPE_TMPL[KIND[op]]


SHAPES: dict[str, dict] = {op: _shapes_for(op) for op in OPS}


# --------------------------------------------------------------------------- #
# dtype sweep: bf16/fp16 default; +fp32 for a few core ops; the quant-out
# variants sweep their single quant dtype (fp8 / int8) baked into the op name.
# --------------------------------------------------------------------------- #
_FP32_OK = frozenset({"norm_rmsnorm", "norm_layernorm", "norm_layernorm_nobias",
                      "norm_l2norm", "norm_rmsnorm_bwd", "norm_layernorm_bwd"})


def op_dtypes(op: str) -> list[str]:
    """The dtype sweep for a norm op (quant override / fp32-eligible / default)."""
    if KIND[op] in _QUANT_KINDS:
        return ["fp8"] if op.endswith("_fp8") else ["int8"]
    if op in _FP32_OK:
        return ["bf16", "fp16", "fp32"]
    return ["bf16", "fp16"]


OP_DTYPES: dict[str, list[str]] = {op: op_dtypes(op) for op in OPS}


def input_dtype(op: str, dtype: str) -> str:
    """torch dtype attr name of the FLOAT activation inputs for (op, dtype).

    The quant-out variants take bf16 activations (dtype selects the fp8/int8 OUTPUT),
    so their inputs are always bfloat16; everything else uses the task dtype."""
    if KIND[op] in _QUANT_KINDS:
        return "bfloat16"
    return DTYPES[dtype][0]


# --------------------------------------------------------------------------- #
# reference.py namespace (fp32 oracle + torch baseline). torch imported lazily.
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import torch
    import torch.nn.functional as F

    spec = _SPECS[op]
    kind = spec["kind"]
    has_bias = spec.get("has_bias", True)
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

    def _wt(n, seed, device):     # affine weight ~ N(1, 0.1)
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
        return xf * r * w.float(), r

    def _ln(xf, w, b, eps=EPS):
        mean = xf.mean(-1, keepdim=True)
        var = (xf - mean).pow(2).mean(-1, keepdim=True)
        r = torch.rsqrt(var + eps)
        y = (xf - mean) * r * w.float()
        if b is not None:
            y = y + b.float()
        return y, mean, r

    def _silu(t):
        return t * torch.sigmoid(t)

    def _quant(normed):
        amax = normed.abs().amax(-1, keepdim=True)
        scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
        q = normed / scale
        if dtype == "int8":
            q = q.round().clamp(-qmax, qmax)
        return q.to(q_torch), scale.squeeze(-1).to(torch.float32)

    ns_extra = {}

    # ====================================================================== #
    # FORWARD CORE
    # ====================================================================== #
    if kind == "rmsnorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _wt(N, seed + 1, device))

        def ref_fn(x, w):
            y, _ = _rms(x.float(), w)
            return y.to(x.dtype)

        def baseline_fn(x, w):
            y, _ = _rms(x.float(), w)
            return y.to(x.dtype)

        arity = 2

    elif kind == "layernorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            xs = [_randn((M, N), seed, device), _wt(N, seed + 1, device)]
            if has_bias:
                xs.append(_bs(N, seed + 2, device))
            return tuple(xs)

        def ref_fn(*a):
            x, w = a[0], a[1]
            b = a[2] if has_bias else None
            y, _, _ = _ln(x.float(), w, b)
            return y.to(x.dtype)

        def baseline_fn(*a):
            x, w = a[0], a[1]
            b = a[2] if has_bias else None
            return F.layer_norm(x.float(), (x.shape[-1],), w.float(),
                                b.float() if b is not None else None, EPS).to(x.dtype)

        arity = 3 if has_bias else 2

    elif kind == "groupnorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, C = shape["M"], shape["N"]
            return (_randn((M, C), seed, device), _wt(C, seed + 1, device),
                    _bs(C, seed + 2, device))

        def ref_fn(x, w, b):
            M, C = x.shape
            xf = x.float().reshape(M, NUM_GROUPS, C // NUM_GROUPS)
            mean = xf.mean(-1, keepdim=True)
            var = (xf - mean).pow(2).mean(-1, keepdim=True)
            xhat = ((xf - mean) * torch.rsqrt(var + EPS)).reshape(M, C)
            return (xhat * w.float() + b.float()).to(x.dtype)

        def baseline_fn(x, w, b):
            return F.group_norm(x.float(), NUM_GROUPS, w.float(), b.float(), EPS).to(x.dtype)

        arity = 3

    elif kind == "instancenorm":
        def get_inputs(shape, device="cuda", seed=0):
            N, C, L = shape["N"], shape["C"], shape["L"]
            return (_randn((N, C, L), seed, device), _wt(C, seed + 1, device),
                    _bs(C, seed + 2, device))

        def ref_fn(x, w, b):
            xf = x.float()
            mean = xf.mean(-1, keepdim=True)
            var = (xf - mean).pow(2).mean(-1, keepdim=True)
            xhat = (xf - mean) * torch.rsqrt(var + EPS)
            return (xhat * w.float().view(1, -1, 1) + b.float().view(1, -1, 1)).to(x.dtype)

        def baseline_fn(x, w, b):
            return F.instance_norm(x.float(), weight=w.float(), bias=b.float(), eps=EPS).to(x.dtype)

        arity = 3

    elif kind == "batchnorm":
        def get_inputs(shape, device="cuda", seed=0):
            N, C, L = shape["N"], shape["C"], shape["L"]
            return (_randn((N, C, L), seed, device), _wt(C, seed + 1, device),
                    _bs(C, seed + 2, device))

        def ref_fn(x, w, b):
            xf = x.float()
            mean = xf.mean(dim=(0, 2))
            var = xf.var(dim=(0, 2), unbiased=False)
            xhat = (xf - mean.view(1, -1, 1)) * torch.rsqrt(var + EPS).view(1, -1, 1)
            return (xhat * w.float().view(1, -1, 1) + b.float().view(1, -1, 1)).to(x.dtype)

        def baseline_fn(x, w, b):
            return F.batch_norm(x.float(), None, None, w.float(), b.float(),
                                True, 0.0, EPS).to(x.dtype)

        arity = 3

    elif kind == "l2norm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device),)

        def ref_fn(x):
            xf = x.float()
            denom = xf.norm(dim=-1, keepdim=True).clamp(min=L2_EPS)
            return (xf / denom).to(x.dtype)

        def baseline_fn(x):
            return F.normalize(x.float(), p=2.0, dim=-1, eps=L2_EPS).to(x.dtype)

        arity = 1

    elif kind == "weightnorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            g = torch.Generator(device=device).manual_seed(seed + 1)
            gain = (torch.randn((M, 1), generator=g, device=device,
                                dtype=torch.float32) * 0.1 + 1.0).to(act_tdt)
            return (_randn((M, N), seed, device), gain)

        def ref_fn(v, g):
            vf = v.float()
            n = vf.norm(dim=-1, keepdim=True)
            return (g.float() * vf / n).to(v.dtype)

        baseline_fn = ref_fn
        arity = 2

    # ====================================================================== #
    # FORWARD + SAVED STATS (training forward returns mean/rstd)
    # ====================================================================== #
    elif kind == "rmsnorm_stats":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _wt(N, seed + 1, device))

        def ref_fn(x, w):
            y, r = _rms(x.float(), w)
            return y.to(x.dtype), r.squeeze(-1)

        baseline_fn = ref_fn
        arity = 2

    elif kind == "layernorm_stats":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _wt(N, seed + 1, device),
                    _bs(N, seed + 2, device))

        def ref_fn(x, w, b):
            y, mean, r = _ln(x.float(), w, b)
            return y.to(x.dtype), mean.squeeze(-1), r.squeeze(-1)

        baseline_fn = ref_fn
        arity = 3

    elif kind == "groupnorm_stats":
        def get_inputs(shape, device="cuda", seed=0):
            M, C = shape["M"], shape["N"]
            return (_randn((M, C), seed, device), _wt(C, seed + 1, device),
                    _bs(C, seed + 2, device))

        def ref_fn(x, w, b):
            M, C = x.shape
            xf = x.float().reshape(M, NUM_GROUPS, C // NUM_GROUPS)
            mean = xf.mean(-1, keepdim=True)
            var = (xf - mean).pow(2).mean(-1, keepdim=True)
            r = torch.rsqrt(var + EPS)
            xhat = ((xf - mean) * r).reshape(M, C)
            y = (xhat * w.float() + b.float()).to(x.dtype)
            return y, mean.reshape(M, NUM_GROUPS), r.reshape(M, NUM_GROUPS)

        baseline_fn = ref_fn
        arity = 3

    elif kind == "batchnorm_stats":
        def get_inputs(shape, device="cuda", seed=0):
            N, C, L = shape["N"], shape["C"], shape["L"]
            return (_randn((N, C, L), seed, device), _wt(C, seed + 1, device),
                    _bs(C, seed + 2, device))

        def ref_fn(x, w, b):
            xf = x.float()
            mean = xf.mean(dim=(0, 2))
            var = xf.var(dim=(0, 2), unbiased=False)
            r = torch.rsqrt(var + EPS)
            xhat = (xf - mean.view(1, -1, 1)) * r.view(1, -1, 1)
            y = (xhat * w.float().view(1, -1, 1) + b.float().view(1, -1, 1)).to(x.dtype)
            return y, mean, r

        baseline_fn = ref_fn
        arity = 3

    elif kind == "groupnorm_silu":
        def get_inputs(shape, device="cuda", seed=0):
            M, C = shape["M"], shape["N"]
            return (_randn((M, C), seed, device), _wt(C, seed + 1, device),
                    _bs(C, seed + 2, device))

        def ref_fn(x, w, b):
            M, C = x.shape
            xf = x.float().reshape(M, NUM_GROUPS, C // NUM_GROUPS)
            mean = xf.mean(-1, keepdim=True)
            var = (xf - mean).pow(2).mean(-1, keepdim=True)
            xhat = ((xf - mean) * torch.rsqrt(var + EPS)).reshape(M, C)
            return _silu(xhat * w.float() + b.float()).to(x.dtype)

        def baseline_fn(x, w, b):
            return F.silu(F.group_norm(x.float(), NUM_GROUPS, w.float(), b.float(), EPS)).to(x.dtype)

        arity = 3

    # ====================================================================== #
    # FUSED: residual add / gate / swiglu
    # ====================================================================== #
    elif kind == "add_rmsnorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _randn((M, N), seed + 1, device),
                    _wt(N, seed + 2, device))

        def ref_fn(x, residual, w):
            added = x.float() + residual.float()
            y, _ = _rms(added, w)
            return y.to(x.dtype), added.to(x.dtype)   # (normed, new residual)

        baseline_fn = ref_fn
        arity = 3

    elif kind == "add_layernorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _randn((M, N), seed + 1, device),
                    _wt(N, seed + 2, device), _bs(N, seed + 3, device))

        def ref_fn(x, residual, w, b):
            added = x.float() + residual.float()
            y, _, _ = _ln(added, w, b)
            return y.to(x.dtype), added.to(x.dtype)

        baseline_fn = ref_fn
        arity = 4

    elif kind == "rmsnorm_swiglu":
        def get_inputs(shape, device="cuda", seed=0):
            M, W = shape["M"], shape["N"]   # N = input width = 2*H
            return (_randn((M, W), seed, device), _wt(W, seed + 1, device))

        def ref_fn(x, w):
            normed, _ = _rms(x.float(), w)   # RMS over the full 2H row
            h = normed.shape[-1] // 2
            a, u = normed[:, :h], normed[:, h:]
            return (_silu(a) * u).to(x.dtype)

        baseline_fn = ref_fn
        arity = 2

    elif kind == "rmsnorm_gated":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _wt(N, seed + 1, device),
                    _randn((M, N), seed + 2, device))

        def ref_fn(x, w, gate):
            normed, _ = _rms(x.float(), w)
            return (normed * _silu(gate.float())).to(x.dtype)

        baseline_fn = ref_fn
        arity = 3

    # ====================================================================== #
    # FUSED: norm + output quantization (fp8 / int8), per-row (per-token) scale
    # ====================================================================== #
    elif kind == "rmsnorm_quant":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _wt(N, seed + 1, device))

        def ref_fn(x, w):
            normed, _ = _rms(x.float(), w)
            return _quant(normed)              # (q [M,N] fp8/int8, scale [M] fp32)

        baseline_fn = ref_fn
        arity = 2

    elif kind == "layernorm_quant":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _wt(N, seed + 1, device),
                    _bs(N, seed + 2, device))

        def ref_fn(x, w, b):
            normed, _, _ = _ln(x.float(), w, b)
            return _quant(normed)

        baseline_fn = ref_fn
        arity = 3

    elif kind == "add_rmsnorm_quant":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _randn((M, N), seed + 1, device),
                    _wt(N, seed + 2, device))

        def ref_fn(x, residual, w):
            added = x.float() + residual.float()
            normed, _ = _rms(added, w)
            q, scale = _quant(normed)
            return q, scale, added.to(x.dtype)   # (q, scale, new residual)

        baseline_fn = ref_fn
        arity = 3

    # ====================================================================== #
    # QK-norm (per-head norm on q,k) / dropout-fused norm
    # ====================================================================== #
    elif kind == "qk_rmsnorm":
        def get_inputs(shape, device="cuda", seed=0):
            B, S, H, D = shape["B"], shape["S"], shape["H"], shape["D"]
            return (_randn((B, S, H, D), seed, device), _randn((B, S, H, D), seed + 1, device),
                    _wt(D, seed + 2, device), _wt(D, seed + 3, device))

        def ref_fn(q, k, wq, wk):
            qn, _ = _rms(q.float(), wq)
            kn, _ = _rms(k.float(), wk)
            return qn.to(q.dtype), kn.to(k.dtype)

        baseline_fn = ref_fn
        arity = 4

    elif kind == "qk_layernorm":
        def get_inputs(shape, device="cuda", seed=0):
            B, S, H, D = shape["B"], shape["S"], shape["H"], shape["D"]
            return (_randn((B, S, H, D), seed, device), _randn((B, S, H, D), seed + 1, device),
                    _wt(D, seed + 2, device), _wt(D, seed + 3, device),
                    _bs(D, seed + 4, device), _bs(D, seed + 5, device))

        def ref_fn(q, k, wq, wk, bq, bk):
            qn, _, _ = _ln(q.float(), wq, bq)
            kn, _, _ = _ln(k.float(), wk, bk)
            return qn.to(q.dtype), kn.to(k.dtype)

        baseline_fn = ref_fn
        arity = 6

    elif kind == "dropout_rmsnorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _wt(N, seed + 1, device),
                    _mask((M, N), seed + 2, device))

        def ref_fn(x, w, mask):
            normed, _ = _rms(x.float(), w)
            return (normed * mask.float() * INV_KEEP).to(x.dtype)

        baseline_fn = ref_fn
        arity = 3

    elif kind == "dropout_layernorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _wt(N, seed + 1, device),
                    _bs(N, seed + 2, device), _mask((M, N), seed + 3, device))

        def ref_fn(x, w, b, mask):
            y, _, _ = _ln(x.float(), w, b)
            return (y * mask.float() * INV_KEEP).to(x.dtype)

        baseline_fn = ref_fn
        arity = 4

    # ====================================================================== #
    # BACKWARD: oracle = torch AUTOGRAD on the fp32 forward (ground truth).
    # dweight/dbias reduce over the token (M) axis; dx is the per-row Jacobian.
    # ====================================================================== #
    elif kind in ("rmsnorm_bwd", "layernorm_bwd", "layernorm_nobias_bwd", "groupnorm_bwd"):
        def get_inputs(shape, device="cuda", seed=0):
            if kind == "groupnorm_bwd":
                M, C = shape["M"], shape["N"]
                return (_randn((M, C), seed, device), _wt(C, seed + 1, device),
                        _randn((M, C), seed + 2, device))
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _wt(N, seed + 1, device),
                    _randn((M, N), seed + 2, device))

        def ref_fn(x, w, dy):
            xf = x.float().detach().requires_grad_(True)
            wf = w.float().detach().requires_grad_(True)
            if kind == "rmsnorm_bwd":
                r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
                y = xf * r * wf
                y.backward(dy.float())
                return xf.grad.detach(), wf.grad.detach()
            if kind == "groupnorm_bwd":
                bf = torch.zeros_like(wf).requires_grad_(True)
                y = F.group_norm(xf, NUM_GROUPS, wf, bf, EPS)
                y.backward(dy.float())
                return xf.grad.detach(), wf.grad.detach(), bf.grad.detach()
            # layernorm (with / without bias)
            N = xf.shape[-1]
            if kind == "layernorm_bwd":
                bf = torch.zeros_like(wf).requires_grad_(True)
                y = F.layer_norm(xf, (N,), wf, bf, EPS)
                y.backward(dy.float())
                return xf.grad.detach(), wf.grad.detach(), bf.grad.detach()
            y = F.layer_norm(xf, (N,), wf, None, EPS)
            y.backward(dy.float())
            return xf.grad.detach(), wf.grad.detach()

        def baseline_fn(x, w, dy):
            xf = x.detach().clone().float().requires_grad_(True)
            wf = w.detach().clone().float().requires_grad_(True)
            if kind == "rmsnorm_bwd":
                r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
                (xf * r * wf).backward(dy.float())
                return xf.grad.detach(), wf.grad.detach()
            if kind == "groupnorm_bwd":
                bf = torch.zeros_like(wf).requires_grad_(True)
                F.group_norm(xf, NUM_GROUPS, wf, bf, EPS).backward(dy.float())
                return xf.grad.detach(), wf.grad.detach(), bf.grad.detach()
            N = xf.shape[-1]
            if kind == "layernorm_bwd":
                bf = torch.zeros_like(wf).requires_grad_(True)
                F.layer_norm(xf, (N,), wf, bf, EPS).backward(dy.float())
                return xf.grad.detach(), wf.grad.detach(), bf.grad.detach()
            F.layer_norm(xf, (N,), wf, None, EPS).backward(dy.float())
            return xf.grad.detach(), wf.grad.detach()

        arity = 3

    elif kind == "l2norm_bwd":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), seed, device), _randn((M, N), seed + 1, device))

        def ref_fn(x, dy):
            xf, dyf = x.float(), dy.float()
            n = xf.norm(dim=-1, keepdim=True)
            y = xf / n
            c = (y * dyf).sum(-1, keepdim=True)
            return (dyf - y * c) / n          # dx (fp32)

        def baseline_fn(x, dy):
            xf = x.detach().clone().float().requires_grad_(True)
            F.normalize(xf, p=2.0, dim=-1, eps=L2_EPS).backward(dy.float())
            return xf.grad.detach()

        arity = 2

    else:
        raise ValueError(f"unknown norm kind {kind!r} for op {op!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op,
          "dtype_name": dtype, "family": f"breadth_{op}",
          "mutates_input": op in NORM_MUTATES_INPUT}
    ns[f"{op}_ref"] = ref_fn
    ns.update(ns_extra)
    return ns


# --------------------------------------------------------------------------- #
# Naive COMPILING + correct Triton seeds (the policy's starting point).
# One row/group per program, fp32 reduction. Each defines ``def <op>(*inputs)``.
# --------------------------------------------------------------------------- #
_HDR = "from __future__ import annotations\nimport torch, triton, triton.language as tl\n"

_RMS = '''"""GENERATED breadth {op} seed ({dtype}) - RMSNorm. One program/row: fp32
mean-square over N, rsqrt, weight, {tldt} store."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (x * rstd * w).to({tldt}), mask=mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, eps: float = {eps}) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _{op}_kernel[(M,)](x, weight, y, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
'''

_RMS_STATS = '''"""GENERATED breadth {op} seed ({dtype}) - RMSNorm returning (y, rstd)."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, y_ptr, r_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (x * rstd * w).to({tldt}), mask=mask)
    tl.store(r_ptr + row, rstd)


def {op}(x: torch.Tensor, weight: torch.Tensor, eps: float = {eps}):
    M, N = x.shape
    y = torch.empty_like(x)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)
    _{op}_kernel[(M,)](x, weight, y, rstd, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y, rstd
'''

_LN = '''"""GENERATED breadth {op} seed ({dtype}) - LayerNorm ({bias_desc}). One
program/row: fp32 mean+var, affine, {tldt} store."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, b_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
{bias_load}    out = xc * rstd * w{bias_add}
    tl.store(y_ptr + row * sm + offs, out.to({tldt}), mask=mask)


def {op}({args}, eps: float = {eps}) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _{op}_kernel[(M,)](x, weight, {bias_arg}, y, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
'''

_LN_STATS = '''"""GENERATED breadth {op} seed ({dtype}) - LayerNorm returning (y, mean, rstd)."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, b_ptr, y_ptr, m_ptr, r_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
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
    tl.store(m_ptr + row, mean)
    tl.store(r_ptr + row, rstd)


def {op}(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = {eps}):
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)
    _{op}_kernel[(M,)](x, weight, bias, y, mean, rstd, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y, mean, rstd
'''

_GN = '''"""GENERATED breadth {op} seed ({dtype}) - GroupNorm{act_desc}. One program per
(row, group): fp32 mean+var over the group width, per-channel affine, {tldt} store."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, b_ptr, y_ptr, sm, G, WD, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // G
    grp = pid % G
    t = tl.arange(0, BLOCK)
    mask = t < WD
    col = grp * WD + t
    base = row * sm
    x = tl.load(x_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / WD
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / WD
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + col, mask=mask, other=0.0).to(tl.float32)
    v = xc * rstd * w + b
    out = {act_expr}
    tl.store(y_ptr + base + col, out.to({tldt}), mask=mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = {eps}) -> torch.Tensor:
    M, C = x.shape
    G = {num_groups}
    WD = C // G
    y = torch.empty_like(x)
    _{op}_kernel[(M * G,)](x, weight, bias, y, x.stride(0), G, WD, eps,
                           BLOCK=triton.next_power_of_2(WD), num_warps=4)
    return y
'''

_GN_STATS = '''"""GENERATED breadth {op} seed ({dtype}) - GroupNorm returning (y, mean, rstd) per group."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, b_ptr, y_ptr, m_ptr, r_ptr, sm, G, WD, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // G
    grp = pid % G
    t = tl.arange(0, BLOCK)
    mask = t < WD
    col = grp * WD + t
    base = row * sm
    x = tl.load(x_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / WD
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / WD
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + col, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + col, (xc * rstd * w + b).to({tldt}), mask=mask)
    tl.store(m_ptr + pid, mean)
    tl.store(r_ptr + pid, rstd)


def {op}(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = {eps}):
    M, C = x.shape
    G = {num_groups}
    WD = C // G
    y = torch.empty_like(x)
    mean = torch.empty((M, G), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M, G), device=x.device, dtype=torch.float32)
    _{op}_kernel[(M * G,)](x, weight, bias, y, mean, rstd, x.stride(0), G, WD, eps,
                           BLOCK=triton.next_power_of_2(WD), num_warps=4)
    return y, mean, rstd
'''

_IN = '''"""GENERATED breadth {op} seed ({dtype}) - InstanceNorm. One program per (n, c):
single-pass fp32 sum & sum-of-squares over the spatial L, normalize + affine, {tldt} store."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, b_ptr, y_ptr, C, L, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    c = pid % C
    base = pid * L
    s = 0.0
    ss = 0.0
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < L
        x = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        s += tl.sum(x, axis=0)
        ss += tl.sum(x * x, axis=0)
    mean = s / L
    var = ss / L - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + c).to(tl.float32)
    b = tl.load(b_ptr + c).to(tl.float32)
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < L
        x = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + base + offs, ((x - mean) * rstd * w + b).to({tldt}), mask=m)


def {op}(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = {eps}) -> torch.Tensor:
    N, C, L = x.shape
    xc = x.contiguous()
    y = torch.empty_like(xc)
    _{op}_kernel[(N * C,)](xc, weight, bias, y, C, L, eps, BLOCK=1024, num_warps=4)
    return y
'''

_BN = '''"""GENERATED breadth {op} seed ({dtype}) - BatchNorm (train stats). Kernel 1:
per-channel fp32 mean/var reduced across (N, L) - the cross-batch reduction. Kernel 2:
normalize + affine per element. {tldt} store.{ret_desc}"""
''' + _HDR + '''

@triton.jit
def _{op}_stats_kernel(x_ptr, m_ptr, r_ptr, N, C, L, eps, BLOCK: tl.constexpr):
    c = tl.program_id(0)
    s = 0.0
    ss = 0.0
    for n in range(0, N):
        base = n * C * L + c * L
        for start in range(0, L, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            m = offs < L
            x = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            s += tl.sum(x, axis=0)
            ss += tl.sum(x * x, axis=0)
    cnt = N * L
    mean = s / cnt
    var = ss / cnt - mean * mean
    tl.store(m_ptr + c, mean)
    tl.store(r_ptr + c, 1.0 / tl.sqrt(var + eps))


@triton.jit
def _{op}_apply_kernel(x_ptr, m_ptr, r_ptr, w_ptr, b_ptr, y_ptr, C, L, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    c = pid % C
    base = pid * L
    mean = tl.load(m_ptr + c)
    rstd = tl.load(r_ptr + c)
    w = tl.load(w_ptr + c).to(tl.float32)
    b = tl.load(b_ptr + c).to(tl.float32)
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < L
        x = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + base + offs, ((x - mean) * rstd * w + b).to({tldt}), mask=m)


def {op}(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = {eps}):
    N, C, L = x.shape
    xc = x.contiguous()
    y = torch.empty_like(xc)
    mean = torch.empty((C,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((C,), device=x.device, dtype=torch.float32)
    _{op}_stats_kernel[(C,)](xc, mean, rstd, N, C, L, eps, BLOCK=1024, num_warps=4)
    _{op}_apply_kernel[(N * C,)](xc, mean, rstd, weight, bias, y, C, L, BLOCK=1024, num_warps=4)
    return {ret}
'''

_L2 = '''"""GENERATED breadth {op} seed ({dtype}) - row L2-normalize x / max(||x||, eps)."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    n = tl.sqrt(tl.sum(x * x, axis=0))
    denom = tl.maximum(n, eps)
    tl.store(y_ptr + row * sm + offs, (x / denom).to({tldt}), mask=mask)


def {op}(x: torch.Tensor, eps: float = {l2eps}) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _{op}_kernel[(M,)](x, y, x.stride(0), N, eps, BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
'''

_WN = '''"""GENERATED breadth {op} seed ({dtype}) - weight normalization g * v / ||v|| per row."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(v_ptr, g_ptr, y_ptr, sm, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    v = tl.load(v_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    n = tl.sqrt(tl.sum(v * v, axis=0))
    g = tl.load(g_ptr + row).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (v * (g / n)).to({tldt}), mask=mask)


def {op}(v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    M, N = v.shape
    y = torch.empty_like(v)
    _{op}_kernel[(M,)](v, g, y, v.stride(0), N, BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
'''

_ADD_RMS = '''"""GENERATED breadth {op} seed ({dtype}) - fused add-residual + RMSNorm. Returns
(y, added): added = x + residual is the NEW residual (fresh tensor)."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, res_ptr, w_ptr, y_ptr, added_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    added = x + r
    tl.store(added_ptr + base + offs, added.to({tldt}), mask=mask)
    var = tl.sum(added * added, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (added * rstd * w).to({tldt}), mask=mask)


def {op}(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = {eps}):
    M, N = x.shape
    y = torch.empty_like(x)
    added = torch.empty_like(x)
    _{op}_kernel[(M,)](x, residual, weight, y, added, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y, added
'''

_ADD_LN = '''"""GENERATED breadth {op} seed ({dtype}) - fused add-residual + LayerNorm. Returns
(y, added) where added = x + residual is the new residual."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, res_ptr, w_ptr, b_ptr, y_ptr, added_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    added = x + r
    tl.store(added_ptr + base + offs, added.to({tldt}), mask=mask)
    mean = tl.sum(added, axis=0) / N
    xc = tl.where(mask, added - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (xc * rstd * w + b).to({tldt}), mask=mask)


def {op}(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
         bias: torch.Tensor, eps: float = {eps}):
    M, N = x.shape
    y = torch.empty_like(x)
    added = torch.empty_like(x)
    _{op}_kernel[(M,)](x, residual, weight, bias, y, added, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y, added
'''

_SWIGLU = '''"""GENERATED breadth {op} seed ({dtype}) - RMSNorm(over 2H) then SwiGLU gate
silu(a)*b on the two halves. One program/row, fp32 reduction, {tldt} store."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, y_ptr, sxm, sym, TWOH, H, eps,
                 BLOCK2: tl.constexpr, BLOCKH: tl.constexpr):
    row = tl.program_id(0)
    offs2 = tl.arange(0, BLOCK2)
    m2 = offs2 < TWOH
    x = tl.load(x_ptr + row * sxm + offs2, mask=m2, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / TWOH
    rstd = 1.0 / tl.sqrt(var + eps)
    offh = tl.arange(0, BLOCKH)
    mh = offh < H
    a = tl.load(x_ptr + row * sxm + offh, mask=mh, other=0.0).to(tl.float32)
    wa = tl.load(w_ptr + offh, mask=mh, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + row * sxm + H + offh, mask=mh, other=0.0).to(tl.float32)
    wb = tl.load(w_ptr + H + offh, mask=mh, other=0.0).to(tl.float32)
    an = a * rstd * wa
    bn = b * rstd * wb
    out = (an * tl.sigmoid(an)) * bn
    tl.store(y_ptr + row * sym + offh, out.to({tldt}), mask=mh)


def {op}(x: torch.Tensor, weight: torch.Tensor, eps: float = {eps}) -> torch.Tensor:
    M, TWOH = x.shape
    H = TWOH // 2
    y = torch.empty((M, H), device=x.device, dtype=x.dtype)
    _{op}_kernel[(M,)](x, weight, y, x.stride(0), y.stride(0), TWOH, H, eps,
                       BLOCK2=triton.next_power_of_2(TWOH),
                       BLOCKH=triton.next_power_of_2(H), num_warps=8)
    return y
'''

_GATED = '''"""GENERATED breadth {op} seed ({dtype}) - gated RMSNorm: rmsnorm(x)*w * silu(gate)."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, g_ptr, y_ptr, sm, sg, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * sg + offs, mask=mask, other=0.0).to(tl.float32)
    out = (x * rstd * w) * (g * tl.sigmoid(g))
    tl.store(y_ptr + row * sm + offs, out.to({tldt}), mask=mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float = {eps}) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _{op}_kernel[(M,)](x, weight, gate, y, x.stride(0), gate.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
'''

_RMS_QUANT = '''"""GENERATED breadth {op} seed ({dtype}) - RMSNorm + per-row (per-token) {dtype}
output quant. Returns (q, scale): scale = amax(normed)/{qmax}; q = normed/scale."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, q_ptr, s_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = x * rstd * w
    amax = tl.max(tl.abs(normed), axis=0)
    scale = tl.where(amax > 0.0, amax / {qmax}, 1.0)
    qv = normed / scale
    tl.store(q_ptr + row * sm + offs, {quant_expr}, mask=mask)
    tl.store(s_ptr + row, scale)


def {op}(x: torch.Tensor, weight: torch.Tensor, eps: float = {eps}):
    M, N = x.shape
    q = torch.empty((M, N), device=x.device, dtype={q_dt})
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    _{op}_kernel[(M,)](x, weight, q, s, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return q, s
'''

_LN_QUANT = '''"""GENERATED breadth {op} seed ({dtype}) - LayerNorm + per-row {dtype} output quant.
Returns (q, scale)."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, b_ptr, q_ptr, s_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
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
    normed = xc * rstd * w + b
    amax = tl.max(tl.abs(normed), axis=0)
    scale = tl.where(amax > 0.0, amax / {qmax}, 1.0)
    qv = normed / scale
    tl.store(q_ptr + row * sm + offs, {quant_expr}, mask=mask)
    tl.store(s_ptr + row, scale)


def {op}(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = {eps}):
    M, N = x.shape
    q = torch.empty((M, N), device=x.device, dtype={q_dt})
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    _{op}_kernel[(M,)](x, weight, bias, q, s, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return q, s
'''

_ADD_RMS_QUANT = '''"""GENERATED breadth {op} seed ({dtype}) - fused add-residual + RMSNorm + per-row
{dtype} quant. Returns (q, scale, added) where added = x + residual (new residual)."""
''' + _HDR + '''

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


def {op}(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = {eps}):
    M, N = x.shape
    q = torch.empty((M, N), device=x.device, dtype={q_dt})
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    added = torch.empty_like(x)
    _{op}_kernel[(M,)](x, residual, weight, q, s, added, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return q, s, added
'''

_QK_RMS = '''"""GENERATED breadth {op} seed ({dtype}) - per-head RMSNorm on q, k over the
head-dim D (one program per (b,s,h) row). Returns (q_normed, k_normed)."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, y_ptr, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * D
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (x * rstd * w).to({tldt}), mask=mask)


def {op}(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, eps: float = {eps}):
    D = q.shape[-1]
    qc, kc = q.contiguous(), k.contiguous()
    rows = qc.numel() // D
    qn = torch.empty_like(qc)
    kn = torch.empty_like(kc)
    B = triton.next_power_of_2(D)
    _{op}_kernel[(rows,)](qc, wq, qn, D, eps, BLOCK=B, num_warps=4)
    _{op}_kernel[(rows,)](kc, wk, kn, D, eps, BLOCK=B, num_warps=4)
    return qn, kn
'''

_QK_LN = '''"""GENERATED breadth {op} seed ({dtype}) - per-head LayerNorm on q, k over the
head-dim D. Returns (q_normed, k_normed)."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, b_ptr, y_ptr, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * D
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (xc * rstd * w + b).to({tldt}), mask=mask)


def {op}(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor,
         bq: torch.Tensor, bk: torch.Tensor, eps: float = {eps}):
    D = q.shape[-1]
    qc, kc = q.contiguous(), k.contiguous()
    rows = qc.numel() // D
    qn = torch.empty_like(qc)
    kn = torch.empty_like(kc)
    B = triton.next_power_of_2(D)
    _{op}_kernel[(rows,)](qc, wq, bq, qn, D, eps, BLOCK=B, num_warps=4)
    _{op}_kernel[(rows,)](kc, wk, bk, kn, D, eps, BLOCK=B, num_warps=4)
    return qn, kn
'''

_DROP_RMS = '''"""GENERATED breadth {op} seed ({dtype}) - RMSNorm then inverted dropout with a
supplied deterministic mask: y = rmsnorm(x)*w * mask * {inv_keep}."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, msk_ptr, y_ptr, sm, N, eps, inv_keep, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(msk_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (x * rstd * w * d * inv_keep).to({tldt}), mask=mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor, eps: float = {eps}) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _{op}_kernel[(M,)](x, weight, mask, y, x.stride(0), N, eps, {inv_keep},
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
'''

_DROP_LN = '''"""GENERATED breadth {op} seed ({dtype}) - LayerNorm then inverted dropout with a
supplied deterministic mask: y = layernorm(x)*mask*{inv_keep}."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, b_ptr, msk_ptr, y_ptr, sm, N, eps, inv_keep, BLOCK_N: tl.constexpr):
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
    d = tl.load(msk_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, ((xc * rstd * w + b) * d * inv_keep).to({tldt}), mask=mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor,
         eps: float = {eps}) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _{op}_kernel[(M,)](x, weight, bias, mask, y, x.stride(0), N, eps, {inv_keep},
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
'''

_RMS_BWD = '''"""GENERATED breadth {op} seed ({dtype}) - RMSNorm BACKWARD. Per-row dx =
rstd*g - (rstd^3/N)*x*sum(g*x), g = dy*w; dweight = sum_rows dy*xhat via atomic add
(the cross-token reduction). Returns (dx, dweight) fp32."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, dy_ptr, dx_ptr, dw_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = dy * w
    s = tl.sum(g * x, axis=0)
    dx = rstd * g - (rstd * rstd * rstd / N) * x * s
    tl.store(dx_ptr + base + offs, dx, mask=mask)
    tl.atomic_add(dw_ptr + offs, dy * x * rstd, mask=mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor, eps: float = {eps}):
    M, N = x.shape
    dx = torch.empty((M, N), device=x.device, dtype=torch.float32)
    dw = torch.zeros((N,), device=x.device, dtype=torch.float32)
    _{op}_kernel[(M,)](x, weight, dy, dx, dw, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return dx, dw
'''

_LN_BWD = '''"""GENERATED breadth {op} seed ({dtype}) - LayerNorm BACKWARD. Per-row dx =
rstd*(g - mean(g) - xhat*mean(g*xhat)), g = dy*w; dweight = sum_rows dy*xhat and
dbias = sum_rows dy via atomic add. Returns (dx, dweight, dbias) fp32."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, dy_ptr, dx_ptr, dw_ptr, db_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xhat = xc * rstd
    g = dy * w
    sg = tl.sum(g, axis=0) / N
    sgx = tl.sum(g * xhat, axis=0) / N
    dx = rstd * (g - sg - xhat * sgx)
    tl.store(dx_ptr + base + offs, dx, mask=mask)
    tl.atomic_add(dw_ptr + offs, dy * xhat, mask=mask)
    tl.atomic_add(db_ptr + offs, dy, mask=mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor, eps: float = {eps}):
    M, N = x.shape
    dx = torch.empty((M, N), device=x.device, dtype=torch.float32)
    dw = torch.zeros((N,), device=x.device, dtype=torch.float32)
    db = torch.zeros((N,), device=x.device, dtype=torch.float32)
    _{op}_kernel[(M,)](x, weight, dy, dx, dw, db, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return dx, dw, db
'''

_LN_NOBIAS_BWD = '''"""GENERATED breadth {op} seed ({dtype}) - LayerNorm (no bias) BACKWARD.
Per-row dx = rstd*(g - mean(g) - xhat*mean(g*xhat)); dweight = sum_rows dy*xhat.
Returns (dx, dweight) fp32."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, dy_ptr, dx_ptr, dw_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xhat = xc * rstd
    g = dy * w
    sg = tl.sum(g, axis=0) / N
    sgx = tl.sum(g * xhat, axis=0) / N
    dx = rstd * (g - sg - xhat * sgx)
    tl.store(dx_ptr + base + offs, dx, mask=mask)
    tl.atomic_add(dw_ptr + offs, dy * xhat, mask=mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor, eps: float = {eps}):
    M, N = x.shape
    dx = torch.empty((M, N), device=x.device, dtype=torch.float32)
    dw = torch.zeros((N,), device=x.device, dtype=torch.float32)
    _{op}_kernel[(M,)](x, weight, dy, dx, dw, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return dx, dw
'''

_GN_BWD = '''"""GENERATED breadth {op} seed ({dtype}) - GroupNorm BACKWARD. Per (row, group)
LayerNorm-style dx over the group width; dweight = sum_rows dy*xhat and dbias =
sum_rows dy per channel via atomic add. Returns (dx, dweight, dbias) fp32."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, dy_ptr, dx_ptr, dw_ptr, db_ptr, sm, G, WD, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // G
    grp = pid % G
    t = tl.arange(0, BLOCK)
    mask = t < WD
    col = grp * WD + t
    base = row * sm
    x = tl.load(x_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / WD
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / WD
    rstd = 1.0 / tl.sqrt(var + eps)
    xhat = xc * rstd
    g = dy * w
    sg = tl.sum(g, axis=0) / WD
    sgx = tl.sum(g * xhat, axis=0) / WD
    dx = rstd * (g - sg - xhat * sgx)
    tl.store(dx_ptr + base + col, dx, mask=mask)
    tl.atomic_add(dw_ptr + col, dy * xhat, mask=mask)
    tl.atomic_add(db_ptr + col, dy, mask=mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor, eps: float = {eps}):
    M, C = x.shape
    G = {num_groups}
    WD = C // G
    dx = torch.empty((M, C), device=x.device, dtype=torch.float32)
    dw = torch.zeros((C,), device=x.device, dtype=torch.float32)
    db = torch.zeros((C,), device=x.device, dtype=torch.float32)
    _{op}_kernel[(M * G,)](x, weight, dy, dx, dw, db, x.stride(0), G, WD, eps,
                           BLOCK=triton.next_power_of_2(WD), num_warps=4)
    return dx, dw, db
'''

_L2_BWD = '''"""GENERATED breadth {op} seed ({dtype}) - L2-normalize BACKWARD. Per-row
dx = (dy - y*(y . dy)) / ||x||, y = x/||x||. Returns dx fp32."""
''' + _HDR + '''

@triton.jit
def _{op}_kernel(x_ptr, dy_ptr, dx_ptr, sm, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    n = tl.sqrt(tl.sum(x * x, axis=0))
    y = x / n
    c = tl.sum(y * dy, axis=0)
    tl.store(dx_ptr + base + offs, (dy - y * c) / n, mask=mask)


def {op}(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    dx = torch.empty((M, N), device=x.device, dtype=torch.float32)
    _{op}_kernel[(M,)](x, dy, dx, x.stride(0), N,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return dx
'''


def _quant_bits(dtype: str):
    """(triton store expression on ``qv``, torch output dtype literal, qmax literal)."""
    if dtype == "fp8":
        return "qv.to(tl.float8e4nv)", "torch.float8_e4m3fn", "448.0"
    # int8: round half away from zero (add +/-0.5 then truncate-on-cast), clamp.
    expr = ("(tl.minimum(tl.maximum(qv + tl.where(qv >= 0.0, 0.5, -0.5), -127.0), "
            "127.0)).to(tl.int8)")
    return expr, "torch.int8", "127.0"


def seed_source(op: str, dtype: str) -> str:
    kind = KIND[op]
    has_bias = _SPECS[op].get("has_bias", True)
    # quant ops keep bf16 activations; everything else stores in the task dtype
    tldt = "tl.bfloat16" if kind in _QUANT_KINDS else DTYPES[dtype][1]
    eps = EPS

    if kind == "rmsnorm":
        return _RMS.format(op=op, dtype=dtype, tldt=tldt, eps=eps)
    if kind == "rmsnorm_stats":
        return _RMS_STATS.format(op=op, dtype=dtype, tldt=tldt, eps=eps)
    if kind == "layernorm":
        if has_bias:
            args = "x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor"
            return _LN.format(op=op, dtype=dtype, tldt=tldt, eps=eps, bias_desc="with bias",
                              bias_add=" + b", args=args, bias_arg="bias",
                              bias_load="    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)\n")
        args = "x: torch.Tensor, weight: torch.Tensor"
        # no-bias: pass the weight ptr as a dummy for b_ptr (never used since bias_add empty)
        return _LN.format(op=op, dtype=dtype, tldt=tldt, eps=eps, bias_desc="no bias",
                          bias_add="", args=args, bias_arg="weight", bias_load="")
    if kind == "layernorm_stats":
        return _LN_STATS.format(op=op, dtype=dtype, tldt=tldt, eps=eps)
    if kind == "groupnorm":
        return _GN.format(op=op, dtype=dtype, tldt=tldt, eps=eps, num_groups=NUM_GROUPS,
                          act_desc="", act_expr="v")
    if kind == "groupnorm_silu":
        return _GN.format(op=op, dtype=dtype, tldt=tldt, eps=eps, num_groups=NUM_GROUPS,
                          act_desc=" + SiLU", act_expr="v * tl.sigmoid(v)")
    if kind == "groupnorm_stats":
        return _GN_STATS.format(op=op, dtype=dtype, tldt=tldt, eps=eps, num_groups=NUM_GROUPS)
    if kind == "instancenorm":
        return _IN.format(op=op, dtype=dtype, tldt=tldt, eps=eps)
    if kind == "batchnorm":
        return _BN.format(op=op, dtype=dtype, tldt=tldt, eps=eps, ret="y", ret_desc="")
    if kind == "batchnorm_stats":
        return _BN.format(op=op, dtype=dtype, tldt=tldt, eps=eps, ret="y, mean, rstd",
                          ret_desc=" Returns (y, mean, rstd).")
    if kind == "l2norm":
        return _L2.format(op=op, dtype=dtype, tldt=tldt, l2eps=L2_EPS)
    if kind == "weightnorm":
        return _WN.format(op=op, dtype=dtype, tldt=tldt)
    if kind == "add_rmsnorm":
        return _ADD_RMS.format(op=op, dtype=dtype, tldt=tldt, eps=eps)
    if kind == "add_layernorm":
        return _ADD_LN.format(op=op, dtype=dtype, tldt=tldt, eps=eps)
    if kind == "rmsnorm_swiglu":
        return _SWIGLU.format(op=op, dtype=dtype, tldt=tldt, eps=eps)
    if kind == "rmsnorm_gated":
        return _GATED.format(op=op, dtype=dtype, tldt=tldt, eps=eps)
    if kind in _QUANT_KINDS:
        qexpr, q_dt, qmax = _quant_bits(dtype)
        tmpl = {"rmsnorm_quant": _RMS_QUANT, "layernorm_quant": _LN_QUANT,
                "add_rmsnorm_quant": _ADD_RMS_QUANT}[kind]
        return tmpl.format(op=op, dtype=dtype, eps=eps, qmax=qmax, quant_expr=qexpr, q_dt=q_dt)
    if kind == "qk_rmsnorm":
        return _QK_RMS.format(op=op, dtype=dtype, tldt=tldt, eps=eps)
    if kind == "qk_layernorm":
        return _QK_LN.format(op=op, dtype=dtype, tldt=tldt, eps=eps)
    if kind == "dropout_rmsnorm":
        return _DROP_RMS.format(op=op, dtype=dtype, tldt=tldt, eps=eps, inv_keep=INV_KEEP)
    if kind == "dropout_layernorm":
        return _DROP_LN.format(op=op, dtype=dtype, tldt=tldt, eps=eps, inv_keep=INV_KEEP)
    if kind == "rmsnorm_bwd":
        return _RMS_BWD.format(op=op, dtype=dtype, eps=eps)
    if kind == "layernorm_bwd":
        return _LN_BWD.format(op=op, dtype=dtype, eps=eps)
    if kind == "layernorm_nobias_bwd":
        return _LN_NOBIAS_BWD.format(op=op, dtype=dtype, eps=eps)
    if kind == "groupnorm_bwd":
        return _GN_BWD.format(op=op, dtype=dtype, eps=eps, num_groups=NUM_GROUPS)
    if kind == "l2norm_bwd":
        return _L2_BWD.format(op=op, dtype=dtype)
    raise ValueError(f"unknown norm kind {kind!r} for op {op!r}")
