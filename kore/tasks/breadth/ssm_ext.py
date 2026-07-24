"""Breadth SEQUENCE-MODEL EXTENSION engine - the hard frontier of sub-quadratic
linear-recurrence / associative-scan kernels (torch-baselined).

This EXTENDS the ``kore.tasks.breadth.seq`` family (which already ships cumsum /
cumprod / assoc_scan_segmented / selective_scan / ssd_chunk_scan / linear_attention
/ causal_conv1d) with the *modern* linear-attention & state-space operators that
dominate today's sub-quadratic architectures - and that are among the HARDEST GPU
kernels in existence: a SEQUENTIAL recurrence (an associative scan) over the time
axis whose naive form is memory- and latency-bound, so a chunked / parallel-scan
Triton kernel has huge headroom over the torch-eager sequential baseline.

Every op here is a HARD sub-quadratic sequence kernel (no trivial ops). Config
variants (chunk size, state size, feature map, gate/decay type, head count) are
distinct op NAMES so the RL policy sweeps real modern architectures:

  * Mamba-2 SSD chunked scan (multi-head, dt-discretized scalar decay), chunk
    {64,128,256} x state N {64,128}.
  * Mamba-1 selective SSM at larger state N {64,128}.
  * Gated Linear Attention (GLA) - data-dependent per-key-dim gate, chunk variants.
  * Gated retention - data-dependent SCALAR decay, chunk variants.
  * RetNet parallel/chunkwise retention (fixed multi-scale decay) + lightning-attn.
  * RWKV-6 (data-dependent decay) & RWKV-5 (fixed decay) wkv recurrence (+ bonus).
  * (Gated) DeltaNet / delta-rule linear attention (the delta update).
  * Chunked causal linear attention with feature maps elu+1 / relu / relu^2 /
    softmax-kernel (exp), normalized & unnormalized.
  * HGRN / GILR gated linear RNNs, HGRN2 outer-product gated expansion.
  * log-cumsum-exp scan, cumulative max/min, Blelloch segmented scan, the LRU
    complex diagonal linear recurrence, and the S4D diagonal LTI SSM.
  * Mamba conv+SSM: causal depthwise conv fused with the (SSD / selective) scan.

Contract mirrors ``kore.tasks.breadth.seq`` (so the shared ``_genops`` driver +
the genv-style generator consume it unchanged):

    OPS / OP_DTYPES / SHAPES              module-level task catalog (every op is
        prefixed ``ssm_``)
    make_reference(op, dtype) -> dict     reference.py namespace (parse_shape,
        get_inputs, ref_fn EXACT fp32 recurrence oracle, baseline_fn torch-eager
        sequential recurrence, arity, entry_name, dtype_name, family=f"breadth_{op}",
        mutates_input, + time_axis / seq_input_idx layout metadata)
    seed_source(op, dtype) -> str         a naive, COMPILING, CORRECT SEQUENTIAL
        scan Triton seed (one program per (batch, channel/head), a running-state
        loop over time) - the honest starting point the policy learns to chunk.

CORRECTNESS is paramount and EXACT: every ``ref_fn`` writes the canonical math and
computes in fp32 (scans accumulate in fp32) then casts back to the task dtype. The
scans are cross-checked TWO ways in tests/test_ssm_ext.py: the sequential
recurrence oracle is asserted equal (tight fp32 tol) to a SECOND, independent
formulation - the O(L^2) decay-weighted attention-matrix / quadratic dual for
linear-attention & SSD ops, the triangular WY solve for DeltaNet, torch native
logcumsumexp / cummax / cummin for the scan primitives, the closed-form product
sum for gated scans, the complex closed form for the LRU, and the LTI convolution
kernel for S4D - so a wrong oracle is caught with certainty. torch/triton are
imported lazily (registry discovery never needs a GPU).
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# --------------------------------------------------------------------------- #
# task catalog: op -> (family, config). Every op is prefixed ``ssm_``.
# --------------------------------------------------------------------------- #
_SPECS: dict[str, tuple[str, dict]] = {
    # ---- Mamba-2 SSD (multi-head, dt-discretized scalar decay); chunk x N ----
    "ssm_mamba2_ssd_c64_n64":   ("mamba2_ssd", {"H": 8, "P": 64, "N": 64,  "chunk": 64}),
    "ssm_mamba2_ssd_c64_n128":  ("mamba2_ssd", {"H": 8, "P": 64, "N": 128, "chunk": 64}),
    "ssm_mamba2_ssd_c128_n64":  ("mamba2_ssd", {"H": 8, "P": 64, "N": 64,  "chunk": 128}),
    "ssm_mamba2_ssd_c128_n128": ("mamba2_ssd", {"H": 8, "P": 64, "N": 128, "chunk": 128}),
    "ssm_mamba2_ssd_c256_n64":  ("mamba2_ssd", {"H": 8, "P": 64, "N": 64,  "chunk": 256}),
    "ssm_mamba2_ssd_c256_n128": ("mamba2_ssd", {"H": 8, "P": 64, "N": 128, "chunk": 256}),
    # ---- Mamba-1 selective SSM at larger state N ----
    "ssm_selective_scan_n64":   ("selective", {"N": 64}),
    "ssm_selective_scan_n128":  ("selective", {"N": 128}),
    # ---- Gated Linear Attention (data-dependent per-key-dim gate) ----
    "ssm_gla_c64":  ("gla", {"H": 8, "Dh": 64, "chunk": 64}),
    "ssm_gla_c128": ("gla", {"H": 8, "Dh": 64, "chunk": 128}),
    "ssm_gla_c256": ("gla", {"H": 8, "Dh": 64, "chunk": 256}),
    # ---- Gated retention (data-dependent SCALAR decay) ----
    "ssm_gated_retention_c64":  ("gated_retention", {"H": 8, "Dh": 64, "chunk": 64}),
    "ssm_gated_retention_c128": ("gated_retention", {"H": 8, "Dh": 64, "chunk": 128}),
    # ---- RetNet retention (fixed multi-scale decay) + lightning attention ----
    "ssm_retnet_c64":        ("retention", {"H": 4, "Dh": 64, "chunk": 64,  "decay": "retnet"}),
    "ssm_retnet_c128":       ("retention", {"H": 4, "Dh": 64, "chunk": 128, "decay": "retnet"}),
    "ssm_retnet_c256":       ("retention", {"H": 4, "Dh": 64, "chunk": 256, "decay": "retnet"}),
    "ssm_retnet_multiscale": ("retention", {"H": 8, "Dh": 64, "chunk": 128, "decay": "retnet"}),
    "ssm_lightning_attn":    ("retention", {"H": 8, "Dh": 64, "chunk": 128, "decay": "lightning"}),
    # ---- RWKV wkv recurrence (num/den + bonus) ----
    "ssm_rwkv6_wkv": ("rwkv", {"data_dependent": True}),
    "ssm_rwkv5_wkv": ("rwkv", {"data_dependent": False}),
    # ---- (Gated) DeltaNet / delta rule ----
    "ssm_deltanet":       ("delta", {"gated": False, "chunk": 64,  "H": 8, "Dh": 64}),
    "ssm_gated_deltanet": ("delta", {"gated": True,  "chunk": 64,  "H": 8, "Dh": 64}),
    "ssm_deltanet_c128":  ("delta", {"gated": False, "chunk": 128, "H": 8, "Dh": 64}),
    # ---- Chunked causal linear attention (feature maps) ----
    "ssm_linattn_norm":           ("linattn", {"fmap": "elu",   "normalize": True,  "chunk": 64,  "H": 8, "Dh": 64}),
    "ssm_linattn_relu":           ("linattn", {"fmap": "relu",  "normalize": False, "chunk": 64,  "H": 8, "Dh": 64}),
    "ssm_linattn_relu2":          ("linattn", {"fmap": "relu2", "normalize": False, "chunk": 64,  "H": 8, "Dh": 64}),
    "ssm_linattn_softmax_kernel": ("linattn", {"fmap": "exp",   "normalize": True,  "chunk": 64,  "H": 8, "Dh": 64}),
    "ssm_linattn_elu_c128":       ("linattn", {"fmap": "elu",   "normalize": True,  "chunk": 128, "H": 8, "Dh": 64}),
    # ---- HGRN / GILR gated linear RNNs + HGRN2 outer-product expansion ----
    "ssm_hgrn":  ("hgrn", {}),
    "ssm_gilr":  ("gilr", {}),
    "ssm_hgrn2": ("hgrn2", {"H": 8, "Dh": 64}),
    # ---- scan primitives ----
    "ssm_logcumsumexp":  ("scan_lse", {}),
    "ssm_cummax":        ("scan_cummax", {}),
    "ssm_cummin":        ("scan_cummin", {}),
    "ssm_segmented_scan": ("segmented_scan", {}),
    "ssm_lru":           ("lru", {}),
    "ssm_s4d":           ("s4d", {"N": 64}),
    # ---- Mamba conv+SSM fused ----
    "ssm_conv_ssd_n64":   ("conv_ssd", {"N": 64,  "K": 4, "chunk": 128}),
    "ssm_conv_ssd_n128":  ("conv_ssd", {"N": 128, "K": 4, "chunk": 128}),
    "ssm_conv_selective": ("conv_selective", {"N": 64, "K": 4}),
}

OPS: tuple[str, ...] = tuple(_SPECS)
OP_FAMILY: dict[str, str] = {op: fam for op, (fam, _) in _SPECS.items()}
OP_CONFIG: dict[str, dict] = {op: dict(cfg) for op, (_, cfg) in _SPECS.items()}

# bf16/fp16 I/O sweep (the fp32 oracle casts back; scans accumulate in fp32 in
# BOTH the oracle and the seed). Materialized dict so a generator iterates directly.
DEFAULT_DTYPES: tuple[str, ...] = ("bf16", "fp16")
OP_DTYPES: dict[str, tuple[str, ...]] = {op: DEFAULT_DTYPES for op in OPS}


def op_dtypes(op: str) -> tuple[str, ...]:
    """The dtype sweep for an op (per-op override or the global default)."""
    return OP_DTYPES.get(op, DEFAULT_DTYPES)


def op_names() -> list[str]:
    return list(OPS)


# Shared hyper-params baked identically into the fp32 oracle, the seed, and the
# independent test dual (so a wrong constant is caught, not hidden).
LINATTN_EPS = 1e-6      # linear-attention normalizer floor
DELTA_EPS = 1e-6        # DeltaNet key L2-normalization floor


def retention_gamma(H: int, mode: str = "retnet") -> list[float]:
    """Per-head FIXED retention decay gamma_h in (0,1) (data-independent).

    * ``retnet``    - RetNet multi-scale decay gamma_h = 1 - 2^(-5-h).
    * ``lightning`` - Lightning-attention / TransNormer geometric decay
                      gamma_h = exp(-2^(-(h+1))).
    Pure-python (no torch) so the oracle, the seed and the test share it exactly.
    """
    import math
    if mode == "lightning":
        return [math.exp(-(2.0 ** (-(h + 1)))) for h in range(H)]
    return [1.0 - 2.0 ** (-5 - h) for h in range(H)]


# --------------------------------------------------------------------------- #
# layout metadata (per family): the time axis of the output, the index of the
# time-varying VALUE input to perturb, so the tests can assert causality
# generically (perturb the value input's FUTURE -> the past output is unchanged).
# --------------------------------------------------------------------------- #
_TIME_AXIS: dict[str, int] = {
    "mamba2_ssd": 1, "selective": 1, "gla": 2, "gated_retention": 2,
    "retention": 2, "rwkv": 1, "delta": 2, "linattn": 2, "hgrn": 2, "gilr": 2,
    "hgrn2": 2, "scan_lse": 2, "scan_cummax": 2, "scan_cummin": 2,
    "segmented_scan": 2, "lru": 2, "s4d": 2, "conv_ssd": 1, "conv_selective": 1,
}
_VALUE_INPUT_IDX: dict[str, int] = {
    "mamba2_ssd": 0, "selective": 0, "gla": 2, "gated_retention": 2,
    "retention": 2, "rwkv": 1, "delta": 2, "linattn": 2, "hgrn": 1, "gilr": 2,
    "hgrn2": 1, "scan_lse": 0, "scan_cummax": 0, "scan_cummin": 0,
    "segmented_scan": 0, "lru": 0, "s4d": 0, "conv_ssd": 0, "conv_selective": 0,
}


# --------------------------------------------------------------------------- #
# Realistic sequence-model shapes: B in {1,4}, L in {2048,4096,8192} (+ a
# non-power-of-2 tail), D in {2048,4096}, state N in {16,64,128}, heads, chunk in
# {64,128,256}. The scan axis is L. Only PARSED (round-tripped) in tests; the tiny
# numeric checks use their own CPU shapes.
# --------------------------------------------------------------------------- #
def _heads_shapes(cfg: dict) -> dict:
    H, Dh = cfg["H"], cfg["Dh"]
    ch = cfg.get("chunk")

    def mk(B, L):
        d = {"B": B, "H": H, "L": L, "Dh": Dh}
        if ch is not None:
            d["chunk"] = ch
        return d

    return {"minimal": mk(1, 256), "primary": mk(2, 2048),
            "validation": [mk(4, 4096), mk(1, 8192), mk(2, 2047)]}


def _ssd_shapes(cfg: dict) -> dict:
    H, P, N, ch = cfg["H"], cfg["P"], cfg["N"], cfg["chunk"]

    def mk(B, L):
        return {"B": B, "L": L, "H": H, "P": P, "N": N, "chunk": ch}

    return {"minimal": mk(1, 256), "primary": mk(2, 2048),
            "validation": [mk(4, 4096), mk(1, 8192), mk(2, 2047)]}


def _selective_shapes(cfg: dict) -> dict:
    N = cfg["N"]

    def mk(B, L, D):
        return {"B": B, "L": L, "D": D, "N": N}

    return {"minimal": mk(1, 256, 256), "primary": mk(2, 2048, 2048),
            "validation": [mk(4, 4096, 1024), mk(1, 4096, 2048), mk(2, 2047, 2048)]}


def _rwkv_shapes(cfg: dict) -> dict:
    def mk(B, L, C):
        return {"B": B, "L": L, "C": C}

    return {"minimal": mk(1, 256, 256), "primary": mk(2, 2048, 2048),
            "validation": [mk(4, 4096, 1024), mk(1, 8192, 2048), mk(2, 2047, 2048)]}


def _scan_shapes(cfg: dict) -> dict:  # x[B, D, L], scan over the last (time) dim
    def mk(B, D, L):
        return {"B": B, "D": D, "L": L}

    return {"minimal": mk(1, 64, 256), "primary": mk(2, 2048, 2048),
            "validation": [mk(4, 1024, 4096), mk(1, 4096, 8192), mk(2, 1536, 2047)]}


def _s4d_shapes(cfg: dict) -> dict:
    N = cfg["N"]

    def mk(B, D, L):
        return {"B": B, "D": D, "L": L, "N": N}

    return {"minimal": mk(1, 64, 256), "primary": mk(2, 2048, 2048),
            "validation": [mk(4, 1024, 4096), mk(1, 4096, 2048), mk(2, 1536, 2047)]}


def _conv_ssd_shapes(cfg: dict) -> dict:
    N, K, ch = cfg["N"], cfg["K"], cfg["chunk"]

    def mk(B, L, D):
        return {"B": B, "L": L, "D": D, "N": N, "K": K, "chunk": ch}

    return {"minimal": mk(1, 256, 256), "primary": mk(2, 2048, 2048),
            "validation": [mk(4, 4096, 1024), mk(1, 4096, 2048), mk(2, 2047, 2048)]}


def _conv_selective_shapes(cfg: dict) -> dict:
    N, K = cfg["N"], cfg["K"]

    def mk(B, L, D):
        return {"B": B, "L": L, "D": D, "N": N, "K": K}

    return {"minimal": mk(1, 256, 256), "primary": mk(2, 2048, 2048),
            "validation": [mk(4, 4096, 1024), mk(1, 4096, 2048), mk(2, 2047, 2048)]}


_SHAPE_BUILDERS = {
    "mamba2_ssd": _ssd_shapes, "selective": _selective_shapes,
    "gla": _heads_shapes, "gated_retention": _heads_shapes,
    "retention": _heads_shapes, "rwkv": _rwkv_shapes, "delta": _heads_shapes,
    "linattn": _heads_shapes, "hgrn": _scan_shapes, "gilr": _scan_shapes,
    "hgrn2": _heads_shapes, "scan_lse": _scan_shapes, "scan_cummax": _scan_shapes,
    "scan_cummin": _scan_shapes, "segmented_scan": _scan_shapes,
    "lru": _scan_shapes, "s4d": _s4d_shapes, "conv_ssd": _conv_ssd_shapes,
    "conv_selective": _conv_selective_shapes,
}

SHAPES: dict[str, dict] = {
    op: _SHAPE_BUILDERS[OP_FAMILY[op]](OP_CONFIG[op]) for op in OPS
}


# --------------------------------------------------------------------------- #
# reference.py namespace (EXACT fp32 recurrence oracle + torch-eager baseline)
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import torch
    import torch.nn.functional as F

    fam = OP_FAMILY[op]
    cfg = OP_CONFIG[op]
    tdt = getattr(torch, DTYPES[dtype][0])

    def _randn(shape, device, seed, scale=1.0):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape, generator=g, device=device,
                            dtype=torch.float32) * scale).to(tdt)

    def _rand01(shape, device, seed, bias=0.0):
        """Gate/decay fill in (0, 1) via sigmoid(randn + bias)."""
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.sigmoid(torch.randn(shape, generator=g, device=device,
                                         dtype=torch.float32) + bias).to(tdt)

    def _neg_exp(shape, device, seed, scale=0.5):
        """Strictly-negative S4D-style state matrix A = -exp(scale*randn)."""
        g = torch.Generator(device=device).manual_seed(seed)
        return (-torch.exp(torch.randn(shape, generator=g, device=device,
                                       dtype=torch.float32) * scale)).to(tdt)

    def _bern01(shape, device, seed, p=0.12):
        """Bernoulli {0,1} float fill (segment-reset flags)."""
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.rand(shape, generator=g, device=device,
                           dtype=torch.float32) < p).to(tdt)

    # ===================================================================== #
    # scan primitives
    # ===================================================================== #
    if fam == "scan_lse":
        # numerically-stable streaming log-cumsum-exp over the last dim.
        def get_inputs(shape, device="cuda", seed=0):
            B, D, L = shape["B"], shape["D"], shape["L"]
            return (_randn((B, D, L), device, seed),)

        def _core(x):
            L = x.shape[-1]
            m = torch.full(x.shape[:-1], float("-inf"), dtype=x.dtype, device=x.device)
            s = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
            out = torch.empty_like(x)
            for t in range(L):
                xt = x[..., t]
                mn = torch.maximum(m, xt)
                s = s * torch.exp(m - mn) + torch.exp(xt - mn)
                m = mn
                out[..., t] = m + torch.log(s)
            return out

        def ref_fn(x):
            return _core(x.float()).to(x.dtype)

        def baseline_fn(x):
            return _core(x)

        arity = 1

    elif fam in ("scan_cummax", "scan_cummin"):
        _redop = torch.maximum if fam == "scan_cummax" else torch.minimum

        def get_inputs(shape, device="cuda", seed=0):
            B, D, L = shape["B"], shape["D"], shape["L"]
            return (_randn((B, D, L), device, seed),)

        def _core(x):
            L = x.shape[-1]
            out = torch.empty_like(x)
            acc = x[..., 0].clone()
            out[..., 0] = acc
            for t in range(1, L):
                acc = _redop(acc, x[..., t])
                out[..., t] = acc
            return out

        def ref_fn(x):
            return _core(x.float()).to(x.dtype)

        def baseline_fn(x):
            return _core(x)

        arity = 1

    elif fam == "segmented_scan":
        # Blelloch segmented cumulative sum: reset flag r_t==1 starts a new
        # segment (h_t = x_t), else h_t = h_{t-1} + x_t. Equivalently the gated
        # scan h_t = (1 - r_t) * h_{t-1} + x_t.
        def get_inputs(shape, device="cuda", seed=0):
            B, D, L = shape["B"], shape["D"], shape["L"]
            x = _randn((B, D, L), device, seed)
            reset = _bern01((B, D, L), device, seed + 1)
            return (x, reset)

        def _core(x, reset):
            L = x.shape[-1]
            out = torch.empty_like(x)
            h = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
            for t in range(L):
                h = (1.0 - reset[..., t]) * h + x[..., t]
                out[..., t] = h
            return out

        def ref_fn(x, reset):
            return _core(x.float(), reset.float()).to(x.dtype)

        def baseline_fn(x, reset):
            return _core(x, reset)

        arity = 2

    elif fam == "lru":
        # Linear Recurrent Unit: complex diagonal linear recurrence
        # h_t = lam * h_{t-1} + b_t, lam = nu * exp(i*theta), |lam| = nu in (0,1).
        # Complex numbers are carried as a trailing size-2 (real, imag) axis.
        def get_inputs(shape, device="cuda", seed=0):
            B, D, L = shape["B"], shape["D"], shape["L"]
            x = _randn((B, D, L, 2), device, seed)
            nu = _rand01((D,), device, seed + 1, bias=1.0)     # magnitude in (0,1), biased high
            theta = _randn((D,), device, seed + 2)             # phase
            return (x, nu, theta)

        def _core(x, nu, theta):
            lr = nu * torch.cos(theta)                          # [D]
            li = nu * torch.sin(theta)
            B, D, L, _ = x.shape
            hr = torch.zeros(B, D, dtype=x.dtype, device=x.device)
            hi = torch.zeros(B, D, dtype=x.dtype, device=x.device)
            out = torch.empty_like(x)
            for t in range(L):
                br, bi = x[:, :, t, 0], x[:, :, t, 1]
                hrn = lr * hr - li * hi + br
                hin = li * hr + lr * hi + bi
                hr, hi = hrn, hin
                out[:, :, t, 0] = hr
                out[:, :, t, 1] = hi
            return out

        def ref_fn(x, nu, theta):
            return _core(x.float(), nu.float(), theta.float()).to(x.dtype)

        def baseline_fn(x, nu, theta):
            return _core(x, nu, theta)

        arity = 3

    elif fam == "s4d":
        # S4D diagonal LTI SSM (time-invariant): per channel d a size-N diagonal
        # state. Abar = dt*A (<0), Bbar = dt*B ; a = exp(Abar) in (0,1).
        #   h_t[d,n] = a[d,n] h_{t-1}[d,n] + Bbar[d,n] u_t[d]
        #   y_t[d]   = sum_n C[d,n] h_t[d,n]
        N = cfg["N"]

        def get_inputs(shape, device="cuda", seed=0):
            B, D, L = shape["B"], shape["D"], shape["L"]
            u = _randn((B, D, L), device, seed)
            Abar = _neg_exp((D, N), device, seed + 1)
            Bbar = _randn((D, N), device, seed + 2, scale=1.0 / (N ** 0.5))
            C = _randn((D, N), device, seed + 3, scale=1.0 / (N ** 0.5))
            return (u, Abar, Bbar, C)

        def _core(u, Abar, Bbar, C):
            a = torch.exp(Abar)                                 # [D,N]
            Bs, D, L = u.shape
            N_ = Abar.shape[1]
            h = torch.zeros(Bs, D, N_, dtype=u.dtype, device=u.device)
            y = torch.empty(Bs, D, L, dtype=u.dtype, device=u.device)
            for t in range(L):
                h = a * h + Bbar * u[:, :, t, None]
                y[:, :, t] = (C * h).sum(-1)
            return y

        def ref_fn(u, Abar, Bbar, C):
            return _core(u.float(), Abar.float(), Bbar.float(), C.float()).to(u.dtype)

        def baseline_fn(u, Abar, Bbar, C):
            return _core(u, Abar, Bbar, C)

        arity = 4

    # ===================================================================== #
    # matrix-state linear attention: GLA / gated-retention / retention /
    # feature-map linear attention / HGRN2 / DeltaNet
    # ===================================================================== #
    elif fam == "gla":
        # Gated Linear Attention: data-dependent gate on the KEY (feature) dim.
        # q/k/v[B,H,L,Dh], gate logits gl[B,H,L,Dh]; alpha=sigmoid(gl) in (0,1).
        #   S_t[i,j] = alpha_t[i] S_{t-1}[i,j] + k_t[i] v_t[j] ; y_t[j]=sum_i q_t[i] S_t[i,j]
        def get_inputs(shape, device="cuda", seed=0):
            B, H, L, Dh = shape["B"], shape["H"], shape["L"], shape["Dh"]
            q = _randn((B, H, L, Dh), device, seed, scale=0.5)
            k = _randn((B, H, L, Dh), device, seed + 1, scale=0.5)
            v = _randn((B, H, L, Dh), device, seed + 2)
            gl = _randn((B, H, L, Dh), device, seed + 3)
            return (q, k, v, gl)

        def _core(q, k, v, gl):
            Bs, H, L, Dk = q.shape
            alpha = torch.sigmoid(gl)
            S = torch.zeros(Bs, H, Dk, v.shape[-1], dtype=q.dtype, device=q.device)
            y = torch.empty_like(v)
            for t in range(L):
                S = alpha[:, :, t, :, None] * S + k[:, :, t, :, None] * v[:, :, t, None, :]
                y[:, :, t] = (q[:, :, t, :, None] * S).sum(-2)
            return y

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 4

    elif fam == "gated_retention":
        # Data-dependent SCALAR decay per (b,h,t): a_t=sigmoid(gl[b,h,t]).
        #   S_t = a_t S_{t-1} + k_t (outer) v_t ; y_t = q_t^T S_t
        def get_inputs(shape, device="cuda", seed=0):
            B, H, L, Dh = shape["B"], shape["H"], shape["L"], shape["Dh"]
            q = _randn((B, H, L, Dh), device, seed, scale=0.5)
            k = _randn((B, H, L, Dh), device, seed + 1, scale=0.5)
            v = _randn((B, H, L, Dh), device, seed + 2)
            gl = _randn((B, H, L), device, seed + 3)
            return (q, k, v, gl)

        def _core(q, k, v, gl):
            Bs, H, L, Dk = q.shape
            a = torch.sigmoid(gl)                              # [B,H,L]
            S = torch.zeros(Bs, H, Dk, v.shape[-1], dtype=q.dtype, device=q.device)
            y = torch.empty_like(v)
            for t in range(L):
                S = a[:, :, t, None, None] * S + k[:, :, t, :, None] * v[:, :, t, None, :]
                y[:, :, t] = (q[:, :, t, :, None] * S).sum(-2)
            return y

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 4

    elif fam == "retention":
        # RetNet / lightning retention: FIXED per-head decay gamma_h.
        #   S_t = gamma_h S_{t-1} + k_t (outer) v_t ; y_t = q_t^T S_t
        H = cfg["H"]
        mode = cfg["decay"]
        gamma_list = retention_gamma(H, mode)

        def get_inputs(shape, device="cuda", seed=0):
            B, Hh, L, Dh = shape["B"], shape["H"], shape["L"], shape["Dh"]
            q = _randn((B, Hh, L, Dh), device, seed, scale=0.5)
            k = _randn((B, Hh, L, Dh), device, seed + 1, scale=0.5)
            v = _randn((B, Hh, L, Dh), device, seed + 2)
            return (q, k, v)

        def _core(q, k, v):
            Bs, Hh, L, Dk = q.shape
            gamma = torch.tensor(gamma_list, dtype=q.dtype, device=q.device)
            S = torch.zeros(Bs, Hh, Dk, v.shape[-1], dtype=q.dtype, device=q.device)
            y = torch.empty_like(v)
            for t in range(L):
                S = gamma[None, :, None, None] * S + k[:, :, t, :, None] * v[:, :, t, None, :]
                y[:, :, t] = (q[:, :, t, :, None] * S).sum(-2)
            return y

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 3

    elif fam == "linattn":
        # Causal linear attention with a feature map phi and optional denominator
        # normalization. S_t = S_{t-1}+phi(k_t)(outer)v_t; z_t = z_{t-1}+phi(k_t);
        # num_t = phi(q_t)^T S_t ; y_t = num_t / (phi(q_t).z_t + eps) [normalized].
        fmap = cfg["fmap"]
        normalize = cfg["normalize"]
        _sc = 0.25 if fmap == "exp" else 0.5

        def _phi(x):
            if fmap == "elu":
                return F.elu(x) + 1.0
            if fmap == "relu":
                return F.relu(x)
            if fmap == "relu2":
                return F.relu(x) ** 2
            return torch.exp(x)                                # softmax kernel

        def get_inputs(shape, device="cuda", seed=0):
            B, H, L, Dh = shape["B"], shape["H"], shape["L"], shape["Dh"]
            q = _randn((B, H, L, Dh), device, seed, scale=_sc)
            k = _randn((B, H, L, Dh), device, seed + 1, scale=_sc)
            v = _randn((B, H, L, Dh), device, seed + 2)
            return (q, k, v)

        def _core(q, k, v):
            Bs, H, L, Dk = q.shape
            pq, pk = _phi(q), _phi(k)
            S = torch.zeros(Bs, H, Dk, v.shape[-1], dtype=q.dtype, device=q.device)
            z = torch.zeros(Bs, H, Dk, dtype=q.dtype, device=q.device)
            y = torch.empty_like(v)
            for t in range(L):
                S = S + pk[:, :, t, :, None] * v[:, :, t, None, :]
                num = (pq[:, :, t, :, None] * S).sum(-2)       # [B,H,Dv]
                if normalize:
                    z = z + pk[:, :, t]
                    den = (pq[:, :, t] * z).sum(-1, keepdim=True) + LINATTN_EPS
                    y[:, :, t] = num / den
                else:
                    y[:, :, t] = num
            return y

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 3

    elif fam == "hgrn2":
        # HGRN2 outer-product gated expansion: the gate is TIED to the input
        # (key = 1 - alpha). q/v[B,H,L,Dh], gate logits gl[B,H,L,Dh].
        #   S_t[i,j] = alpha_t[i] S_{t-1}[i,j] + (1-alpha_t[i]) v_t[j]
        #   y_t[j]   = sum_i q_t[i] S_t[i,j]
        def get_inputs(shape, device="cuda", seed=0):
            B, H, L, Dh = shape["B"], shape["H"], shape["L"], shape["Dh"]
            q = _randn((B, H, L, Dh), device, seed, scale=0.5)
            v = _randn((B, H, L, Dh), device, seed + 1)
            gl = _randn((B, H, L, Dh), device, seed + 2)
            return (q, v, gl)

        def _core(q, v, gl):
            Bs, H, L, Dk = q.shape
            alpha = torch.sigmoid(gl)
            S = torch.zeros(Bs, H, Dk, v.shape[-1], dtype=q.dtype, device=q.device)
            y = torch.empty_like(v)
            for t in range(L):
                S = (alpha[:, :, t, :, None] * S
                     + (1.0 - alpha[:, :, t])[:, :, :, None] * v[:, :, t, None, :])
                y[:, :, t] = (q[:, :, t, :, None] * S).sum(-2)
            return y

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 3

    elif fam == "delta":
        # (Gated) DeltaNet delta-rule linear attention. k is L2-normalized.
        #   S'      = alpha_t S_{t-1}        (gated: scalar decay; else identity)
        #   pred_t  = S'^T k_t               (readout of the decayed state)
        #   S_t     = S' + beta_t k_t (outer)(v_t - pred_t)   (the delta update)
        #   y_t     = q_t^T S_t
        gated = cfg["gated"]

        def get_inputs(shape, device="cuda", seed=0):
            B, H, L, Dh = shape["B"], shape["H"], shape["L"], shape["Dh"]
            q = _randn((B, H, L, Dh), device, seed, scale=0.5)
            k = _randn((B, H, L, Dh), device, seed + 1, scale=0.5)
            v = _randn((B, H, L, Dh), device, seed + 2)
            beta = _randn((B, H, L), device, seed + 3)         # -> sigmoid write strength
            if gated:
                al = _randn((B, H, L), device, seed + 4) + 2.0  # -> sigmoid ~ 0.85-0.98
                return (q, k, v, al, beta)
            return (q, k, v, beta)

        def _core(*xs):
            if gated:
                q, k, v, al, be = xs
                alpha = torch.sigmoid(al)
            else:
                q, k, v, be = xs
                alpha = None
            beta = torch.sigmoid(be)
            k = k / (k.norm(dim=-1, keepdim=True) + DELTA_EPS)
            Bs, H, L, Dk = q.shape
            S = torch.zeros(Bs, H, Dk, v.shape[-1], dtype=q.dtype, device=q.device)
            y = torch.empty_like(v)
            for t in range(L):
                if alpha is not None:
                    S = alpha[:, :, t, None, None] * S
                pred = (k[:, :, t, :, None] * S).sum(-2)       # [B,H,Dv]
                S = S + (beta[:, :, t, None, None] * k[:, :, t, :, None]
                         * (v[:, :, t] - pred)[:, :, None, :])
                y[:, :, t] = (q[:, :, t, :, None] * S).sum(-2)
            return y

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 5 if gated else 4

    # ===================================================================== #
    # state-[N] recurrences: Mamba-2 SSD, Mamba-1 selective, conv+SSM fusions
    # ===================================================================== #
    elif fam == "mamba2_ssd":
        # Mamba-2 SSD (state-space duality), multi-head, dt-discretized SCALAR
        # decay per head. Layout x[B,L,H,P], dt[B,L,H] (raw), A[H] (<0),
        # B_/C[B,L,H,N] -> y[B,L,H,P].
        #   dt_t = softplus(dt) ; a_t = exp(dt_t * A_h)   (scalar decay per head)
        #   g_t[p,n] = a_t g_{t-1}[p,n] + (dt_t x_t[p]) B_t[n]   (rank-1 input, dt-scaled)
        #   y_t[p]   = sum_n C_t[n] g_t[p,n]
        H, P, N = cfg["H"], cfg["P"], cfg["N"]

        def get_inputs(shape, device="cuda", seed=0):
            B, L = shape["B"], shape["L"]
            Hh, Pp, Nn = shape["H"], shape["P"], shape["N"]
            x = _randn((B, L, Hh, Pp), device, seed)
            dt = _randn((B, L, Hh), device, seed + 1, scale=0.5)
            A = _neg_exp((Hh,), device, seed + 2)
            B_ = _randn((B, L, Hh, Nn), device, seed + 3)
            C = _randn((B, L, Hh, Nn), device, seed + 4)
            return (x, dt, A, B_, C)

        def _core(x, dt, A, B_, C):
            Bs, L, Hh, Pp = x.shape
            Nn = B_.shape[-1]
            dts = F.softplus(dt)                                 # [B,L,H]
            a = torch.exp(dts * A)                               # [B,L,H]
            h = torch.zeros(Bs, Hh, Pp, Nn, dtype=x.dtype, device=x.device)
            y = torch.empty(Bs, L, Hh, Pp, dtype=x.dtype, device=x.device)
            for t in range(L):
                dtx = dts[:, t, :, None] * x[:, t]              # [B,H,P]
                upd = dtx[:, :, :, None] * B_[:, t, :, None, :]  # [B,H,P,N]
                h = a[:, t, :, None, None] * h + upd
                y[:, t] = (h * C[:, t, :, None, :]).sum(-1)      # [B,H,P]
            return y

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 5

    elif fam == "selective":
        # Mamba-1 selective SSM core at larger state N. u/delta[B,L,D], A[D,N],
        # B_/C[B,L,N], D_[D] -> y[B,L,D] (delta_softplus=True, with the D skip).
        N = cfg["N"]

        def get_inputs(shape, device="cuda", seed=0):
            B, L, D = shape["B"], shape["L"], shape["D"]
            Nn = shape["N"]
            u = _randn((B, L, D), device, seed)
            delta = _randn((B, L, D), device, seed + 1, scale=0.5)
            A = _neg_exp((D, Nn), device, seed + 2)
            B_ = _randn((B, L, Nn), device, seed + 3)
            C = _randn((B, L, Nn), device, seed + 4)
            D_ = _randn((D,), device, seed + 5)
            return (u, delta, A, B_, C, D_)

        def _core(u, delta, A, B_, C, D_):
            Bs, L, D = u.shape
            dt = F.softplus(delta)                              # [B,L,D]
            h = torch.zeros((Bs, D, A.shape[1]), dtype=u.dtype, device=u.device)
            y = torch.empty((Bs, L, D), dtype=u.dtype, device=u.device)
            for t in range(L):
                dt_t = dt[:, t]                                 # [B,D]
                dA = torch.exp(dt_t[:, :, None] * A)            # [B,D,N]
                dBu = dt_t[:, :, None] * B_[:, t, None, :] * u[:, t, :, None]
                h = dA * h + dBu
                y[:, t] = (h * C[:, t, None, :]).sum(-1)        # [B,D]
            return y + u * D_

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 6

    elif fam == "conv_ssd":
        # Mamba conv+SSM: causal depthwise conv (+ SiLU) fused into a scalar-decay
        # SSD scan. x[B,L,D], conv_w[D,K], a[B,L] (decay logit), B_/C[B,L,N] ->
        # y[B,L,D]. xc = silu(causal_depthwise_conv(x)) ; then the SSD recurrence
        #   h_t = sigmoid(a_t) h_{t-1} + xc_t (outer) B_t ; y_t = sum_n C_t h_t.
        N, K = cfg["N"], cfg["K"]

        def get_inputs(shape, device="cuda", seed=0):
            B, L, D = shape["B"], shape["L"], shape["D"]
            Nn, Kk = shape["N"], shape["K"]
            x = _randn((B, L, D), device, seed)
            conv_w = _randn((D, Kk), device, seed + 1, scale=1.0 / (Kk ** 0.5))
            a = _randn((B, L), device, seed + 2, scale=0.5)     # decay logit -> sigmoid
            B_ = _randn((B, L, Nn), device, seed + 3)
            C = _randn((B, L, Nn), device, seed + 4)
            return (x, conv_w, a, B_, C)

        def _core(x, conv_w, a, B_, C):
            Bs, L, D = x.shape
            Nn, Kk = B_.shape[-1], conv_w.shape[1]
            xt = x.transpose(1, 2)                              # [B,D,L]
            xc = F.conv1d(F.pad(xt, (Kk - 1, 0)), conv_w[:, None, :], None, groups=D)
            xc = F.silu(xc).transpose(1, 2)                     # [B,L,D]
            dec = torch.sigmoid(a)                              # [B,L]
            h = torch.zeros(Bs, D, Nn, dtype=x.dtype, device=x.device)
            y = torch.empty(Bs, L, D, dtype=x.dtype, device=x.device)
            for t in range(L):
                h = dec[:, t, None, None] * h + xc[:, t, :, None] * B_[:, t, None, :]
                y[:, t] = (h * C[:, t, None, :]).sum(-1)
            return y

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 5

    elif fam == "conv_selective":
        # Mamba conv+SSM: causal depthwise conv (+ SiLU) fused into the Mamba-1
        # selective scan. u[B,L,D], conv_w[D,K], delta[B,L,D], A[D,N], B_/C[B,L,N]
        # -> y[B,L,D]. uc = silu(causal_conv(u)); dt=softplus(delta);
        #   h_t = exp(dt_t A) h_{t-1} + (dt_t B_t) uc_t ; y_t = sum_n C_t h_t.
        N, K = cfg["N"], cfg["K"]

        def get_inputs(shape, device="cuda", seed=0):
            B, L, D = shape["B"], shape["L"], shape["D"]
            Nn, Kk = shape["N"], shape["K"]
            u = _randn((B, L, D), device, seed)
            conv_w = _randn((D, Kk), device, seed + 1, scale=1.0 / (Kk ** 0.5))
            delta = _randn((B, L, D), device, seed + 2, scale=0.5)
            A = _neg_exp((D, Nn), device, seed + 3)
            B_ = _randn((B, L, Nn), device, seed + 4)
            C = _randn((B, L, Nn), device, seed + 5)
            return (u, conv_w, delta, A, B_, C)

        def _core(u, conv_w, delta, A, B_, C):
            Bs, L, D = u.shape
            Nn, Kk = A.shape[1], conv_w.shape[1]
            ut = u.transpose(1, 2)
            uc = F.conv1d(F.pad(ut, (Kk - 1, 0)), conv_w[:, None, :], None, groups=D)
            uc = F.silu(uc).transpose(1, 2)                    # [B,L,D]
            dt = F.softplus(delta)
            h = torch.zeros(Bs, D, Nn, dtype=u.dtype, device=u.device)
            y = torch.empty(Bs, L, D, dtype=u.dtype, device=u.device)
            for t in range(L):
                dt_t = dt[:, t]
                dA = torch.exp(dt_t[:, :, None] * A)
                dBu = dt_t[:, :, None] * B_[:, t, None, :] * uc[:, t, :, None]
                h = dA * h + dBu
                y[:, t] = (h * C[:, t, None, :]).sum(-1)
            return y

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 6

    # ===================================================================== #
    # elementwise gated recurrences: RWKV wkv, HGRN, GILR
    # ===================================================================== #
    elif fam == "rwkv":
        # RWKV-5/6 channel-wise wkv with a bonus. k/v[B,L,C]; the recurrence keeps
        # a numerator S and denominator Z per channel:
        #   wkv_t = (S_{t-1} + exp(u+k_t) v_t) / (Z_{t-1} + exp(u+k_t))
        #   S_t = exp(-w_t) S_{t-1} + exp(k_t) v_t ; Z_t = exp(-w_t) Z_{t-1} + exp(k_t)
        # RWKV-6: data-dependent decay w_t=softplus(wl[b,t]); RWKV-5: fixed w=softplus(w[c]).
        data_dep = cfg["data_dependent"]

        def get_inputs(shape, device="cuda", seed=0):
            B, L, C = shape["B"], shape["L"], shape["C"]
            k = _randn((B, L, C), device, seed, scale=0.5)
            v = _randn((B, L, C), device, seed + 1)
            if data_dep:
                w = _randn((B, L, C), device, seed + 2, scale=0.5)     # decay logits
            else:
                w = _randn((C,), device, seed + 2, scale=0.5)          # fixed decay logit
            u = _randn((C,), device, seed + 3, scale=0.5)             # bonus
            return (k, v, w, u)

        def _core(k, v, w, u):
            Bs, L, C = k.shape
            S = torch.zeros(Bs, C, dtype=k.dtype, device=k.device)
            Z = torch.zeros(Bs, C, dtype=k.dtype, device=k.device)
            y = torch.empty_like(v)
            for t in range(L):
                kt, vt = k[:, t], v[:, t]
                ek = torch.exp(kt)
                eb = torch.exp(u + kt)
                y[:, t] = (S + eb * vt) / (Z + eb)
                wt = F.softplus(w[:, t]) if data_dep else F.softplus(w)
                dec = torch.exp(-wt)
                S = dec * S + ek * vt
                Z = dec * Z + ek
            return y

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 4

    elif fam == "hgrn":
        # HGRN gated linear RNN: h_t = f_t h_{t-1} + (1-f_t) g_t, f=sigmoid(f_l).
        def get_inputs(shape, device="cuda", seed=0):
            B, D, L = shape["B"], shape["D"], shape["L"]
            f_l = _randn((B, D, L), device, seed)
            g = _randn((B, D, L), device, seed + 1)
            return (f_l, g)

        def _core(f_l, g):
            L = f_l.shape[-1]
            f = torch.sigmoid(f_l)
            out = torch.empty_like(g)
            h = torch.zeros(g.shape[:-1], dtype=g.dtype, device=g.device)
            for t in range(L):
                h = f[..., t] * h + (1.0 - f[..., t]) * g[..., t]
                out[..., t] = h
            return out

        def ref_fn(f_l, g):
            return _core(f_l.float(), g.float()).to(f_l.dtype)

        def baseline_fn(f_l, g):
            return _core(f_l, g)

        arity = 2

    elif fam == "gilr":
        # GILR gated impulse linear recurrent: h_t = f_t h_{t-1} + i_t z_t,
        # f=sigmoid(f_l) (forget gate), i=sigmoid(i_l) (input gate).
        def get_inputs(shape, device="cuda", seed=0):
            B, D, L = shape["B"], shape["D"], shape["L"]
            f_l = _randn((B, D, L), device, seed)
            i_l = _randn((B, D, L), device, seed + 1)
            z = _randn((B, D, L), device, seed + 2)
            return (f_l, i_l, z)

        def _core(f_l, i_l, z):
            L = f_l.shape[-1]
            f = torch.sigmoid(f_l)
            i = torch.sigmoid(i_l)
            out = torch.empty_like(z)
            h = torch.zeros(z.shape[:-1], dtype=z.dtype, device=z.device)
            for t in range(L):
                h = f[..., t] * h + i[..., t] * z[..., t]
                out[..., t] = h
            return out

        def ref_fn(f_l, i_l, z):
            return _core(f_l.float(), i_l.float(), z.float()).to(f_l.dtype)

        def baseline_fn(f_l, i_l, z):
            return _core(f_l, i_l, z)

        arity = 3

    else:
        raise NotImplementedError(f"family {fam!r} for op {op!r} not yet implemented")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op,
          "dtype_name": dtype, "family": f"breadth_{op}", "mutates_input": False,
          "time_axis": _TIME_AXIS[fam], "seq_input_idx": _VALUE_INPUT_IDX[fam]}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Naive (correct, compiling) SEQUENTIAL Triton seeds - the policy's start.
# One program per row / channel / head keeps a running state and loops over the
# time axis (fp32 math, {tldt} store). Correct-but-slow; the policy replaces the
# serial loop with a chunked / parallel prefix scan. {dtype} lands in the
# docstring; {tldt} is the tl store dtype literal.
# --------------------------------------------------------------------------- #
_SEED_LOGCUMSUMEXP = '''"""GENERATED breadth ssm_logcumsumexp seed ({dtype}). Stable log-cumsum-exp over
the last dim. One program per flattened row; a streaming (running max m, running
sum s) fp32 scan. The policy replaces the serial loop with a parallel prefix
(log-space) scan. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_logcumsumexp_kernel(x_ptr, y_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    m = -1.0e30
    s = 0.0
    for i in range(0, L):
        v = tl.load(x_ptr + base + i).to(tl.float32)
        mn = tl.maximum(m, v)
        s = s * tl.exp(m - mn) + tl.exp(v - mn)
        m = mn
        tl.store(y_ptr + base + i, (m + tl.log(s)).to({tldt}))


def ssm_logcumsumexp(x: torch.Tensor) -> torch.Tensor:
    L = x.shape[-1]
    xf = x.contiguous().reshape(-1, L)
    y = torch.empty_like(xf)
    _ssm_logcumsumexp_kernel[(xf.shape[0],)](xf, y, L, xf.stride(0), num_warps=1)
    return y.reshape(x.shape)
'''

_SEED_CUMMAX = '''"""GENERATED breadth ssm_cummax seed ({dtype}). Cumulative maximum over the last
dim. One program per flattened row; a sequential running-max scan (naive but
correct). The policy replaces the serial loop with a parallel prefix (max) scan.
{tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_cummax_kernel(x_ptr, y_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    acc = -1.0e30
    for i in range(0, L):
        v = tl.load(x_ptr + base + i).to(tl.float32)
        acc = tl.maximum(acc, v)
        tl.store(y_ptr + base + i, acc.to({tldt}))


def ssm_cummax(x: torch.Tensor) -> torch.Tensor:
    L = x.shape[-1]
    xf = x.contiguous().reshape(-1, L)
    y = torch.empty_like(xf)
    _ssm_cummax_kernel[(xf.shape[0],)](xf, y, L, xf.stride(0), num_warps=1)
    return y.reshape(x.shape)
'''

_SEED_CUMMIN = '''"""GENERATED breadth ssm_cummin seed ({dtype}). Cumulative minimum over the last
dim. One program per flattened row; a sequential running-min scan (naive but
correct). The policy replaces the serial loop with a parallel prefix (min) scan.
{tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_cummin_kernel(x_ptr, y_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    acc = 1.0e30
    for i in range(0, L):
        v = tl.load(x_ptr + base + i).to(tl.float32)
        acc = tl.minimum(acc, v)
        tl.store(y_ptr + base + i, acc.to({tldt}))


def ssm_cummin(x: torch.Tensor) -> torch.Tensor:
    L = x.shape[-1]
    xf = x.contiguous().reshape(-1, L)
    y = torch.empty_like(xf)
    _ssm_cummin_kernel[(xf.shape[0],)](xf, y, L, xf.stride(0), num_warps=1)
    return y.reshape(x.shape)
'''

_SEED_SEGMENTED_SCAN = '''"""GENERATED breadth ssm_segmented_scan seed ({dtype}). Blelloch segmented
cumulative sum: reset flag r_t==1 starts a new segment. Gated recurrence
h_t = (1 - r_t) * h_{{t-1}} + x_t over the last dim. One program per flattened
row; sequential fp32 scan (the associative segmented operator makes this a
parallel prefix scan the policy builds). Inputs x[...,L], reset[...,L]. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_segmented_scan_kernel(x_ptr, r_ptr, h_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    h = 0.0
    for i in range(0, L):
        xv = tl.load(x_ptr + base + i).to(tl.float32)
        rv = tl.load(r_ptr + base + i).to(tl.float32)
        h = (1.0 - rv) * h + xv
        tl.store(h_ptr + base + i, h.to({tldt}))


def ssm_segmented_scan(x: torch.Tensor, reset: torch.Tensor) -> torch.Tensor:
    L = x.shape[-1]
    xf = x.contiguous().reshape(-1, L)
    rf = reset.contiguous().reshape(-1, L)
    h = torch.empty_like(xf)
    _ssm_segmented_scan_kernel[(xf.shape[0],)](xf, rf, h, L, xf.stride(0), num_warps=1)
    return h.reshape(x.shape)
'''

_SEED_LRU = '''"""GENERATED breadth ssm_lru seed ({dtype}). Linear Recurrent Unit: complex
diagonal recurrence h_t = lam*h_{{t-1}} + b_t, lam = nu*exp(i*theta). Complex is
carried as a trailing size-2 (real, imag) axis. One program per (batch, channel)
keeps the fp32 complex state (hr, hi) and scans over L (the policy parallelizes it
via an associative complex scan). lam's (lr, li) are precomputed. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_lru_kernel(x_ptr, lr_ptr, li_ptr, y_ptr, D, L, srow, sl):
    row = tl.program_id(0)
    d = row % D
    lr = tl.load(lr_ptr + d).to(tl.float32)
    li = tl.load(li_ptr + d).to(tl.float32)
    base = row * srow
    hr = 0.0
    hi = 0.0
    for i in range(0, L):
        br = tl.load(x_ptr + base + i * sl + 0).to(tl.float32)
        bi = tl.load(x_ptr + base + i * sl + 1).to(tl.float32)
        hrn = lr * hr - li * hi + br
        hin = li * hr + lr * hi + bi
        hr = hrn
        hi = hin
        tl.store(y_ptr + base + i * sl + 0, hr.to({tldt}))
        tl.store(y_ptr + base + i * sl + 1, hi.to({tldt}))


def ssm_lru(x: torch.Tensor, nu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    B, D, L, _ = x.shape
    lr = (nu * torch.cos(theta)).contiguous()
    li = (nu * torch.sin(theta)).contiguous()
    xf = x.contiguous().reshape(B * D, L, 2)
    y = torch.empty_like(xf)
    _ssm_lru_kernel[(B * D,)](xf, lr, li, y, D, L, xf.stride(0), xf.stride(1), num_warps=1)
    return y.reshape(x.shape)
'''

_SEED_S4D = '''"""GENERATED breadth ssm_s4d seed ({dtype}). S4D diagonal LTI SSM. u[B,D,L],
Abar/Bbar/C[D,N] -> y[B,D,L]. One program per (b, d) keeps an fp32 diagonal state
h[N] and scans over L: a=exp(Abar); h = a*h + Bbar*u; y = sum_n C*h. Naive
sequential scan; the policy exploits time-invariance (long convolution / chunked
scan). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_s4d_kernel(u_ptr, Abar_ptr, Bbar_ptr, C_ptr, y_ptr, D, L, N,
                    su_b, su_d, su_l, sA_d, sA_n, NB: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D
    n = tl.arange(0, NB)
    nmask = n < N
    Aoff = d * sA_d + n * sA_n
    a = tl.exp(tl.load(Abar_ptr + Aoff, mask=nmask, other=0.0).to(tl.float32))
    Bb = tl.load(Bbar_ptr + Aoff, mask=nmask, other=0.0).to(tl.float32)
    Cc = tl.load(C_ptr + Aoff, mask=nmask, other=0.0).to(tl.float32)
    h = tl.zeros([NB], dtype=tl.float32)
    base = b * su_b + d * su_d
    for l in range(0, L):
        uv = tl.load(u_ptr + base + l * su_l).to(tl.float32)
        h = a * h + Bb * uv
        y_v = tl.sum(tl.where(nmask, Cc * h, 0.0), axis=0)
        tl.store(y_ptr + base + l * su_l, y_v.to({tldt}))


def ssm_s4d(u, Abar, Bbar, C):
    Bsz, D, L = u.shape
    N = Abar.shape[1]
    u = u.contiguous(); Abar = Abar.contiguous(); Bbar = Bbar.contiguous(); C = C.contiguous()
    y = torch.empty_like(u)
    NB = triton.next_power_of_2(N)
    _ssm_s4d_kernel[(Bsz * D,)](
        u, Abar, Bbar, C, y, D, L, N,
        u.stride(0), u.stride(1), u.stride(2),
        Abar.stride(0), Abar.stride(1), NB=NB, num_warps=1)
    return y
'''


_SEED_MAMBA2_SSD = '''"""GENERATED breadth {op} seed ({dtype}). Mamba-2 SSD, multi-head scalar decay.
x[B,L,H,P], dt[B,L,H], A[H], B_/C[B,L,H,N] -> y[B,L,H,P]. One program per (b,h,p)
keeps an fp32 state h[N] and scans over L: dt=softplus; a=exp(dt*A); h=a*h+(dt*x)*B_;
y=sum_n C*h. Naive sequential scan; a real SSD kernel processes time in chunks
(matmul the intra-chunk term). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, y_ptr, L, H, P, N,
                 sx_b, sx_l, sx_h, sx_p, sdt_b, sdt_l, sdt_h, sA_h,
                 sB_b, sB_l, sB_h, sB_n, NB: tl.constexpr):
    pid = tl.program_id(0)
    p = pid % P
    tmp = pid // P
    hh = tmp % H
    b = tmp // H
    n = tl.arange(0, NB)
    nmask = n < N
    A_h = tl.load(A_ptr + hh * sA_h).to(tl.float32)
    state = tl.zeros([NB], dtype=tl.float32)
    for l in range(0, L):
        dt = tl.load(dt_ptr + b * sdt_b + l * sdt_l + hh * sdt_h).to(tl.float32)
        dt = tl.where(dt > 20.0, dt, tl.log(1.0 + tl.exp(dt)))
        a = tl.exp(dt * A_h)
        xoff = b * sx_b + l * sx_l + hh * sx_h + p * sx_p
        xv = tl.load(x_ptr + xoff).to(tl.float32)
        boff = b * sB_b + l * sB_l + hh * sB_h + n * sB_n
        Bv = tl.load(B_ptr + boff, mask=nmask, other=0.0).to(tl.float32)
        Cv = tl.load(C_ptr + boff, mask=nmask, other=0.0).to(tl.float32)
        state = a * state + (dt * xv) * Bv
        y_v = tl.sum(tl.where(nmask, Cv * state, 0.0), axis=0)
        tl.store(y_ptr + xoff, y_v.to({tldt}))


def {op}(x, dt, A, B_, C):
    B, L, H, P = x.shape
    N = B_.shape[-1]
    x = x.contiguous(); dt = dt.contiguous(); A = A.contiguous()
    B_ = B_.contiguous(); C = C.contiguous()
    y = torch.empty_like(x)
    NB = triton.next_power_of_2(N)
    _{op}_kernel[(B * H * P,)](
        x, dt, A, B_, C, y, L, H, P, N,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        dt.stride(0), dt.stride(1), dt.stride(2), A.stride(0),
        B_.stride(0), B_.stride(1), B_.stride(2), B_.stride(3),
        NB=NB, num_warps=1)
    return y
'''

_SEED_SELECTIVE = '''"""GENERATED breadth {op} seed ({dtype}). Mamba-1 selective SSM core. u/delta[B,L,D],
A[D,N], B_/C[B,L,N], D_[D] -> y[B,L,D]. One program per (b,d) keeps an fp32 state
h[N] and scans over L: dt=softplus(delta); dA=exp(dt*A); dBu=dt*B_*u; h=dA*h+dBu;
y=sum_n C*h + D_*u. Naive sequential scan (the policy fuses/chunks it). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, Dskip_ptr, y_ptr, L, D, N,
                 su_b, su_l, su_d, sA_d, sA_n, sB_b, sB_l, sB_n, sDs, NB: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D
    n = tl.arange(0, NB)
    nmask = n < N
    Arow = tl.load(A_ptr + d * sA_d + n * sA_n, mask=nmask, other=0.0).to(tl.float32)
    Dd = tl.load(Dskip_ptr + d * sDs).to(tl.float32)
    h = tl.zeros([NB], dtype=tl.float32)
    for l in range(0, L):
        off_ld = b * su_b + l * su_l + d * su_d
        u_v = tl.load(u_ptr + off_ld).to(tl.float32)
        dt = tl.load(delta_ptr + off_ld).to(tl.float32)
        dt = tl.where(dt > 20.0, dt, tl.log(1.0 + tl.exp(dt)))
        off_ln = b * sB_b + l * sB_l + n * sB_n
        Bv = tl.load(B_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        Cv = tl.load(C_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        h = tl.exp(dt * Arow) * h + (dt * Bv * u_v)
        y_v = tl.sum(tl.where(nmask, Cv * h, 0.0), axis=0) + Dd * u_v
        tl.store(y_ptr + off_ld, y_v.to({tldt}))


def {op}(u, delta, A, B_, C, D_):
    Bsz, L, D = u.shape
    N = A.shape[1]
    u = u.contiguous(); delta = delta.contiguous(); A = A.contiguous()
    B_ = B_.contiguous(); C = C.contiguous(); D_ = D_.contiguous()
    y = torch.empty_like(u)
    NB = triton.next_power_of_2(N)
    _{op}_kernel[(Bsz * D,)](
        u, delta, A, B_, C, D_, y, L, D, N,
        u.stride(0), u.stride(1), u.stride(2), A.stride(0), A.stride(1),
        B_.stride(0), B_.stride(1), B_.stride(2), D_.stride(0), NB=NB, num_warps=1)
    return y
'''

_SEED_CONV_SSD = '''"""GENERATED breadth {op} seed ({dtype}). Mamba conv+SSM: causal depthwise conv
(+SiLU) then a scalar-decay SSD scan. x[B,L,D], conv_w[D,K], a[B,L], B_/C[B,L,N] ->
y[B,L,D]. A Triton causal-convolution kernel materializes the SiLU projection; one
scan program per (b,d) then keeps an fp32 state h[N] over L. The policy fuses the
two honest Triton stages and chunks the recurrence. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(xc_ptr, a_ptr, B_ptr, C_ptr, y_ptr, L, D, N,
                 s_b, s_l, s_d, sa_b, sa_l, sB_b, sB_l, sB_n, NB: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D
    n = tl.arange(0, NB)
    nmask = n < N
    h = tl.zeros([NB], dtype=tl.float32)
    for l in range(0, L):
        dec = tl.sigmoid(tl.load(a_ptr + b * sa_b + l * sa_l).to(tl.float32))
        xoff = b * s_b + l * s_l + d * s_d
        xv = tl.load(xc_ptr + xoff).to(tl.float32)
        boff = b * sB_b + l * sB_l + n * sB_n
        Bv = tl.load(B_ptr + boff, mask=nmask, other=0.0).to(tl.float32)
        Cv = tl.load(C_ptr + boff, mask=nmask, other=0.0).to(tl.float32)
        h = dec * h + xv * Bv
        y_v = tl.sum(tl.where(nmask, Cv * h, 0.0), axis=0)
        tl.store(y_ptr + xoff, y_v.to({tldt}))


def {op}(x, conv_w, a, B_, C):
    B, L, D = x.shape
    N = B_.shape[-1]
    xc = _causal_conv_silu(x, conv_w)
    a = a.contiguous(); B_ = B_.contiguous(); C = C.contiguous()
    y = torch.empty_like(xc)
    NB = triton.next_power_of_2(N)
    _{op}_kernel[(B * D,)](
        xc, a, B_, C, y, L, D, N,
        xc.stride(0), xc.stride(1), xc.stride(2), a.stride(0), a.stride(1),
        B_.stride(0), B_.stride(1), B_.stride(2), NB=NB, num_warps=1)
    return y
'''

_SEED_CONV_SELECTIVE = '''"""GENERATED breadth {op} seed ({dtype}). Mamba conv+SSM: causal depthwise conv
(+SiLU) then the Mamba-1 selective scan. u[B,L,D], conv_w[D,K], delta[B,L,D],
A[D,N], B_/C[B,L,N] -> y[B,L,D]. A Triton causal-convolution kernel materializes
the SiLU projection; one scan program per (b,d) then computes the selective
recurrence. The policy fuses the two honest Triton stages and chunks it. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(uc_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, y_ptr, L, D, N,
                 su_b, su_l, su_d, sA_d, sA_n, sB_b, sB_l, sB_n, NB: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D
    n = tl.arange(0, NB)
    nmask = n < N
    Arow = tl.load(A_ptr + d * sA_d + n * sA_n, mask=nmask, other=0.0).to(tl.float32)
    h = tl.zeros([NB], dtype=tl.float32)
    for l in range(0, L):
        off_ld = b * su_b + l * su_l + d * su_d
        uv = tl.load(uc_ptr + off_ld).to(tl.float32)
        dt = tl.load(delta_ptr + off_ld).to(tl.float32)
        dt = tl.where(dt > 20.0, dt, tl.log(1.0 + tl.exp(dt)))
        off_ln = b * sB_b + l * sB_l + n * sB_n
        Bv = tl.load(B_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        Cv = tl.load(C_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        h = tl.exp(dt * Arow) * h + (dt * Bv) * uv
        y_v = tl.sum(tl.where(nmask, Cv * h, 0.0), axis=0)
        tl.store(y_ptr + off_ld, y_v.to({tldt}))


def {op}(u, conv_w, delta, A, B_, C):
    B, L, D = u.shape
    N = A.shape[1]
    uc = _causal_conv_silu(u, conv_w)
    delta = delta.contiguous(); A = A.contiguous(); B_ = B_.contiguous(); C = C.contiguous()
    y = torch.empty_like(uc)
    NB = triton.next_power_of_2(N)
    _{op}_kernel[(B * D,)](
        uc, delta, A, B_, C, y, L, D, N,
        uc.stride(0), uc.stride(1), uc.stride(2), A.stride(0), A.stride(1),
        B_.stride(0), B_.stride(1), B_.stride(2), NB=NB, num_warps=1)
    return y
'''

_SEED_CAUSAL_CONV_SILU = '''

@triton.jit
def _causal_conv_silu_kernel(x_ptr, w_ptr, out_ptr, L, D,
                             sx_b, sx_l, sx_d, sw_d, sw_k,
                             K: tl.constexpr, BLOCK_D: tl.constexpr):
    bl = tl.program_id(0)
    b = bl // L
    pos = bl % L
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    dmask = d < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for k in range(0, K):
        src_pos = pos - (K - 1) + k
        valid = dmask & (src_pos >= 0)
        xv = tl.load(x_ptr + b * sx_b + src_pos * sx_l + d * sx_d,
                     mask=valid, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + d * sw_d + k * sw_k,
                     mask=dmask, other=0.0).to(tl.float32)
        acc += xv * wv
    acc = acc * tl.sigmoid(acc)
    tl.store(out_ptr + b * sx_b + pos * sx_l + d * sx_d,
             acc.to(out_ptr.dtype.element_ty), mask=dmask)


def _causal_conv_silu(x, conv_w):
    B, L, D = x.shape
    K = conv_w.shape[1]
    x = x.contiguous()
    conv_w = conv_w.contiguous()
    out = torch.empty_like(x)
    BLOCK_D = 256
    _causal_conv_silu_kernel[(B * L, triton.cdiv(D, BLOCK_D))](
        x, conv_w, out, L, D,
        x.stride(0), x.stride(1), x.stride(2),
        conv_w.stride(0), conv_w.stride(1),
        K=K, BLOCK_D=BLOCK_D)
    return out
'''


_SEED_GLA = '''"""GENERATED breadth {op} seed ({dtype}). Gated Linear Attention (per-key-dim
data-dependent gate). q/k/v[B,H,L,Dh], gate logits gl[B,H,L,Dh] -> y[B,H,L,Dh].
One program per (b,h,e) keeps the fp32 state COLUMN s[Dh] (= S[:, e]) and scans
over L: a=sigmoid(gl); s = a*s + phi(k_l)*v_l[e]; y_l[e]=sum_d q_l[d]*s[d]. Naive
sequential scan (the policy chunks/parallelizes it). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(q_ptr, k_ptr, v_ptr, g_ptr, y_ptr, H, L, Dh, s_b, s_h, s_l, s_d,
                 DB: tl.constexpr):
    pid = tl.program_id(0)
    e = pid % Dh
    tmp = pid // Dh
    hh = tmp % H
    bb = tmp // H
    dd = tl.arange(0, DB)
    dmask = dd < Dh
    bh = bb * s_b + hh * s_h
    s = tl.zeros([DB], dtype=tl.float32)
    for l in range(0, L):
        off = bh + l * s_l
        krow = tl.load(k_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        grow = tl.load(g_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        a = tl.sigmoid(grow)
        v_le = tl.load(v_ptr + off + e * s_d).to(tl.float32)
        s = a * s + krow * v_le
        qrow = tl.load(q_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        y_le = tl.sum(tl.where(dmask, qrow * s, 0.0), axis=0)
        tl.store(y_ptr + off + e * s_d, y_le.to({tldt}))


def {op}(q, k, v, gl):
    B, H, L, Dh = q.shape
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); gl = gl.contiguous()
    y = torch.empty_like(v)
    DB = triton.next_power_of_2(Dh)
    _{op}_kernel[(B * H * Dh,)](q, k, v, gl, y, H, L, Dh,
                               q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                               DB=DB, num_warps=1)
    return y
'''

_SEED_GATED_RETENTION = '''"""GENERATED breadth {op} seed ({dtype}). Gated retention (data-dependent SCALAR
decay). q/k/v[B,H,L,Dh], gate logits gl[B,H,L] -> y[B,H,L,Dh]. One program per
(b,h,e) keeps the fp32 state column s[Dh] and scans over L: a=sigmoid(gl_l);
s = a*s + k_l*v_l[e]; y_l[e]=sum_d q_l[d]*s[d]. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(q_ptr, k_ptr, v_ptr, g_ptr, y_ptr, H, L, Dh, s_b, s_h, s_l, s_d,
                 sg_b, sg_h, sg_l, DB: tl.constexpr):
    pid = tl.program_id(0)
    e = pid % Dh
    tmp = pid // Dh
    hh = tmp % H
    bb = tmp // H
    dd = tl.arange(0, DB)
    dmask = dd < Dh
    bh = bb * s_b + hh * s_h
    gh = bb * sg_b + hh * sg_h
    s = tl.zeros([DB], dtype=tl.float32)
    for l in range(0, L):
        off = bh + l * s_l
        a = tl.sigmoid(tl.load(g_ptr + gh + l * sg_l).to(tl.float32))
        krow = tl.load(k_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        v_le = tl.load(v_ptr + off + e * s_d).to(tl.float32)
        s = a * s + krow * v_le
        qrow = tl.load(q_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        y_le = tl.sum(tl.where(dmask, qrow * s, 0.0), axis=0)
        tl.store(y_ptr + off + e * s_d, y_le.to({tldt}))


def {op}(q, k, v, gl):
    B, H, L, Dh = q.shape
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); gl = gl.contiguous()
    y = torch.empty_like(v)
    DB = triton.next_power_of_2(Dh)
    _{op}_kernel[(B * H * Dh,)](q, k, v, gl, y, H, L, Dh,
                               q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                               gl.stride(0), gl.stride(1), gl.stride(2),
                               DB=DB, num_warps=1)
    return y
'''

_SEED_HGRN2 = '''"""GENERATED breadth {op} seed ({dtype}). HGRN2 outer-product gated expansion
(gate-tied key = 1 - alpha). q/v[B,H,L,Dh], gate logits gl[B,H,L,Dh] -> y[B,H,L,Dh].
One program per (b,h,e) keeps the fp32 state column s[Dh] and scans over L:
a=sigmoid(gl); s = a*s + (1-a)*v_l[e]; y_l[e]=sum_d q_l[d]*s[d]. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(q_ptr, v_ptr, g_ptr, y_ptr, H, L, Dh, s_b, s_h, s_l, s_d,
                 DB: tl.constexpr):
    pid = tl.program_id(0)
    e = pid % Dh
    tmp = pid // Dh
    hh = tmp % H
    bb = tmp // H
    dd = tl.arange(0, DB)
    dmask = dd < Dh
    bh = bb * s_b + hh * s_h
    s = tl.zeros([DB], dtype=tl.float32)
    for l in range(0, L):
        off = bh + l * s_l
        a = tl.sigmoid(tl.load(g_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32))
        v_le = tl.load(v_ptr + off + e * s_d).to(tl.float32)
        s = a * s + (1.0 - a) * v_le
        qrow = tl.load(q_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        y_le = tl.sum(tl.where(dmask, qrow * s, 0.0), axis=0)
        tl.store(y_ptr + off + e * s_d, y_le.to({tldt}))


def {op}(q, v, gl):
    B, H, L, Dh = q.shape
    q = q.contiguous(); v = v.contiguous(); gl = gl.contiguous()
    y = torch.empty_like(v)
    DB = triton.next_power_of_2(Dh)
    _{op}_kernel[(B * H * Dh,)](q, v, gl, y, H, L, Dh,
                               q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                               DB=DB, num_warps=1)
    return y
'''

_SEED_HGRN = '''"""GENERATED breadth {op} seed ({dtype}). HGRN gated linear RNN over the last dim:
h_t = f_t h_{{t-1}} + (1-f_t) g_t, f=sigmoid(f_l). One program per flattened row;
sequential fp32 scan (the policy builds the parallel prefix scan). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(f_ptr, g_ptr, h_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    h = 0.0
    for i in range(0, L):
        f = tl.sigmoid(tl.load(f_ptr + base + i).to(tl.float32))
        g = tl.load(g_ptr + base + i).to(tl.float32)
        h = f * h + (1.0 - f) * g
        tl.store(h_ptr + base + i, h.to({tldt}))


def {op}(f_l, g):
    L = f_l.shape[-1]
    ff = f_l.contiguous().reshape(-1, L)
    gg = g.contiguous().reshape(-1, L)
    h = torch.empty_like(ff)
    _{op}_kernel[(ff.shape[0],)](ff, gg, h, L, ff.stride(0), num_warps=1)
    return h.reshape(g.shape)
'''

_SEED_GILR = '''"""GENERATED breadth {op} seed ({dtype}). GILR gated impulse linear recurrent over
the last dim: h_t = f_t h_{{t-1}} + i_t z_t, f=sigmoid(f_l), i=sigmoid(i_l). One
program per flattened row; sequential fp32 scan. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(f_ptr, i_ptr, z_ptr, h_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    h = 0.0
    for t in range(0, L):
        f = tl.sigmoid(tl.load(f_ptr + base + t).to(tl.float32))
        ii = tl.sigmoid(tl.load(i_ptr + base + t).to(tl.float32))
        z = tl.load(z_ptr + base + t).to(tl.float32)
        h = f * h + ii * z
        tl.store(h_ptr + base + t, h.to({tldt}))


def {op}(f_l, i_l, z):
    L = f_l.shape[-1]
    ff = f_l.contiguous().reshape(-1, L)
    ii = i_l.contiguous().reshape(-1, L)
    zz = z.contiguous().reshape(-1, L)
    h = torch.empty_like(ff)
    _{op}_kernel[(ff.shape[0],)](ff, ii, zz, h, L, ff.stride(0), num_warps=1)
    return h.reshape(z.shape)
'''


def _phi_expr(fmap: str, var: str) -> str:
    if fmap == "elu":
        return f"tl.where({var} > 0.0, {var} + 1.0, tl.exp({var}))"
    if fmap == "relu":
        return f"tl.maximum({var}, 0.0)"
    if fmap == "relu2":
        return f"tl.maximum({var}, 0.0) * tl.maximum({var}, 0.0)"
    return f"tl.exp({var})"


def _seed_retention(op: str, tldt: str, gamma_list: list) -> str:
    return f'''"""GENERATED breadth {op} seed. RetNet/lightning retention (FIXED per-head decay
gamma_h). q/k/v[B,H,L,Dh] -> y[B,H,L,Dh]. One program per (b,h,e) keeps the fp32
state column s[Dh] and scans over L: s = gamma_h*s + k_l*v_l[e]; y_l[e]=sum_d
q_l[d]*s[d]. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(q_ptr, k_ptr, v_ptr, gamma_ptr, y_ptr, H, L, Dh, s_b, s_h, s_l, s_d,
                 DB: tl.constexpr):
    pid = tl.program_id(0)
    e = pid % Dh
    tmp = pid // Dh
    hh = tmp % H
    bb = tmp // H
    dd = tl.arange(0, DB)
    dmask = dd < Dh
    bh = bb * s_b + hh * s_h
    g = tl.load(gamma_ptr + hh).to(tl.float32)
    s = tl.zeros([DB], dtype=tl.float32)
    for l in range(0, L):
        off = bh + l * s_l
        krow = tl.load(k_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        v_le = tl.load(v_ptr + off + e * s_d).to(tl.float32)
        s = g * s + krow * v_le
        qrow = tl.load(q_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        y_le = tl.sum(tl.where(dmask, qrow * s, 0.0), axis=0)
        tl.store(y_ptr + off + e * s_d, y_le.to({tldt}))


def {op}(q, k, v):
    B, H, L, Dh = q.shape
    gamma = torch.tensor({gamma_list!r}, dtype=torch.float32, device=q.device)
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    y = torch.empty_like(v)
    DB = triton.next_power_of_2(Dh)
    _{op}_kernel[(B * H * Dh,)](q, k, v, gamma, y, H, L, Dh,
                               q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                               DB=DB, num_warps=1)
    return y
'''


def _seed_linattn(op: str, tldt: str, cfg: dict) -> str:
    phik = _phi_expr(cfg["fmap"], "krow")
    phiq = _phi_expr(cfg["fmap"], "qrow")
    if cfg["normalize"]:
        z_init = "    z = tl.zeros([DB], dtype=tl.float32)\n"
        z_upd = "        z = z + phik\n"
        y_out = (f"        den = tl.sum(tl.where(dmask, phiq * z, 0.0), axis=0) + {LINATTN_EPS}\n"
                 "        y_le = num / den\n")
    else:
        z_init = ""
        z_upd = ""
        y_out = "        y_le = num\n"
    return f'''"""GENERATED breadth {op} seed. Causal linear attention, feature map {cfg["fmap"]}
(normalize={cfg["normalize"]}). q/k/v[B,H,L,Dh] -> y[B,H,L,Dh]. One program per
(b,h,e) keeps the fp32 state column s[Dh] (and normalizer z[Dh]) and scans over L:
s += phi(k_l)*v_l[e]; num = sum_d phi(q_l)[d]*s[d]. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(q_ptr, k_ptr, v_ptr, y_ptr, H, L, Dh, s_b, s_h, s_l, s_d,
                 DB: tl.constexpr):
    pid = tl.program_id(0)
    e = pid % Dh
    tmp = pid // Dh
    hh = tmp % H
    bb = tmp // H
    dd = tl.arange(0, DB)
    dmask = dd < Dh
    bh = bb * s_b + hh * s_h
    s = tl.zeros([DB], dtype=tl.float32)
{z_init}    for l in range(0, L):
        off = bh + l * s_l
        krow = tl.load(k_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        phik = {phik}
        phik = tl.where(dmask, phik, 0.0)
        v_le = tl.load(v_ptr + off + e * s_d).to(tl.float32)
        s = s + phik * v_le
        qrow = tl.load(q_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        phiq = {phiq}
        num = tl.sum(tl.where(dmask, phiq * s, 0.0), axis=0)
{z_upd}{y_out}        tl.store(y_ptr + off + e * s_d, y_le.to({tldt}))


def {op}(q, k, v):
    B, H, L, Dh = q.shape
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    y = torch.empty_like(v)
    DB = triton.next_power_of_2(Dh)
    _{op}_kernel[(B * H * Dh,)](q, k, v, y, H, L, Dh,
                               q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                               DB=DB, num_warps=1)
    return y
'''


def _seed_delta(op: str, tldt: str, gated: bool) -> str:
    if gated:
        sig = "q_ptr, k_ptr, v_ptr, a_ptr, be_ptr, y_ptr"
        gstr = "sa_b, sa_h, sa_l, "
        decay = ("        a = tl.sigmoid(tl.load(a_ptr + ah + l * sa_l).to(tl.float32))\n"
                 "        s = a * s\n")
        ah_init = "    ah = bb * sa_b + hh * sa_h\n"
        entry_args = "q, k, v, alpha, beta"
        launch = ("q, k, v, alpha, beta, y, H, L, Dh,\n"
                  "        q.stride(0), q.stride(1), q.stride(2), q.stride(3),\n"
                  "        alpha.stride(0), alpha.stride(1), alpha.stride(2),\n"
                  "        beta.stride(0), beta.stride(1), beta.stride(2),")
    else:
        sig = "q_ptr, k_ptr, v_ptr, be_ptr, y_ptr"
        gstr = ""
        decay = ""
        ah_init = ""
        entry_args = "q, k, v, beta"
        launch = ("q, k, v, beta, y, H, L, Dh,\n"
                  "        q.stride(0), q.stride(1), q.stride(2), q.stride(3),\n"
                  "        beta.stride(0), beta.stride(1), beta.stride(2),")
    return f'''"""GENERATED breadth {op} seed. (Gated) DeltaNet delta-rule linear attention.
k is L2-normalized. One program per (b,h,e) keeps the fp32 state column s[Dh] and
scans over L: [s=alpha*s;] pred=sum_d k_l[d]*s[d]; s += beta*k_l*(v_l[e]-pred);
y_l[e]=sum_d q_l[d]*s[d]. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel({sig}, H, L, Dh, s_b, s_h, s_l, s_d,
                 {gstr}sbe_b, sbe_h, sbe_l, DB: tl.constexpr):
    pid = tl.program_id(0)
    e = pid % Dh
    tmp = pid // Dh
    hh = tmp % H
    bb = tmp // H
    dd = tl.arange(0, DB)
    dmask = dd < Dh
    bh = bb * s_b + hh * s_h
    beh = bb * sbe_b + hh * sbe_h
{ah_init}    s = tl.zeros([DB], dtype=tl.float32)
    for l in range(0, L):
        off = bh + l * s_l
        beta = tl.sigmoid(tl.load(be_ptr + beh + l * sbe_l).to(tl.float32))
        krow = tl.load(k_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
{decay}        pred = tl.sum(tl.where(dmask, krow * s, 0.0), axis=0)
        v_le = tl.load(v_ptr + off + e * s_d).to(tl.float32)
        s = s + beta * krow * (v_le - pred)
        qrow = tl.load(q_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        y_le = tl.sum(tl.where(dmask, qrow * s, 0.0), axis=0)
        tl.store(y_ptr + off + e * s_d, y_le.to({tldt}))


def {op}({entry_args}):
    B, H, L, Dh = q.shape
    k = k / (k.norm(dim=-1, keepdim=True) + {DELTA_EPS})
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    beta = beta.contiguous()
    {"alpha = alpha.contiguous()" if gated else ""}
    y = torch.empty_like(v)
    DB = triton.next_power_of_2(Dh)
    _{op}_kernel[(B * H * Dh,)](
        {launch}
        DB=DB, num_warps=1)
    return y
'''


def _seed_rwkv(op: str, tldt: str, data_dep: bool) -> str:
    if data_dep:
        w_load = "tl.load(w_ptr + b * sw_b + l * sw_l + c * sw_c).to(tl.float32)"
        w_sig = "sw_b, sw_l, sw_c, "
        w_launch = "w.stride(0), w.stride(1), w.stride(2),"
    else:
        w_load = "tl.load(w_ptr + c * sw_c).to(tl.float32)"
        w_sig = "sw_c, "
        w_launch = "w.stride(0),"
    return f'''"""GENERATED breadth {op} seed. RWKV wkv (num/den + bonus), data_dependent={data_dep}.
k/v[B,L,C] -> y[B,L,C]. One program per (b,c) keeps fp32 (S, Z) and scans over L:
wkv=(S+exp(u+k)v)/(Z+exp(u+k)); w=softplus(decay); S=exp(-w)S+exp(k)v;
Z=exp(-w)Z+exp(k). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(k_ptr, v_ptr, w_ptr, u_ptr, y_ptr, C, L,
                 sk_b, sk_l, sk_c, {w_sig}su_c, SP: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C
    uc = tl.load(u_ptr + c * su_c).to(tl.float32)
    S = 0.0
    Z = 0.0
    for l in range(0, L):
        off = b * sk_b + l * sk_l + c * sk_c
        kt = tl.load(k_ptr + off).to(tl.float32)
        vt = tl.load(v_ptr + off).to(tl.float32)
        ek = tl.exp(kt)
        eb = tl.exp(uc + kt)
        tl.store(y_ptr + off, ((S + eb * vt) / (Z + eb)).to({tldt}))
        wl = {w_load}
        wt = tl.where(wl > 20.0, wl, tl.log(1.0 + tl.exp(wl)))
        dec = tl.exp(-wt)
        S = dec * S + ek * vt
        Z = dec * Z + ek


def {op}(k, v, w, u):
    B, L, C = k.shape
    k = k.contiguous(); v = v.contiguous(); w = w.contiguous(); u = u.contiguous()
    y = torch.empty_like(v)
    _{op}_kernel[(B * C,)](k, v, w, u, y, C, L,
                          k.stride(0), k.stride(1), k.stride(2),
                          {w_launch} u.stride(0), SP=0, num_warps=1)
    return y
'''


def seed_source(op: str, dtype: str) -> str:
    tldt = DTYPES[dtype][1]
    fam = OP_FAMILY[op]
    cfg = OP_CONFIG[op]
    if fam == "gla":
        return _SEED_GLA.format(op=op, dtype=dtype, tldt=tldt)
    if fam == "gated_retention":
        return _SEED_GATED_RETENTION.format(op=op, dtype=dtype, tldt=tldt)
    if fam == "hgrn2":
        return _SEED_HGRN2.format(op=op, dtype=dtype, tldt=tldt)
    if fam == "hgrn":
        return _SEED_HGRN.format(op=op, dtype=dtype, tldt=tldt)
    if fam == "gilr":
        return _SEED_GILR.format(op=op, dtype=dtype, tldt=tldt)
    if fam == "retention":
        return _seed_retention(op, tldt, retention_gamma(cfg["H"], cfg["decay"]))
    if fam == "linattn":
        return _seed_linattn(op, tldt, cfg)
    if fam == "delta":
        return _seed_delta(op, tldt, cfg["gated"])
    if fam == "rwkv":
        return _seed_rwkv(op, tldt, cfg["data_dependent"])
    if fam == "mamba2_ssd":
        return _SEED_MAMBA2_SSD.format(op=op, dtype=dtype, tldt=tldt)
    if fam == "selective":
        return _SEED_SELECTIVE.format(op=op, dtype=dtype, tldt=tldt)
    if fam == "conv_ssd":
        return (_SEED_CONV_SSD + _SEED_CAUSAL_CONV_SILU).format(
            op=op, dtype=dtype, tldt=tldt)
    if fam == "conv_selective":
        return (_SEED_CONV_SELECTIVE + _SEED_CAUSAL_CONV_SILU).format(
            op=op, dtype=dtype, tldt=tldt)
    if fam == "scan_lse":
        return _SEED_LOGCUMSUMEXP.format(dtype=dtype, tldt=tldt)
    if fam == "scan_cummax":
        return _SEED_CUMMAX.format(dtype=dtype, tldt=tldt)
    if fam == "scan_cummin":
        return _SEED_CUMMIN.format(dtype=dtype, tldt=tldt)
    if fam == "segmented_scan":
        return _SEED_SEGMENTED_SCAN.format(dtype=dtype, tldt=tldt)
    if fam == "lru":
        return _SEED_LRU.format(dtype=dtype, tldt=tldt)
    if fam == "s4d":
        return _SEED_S4D.format(dtype=dtype, tldt=tldt)
    raise NotImplementedError(f"seed for family {fam!r} (op {op!r}) not yet implemented")
