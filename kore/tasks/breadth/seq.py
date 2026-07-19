"""Breadth SEQUENCE-MODEL + CONV1D task-authoring engine (torch-baselined).

Widens the KORE suite with the state-space / linear-recurrence operator families
that dominate the modern sub-quadratic sequence models (Mamba-1 / Mamba-2 / linear
attention / gated linear RNNs) but that the vendor-baselined core (norms /
activations / GEMM / attention / RoPE) and the other breadth engines (conv2d,
sort/sparse, train ops) never covered. These ops are the archetypal "hard for a
GPU" class: a SEQUENTIAL recurrence over the time axis (an associative scan) whose
naive form is memory-bound and latency-bound, so a fused/chunked Triton kernel has
genuine headroom over the torch-eager multi-kernel baseline.

Unlike the vendor tasks (graded against AITER), these grade against the honest
torch reference: the correctness ORACLE is a torch fp32 reference (``ref_fn``) and
the perf BASELINE is the eager torch computation (``baseline_fn``) - the naive
Python-loop-over-time recurrence a fused Triton scan must beat.

Op families
-----------
  * ``cumsum``               - row-wise cumulative sum over the last dim.
  * ``cumprod``              - row-wise cumulative product over the last dim.
  * ``assoc_scan_segmented`` - gated (segmented) associative scan, the first-order
        linear recurrence ``h_t = a_t * h_{t-1} + b_t`` (a gate ``a_t == 0`` marks a
        segment boundary / state reset, so the gated form subsumes segmentation).
  * ``selective_scan``       - the Mamba-1 selective SSM core (the canonical hard
        kernel): the discretized input-dependent state-space recurrence.
  * ``ssd_chunk_scan``       - a simplified Mamba-2 SSD scalar-decay state-space scan.
  * ``linear_attention``     - causal linear attention with feature map phi=elu+1.
  * ``causal_conv1d``        - depthwise causal 1D convolution (the Mamba short conv).

Contract mirrors ``kore/tasks/vendor_ops.py`` (and the sibling breadth engines) so
the shared ``_genops`` driver + the genv-style generator consume it unchanged:

    OPS / OP_DTYPES / SHAPES              module-level task catalog
    make_reference(op, dtype) -> dict     reference.py namespace (parse_shape,
        get_inputs, ref_fn fp32 oracle, baseline_fn torch, arity, entry_name,
        dtype_name, family=f"breadth_{op}", mutates_input)
    seed_source(op, dtype) -> str         a naive, COMPILING, correct Triton seed
        (defines ``def <op>(*inputs)``) - the policy's starting point.

CORRECTNESS is paramount: every ``ref_fn`` computes in fp32 and casts back to the
task dtype, and is validated on CPU against an INDEPENDENT torch computation (see
tests/test_seq.py - the recurrences are cross-checked against their closed-form /
quadratic-attention / im2col-free definitions). The naive Triton seeds are
correct-but-slow SEQUENTIAL scans (one program per row/channel, a running-state
loop over time) - the honest starting point the policy learns to parallelize
(Blelloch prefix scan / chunked SSD). torch/triton are imported lazily (registry
discovery never needs a GPU).
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# --------------------------------------------------------------------------- #
# task catalog
# --------------------------------------------------------------------------- #
OPS: list[str] = [
    "cumsum",
    "cumprod",
    "assoc_scan_segmented",
    "selective_scan",
    "ssd_chunk_scan",
    "linear_attention",
    "causal_conv1d",
]

# bf16/fp16 sweep (matches the vendor + sibling-breadth default); the fp32 oracle
# casts back. Materialized dict so a generator can iterate OPS x dtypes directly.
DEFAULT_DTYPES: list[str] = ["bf16", "fp16"]
OP_DTYPES: dict[str, list[str]] = {op: list(DEFAULT_DTYPES) for op in OPS}


def op_dtypes(op: str) -> list[str]:
    """The dtype sweep for a breadth op (per-op override or the global default)."""
    return OP_DTYPES.get(op, DEFAULT_DTYPES)


# --------------------------------------------------------------------------- #
# Realistic sequence-model shapes (B=1-4, L=512-4096, D=1024-4096, N=16, K=4).
# Row/scan ops carry the scan axis LAST (cumsum over L). SSM ops use the
# (batch, seqlen, dim) + state-dim N layout. Linear attention is multi-head
# (B, H, L, Dh) with a small head dim. A non-power-of-2 L tail stresses masking.
# --------------------------------------------------------------------------- #
_SCAN_SHAPES = {  # x[B, D, L] ; cumulative scan over the last (time) dim L
    "minimal": {"B": 1, "D": 64, "L": 256},
    "primary": {"B": 2, "D": 2048, "L": 2048},
    "validation": [
        {"B": 4, "D": 1024, "L": 4096},
        {"B": 1, "D": 4096, "L": 1024},
        {"B": 2, "D": 1536, "L": 2047},   # non-pow2 L tail
    ],
}
_SSM_SHAPES = {  # u/delta[B, L, D], A[D, N], B_/C[B, L, N], D_[D]  (Mamba layout)
    "minimal": {"B": 1, "L": 128, "D": 256, "N": 16},
    "primary": {"B": 2, "L": 2048, "D": 2048, "N": 16},
    "validation": [
        {"B": 4, "L": 1024, "D": 1024, "N": 16},
        {"B": 1, "L": 4096, "D": 1536, "N": 16},
        {"B": 2, "L": 2047, "D": 1024, "N": 16},   # non-pow2 L tail
    ],
}
_LINATTN_SHAPES = {  # q/k/v[B, H, L, Dh] ; small head dim so the state [Dh, Dh] fits
    "minimal": {"B": 1, "H": 2, "L": 128, "Dh": 32},
    "primary": {"B": 2, "H": 16, "L": 1024, "Dh": 64},
    "validation": [
        {"B": 1, "H": 32, "L": 2048, "Dh": 64},
        {"B": 2, "H": 8, "L": 512, "Dh": 128},
        {"B": 1, "H": 16, "L": 1023, "Dh": 64},   # non-pow2 L tail
    ],
}
_CONV1D_SHAPES = {  # x[B, D, L], weight[D, K], bias[D] ; depthwise causal short conv
    "minimal": {"B": 1, "D": 256, "L": 256, "K": 4},
    "primary": {"B": 2, "D": 2048, "L": 2048, "K": 4},
    "validation": [
        {"B": 4, "D": 1024, "L": 4096, "K": 4},
        {"B": 1, "D": 4096, "L": 1024, "K": 4},
        {"B": 2, "D": 1536, "L": 2047, "K": 4},   # non-pow2 L tail
    ],
}

SHAPES: dict[str, dict] = {
    "cumsum": _SCAN_SHAPES,
    "cumprod": _SCAN_SHAPES,
    "assoc_scan_segmented": _SCAN_SHAPES,
    "selective_scan": _SSM_SHAPES,
    "ssd_chunk_scan": _SSM_SHAPES,
    "linear_attention": _LINATTN_SHAPES,
    "causal_conv1d": _CONV1D_SHAPES,
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
        return (torch.randn(shape, generator=g, device=device,
                            dtype=torch.float32) * scale).to(tdt)

    def _rand01(shape, device, seed):
        """Gate/decay fill in (0, 1) via sigmoid(randn) - keeps the recurrence stable."""
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.sigmoid(torch.randn(shape, generator=g, device=device,
                                         dtype=torch.float32)).to(tdt)

    def _neg_exp(shape, device, seed, scale=0.5):
        """Strictly-negative fill A = -exp(scale*randn) (the S4D-style SSM state matrix,
        so the discrete decay exp(dt*A) with dt>0 lands in (0, 1))."""
        g = torch.Generator(device=device).manual_seed(seed)
        return (-torch.exp(torch.randn(shape, generator=g, device=device,
                                       dtype=torch.float32) * scale)).to(tdt)

    def _pos_near_one(shape, device, seed, scale=0.02):
        """Positive fill exp(scale*randn) ~ 1 - a well-conditioned cumprod input (the
        product stays O(1) over long L instead of over/underflowing)."""
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.exp(torch.randn(shape, generator=g, device=device,
                                     dtype=torch.float32) * scale).to(tdt)

    # ---------------------------------------------------------- CUMULATIVE SCANS
    if op == "cumsum":
        def get_inputs(shape, device="cuda", seed=0):
            B, D, L = shape["B"], shape["D"], shape["L"]
            return (_randn((B, D, L), device, seed),)

        def ref_fn(x):
            return x.float().cumsum(-1).to(x.dtype)

        def baseline_fn(x):
            return x.cumsum(-1)

        arity = 1

    elif op == "cumprod":
        def get_inputs(shape, device="cuda", seed=0):
            B, D, L = shape["B"], shape["D"], shape["L"]
            return (_pos_near_one((B, D, L), device, seed),)

        def ref_fn(x):
            return x.float().cumprod(-1).to(x.dtype)

        def baseline_fn(x):
            return x.cumprod(-1)

        arity = 1

    elif op == "assoc_scan_segmented":
        # Gated (segmented) associative scan: h_t = a_t * h_{t-1} + b_t, h_{-1}=0.
        # The gate a_t encodes segment resets (a_t==0 -> h_t=b_t starts a new segment),
        # so this single first-order linear recurrence is the general segmented scan.
        def get_inputs(shape, device="cuda", seed=0):
            B, D, L = shape["B"], shape["D"], shape["L"]
            a = _rand01((B, D, L), device, seed)             # gate/decay in (0,1)
            b = _randn((B, D, L), device, seed + 1, scale=0.5)
            return (a, b)

        def _core(a, b):
            L = a.shape[-1]
            h = torch.zeros(a.shape[:-1], dtype=a.dtype, device=a.device)
            out = torch.empty_like(a)
            for t in range(L):
                h = a[..., t] * h + b[..., t]
                out[..., t] = h
            return out

        def ref_fn(a, b):
            return _core(a.float(), b.float()).to(a.dtype)

        def baseline_fn(a, b):
            return _core(a, b)

        arity = 2

    # ------------------------------------------------------- STATE-SPACE MODELS
    elif op == "selective_scan":
        # Mamba-1 selective SSM core (the canonical hard kernel), mirroring the
        # reference math of mamba_ssm.ops.selective_scan_ref (delta_softplus=True,
        # with the D skip, no z-gate). Layout: u/delta[B,L,D], A[D,N], B_/C[B,L,N],
        # D_[D] -> y[B,L,D].
        #   dt      = softplus(delta)                          (positive time step)
        #   dA      = exp(dt * A)                              (ZOH discretization of A)
        #   dBu     = dt * B_ * u                              (Euler discretization of B)
        #   h_l     = dA_l * h_{l-1} + dBu_l                   (per (b,d): state over N)
        #   y_l[d]  = sum_n C_l[n] * h_l[d,n]  +  D_[d]*u_l[d] (readout + skip)
        def get_inputs(shape, device="cuda", seed=0):
            B, L, D, N = shape["B"], shape["L"], shape["D"], shape["N"]
            u = _randn((B, L, D), device, seed)
            delta = _randn((B, L, D), device, seed + 1, scale=0.5)   # raw (pre-softplus)
            A = _neg_exp((D, N), device, seed + 2)                   # negative state matrix
            B_ = _randn((B, L, N), device, seed + 3)
            C = _randn((B, L, N), device, seed + 4)
            D_ = _randn((D,), device, seed + 5)
            return (u, delta, A, B_, C, D_)

        def _core(u, delta, A, B_, C, D_):
            Bs, L, D = u.shape
            N = A.shape[1]
            dt = F.softplus(delta)                              # [B,L,D]
            h = torch.zeros((Bs, D, N), dtype=u.dtype, device=u.device)
            y = torch.empty((Bs, L, D), dtype=u.dtype, device=u.device)
            for t in range(L):
                dt_t = dt[:, t]                                 # [B,D]
                dA = torch.exp(dt_t[:, :, None] * A)            # [B,D,N]  (A [D,N] bcast)
                dBu = dt_t[:, :, None] * B_[:, t, None, :] * u[:, t, :, None]  # [B,D,N]
                h = dA * h + dBu                                # [B,D,N]
                y[:, t] = (h * C[:, t, None, :]).sum(-1)        # [B,D]
            return y + u * D_                                   # skip (D_ [D] bcast)

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 6

    elif op == "ssd_chunk_scan":
        # Simplified Mamba-2 SSD (state-space duality) scalar-decay scan. Each step
        # applies a single SCALAR decay a_t in (0,1) to the whole [D,N] state, adds the
        # rank-1 input update x_t (outer) B_t, and reads out via C_t. A "chunked" SSD
        # kernel processes blocks of time but must reproduce this sequential recurrence.
        #   h_l = a_l * h_{l-1} + x_l (outer) B_l ;  y_l[d] = sum_n C_l[n] * h_l[d,n]
        # Layout: x[B,L,D], a[B,L], B_/C[B,L,N] -> y[B,L,D].
        def get_inputs(shape, device="cuda", seed=0):
            B, L, D, N = shape["B"], shape["L"], shape["D"], shape["N"]
            x = _randn((B, L, D), device, seed)
            a = _rand01((B, L), device, seed + 1)               # scalar decay in (0,1)
            B_ = _randn((B, L, N), device, seed + 2)
            C = _randn((B, L, N), device, seed + 3)
            return (x, a, B_, C)

        def _core(x, a, B_, C):
            Bs, L, D = x.shape
            N = B_.shape[-1]
            h = torch.zeros((Bs, D, N), dtype=x.dtype, device=x.device)
            y = torch.empty((Bs, L, D), dtype=x.dtype, device=x.device)
            for t in range(L):
                h = a[:, t, None, None] * h + x[:, t, :, None] * B_[:, t, None, :]  # [B,D,N]
                y[:, t] = (h * C[:, t, None, :]).sum(-1)        # [B,D]
            return y

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 4

    # -------------------------------------------------------- LINEAR ATTENTION
    elif op == "linear_attention":
        # Causal linear attention (unnormalized), feature map phi(x) = elu(x) + 1.
        # State S_t = sum_{s<=t} phi(k_s) (outer) v_s  (a [Dh_feat, Dh_val] matrix);
        # y_t = phi(q_t) . S_t == sum_{s<=t} (phi(q_t).phi(k_s)) v_s (the linear-time
        # dual of quadratic attention). Layout: q/k/v[B,H,L,Dh] -> y[B,H,L,Dh].
        def get_inputs(shape, device="cuda", seed=0):
            B, H, L, Dh = shape["B"], shape["H"], shape["L"], shape["Dh"]
            q = _randn((B, H, L, Dh), device, seed, scale=0.5)
            k = _randn((B, H, L, Dh), device, seed + 1, scale=0.5)
            v = _randn((B, H, L, Dh), device, seed + 2)
            return (q, k, v)

        def _core(q, k, v):
            Bs, H, L, Dh = q.shape
            pq = F.elu(q) + 1.0
            pk = F.elu(k) + 1.0
            S = torch.zeros((Bs, H, Dh, Dh), dtype=q.dtype, device=q.device)  # [feat, val]
            y = torch.empty((Bs, H, L, Dh), dtype=q.dtype, device=q.device)
            for t in range(L):
                S = S + pk[:, :, t, :, None] * v[:, :, t, None, :]  # [B,H,Dh,Dh]
                y[:, :, t] = (pq[:, :, t, :, None] * S).sum(-2)     # sum over feat -> [B,H,Dh]
            return y

        def ref_fn(*xs):
            return _core(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            return _core(*xs)

        arity = 3

    # ---------------------------------------------------------- CAUSAL CONV1D
    elif op == "causal_conv1d":
        # Depthwise causal 1D convolution (Mamba's short conv): each channel is
        # convolved with its own length-K kernel; left-pad by K-1 (no right pad) so
        # y[t] depends only on x[t-K+1 .. t]. Layout: x[B,D,L], weight[D,K], bias[D].
        def get_inputs(shape, device="cuda", seed=0):
            B, D, L, K = shape["B"], shape["D"], shape["L"], shape["K"]
            x = _randn((B, D, L), device, seed)
            weight = _randn((D, K), device, seed + 1, scale=1.0 / (K ** 0.5))
            bias = _randn((D,), device, seed + 2, scale=0.1)
            return (x, weight, bias)

        def _core(x, weight, bias):
            D = x.shape[1]
            K = weight.shape[1]
            xpad = F.pad(x, (K - 1, 0))                          # causal left pad
            return F.conv1d(xpad, weight[:, None, :], bias, groups=D)   # [B,D,L]

        def ref_fn(x, weight, bias):
            return _core(x.float(), weight.float(), bias.float()).to(x.dtype)

        def baseline_fn(x, weight, bias):
            return _core(x, weight, bias)

        arity = 3

    else:
        raise ValueError(f"unknown breadth op {op!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op,
          "dtype_name": dtype, "family": f"breadth_{op}", "mutates_input": False}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Naive (correct, compiling) Triton seeds - the policy's starting point.
# Each is a SEQUENTIAL scan: one program per row/channel keeps a running state and
# loops over the time axis (fp32 math, {tldt} store). Correct-but-slow; the policy
# is expected to replace the serial loop with a parallel prefix / chunked scan.
# {dtype} lands only in the docstring; {tldt} is the tl store dtype literal.
# --------------------------------------------------------------------------- #
_CUMSUM_SEED = '''"""GENERATED breadth cumsum seed ({dtype}). x[..., L] -> cumulative sum over the last dim.
One program per flattened row; a sequential fp32 running-sum scan over L (naive but
correct). The policy replaces the serial loop with a parallel prefix (Blelloch) scan.
{tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cumsum_kernel(x_ptr, y_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    acc = 0.0
    for i in range(0, L):
        v = tl.load(x_ptr + base + i).to(tl.float32)
        acc = acc + v
        tl.store(y_ptr + base + i, acc.to({tldt}))


def cumsum(x: torch.Tensor) -> torch.Tensor:
    L = x.shape[-1]
    xf = x.contiguous().reshape(-1, L)
    y = torch.empty_like(xf)
    _cumsum_kernel[(xf.shape[0],)](xf, y, L, xf.stride(0), num_warps=1)
    return y.reshape(x.shape)
'''

_CUMPROD_SEED = '''"""GENERATED breadth cumprod seed ({dtype}). x[..., L] -> cumulative product over last dim.
One program per flattened row; a sequential fp32 running-product scan over L (naive
but correct). The policy replaces the serial loop with a parallel prefix scan. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cumprod_kernel(x_ptr, y_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    acc = 1.0
    for i in range(0, L):
        v = tl.load(x_ptr + base + i).to(tl.float32)
        acc = acc * v
        tl.store(y_ptr + base + i, acc.to({tldt}))


def cumprod(x: torch.Tensor) -> torch.Tensor:
    L = x.shape[-1]
    xf = x.contiguous().reshape(-1, L)
    y = torch.empty_like(xf)
    _cumprod_kernel[(xf.shape[0],)](xf, y, L, xf.stride(0), num_warps=1)
    return y.reshape(x.shape)
'''

_ASSOC_SCAN_SEED = '''"""GENERATED breadth assoc_scan_segmented seed ({dtype}). Gated linear recurrence
h_t = a_t*h_{{t-1}} + b_t (h_{{-1}}=0) over the last dim. One program per flattened row;
sequential fp32 scan (naive but correct; the associative operator (a,b) makes this a
parallel prefix scan the policy is expected to build). Inputs a[...,L], b[...,L]. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _assoc_scan_segmented_kernel(a_ptr, b_ptr, h_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    h = 0.0
    for i in range(0, L):
        av = tl.load(a_ptr + base + i).to(tl.float32)
        bv = tl.load(b_ptr + base + i).to(tl.float32)
        h = av * h + bv
        tl.store(h_ptr + base + i, h.to({tldt}))


def assoc_scan_segmented(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    L = a.shape[-1]
    af = a.contiguous().reshape(-1, L)
    bf = b.contiguous().reshape(-1, L)
    h = torch.empty_like(af)
    _assoc_scan_segmented_kernel[(af.shape[0],)](af, bf, h, L, af.stride(0), num_warps=1)
    return h.reshape(a.shape)
'''

_SELECTIVE_SCAN_SEED = '''"""GENERATED breadth selective_scan seed ({dtype}). Mamba-1 selective SSM core.
u/delta[B,L,D], A[D,N], B_/C[B,L,N], D_[D] -> y[B,L,D]. One program per (b, d) keeps
an fp32 state h[N] and scans over L: dt=softplus(delta); dA=exp(dt*A); dBu=dt*B_*u;
h=dA*h+dBu; y=sum_n C*h + D_*u. Naive sequential scan (the policy fuses/chunks it). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _selective_scan_kernel(u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, Dskip_ptr, y_ptr,
                           L, D, N,
                           su_b, su_l, su_d, sA_d, sA_n, sB_b, sB_l, sB_n, sDs,
                           NB: tl.constexpr):
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
        dt = tl.where(dt > 20.0, dt, tl.log(1.0 + tl.exp(dt)))   # softplus
        off_ln = b * sB_b + l * sB_l + n * sB_n
        Bv = tl.load(B_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        Cv = tl.load(C_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        dA = tl.exp(dt * Arow)
        dBu = dt * Bv * u_v
        h = dA * h + dBu
        y_v = tl.sum(tl.where(nmask, Cv * h, 0.0), axis=0) + Dd * u_v
        tl.store(y_ptr + off_ld, y_v.to({tldt}))


def selective_scan(u, delta, A, B_, C, D_):
    Bsz, L, D = u.shape
    N = A.shape[1]
    u = u.contiguous(); delta = delta.contiguous(); A = A.contiguous()
    B_ = B_.contiguous(); C = C.contiguous(); D_ = D_.contiguous()
    y = torch.empty_like(u)
    NB = triton.next_power_of_2(N)
    _selective_scan_kernel[(Bsz * D,)](
        u, delta, A, B_, C, D_, y, L, D, N,
        u.stride(0), u.stride(1), u.stride(2),
        A.stride(0), A.stride(1),
        B_.stride(0), B_.stride(1), B_.stride(2),
        D_.stride(0), NB=NB, num_warps=1)
    return y
'''

_SSD_SEED = '''"""GENERATED breadth ssd_chunk_scan seed ({dtype}). Simplified Mamba-2 SSD scalar-decay scan.
x[B,L,D], a[B,L] (scalar decay), B_/C[B,L,N] -> y[B,L,D]. One program per (b, d) keeps an
fp32 state h[N] and scans over L: h = a*h + x*B_ ; y = sum_n C*h. Naive sequential scan; a
real SSD kernel processes time in chunks (matmul the intra-chunk term). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssd_chunk_scan_kernel(x_ptr, a_ptr, B_ptr, C_ptr, y_ptr, L, D, N,
                           sx_b, sx_l, sx_d, sa_b, sa_l, sB_b, sB_l, sB_n,
                           NB: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D
    n = tl.arange(0, NB)
    nmask = n < N
    h = tl.zeros([NB], dtype=tl.float32)
    for l in range(0, L):
        a_v = tl.load(a_ptr + b * sa_b + l * sa_l).to(tl.float32)
        off_ld = b * sx_b + l * sx_l + d * sx_d
        x_v = tl.load(x_ptr + off_ld).to(tl.float32)
        off_ln = b * sB_b + l * sB_l + n * sB_n
        Bv = tl.load(B_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        Cv = tl.load(C_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        h = a_v * h + x_v * Bv
        y_v = tl.sum(tl.where(nmask, Cv * h, 0.0), axis=0)
        tl.store(y_ptr + off_ld, y_v.to({tldt}))


def ssd_chunk_scan(x, a, B_, C):
    Bsz, L, D = x.shape
    N = B_.shape[-1]
    x = x.contiguous(); a = a.contiguous(); B_ = B_.contiguous(); C = C.contiguous()
    y = torch.empty_like(x)
    NB = triton.next_power_of_2(N)
    _ssd_chunk_scan_kernel[(Bsz * D,)](
        x, a, B_, C, y, L, D, N,
        x.stride(0), x.stride(1), x.stride(2),
        a.stride(0), a.stride(1),
        B_.stride(0), B_.stride(1), B_.stride(2), NB=NB, num_warps=1)
    return y
'''

_LINEAR_ATTENTION_SEED = '''"""GENERATED breadth linear_attention seed ({dtype}). Causal linear attention, phi=elu+1.
q/k/v[B,H,L,Dh] -> y[B,H,L,Dh]. One program per (b, h, e) keeps an fp32 state column s[Dh]
(= S[:, e]) and scans over L: s += phi(k_l) * v_l[e]; y_l[e] = sum_d phi(q_l)[d] * s[d].
Naive sequential scan (the policy chunks/parallelizes it). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _linear_attention_kernel(q_ptr, k_ptr, v_ptr, y_ptr, H, L, Dh,
                             s_b, s_h, s_l, s_d, DB: tl.constexpr):
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
        krow = tl.load(k_ptr + bh + l * s_l + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        phik = tl.where(krow > 0.0, krow + 1.0, tl.exp(krow))     # elu(k)+1
        v_le = tl.load(v_ptr + bh + l * s_l + e * s_d).to(tl.float32)
        s = s + phik * v_le
        qrow = tl.load(q_ptr + bh + l * s_l + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        phiq = tl.where(qrow > 0.0, qrow + 1.0, tl.exp(qrow))     # elu(q)+1
        y_le = tl.sum(tl.where(dmask, phiq * s, 0.0), axis=0)
        tl.store(y_ptr + bh + l * s_l + e * s_d, y_le.to({tldt}))


def linear_attention(q, k, v):
    Bsz, H, L, Dh = q.shape
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    y = torch.empty_like(q)
    DB = triton.next_power_of_2(Dh)
    _linear_attention_kernel[(Bsz * H * Dh,)](
        q, k, v, y, H, L, Dh,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), DB=DB, num_warps=1)
    return y
'''

_CAUSAL_CONV1D_SEED = '''"""GENERATED breadth causal_conv1d seed ({dtype}). Depthwise causal 1D conv (Mamba short conv).
x[B,D,L], weight[D,K], bias[D] -> y[B,D,L]. One program per (b, d); for each output time t,
y[t] = bias + sum_k weight[k] * x[t-(K-1)+k] (left-causal, x=0 for t<0). fp32 accumulate,
{tldt} store. Naive per-time loop; the policy vectorizes over the time axis."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _causal_conv1d_kernel(x_ptr, w_ptr, b_ptr, y_ptr, D, L, K,
                          sx_b, sx_d, sx_l, sw_d, sw_k, sb_d, KB: tl.constexpr):
    pid = tl.program_id(0)
    bb = pid // D
    d = pid % D
    kk = tl.arange(0, KB)
    kmask = kk < K
    wrow = tl.load(w_ptr + d * sw_d + kk * sw_k, mask=kmask, other=0.0).to(tl.float32)
    bias = tl.load(b_ptr + d * sb_d).to(tl.float32)
    base = bb * sx_b + d * sx_d
    for t in range(0, L):
        idx = t - (K - 1) + kk                                   # [KB] input positions
        vmask = kmask & (idx >= 0)
        xv = tl.load(x_ptr + base + idx * sx_l, mask=vmask, other=0.0).to(tl.float32)
        acc = tl.sum(tl.where(vmask, wrow * xv, 0.0), axis=0) + bias
        tl.store(y_ptr + base + t * sx_l, acc.to({tldt}))


def causal_conv1d(x, weight, bias):
    Bsz, D, L = x.shape
    K = weight.shape[1]
    x = x.contiguous(); weight = weight.contiguous(); bias = bias.contiguous()
    y = torch.empty_like(x)
    KB = triton.next_power_of_2(K)
    _causal_conv1d_kernel[(Bsz * D,)](
        x, weight, bias, y, D, L, K,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1), bias.stride(0), KB=KB, num_warps=1)
    return y
'''


def seed_source(op: str, dtype: str) -> str:
    tldt = DTYPES[dtype][1]
    if op == "cumsum":
        return _CUMSUM_SEED.format(dtype=dtype, tldt=tldt)
    if op == "cumprod":
        return _CUMPROD_SEED.format(dtype=dtype, tldt=tldt)
    if op == "assoc_scan_segmented":
        return _ASSOC_SCAN_SEED.format(dtype=dtype, tldt=tldt)
    if op == "selective_scan":
        return _SELECTIVE_SCAN_SEED.format(dtype=dtype, tldt=tldt)
    if op == "ssd_chunk_scan":
        return _SSD_SEED.format(dtype=dtype, tldt=tldt)
    if op == "linear_attention":
        return _LINEAR_ATTENTION_SEED.format(dtype=dtype, tldt=tldt)
    if op == "causal_conv1d":
        return _CAUSAL_CONV1D_SEED.format(dtype=dtype, tldt=tldt)
    raise ValueError(f"unknown breadth op {op!r}")


def op_names() -> list[str]:
    return list(OPS)
