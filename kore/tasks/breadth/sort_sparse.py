"""Breadth task engine: SORT/SELECT + SPARSE operator families (torch-baselined).

These widen the KORE suite past the vendor-baselined core with two high-value
classes the block interior never covered:

  * SORT / SELECT  - the sampling / routing tail of a transformer: top-k values,
    argmax, full row sort, and nucleus (top-p) renormalization. Data-dependent
    control flow (the honest "hard for a GPU" class).
  * SPARSE          - structured/unstructured sparse GEMM: 2:4 structured weights,
    block-sparse GEMM, sparse@dense (SpMM), and sampled dense-dense (SDDMM). The
    reference is a DENSE torch compute over a (dense tensor + mask) pair so the
    grade stays SNR-comparable (no CSR/index plumbing in the oracle).

Contract (mirrors kore.tasks.vendor_ops so the generic _genops driver works):

  * ``make_reference(op, dtype) -> ns`` builds the reference.py namespace
    (parse_shape / get_inputs / ref_fn fp32 oracle / baseline_fn torch path /
    arity / entry_name / dtype_name / family=``breadth_{op}``).
  * ``seed_source(op, dtype) -> str`` returns a COMPILING Triton starter kernel
    (the policy's optimization seed).

CORRECTNESS of the reference is paramount and exact (fp32 oracle). The naive
Triton seeds must COMPILE and define the entry fn; where the op has genuine
data-dependent control flow (sort / top-p) the seed is a simple-but-correct
selection loop (O(N^2) per row) that the teacher/policy is expected to replace
with a real bitonic/partial sort. The structured-sparse seeds do the (data-
dependent) mask selection host-side in torch and run the GEMM in Triton - a
correct, honest partial-fusion starting point.

torch/triton are imported lazily (registry discovery never needs a GPU).
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# --------------------------------------------------------------------------- #
# op set + dtype sweep
# --------------------------------------------------------------------------- #
OPS: tuple[str, ...] = (
    # sort / select
    "topk_values", "argmax_lastdim", "sort_lastdim", "topp_mask",
    # sparse
    "sparse_2to4_apply", "block_sparse_matmul", "spmm_csr", "sddmm",
)

# breadth tasks are torch-baselined (the honest eager/compile bar), swept over the
# two serving activation dtypes - same default as the vendor plain-float ops.
DEFAULT_DTYPES: tuple[str, ...] = ("bf16", "fp16")

# per-op dtype override (all ops use the default here; kept as the vendor-style ABI
# hook so a future op can pin its own sweep). Fully materialized for direct use.
OP_DTYPES: dict[str, tuple[str, ...]] = {op: DEFAULT_DTYPES for op in OPS}


def op_dtypes(op: str) -> tuple[str, ...]:
    """The dtype sweep for a breadth op (per-op override or the global default)."""
    return OP_DTYPES.get(op, DEFAULT_DTYPES)


# --------------------------------------------------------------------------- #
# op hyper-params (baked into both the fp32 oracle and the seed defaults)
# --------------------------------------------------------------------------- #
TOPK_K = 8            # top-k values width
TOPP_P = 0.9          # nucleus (top-p) mass threshold
SP_GROUP = 4          # 2:4 structured sparsity group size (keep 2 of every 4)
SP_KEEP = 2
BLK_K = 32            # block-sparse GEMM block size along K
BLK_N = 32            # block-sparse GEMM block size along N
SPMM_DENSITY = 0.3    # fraction of nonzeros in the SpMM sparse operand
SDDMM_DENSITY = 0.25  # fraction of sampled (kept) entries in SDDMM
BLOCK_DENSITY = 0.5   # fraction of active blocks in block-sparse GEMM

# --------------------------------------------------------------------------- #
# Realistic shapes per op (minimal for smoke, primary for the headline grade,
# validation for the sweep). Row ops: x[M, N] over the last dim. GEMM ops:
# x/a[M, K] @ w/b[K, N] -> [M, N].
# --------------------------------------------------------------------------- #
_TOPK_SHAPES = {  # top-k over a wide row (attention logits / vocab shard)
    "minimal": {"M": 64, "N": 256},
    "primary": {"M": 4096, "N": 8192},
    "validation": [{"M": 8192, "N": 4096}, {"M": 2048, "N": 32768},
                   {"M": 4096, "N": 8191}],   # large vocab, non-pow2 tail
}
_ARGMAX_SHAPES = {  # argmax over a wide row (router / classifier)
    "minimal": {"M": 64, "N": 256},
    "primary": {"M": 4096, "N": 8192},
    "validation": [{"M": 8192, "N": 4096}, {"M": 2048, "N": 32768},
                   {"M": 4096, "N": 8191}],
}
_SORT_SHAPES = {  # full ascending row sort (moderate width; naive seed is O(N^2))
    "minimal": {"M": 64, "N": 128},
    "primary": {"M": 2048, "N": 2048},
    "validation": [{"M": 4096, "N": 1024}, {"M": 1024, "N": 4096},
                   {"M": 2048, "N": 2047}],   # non-pow2 tail
}
_TOPP_SHAPES = {  # nucleus renormalization over logits (moderate width; O(N^2) seed)
    "minimal": {"M": 64, "N": 256},
    "primary": {"M": 2048, "N": 4096},
    "validation": [{"M": 4096, "N": 8192}, {"M": 1024, "N": 8192},
                   {"M": 2048, "N": 4095}],   # non-pow2 tail
}
_S24_SHAPES = {  # 2:4 structured-sparse weight GEMM (N multiple of 4)
    "minimal": {"M": 64, "K": 128, "N": 128},
    "primary": {"M": 4096, "K": 4096, "N": 4096},
    "validation": [{"M": 2048, "K": 8192, "N": 4096}, {"M": 8192, "K": 1024, "N": 2048},
                   {"M": 4096, "K": 4096, "N": 4092}],   # N%4==0 non-pow2 tail
}
_BLOCK_SHAPES = {  # block-sparse GEMM (K,N multiples of BLK_K/BLK_N = 32)
    "minimal": {"M": 64, "K": 128, "N": 128},
    "primary": {"M": 4096, "K": 4096, "N": 4096},
    "validation": [{"M": 2048, "K": 2048, "N": 4096}, {"M": 1024, "K": 4096, "N": 2048},
                   {"M": 4096, "K": 2048, "N": 2048}],
}
_SPMM_SHAPES = {  # sparse(A) @ dense(B); A carries an element-wise mask
    "minimal": {"M": 64, "K": 128, "N": 128},
    "primary": {"M": 4096, "K": 4096, "N": 4096},
    "validation": [{"M": 2048, "K": 8192, "N": 4096}, {"M": 8192, "K": 1024, "N": 2048},
                   {"M": 4096, "K": 4095, "N": 4096}],   # non-pow2 K
}
_SDDMM_SHAPES = {  # sampled dense-dense: mask ⊙ (A @ B); mask over the [M,N] output
    "minimal": {"M": 64, "K": 128, "N": 128},
    "primary": {"M": 4096, "K": 4096, "N": 4096},
    "validation": [{"M": 2048, "K": 4096, "N": 4096}, {"M": 8192, "K": 1024, "N": 2048},
                   {"M": 4096, "K": 2048, "N": 4095}],   # non-pow2 N
}

SHAPES: dict[str, dict] = {
    "topk_values": _TOPK_SHAPES, "argmax_lastdim": _ARGMAX_SHAPES,
    "sort_lastdim": _SORT_SHAPES, "topp_mask": _TOPP_SHAPES,
    "sparse_2to4_apply": _S24_SHAPES, "block_sparse_matmul": _BLOCK_SHAPES,
    "spmm_csr": _SPMM_SHAPES, "sddmm": _SDDMM_SHAPES,
}


# --------------------------------------------------------------------------- #
# torch helpers (shared by fp32 oracle + torch baseline)
# --------------------------------------------------------------------------- #
def _sparsify_2to4_lastdim(w):
    """Keep the 2 largest-magnitude of every contiguous group of 4 along the last
    dim (zero the other 2). ``w`` last dim must be a multiple of 4. Ties are
    measure-zero for continuous inputs (top-k picks a well-defined set)."""
    import torch
    *lead, L = w.shape
    g = w.reshape(*lead, L // SP_GROUP, SP_GROUP)
    keep_idx = g.abs().topk(SP_KEEP, dim=-1).indices
    keep = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, keep_idx, True)
    return torch.where(keep, g, torch.zeros_like(g)).reshape(*lead, L)


def _apply_block_mask(w, mask):
    """Expand a block mask [K//BLK_K, N//BLK_N] over w[K, N] (zero inactive blocks)."""
    K, N = w.shape
    Kb, Nb = mask.shape
    bk, bn = K // Kb, N // Nb
    wm = w.reshape(Kb, bk, Nb, bn) * mask.reshape(Kb, 1, Nb, 1).to(w.dtype)
    return wm.reshape(K, N)


def _nucleus(probs, p):
    """Top-p (nucleus) renormalization of a probability row. Keep the smallest
    descending prefix whose EXCLUSIVE cumulative mass <= p (i.e. always keep the
    crossing token), zero the rest, renormalize to sum 1. Matches the standard
    HF top-p mask (shift-right of cumsum > p)."""
    import torch
    sp, si = torch.sort(probs, dim=-1, descending=True)
    excl = sp.cumsum(dim=-1) - sp                 # exclusive prefix mass
    keep_sorted = excl <= p
    keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, si, keep_sorted)
    masked = torch.where(keep, probs, torch.zeros_like(probs))
    return masked / masked.sum(dim=-1, keepdim=True)


# --------------------------------------------------------------------------- #
# reference.py namespace (torch fp32 oracle + torch baseline)
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import torch

    tdt = getattr(torch, DTYPES[dtype][0])

    def _randn(shape, device, seed, scale=1.0):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape, generator=g, device=device, dtype=torch.float32) * scale).to(tdt)

    def _mask(shape, device, seed, density):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.rand(shape, generator=g, device=device, dtype=torch.float32) < density).to(tdt)

    # -------------------- SORT / SELECT ------------------------------------- #
    if op == "topk_values":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["N"]), device, seed, scale=2.0),)

        def ref_fn(x):
            return torch.topk(x.float(), TOPK_K, dim=-1).values.to(x.dtype)

        def baseline_fn(x):
            return torch.topk(x, TOPK_K, dim=-1).values

        arity = 1

    elif op == "argmax_lastdim":
        # SNR-safe: return the max VALUE per row (== value gathered at argmax), not
        # the integer index (indices are not SNR/allclose-comparable and are
        # ill-defined under ties).
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["N"]), device, seed, scale=2.0),)

        def ref_fn(x):
            return x.float().amax(dim=-1).to(x.dtype)

        def baseline_fn(x):
            return x.amax(dim=-1)

        arity = 1

    elif op == "sort_lastdim":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["N"]), device, seed),)

        def ref_fn(x):
            return torch.sort(x.float(), dim=-1).values.to(x.dtype)

        def baseline_fn(x):
            return torch.sort(x, dim=-1).values

        arity = 1

    elif op == "topp_mask":
        def get_inputs(shape, device="cuda", seed=0):
            return (_randn((shape["M"], shape["N"]), device, seed, scale=2.0),)  # logits

        def ref_fn(x):
            probs = torch.softmax(x.float(), dim=-1)
            return _nucleus(probs, TOPP_P).to(x.dtype)

        def baseline_fn(x):
            probs = torch.softmax(x.float(), dim=-1)
            return _nucleus(probs, TOPP_P).to(x.dtype)

        arity = 1

    # -------------------- SPARSE -------------------------------------------- #
    elif op == "sparse_2to4_apply":
        def get_inputs(shape, device="cuda", seed=0):
            M, K, N = shape["M"], shape["K"], shape["N"]
            x = _randn((M, K), device, seed)
            w = _randn((K, N), device, seed + 1, scale=1.0 / (K ** 0.5))
            return (x, w)

        def ref_fn(x, w):
            ws = _sparsify_2to4_lastdim(w.float())
            return (x.float() @ ws).to(x.dtype)

        def baseline_fn(x, w):
            return x @ _sparsify_2to4_lastdim(w)

        arity = 2

    elif op == "block_sparse_matmul":
        def get_inputs(shape, device="cuda", seed=0):
            M, K, N = shape["M"], shape["K"], shape["N"]
            x = _randn((M, K), device, seed)
            w = _randn((K, N), device, seed + 1, scale=1.0 / (K ** 0.5))
            mask = _mask((K // BLK_K, N // BLK_N), device, seed + 2, BLOCK_DENSITY)
            return (x, w, mask)

        def ref_fn(x, w, mask):
            wm = _apply_block_mask(w.float(), mask.float())
            return (x.float() @ wm).to(x.dtype)

        def baseline_fn(x, w, mask):
            return x @ _apply_block_mask(w, mask)

        arity = 3

    elif op == "spmm_csr":
        # Sparse(A) @ dense(B), A represented DENSE + an element-wise mask (so the
        # oracle is a plain masked matmul, SNR-comparable - no CSR index plumbing).
        def get_inputs(shape, device="cuda", seed=0):
            M, K, N = shape["M"], shape["K"], shape["N"]
            a = _randn((M, K), device, seed)
            b = _randn((K, N), device, seed + 1, scale=1.0 / (K ** 0.5))
            mask = _mask((M, K), device, seed + 2, SPMM_DENSITY)
            return (a, b, mask)

        def ref_fn(a, b, mask):
            am = a.float() * mask.float()
            return (am @ b.float()).to(a.dtype)

        def baseline_fn(a, b, mask):
            return (a * mask) @ b

        arity = 3

    elif op == "sddmm":
        # Sampled dense-dense: C = mask ⊙ (A @ B), mask over the [M,N] output.
        def get_inputs(shape, device="cuda", seed=0):
            M, K, N = shape["M"], shape["K"], shape["N"]
            a = _randn((M, K), device, seed, scale=1.0 / (K ** 0.25))
            b = _randn((K, N), device, seed + 1, scale=1.0 / (K ** 0.25))
            mask = _mask((M, N), device, seed + 2, SDDMM_DENSITY)
            return (a, b, mask)

        def ref_fn(a, b, mask):
            return ((a.float() @ b.float()) * mask.float()).to(a.dtype)

        def baseline_fn(a, b, mask):
            return (a @ b) * mask

        arity = 3

    else:
        raise ValueError(f"unknown breadth op {op!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op,
          "dtype_name": dtype, "family": f"breadth_{op}", "mutates_input": False}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Triton starter seeds (COMPILING; the policy optimizes these).
# {dtype} lands only in the docstring; {tldt} is the tl store dtype literal.
# --------------------------------------------------------------------------- #
_TOPK_VALUES_SEED = '''"""GENERATED breadth top-k values seed ({dtype}). x[M,N] -> top-k values per row.
Naive: load the row, iteratively pull the running max K times (tl.max/argmax with
masking of the just-taken lane). O(K*N) - cheap - and CORRECT for the returned
VALUES (descending, ties are value-identical). {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _topk_values_kernel(x_ptr, o_ptr, sx, so, N, K, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
    for k in range(0, K):
        v = tl.max(x, axis=0)
        j = tl.argmax(x, axis=0)
        tl.store(o_ptr + row * so + k, v.to({tldt}))
        x = tl.where(offs == j, -float("inf"), x)


def topk_values(x: torch.Tensor, k: int = 8) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M, k), device=x.device, dtype=x.dtype)
    _topk_values_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, k,
                              BLOCK_N=triton.next_power_of_2(N), num_warps=4)
    return o
'''


_ARGMAX_SEED = '''"""GENERATED breadth argmax-lastdim seed ({dtype}). x[M,N] -> max value per row.
One program/row: fp32 masked load, tl.max reduction (== value at argmax), {tldt}
store. SNR-safe (returns the VALUE, not the index)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _argmax_lastdim_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
    v = tl.max(x, axis=0)
    tl.store(o_ptr + row, v.to({tldt}))


def argmax_lastdim(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=x.dtype)
    _argmax_lastdim_kernel[(M,)](x, o, x.stride(0), N,
                                 BLOCK_N=triton.next_power_of_2(N), num_warps=4)
    return o
'''


_SORT_SEED = '''"""GENERATED breadth row-sort seed ({dtype}). x[M,N] -> ascending sorted rows.
NAIVE + CORRECT selection sort: load the row (pad with +inf), N times pull the
running min into the next output slot and mask it out. O(N^2)/row - a partial
starting point; the teacher is expected to replace it with a bitonic sort. {tldt}."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _sort_lastdim_kernel(x_ptr, o_ptr, sx, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=float("inf")).to(tl.float32)
    for i in range(0, N):
        v = tl.min(x, axis=0)
        j = tl.argmin(x, axis=0)
        tl.store(o_ptr + row * so + i, v.to({tldt}))
        x = tl.where(offs == j, float("inf"), x)


def sort_lastdim(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    _sort_lastdim_kernel[(M,)](x, o, x.stride(0), o.stride(0), N,
                               BLOCK_N=triton.next_power_of_2(N), num_warps=4)
    return o
'''


_TOPP_SEED = '''"""GENERATED breadth nucleus (top-p) seed ({dtype}). logits[M,N] -> renormalized
top-p probabilities. Pass 1 fp32 softmax; then a selection loop keeps tokens in
descending prob order while the EXCLUSIVE cumulative mass <= p (arithmetic masks,
no data-dependent branch), then renormalizes the kept set. O(N^2)/row - a correct
partial seed the teacher is expected to replace with a real top-p. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _topp_mask_kernel(x_ptr, o_ptr, sx, so, N, P, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.where(mask, tl.exp(x - m), 0.0)
    probs = e / tl.sum(e, axis=0)
    work = tl.where(mask, probs, -1.0)
    keep = tl.zeros([BLOCK_N], dtype=tl.float32)
    cum = 0.0
    for i in range(0, N):
        v = tl.max(work, axis=0)
        j = tl.argmax(work, axis=0)
        tk = tl.where(cum <= P, 1.0, 0.0)
        is_j = tl.where(offs == j, 1.0, 0.0)
        keep = keep + is_j * tk
        cum = cum + v * tk
        work = tl.where(offs == j, -1.0, work)
    kp = probs * keep
    denom = tl.sum(kp, axis=0)
    tl.store(o_ptr + row * so + offs, (kp / denom).to({tldt}), mask=mask)


def topp_mask(x: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    _topp_mask_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, float(p),
                            BLOCK_N=triton.next_power_of_2(N), num_warps=4)
    return o
'''


# Shared plain-GEMM prelude (docstring + kernel + launcher). The structured-sparse
# ops build their sparse operand host-side (data-dependent selection) then call
# this Triton GEMM - a correct, compiling partial-fusion seed.
_GEMM_PRELUDE = '''"""GENERATED breadth {optag} seed ({dtype}). Correct partial-fusion starting point:
the (data-dependent) sparsity selection is done host-side in torch, the dense GEMM
runs in this Triton kernel (fp32 accumulate, {tldt} store). 2D tiling + K-mask
(ROCm/CDNA-safe). The teacher is expected to fuse the masking into the kernel."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                 sam, sak, sbk, sbn, scm, scn,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        krem = K - k * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < krem), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < krem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    cmask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to({tldt}), mask=cmask)


def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _gemm_kernel[grid](a, b, c, M, N, K,
                       a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                       c.stride(0), c.stride(1),
                       BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2)
    return c
'''


_S24_WRAP = '''

def sparse_2to4_apply(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    K, N = w.shape
    g = w.reshape(K, N // 4, 4)
    idx = g.abs().topk(2, dim=-1).indices
    keep = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx, True)
    ws = torch.where(keep, g, torch.zeros_like(g)).reshape(K, N).contiguous()
    return _gemm(x, ws)
'''


_BLOCK_WRAP = '''

def block_sparse_matmul(x: torch.Tensor, w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    K, N = w.shape
    Kb, Nb = mask.shape
    bk, bn = K // Kb, N // Nb
    wm = (w.reshape(Kb, bk, Nb, bn) * mask.reshape(Kb, 1, Nb, 1).to(w.dtype)).reshape(K, N).contiguous()
    return _gemm(x, wm)
'''


_SPMM_WRAP = '''

def spmm_csr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    am = (a * mask.to(a.dtype)).contiguous()
    return _gemm(am, b)
'''


# SDDMM fuses the output mask into the GEMM epilogue (mask is over the [M,N] out).
_SDDMM_SEED = '''"""GENERATED breadth sddmm seed ({dtype}). C = mask ⊙ (A @ B), mask over [M,N].
Dense GEMM (fp32 accumulate) with the sampling mask applied in the epilogue before
the {tldt} store. Correct + compiling; a real SDDMM would skip masked-out tiles."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _sddmm_kernel(a_ptr, b_ptr, m_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, smm, smn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        krem = K - k * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < krem), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < krem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    cmask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    mm = tl.load(m_ptr + offs_m[:, None] * smm + offs_n[None, :] * smn,
                 mask=cmask, other=0.0).to(tl.float32)
    acc = acc * mm
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, acc.to({tldt}), mask=cmask)


def sddmm(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _sddmm_kernel[grid](a, b, mask, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        mask.stride(0), mask.stride(1), c.stride(0), c.stride(1),
                        BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2)
    return c
'''


def seed_source(op: str, dtype: str) -> str:
    tldt = DTYPES[dtype][1]
    if op == "topk_values":
        return _TOPK_VALUES_SEED.format(dtype=dtype, tldt=tldt)
    if op == "argmax_lastdim":
        return _ARGMAX_SEED.format(dtype=dtype, tldt=tldt)
    if op == "sort_lastdim":
        return _SORT_SEED.format(dtype=dtype, tldt=tldt)
    if op == "topp_mask":
        return _TOPP_SEED.format(dtype=dtype, tldt=tldt)
    if op == "sparse_2to4_apply":
        return (_GEMM_PRELUDE.format(optag="2:4 sparse GEMM", dtype=dtype, tldt=tldt)
                + _S24_WRAP)
    if op == "block_sparse_matmul":
        return (_GEMM_PRELUDE.format(optag="block-sparse GEMM", dtype=dtype, tldt=tldt)
                + _BLOCK_WRAP)
    if op == "spmm_csr":
        return (_GEMM_PRELUDE.format(optag="SpMM (sparse@dense)", dtype=dtype, tldt=tldt)
                + _SPMM_WRAP)
    if op == "sddmm":
        return _SDDMM_SEED.format(dtype=dtype, tldt=tldt)
    raise ValueError(f"unknown breadth op {op!r}")


def op_names() -> list[str]:
    return list(OPS)
