"""Vendor-baselined task authoring engine: hard ops graded against REAL AITER kernels.

Unlike the generated elementwise/fusion tasks (torch-eager baseline), these tasks
grade the policy against the ACTUAL production vendor kernel AMD's serving stack
calls — ``aiter.rms_norm`` / ``aiter.layer_norm`` / ``aiter.silu_and_mul`` /
``aiter.gelu_tanh_and_mul`` — the honest "beat the vendor library" bar. Each task
is authored semi-automatically from a per-op template + a model-shape/dtype sweep,
so the vendor-baselined suite scales without hundreds of hand-written files.

Contract (matches _genops so the generic driver works): make_vendor_reference()
returns the reference.py namespace (parse_shape/get_inputs/ref_fn oracle/baseline_fn
AITER/arity/entry_name); vendor_seed_source() returns a REAL Triton starter kernel;
the driver is the shared kore.tasks._genops.driver_main.

torch/aiter imported lazily (registry discovery never needs a GPU/aiter).
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# op -> family metadata; each op has a bespoke oracle/baseline/seed (below).
VENDOR_OPS: tuple[str, ...] = ("rmsnorm", "layernorm", "silu_mul", "gelu_mul",
                               "softmax", "gemm_a8w8", "fused_add_rmsnorm", "rope",
                               "topk_softmax", "batched_gemm", "gemm_a8w8_blockscale",
                               # v3 frontier additions (RoPE variants; torch.compile bar)
                               "rope_gptj", "rope_partial")

# ops whose vendor BASELINE mutates its inputs in place (so the bench loop must
# feed a fresh clone each timed call — see _genops._run_bench mutates_input path).
VENDOR_MUTATES_INPUT: frozenset[str] = frozenset({"fused_add_rmsnorm"})

# Real production shapes (hidden dims / gated-MLP widths) per op class, per the
# KORE-Bench blueprint (Llama-3 / Qwen3 / Mixtral / DeepSeek-V3).
_NORM_SHAPES = {  # x[M, N] ; N = hidden
    "minimal": {"M": 64, "N": 2048},
    "primary": {"M": 4096, "N": 8192},
    "validation": [{"M": 8192, "N": 4096}, {"M": 2048, "N": 7168},
                   {"M": 4096, "N": 8191}],   # DeepSeek hidden + non-pow2 tail
}
_GATE_SHAPES = {  # x[M, 2*inter] ; N = 2*inter (input width)
    "minimal": {"M": 64, "N": 1024},
    "primary": {"M": 4096, "N": 28672},        # Llama-3 8B MLP: 2*14336
    "validation": [{"M": 8192, "N": 22016}, {"M": 2048, "N": 8192},
                   {"M": 4096, "N": 28670}],   # 2*11008, small, non-pow2 tail
}

_SOFTMAX_SHAPES = {  # x[M, N] ; softmax over N (attention logits / vocab rows)
    "minimal": {"M": 64, "N": 1024},
    "primary": {"M": 8192, "N": 8192},
    "validation": [{"M": 4096, "N": 32768}, {"M": 16384, "N": 2048},
                   {"M": 8192, "N": 8191}],   # large vocab, wide batch, non-pow2 tail
}
_FP8_GEMM_SHAPES = {  # XQ[M,K] @ WQ[N,K]^T -> [M,N] bf16 (fp8 a8w8 serving GEMM)
    "minimal": {"M": 128, "N": 128, "K": 256},
    "primary": {"M": 4096, "N": 4096, "K": 4096},
    "validation": [{"M": 8192, "N": 8192, "K": 1024}, {"M": 2048, "N": 14336, "K": 8192},
                   {"M": 4096, "N": 4096, "K": 4095}],  # MLP up-proj, decode, non-pow2 K
}

_ROPE_SHAPES = {  # x[S,B,H,D] NEOX rotary embedding; freqs[S,1,1,D//2] angles
    "minimal": {"S": 128, "B": 1, "H": 8, "D": 64},
    "primary": {"S": 4096, "B": 1, "H": 32, "D": 128},   # Llama-3 8B attention
    "validation": [{"S": 2048, "B": 2, "H": 32, "D": 128}, {"S": 8192, "B": 1, "H": 40, "D": 128},
                   {"S": 4096, "B": 1, "H": 32, "D": 64}],  # batched, GQA-wide, half head-dim
}

_TOPK_SHAPES = {  # gate[M,E] MoE router; softmax over E experts -> top-k -> renorm
    "minimal": {"M": 64, "E": 8, "topk": 2},
    "primary": {"M": 4096, "E": 32, "topk": 8},          # 32-expert router, top-8
    "validation": [{"M": 8192, "E": 8, "topk": 2},        # Mixtral 8x, top-2
                   {"M": 2048, "E": 64, "topk": 8},
                   {"M": 4096, "E": 256, "topk": 8}],     # DeepSeek-V3 256-expert, top-8
}

_BLOCKSCALE_SHAPES = {  # DeepSeek-V3 block-scaled fp8 GEMM; N,K multiples of 128
    "minimal": {"M": 128, "N": 256, "K": 256},
    "primary": {"M": 4096, "N": 4096, "K": 4096},
    "validation": [{"M": 8192, "N": 8192, "K": 1024}, {"M": 2048, "N": 14336, "K": 4096},
                   {"M": 4095, "N": 512, "K": 512}],  # decode, MLP up-proj, non-aligned M
}

_BATCHED_GEMM_SHAPES = {  # A[B,M,K] @ B[B,N,K]^T -> [B,M,N] bf16 (batched attn/MoE proj)
    "minimal": {"B": 2, "M": 128, "N": 128, "K": 256},
    "primary": {"B": 8, "M": 512, "N": 512, "K": 512},
    "validation": [{"B": 16, "M": 256, "N": 256, "K": 1024},
                   {"B": 4, "M": 1024, "N": 1024, "K": 512},
                   {"B": 8, "M": 512, "N": 513, "K": 512}],  # batched, wide, non-pow2 N
}

VENDOR_SHAPES = {"rmsnorm": _NORM_SHAPES, "layernorm": _NORM_SHAPES,
                 "silu_mul": _GATE_SHAPES, "gelu_mul": _GATE_SHAPES,
                 "softmax": _SOFTMAX_SHAPES, "gemm_a8w8": _FP8_GEMM_SHAPES,
                 "fused_add_rmsnorm": _NORM_SHAPES, "rope": _ROPE_SHAPES,
                 "topk_softmax": _TOPK_SHAPES, "batched_gemm": _BATCHED_GEMM_SHAPES,
                 "gemm_a8w8_blockscale": _BLOCKSCALE_SHAPES,
                 "rope_gptj": _ROPE_SHAPES, "rope_partial": _ROPE_SHAPES}
VENDOR_DTYPES = ("bf16", "fp16")
ROPE_BASE = 10000.0
# Per-op dtype override (defaults to VENDOR_DTYPES). a8w8 GEMM sweeps fp8 + int8
# (both 8-bit-in / bf16-out); batched GEMM is bf16-only (aiter.batched_gemm_bf16).
VENDOR_OP_DTYPES = {"gemm_a8w8": ("fp8", "int8"), "batched_gemm": ("bf16",),
                    "gemm_a8w8_blockscale": ("fp8",)}


def vendor_op_dtypes(op: str) -> tuple[str, ...]:
    """The dtype sweep for a vendor op (per-op override or the global default)."""
    return VENDOR_OP_DTYPES.get(op, VENDOR_DTYPES)


EPS = 1e-6
INT8_MAX = 127.0


BLK = 128  # DeepSeek-V3 block size (1x128 activation, 128x128 weight)


def _quant_1x128(x):
    """Per-1x128 (per-token-group) fp8 quant. x[M,K] -> (xq[M,K] fp8, xs[M,K//128])."""
    import torch

    from kore.tasks.aiter_ref import FP8_DTYPE, FP8_MAX
    M, K = x.shape
    xb = x.view(M, K // BLK, BLK)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    s = amax / FP8_MAX
    xq = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).view(M, K)
    return xq, s.squeeze(-1).to(torch.float32).contiguous()


def _quant_128x128(w):
    """Per-128x128 block fp8 quant. w[N,K] -> (wq[N,K] fp8, ws[N//128,K//128])."""
    import torch

    from kore.tasks.aiter_ref import FP8_DTYPE, FP8_MAX
    N, K = w.shape
    wb = w.view(N // BLK, BLK, K // BLK, BLK)
    amax = wb.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-8)
    s = amax / FP8_MAX
    wq = (wb / s).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).view(N, K)
    return wq, s.squeeze(1).squeeze(-1).to(torch.float32).contiguous()


def _blockscale_ref(xq, wq, xs, ws):
    """Exact fp32 dequant-matmul oracle for block-scaled fp8 GEMM -> bf16."""
    import torch
    M, K = xq.shape
    N = wq.shape[0]
    xd = (xq.view(M, K // BLK, BLK).float() * xs[:, :, None]).view(M, K)
    wd = (wq.view(N // BLK, BLK, K // BLK, BLK).float() * ws[:, None, :, None]).view(N, K)
    return (xd @ wd.t()).to(torch.bfloat16)


def _quant_a8w8(x, qdtype: str):
    """Per-tensor symmetric quantization to fp8 (e4m3fnuz) or int8, returning
    ``(q, scale)`` with ``x ~= q.float() * scale`` (scale is a scalar fp32 tensor)."""
    import torch

    from kore.tasks import aiter_ref

    if qdtype == "fp8":
        return aiter_ref.per_tensor_quant_fp8(x)
    if qdtype == "int8":
        amax = x.abs().max().clamp(min=1e-12)
        scale = (amax / INT8_MAX).to(torch.float32)
        q = (x.float() / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        return q, scale.reshape(())
    raise ValueError(f"unknown a8w8 quant dtype {qdtype!r}")


# --------------------------------------------------------------------------- #
# reference.py namespace (torch fp32 oracle + AITER production baseline)
# --------------------------------------------------------------------------- #
def make_vendor_reference(op: str, dtype: str) -> dict:
    import torch
    import torch.nn.functional as F
    from kore.tasks import aiter_ref

    tdt = getattr(torch, DTYPES[dtype][0])

    def _randn(shape, device, seed, scale=1.0):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape, generator=g, device=device, dtype=torch.float32) * scale).to(tdt)

    if op == "rmsnorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            x = _randn((M, N), device, seed)
            w = (torch.randn((N,), generator=torch.Generator(device=device).manual_seed(seed + 1),
                             device=device, dtype=torch.float32) * 0.1 + 1.0).to(tdt)
            return (x, w)

        def ref_fn(x, w):
            xf = x.float()
            y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
            return y.to(x.dtype)

        def baseline_fn(x, w):
            return aiter_ref.aiter_rms_norm(x, w, EPS)

        arity = 2

    elif op == "layernorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            x = _randn((M, N), device, seed)
            g = torch.Generator(device=device).manual_seed(seed + 1)
            w = (torch.randn((N,), generator=g, device=device, dtype=torch.float32) * 0.1 + 1.0).to(tdt)
            b = _randn((N,), device, seed + 2, scale=0.1)
            return (x, w, b)

        def ref_fn(x, w, b):
            return F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), EPS).to(x.dtype)

        def baseline_fn(x, w, b):
            return aiter_ref.aiter_layer_norm(x, w, b, EPS)

        arity = 3

    elif op in ("silu_mul", "gelu_mul"):
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]  # N = 2*inter
            return (_randn((M, N), device, seed),)

        if op == "silu_mul":
            def ref_fn(x):
                inter = x.shape[-1] // 2
                g, u = x[:, :inter].float(), x[:, inter:].float()
                return (F.silu(g) * u).to(x.dtype)

            def baseline_fn(x):
                return aiter_ref.aiter_silu_and_mul(x)
        else:
            def ref_fn(x):
                inter = x.shape[-1] // 2
                g, u = x[:, :inter].float(), x[:, inter:].float()
                return (F.gelu(g, approximate="tanh") * u).to(x.dtype)

            def baseline_fn(x):
                return aiter_ref.aiter_gelu_tanh_and_mul(x)

        arity = 1

    elif op == "softmax":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), device, seed, scale=2.0),)  # logit-scale rows

        def ref_fn(x):
            return torch.softmax(x.float(), dim=-1).to(x.dtype)

        def baseline_fn(x):
            return aiter_ref.torch_softmax_lastdim(x)  # ROCm MIOpen fused softmax

        arity = 1

    elif op == "gemm_a8w8":
        qdtype = dtype  # "fp8" or "int8"

        def get_inputs(shape, device="cuda", seed=0):
            M, N, K = shape["M"], shape["N"], shape["K"]
            g = torch.Generator(device=device).manual_seed(seed)
            a = torch.randn((M, K), generator=g, device=device, dtype=torch.float32)
            w = torch.randn((N, K), generator=g, device=device, dtype=torch.float32)
            xq, sx = _quant_a8w8(a, qdtype)
            wq, sw = _quant_a8w8(w, qdtype)
            return (xq, wq, sx.repeat(M, 1).contiguous(), sw.repeat(1, N).contiguous())

        def ref_fn(xq, wq, x_scale, w_scale):
            a_deq = xq.float() * x_scale.float()               # [M,K]
            w_deq = wq.float() * w_scale.float().reshape(-1, 1)  # [N,K]
            return (a_deq @ w_deq.t()).to(torch.bfloat16)

        def baseline_fn(xq, wq, x_scale, w_scale):
            return aiter_ref.aiter_gemm_a8w8(xq, wq, x_scale, w_scale,
                                             out_dtype=torch.bfloat16)

        arity = 4

    elif op == "fused_add_rmsnorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            x = _randn((M, N), device, seed)
            residual = _randn((M, N), device, seed + 1)
            w = (torch.randn((N,), generator=torch.Generator(device=device).manual_seed(seed + 2),
                             device=device, dtype=torch.float32) * 0.1 + 1.0).to(tdt)
            return (x, residual, w)

        def ref_fn(x, residual, w):
            added = x.float() + residual.float()
            var = added.pow(2).mean(dim=-1, keepdim=True)
            y = added * torch.rsqrt(var + EPS) * w.float()
            return y.to(x.dtype), added.to(x.dtype)   # (normed, new_residual)

        def baseline_fn(x, residual, w):
            return aiter_ref.aiter_fused_add_rms_norm(x, residual, w, EPS)

        arity = 3

    elif op == "rope":
        def get_inputs(shape, device="cuda", seed=0):
            S, B, H, D = shape["S"], shape["B"], shape["H"], shape["D"]
            g = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn((S, B, H, D), generator=g, device=device, dtype=torch.float32).to(tdt)
            inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, D, 2, device=device,
                                                          dtype=torch.float32) / D))
            t = torch.arange(S, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq).view(S, 1, 1, D // 2).contiguous()
            return (x, freqs)

        def ref_fn(x, freqs):
            xf = x.float()
            D = xf.shape[-1]
            cos = torch.cos(freqs).float()
            sin = torch.sin(freqs).float()
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
            x1, x2 = xf[..., : D // 2], xf[..., D // 2:]
            rot = torch.cat([-x2, x1], dim=-1)
            return (xf * cos + rot * sin).to(x.dtype)

        def baseline_fn(x, freqs):
            return aiter_ref.aiter_rope_neox(x, freqs)

        arity = 2

    elif op == "topk_softmax":
        from kore.tasks.aiter_ref_attn import aiter_topk_softmax

        def _to_dense(topk_weights, topk_ids, E):
            M = topk_weights.shape[0]
            dense = torch.zeros((M, E), device=topk_weights.device, dtype=torch.float32)
            dense.scatter_(1, topk_ids.long(), topk_weights.float())
            return dense

        def get_inputs(shape, device="cuda", seed=0):
            M, E = shape["M"], shape["E"]
            gate = torch.randn((M, E), generator=torch.Generator(device=device).manual_seed(seed),
                               device=device, dtype=torch.float32).to(tdt)
            return (gate, int(shape["topk"]))

        def ref_fn(gate, topk):
            E = gate.shape[1]
            sm = torch.softmax(gate.float(), dim=-1)
            tw, ti = torch.topk(sm, topk, dim=-1)
            tw = tw / tw.sum(dim=-1, keepdim=True)
            return _to_dense(tw, ti, E)                    # dense [M,E], order-independent

        def baseline_fn(gate, topk):
            w, ids = aiter_topk_softmax(gate, topk, True)
            return _to_dense(w, ids, gate.shape[1])

        arity = 2

    elif op == "batched_gemm":
        def get_inputs(shape, device="cuda", seed=0):
            B, M, N, K = shape["B"], shape["M"], shape["N"], shape["K"]
            g = torch.Generator(device=device).manual_seed(seed)
            sc = 1.0 / (K ** 0.5)   # keep accumulated magnitude ~O(1) for stable bf16
            a = (torch.randn((B, M, K), generator=g, device=device, dtype=torch.float32) * sc).to(tdt)
            b = (torch.randn((B, N, K), generator=g, device=device, dtype=torch.float32) * sc).to(tdt)
            return (a, b)

        def ref_fn(a, b):
            return torch.bmm(a.float(), b.transpose(1, 2).float()).to(a.dtype)  # A@B^T per batch

        def baseline_fn(a, b):
            return aiter_ref.aiter_batched_gemm_bf16(a, b)

        arity = 2

    elif op == "gemm_a8w8_blockscale":
        def get_inputs(shape, device="cuda", seed=0):
            M, N, K = shape["M"], shape["N"], shape["K"]
            g = torch.Generator(device=device).manual_seed(seed)
            X = torch.randn((M, K), generator=g, device=device, dtype=torch.float32) * 0.2
            W = torch.randn((N, K), generator=g, device=device, dtype=torch.float32) * 0.2
            xq, xs = _quant_1x128(X)
            wq, ws = _quant_128x128(W)
            return (xq, wq, xs, ws)

        def ref_fn(xq, wq, xs, ws):
            return _blockscale_ref(xq, wq, xs, ws)

        def baseline_fn(xq, wq, xs, ws):
            import aiter
            return aiter.gemm_a8w8_blockscale(xq, wq, xs, ws, dtype=torch.bfloat16)

        arity = 4

    elif op == "rope_gptj":
        # GPT-J / interleaved RoPE: rotate adjacent pairs (x[2i], x[2i+1]) by angle[i]
        # (vs NEOX which rotates x[i] with x[i+D/2]). Many models (GPT-J/NeoX-interleaved)
        # use this layout. No dedicated aiter kernel -> torch is the honest bar.
        def get_inputs(shape, device="cuda", seed=0):
            S, B, H, D = shape["S"], shape["B"], shape["H"], shape["D"]
            g = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn((S, B, H, D), generator=g, device=device, dtype=torch.float32).to(tdt)
            inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, D, 2, device=device,
                                                         dtype=torch.float32) / D))
            t = torch.arange(S, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq).view(S, 1, 1, D // 2).contiguous()
            return (x, freqs)

        def ref_fn(x, freqs):
            xf = x.float()
            cos = torch.cos(freqs).float()   # [S,1,1,D//2]
            sin = torch.sin(freqs).float()
            x1 = xf[..., 0::2]               # even lanes
            x2 = xf[..., 1::2]               # odd lanes
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            out = torch.empty_like(xf)
            out[..., 0::2] = o1
            out[..., 1::2] = o2
            return out.to(x.dtype)

        def baseline_fn(x, freqs):
            return ref_fn(x, freqs)          # torch bar (no vendor gptj kernel)

        arity = 2

    elif op == "rope_partial":
        # Partial-rotary RoPE (GPT-NeoX partial / Phi-style): rotate only the first
        # rotary_dim = D//2 lanes (NEOX half-split within that band), pass the rest
        # through unchanged. Common in models that keep part of the head-dim un-rotated.
        def get_inputs(shape, device="cuda", seed=0):
            S, B, H, D = shape["S"], shape["B"], shape["H"], shape["D"]
            rot = D // 2                     # rotary_dim
            g = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn((S, B, H, D), generator=g, device=device, dtype=torch.float32).to(tdt)
            inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, rot, 2, device=device,
                                                         dtype=torch.float32) / rot))
            t = torch.arange(S, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq).view(S, 1, 1, rot // 2).contiguous()
            return (x, freqs)

        def ref_fn(x, freqs):
            xf = x.float()
            D = xf.shape[-1]
            rot = D // 2
            cos = torch.cos(freqs).float()
            sin = torch.sin(freqs).float()
            cos = torch.cat([cos, cos], dim=-1)   # [S,1,1,rot]
            sin = torch.cat([sin, sin], dim=-1)
            xr = xf[..., :rot]
            xp = xf[..., rot:]                     # pass-through (un-rotated) band
            x1, x2 = xr[..., : rot // 2], xr[..., rot // 2:]
            rotd = torch.cat([-x2, x1], dim=-1)
            out_r = xr * cos + rotd * sin
            return torch.cat([out_r, xp], dim=-1).to(x.dtype)

        def baseline_fn(x, freqs):
            return ref_fn(x, freqs)

        arity = 2

    else:
        raise ValueError(f"unknown vendor op {op!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op, "dtype_name": dtype,
          "family": f"vendor_{op}", "mutates_input": op in VENDOR_MUTATES_INPUT}
    ns["adversarial_inputs"] = _make_adversarial_inputs(op, get_inputs, dtype)
    ns[f"{op}_ref"] = ref_fn
    return ns


def _make_adversarial_inputs(op: str, get_inputs, dtype: str = "bf16"):
    """Op-class-aware adversarial input battery (verification-in-the-loop).

    Plain-float vendor ops reuse the generic float fills (zeros/ones/large/small/
    sign-alt), which are exhaustive of the qualitative regimes. Quantized GEMM ops
    must build the battery in FLOAT then quantize, so the fp8 codes + scales stay a
    valid dequantizable pair (filling the fp8 tensors directly would be nonsense)."""
    import torch

    from kore.tasks._genops import _adversarial_fills

    def _generic(shape, device="cuda", seed=0):
        return list(_adversarial_fills(get_inputs(shape, device=device, seed=seed)))

    if op == "topk_softmax":
        # Constant gates create expert TIES (softmax uniform), so different correct
        # routers legitimately disagree — the generic fills would false-reject. Use
        # strictly-DISTINCT integer ramps (bf16/fp16-exact for E<=256) so the top-k
        # selection is unambiguous, exercising softmax numerics + selection order.
        tdt = getattr(torch, DTYPES[dtype][0])

        def _topk_adv(shape, device="cuda", seed=0):
            M, E, topk = shape["M"], shape["E"], int(shape["topk"])
            base = torch.arange(E, device=device, dtype=torch.float32).view(1, E).repeat(M, 1)
            regimes = {"ramp": base, "neg_ramp": -base, "shifted": base - (E // 2)}
            return [(name, (g.to(tdt), topk)) for name, g in regimes.items()]

        return _topk_adv

    if op == "gemm_a8w8_blockscale":
        def _blockscale_adv(shape, device="cuda", seed=0):
            M, N, K = shape["M"], shape["N"], shape["K"]
            regimes = {
                "zeros": (torch.zeros((M, K)), torch.zeros((N, K))),
                "ones": (torch.ones((M, K)), torch.ones((N, K))),
                "large": (torch.full((M, K), 10.0), torch.full((N, K), 10.0)),
                "small": (torch.full((M, K), 1e-2), torch.full((N, K), 1e-2)),
            }
            out = []
            for name, (X, W) in regimes.items():
                X = X.to(device=device, dtype=torch.float32)
                W = W.to(device=device, dtype=torch.float32)
                xq, xs = _quant_1x128(X)
                wq, ws = _quant_128x128(W)
                out.append((name, (xq, wq, xs, ws)))
            return out
        return _blockscale_adv

    if op != "gemm_a8w8":
        return _generic

    def _quant_gemm(shape, device="cuda", seed=0):
        M, N, K = shape["M"], shape["N"], shape["K"]
        regimes = {
            "zeros": (torch.zeros((M, K)), torch.zeros((N, K))),
            "ones": (torch.ones((M, K)), torch.ones((N, K))),
            "large": (torch.full((M, K), 100.0), torch.full((N, K), 100.0)),
            "small": (torch.full((M, K), 1e-2), torch.full((N, K), 1e-2)),
            "mixed_sign": (torch.ones((M, K)).cumsum(1) % 2 * 2 - 1,
                           torch.ones((N, K)).cumsum(1) % 2 * 2 - 1),
        }
        out = []
        for name, (a, w) in regimes.items():
            a = a.to(device=device, dtype=torch.float32)
            w = w.to(device=device, dtype=torch.float32)
            xq, sx = _quant_a8w8(a, dtype)   # dtype: "fp8" | "int8"
            wq, sw = _quant_a8w8(w, dtype)
            out.append((name, (xq, wq, sx.repeat(M, 1).contiguous(),
                               sw.repeat(1, N).contiguous())))
        return out

    return _quant_gemm


# --------------------------------------------------------------------------- #
# Real Triton starter seeds (the policy optimizes these against the AITER bar)
# --------------------------------------------------------------------------- #
_RMSNORM_SEED = '''"""GENERATED vendor-baselined RMSNorm seed ({dtype}) vs aiter.rms_norm.
One program/row: fp32 mean-square, rsqrt, weight, {tldt} store. Regenerate via
kore/tasks/generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (x * rstd * w).to({tldt}), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _rmsnorm_kernel[(M,)](x, weight, y, x.stride(0), N, eps,
                          BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
'''

_LAYERNORM_SEED = '''"""GENERATED vendor-baselined LayerNorm seed ({dtype}) vs aiter.layer_norm.
One program/row: fp32 mean+var, affine, {tldt} store. Regenerate via
kore/tasks/generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
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


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _layernorm_kernel[(M,)](x, weight, bias, y, x.stride(0), N, eps,
                            BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
'''

_GATE_SEED = '''"""GENERATED vendor-baselined {op} seed ({dtype}) vs aiter {op}.
Gated MLP activation x[M,2*inter] -> {op_desc}(gate)*up [M,inter], {tldt} store.
Regenerate via kore/tasks/generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, y_ptr, sxm, sym, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    gate = tl.load(x_ptr + row * sxm + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * sxm + N + offs, mask=mask, other=0.0).to(tl.float32)
    act = {act_expr}
    tl.store(y_ptr + row * sym + offs, (act * up).to({tldt}), mask=mask)


def {op}(x: torch.Tensor) -> torch.Tensor:
    M, two_n = x.shape
    N = two_n // 2
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _{op}_kernel[grid](x, y, x.stride(0), y.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)
    return y
'''


_SOFTMAX_SEED = '''"""GENERATED vendor-baselined row-softmax seed ({dtype}) vs torch/MIOpen softmax.
Online (streaming) softmax: pass 1 running max+sum, pass 2 normalize+store, so any
row width N fits regardless of BLOCK_N. Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, sm, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    m = -float("inf")
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + base + offs, (tl.exp(x - m) / s).to({tldt}), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _softmax_kernel[(M,)](x, y, x.stride(0), N, BLOCK_N=1024, num_warps=8)
    return y
'''

_FP8_GEMM_SEED = '''"""GENERATED vendor-baselined a8w8 GEMM seed ({dtype}) vs aiter.gemm_a8w8.
Y = (XQ*x_scale) @ (WQ*w_scale)^T, bf16 out. 8-bit (fp8/int8) operands up-converted
in-register to fp32, fp32 accumulate, scales on the accumulator. Dtype-agnostic:
the load->fp32 path handles both fp8 e4m3fnuz and int8. Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _gemm_a8w8_kernel(a_ptr, b_ptr, c_ptr, xs_ptr, ws_ptr, M, N, K,
                      stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=kmask, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    xs = tl.load(xs_ptr + offs_cm, mask=offs_cm < M, other=0.0).to(tl.float32)
    ws = tl.load(ws_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    acc = acc * xs[:, None] * ws[None, :]
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def gemm_a8w8(xq: torch.Tensor, wq: torch.Tensor,
              x_scale: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    M, K = xq.shape
    N, _ = wq.shape
    c = torch.empty((M, N), device=xq.device, dtype=torch.bfloat16)
    xs = x_scale.reshape(-1).contiguous()
    ws = w_scale.reshape(-1).contiguous()
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 128, 64, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _gemm_a8w8_kernel[grid](xq, wq, c, xs, ws, M, N, K,
                            xq.stride(0), xq.stride(1), wq.stride(0), wq.stride(1),
                            c.stride(0), c.stride(1),
                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                            GROUP_M=GROUP_M, num_warps=4, num_stages=2)
    return c
'''


_FUSED_ADD_RMSNORM_SEED = '''"""GENERATED vendor-baselined fused add-RMSNorm seed ({dtype}) vs aiter.fused_add_rms_norm_cu.
added = x + residual (the new residual); y = RMSNorm(added) * weight. One program
per row, fp32 accumulate, {tldt} store. Returns (y, added) — the candidate writes
NEW tensors (the vendor baseline is in-place). Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, y_ptr, added_ptr, sm, N, eps,
                              BLOCK_N: tl.constexpr):
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


def fused_add_rmsnorm(x, residual, weight, eps: float = 1e-6):
    M, N = x.shape
    y = torch.empty_like(x)
    added = torch.empty_like(x)
    _fused_add_rmsnorm_kernel[(M,)](x, residual, weight, y, added, x.stride(0), N, eps,
                                    BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y, added
'''


_ROPE_SEED = '''"""GENERATED vendor-baselined NEOX RoPE seed ({dtype}) vs aiter.rope_fwd.
x[S,B,H,D], freqs[S,1,1,D//2] angles. One program per (s,b,h) row; half-width
rotate-NEOX identity (o1=x1*cos-x2*sin, o2=x2*cos+x1*sin), fp32 math, {tldt} store.
Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _rope_kernel(x_ptr, f_ptr, y_ptr, B, H, D,
                 sxs, sxb, sxh, sxd, sfs, HALF: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    tmp = pid // H
    b = tmp % B
    s = tmp // B
    base = s * sxs + b * sxb + h * sxh
    offs = tl.arange(0, HALF)
    x1 = tl.load(x_ptr + base + offs * sxd).to(tl.float32)
    x2 = tl.load(x_ptr + base + (offs + HALF) * sxd).to(tl.float32)
    theta = tl.load(f_ptr + s * sfs + offs).to(tl.float32)
    cos = tl.cos(theta)
    sin = tl.sin(theta)
    tl.store(y_ptr + base + offs * sxd, (x1 * cos - x2 * sin).to({tldt}))
    tl.store(y_ptr + base + (offs + HALF) * sxd, (x2 * cos + x1 * sin).to({tldt}))


def rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    S, B, H, D = x.shape
    y = torch.empty_like(x)
    f = freqs.reshape(S, D // 2)
    _rope_kernel[(S * B * H,)](x, f, y, B, H, D,
                               x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                               f.stride(0), HALF=D // 2, num_warps=4)
    return y
'''


_TOPK_SOFTMAX_SEED = '''"""GENERATED vendor-baselined MoE router seed ({dtype}) vs aiter.topk_softmax.
gate[M,E] -> fp32 softmax over experts -> top-k (masked argmax) -> renorm; returned
as a DENSE [M,E] weight tensor (order-independent grading; the vendor baseline is
scattered to dense the same way). Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _topk_softmax_kernel(gate_ptr, w_ptr, id_ptr, sg_m, sw_m, sid_m, E, topk,
                         EMAX: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, EMAX)
    mask = offs < E
    g = tl.load(gate_ptr + row * sg_m + offs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(g, axis=0)
    ex = tl.where(mask, tl.exp(g - m), 0.0)
    probs = tl.where(mask, ex / tl.sum(ex, axis=0), -1.0)
    pw = probs
    wsum = 0.0
    for _ in range(0, topk):
        wsum += tl.max(pw, axis=0)
        pw = tl.where(offs == tl.argmax(pw, axis=0), -1.0, pw)
    pw = probs
    for k in range(0, topk):
        bv = tl.max(pw, axis=0)
        bi = tl.argmax(pw, axis=0)
        tl.store(id_ptr + row * sid_m + k, bi.to(tl.int32))
        tl.store(w_ptr + row * sw_m + k, bv / wsum)
        pw = tl.where(offs == bi, -1.0, pw)


def topk_softmax(gate: torch.Tensor, topk: int) -> torch.Tensor:
    M, E = gate.shape
    w = torch.empty((M, topk), device=gate.device, dtype=torch.float32)
    ids = torch.empty((M, topk), device=gate.device, dtype=torch.int32)
    _topk_softmax_kernel[(M,)](gate, w, ids, gate.stride(0), w.stride(0), ids.stride(0),
                              E, topk, EMAX=triton.next_power_of_2(E), num_warps=4)
    dense = torch.zeros((M, E), device=gate.device, dtype=torch.float32)
    dense.scatter_(1, ids.long(), w)
    return dense
'''


_BATCHED_GEMM_SEED = '''"""GENERATED vendor-baselined batched GEMM seed ({dtype}) vs aiter.batched_gemm_bf16.
C[b] = A[b] @ B[b]^T per batch; A[B,M,K], B[B,N,K] -> C[B,M,N] bf16, fp32 accumulate.
One program per (batch, m-tile, n-tile). Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _bgemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  sab, sam, sak, sbb, sbn, sbk, scb, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    per_batch = num_m * num_n
    batch = pid // per_batch
    rem = pid % per_batch
    pid_m = rem // num_n
    pid_n = rem % num_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + batch * sab + (offs_m[:, None] * sam + offs_k[None, :] * sak)
    b_ptrs = b_ptr + batch * sbb + (offs_n[None, :] * sbn + offs_k[:, None] * sbk)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kmask = offs_k < K - k * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & kmask[:, None], other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c_ptrs = c_ptr + batch * scb + offs_m[:, None] * scm + offs_n[None, :] * scn
    cmask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=cmask)


def batched_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, M, K = a.shape
    N = b.shape[1]
    c = torch.empty((B, M, N), device=a.device, dtype=torch.bfloat16)
    BM, BN, BK = 64, 64, 64
    grid = (B * triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _bgemm_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), a.stride(2),
                        b.stride(0), b.stride(1), b.stride(2),
                        c.stride(0), c.stride(1), c.stride(2),
                        BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2)
    return c
'''


_BLOCKSCALE_SEED = '''"""GENERATED vendor-baselined block-scaled fp8 GEMM seed ({dtype}) vs aiter.gemm_a8w8_blockscale.
DeepSeek-V3 blockscale: XQ[M,K] fp8 with x_scale[M,K//128] (1x128), WQ[N,K] fp8 with
w_scale[N//128,K//128] (128x128) -> Y = X_deq @ W_deq^T bf16. Per-128-K-block dequant
applied on the fp32 accumulator. Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _bs_kernel(a_ptr, b_ptr, c_ptr, xs_ptr, ws_ptr, M, N, K, KB,
               sam, sak, sbn, sbk, scm, scn, sxm, sxk, swn, swk,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    num_n = N // BN                       # BN == 128 == weight n-block
    pid_m = pid // num_n
    pid_n = pid % num_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, KB):
        koff = kb * BK + offs_k
        a = tl.load(a_ptr + offs_m[:, None] * sam + koff[None, :] * sak,
                    mask=offs_m[:, None] < M, other=0.0).to(tl.float32)   # [BM,BK]
        b = tl.load(b_ptr + offs_n[None, :] * sbn + koff[:, None] * sbk,
                    mask=offs_n[None, :] < N, other=0.0).to(tl.float32)   # [BK,BN]
        p = tl.dot(a, b)                                                  # [BM,BN]
        xs = tl.load(xs_ptr + offs_m * sxm + kb * sxk,
                     mask=offs_m < M, other=0.0).to(tl.float32)           # [BM]
        ws = tl.load(ws_ptr + pid_n * swn + kb * swk).to(tl.float32)      # scalar
        acc += p * xs[:, None] * ws
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gemm_a8w8_blockscale(xq, wq, x_scale, w_scale) -> torch.Tensor:
    M, K = xq.shape
    N = wq.shape[0]
    c = torch.empty((M, N), device=xq.device, dtype=torch.bfloat16)
    BM, BN, BK = 64, 128, 128
    grid = (triton.cdiv(M, BM) * (N // BN),)
    _bs_kernel[grid](xq, wq, c, x_scale, w_scale, M, N, K, K // BK,
                     xq.stride(0), xq.stride(1), wq.stride(0), wq.stride(1),
                     c.stride(0), c.stride(1), x_scale.stride(0), x_scale.stride(1),
                     w_scale.stride(0), w_scale.stride(1),
                     BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2)
    return c
'''


_ROPE_GPTJ_SEED = '''"""GENERATED GPT-J (interleaved) RoPE seed ({dtype}).
x[S,B,H,D], freqs[S,1,1,D//2] angles. Rotates interleaved pairs (x[2i], x[2i+1])
by angle[i]: o[2i]=x1*cos-x2*sin, o[2i+1]=x2*cos+x1*sin. fp32 math, {tldt} store.
Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _rope_gptj_kernel(x_ptr, f_ptr, y_ptr, B, H, D,
                      sxs, sxb, sxh, sxd, sfs, HALF: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    tmp = pid // H
    b = tmp % B
    s = tmp // B
    base = s * sxs + b * sxb + h * sxh
    offs = tl.arange(0, HALF)
    x1 = tl.load(x_ptr + base + (2 * offs) * sxd).to(tl.float32)
    x2 = tl.load(x_ptr + base + (2 * offs + 1) * sxd).to(tl.float32)
    theta = tl.load(f_ptr + s * sfs + offs).to(tl.float32)
    cos = tl.cos(theta)
    sin = tl.sin(theta)
    tl.store(y_ptr + base + (2 * offs) * sxd, (x1 * cos - x2 * sin).to({tldt}))
    tl.store(y_ptr + base + (2 * offs + 1) * sxd, (x2 * cos + x1 * sin).to({tldt}))


def rope_gptj(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    S, B, H, D = x.shape
    y = torch.empty_like(x)
    f = freqs.reshape(S, D // 2)
    _rope_gptj_kernel[(S * B * H,)](x, f, y, B, H, D,
                                    x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                                    f.stride(0), HALF=D // 2, num_warps=4)
    return y
'''


_ROPE_PARTIAL_SEED = '''"""GENERATED partial-rotary RoPE seed ({dtype}).
x[S,B,H,D]; rotate only the first rotary_dim = D//2 lanes (NEOX half-split within
that band: pair i with i+rot/2), pass the remaining lanes through unchanged.
freqs[S,1,1,rot//2]. fp32 math, {tldt} store. Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _rope_partial_kernel(x_ptr, f_ptr, y_ptr, B, H, D,
                         sxs, sxb, sxh, sxd, sfs, ROT: tl.constexpr, QUART: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    tmp = pid // H
    b = tmp % B
    s = tmp // B
    base = s * sxs + b * sxb + h * sxh
    offs = tl.arange(0, QUART)                       # rot//2 rotation angles
    x1 = tl.load(x_ptr + base + offs * sxd).to(tl.float32)
    x2 = tl.load(x_ptr + base + (offs + QUART) * sxd).to(tl.float32)
    theta = tl.load(f_ptr + s * sfs + offs).to(tl.float32)
    cos = tl.cos(theta)
    sin = tl.sin(theta)
    tl.store(y_ptr + base + offs * sxd, (x1 * cos - x2 * sin).to({tldt}))
    tl.store(y_ptr + base + (offs + QUART) * sxd, (x2 * cos + x1 * sin).to({tldt}))
    poffs = ROT + tl.arange(0, ROT)                  # pass-through lanes [rot, D)
    xp = tl.load(x_ptr + base + poffs * sxd, mask=poffs < D, other=0.0)
    tl.store(y_ptr + base + poffs * sxd, xp, mask=poffs < D)


def rope_partial(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    S, B, H, D = x.shape
    rot = D // 2
    y = torch.empty_like(x)
    f = freqs.reshape(S, rot // 2)
    _rope_partial_kernel[(S * B * H,)](x, f, y, B, H, D,
                                       x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                                       f.stride(0), ROT=rot, QUART=rot // 2, num_warps=4)
    return y
'''


def vendor_seed_source(op: str, dtype: str) -> str:
    tldt = DTYPES[dtype][1]
    if op == "rope_gptj":
        return _ROPE_GPTJ_SEED.format(dtype=dtype, tldt=tldt)
    if op == "rope_partial":
        return _ROPE_PARTIAL_SEED.format(dtype=dtype, tldt=tldt)
    if op == "gemm_a8w8_blockscale":
        return _BLOCKSCALE_SEED.format(dtype=dtype)
    if op == "batched_gemm":
        return _BATCHED_GEMM_SEED.format(dtype=dtype)
    if op == "topk_softmax":
        return _TOPK_SOFTMAX_SEED.format(dtype=dtype)
    if op == "softmax":
        return _SOFTMAX_SEED.format(dtype=dtype, tldt=tldt)
    if op == "gemm_a8w8":
        return _FP8_GEMM_SEED.format(dtype=dtype)
    if op == "fused_add_rmsnorm":
        return _FUSED_ADD_RMSNORM_SEED.format(dtype=dtype, tldt=tldt)
    if op == "rope":
        return _ROPE_SEED.format(dtype=dtype, tldt=tldt)
    if op == "rmsnorm":
        return _RMSNORM_SEED.format(dtype=dtype, tldt=tldt)
    if op == "layernorm":
        return _LAYERNORM_SEED.format(dtype=dtype, tldt=tldt)
    if op == "silu_mul":
        return _GATE_SEED.format(op="silu_mul", op_desc="silu", dtype=dtype, tldt=tldt,
                                 act_expr="gate * tl.sigmoid(gate)")
    if op == "gelu_mul":
        gelu = ("0.5 * gate * (1.0 + (2.0 * tl.sigmoid(2.0 * (0.7978845608028654 * "
                "(gate + 0.044715 * gate * gate * gate))) - 1.0))")
        return _GATE_SEED.format(op="gelu_mul", op_desc="gelu_tanh", dtype=dtype, tldt=tldt,
                                 act_expr=gelu)
    raise ValueError(op)
