# P0 (roofline / Speed-of-Light) results — native gfx950 (MI350X)

Status: **native stack working; check (b) PASSES; checks (a)/(c) partially blocked (documented below).**

## Node & stack
- Host `cv350-...` — 8× **gfx950** (AMD Instinct **MI350X**, CDNA4), ROCm 7.2.3, `rocprofv3`.
- **Native stack (no override):** `torch 2.10.0+rocm7.0` (hip 7.0.51831) + `pytorch-triton-rocm 3.5.1`.
  `torch.cuda.get_arch_list()` includes `gfx950`; device reports `AMD Instinct MI350X`. Verified
  real bf16 matmul + Triton kernels run natively.
- Disk 2.2 GB/s, 3 TB RAM. (Torch's public rocm6.4 wheel rejects gfx950; rocm7.0 is required and works.)

## Roofline model
`T_min = max(W_flops/P_peak, Q_bytes/B_peak)`, `eta = T_min/T_measured` (SOL attainment).
gfx950 peaks (approx, env-overridable via `KORE_PEAK_{BF16,FP8,HBM_BW}`): HBM 8.0 TB/s, bf16 2.5 PF/s, fp8 5.0 PF/s.

## Full native sweep — all 15 tasks (seed kernels)
Every seed is **correct** on native gfx950 (torch fp32 oracle). η = SOL attained; speedup vs vendor
where a vendor baseline was available (torch/hipBLASLt; aiter pending — see below).

| task | bound | η (SOL) | speedup vs vendor | note |
|---|---|---|---|---|
| silu_mul_bf16 | memory | 38.2% | – | aiter pending |
| fused_add_rmsnorm_bf16 | memory | 37.0% | – | aiter pending |
| gelu_tanh_bf16 | memory | 35.1% | 0.90× | torch F.gelu |
| rmsnorm_aiter | memory | 34.1% | – | aiter pending |
| layernorm_bf16 | memory | 33.7% | – | aiter pending |
| gemm_bf16 | compute | 21.4% | **0.46×** | hipBLASLt |
| softmax_bf16 | memory | 16.4% | 0.59× | torch.softmax |
| quant_fp8_pertoken | memory | 12.9% | – | aiter pending |
| flash_attn_decode_bf16 | memory | 10.1% | – | aiter pending |
| rope_bf16 | memory | 9.8% | – | aiter pending |
| paged_attn_decode_bf16 | memory | 2.2% | – | aiter pending |
| flash_attn_prefill_bf16 | compute | 1.3% | – | aiter pending |
| fused_moe_silu_bf16 | compute | 0.8% | – | aiter pending |
| gemm_fp8_a8w8 | compute | 0.6% | – | aiter pending |
| topk_softmax_bf16 | memory | 0.2% | – | aiter pending |

**Reading:** the naive seeds sit far below SOL exactly on the hard operators (attention, MoE, fp8
GEMM at <3%), and closer on memory-bound norms/activations (34–38%). The vendor-beating gap is
real: seed GEMM is 0.46× hipBLASLt.

## The three checks (native, PMC on)
```
(a) eta predicts speedup   : rho=0.50 (n=3)     -> WEAK   (only torch-baseline tasks have a vendor speedup yet)
(b) residual decomp R^2    : 0.997  (n=17)      -> PASS   (residual = stall + occupancy-deficit time; STRONG)
(c) monotone-in-valley frac: -      (pairs=1)   -> SKIP   (need >=3 correct kernels/task as trajectories)
```

**check (b) is the headline preliminary result:** on real gfx950, the runtime residual
`(T_measured - T_min)` decomposes into counter-derived named terms (memory-stall time +
occupancy-deficit time) with **R² = 0.997** across 17 kernels. The "named gradient" the KORE
paradigm relies on is measurable — not drowned by cross-terms.

## PMC (gfx950 fix)
gfx950/CDNA4 renamed raw counters, so the original SQ_* names collected nothing. Now using
rocprofv3 **derived metrics** (`OccupancyPercent`, `MemUnitStalled`, `MfmaUtil`, `GRBM_GUI_ACTIVE`),
parsing the long-format `*_counter_collection.csv` and picking the longest-running compute kernel.

## Remaining for a complete GO/PARTIAL/FALLBACK/PIVOT verdict
1. **check (a) needs more vendor baselines.** aiter was cloned + built for gfx950, and
   `aiter_ref.py` is now version-robust (resolves `aiter.ops.*`). BUT the current aiter kernels
   require **triton ≥ 3.6.0**, while the torch 2.10+rocm7.0 stack ships **triton 3.5.1**. Options:
   (i) move to AMD's `torch 2.11+rocm7.13` + `triton 3.6` gfx950 nightly (another ~4.8 GB), or
   (ii) use aiter CK `_cu` variants (no triton dep) with adjusted signatures, or
   (iii) add torch-optimized baselines (F.rms_norm/F.layer_norm/SDPA) as the framework bar.
2. **check (c) needs trajectories** — ≥3 correct kernels per task (seed + variants) so the
   dominant-residual-falls-while-wall-flat test has data. Generate via the mutation/evolve operators.

## Reproduce
```bash
source ~/kore-venv/bin/activate
python -m kore.analysis.p0_sol --arch gfx950 --warmup 5 --iters 20 \
  --max-kernels-per-task 6 --out runs/p0_full.json      # native, no override
```
