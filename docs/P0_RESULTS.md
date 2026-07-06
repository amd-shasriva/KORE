# P0 (roofline / Speed-of-Light) — preliminary results on gfx950 (MI350)

Status: **pipeline validated on-GPU; full statistical verdict pending more vendor baselines.**

## Node
- Host: `cv350-...` — 8× **gfx950** (MI350-class, CDNA4), ROCm 7.2.3, `rocprofv3` present.
- Disk 2.2 GB/s, 3 TB RAM. Node is healthy.

## Software stack (how P0 runs today)
The only public PyTorch-ROCm wheel that installed cleanly is `torch==2.9.1+rocm6.4`, whose
bundled ROCm-6.4 runtime **rejects gfx950** (`Unsupported HSA device gfx950 ... hipErrorNoDevice`).
Workaround for **preliminary** runs: `HSA_OVERRIDE_GFX_VERSION=9.4.2`, which makes HIP treat the
CDNA4 device as its CDNA3 sibling **gfx942**. gfx942 kernels are ISA-compatible enough to execute
on gfx950 (verified: real bf16 matmul + a Triton kernel run correctly).

- `torch 2.9.1+rocm6.4` + `pytorch-triton-rocm 3.5.1`, run under `HSA_OVERRIDE_GFX_VERSION=9.4.2`.
- **Caveat:** these are gfx942-compiled kernels on gfx950 hardware → performance is *indicative,
  not native-gfx950-tuned*. Fine for validating the **methodology**; not for headline perf claims.
- **Native path identified** for real runs: `torch 2.10.0+rocm7.0`
  (`https://download.pytorch.org/whl/rocm7.0`) or AMD's `torch 2.11.0+rocm7.13` gfx950 nightly
  (`https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/`).

## Roofline model (kore/analysis/rooflines.py)
`T_min = max(W_flops / P_peak, Q_bytes / B_peak)`, `eta = T_min / T_measured` (SOL attainment).
gfx950 peaks used (approximate; env-overridable via `KORE_PEAK_{BF16,FP8,HBM_BW}`):
HBM 8.0 TB/s, bf16 2.5 PFLOP/s, fp8 5.0 PFLOP/s.

| task | dtype | bound | AI (FLOP/B) | T_min |
|---|---|---|---|---|
| gemm_bf16 (4096³) | bf16 | compute | 1365 | 55.0 µs |
| softmax_bf16 (4096²) | bf16 | memory | 1.25 | 8.4 µs |

## Preliminary measurements (seed kernels, gfx942-override)
| task | correct | η (SOL) | speedup vs vendor | vendor |
|---|---|---|---|---|
| gemm_bf16 | ✅ | 19.1% | **0.974×** | hipBLASLt (torch.matmul) |
| softmax_bf16 | ✅ | 16.0% | 0.581× | torch.softmax |

**Reading:** the seed GEMM is correct but ~0.97× the vendor and only ~19% of the roofline —
the "correct-but-slow wall" appears in the very first measurement. This is exactly the regime
KORE targets.

## What's validated
- End-to-end P0 path on real GPU: stage kernel → Triton compile → correctness (SNR) →
  candidate bench → vendor bench → analytic `T_min` → `eta` + speedup. ✅
- `kore/analysis/{rooflines,p0_sol}.py` + 17 CPU tests. ✅

## What a full GO/PARTIAL/FALLBACK/PIVOT verdict still needs
1. **More vendor baselines** — install `aiter` (gfx950) to unlock rmsnorm/silu/layernorm/rope/
   quant/moe/attention baselines (→ ≥5–10 `(eta, speedup)` points for check (a)).
2. **More correct candidate kernels per task** — current `data/groups/*` candidates largely fail
   under the override; need fresh datagen or the value-model candidates for check (c) trajectories.
3. **PMC decomposition** — validate `rocprofv3` counter collection under profiling (check (b)).
4. **Native gfx950 torch** (rocm7.0) for non-override, headline-grade perf numbers.

## Reproduce
```bash
source ~/kore-venv/bin/activate
HSA_OVERRIDE_GFX_VERSION=9.4.2 python -m kore.analysis.p0_sol \
  --tasks gemm_bf16,softmax_bf16 --arch gfx950 --warmup 5 --iters 20 --no-pmc \
  --out runs/p0_override.json
```
