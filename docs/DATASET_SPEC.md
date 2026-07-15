# KORE Training-Data Design Specification (v1)

Target: frontier ROCm GPU-kernel-generation model. Hardware: **AMD Instinct MI350X, CDNA4, `gfx950`** (the sole KORE target), wavefront 64, 256 CUs, 160 KiB LDS/CU, FP8 = **OCP** `e4m3fn`, with native MX-FP4 / MX-FP6 scaled-MFMA. (The previous-gen MI300X / MI325X `gfx942` / CDNA3 used the **FNUZ** FP8 variant and 64 KB LDS/CU; it appears here only as legacy training-data provenance, never as a target - see Section 1.2.)

This spec is directly implementable against the KORE codebase (`kore/kore/data/*`, `kore/kore/tasks/*`, `kore/kore/verifier/*`) and the record schemas in `kore/kore/data/schemas.py` (`RepairRecord`, `RankedGroupRecord`, `WinRecord`).

---

## 0. What makes this data genuinely the best (design principles)

Five principles derived from the evidence (Kevin, ConCuR, GEAK, KORE.pdf) that every subsequent section enforces:

1. **The baseline is the production kernel, not torch.** Every performance label is a speedup vs the *real* serving op (AITER / hipBLASLt / rocBLAS / CK), measured on-box (`--impl reference`), never vs `torch.matmul`/eager. This is the single most important differentiator from KernelBench/Kevin (which use PyTorch Eager) and is already the KORE convention (`kore/kore/tasks/aiter_ref.py`, every `task.yaml` `comparison_baseline`). A model that only beats torch is worthless in production; a model that beats AITER is best-in-world.
2. **Measured, not asserted.** Only *executed* outcomes enter the corpus. Every correctness/speedup label is produced by the verifier on real gfx950 silicon, re-verified independently, with a variance gate. No teacher-claimed number is ever trusted (Section 4).
3. **Learn from the abundant, RL-manufacture the scarce.** Repairs and ranked candidates are cheap and plentiful; strong wins are scarce (~15-20 per ~300 audited trajectories per KORE.pdf §2). The curriculum warm-starts on the plentiful (repair SFT → DPO/RFT) then uses RL to produce wins (Kevin's result: 56%→82% correct via multi-turn RL).
4. **Hard negatives are first-class data, labeled.** Reward hacking is the dominant failure at small scale (Kevin §6.2: 7B copies reference, recycles reference output tensor, wraps in try/except). We *manufacture* these as labeled negatives so the reward model / DPO explicitly learns to reject them (Section 2.6).
5. **Conciseness of reasoning is a quality signal.** ConCuR's central finding: for a fixed task, the *shortest* CoT that achieves the *highest* speedup is the best training example. We adopt CoT-length × speedup as a curation score for reasoning traces (Section 3.6).

**Sources.** KORE.pdf (`/root/Kore-rl/plans/KORE.pdf`); Kevin arXiv:2507.11948 (https://arxiv.org/abs/2507.11948, https://cognition.ai/blog/kevin-32b); ConCuR arXiv:2510.07356 (https://arxiv.org/html/2510.07356v1); KernelBench arXiv:2502.10517; GEAK arXiv:2507.23194 (https://www.arxiv.org/pdf/2507.23194), ROCm GEAK blogs (https://rocm.blogs.amd.com/artificial-intelligence/geak-agents-family/README.html); KernelBook (https://huggingface.co/datasets/GPUMODE/KernelBook); MI350X / CDNA4 specs (https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html, CDNA4 white paper), with the legacy MI300X / MI325X / CDNA3 white paper kept only for reference.

---

## 1. Coverage matrix - operator families × dtypes × shape regimes × backends

### 1.1 Priority tiering (essential vs optional)

We rank operator families by Amdahl weight `f` (fraction of inference GPU time), because that is exactly the KORE objective (KORE.pdf §1). Data volume is allocated proportional to `f`, not uniformly.

| Tier | Rationale | Families |
|---|---|---|
| **P0 (essential - must saturate)** | Dominate inference time on every deployed model | Dense/quantized **GEMM** (fp16/bf16/fp8), **Flash-Attention prefill+decode** (causal, GQA), **fused MoE** GEMM+routing, **RMSNorm / fused-add-RMSNorm**, **SiLU/SwiGLU gated MLP**, **fp8 quant/dequant** (per-tensor + per-token/rowwise + block-scale) |
| **P1 (essential - high value, model-specific)** | Large `f` on flagship models | **MLA decode + prefill** (DeepSeek-V3), **paged-KV attention** (vLLM/SGLang), **RoPE** (incl. partial/NTK), **grouped/batched GEMM**, **softmax / online-softmax**, **LayerNorm**, **sampling** (top-k/top-p/argmax), **KV-cache write/gather/copy** |
| **P2 (optional - differentiators / frontier)** | Smaller `f` or emerging; give the model a moat | **Sliding-window / block-sparse attention** (VSA), **sparse-MLA decode**, attention **backward** (SLA bwd), **MoE MXFP4** (gfx950), **MXFP8 grouped GEMM** (gfx950), **comms-fusion** (all-reduce/all-gather + GEMM), **int8** GEMM, **speculative-decode** verify |
| **P3 (breadth - cheap synthetic only)** | Long tail; teaches idiomatic Triton | elementwise (add/mul/cast), reductions, cumsum, `argmax`, dropout, activations (gelu/relu/tanh), loss functions, conv (from KernelBench L1) |

**Rule:** P0 must have data in *every* dtype × shape-regime × backend cell that is physically meaningful (Section 1.5 marks the impossible cells). P1 must cover all shape regimes in bf16 + fp8. P2 gets primary+1 validation shape each. P3 is synthetic-only breadth (Section 3.5).

### 1.2 dtype axis (gfx950-correct)

| dtype | gfx950 encoding | SNR gate | Notes / edge |
|---|---|---|---|
| bf16 | native | **30 dB** | default accum fp32 |
| fp16 | native | **30 dB** (40 for pure GEMM per `gemm_fp16.yaml`) | overflow risk at large K |
| fp8 e4m3 | **`float8_e4m3fn` (OCP, max 448)** - NOT FNUZ (legacy gfx942 used `e4m3fnuz`, max 240) | **25 dB** | wrong variant silently mismatches AITER/hipBLASLt (`aiter_ref.py`) |
| fp8 e5m2 | `float8_e5m2` (OCP; legacy gfx942 used `e5m2fnuz`) | 25 dB | KV-cache / attention accumulation |
| int8 | native | 30 dB (exact-ish) | per-row/col scales; W8A8 |
| **mxfp4** (OCP microscaling, e2m1 + e8m0/32) | **gfx950 ONLY** (CDNA4 scaled MFMA) | 25 dB | native target capability; on legacy gfx942 it was emulated/dequant path only |
| **mxfp8** (OCP e4m3 + e8m0/32) | **gfx950 ONLY** | 25 dB | K must be multiple of 32 (scale group) |
| fp32 | native | 40 dB | reference/oracle only; rarely a kernel target |

**fp8 scale granularities to cover (all are distinct kernels):** per-tensor, per-token (rowwise, A), per-channel (colwise, W), 128×128 block-scale (DeepSeek), 1×128 / 128×1. These are separate coverage cells because the scale-application code path differs and is a top source of SNR failures.

### 1.3 Shape-regime axis

| Regime | Definition | Why it matters |
|---|---|---|
| **decode** | `seq_q = 1`, batch `1..N`, long KV (up to 32K-128K) | latency-bound, memory-bound, tiny-M GEMM; dominates serving |
| **prefill** | large `seq_q` (512-8192), full attention | compute-bound; MFMA utilization |
| **chunked-prefill / mixed** | `seq_q` in {128, 256, 512}, ragged batch | vLLM/SGLang default; varlen masking |
| **small-batch decode** | batch {1, 8, 64} | occupancy cliffs, tail effects |
| **peak-throughput** | batch {256, 512, 2048} | L2 thrashing (skill `moe_cache_thrashing_past_batch_480`) |

### 1.4 Backend axis

| Backend | Role in curriculum | Volume share |
|---|---|---|
| **Triton** | Primary. Fastest correct loop, plugs into verifier, full-source output (KORE.pdf §3). | ~70% of code volume |
| **HIP / CK** | Stage-2/3 for P0/P1 ops where Triton hits a ceiling (e.g. MLA ASM parity, block-sparse). Emit full `.hip`/`.hpp` source. | ~25% |
| **FlyDSL** | P2 only, where KernelForge shows wins (MoE stage1). Optional. | ~5% |

**Rule:** every P0/P1 op ships a Triton seed first (`seed_triton.py`), matching the existing KORE task layout. HIP/CK variants are added only after the Triton baseline and its AITER bar are measured (KernelForge phase discipline: Phase A measures baseline before any rewrite).

### 1.5 The concrete coverage table (with essential/optional and impossible cells)

Legend: ✅ essential (must have data), ➕ optional/differentiator, ⛔ physically N/A (dtype does not apply to this family), 🔷 MX path, native on gfx950/CDNA4 (the target; emulated only on legacy gfx942).

| Family | bf16 | fp16 | fp8-OCP | int8 | mxfp4/mxfp8 | decode | prefill | Triton | HIP/CK |
|---|---|---|---|---|---|---|---|---|---|
| GEMM dense | ✅ | ✅ | ✅ | ➕ | 🔷 | ✅(tiny-M) | ✅ | ✅ | ✅ |
| GEMM grouped (MoE) | ✅ | ➕ | ✅ | ➕ | 🔷 | ✅ | ✅ | ✅ | ➕ |
| GEMM batched | ✅ | ➕ | ➕ | ⛔ | ⛔ | ✅ | ✅ | ✅ | ➕ |
| Flash-attn prefill (causal, GQA) | ✅ | ➕ | ✅ | ⛔ | ⛔ | ⛔ | ✅ | ✅ | ✅ |
| Flash-attn decode (paged KV) | ✅ | ➕ | ✅ | ⛔ | ⛔ | ✅ | ⛔ | ✅ | ✅ |
| MLA decode (DeepSeek) | ✅ | ⛔ | ✅ | ⛔ | ⛔ | ✅ | ➕ | ✅ | ✅ |
| MLA prefill | ✅ | ⛔ | ➕ | ⛔ | ⛔ | ⛔ | ✅ | ✅ | ➕ |
| Sliding-window attn | ➕ | ⛔ | ➕ | ⛔ | ⛔ | ✅ | ✅ | ✅ | ➕ |
| Block-sparse attn (VSA) | ➕ | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | ✅ | ➕ | ✅ |
| Attn backward (SLA) | ➕ | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | ✅ | ➕ | ➕ |
| MoE fused (gate+up+SwiGLU) | ✅ | ➕ | ✅ | ➕ | 🔷 | ✅ | ✅ | ✅ | ➕ |
| MoE routing/scatter/sort | ✅ | ⛔ | ➕ | ⛔ | ⛔ | ✅ | ✅ | ✅ | ➕ |
| RMSNorm / fused-add-RMSNorm | ✅ | ➕ | ➕(quant-fused) | ⛔ | ⛔ | ✅ | ✅ | ✅ | ➕ |
| LayerNorm | ✅ | ➕ | ⛔ | ⛔ | ⛔ | ✅ | ✅ | ✅ | ➕ |
| SiLU/SwiGLU/GeGLU + mul | ✅ | ➕ | ✅(fused quant) | ⛔ | 🔷 | ✅ | ✅ | ✅ | ➕ |
| RoPE (full/partial/NTK) | ✅ | ➕ | ⛔ | ⛔ | ⛔ | ✅ | ✅ | ✅ | ➕ |
| Softmax / online-softmax | ✅ | ➕ | ⛔ | ⛔ | ⛔ | ✅ | ✅ | ✅ | ➕ |
| fp8 quant/dequant (all granularities) | ✅ | ➕ | ✅ | ✅ | 🔷 | ✅ | ✅ | ✅ | ➕ |
| KV-cache write/gather/reshape | ✅ | ➕ | ✅ | ⛔ | ⛔ | ✅ | ✅ | ✅ | ➕ |
| Sampling (top-k/top-p/argmax) | ✅ | ➕ | ⛔ | ⛔ | ⛔ | ✅ | ⛔ | ✅ | ➕ |
| Comms-fusion (AR/AG + GEMM) | ➕ | ➕ | ➕ | ⛔ | ⛔ | ✅ | ✅ | ➕ | ✅ |
| Elementwise/reduction (breadth) | ✅ | ✅ | ➕ | ➕ | ⛔ | ✅ | ✅ | ✅ | ⛔ |

### 1.6 Concrete shape lists (representative production model dims)

These are the exact shapes to bake into `task.yaml` `shapes:` blocks (`minimal` / `primary` / `validation` / plus `decode` and `prefill` variants). Hidden dims below are real config values.

**Model reference dims**

| Model | hidden | heads (q/kv) | head_dim | inter (MLP) | experts / top-k | vocab | notes |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 4096 | 32/8 (GQA) | 128 | 14336 | dense | 128256 | |
| Llama-3.1-70B | 8192 | 64/8 | 128 | 28672 | dense | 128256 | |
| Llama-4 Scout (MoE) | 5120 | 40/8 | 128 | 8192 | 16/1(+shared) | 202048 | interleaved |
| Qwen3-14B | 5120 | 40/8 | 128 | 17408 | dense | 151936 | KORE base |
| Qwen3-32B | 5120 | 64/8 | 128 | 25600 | dense | 151936 | KORE RL model |
| Qwen3-235B-A22B (MoE) | 4096 | 64/4 | 128 | 1536 (expert) | 128/8 | 151936 | |
| DeepSeek-V3 / R1 (MLA+MoE) | 7168 | 128/128→MLA | qk=576 (512+64), v=512 | 18432 dense / 2048 expert | 256/8 (+1 shared) | 129280 | kv_lora_rank=512, rope=64 |
| Mixtral-8x7B | 4096 | 32/8 | 128 | 14336 | 8/2 | 32000 | |

**GEMM (dense + fp8) - M×N×K.** Include square (compute-bound) and tall-skinny decode (memory-bound):
- Prefill/compute: `4096×4096×4096`, `8192×8192×4096`, `8192×14336×4096` (Llama MLP up), `4096×28672×8192` (70B), `2048×5120×5120` (Qwen3).
- Decode/tiny-M (seq_q=1..N): `M∈{1,8,16,32,64,128}` × `N,K` from projections: `N=6144,K=4096` (Llama-8B qkv fused), `N=4096,K=4096` (o_proj), `N=28672,K=8192` (70B gate_up), `N=headdim*…`.
- Vocab/logits GEMM: `M∈{1,64,2048}×N=128256×K=4096` (huge-N tail).
- **Edge shapes (must include):** `K=4095` (non-multiple-of-32 → fp8 illegal, must fail/guard), `K=8191`, `M=1` (pure GEMV), `N=1`, `M=17` (non-pow2 tail), giant-K `K=28672`.

**Attention (flash) - (batch, q_heads, kv_heads, seq_q, seq_kv, head_dim, causal):**
- Prefill: `(4,32,8,4096,4096,128,causal)`, `(2,16,16,2048,2048,128,causal)`, `(8,64,8,8192,8192,128,causal)`, `(1,1,1,256,256,128,noncausal)` [minimal].
- Decode (seq_q=1, long KV): `(bs,32,8,1,S,128)` for `bs∈{1,8,64,128}`, `S∈{1024,4096,16384,32768,131072}`.
- Head-dim edges: `head_dim∈{64,128,192(!),256}`; `192` (non-pow2-ish, DeepSeek qk path) and `256` stress LDS.
- GQA ratios: q/kv `∈{4:1, 8:1, 16:1, 1:1(MHA)}`.

**MLA decode (DeepSeek-V3 constants - do NOT vary; from `mla_deepseekv3_decode.yaml`):** `kv_lora_rank=512, qk_rope_head_dim=64, qk_head_dim=576, v_head_dim=512, page_size=16, nhead=128, nhead_kv=1`; `batch∈{1,4,8}`, `max_seqlen_kv∈{512,4096,32768}`. Prefill adds `max_seqlen_q=4096`.

**MoE - (tokens, experts, top_k, hidden, inter):**
- Decode worst-case: `(1, 256, 8, 7168, 2048)` (DeepSeek), `(1, 8, 2, 4096, 14336)` (Mixtral).
- Prefill scaling: tokens `∈{128, 512, 2048, 8192}`.
- Grouped-GEMM MoE (from `mxfp8_grouped_gemm.yaml`): `total_m=65536, hidden=2880, inter=5760, G=32`, plus the **real unbalanced trace** tokens-per-expert `[327,105,1843,2724,...,16053,...,14682,...]` (sum 65536, G=32) - this jagged/0-token/16K-giant-expert case is a mandatory edge (Section 2.2).

**Norm / activation - (M, N):**
- RMSNorm: `M∈{1,8,64,2048,4096,8192}` × `N∈{4096,5120,7168,8192}`; edge `N=8191` (non-pow2), `N=512` (tiny).
- SiLU+mul: input `(M, 2*inter)`; `inter∈{14336,11008,17408,25600,2048}`; `M∈{1,64,2048,4096,8192}`.

**RoPE - (batch, heads, seq, head_dim, rotary_dim):** full (`rotary_dim=head_dim=128`), partial (`rotary_dim=64` of 192, DeepSeek), NTK-scaled; `seq∈{1(decode),2048,8192}`.

**KV-cache ops:** page sizes `∈{1,16,32}`; block tables with holes; `num_blocks∈{...}`; dtype bf16 + fp8 KV.

---

## 2. Edge cases that MUST be in the data

Each edge case below is a *data requirement*: there must exist (a) at least one **verification shape/input** that exercises it, and (b) at least one **repair transition** (broken→fixed) whose failure is caused by mishandling it. Column "How to inject" maps to `kore/kore/data/mutate.py` mutators (existing or to-add).

### 2.1 Numerical edge cases

| # | Edge case | Must-have input/shape | How to inject (mutator) | Expected verdict |
|---|---|---|---|---|
| N1 | fp32-accumulator dropped (accumulate in bf16/fp16) over large K | `K≥8192` GEMM | `break_accumulator_dtype`, `break_dtype_cast` (exist) | SNR fail |
| N2 | Large-K accumulation overflow in fp16 | `K=28672`, fp16, large magnitudes | scale inputs ×1e2 in a fuzz seed | SNR fail / inf |
| N3 | fp8 scaling extremes: amax→0 (all-zero tile) and amax huge (→ clamp at FP8_MAX=448 for OCP `e4m3fn`; legacy gfx942 FNUZ clamps at 240) | per-tensor + per-token quant | zero-row input; ×1e4 input | must not NaN; SNR pass |
| N4 | Denormal / underflow in fp8 (values < smallest normal) | tiny magnitudes (1e-4) | fuzz seed | correct rounding |
| N5 | NaN/Inf propagation guard (softmax with -inf masked rows, all-masked row) | attention fully-masked row; RMSNorm zero-variance row | `break_eps` (exist, drops `+eps`) | must produce 0/defined, not NaN |
| N6 | Softmax overflow (no max-subtraction / online-softmax rescale bug) | large logits (seq 32K, scale off) | `break_scale` (exist) | SNR fail |
| N7 | Catastrophic cancellation in variance (RMSNorm/LayerNorm) | large mean + small var | reduction-order mutation | SNR fail |
| N8 | Wrong fp8 variant (legacy FNUZ `e4m3fnuz` instead of target OCP `e4m3fn`) | any fp8 task | new mutator `break_fp8_variant` | SNR fail vs AITER |

### 2.2 Shape edge cases

| # | Edge case | Must-have input | Notes |
|---|---|---|---|
| S1 | Non-power-of-2 dims | `N=8191`, `seq=4095`, `head_dim=192` | `tl.arange` needs pow2 length → tests masking, not tile=dim |
| S2 | Tiny | `M=1`, `N=1`, `seq=1`, `batch=1`, single expert token | GEMV/decode degeneracy, grid=1 |
| S3 | Huge | `K=28672`, `seq_kv=131072`, `total_m=65536` | multi-tile K, split-KV, HBM pressure |
| S4 | Ragged / varlen (cu_seqlens) | attention with `seqlens=[13,4096,1,777]` | varlen FMHA; must not read across sequence boundaries |
| S5 | Tail / masking (dim % BLOCK ≠ 0) | `M=4097` with BLOCK_M=128 | boundary mask on last tile |
| S6 | K not multiple of 32 for fp8/MX | `K=4095` fp8, `K=48` mxfp8 | MX scale-group constraint → must reject or pad-and-guard |
| S7 | Misaligned base pointers (offset by 1 elem) | slice a tensor `x[1:]` then pass | vectorized loads (dwordx4) misalign |
| S8 | Head-dim edge | `head_dim∈{64,192,256}` | 192 not pow2; 256 stresses LDS |
| S9 | 0-token / giant expert (MoE) | unbalanced trace incl. `M_g=0` and `M_g=16053` | jagged scale layout (skill `mxfp8_jagged_scale_layout`) |
| S10 | Batch-of-1 vs batch-of-2048 (occupancy) | both extremes per op | grid under/over-subscription |

### 2.3 Layout / stride edge cases

| # | Edge case | Must-have input | Notes |
|---|---|---|---|
| L1 | Transposed operands | GEMM `A^T`, `B^T`, both (KernelBench L1 16/17/18) | stride swap |
| L2 | Non-contiguous / sliced | `x[:, ::2]`, `x.transpose(0,1)` (no `.contiguous()`) | stride-aware loads |
| L3 | Permuted fp8 MFMA operand layout | fp8×fp8 and fp8×fp4 MFMA | `mfma_16x16x128_f8f6f4_operand_layout` skill: FP4 B natural, FP8 A **non-contiguous** per-lane; getting this wrong → subtle SNR fail (needs K-position probe, not sum-invariant test) |
| L4 | Weight layout `[N,K]` vs `[K,N]`, `trans_b` | fp8 GEMM `WQ[N,K]` (KORE `gemm_fp8_a8w8`) | matches AITER CK layout |
| L5 | Scale layout N-first `[G,N,K//32]` | mxfp8 grouped GEMM | "MATTERS for native v_mfma_scale lowering" (`mxfp8_grouped_gemm.yaml`) |
| L6 | Paged KV indirection (kv_indices) | MLA/paged-attn `page_size=16` | gather via block table |
| L7 | LDS swizzle / padding pitfall | fp8 permuted store | `triton_permuted_store_blocking_layout_pitfall`, `triton_fp8_permute_lds_tile` skills |

### 2.4 Concurrency / determinism edge cases

| # | Edge case | Must-have test | Notes |
|---|---|---|---|
| C1 | Cross-workgroup race | split-K GEMM with atomic accumulate; sparse attn BLOCK_M=64 | skill `sparse_block_m_128_guard`: BLOCK_M=64 causes **silent** cross-WG corruption; BLOCK_M=128 required. Data must contain the broken (=64) → fixed (=128) repair. |
| C2 | Atomic nondeterminism | atomic-add reduction | determinism rerun (5 seeds) must be stable within tol; flag if not |
| C3 | Race-only-at-scale | correct at seq=4096, corrupt at seq=65536 | multi-shape verification catches (Section 4.1) |
| C4 | LSE reduction order (MLA stage-2) | two-stage online softmax | `mla_online_softmax_with_lse` skill |

### 2.5 Resource / occupancy edge cases

| # | Edge case | Detection | Notes |
|---|---|---|---|
| R1 | VGPR spill (>256 VGPR/thread) | compiler output parse (`verifier/parsers/compiler_output.py`), rocprofv3 VGPR count | tile too big; skills `reduce_vgpr_reload_lds`, `moe_register_aliasing` |
| R2 | LDS overflow (>160 KiB/CU on gfx950) | compile fail / occupancy=0 | **hard gfx950 limit 160 KiB/CU** (legacy gfx942 was 64 KB; do not carry the 64 KB tile assumption onto the gfx950 target) |
| R3 | Occupancy cliff | rocprofv3 occupancy; a tile change halving waves/CU | skill `warp_tile_lds_coupling` |
| R4 | Register spill to scratch (silent slowdown) | perf regression w/ correct result | must be caught by speedup label, not correctness |
| R5 | num_warps/num_stages mis-tune (pipeline stall) | benchmark | `num_warps∈{4,8}`, tune `num_stages` (SYSTEM_PROMPT) |

### 2.6 Adversarial / reward-hacking negatives (LABELED)

These are the highest-leverage negatives (Kevin §6.2). Each is generated deliberately and stored with an explicit label so DPO/reward learns to *reject* it. Add a `RepairRecord.failure_class` value `"reward_hack"` and a hard-negative flag on candidates.

| # | Hack | How to detect (verifier rule) | Data use |
|---|---|---|---|
| H1 | **Copy the reference** / call the oracle | AST scan: kernel imports/calls `reference`, `matmul_ref`, `torch.matmul`, `F.*` | DPO `rejected`; label `reward_hack:copy_reference` |
| H2 | **Calls vendor lib** (aiter/rocBLAS/hipBLASLt/rocblas) | AST/string scan for `aiter.`, `hipblaslt`, `rocblas`, `torch.nn.functional` (SYSTEM_PROMPT already forbids) | DPO `rejected`; label `reward_hack:vendor_call` |
| H3 | **try/except fallback** to torch on kernel failure | AST: `try` wrapping the kernel call with a torch fallback | `rejected`; label `reward_hack:try_except_fallback` |
| H4 | **Recycle reference output tensor** (Kevin's harness bug) | run candidate FIRST, reference AFTER (KORE `driver.py` already loads candidate then computes ref - enforce ordering); allocate fresh output, poison-fill (NaN) input `out` | `rejected` + harness fix is mandatory |
| H5 | **Partial compute** passes weak check | fuzz multiple randomized inputs + multiple shapes; compute-only-corner still fails worst-shape SNR | negative + drives multi-shape gate |
| H6 | **Hardcoded output for the test shape** | randomized shapes at verify time (held-out shape not shown to model) | negative |
| H7 | **No-op / identity** when reference≈identity | include non-identity refs; SNR fail | negative |
| H8 | **Timing hack** (empty kernel, async not synced) | `torch.cuda.synchronize()` + CUDA events (KORE `bench` uses events + sync) | reject via correctness gate (speed only counts if correct) |
| H9 | **Format hack** (claims FULL_KERNEL but reuses parent) | dedup by source hash; `extract_kernel` returns parent | drop / negative |

**Volume target for hard negatives:** ≥ **8% of all Stage-2 DPO pairs** must have a reward-hack `rejected` side, spread across H1-H9, with H1/H2/H3 the majority (they are what a 14B base actually does). This is what prevents the 7B/14B reward-hacking collapse KORE.pdf and Kevin both document.

---

## 3. Data types, volumes, and mixing ratios per training stage

Record types (from `schemas.py`): **RepairRecord** (broken→fixed turn), **RankedGroupRecord** (k candidates + preference pairs), **WinRecord** (full winning trajectory). Plus two auxiliary corpora: **SyntheticPair** (KernelBook torch→Triton, `data/synthetic.py`) and **ReasoningTrace** (ConCuR-style CoT). We recommend the following counts. "Op-cells" = the ~24 essential (op × dtype × regime) cells from Section 1.5.

### 3.1 Target volumes (whole program)

| Corpus | Record type | Count (target) | Source split | Per op-cell |
|---|---|---|---|---|
| Synthetic torch→Triton (breadth) | SyntheticPair | **20,000-40,000** | KernelBook (18,162) + KORE-generated Inductor captures | broad |
| Repair transitions | RepairRecord | **60,000** | 70% injected-breakage (mutators), 30% natural teacher failures | ~2,500 |
| Reasoning traces (concise CoT) | ReasoningTrace | **6,000** | teacher (Opus) + self-gen, ConCuR-curated | ~250 |
| Ranked candidate groups | RankedGroupRecord | **12,000 groups** (k=3-7 → ~48k pairs) | 60% self-gen, 40% teacher | ~500 groups |
| Verified wins (RFT/SFT-on-wins) | WinRecord | **2,500-4,000** | RL + teacher evolve | ~120 |
| Hard negatives (labeled) | RepairRecord/DPO neg | **8,000** | deliberate injection (Section 2.6) | ~330 |

Rationale for the pyramid: it mirrors the natural scarcity (KORE.pdf §2: of ~300 audited trajectories only ~15-20 wins). Repairs and rankings are cheap to manufacture at scale; wins are expensive (each needs a real evolve loop with GPU benchmarks) so we keep them scarce but high-purity. KernelBook gives cheap breadth (correct-by-construction Inductor output).

### 3.2 Per-stage mixing ratios

**Stage 1 - repair-weighted SFT** (14B bring-up). Goal: raise compile+correctness far above base; learn to fix.

| Component | Ratio | Notes |
|---|---|---|
| RepairRecord (broken→fixed) | **55%** | the namesake weighting; weight `compile_fail : snr_fail ≈ 40:60` |
| SyntheticPair (torch→Triton) | **25%** | idiomatic Triton breadth; correct-by-construction |
| ReasoningTrace (concise CoT) | **12%** | ConCuR: short CoT + high speedup teaches good reasoning |
| WinRecord (SFT-on-wins) | **8%** | a few strong finals so the model sees "good" |

**Stage 2 - RFT + DPO** (learn to rank). 

| Component | Ratio | Notes |
|---|---|---|
| DPO pairs from RankedGroupRecord | **70%** | `faster-correct > slower-correct > incorrect > non-compiling` (`gen_groups.rank_candidates`) |
| - of which hard-negative pairs | **≥8% (of DPO)** | Section 2.6 |
| RFT (SFT on best-of-group + wins) | **30%** | `build_rft`: rank-0 candidate per group + win trajectories |

**Stage 3 - multi-turn GRPO** (32B, learn to win). Data is *on-policy* (generated by the current policy against the fixed verifier); the static corpus only seeds it.

| Component | Role |
|---|---|
| WinRecord (seed) | warm-start / off-policy replay buffer |
| On-policy trajectories | 16 parallel × 4 refinement turns (train), 8 turns (test) - Kevin recipe; each turn = one training sample; summarize prior CoTs to bound context; discounted intermediate reward across turns |
| Group-relative advantage | `A_i=(r_i-mean r)/(std r+ε)`; reward-conditioned + turn-level credit to avoid zero-advantage collapse on ties (KORE.pdf §3, refs [8][9][10]) |

### 3.3 Memory-bound vs compute-bound balance

Serving is dominated by memory-bound decode, but compute-bound prefill has the biggest per-kernel speedup ceilings. Split code-volume roughly:

| Regime | Share | Why |
|---|---|---|
| Memory-bound (decode GEMV, norms, activations, KV ops, MoE decode) | **55%** | matches inference time distribution; teaches bandwidth/occupancy reasoning |
| Compute-bound (prefill attention, square GEMM, MoE prefill) | **35%** | biggest MFMA-utilization wins; roofline reasoning |
| Latency-bound (tiny kernels, launch overhead, fusion) | **10%** | fusion opportunities (KernelBench L2 style) |

Label every task with `roofline_class ∈ {memory, compute, latency}` (computed from arithmetic intensity vs the gfx950 ridge point; use MAF ≈ 50% of peak per skill `roofline_use_maf_not_peak`: bf16≈1150, fp8≈2300 TFLOPs).

### 3.4 Difficulty distribution (curriculum)

Use **average reasoning-trace length as the difficulty proxy** (ConCuR's validated metric). Target distribution across the corpus:

| Difficulty | Share (SFT) | Share (RL) | Examples |
|---|---|---|---|
| Easy (single op, 1 tile) | 40% → 15% | elementwise, RMSNorm, SiLU, softmax |
| Medium (tiling/pipelining) | 40% → 45% | dense GEMM, fp8 GEMM, flash-attn prefill |
| Hard (fusion, multi-stage, sparse) | 20% → 40% | MoE fused, MLA decode, block-sparse, grouped GEMM |

Curriculum schedule: Stage 1 skews easy→medium; Stage 3 RL skews medium→hard (that is where wins are scarce and RL adds value). Anneal the mix over training steps.

### 3.5 Teacher (Opus) vs self-generated vs synthetic

| Source | Overall share | Where used | Why |
|---|---|---|---|
| **Synthetic (KernelBook / Inductor)** | ~35% by count | Stage 1 breadth only | cheap, correct-by-construction, but *not* AMD-optimized (torch.compile targets NVIDIA idioms) → breadth not depth |
| **Teacher (Claude Opus over KernelForge evolve)** | ~30% | Stage 1 repairs, Stage 2 seed groups, hard finals, CoT traces | high quality on hard ops; the scarce-win manufacturer pre-RL; expensive |
| **Self-generated (policy)** | ~35%, rising to ~100% by Stage 3 | Stage 2 groups (on-policy relabel), Stage 3 GRPO | keeps training on states the policy actually visits (DAgger argument, KORE.pdf ref [5]); prevents distribution shift |

Guidance: teacher share should *decline* across stages (30% → ~0% at Stage 3). Over-reliance on teacher wins overfits to teacher idioms; the KORE goal is to match the teacher's kernel quality at far lower cost, so RL on self-generated + verifier reward is what closes the gap.

### 3.6 Reasoning-trace curation (ConCuR rule, adapted)

For each task, sample **5** candidate (CoT, kernel) pairs from the teacher. Keep a trace iff:
- (a) it is the **shortest CoT among the 5 that also achieves the max measured speedup** (ConCuR part a, ~80% of traces), OR
- (b) speedup vs AITER baseline **> 1.5×** regardless of length (ConCuR part b - high-value kernels), OR
- (c) needed to balance single-op vs fusion paradigm ratio (ConCuR part c).

Strip chain-of-thought from prior turns in multi-turn contexts (summarize) to prevent context explosion (Kevin recipe). Store `cot_tokens` and `speedup` on every ReasoningTrace for later re-curation.

---

## 4. Quality, verification, and hygiene

### 4.1 Verification rigor (the gate that makes labels trustworthy)

Every correctness/perf label MUST be produced by this pipeline (extends `verifier/test.py`, `verifier/bench.py`, and each `driver.py`):

1. **Multi-shape correctness.** Score the **worst** shape, not the average (KORE.pdf §4: "score the worst shape"). Run `minimal + primary + all validation` shapes; a kernel passes only if the *minimum* SNR over shapes ≥ threshold. KORE `driver.run_correctness` already computes `worst = min(...)` over seeds - extend to iterate shapes too.
2. **Randomized-input fuzzing.** ≥5 random seeds per shape (KORE uses seeds `[0]` normally, `[0..4]` in `stability`/`determinism` mode - make ≥5 the default for corpus generation). Include the numerical stress seeds from Section 2.1 (×1e2, ×1e-4, zero-rows).
3. **Held-out verification shapes.** Verify on at least one shape **never shown to the model** (defeats hardcoding, H6). KORE.pdf §4 explicitly includes held-out shapes in the SNR check.
4. **SNR gate + allclose.** SNR ≥ 30 dB (bf16/fp16), ≥ 25 dB (fp8/MX), with `atol=rtol=1e-2` allclose as secondary (KernelBench convention). fp8 GEMM driver uses `atol=5e-1,rtol=5e-2` with SNR as the real gate.
5. **Determinism reruns.** Re-run the winning kernel ≥3× (fresh process). If SNR/wall variance exceeds tolerance (nondeterminism, C2), **flag and quarantine** - do not admit to corpus.
6. **Independent re-verification.** Re-verify accepted wins in a *separate process / separate harness invocation* (KORE.pdf §4: "re-verify independently"). Candidate is loaded and run **before** the reference (defeats output-recycling hack H4).
7. **Timing hygiene.** Warmup (≥10 iters) + median of ≥30 timed iters + CUDA-event timing + `torch.cuda.synchronize()` + **variance gate CV < 3%** (KORE.pdf §4). Reject measurements with CV ≥ 3%.
8. **Reward is lexicographic.** `r = 1[correct] · log(T_base / T_cand)`. Speed counts only if correct; a fast-but-wrong kernel scores 0. No dense/intermediate compile-or-run reward (KORE.pdf §4; prevents over-optimization cheating).
9. **Baseline = production op, measured on-box.** `--impl reference` calls AITER/hipBLASLt/rocBLAS/CK for `T_base`; never a torch fallback.

### 4.2 Anti-cheat AST/static gate (pre-execution)

Before a candidate is even benchmarked, run a static scan (cheap, deterministic). Reject + label as reward-hack (Section 2.6) if it: imports/calls `torch.nn.functional`, `torch.matmul`, `aiter`, `rocblas`, `hipblaslt`; references the `reference`/`matmul_ref` symbols; wraps the kernel entry in `try/except` with a non-kernel fallback; or the extracted source hash equals the parent's (no-op turn). This is GEAK's "call accuracy" idea plus Kevin's rule-based checks, run as a data filter.

### 4.3 Dedup / near-dup detection

- **Exact dedup:** `dedup_by_source_hash` (exists in `build_datasets.py`) on the representative source per record.
- **Near-dup:** add a normalized-AST hash (strip comments/whitespace/rename locals) + MinHash/Jaccard over token-shingles; drop pairs with Jaccard > 0.9 within the same op-cell. Prevents the corpus from being dominated by trivially-perturbed clones (a real risk with mutator-generated repairs).
- **Trajectory dedup:** two WinRecords whose `final_source` near-match collapse to one; keep the higher speedup.

### 4.4 Leakage control (held-out ops / shapes / arch)

Use `leakage_split(by=("operation","shape"))` (exists) so **no op×shape group crosses train/val/test**. Additionally hold out:
- **≥1 whole operator family** end-to-end (KORE.pdf §5 evaluates on "one held-out operator family"). Recommend holding out **grouped-GEMM MoE** or **MLA prefill** as the never-trained eval family.
- **Held-out shapes within trained ops** (Section 4.1 #3).
- **Held-out arch (OOD probe):** gfx950 is the *target* and is always trained on, never held out. The generalization split is by operator family (MLA latent attention + paged-attention KV-decode are the reserved families); if an arch OOD probe is wanted, reserve a small slice from a foreign arch outside the gfx950/gfx942 lineage. Never invert this by holding out gfx950.
- **Provenance-based holdout:** hold out entire source repos (e.g. all kernels from one GitHub repo in KernelBook) from train to prevent memorization.

### 4.5 Labeling & provenance (schema additions)

Extend each record with a metadata block (all cheap to compute, critical for curation and audits):

```
meta = {
  "operator_family": "gemm|attention|moe|norm|activation|rope|kvcache|quant|sampling|comms|elementwise",
  "dtype": "bf16|fp16|fp8_e4m3fn|fp8_e5m2|fp8_e4m3fnuz|fp8_e5m2fnuz|int8|mxfp4|mxfp8|fp32",  # OCP e4m3fn/e5m2 = gfx950 target; *fnuz = legacy gfx942
  "regime": "decode|prefill|chunked|small_batch|peak",
  "roofline_class": "memory|compute|latency",
  "difficulty": "easy|medium|hard",         # or cot_tokens bucket
  "backend": "triton|hip|ck|flydsl",
  "arch": "gfx950|gfx942",                  # gfx950 = target; gfx942 = legacy/reference
  "shape_key": "M4096_N4096_K4096",
  "provenance": {"source": "teacher|self|synthetic|kernelbook", "model": "...", "commit": "...", "license": "..."},
  "verify": {"snr_db_worst": 41.2, "shapes_tested": 5, "seeds": 5, "cv": 0.021,
             "reverified": true, "baseline": "aiter_gemm_a8w8", "baseline_wall_us": 812.0},
  "hard_negative": null | "reward_hack:copy_reference"
}
```

### 4.6 Keeping only trustworthy measured outcomes (admission policy)

A record is admitted to the training corpus iff: verified on ≥ N_shapes (≥3) and ≥5 seeds; passed the worst-shape SNR gate; timing CV < 3%; independently re-verified; passed the anti-cheat static gate; not a near-dup; carries full provenance. **Speedup labels missing a measured `baseline_wall_us` are dropped** (no teacher-claimed speedups). Quarantine (don't delete) rejects for later analysis and as hard negatives.

---

## 5. Existing datasets to reuse/adapt - and their gaps for AMD

| Dataset | Local path / URL | What it gives KORE | Gap for gfx950/AMD | Action |
|---|---|---|---|---|
| **KernelBench** (L1 100, L2 100, L3 50, L4 20) | `repos/KernelBench/KernelBench/level{1..4}` | 270 PyTorch reference modules; the *task specifications* (op + reference) and the `fast_p` metric | CUDA-oriented; baseline is **torch eager**, not AITER; no Triton/HIP AMD kernels; no fp8/MoE/MLA depth | Reuse the **reference modules as task specs**; re-target baseline to AITER on-box; port to KORE `task.yaml` + `driver.py`. Use L1 (basic ops) + L2 (fusion) for breadth; adopt `fast_p@1.2` as the win metric (KORE.pdf §5). |
| **GEAK-eval TritonBench-revised** (184) | `repos/GEAK-eval/geak_eval/data/TritonBench/data/TritonBench_G_v1/*` (185 files) | 184 real Triton kernels with harnesses; attention/GLA/retention/linear-attn breadth | Some kernels needed AMD fixes (GEAK: 37/184 failed on AMD - shared-mem errors, invalid HIP args); harnesses assume NVIDIA idioms | Reuse as **held-out eval + SFT breadth**; run through the AMD-fix pass GEAK did; do NOT leak into train (it's an eval bench). |
| **GEAK-eval ROCm bench** (30-31) | `repos/GEAK-eval/geak_eval/data/ROCm/data/ROCm_v1/*` (31 files) | Real AMD ROCm kernels: `gemm.py`, `layernorm.py`, `moe_gemm.py`, `rmsnorm_fwd/bwd.py`, `test_flashattention_fwd.py`, `test_chained_dot_fp8.py`, `test_matmul_MXFP.py` | Only 31 kernels; the *most* AMD-authentic set available but tiny | **Highest-value seed set.** Use as gold seeds for repair-mutation and as the primary held-out AMD eval. `test_chained_dot_fp8` and `test_matmul_MXFP` directly inform fp8/MX coverage. |
| **KernelBook** (18,162 torch↔Triton) | https://huggingface.co/datasets/GPUMODE/KernelBook (+ local `data/synthetic.py` regenerates via Inductor) | Massive breadth of correct-by-construction torch→Triton pairs; idiomatic Triton | Inductor targets NVIDIA; **not** MFMA/gfx950-tuned; no fp8/MoE/MLA; no perf-vs-AITER labels | Use for **Stage-1 breadth only** (25% of SFT). Regenerate on gfx950 with `torch.compile` to capture AMD-flavored Triton where possible. Never use for perf labels. |
| **ConCuR** (4,892 CoT+CUDA) | https://arxiv.org/abs/2510.07356 | The *curation method* (short-CoT×high-speedup) and evidence that concise CoT → better kernels | CUDA, not Triton/HIP; torch-eager baseline | Reuse the **curation pipeline** (Section 3.6), not the data. Regenerate Triton/HIP CoT traces with the Opus teacher on KORE tasks. |
| **KernelForge task suite** | `repos/KernelForge-main/tasks/*.yaml` | 9 production-grade AMD task specs with real shapes, constraints, phase gates, and measured results (MLA, MoE-MXFP4, MXFP8 grouped GEMM, flash-attn, sparse attn, SLA bwd, gemm fp16) | gfx950-targeted for MX ops; not a "dataset" but a spec source | **Directly port to KORE `task.yaml`** - these are the P1/P2 coverage cells with real dims + the unbalanced MoE trace. |
| **KernelForge knowledge_base/skills** (37 skills) | `repos/KernelForge-main/knowledge_base/skills/*.json` | Distilled AMD optimization + pitfall knowledge (VGPR/LDS/MFMA/MoE/MLA/sparse) | - | Mine as **tuning_hints** for prompts and as the source list for edge cases in Section 2 (each skill ⇒ a repair transition). |

### 5.1 Net-new data KORE must create (nobody else has it)

1. **AMD-baseline perf labels** - speedup vs AITER/hipBLASLt/rocBLAS/CK measured on gfx950. This is the moat; no public dataset has it.
2. **fp8-OCP correctness data** at scale (per-tensor/per-token/block-scale) - the OCP-vs-FNUZ trap (Section 2.1 N8, `aiter_ref.py`).
3. **MLA decode/prefill + paged-KV + MoE-fused** repair/win data with DeepSeek-V3 constants.
4. **Labeled reward-hack negatives** (Section 2.6) - deliberately manufactured, absent from all public sets.
5. **Multi-turn AMD evolve trajectories** with per-turn verifier feedback (the Stage-3 substrate).

---

## 6. Master edge-case checklist (implementation acceptance test)

The corpus is "done" only when every row below has ≥1 verification shape AND ≥1 repair transition:

- [ ] N1-N8 numerical (fp32-accum, large-K overflow, fp8 amax extremes, denormal, NaN/all-masked-row, softmax overflow, variance cancellation, fp8-OCP-vs-FNUZ)
- [ ] S1-S10 shape (non-pow2, tiny, huge, ragged/varlen, tail-mask, K%32≠0 fp8/MX, misaligned ptr, head-dim 64/192/256, 0-token+giant expert, batch extremes)
- [ ] L1-L7 layout (transposed, non-contiguous, permuted fp8 MFMA operand, weight [N,K]/trans_b, N-first scale layout, paged-KV indirection, LDS swizzle pitfall)
- [ ] C1-C4 concurrency (BLOCK_M=64 sparse corruption, atomic nondeterminism, race-only-at-scale, LSE reduction order)
- [ ] R1-R5 resource (VGPR spill >256, LDS overflow >160 KiB, occupancy cliff, scratch spill, num_warps/stages)
- [ ] H1-H9 reward-hack negatives, labeled, ≥8% of DPO pairs
- [ ] Every P0 op has data in every meaningful dtype×regime×backend cell (Section 1.5)
- [ ] ≥1 whole operator family held out end-to-end for eval
- [ ] fp8 tasks gate at 25 dB, bf16/fp16 at 30 dB; all baselines are on-box production ops

## 7. Direct implementation order (maps to existing code)

1. Port KernelForge `tasks/*.yaml` + GEAK-eval ROCm seeds → KORE `tasks/<id>/{task.yaml, reference.py, driver.py, seed_triton.py}` (follow `gemm_fp8_a8w8` layout). Fill the Section 1.5 P0/P1 cells first.
2. Extend `data/mutate.py` with mutators for N8 (`break_fp8_variant`), C1 (`break_block_m_to_64`), L-family (stride/layout), R2 (`break_lds_overflow`); wire families in `OP_FAMILY_MUTATORS`.
3. Add the anti-cheat static gate (Section 4.2) into `verifier/` and the metadata block (Section 4.5) into `schemas.py`.
4. Make multi-shape × ≥5-seed × held-out-shape × CV<3% the default in `driver.py`/`verifier/test.py`/`bench.py`.
5. Generate corpus at the Section 3.1 volumes with the Section 3.2 stage ratios; enforce Section 4.6 admission; split with `leakage_split`.
6. Curate reasoning traces via Section 3.6; label difficulty by CoT length.
7. Assemble Stage-1/2/3 mixes; verify against the Section 6 checklist before each stage.
