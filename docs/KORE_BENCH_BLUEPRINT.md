# KORE-Bench: an open benchmark of hard AMD ROCm/Triton kernel-optimization tasks

**Goal.** An open benchmark of hard GPU-kernel-optimization tasks for **AMD Instinct MI350X
(`gfx950` / CDNA4, the KORE target)**, where every task is graded against a **real production
vendor baseline** (AITER / hipBLASLt / Composable Kernel / AOTriton), not torch-eager. 200–400
tasks spanning every operator family that matters for LLM training and inference. Serves both KORE
RL training and the open-source / AMD community.

This document is the build spec. It is grounded in the KORE task ABI
(`kore/kore/tasks/base.py`, `_genops.py`, `aiter_ref.py`, `aiter_ref_attn.py`,
`registry.py`) and complements `DATASET_SPEC.md` (which covers the *training-record*
corpus). This doc covers the **task artifacts themselves** (the environments), the
authoring harness that produces them at scale, and the release plan for KORE-Bench.

---

## 0. What "task" means here, and what distinguishes KORE-Bench

A KORE task is a directory `kore/tasks/<id>/`:

- `task.yaml` - metadata: `task_id, operation, dtype, backend, gpu_target,
  seed_kernel_name, snr_threshold, shapes{minimal, primary, validation[]}, targets{snr_db,
  comparison_baseline}`.
- `reference.py` - the correctness **oracle**: `parse_shape`, `get_inputs(shape, device,
  seed)`, a torch-**fp32** `*_ref(...)`, and (for vendor tasks) the inputs pre-shaped to the
  vendor's exact layout.
- `seed_triton.py` - a *compiling but naive* starter kernel (the policy's edit target).
- `driver.py` - the verifier contract:
  - correctness: writes candidate to `kernel.py`, runs ≥5 reseeded trials, prints
    `SNR: <db>` / `allclose: <bool>` / `max_diff`.
  - `--bench-mode --impl {reference|candidate}`: cold-cache (L2-flushed) CUDA-event median
    timing; `reference` is the **real vendor op**; post-timing anti-hack re-verification on
    the cached module.

Five properties define KORE-Bench:

1. **Real-vendor baseline, measured on silicon.** `--impl reference` calls the exact kernel
   the production serving stack calls (AITER `flash_attn_varlen_func`, hipBLASLt via
   `tgemm.mm`, CK `gemm_a8w8`, AOTriton SDPA). Beating torch-eager has no production value; the
   bar is the production vendor kernel. Public kernel benchmarks (KernelBench, TritonBench,
   KernelBook) bench against torch-eager on NVIDIA.
2. **Headroom-verified.** A task is admitted only if a hand-written expert Triton/HIP kernel
   can get within, or beat, a defined fraction of the vendor kernel - i.e. the op has *real*
   optimization headroom vs the vendor kernel on `gfx950` (the target). Ops where AITER is already
   at roofline and unbeatable in Triton are dropped or demoted to a `parity` tier.
3. **gfx950/CDNA4-correct numerics.** fp8 = **OCP** (`float8_e4m3fn`, max 448), not legacy
   FNUZ (`e4m3fnuz`, max 240, the gfx942 encoding); MX (e2m1/e4m3 + e8m0/32) is
   **gfx950-only**; LDS ≤ 160 KiB on gfx950 (legacy gfx942 was 64 KB). These traps are baked
   into the tasks.
4. **Exhaustive, stratified coverage** across families × dtypes × shape-regimes × difficulty,
   with a completeness argument (§4) tied to the actual production op inventory (the ATOM
   `model_ops_guide` maps every LLM inference op to its AITER kernel - that map *is* our
   coverage target).
5. **Leakage-safe, released as an artifact.** Whole families/shapes plus a foreign-arch slice held out (gfx950 is the target, never held out); a public
   `KORE-Bench` release with a `fast_p` leaderboard vs vendor baselines that AMD and the
   community can run.

---

## 1. THE COMPLETE OPERATOR TAXONOMY (target 200-400 tasks)

Notation for each row: **op × variant × dtype × regime → (vendor baseline, oracle,
production shapes, difficulty tier)**.

- **Difficulty tiers:** `T1` easy (single tile, 1 pass), `T2` medium (tiling/pipelining/online
  softmax), `T3` hard (multi-stage fusion, paged indirection, grouped/jagged), `T4` frontier
  (MLA/sparse/MX/comm-fused, HIP/CK often required to beat vendor).
- **dtypes:** bf16, fp16, fp8=`e4m3fn`/`e5m2` (OCP; legacy gfx942 used FNUZ `e4m3fnuz`/`e5m2fnuz`), int8, mxfp4/mxfp8 (gfx950).
- **SNR gates:** bf16/fp16 30 dB (40 for pure fp32/GEMM-fp16), fp8/MX 25 dB.
- **Baseline column** gives the EXACT vendor symbol. Sources: `aiter_ref.py`,
  `aiter_ref_attn.py`, ROCm/ATOM `docs/model_ops_guide.md`, ROCm/aiter `op_tests/`,
  ROCm blogs, AOTriton README.

Model reference dims used throughout (real config values):

| Model | hidden | q/kv heads | head_dim | inter | experts/topk | vocab |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 4096 | 32/8 | 128 | 14336 | dense | 128256 |
| Llama-3.1-70B | 8192 | 64/8 | 128 | 28672 | dense | 128256 |
| Qwen3-14B / 32B | 5120 | 40/8, 64/8 | 128 | 17408 / 25600 | dense | 151936 |
| Qwen3-235B-A22B | 4096 | 64/4 | 128 | 1536 (expert) | 128/8 | 151936 |
| Mixtral-8x7B | 4096 | 32/8 | 128 | 14336 | 8/2 | 32000 |
| DeepSeek-V3/R1 (MLA+MoE) | 7168 | 128→MLA | qk=576(512+64), v=512 | 18432 / 2048 | 256/8(+1 shared) | 129280 |
| Llama-4 Scout (MoE) | 5120 | 40/8 | 128 | 8192 | 16/1(+shared) | 202048 |

### 1.1 GEMM family (target ~70 tasks)

The workhorse. Baseline discipline: **bf16/fp16 dense → hipBLASLt via
`aiter.tuned_gemm.tgemm.mm` / `torch.matmul`** (ROCm lowers bf16 matmul to hipBLASLt);
**quantized → CK `gemm_a8w8*` / `gemm_a4w4`**.

| # | op × variant | dtype | regime | vendor baseline (exact) | oracle | prod shapes (M×N×K) | tier |
|---|---|---|---|---|---|---|---|
| G1 | dense GEMM square | bf16 | prefill | `torch.matmul`→hipBLASLt (`hipblaslt_gemm_bf16`) | fp32 A@B | 4096³, 8192×8192×4096 | T2 |
| G2 | dense GEMM MLP-up | bf16 | prefill | hipBLASLt `tgemm.mm` | fp32 A@B | 8192×14336×4096, 4096×28672×8192 | T2 |
| G3 | dense GEMM fp16 | fp16 | prefill | hipBLASLt | fp32 (40 dB) | 4096³ (KF `gemm_fp16.yaml`) | T2 |
| G4 | skinny/tall-M decode GEMV | bf16 | decode | hipBLASLt | fp32 | M∈{1,8,16,32,64}×{N=6144,K=4096}(qkv), {4096,4096}(o) | T3 (tiny-M occupancy) |
| G5 | huge-N logits/LM-head | bf16 | decode+prefill | hipBLASLt `ParallelLMHead tgemm.mm` | fp32 | M∈{1,64,2048}×N=128256×K=4096 | T3 |
| G6 | fp8 per-tensor GEMM | fp8 e4m3fn | prefill+decode | `tgemm.mm` w/ scale_a,scale_b (hipBLASLt) | fp32 of dequant | 4096³, M∈{1..64}×4096×4096 | T2 |
| G7 | fp8 per-token/channel GEMM (a8w8) | fp8 | prefill+decode | `aiter.gemm_a8w8` / `gemm_a8w8_bpreshuffle` (CK) | fp32 of dequant | KORE `gemm_fp8_a8w8` shapes | T3 |
| G8 | fp8 block-scale 1×128 GEMM | fp8 | prefill+decode | `aiter.gemm_a8w8_blockscale_bpreshuffle` (CK) | fp32 blockwise dequant | DeepSeek 7168×… (128-block) | T3 |
| G9 | int8 W8A8 per-token GEMM | int8 | prefill+decode | `aiter.gemm_a8w8` (CK, `dtypes.i8`) | fp32 of dequant | 4096³, decode tiny-M | T2 |
| G10 | mxfp4 GEMM (a4w4) | mxfp4 | prefill | `aiter.gemm_a4w4` (CK, 1×32 e8m0) | fp32 dequant | **gfx950** 2880/5760 | T4 |
| G11 | grouped GEMM (MoE gate_up) bf16 | bf16 | prefill+decode | `aiter` grouped GEMM / `fused_moe` GEMM stage | fp32 per-group | KF `mxfp8_grouped_gemm` shapes bf16 | T3 |
| G12 | grouped GEMM mxfp8 (fwd/dgrad/wgrad) | mxfp8 | prefill | Primus-Turbo `grouped_gemm_mxfp8_kernel` (bf16 grouped baseline) | fp32 per-group | **gfx950** total_m=65536,h=2880,i=5760,G=32 + unbalanced trace | T4 |
| G13 | batched GEMM (BMM) | bf16 | prefill | `torch.bmm`→hipBLASLt | fp32 | attention proj batched | T2 |
| G14 | batched fp8 BMM (MLA proj) | fp8 | decode | `aiter batched_gemm_a8w8_a_per_token_group_prequant...` | fp32 | DeepSeek MLA q/k up-proj | T4 |
| G15 | split-K GEMM (tiny-M, giant-K) | bf16/fp8 | decode | hipBLASLt / CK | fp32 | M=16×N=4096×K=28672 | T3 (atomic race edge) |
| G16 | epilogue-fused GEMM (bias/act) | bf16/fp16 | prefill | torch matmul+bias+act chain (multi-kernel → hipBLASLt) | fp32 | `_genops` gemm_fusion 7 variants × {bf16,fp16} | T2 |
| G17 | GEMM + fp8 dequant/requant epilogue | fp8 | prefill | `tgemm.mm` + fused scale | fp32 | 4096³ | T3 |

**Shape edges (mandatory across G1-G17):** `K=4095` (fp8 illegal → must guard/reject),
`K=8191`, `M=1`/`N=1` (GEMV), `M=17`/`M=4097` (tail-mask), giant `K=28672`. Transposed
operands `A^T,B^T,both`; non-contiguous/sliced; weight `[N,K]` vs `[K,N]` (`trans_b`).

### 1.2 Attention family (target ~80 tasks - the richest)

Baselines: **prefill → `aiter.flash_attn_varlen_func` (CK/ASM FMHA) or
`aiter.flash_attn_func`; AOTriton SDPA (`F.scaled_dot_product_attention` on ROCm) as the
open baseline**. **Decode → `aiter.pa_fwd_asm` / `pa_persistent_fwd` / `pa_decode_gluon`
(paged).** **MLA → `aiter.mla.mla_decode_fwd` / `mla_prefill_fwd`.**

| # | op × variant | dtype | regime | vendor baseline (exact) | oracle | prod shapes (B,Hq,Hkv,Sq,Skv,D,causal) | tier |
|---|---|---|---|---|---|---|---|
| A1 | prefill MHA causal | bf16 | prefill | `aiter.flash_attn_func` (causal) | fp32 SDPA | (2,16,16,2048,2048,128,C) | T2 |
| A2 | prefill GQA causal (KORE `flash_attn_prefill`) | bf16 | prefill | `aiter.flash_attn_func` GQA | fp32 SDPA enable_gqa | (4,32,8,4096,4096,128,C),(8,64,8,8192³,128,C) | T3 |
| A3 | prefill MQA | bf16 | prefill | `aiter.flash_attn_func` (Hkv=1) | fp32 | (4,32,1,4096,4096,128,C) | T3 |
| A4 | prefill non-causal (bidir) | bf16 | prefill | `aiter.flash_attn_func` | fp32 | (1,16,16,512,512,128,NC) | T2 |
| A5 | varlen/ragged prefill (cu_seqlens) | bf16 | chunked | `aiter.flash_attn_varlen_func` | fp32 per-seq | seqlens=[13,4096,1,777] | T3 |
| A6 | chunked-prefill (mixed q) | bf16 | chunked | `flash_attn_varlen_func` | fp32 | Sq∈{128,256,512} ragged | T3 |
| A7 | head-dim edges | bf16 | prefill | `aiter.flash_attn_func` | fp32 | D∈{64,192(DSV3 qk),256} | T3 (LDS) |
| A8 | softcap (Gemma-2/Grok logit cap) | bf16 | prefill | `flash_attn_func(..., softcap=…)` | fp32 tanh-cap | (4,32,8,4096,…,128), cap=30/50 | T3 |
| A9 | ALiBi bias prefill | bf16 | prefill | `flash_attn_func` alibi_slopes | fp32 +bias | Hq=32 slopes | T3 |
| A10 | sliding-window attn (SWA) | bf16 | prefill+decode | `flash_attn_func(window_size=…)` / Triton unified | fp32 windowed | window∈{1024,4096} (Mistral) | T3 |
| A11 | decode paged (KORE `paged_attn_decode`) | bf16 | decode | `aiter.paged_attention_rocm` / `pa_fwd_asm` | fp32 gather | bs∈{1,8,64,128}, S∈{1k,4k,16k,32k,128k}, page=16 | T3 |
| A12 | decode paged persistent (block=1024) | bf16 | decode | `aiter.pa_persistent_fwd` | fp32 | bs∈{8,64}, S=32k, block=1024 | T4 |
| A13 | decode paged Triton (gluon) | bf16 | decode | `torch.ops.aiter.pa_decode_gluon` | fp32 | bs∈{1,64}, D≠128 or SWA | T3 |
| A14 | fp8-KV paged decode | fp8 e4m3fn/e5m2 | decode | `pa_fwd_asm` w/ fp8 KV (`reshape_and_cache_with_pertoken_quant`) | fp32 dequant KV | bs∈{8,64}, S∈{4k,32k} | T4 |
| A15 | GQA decode long-context | bf16 | decode | `pa_fwd_asm` | fp32 | (128,32,8,1,131072,128) | T3 |
| A16 | MLA decode (DeepSeek-V3) | bf16 | decode | `aiter.mla.mla_decode_fwd` (ASM); KF `mla_deepseekv3_decode` | fp32 2-stage LSE | kv_lora=512,rope=64,qk=576,v=512,nhead=128,kv=1,page=16, S∈{512,4k,32k} | T4 |
| A17 | MLA prefill | bf16 | prefill | `aiter.mla.mla_prefill_fwd` | fp32 | Sq=4096, DSV3 dims | T4 |
| A18 | MLA fp8-KV decode | fp8 | decode | `mla_decode_fwd` fp8 (KF phase D) | fp32 dequant | DSV3 fp8 KV per-head scale | T4 |
| A19 | sparse MLA decode (DSV4 DSA) | bf16/fp8 | decode | `aiter pa_sparse_prefill_*_opus` / KF `dsv4_sparse_mla_decode_hip` | fp32 masked | **gfx950** indexer top-k KV | T4 |
| A20 | block-sparse attn fwd (VSA) | bf16 | prefill | CK block-sparse (KF `vsa_sparse_attn_fwd`); Triton baseline | fp32 block-masked (cos≥0.9999) | (1,12,49152,128), sparsity 0.10, block_kv∈{64,128} | T4 |
| A21 | attention backward (SLA/FA bwd) | bf16 | prefill | AOTriton FA bwd / `aiter` bwd; KF `sla_bwd_attention` | fp32 autograd | (2,16,2048,128) dQ,dK,dV | T4 |
| A22 | prefill (AOTriton open baseline) | bf16/fp16 | prefill | AOTriton SDPA (`F.scaled_dot_product_attention`) | fp32 | D≤256, arbitrary S | T2 |

### 1.3 MoE family (target ~45 tasks)

Baselines: **`aiter.fused_moe.fused_moe` (CK 2-stage, weights pre-shuffled via
`shuffle_weight` at load, outside timing) / `asm_moe` (ASM)**; routing →
`topk_softmax`/`grouped_topk`/`biased_grouped_topk`; sort/align → `moe_sorting` /
`moe_align_block_size`.

| # | op × variant | dtype | regime | vendor baseline (exact) | oracle | prod shapes (tokens,E,topk,hidden,inter) | tier |
|---|---|---|---|---|---|---|---|
| M1 | fused MoE bf16 SiLU (KORE `fused_moe_silu`) | bf16 | prefill+decode | `aiter.fused_moe.fused_moe` (Silu, QuantType.No) | fp32 per-expert gated MLP | Mixtral (·,8,2,4096,14336); jagged 32-expert + 0-token | T3 |
| M2 | fused MoE bf16 GeLU/SwiGLU | bf16 | prefill | `fused_moe` (Gelu/Swiglu act) | fp32 | Llama-4 (·,16,1,5120,8192) | T3 |
| M3 | fused MoE fp8 per-token | fp8 | prefill+decode | `fused_moe` w/ QuantType per-token / `asm_moe` a16 | fp32 dequant | DeepSeek (·,256,8,7168,2048) | T4 |
| M4 | fused MoE fp8 per-tensor | fp8 | prefill | `fused_moe` per-tensor | fp32 | 235B (·,128,8,4096,1536) | T4 |
| M5 | fused MoE mxfp4 (w4a16) | mxfp4 | prefill | `aiter.fused_moe` mxfp4 / Triton `triton_kernel_moe_forward`; KF `moe_mxfp4` | fp32 dequant | **gfx950** gpt-oss dims | T4 |
| M6 | grouped-GEMM MoE (gate_up only) | bf16/fp8 | prefill | `aiter` grouped GEMM stage | fp32 grouped | total_m=65536,G=32 | T3 |
| M7 | shared-expert-fused MoE | bf16 | prefill+decode | `fused_moe` w/ shared-expert IDs appended | fp32 routed+shared | DSV3 (+1 shared), Llama-4 | T4 |
| M8 | top-k softmax router (KORE `topk_softmax`) | bf16→fp32 | both | `aiter.topk_softmax` | fp32 softmax+topk(+renorm) | (M,E)∈{(4096,8),(8192,256)} | T2 |
| M9 | grouped top-k router | bf16 | both | `aiter.grouped_topk` | fp32 group-then-topk | DSV3 group routing | T3 |
| M10 | biased grouped top-k (DSV3) | bf16 | both | `aiter.biased_grouped_topk` | fp32 +bias | DSV3 256/8 | T3 |
| M11 | MoE sorting / align-block-size | int32 | both | `aiter.moe_sorting` / `moe_align_block_size` | ref sort+pad | tokens=8192,E=256,block=64 | T3 (scatter) |
| M12 | MoE scatter/gather (permute tokens) | bf16 | both | aiter permute / FlagGems `moe` gather | fp32 index | jagged per-expert | T3 |
| M13 | MoE `moe_sum` reduce over topk | bf16 | both | FlagGems `moe_sum` | fp32 weighted sum | (M,topk,D) | T2 |
| M14 | SwiGLU+quant fusion (MoE a-operand) | fp8/mxfp8 | prefill | `fused_silu_mul_fp8_*` / KF Primus swiglu_quant | fp32 | emits [M,N//32] e8m0 | T4 |

**Mandatory MoE edges:** unbalanced trace `[327,105,1843,2724,…,16053,…,14682,…]` sum
65536 G=32 (0-token expert + 16K giant expert), `moe_cache_thrashing_past_batch_480`
(batch>480 L2 thrash), `sparse_block_m_128_guard` (BLOCK_M=64 silent cross-WG corruption →
must use 128).

### 1.4 Normalization + fusion family (target ~35 tasks)

Baselines: `aiter.rms_norm`, `rmsnorm2d_fwd`, `rmsnorm2d_fwd_with_add`,
`fused_add_rms_norm_cu`, `layernorm2d_fwd`, `layernorm2d_fwd_with_add`,
`fused_rms_fp8_per_tensor_static_quant`, `fused_rms_mxfp4_quant`,
`tensor_model_parallel_fused_allreduce_rmsnorm`.

| # | op × variant | dtype | regime | vendor baseline | oracle | shapes (M×N) | tier |
|---|---|---|---|---|---|---|---|
| N1 | RMSNorm (KORE `rmsnorm_aiter`) | bf16 | both | `aiter.rms_norm` | fp32 rms | M∈{1,8,64,2048,4096,8192}×N∈{4096,5120,7168,8192}; N=8191,512 | T1 |
| N2 | fused-add RMSNorm (KORE `fused_add_rmsnorm`) | bf16 | both | `aiter.fused_add_rms_norm_cu` (in-place resid) | fp32 | same; returns (normed,new_resid) | T2 |
| N3 | RMSNorm + fp8 quant | fp8 | both | `fused_rms_fp8_per_tensor_static_quant` | fp32 norm→fp8 | DSV3 7168 | T3 |
| N4 | RMSNorm + mxfp4 quant | mxfp4 | both | `fused_rms_mxfp4_quant` | fp32 norm→mxfp4 | **gfx950** | T4 |
| N5 | RMSNorm + pad | bf16 | both | `fused_add_rmsnorm_pad` | fp32 + pad | pad-to-multiple | T2 |
| N6 | RMSNorm backward | bf16 | train | `aiter`/GEAK `rmsnorm_bwd.py` | fp32 autograd | 4096×8192 | T3 |
| N7 | LayerNorm (KORE `layernorm`) | bf16 | both | `aiter.layer_norm`/`layernorm2d_fwd` | fp32 mean/var | 4096×8192 | T2 |
| N8 | fused-add LayerNorm | bf16 | both | `layernorm2d_fwd_with_add` | fp32 | 4096×8192 | T2 |
| N9 | AllReduce + RMSNorm (comm-fused) | bf16 | both | `tensor_model_parallel_fused_allreduce_rmsnorm` | fp32 (single-GPU AR=identity) | TP=1 measurement | T4 |

**Edges:** zero-variance row (must not NaN - `break_eps`), large-mean+small-var
(catastrophic cancellation), N non-pow2.

### 1.5 Quantization family (target ~30 tasks)

Baselines: `aiter.get_hip_quant(QuantType)`, `aiter.get_triton_quant`,
`aiter.pertoken_quant`, `aiter.dynamic_per_token_scaled_quant`,
`per_token_group_quant_fp8` (FlagGems). QuantTypes: `per_Tensor, per_Token, per_1x128,
per_1x32, per_Channel`.

| # | op × variant | dtype | regime | vendor baseline | oracle | shapes | tier |
|---|---|---|---|---|---|---|---|
| Q1 | dynamic per-token fp8 quant (KORE `quant_fp8_pertoken`) | fp8 e4m3fn | both | `aiter.dynamic_per_token_scaled_quant` | fp32 rowwise amax/scale | M×N∈{4096,8192}×{4096,7168} | T2 |
| Q2 | per-tensor static fp8 quant | fp8 | both | `get_hip_quant(per_Tensor)` | fp32 | 4096×8192 | T1 |
| Q3 | per-channel (colwise, W) fp8 quant | fp8 | load | `get_hip_quant(per_Channel)` | fp32 | N=4096 weights | T2 |
| Q4 | block-scale 1×128 fp8 quant | fp8 | both | `get_hip_quant(per_1x128)` | fp32 blockwise | DSV3 128-block | T3 |
| Q5 | per-token-group fp8 quant | fp8 | both | FlagGems `per_token_group_quant_fp8` | fp32 group | group=128 | T3 |
| Q6 | int8 per-token quant | int8 | both | `pertoken_quant(dtypes.i8)` | fp32 | 4096×8192 | T2 |
| Q7 | mxfp4 1×32 quant (e8m0 scale) | mxfp4 | both | `get_hip_quant(per_1x32)` / `fp4_utils.e8m0_shuffle` | fp32 | **gfx950** K%32==0 | T4 |
| Q8 | mxfp8 rowwise quant | mxfp8 | both | Primus `mxfp8_quant_kernels` | fp32 | **gfx950** [M,N//32] e8m0 | T4 |
| Q9 | fp8 dequant (unpack) | fp8→bf16 | both | aiter dequant | fp32 | 4096×8192 | T1 |
| Q10 | fused dual-quant (KF skill) | fp8 | both | `fused_dual_quant_hbm_reuse` | fp32 | HBM-reuse | T3 |
| Q11 | KV per-token quant write | fp8 | decode | `reshape_and_cache_with_pertoken_quant` | fp32 | paged KV | T3 |

**Edges (mandatory):** amax→0 all-zero tile (no NaN), amax huge (clamp at FP8_MAX=448 for OCP `e4m3fn`; legacy gfx942 FNUZ clamps at 240),
denormal/underflow (1e-4), **wrong fp8 variant** (legacy FNUZ `e4m3fnuz` instead of target OCP `e4m3fn` → SNR fail vs
AITER), K%32≠0 for MX (reject).

### 1.6 RoPE family (target ~15 tasks)

Baselines: `aiter.rope_fwd`, `aiter.rope_bwd`, `aiter.rope_cached_positions_2c_fwd_inplace`,
`fused_qk_rope_reshape_and_cache`, `fused_qk_rope_concat_and_cache_mla`.

| # | op × variant | dtype | regime | vendor baseline | oracle | shapes (B,H,S,D,rotary_dim) | tier |
|---|---|---|---|---|---|---|---|
| R1 | RoPE NEOX full (KORE `rope`) | bf16 | both | `aiter.rope_fwd` (rotate_style=0) | fp32 rotate | S∈{1,2048,8192}, D=128, rd=128 | T2 |
| R2 | RoPE GPT-J interleaved | bf16 | both | `aiter.rope_fwd` (rotate_style=1) | fp32 | D=128 interleaved | T2 |
| R3 | partial RoPE (DSV3 rd=64 of 192) | bf16 | both | `aiter.rope_fwd` (nope_first) | fp32 partial | rd=64, D=192 | T3 |
| R4 | RoPE + reshape + KV cache write | bf16 | decode | `fused_qk_rope_reshape_and_cache` | fp32 | paged, page=16 | T3 |
| R5 | fused QK-norm + RoPE + cache + quant | fp8 | decode | `fused_qk_norm_rope_cache_quant_shuffle` | fp32 | Qwen3 q_norm/k_norm | T4 |
| R6 | RoPE + MLA concat+cache | bf16 | decode | `fused_qk_rope_concat_and_cache_mla` | fp32 | DSV3 | T4 |
| R7 | RoPE backward | bf16 | train | `aiter.rope_bwd` | fp32 autograd | 2048×128 | T3 |
| R8 | NTK/YaRN-scaled RoPE | bf16 | both | `aiter.rope_fwd` scaled freqs | fp32 scaled | long-context freqs | T3 |

### 1.7 Activation + gating family (target ~25 tasks)

Baselines: `aiter.silu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul`,
`fused_silu_mul_fp8_per_tensor_static_quant`, `F.gelu`/`F.silu` (→ fused ROCm eltwise).
Note: AITER ships only **gated** GELU/SiLU, so standalone activations use the framework
(torch) production path (documented in `aiter_ref.py`).

| # | op × variant | dtype | regime | vendor baseline | oracle | shapes | tier |
|---|---|---|---|---|---|---|---|
| AC1 | SiLU+mul / SwiGLU (KORE `silu_mul`) | bf16 | both | `aiter.silu_and_mul` | fp32 silu(a)*b | (M, 2·inter), inter∈{14336,11008,17408,25600,2048} | T2 |
| AC2 | GeLU+mul / GeGLU | bf16 | both | `aiter.gelu_and_mul` | fp32 gelu(a)*b | inter dims | T2 |
| AC3 | GeLU-tanh+mul | bf16 | both | `aiter.gelu_tanh_and_mul` | fp32 | inter | T2 |
| AC4 | SiLU+mul+fp8 quant | fp8 | both | `fused_silu_mul_fp8_per_tensor_static_quant` | fp32→fp8 | inter, per-tensor | T3 |
| AC5 | SiLU+mul+mxfp4 quant | mxfp4 | both | `fused_reduce_act_mul_and_mxfp4_quant` | fp32→mxfp4 | **gfx950** | T4 |
| AC6 | gelu_tanh standalone (KORE `gelu_tanh`) | bf16 | both | `F.gelu(approximate=tanh)`→ROCm | fp32 | 4096×8192 | T1 |
| AC7 | gated activations (24 `_genops` unary) | bf16/fp16/fp32 | both | torch framework op | fp32 | 4096×8192 | T1 |

(The `_genops.py` unary/binary/reduce/fusion catalog already covers the breadth tail here
- 146 tasks. These stay as T1 breadth with torch-framework baselines.)

### 1.8 Sampling / softmax / top-k family (target ~20 tasks)

Baselines: `aiter.ops.triton.softmax.softmax`, `aiter.ops.triton.topk.topk`,
`aiter.mixed_sample_outer_exponential`, `torch.softmax`→ROCm, Triton
`rejection_greedy_sample_kernel`, FlagGems `scaled_softmax_forward/backward`.

| # | op × variant | dtype | regime | vendor baseline | oracle | shapes | tier |
|---|---|---|---|---|---|---|---|
| S1 | row softmax (KORE `softmax`) | bf16 | both | `torch.softmax`→ROCm | fp32 | (M,N)∈{(4096,8192),(2048,151936)} | T2 |
| S2 | online/streaming softmax | bf16 | both | `torch.softmax` | fp32 online | N=131072 (long) | T3 |
| S3 | log-softmax | bf16 | both | `torch.log_softmax`/FlagGems | fp32 | (·,vocab) | T2 |
| S4 | scaled softmax fwd/bwd | bf16 | train | FlagGems `scaled_softmax_*` | fp32 | attn logits | T3 |
| S5 | top-k (k small) | fp32 | decode | `aiter.ops.triton.topk.topk` | fp32 sort | (M,vocab), k∈{1,8,50} | T3 |
| S6 | top-p (nucleus) filter | fp32 | decode | aiter/vLLM top-p | fp32 cumsum-threshold | (M,vocab) | T3 |
| S7 | argmax / greedy sample | int | decode | `topk(...,1)` | fp32 argmax | (M,vocab) | T2 |
| S8 | temperature exponential sample | fp32 | decode | `aiter.mixed_sample_outer_exponential` | fp32 gumbel-max | (M,vocab), temps | T3 |
| S9 | rejection sample (spec-decode) | int | decode | Triton `rejection_greedy_sample_kernel` | ref sequential accept | draft×target | T4 |

### 1.9 Cross-entropy / loss family (target ~15 tasks - training-side)

Baselines: Liger `LigerCrossEntropyLoss`, `LigerFusedLinearCrossEntropyLoss`,
`LigerKLDIVLoss`, `LigerJSD`, `LigerFusedLinearJSD`, chunked-loss DPO/CPO/ORPO/SimPO/KTO.

| # | op × variant | dtype | regime | vendor baseline | oracle | shapes | tier |
|---|---|---|---|---|---|---|---|
| CE1 | cross-entropy fwd+bwd | bf16 | train | `LigerCrossEntropyLoss` | fp32 CE | (M,vocab)=(8192,128256) | T3 |
| CE2 | fused-linear cross-entropy (FLCE) | bf16 | train | `LigerFusedLinearCrossEntropyLoss` | fp32 chunked logits+CE | (M,H)×(H,vocab), chunked | T4 |
| CE3 | KL divergence | bf16 | train | `LigerKLDIVLoss` | fp32 KL | (M,vocab) | T2 |
| CE4 | JSD / fused-linear JSD | bf16 | train | `LigerJSD` / `LigerFusedLinearJSD` | fp32 | distill | T3 |
| CE5 | fused-linear DPO/ORPO/SimPO/KTO | bf16 | train | Liger `chunked_loss.*` | fp32 | pref pairs | T4 |
| CE6 | z-loss / logit-softcap CE | bf16 | train | Liger CE variants | fp32 | (M,vocab) | T3 |

### 1.10 KV-cache + reshape family (target ~15 tasks)

Baselines: `aiter.reshape_and_cache`, `reshape_and_cache_flash`,
`reshape_and_cache_with_pertoken_quant`, `concat_and_cache_mla`.

| # | op × variant | dtype | regime | vendor baseline | oracle | shapes | tier |
|---|---|---|---|---|---|---|---|
| KV1 | KV cache write (bf16) | bf16 | decode | `aiter.reshape_and_cache` | ref scatter | page∈{1,16,32}, vLLM layout | T2 |
| KV2 | KV cache write flash layout | bf16 | decode | `reshape_and_cache_flash` | ref | flash paged | T2 |
| KV3 | fp8-KV cache write + quant | fp8 | decode | `reshape_and_cache_with_pertoken_quant` | fp32 dequant | per-token scales | T3 |
| KV4 | MLA concat+cache | bf16 | decode | `aiter.concat_and_cache_mla` | ref | DSV3 latent+rope | T3 |
| KV5 | KV gather/copy (block table) | bf16 | decode | aiter gather | ref index | holes in block table | T3 |
| KV6 | paged block copy/defrag | bf16 | decode | vLLM block copy | ref | fragmented pages | T2 |

### 1.11 Comm-adjacent family (target ~8 tasks, single-GPU-measurable)

Baselines: `aiter` custom all-reduce / all-gather fused with norm/quant/GEMM. Measured with
TP=1 (comm = identity) so the benchmark is on the *fusion/compute* portion; multi-GPU
variants labeled and held for a distributed harness.

| # | op × variant | dtype | vendor baseline | oracle | tier |
|---|---|---|---|---|---|
| C1 | AR + RMSNorm | bf16 | `tensor_model_parallel_fused_allreduce_rmsnorm` | fp32 | T4 |
| C2 | AG + GEMM (sequence-parallel) | bf16 | aiter AG+GEMM | fp32 | T4 |
| C3 | AR + quant | fp8 | aiter AR quantized | fp32 | T4 |
| C4 | one-shot/two-shot custom all-reduce | bf16 | aiter custom_all_reduce | fp32 identity (TP=1) | T3 |

**Taxonomy totals (target):** GEMM 70, Attention 80, MoE 45, Norm 35, Quant 30, RoPE 15,
Activation 25, Sampling 20, Loss 15, KV-cache 15, Comm 8 ≈ **358 hand-baselined tasks**,
plus the existing 146 `_genops` breadth tasks with framework baselines. Cut/merge to the
200-400 window by dtype/regime pruning where headroom verification (§4.3) fails.

---

## 2. SOURCING AT SCALE - exact IDs, repos, conversion recipe, license

### 2.1 HuggingFace datasets

| ID | Rows | What it gives | Convert-to-KORE recipe | License |
|---|---|---|---|---|
| `ScalingIntelligence/KernelBench` | 270 | L1(100 basic ops)/L2(100 fusions)/L3(50 archs)/L4(20) PyTorch reference `Model` modules + `get_inputs` | Each `Model.forward` → `reference.py` `*_ref` (fp32) + `get_inputs`; re-target baseline from torch-eager to the AITER/hipBLASLt op the module lowers to; emit `task.yaml` + generic `driver.py`. Adopt `fast_p@1.2` metric. | MIT |
| `GPUMODE/KernelBook` | 18,162 | torch↔Triton pairs (Inductor-generated), license/star/commit metadata | Use as **seed_triton.py breadth** + SFT pairs only (NVIDIA-idiom, not MFMA-tuned); never for perf labels. Filter to permissive (`dataset_permissive.parquet`). Regenerate on gfx950 via `torch.compile` for AMD-flavored seeds. | per-repo (metadata-tagged); use permissive subset |
| `facebook/KernelLLM` (+ 8B model) | - | high-quality SFT kernel-gen data | SFT breadth; cross-check idioms | check card (Meta) |
| `GPUMODE/categorized_triton_data_permissive` | filtered | MIT-only Triton snippets categorized | seed_triton breadth by category | MIT |
| `ppbhatt500/kernelbook-triton-reasoning-traces` (+multiturn) | 170 | CoT traces for kernel gen | ReasoningTrace seeds (ConCuR curation) | check card |
| `BonnieWang/KernelBenchX` | 176 specs | Triton-gen task specs + LLM kernel corpus | additional task specs → `reference.py` | check card |

### 2.2 GitHub repos (task specs, seeds, shapes, oracles, baselines)

| Repo | Local path | What to mine | KORE mapping | License |
|---|---|---|---|---|
| **ROCm/aiter** | (clone) | `op_tests/test_*.py` (test_gemm_a8w8, test_mha, test_mla, test_moe, test_quant, test_pa_v1, test_rmsnorm2d, test_rope, test_layernorm, test_silu_and_mul, test_topk_softmax, test_pa_sparse_prefill_opus) | Each op_test = **exact shapes + input gen + the vendor call + a torch reference** → directly becomes `reference.py` oracle + `driver.py` `--impl reference` binding. This is the single richest source of vendor-baselined tasks. | MIT |
| **ROCm/ATOM** | - | `docs/model_ops_guide.md` = the op→AITER-kernel map (§1 baselines) | Coverage target + baseline bindings | MIT-ish (check) |
| **ROCm/composable_kernel** | - | CK example gemms/fmha/grouped-gemm, tuned configs | HIP/CK stage-2 baselines + shape lists | MIT |
| **ROCm/hipBLASLt** | - | GEMM problem sizes, epilogues | dense/fp8 GEMM baseline via `tgemm.mm` | MIT |
| **ROCm/aotriton** | - | FlashAttention fwd/bwd (V3 API, hdim≤256, varlen `PaddedVarlen`/`StridedVarlen`) | open attention baseline for A22/A21 (via torch SDPA on ROCm) | MIT |
| **ROCm/triton** | - | AMD Triton tutorials (matmul, fa, grouped) | seed_triton starters | MIT |
| **vllm-project/vllm** | `repos/vllm` | `vllm/model_executor/layers/{fused_moe,quantization,rotary_embedding}`, ROCm attention backends | production shapes + which aiter op each layer calls | Apache-2.0 |
| **sgl-project/sglang** | - | ROCm kernels, block-scale GEMM, MoE, AR | shapes + baselines | Apache-2.0 |
| **flagos-ai/FlagGems** | - | 216 Triton ops incl. `flash_attention_forward`, `flash_mla`, `fused_moe`, `moe_align_block_size(_triton)`, `grouped_topk`, `moe_sum`, `per_token_group_quant_fp8`, `rms_norm`, `rotary_embedding`, `scaled_softmax_*`, `topk` | seed_triton + a **second cross-check oracle**; ops list = coverage checklist | Apache-2.0 |
| **linkedin/Liger-Kernel** | - | RMSNorm, RoPE, SwiGLU, GeGLU, CrossEntropy, **FusedLinearCrossEntropy**, KLDiv, JSD, chunked DPO/CPO/ORPO/SimPO/KTO, Softmax, Sparsemax | the loss/norm/activation seeds + baselines (training-side family §1.9) | BSD-2 |
| **AMD-AGI/Primus-Turbo** | - | `grouped_gemm_mxfp8_kernel`, `mxfp8_quant_kernels`, `swiglu_quant_kernel` (KF `mxfp8_grouped_gemm` provenance) | gfx950 MX grouped-GEMM tasks | check |
| **KernelForge** | `repos/KernelForge-main/tasks/*.yaml` + `knowledge_base/skills/*.json` | 8 production task specs (MLA, VSA, SLA-bwd, MoE-MXFP4, MXFP8 grouped, gemm fp16) + 37 skills | direct port to `task.yaml`; skills → edge-case + tuning-hint source | internal |
| **GEAK-eval** | `repos/GEAK-eval/.../ROCm_v1` (31), `TritonBench_G_v1` (184) | real AMD ROCm kernels (gemm, layernorm, moe_gemm, rmsnorm fwd/bwd, flashattention_fwd, chained_dot_fp8, matmul_MXFP) + Triton breadth | **highest-value AMD seed set**; use ROCm_v1 as held-out AMD eval, TritonBench_G as SFT/held-out | check GEAK license |

### 2.3 Conversion recipe (repo op_test → KORE task)

For an AITER `op_tests/test_X.py`:
1. Extract the shape sweep (`l_mnk`, dims) → `task.yaml` `shapes{minimal,primary,validation}`.
2. Extract the input generator + quant helper → `reference.py:get_inputs(shape,device,seed)`.
3. Extract the **torch reference** in the test (`ref = ...`) → `reference.py:*_ref` (force fp32
   math, cast to task dtype).
4. Bind the vendor call (`aiter.X(...)`) → `driver.py` `--impl reference` (via a wrapper in
   `aiter_ref*.py`), pre-shuffling weights outside the timed region where required
   (`shuffle_weight`, `e8m0_shuffle`).
5. Set `snr_threshold` by dtype (30/25/40) and `comparison_baseline` = the aiter symbol.
6. Emit a naive `seed_triton.py` (from the per-family template, §3).

---

## 3. AUTHORING HARNESS - semi-automatic task generation

**Principle:** one declarative **task-spec** row → auto-generated
`reference.py + driver.py + seed_triton.py + task.yaml` from a **per-family template** + a
**vendor-baseline binding**. Only the hard, irreducible parts are hand-written. This
generalizes the existing `_genops.py` (which already templates unary/binary/reduce/fusion/
gemm_fusion with a torch baseline) to the **vendor-baselined** families.

### 3.1 Task-spec schema (`specs/<family>/<op>.yaml`)

```yaml
op: gemm_fp8_blockscale          # unique task stem
family: gemm                     # dispatches the family template
variant: block_1x128             # sub-template selector
dtypes: [fp8_e4m3fn]             # OCP (gfx950 target); legacy gfx942 used e4m3fnuz
regimes: [prefill, decode]       # cartesian-expanded
shapes:
  minimal: {M: 8,   N: 512,  K: 512}
  primary: {M: 4096, N: 4096, K: 4096}
  validation: [{M: 1, N: 4096, K: 4096}, {M: 64, N: 4096, K: 4096}]
snr_db: 25.0
oracle:                          # how to build the fp32 reference
  kind: gemm_dequant             # named oracle builder in oracles.py
  quant: {a: per_token, w: per_channel, block: [1,128]}
baseline:                        # the vendor binding (exact)
  symbol: aiter.gemm_a8w8_blockscale_bpreshuffle
  wrapper: aiter_ref.aiter_gemm_a8w8_blockscale
  preshuffle: [w]                # done outside timed region
  layout: {a: [M,K], w: [N,K], trans_b: true}
seed:                            # seed_triton generation
  template: gemm_blockscale      # per-family Triton skeleton
  headroom_note: "fuse dequant into MFMA epilogue"
edges: [K_not_mult_32_reject, amax_zero, misaligned_ptr]
arch: gfx950                     # target (gfx942 = legacy/reference)
tier: T3
provenance: {source: aiter_op_tests, test: test_gemm_a8w8.py, commit: <sha>}
license: MIT
```

### 3.2 The generator (`author.py`)

`author.py spec.yaml` → writes the task dir. Pipeline:
1. **Expand** dtypes × regimes → one task per cell (skip physically-impossible cells per the
   §1.5 coverage table of `DATASET_SPEC.md`: e.g. fp16 attention decode = ⛔).
2. **reference.py** = family template + the named **oracle builder** (`oracles.py`):
   `gemm_dequant`, `sdpa_causal_gqa`, `moe_topk_gated`, `rmsnorm`, `rope_neox`,
   `paged_gather_attn`, `quant_rowwise`, `cross_entropy`, `softmax`, `kvcache_scatter`, ….
   The oracle is always **fp32 math** matching the vendor op's numerics.
3. **driver.py** = the single generic driver (`_genops.driver_main`, extended) parameterized
   by (arity, entry_name, baseline wrapper, snr gate, atol/rtol, preshuffle list, edge
   guards). Correctness + cold bench + post-timing anti-hack are all inherited.
4. **seed_triton.py** = per-family Triton skeleton (naive but compiling) with the op's math
   inlined so the policy has real code to edit.
5. **task.yaml** = emitted from the spec.
6. **Validate** on-box (`author.py --check`): seed compiles, seed passes SNR gate vs oracle,
   vendor baseline runs, and **headroom probe** (§4.3) passes; else the cell is rejected.

### 3.3 What is templated per family vs hand-authored

| Component | Templated (auto) | Hand-authored (per family, once) | Truly per-task hand-work |
|---|---|---|---|
| task.yaml | ✅ from spec | - | shape choices in spec |
| driver.py | ✅ generic `driver_main` | baseline wrapper + edge guards (once) | - |
| reference.py get_inputs | ✅ family template | quant/layout helper (once) | rare bespoke input (jagged MoE trace) |
| reference.py oracle | ✅ named builder | the fp32 oracle math (once/family) | - |
| seed_triton.py | ✅ skeleton | family Triton skeleton (once) | for T4 (MLA/sparse/MX), a real HIP/CK seed is hand-written |
| vendor binding | ✅ from `baseline.symbol` | wrapper in `aiter_ref*.py` (once/op) | ASM dispatch quirks (MLA, persistent PA) |

**Rule of thumb:** T1/T2 (elementwise, GEMM, norm, activation, quant, rope, softmax) are
~100% templated once the family skeleton + oracle + wrapper exist. T3 (paged attn, MoE, FLCE)
need a hand-written oracle + wrapper but templated driver/yaml. **T4 (MLA, sparse, MX
grouped, comm-fused) require a hand-authored seed and often a hand-written HIP/CK reference**
- these are the ~40-60 genuinely-hand-crafted tasks; the other ~300 are semi-auto.

### 3.4 Reuse of existing code

- `_genops.py` `driver_main`, `_snr_db`, `_flush_l2`, `_time_fn`, `_load_candidate`
  (anti-hack module caching) are the driver core - extend, don't rewrite.
- `aiter_ref.py` / `aiter_ref_attn.py` are the wrapper library - grow it op-by-op.
- `base.py:Task.from_dir` already parses the yaml schema; `registry.py` already does the
  family + held-out split. New families just need `operator_family()` cases.

---

## 4. QUALITY + PROVABILITY ("best in world" argument)

### 4.1 Coverage-completeness argument

The claim "definitive" is made **relative to the production op inventory**: the set of ops a
real ROCm serving stack (vLLM/SGLang/ATOM on AITER) dispatches. That inventory is enumerated
in ATOM `docs/model_ops_guide.md` (every `nn.Module.forward` → its AITER kernel). KORE-Bench
is **complete** iff every row of that map has ≥1 task in every physically-meaningful
dtype×regime cell (the §1.5 coverage matrix in `DATASET_SPEC.md`, ✅/➕/⛔/🔷 legend). We ship
a `coverage_report.py` that diffs the authored tasks against the op-map and prints uncovered
cells → the completeness is *checkable*, not asserted.

### 4.2 Real-baseline requirement (hard gate)

A task is invalid unless `driver.py --impl reference` calls a **real vendor kernel** (AITER
symbol / hipBLASLt via `tgemm.mm` / CK / AOTriton SDPA). Framework-torch baselines are
allowed **only** for ops AITER genuinely doesn't provide standalone (dense softmax, standalone
GELU, elementwise breadth) - and these are explicitly labeled `baseline_class: framework` and
capped at the T1/T2 breadth tier. An anti-cheat static gate (from `DATASET_SPEC.md` §4.2)
rejects candidates that call the vendor lib / copy the reference.

### 4.3 Headroom verification (only keep ops with real headroom)

For each candidate task, before admission, run a **headroom probe**: an expert reference
Triton (and for T4, HIP/CK) kernel is benchmarked vs the vendor baseline on the primary
shape. Classify:
- `headroom` (keep, primary tier): expert kernel ≥ 1.1× *or* the vendor kernel is Triton
  (beatable) *or* a documented fusion removes an HBM round-trip → real optimization target.
- `parity` (keep, labeled `parity`): expert kernel lands 0.9-1.1×; still valuable as a "match
  the ASM" task (e.g. MLA decode vs ASM `.co`) but flagged so `fast_p` thresholds are set
  realistically.
- `no_headroom` (drop or demote): vendor kernel is a hand-tuned ASM at roofline and no Triton
  path gets within 2× (e.g. tiny elementwise where AITER = bandwidth-bound optimal). Dropped
  from the hard set; kept only as T1 breadth if pedagogically useful.

This makes "hard" **measured**: every admitted hard task has a demonstrated gap between the
seed and the vendor kernel, and a demonstrated *reachable* target.

### 4.4 Difficulty stratification

Every task carries `tier ∈ {T1,T2,T3,T4}` and `roofline_class ∈ {memory,compute,latency}`
(arithmetic intensity vs the gfx950 MAF ridge: bf16≈1150, fp8≈2300 TFLOPs). Target
distribution: T1 15%, T2 35%, T3 35%, T4 15% (skews harder than KernelBench, whose L1/L2 are
mostly T1/T2). The `fast_p` leaderboard is reported per-tier so progress on hard families is
visible (KernelBench collapses everything into one number and saturates at low `p`).

### 4.5 Verification rigor (per task, inherited from the driver)

Multi-shape worst-case SNR (min over `minimal+primary+validation`), ≥5 reseeded trials,
held-out verification shape not shown to the model, SNR gate (30/25/40 dB) + allclose,
cold-cache CUDA-event median with **CV<3%** variance gate, post-timing anti-hack
re-verification on the cached candidate module, reward lexicographic `r = 1[correct]·log(T_base/T_cand)`.
(All already implemented in `driver.py` / `_genops.driver_main`.)

### 4.6 Leakage-safe held-out split

`taxonomy.py` reserves two structurally-distinct attention leaves as whole families
(`mla` latent attention + `paged_attention` KV-decode), 43 exact stratified
near-generalization task/provenance roots, and any foreign architecture or unreviewed
dtype. Core attention still trains, so the product model stays strong at attention while
the eval measures both near-task and cross-family transfer. For the release we additionally:
- Hold out **≥1 more whole family end-to-end** (recommend **grouped-GEMM MoE**) as an extra
  never-trained generalization eval.
- Held-out **shapes within trained ops**.
- Held-out **arch** (a foreign non-`gfx950`/`gfx942` slice) as an OOD probe.
- **Provenance holdout**: reserve entire source repos (e.g. all GEAK-eval ROCm_v1 kernels)
  as eval-only to prevent memorization.
- Publish a frozen `test` split hash so leaderboard submissions can't train on it.

### 4.7 Release as an open artifact - **KORE-Bench**

- **Package**: `tasks/` dirs (task.yaml + reference.py + seed_triton.py + driver.py) + a
  `manifest.jsonl` (per-task: family, dtype, regime, tier, baseline symbol, shapes, arch,
  license, headroom class, provenance) + `aiter_ref*.py` wrappers + `coverage_report.py`.
- **Runner**: `kore-bench run --impl {candidate,reference} --split {train,test}` producing a
  `fast_p@{1.0,1.2,1.5,2.0}` leaderboard **per family and per tier**, vs the vendor baseline.
- **Repro**: pinned ROCm + AITER version (baselines updated with AITER v0.1.12+ MI355X
  configs), gfx950 (target) + gfx942 (legacy) rows separated.
- **License**: dataset artifacts under a permissive license (task specs original;
  seed/oracle code MIT/Apache); each imported seed retains upstream license in `manifest`.
- **Community/AMD value**: the only public benchmark that grades kernels vs the exact
  production AITER/hipBLASLt/CK/AOTriton kernels on Instinct silicon - usable to (a) track
  Triton-vs-vendor gaps AMD wants closed, (b) train/eval kernel-gen models, (c) file the
  headroom cases upstream to AITER.

---

## 4.8 What's already implemented (`kore/eval/`), vs. what's still aspirational above

Sections 1-4 above are a **build spec** for growing the taxonomy toward 200-400 tasks; the
*evaluation machinery* they assume is largely already real, in `kore/eval/`:

- **`fastp.py`** - the `fast_p` metric itself (fraction correct AND `>p`× faster than baseline).
- **`kernelbench_amd.py`** - the **KernelBench-AMD adapter**: `spec_to_task` maps a KernelBench-style
  problem (Level 1/2 `Model.forward` + input generator) onto a genuine KORE `Task` so it runs through
  KORE's own verified pipeline, and `to_kernelbench_report` renders a KORE result back into the
  field-standard `fast_p@{1.0,1.5,2.0}` shape for cross-comparison. Ships bundled offline fixtures
  (elementwise/gemm/fused) so the bridge is CPU-testable without a real KernelBench checkout.
- **`robust_eval.py`** - **Robust-kbench-style hardening** of the eval-time correctness verdict:
  reseeded random inits, the enumerated adversarial regimes, non-contiguous inputs, a differential
  fp64 oracle (catches a precision downgrade that plain `allclose` waves through), and the
  permutation/homogeneity/additivity metamorphic relations - all pure functions over a
  candidate/reference callable pair, so they're CPU/torch-testable without a GPU.
- **`paired_stats.py`** - **paired significance** for "KORE vs. baseline/Opus" claims: geometric-mean
  speedup ratio as the effect size, a paired bootstrap 95% CI, the exact sign test, and the Wilcoxon
  signed-rank test - because both sides are scored on the same held-out tasks under a matched budget
  (`bakeoff.py`/`vs_opus.py`), so per-task deltas are paired and far more powerful than an unpaired
  comparison.

What's still aspirational (not yet built): the `author.py` semi-automatic task generator (§3)
and the `coverage_report.py` completeness checker (§4.1). The registry inventory has moved
beyond this blueprint's original target; derive its current size, family counts, split reasons,
and digest with `kore.tasks.registry.taxonomy_description()` rather than treating this planning
document as a manifest. Tests: `tests/test_task_taxonomy.py`,
`kore/eval/tests/test_eval_frontier.py`, and `tests/test_korebench.py`.

---

## 5. PRIORITIZED BUILD ORDER

### 5.1 First ~50 tasks (highest value = Amdahl weight × headroom × already-have-baseline)

The registry has grown past this blueprint's original snapshot, and many of the
"first ~50" below already exist with a real vendor binding (marked `(have)`).
Re-derive the live inventory with `registry.taxonomy_description()` and actual task
metadata rather than this list before starting new authoring work.
Order:

1. **GEMM core (10):** G1 bf16 square, G2 MLP-up, G6 fp8 per-tensor, G7 fp8 a8w8 (have),
   G8 fp8 block-scale, G9 int8 W8A8, G4 skinny decode-GEMV, G5 huge-N logits, G16 epilogue-
   fused (have via `_genops`), G3 fp16.
2. **Attention core (10):** A2 GQA prefill (have), A1 MHA prefill, A11 paged decode (have),
   A14 fp8-KV decode, A10 SWA, A5 varlen, A8 softcap, A15 long-context GQA decode, A16 MLA
   decode (KF port), A22 AOTriton-baseline prefill.
3. **MoE core (7):** M1 fused SiLU (have), M3 fp8 per-token MoE, M8 topk_softmax (have),
   M10 biased grouped topk, M11 moe_sorting/align, M6 grouped-GEMM, M7 shared-expert.
4. **Norm + activation (8):** N1 RMSNorm (have), N2 fused-add RMSNorm (have), N3 RMS+fp8
   quant, N7 LayerNorm (have), N8 fused-add LayerNorm, AC1 SiLU-mul (have), AC4 SiLU-mul+fp8
   quant, AC6 gelu_tanh (have).
5. **Quant (5):** Q1 per-token fp8 (have), Q2 per-tensor fp8, Q4 block-scale 1×128, Q6 int8
   per-token, Q11 KV per-token quant.
6. **RoPE + KV (5):** R1 RoPE NEOX (have), R3 partial RoPE (DSV3), R4 RoPE+reshape+cache,
   KV1 KV write, KV3 fp8-KV write.
7. **Sampling + loss (5):** S1 softmax (have), S5 top-k, S8 temperature sample, CE1 cross-
   entropy, CE2 FLCE.

Each first-50 task must pass §4.3 headroom + §4.5 verification before counting as "done".

### 5.2 Path to hundreds

- **Phase A (weeks 1-3): harness.** Build `author.py`, `oracles.py`, per-family seed
  templates, extend `driver_main`; grow `aiter_ref*.py` wrappers for the first-50 baselines.
  Port the 8 KernelForge task specs. → ~50 tasks, harness proven.
- **Phase B (weeks 3-6): auto-expand cells.** Cartesian-expand the first-50 specs across
  dtypes × regimes × shape-edges via `author.py`; mine AITER `op_tests/test_*.py` into specs
  (test_gemm_a8w8, test_mha, test_mla, test_moe, test_quant, test_pa, test_rmsnorm2d,
  test_rope, test_layernorm, test_silu_and_mul, test_topk_softmax). → ~200 tasks.
- **Phase C (weeks 6-9): frontier T4 + breadth.** Hand-author MLA (decode/prefill/fp8),
  sparse MLA (DSV4), VSA block-sparse, SLA bwd, MX grouped-GEMM (gfx950), comm-fused; port
  Liger losses (FLCE/JSD/DPO) and FlagGems MoE/quant ops; fold in KernelBench L1/L2 re-
  baselined + GEAK ROCm_v1 as held-out. → **300-400 tasks**.
- **Phase D (weeks 9-10): release.** Run `coverage_report.py` to green every ✅ cell; freeze
  held-out `test` split; publish KORE-Bench artifact + `fast_p` leaderboard.

### 5.3 Acceptance gate for "definitive"

Ship only when: every ATOM op-map row covered in every ✅ dtype×regime cell; every hard task
headroom-verified; T1/T2/T3/T4 = 15/35/35/15; ≥1 family + a foreign-arch slice + ≥1 source-repo held
out; fp8=OCP e4m3fn everywhere (FNUZ only as the legacy gfx942 reference); MX labeled gfx950; all baselines are on-box vendor ops; leaderboard
reproducible on pinned ROCm+AITER.

---

## Appendix: exact vendor-baseline symbol index (grep-able)

`aiter.tuned_gemm.tgemm.mm` (hipBLASLt dense/fp8-per-tensor) · `aiter.gemm_a8w8` (CK
int8/fp8 per-token) · `aiter.gemm_a8w8_bpreshuffle` (fp8 per-token) ·
`aiter.gemm_a8w8_blockscale_bpreshuffle` (fp8 1×128) · `aiter.gemm_a4w4` (mxfp4 1×32) ·
`aiter.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant` (MLA proj) ·
`aiter.flash_attn_func` / `aiter.flash_attn_varlen_func` (CK/ASM FMHA prefill) ·
`aiter.paged_attention_rocm` / `aiter.pa_fwd_asm` / `aiter.pa_persistent_fwd` /
`torch.ops.aiter.pa_decode_gluon` (decode) · `aiter.mla.mla_decode_fwd` /
`aiter.mla.mla_prefill_fwd` · `aiter.pa_sparse_prefill_fp8_opus` (DSV4 sparse) ·
`aiter.reshape_and_cache` / `reshape_and_cache_flash` /
`reshape_and_cache_with_pertoken_quant` / `concat_and_cache_mla` ·
`aiter.fused_moe.fused_moe` / `aiter.fused_moe_bf16_asm.asm_moe` · `aiter.topk_softmax` /
`grouped_topk` / `biased_grouped_topk` · `aiter.moe_sorting` / `moe_align_block_size` ·
`aiter.rms_norm` / `rmsnorm2d_fwd` / `rmsnorm2d_fwd_with_add` / `fused_add_rms_norm_cu` /
`fused_add_rmsnorm_pad` · `aiter.layer_norm` / `layernorm2d_fwd` /
`layernorm2d_fwd_with_add` · `fused_rms_fp8_per_tensor_static_quant` /
`fused_rms_mxfp4_quant` · `aiter.silu_and_mul` / `gelu_and_mul` / `gelu_tanh_and_mul` /
`fused_silu_mul_fp8_per_tensor_static_quant` / `fused_reduce_act_mul_and_mxfp4_quant` ·
`aiter.rope_fwd` / `rope_bwd` / `rope_cached_positions_2c_fwd_inplace` /
`fused_qk_rope_reshape_and_cache` / `fused_qk_rope_concat_and_cache_mla` /
`fused_qk_norm_rope_cache_quant_shuffle` · `aiter.get_hip_quant(QuantType)` /
`pertoken_quant` / `dynamic_per_token_scaled_quant` · `aiter.mixed_sample_outer_exponential`
/ `aiter.ops.triton.topk.topk` / `aiter.ops.triton.softmax.softmax` ·
`tensor_model_parallel_fused_allreduce_rmsnorm` (comm-fused).

**Sources:** KORE `kore/kore/tasks/{aiter_ref.py, aiter_ref_attn.py, _genops.py, base.py,
registry.py}`; ROCm/ATOM `docs/model_ops_guide.md`; ROCm/aiter `op_tests/` + README + ROCm
blog (AITER); ROCm/aotriton README (0.12b, PyTorch SDPA); linkedin/Liger-Kernel; flagos-ai/
FlagGems (216 ops, Apache-2.0); GPUMODE/KernelBook + ScalingIntelligence/KernelBench (HF);
KernelForge `tasks/*.yaml` + `knowledge_base/skills/*.json`; GEAK-eval ROCm_v1 + TritonBench_G.
