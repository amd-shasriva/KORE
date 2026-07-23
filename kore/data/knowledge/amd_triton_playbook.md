# AMD-Triton Optimization Playbook — MI350X (gfx950/CDNA4) & MI300X (gfx942/CDNA3)

**How to use:** (1) classify the bottleneck FIRST via roofline — compute / bandwidth / latency-occupancy; (2) change **ONE knob per iteration**; (3) verify with autotune timing AND the AMDGCN ISA before trusting. wave = **64 lanes** (num_warps=N → N·64 threads). AMD-only knobs take effect ONLY inside `triton.Config({...})` kwargs — as Python vars they are **silently ignored**.

## 1. Launch-config knobs (start → when to change)

| Knob | Grp | Start | Range | When to change |
|---|---|---|---|---|
| `num_warps` | std | **4** | 1,2,4,8 | **8 = #1 AMD perf bug** → VGPR spill to HBM, 3–5× slower. Go 8 only if VGPR-light & occupancy-bound. Memory-bound: 2/4 |
| `num_stages` | std | 2 GEMM / **1** FA | 1,2 | single GEMM 1–2; fused FA (2 dots) 1; elementwise/reduction 1. **Never 3–4** → buffers loads in LDS → occupancy cliff |
| `matrix_instr_nonkdim` | AMD | **16** | 0,16,32 | 16 = `mfma_16x16` (fewer AGPRs, higher occ, finer sched). 32 needs BLOCK_M & BLOCK_N %32==0; use only if it measurably wins on a big square shape |
| `waves_per_eu` | AMD | 0 (auto) | 0–8 | 2–3 GEMM, 3–4 memory-bound. Set target+1 when 1 granule over a boundary so LLVM shaves VGPRs; back off if it introduces spill |
| `BLOCK_M/N/K` | cexpr | 128,128,64 | pow2, **mult of 64** | primary lever. Min 64 (32 underutilizes MFMA / wastes lanes). Prune configs whose LDS bytes > cap |
| `GROUP_SIZE_M` | cexpr | **8** | 1,4,8,16 | multiple of **XCD=8** for L2 reuse; bigger → more reuse but worse balance on small grids |
| `SPLIT_K` | cexpr | 1 | 1,2,4,8,16 | skinny/decode (M≤64) to reach **≥1024 programs** across 304/256 CUs. Costs C zero-init + atomics; skip if M·N already ≥1024 tiles |
| `kpack` | AMD | 1 | 1,2 | `2` → `ds_read_b128` for fp16/bf16 GEMM with BLOCK_K≥64. **gfx942 only** — forced to 1 & warns on gfx950 |
| `OPTIMIZE_EPILOGUE` | env | **1** for GEMM | 0/1 | drops epilogue convert_layout (512B Tagram/CShuffle hotspot) |

Grid: target **≥1024 programs** to fill 8 XCDs (304 CU MI300X / 256 CU MI350X); tile count a multiple of 8 for XCD balance. `schedule_hint="attention"`/`"memory-bound-attention"` only for FA. `knobs.amd.use_block_pingpong` needs stages>1.

## 2. Occupancy / VGPR / LDS

- **512 VGPR/EU, 16-register granule**, 8 wave-slots/SIMD (cap 8 waves/EU). `waves_per_eu = min(8, floor(512 / round_up16(vgpr)))`. AGPRs (MFMA accumulator) come out of the SAME 512 budget → a fat accumulator silently caps occupancy.

| VGPR reserved | max waves/EU |
|---|---|
| ≤64 | 8 |
| 128 | 4 |
| 176 (e.g. 170 used) | 2 (176×3 > 512) |
| 256 | 2 |
| >256 | **1 — occupancy CLIFF** |

- Prefer **2 waves no-spill over 3-with-spill** (GEMM/attention). MFMA latency is hidden by the pipeline, not by many waves.
- **Scratch MUST be 0** — `.private_segment_fixed_size > 0` = spilling to HBM (3–5× slower).
- **LDS:** 64 KB/CU CDNA3 (32 banks) vs **160 KB/CU CDNA4** (64 banks, ~2× BW). LDS/stage ≈ `(BM·BK + BK·BN)·sizeof·num_stages`; keep **≤ ~80 KB for dual-occupancy** (2 wg/CU on CDNA4).

## 3. Memory — coalescing & LDS banks

- Emit **128-bit `global_load_dwordx4`** (4 fp32 / 8 bf16 / 16 fp8 per lane): base **16-byte aligned**, 64 lanes **contiguous** (`lane i → base+i`, innermost dim along the wave). Misalign / strided → downgrades to scalar `dword` = many transactions.
- Pad leading dims to 16-byte multiples. TN Tagram hotspot: if `K%256==0`, pad `lda=ldb=K+128`.
- **LDS bank conflict** = lanes hit different words in the same bank (**32 banks × 4 B = 128 B row**; N-way conflict = N× cost). Fix: **pad** so `(BK+PAD)%32 != 0` (keep 128-bit align), or **XOR-swizzle** `col ^= (row & mask)` (0 conflicts, 0 wasted LDS — preferred for GEMM). Want `ds_read_b128`/`ds_write_b128`, not `b32`.
- Masked tails: set `knobs.amd.use_buffer_ops` → `buffer_load_dwordx4` (HW bounds-check, no predication branch). **NOT default on many builds**. `knobs.amd.use_async_copy` (`global_load_lds`, skip VGPR staging) is default on gfx950.

## 4. MFMA / `tl.dot`

- **Always route inner products through `tl.dot`** → matrix cores (`v_mfma_*`). Never hand-roll a VALU reduction for a matmul.
- **fp32 accumulate**: MFMA accumulates fp32 in AGPRs even for bf16/fp16/fp8 — never down-cast the accumulator inside the K-loop; cast to output dtype only in the epilogue.
- **16×16 > 32×32**: `mfma_16x16x16` uses fewer AGPRs → higher occupancy + finer scheduling. 32×32 → larger accumulator → register pressure/spill, coarser latency hiding.
- Recommended BLOCK_K: fp16/bf16 **32–64**; fp8 **64–128**.
- Keep MFMAs back-to-back with **multiple accumulator sub-tiles** to hide systolic latency. Do NOT copy NVIDIA producer/consumer wave-specialization — AMD's static register alloc starves "producer" waves → ~80% peak ceiling. Prefer all-waves-compute (8-wave ping-pong / 4-wave interleave).

## 5. fp8 correctness trap (silent 2× error, not a crash)

- **gfx942 / CDNA3 = FNUZ**: use `tl.float8e4b8` (E4M3 fnuz, bias 8) / `tl.float8e5b16`. OCP `float8_e4m3fn` into `tl.dot` → `Unsupported conversion 'f8E4M3FN'`. fp8 MFMA = `v_mfma_f32_16x16x32_fp8_fp8`.
- **gfx950 / CDNA4 = OCP**: `e4m3fn` (bias 7, ±448) / `e5m2`. Re-cast FNUZ checkpoints. **TF32 removed** on CDNA4.
- Wrong dialect (exponent bias differs by 1) = **~2× silent value error**. vLLM/SGLang run `normalize_e4m3fn_to_e4m3fnuz` before the matmul.
- **SNR gate: 25 dB e4m3 / 20 dB e5m2** (fp8 e4m3 noise floor ~28 dB; >30 dB is physically impossible). Gate every fast path vs an fp32 reference (`err_ratio < 0.05`).
- `tl.dot_scaled` (gfx950 MXFP, E8M0 per-32-element scales) emits `v_mfma_scale_f32_32x32x64_f8f6f4` but runs **~24% slower** than plain `tl.dot` (scale-LDS `ds_read_u8` + `s_waitcnt` density). B scale layout is `[N, K//32]` — do NOT transpose it.

## 6. Pitfall library (each = a real failure gate)

- `BLOCK_M=64` **silently corrupts sparse attention** → use **128**.
- `tl.atomic_add` is **unordered across workgroups** → non-deterministic bwd dK/dV (and SPLIT_K) accumulation: passes `allclose`, fails exact match.
- **Reduced dim < 64 wastes lanes** in `tl.sum`/`tl.max` wave reduces → round the reduced dim to a **pow2 ≥ 64** (`BLOCK_SIZE = next_pow2(n_cols)`).
- AMD knobs (`matrix_instr_nonkdim`/`kpack`/`waves_per_eu`) as Python vars are **silently ignored** — must live in `triton.Config({...})`.
- `input_precision="tf32"` is **CDNA3-only** (removed CDNA4). Valid AMD values: `"ieee"`, (CDNA3) `"tf32"`. NVIDIA `"tf32x3"` is not an AMD path.
- Big tile ignoring the LDS cap → occupancy drops to 1 or compile fails. `num_stages=3/4` pipelines *worse* than 2. Clear `~/.triton/cache/` on substantial source edits; autotune on the target shape, not a proxy.

## 7. What good ISA looks like (`AMDGCN_ENABLE_DUMP=1`)

| Check | Good | Bad → retune |
|---|---|---|
| Global load | `global_load_dwordx4` / `buffer_load_dwordx4` | `global_load_dword` (scalar) |
| Masked tail | `buffer_load_*` (HW bounds) | `global_load_*` + `v_cmp` predication |
| LDS | `ds_read_b128` | `ds_read_b32` |
| MFMA | dense `v_mfma_f32_16x16x16` | sparse / gaps = starved core |
| Accumulator | stays in AGPR | `v_accvgpr_read/write` inside loop |
| Scratch | `.private_segment_fixed_size: 0` | nonzero = HBM spill (3–5× slower) |

Dump: `AMDGCN_ENABLE_DUMP=1 MLIR_ENABLE_DUMP=1 TRITON_PRINT_AUTOTUNING=1 TRITON_ALWAYS_COMPILE=1`. Grep `.vgpr_count`, `.group_segment_fixed_size` (LDS), `.private_segment_fixed_size` (scratch).

## 8. Measured transfer rules (tuning DB)

- `num_warps=4` is **~1.1× faster than 8** across MLA-decode / sparse-attn fwd / block-sparse — the general default.
- Sparse attention: **wpe=2 beats wpe=3** (SLA fwd 8.86 vs 13.40 ms).
- **VGPR=256 occupancy cliff**: waves 2→1 (step function). Keep ≤256.
- MoE: **pre-sort tokens by expert** assignment.
- MLA decode: load Q from global **per-MFMA** (L2 reuse), not LDS-pinned (saves ~72 VGPR); use **two-stage split-KV + reduce** for variable-length paged KV.

## 9. Honest caveat (set expectations)

- On **plain dense GEMM, tuned hipBLASLt / aiter usually win** — compiler backends (incl. Triton) under-perform hand-tuned asm/CK on CDNA3/4 GEMM & attention. Triton's real wins are **FUSION** (epilogue/prologue/attention) and **skinny split-K decode**.
- **Peak ≠ achievable**: sustained **~45–55% of matrix peak**; the bar is the best tuned library kernel, never theoretical peak. Theoretical peaks: MI300X FP16/BF16 **1307 TF**, FP8 **2615 TF**; MI355X FP16/BF16 **2.5 PF**, FP8 **5 PF**, FP6/FP4 **10 PF**.
- Author-time `@triton.autotune` does NOT reach a live sglang/vLLM path unless rebound through the aiter seam — then **e2e-gate** (`pct_gpu_time × speedup` beyond the noise band), not isolated TFLOPS.
