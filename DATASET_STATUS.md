# KORE Frontier Dataset — Status (offline, Crusoe-staged, curriculum-enriched)

Built OFFLINE on Crusoe for AMD **CDNA4 / gfx950 / MI355X** kernel generation.
Base model **Qwen/Qwen3-14B** @ `40c069824f4251a91eefaf281ebe4c544efd3e18`.
Reproducible, decontaminated, and enriched with a novel kernel-curriculum layer.

## Per-stage artifacts (packaged under `data/release/`, gzip+split 90MB parts)

| Stage | Rows | Tokens | Source file | Package |
|---|---|---|---|---|
| Midtrain (Stage-0 CPT) | 86,010 chunks | 170.2M | data/b05factory/midtrain/corpus.jsonl (683MB) | data/release/midtrain/ |
| SFT (Stage-1 multicap) | 56,493 | 190.4M | data/b05factory/sft/multicap.jsonl (630MB) | data/release/sft/ |
| DPO (Stage-2, 12% hard neg) | 96,675 pairs | 301.4M | data/b05factory/dpo/pairs.jsonl (1.09GB) | data/release/dpo/ |
| Kernel-curriculum (folded into SFT + midtrain) | 9,965 | 13.3M | kore_offline/curriculum_all.jsonl | data/release/curriculum/ |
| Provenance (wins/groups/repair/agentic) | — | — | data/b05factory/{wins,groups,repair,agentic}/ | data/release/provenance/ |

**Total ~662M training tokens** (>1B effective at 3 CPT epochs). Reassemble: `cd data/release && ./reassemble.sh`

## Kernel-curriculum layer (novel, gfx950-only, teacher = claude-opus-5 via AMD gateway)
Fills the GPU-code scarcity gap (CUDA/HIP <0.01% of pretraining data; Kevin-32B). Tiers:
- **Tier 1 — verifiable analytical kernel-math (8,873)** ⭐ a deterministic solver computes ground
  truth (roofline P=min(pi,beta*I), occupancy, tiling bm*bn/(bm+bn), MFMA/peak TFLOPS, FP8 OCP-format);
  the teacher writes prose and every answer is **auto-verified against the computed number** (≈90% pass;
  wrong answers rejected). No hallucinated math — unique to this dataset.
- **Tier 4 — reasoning traces from OUR real verified wins (724)** ⭐ grounded in real measured gfx950
  speedups (nobody else has real-silicon data at scale).
- Tier 2 grounded QA (298, OSS-Instruct on real KernelBook kernels), Tier 3 concept/curriculum (41,
  also as midtrain pretraining text), Tier 5 evol-instruct (20), Tier 6 distractors (9, pre-RL
  reward-hack immunity). Generators committed under data/release/generators/ for reproducibility.

## Midtrain corpus composition
base 76,054 (kernelbook 21,432 · rocm_hip 14,063 · general_replay 21,486 · docs 8,543 · amd_kernels
5,638 · triton 2,325 · pytorch_triton_pairs 1,784 · kore_tasks 539 · amd_asm 244) + **9,956
kernel_curriculum chunks** = 86,010. 27 verified gfx950 device-code repos; decontaminated vs 6 benches.

## External data (staged offline on Crusoe)
Qwen3-14B (28G) · GPUMODE/KernelBook @ b76504d8 · GPUMODE/kernelbot-data @ 4159cf6b · 28k general
replay · source catalog (data/release/meta/source_metadata.json) · frozen decontam artifact
(data/release/meta/benchmark_artifact.json.gz, 6 benches/20,148 recs) · 5/6 eval benches cached.

## Known deltas (honest)
- Grounded curriculum tiers (QA/concept/evol/distractor) are modest volume — the shared teacher gateway
  rate-limited long-prompt generation; the verifiable Tier-1 math moat is the bulk.
- CDNA4/gfx950 only (CDNA3 out of scope per current goal).
