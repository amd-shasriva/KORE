# KORE Frontier Dataset — Status (offline, Crusoe-staged)

All stages built OFFLINE on Crusoe (gfx950 / CDNA4). Base model **Qwen/Qwen3-14B**
@ `40c069824f4251a91eefaf281ebe4c544efd3e18`. Fully reproducible + decontaminated.

## Per-stage artifacts (packaged under `data/release/`, gzip+split 90MB parts)

| Stage | Rows | Tokens | Source file | Package |
|---|---|---|---|---|
| Midtrain (Stage-0 CPT) | 76,054 chunks | 157.6M | data/b05factory/midtrain/corpus.jsonl (650MB) | data/release/midtrain/ (2 parts) |
| SFT (Stage-1 multicap) | 45,409 | ~140M | data/b05factory/sft/multicap.jsonl (592MB) | data/release/sft/ (2 parts) |
| DPO (Stage-2, 12% hard neg) | 96,675 pairs | ~300M | data/b05factory/dpo/pairs.jsonl (1.09GB) | data/release/dpo/ (1 part) |
| Provenance (wins/groups/repair/agentic) | — | — | data/b05factory/{wins,groups,repair,agentic}/ | data/release/provenance/ (2 parts) |

Reassemble: `cd data/release && ./reassemble.sh`

## Midtrain corpus composition (real, offline, decontaminated)
kore_tasks 539 · pytorch_triton_pairs 1,784 · kernelbook 21,432 · amd_kernels 5,638 ·
triton 2,325 · rocm_hip 14,063 · amd_asm 244 · docs 8,543 · general_replay 21,486
(general_frac 0.2825; near-dedup dropped 56,202; 27 verified device-code repos incl.
pytorch/cutlass/MIOpen/Tensile/ROCm libs + KernelBook + AMD kernelbot submissions).

## SFT mix (canonical, decontaminated vs 6 eval benches)
kernel_repair_opt 0.43 · general_chat 0.18 · general_code 0.13 · math_reasoning 0.13 ·
agentic_tooluse 0.13 · kernel_qa ~0 (teacher-gated slice deferred). general retention 0.44.

## External data (all staged offline on Crusoe; NOT re-fetched at train time)
- Qwen3-14B weights+tokenizer (28G, HF cache) @ 40c06982...
- GPUMODE/KernelBook @ b76504d8... (18,162 pairs materialized)
- GPUMODE/kernelbot-data amd_successful_submissions @ 4159cf6b... (60,357 kernels)
- General replay 28,000 rows (OpenCodeInstruct/OpenThoughts3/tulu-3/ToolACE)
- Source catalog: data/release/meta/source_metadata.json (28 sources + 2 datasets, pinned)
- Frozen decontam artifact: data/release/meta/benchmark_artifact.json.gz (6 benches, 20,148 recs)
- Eval retention benchmarks cached: mmlu, humaneval, livecodebench(+modules), ifeval, mtbench
  (bfcl text in frozen artifact; its HF repo is not load_dataset-able)

## Known deltas (honest)
- kernel_qa SFT slice ~0 (requires a live teacher; deferred).
- >1B midtrain: 157.6M unique is the ungated high-quality ceiling; reach >1B via
  multiple CPT epochs (num_train_epochs) or a gated the-stack CUDA subset (needs HF token).
