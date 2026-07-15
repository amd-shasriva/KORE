#!/usr/bin/env bash
# KORE FULL-SCALE end-to-end 14B run - the complete 70B recipe on 14B, NO scale
# knobs: ALL registered tasks (MLA + paged-KV decode families held out for eval), FULL datagen
# counts, FULL GRPO horizon, full-parameter FSDP across 8x MI325X, every
# best-in-world lever engaged. Durable/resumable from data/full14b/campaign_manifest.json.
set -euo pipefail
cd /root/Kore-rl/kore
export PYTHONPATH=/root/Kore-rl/kore:${PYTHONPATH:-}

# CRITICAL: the FSDP stages need ALL GPUs visible so accelerate assigns one per rank.
# A stale HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES in the calling shell pins every rank
# to a single GPU -> NCCL "Duplicate GPU detected". Clear them here (the parallel
# datagen workers re-pin HIP_VISIBLE_DEVICES per-worker internally).
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES

# --- best-in-world integrity levers (propagate to every stage subprocess) ---
export KORE_VERIFIED_CORRECTNESS=1     # enumerated adversarial no-lucky-pass gate
export KORE_COMPILE_BASELINE=1         # honest compiler-fused baseline (anti speedup-inflation)
export KORE_GENERAL_REPLAY_HF=1        # real SOTA replay (AMD kernels / reasoning / code)
export KORE_BENCH_COLD=1               # cold-cache (L2-flushed) timing
export TORCHINDUCTOR_CACHE_DIR=/root/Kore-rl/.inductor_cache

# No --tasks  -> ALL registered tasks (train = non-held-out; eval = held-out MLA + paged-KV decode).
# No datagen caps -> full defaults (n_repair=50, n_parents=20, k=6, wins_gens=8,
#   n_agentic=16). No --grpo-steps -> full config horizon (+ --adaptive-steps plateau
#   early-stop). dpo-rounds=2, sft_total=20000, eval-n=300, retention gate @ 0.02:
#   all campaign defaults (the real, no-shortcut pipeline). Curriculum / anti-collapse
#   / value-prefilter / RFT / retention gates are ON by default.
python scripts/run_campaign.py \
  --model Qwen/Qwen3-14B \
  --full-ft --use-hf --teacher claude \
  --adaptive-steps \
  --datagen-workers 8 \
  --data-root data/full14b \
  --midtrain-out runs/full/midtrain \
  --sft-out runs/full/sft \
  --dpo-out runs/full/dpo \
  --grpo-out runs/full/grpo \
  --soup-out runs/full/soup
echo "[run_full_14b] campaign process exited with code $?"
