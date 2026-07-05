#!/usr/bin/env bash
# KORE FULL-SCALE end-to-end 14B run — the complete 70B recipe on 14B, NO scale
# knobs: ALL registered tasks (attention family held out for eval), FULL datagen
# counts, FULL GRPO horizon, full-parameter FSDP across 8x MI325X, every
# best-in-world lever engaged. Durable/resumable from data/full14b/campaign_manifest.json.
set -euo pipefail
cd /root/Kore-rl/kore
export PYTHONPATH=/root/Kore-rl/kore:${PYTHONPATH:-}

# --- best-in-world integrity levers (propagate to every stage subprocess) ---
export KORE_VERIFIED_CORRECTNESS=1     # enumerated adversarial no-lucky-pass gate
export KORE_COMPILE_BASELINE=1         # honest compiler-fused baseline (anti speedup-inflation)
export KORE_GENERAL_REPLAY_HF=1        # real SOTA replay (AMD kernels / reasoning / code)
export KORE_BENCH_COLD=1               # cold-cache (L2-flushed) timing
export TORCHINDUCTOR_CACHE_DIR=/root/Kore-rl/.inductor_cache

# No --tasks  -> ALL registered tasks (train = non-held-out; eval = attention family).
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
