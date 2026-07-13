#!/usr/bin/env bash
# KORE v2 rerun — REUSE v1 kernels, don't regenerate.
#
# Pipeline: reverify (re-measure the ~27.6k EXISTING kernels against the strong
# baseline + adversarial battery, NO teacher) -> datagen (ONLY the coverage holes,
# resumes) -> build (v2 SFT/DPO: contract + curation + dedup + provenance + in-context
# DPO) -> midtrain -> sft -> dpo -> grpo -> soup -> eval. All pinned to specific GPUs
# so it never contends with other users on a shared node.
#
# Usage:
#   GPU_IDS=5,6,7 bash scripts/run_v2_reuse_14b.sh      # pin to GPUs 5,6,7
#   bash scripts/run_v2_reuse_14b.sh                    # auto-detect FREE GPUs
#
# Resumable: reverify skips .reverified tasks; datagen skips existing shards; the
# campaign manifest resumes stages. Safe to re-run after an interruption.
set -euo pipefail

# Repo root = parent of this script's dir (robust to the mount path).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Use the KORE venv for python/accelerate/torchrun (bare `python` may be the system
# one without torch). Override with KORE_VENV=/path/to/venv if different.
KORE_VENV="${KORE_VENV:-/home/shasriva/kore-venv}"
if [ -x "$KORE_VENV/bin/python" ]; then
  export PATH="$KORE_VENV/bin:$PATH"
fi
echo "[run_v2_reuse_14b] python=$(command -v python)"

# Clear any stale device pin from the calling shell; per-stage pinning is done via
# --gpu-ids (datagen/reverify re-pin per worker; FSDP training pins via GPU_IDS).
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES

# Best-in-world integrity levers (propagate to every verifier subprocess).
export KORE_VERIFIED_CORRECTNESS=1     # adversarial no-lucky-pass correctness gate
export KORE_COMPILE_BASELINE=1         # honest compiler-fused/vendor baseline
export KORE_GENERAL_REPLAY_HF=1        # real SOTA replay (AMD kernels / reasoning / code)
export KORE_BENCH_COLD=1               # cold-cache (L2-flushed) timing
export KORE_DECONTAM=1                 # eval decontamination
export KORE_CURATE=1                   # curation + balancing
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.inductor_cache"

# Throughput: reverify is compile/CPU-bound with ~idle GPUs (each eval spends most of
# its time importing torch + JIT-compiling, ~2% GPU). This box has 384 cores + 3TB RAM,
# so run K workers PER physical GPU. Timing stays honest because the genops --bench-both
# path measures candidate+reference back-to-back in one process (contention-fair ratio).
export KORE_REVERIFY_WORKERS_PER_GPU="${KORE_REVERIFY_WORKERS_PER_GPU:-8}"

# Optional: attach rocprof counters for grounded reasoning (adds profiling cost).
GROUND_FLAG=""
[ "${GROUND_REASONING:-0}" = "1" ] && GROUND_FLAG="--ground-reasoning"

GPUS="${GPU_IDS:-}"   # empty -> campaign auto-detects FREE GPUs via rocm-smi
echo "[run_v2_reuse_14b] repo=$REPO_ROOT  gpu-ids='${GPUS:-auto-free}'  ground=${GROUND_REASONING:-0}"

python scripts/run_campaign.py \
  --model Qwen/Qwen3-14B \
  --full-ft --use-hf --teacher claude \
  --adaptive-steps \
  --stages reverify,datagen,build,midtrain,sft,dpo,grpo,soup,eval \
  --gpu-ids "$GPUS" \
  $GROUND_FLAG \
  --data-root data/full14b \
  --midtrain-out runs/full/midtrain \
  --sft-out runs/full/sft \
  --dpo-out runs/full/dpo \
  --grpo-out runs/full/grpo \
  --soup-out runs/full/soup
echo "[run_v2_reuse_14b] campaign process exited with code $?"
