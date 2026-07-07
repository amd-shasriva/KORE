#!/usr/bin/env bash
# KORE full-scale 14B campaign — CONDUCTOR launcher (portable paths + project venv).
#
# This is the path-agnostic sibling of scripts/run_full_14b.sh (which hardcodes the
# tas32 dev paths). It resolves the repo root from its own location and uses the
# project virtualenv, so it runs on conductor / any node / NFS unchanged.
#
# RESUMABLE: state lives in <data-root>/campaign_manifest.json. If the node
# reservation ends mid-run, just re-run this script after you re-reserve — every
# already-completed stage whose on-disk artifact is present is skipped, and the
# run continues from where it stopped (files persist under your account).
#
# STAGE PLAN (datagen is only ~74% done, so RESUME it; everything else re-runs):
#   datagen  -> RESUMES additively (shard_done skips the finished task-shards and
#               generates only the missing repair/groups/wins) — needs the teacher
#   agentic  -> RUNS (on-policy tool-use trajectories) — needs the teacher
#   build    -> RUNS (assembles SFT/DPO from datagen + agentic)
#   midtrain -> RE-RUNS (no checkpoint was transferred)
#   sft,dpo,grpo,soup,eval -> RUN
set -euo pipefail

# --- resolve repo root from this script's location (portable) ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- python: prefer the project venv, fall back to python3 ------------------
PY="${KORE_PY:-$HOME/kore-venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
echo "[run_conductor] repo=$REPO_ROOT python=$PY"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Load gitignored secrets (AMD_LLM_API_KEY / AMD_NTID for the Claude teacher used
# by the datagen + agentic stages). Exported so every stage subprocess inherits.
if [ -f "$REPO_ROOT/.env.local" ]; then
  set -a; . "$REPO_ROOT/.env.local"; set +a
  echo "[run_conductor] loaded .env.local (teacher creds)"
fi

# FSDP needs ALL GPUs visible so accelerate assigns one per rank. A stale
# HIP/CUDA_VISIBLE_DEVICES in the calling shell pins every rank to one GPU ->
# NCCL "Duplicate GPU detected". Clear them (datagen workers re-pin per-worker).
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES

# --- best-in-world integrity levers (propagate to every stage subprocess) ----
export KORE_VERIFIED_CORRECTNESS=1     # enumerated adversarial no-lucky-pass gate
export KORE_COMPILE_BASELINE=1         # honest compiler-fused baseline
export KORE_GENERAL_REPLAY_HF=1        # real SOTA replay (AMD kernels/reasoning/code)
export KORE_BENCH_COLD=1               # cold-cache (L2-flushed) timing
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.inductor_cache"

# --- stage selection --------------------------------------------------------
# agentic needs the Claude teacher; include it only if a key is present so the
# campaign "just runs" without one. Override the whole list via KORE_STAGES.
if [ -n "${AMD_LLM_API_KEY:-}" ]; then
  # teacher available -> RESUME datagen (additive: only missing shards) + agentic.
  DEFAULT_STAGES="datagen,agentic,build,midtrain,sft,dpo,grpo,soup,eval"
  echo "[run_conductor] teacher key present -> datagen(resume) + agentic included."
else
  # no teacher -> cannot finish datagen/agentic; train on existing on-disk data only.
  DEFAULT_STAGES="midtrain,build,sft,dpo,grpo,soup,eval"
  echo "[run_conductor] no AMD_LLM_API_KEY -> skipping datagen/agentic (teacher stages)."
  echo "[run_conductor] put AMD_LLM_API_KEY in .env.local to finish datagen + agentic."
fi
STAGES="${KORE_STAGES:-$DEFAULT_STAGES}"
echo "[run_conductor] stages=$STAGES"

# Datagen/agentic are TEACHER-API-bound (each worker mostly waits on a ~1-7s Claude
# call; the GPUs sit idle in between), so throughput scales with concurrency, not
# GPU count. Verified on this node: 3TB RAM / 384 CPUs and the gateway serves 64
# concurrent calls with zero throttling. Oversubscribe the 8 GPUs (workers %% n_gpus
# pins each worker to a GPU) to ~4x for a 4x-faster datagen; benchmark timing stays
# usable because most evals are cached + win detection is ratio-based. Override with
# KORE_DATAGEN_WORKERS (e.g. 8 for pristine timing, 48 for max speed).
DATAGEN_WORKERS="${KORE_DATAGEN_WORKERS:-32}"
echo "[run_conductor] datagen/agentic workers=$DATAGEN_WORKERS (teacher-bound; oversubscribing 8 GPUs)"

# No --tasks -> ALL registered tasks (train = non-held-out; eval = attention family).
# --adaptive-steps -> GRPO plateau early-stop. Full campaign defaults otherwise.
exec "$PY" scripts/run_campaign.py \
  --model Qwen/Qwen3-14B \
  --full-ft --use-hf --teacher claude \
  --adaptive-steps \
  --datagen-workers "$DATAGEN_WORKERS" \
  --stages "$STAGES" \
  --data-root data/full14b \
  --midtrain-out runs/full/midtrain \
  --sft-out runs/full/sft \
  --dpo-out runs/full/dpo \
  --grpo-out runs/full/grpo \
  --soup-out runs/full/soup \
  "$@"
