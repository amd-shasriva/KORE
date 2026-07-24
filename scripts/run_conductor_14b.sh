#!/usr/bin/env bash
# KORE full-scale 14B campaign - CONDUCTOR launcher (portable paths + project venv).
#
# This is the path-agnostic sibling of scripts/run_full_14b.sh (which hardcodes the
# tas32 dev paths). It resolves the repo root from its own location and uses the
# project virtualenv, so it runs on conductor / any node / NFS unchanged.
#
# RESUMABLE: state lives in <data-root>/campaign_manifest.json. If the node
# reservation ends mid-run, just re-run this script after you re-reserve - every
# already-completed stage whose on-disk artifact is present is skipped, and the
# run continues from where it stopped (files persist under your account).
#
# STAGE PLAN (resume-safe; skips whatever already has an on-disk artifact):
#   midtrain -> SKIPS when runs/full/midtrain checkpoint is present (it is).
#   datagen  -> RESUMES additively (shard_done skips finished task-shards and
#               generates only the missing repair/groups/wins) - needs the teacher
#   agentic  -> SYNTH (--agentic synth): CPU-only reconstruction of native tool-use
#               trajectories from the verified repair/wins/groups (minutes, no GPU/
#               teacher) INSTEAD of the tens-of-GPU-hours live harness. See
#               kore/data/synth_agentic.py.
#   build    -> RUNS: also mints gold-wins (from ranked groups) + repair->DPO pairs
#               (both ON by default) before assembling the SFT/DPO mix.
#   sft,dpo,grpo,soup,eval -> RUN
set -euo pipefail

# --- resolve repo root from this script's location (portable) ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/ops_runtime.sh
source "$SCRIPT_DIR/lib/ops_runtime.sh"
kore_deprecated_guard \
  "scripts/run_conductor_14b.sh" \
  "use scripts/spur_supervise_datagen.py for production datagen and submit training through the site scheduler" \
  "bash scripts/run_conductor_14b.sh [--dry-run]" \
  "$@"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- python: prefer the project venv, fall back to python3 ------------------
PY="$(kore_resolve_python "$REPO_ROOT")"
RUNTIME="$(kore_private_runtime)"
kore_require_commands od stat
echo "[run_conductor] repo=$REPO_ROOT python=$PY"

# Put the venv's bin on PATH so console scripts resolve to the venv - critically
# `accelerate`, which scripts/launch_distributed.sh invokes by bare name to drive
# FSDP for midtrain/sft/dpo. Without this the distributed stages die with exit 127.
export PATH="$(dirname "$PY"):$PATH"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Load gitignored secrets (AMD_LLM_API_KEY / AMD_NTID for the Claude teacher used
# by the datagen + agentic stages). Exported so every stage subprocess inherits.
if [ -f "$REPO_ROOT/.env.local" ]; then
  kore_secure_source_env "$REPO_ROOT/.env.local"
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
kore_export_rigor_env

# --- ON-NODE-CALIBRATED gfx950 (MI350X) roofline peaks ----------------------
# Measured on THIS node via `python -m kore.analysis.calibrate_peaks` (matches the
# P0 study within ~1%): HBM 4.64 TB/s (58% of the 8.0 datasheet) and bf16 1.29
# PF/s (52% of 2.5). The GRPO physics-residual reward (reward_mode=residual)
# needs the ATTAINABLE peak, not the datasheet - datasheet peaks make T_min ~2x
# too small and η ~2x too optimistic. fp8 is left at datasheet (not measurable on
# this stack). Override by re-running calibrate_peaks on a fully idle node.
export KORE_PEAK_HBM_BW="${KORE_PEAK_HBM_BW:-4.638812e+12}"
export KORE_PEAK_BF16="${KORE_PEAK_BF16:-1.290287e+15}"

# --- stage selection --------------------------------------------------------
# agentic needs the Claude teacher; include it only if a key is present so the
# campaign "just runs" without one. Override the whole list via KORE_STAGES.
if [ -n "${AMD_LLM_API_KEY:-}" ]; then
  # teacher available -> full pipeline. Order matches run_campaign.DEFAULT_STAGES:
  # midtrain FIRST (it's independent + fail-fasts the 28GB full-FT FSDP setup),
  # then the teacher-bound data stages, then build -> sft(base=midtrain ckpt) ->
  # dpo -> grpo -> soup -> eval. datagen still resumes additively; sft only needs
  # midtrain + build done before it, both of which precede it here.
  DEFAULT_STAGES="midtrain,datagen,agentic,build,sft,dpo,grpo,soup,eval"
  echo "[run_conductor] teacher key present -> full pipeline (midtrain-first, code default)."
else
  # no teacher -> cannot finish datagen/agentic; train on existing on-disk data only.
  DEFAULT_STAGES="midtrain,build,sft,dpo,grpo,soup,eval"
  echo "[run_conductor] no AMD_LLM_API_KEY -> skipping datagen/agentic (teacher stages)."
  echo "[run_conductor] put AMD_LLM_API_KEY in .env.local to finish datagen + agentic."
fi
STAGES="${KORE_STAGES:-$DEFAULT_STAGES}"
echo "[run_conductor] stages=$STAGES"

# Datagen/agentic are TEACHER-API-bound (each worker mostly waits on a ~6-13s Claude
# call; the GPUs sit idle at ~10% in between), so throughput scales with concurrency,
# not GPU count. Verified on this node: 3TB RAM / 384 CPUs and the gateway serves 64
# concurrent calls with zero throttling. Default 64 workers (8 per GPU) ~doubles
# throughput vs 32 with negligible quality impact (GPU eval bursts stay brief; groups
# ranking has a noise-margin gate). Override with KORE_DATAGEN_WORKERS (e.g. 8 for
# pristine bench timing, 32 for a balance).
DATAGEN_WORKERS="${KORE_DATAGEN_WORKERS:-64}"
echo "[run_conductor] datagen/agentic workers=$DATAGEN_WORKERS (teacher-bound; oversubscribing 8 GPUs)"

# No --tasks -> ALL registered tasks (train = non-held-out; eval = held-out MLA + paged-KV decode).
# --adaptive-steps -> GRPO plateau early-stop. Full campaign defaults otherwise.
# --agentic synth : fill the agentic SFT slice from ALREADY-verified records on CPU
#   (skips the tens-of-GPU-hours live harness). --gold-wins / --repair-dpo default ON
#   in run_campaign, so the build stage mints them automatically; override with
#   KORE_AGENTIC (live|synth|both) if you ever want the live rollouts.
AGENTIC_MODE="${KORE_AGENTIC:-synth}"
echo "[run_conductor] agentic mode=$AGENTIC_MODE (synth = CPU reconstruction, no 15-30h live rollouts)"
RUN_ID="${KORE_RUN_ID:-$(kore_new_run_id conductor-14b)}"
mkdir -p runs/full/logs
LOG="runs/full/logs/conductor_${RUN_ID}.log"
COMMAND=("$PY" scripts/run_campaign.py \
  --model Qwen/Qwen3-14B \
  --full-ft --use-hf --teacher claude \
  --adaptive-steps \
  --datagen-workers "$DATAGEN_WORKERS" \
  --stages "$STAGES" \
  --agentic "$AGENTIC_MODE" \
  --data-root data/full14b \
  --midtrain-out runs/full/midtrain \
  --sft-out runs/full/sft \
  --dpo-out runs/full/dpo \
  --grpo-out runs/full/grpo \
  --soup-out runs/full/soup \
  "$@")
set +e
kore_owned_run "$PY" "$REPO_ROOT" "$RUNTIME" "$RUN_ID" "conductor-14b" "$LOG" \
  "${COMMAND[@]}"
rc=$?
set -e
if (( rc != 0 )); then
  echo "[run_conductor] campaign failed rc=$rc run_id=$RUN_ID" >&2
  exit "$rc"
fi
kore_verify "$PY" "$REPO_ROOT" campaign \
  --repo "$REPO_ROOT" \
  --data-root data/full14b \
  --required-stages "$STAGES"
echo "[run_conductor] strict completion verified run_id=$RUN_ID"
