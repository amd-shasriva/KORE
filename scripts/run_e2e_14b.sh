#!/usr/bin/env bash
# KORE end-to-end 14B validation run (the full 70B recipe, on 14B, full-parameter
# FSDP across 8x MI325X). Every stage + every best-in-world lever engaged; sized
# with a representative task set + bounded per-stage work so it COMPLETES end-to-end
# (the point is to verify the whole pipeline works top-to-bottom). Durable/resumable:
# re-running resumes from data/e2e14b/campaign_manifest.json.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/ops_runtime.sh
source "$SCRIPT_DIR/lib/ops_runtime.sh"
kore_deprecated_guard \
  "scripts/run_e2e_14b.sh" \
  "use scripts/spur_supervise_datagen.py for production data and scheduler-native training validation jobs" \
  "bash scripts/run_e2e_14b.sh [--dry-run]" \
  "$@"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
PY="$(kore_resolve_python "$REPO_ROOT")"
RUNTIME="$(kore_private_runtime)"
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
kore_secure_source_env "$REPO_ROOT/.env.local"
kore_require_commands od stat

# --- best-in-world integrity levers (propagate to every stage subprocess) ---
export KORE_VERIFIED_CORRECTNESS=1     # enumerated adversarial no-lucky-pass gate
export KORE_COMPILE_BASELINE=1         # honest compiler-fused baseline (anti speedup-inflation)
export KORE_GENERAL_REPLAY_HF=1        # real SOTA replay (AMD kernels / reasoning / code)
export KORE_BENCH_COLD=1               # cold-cache (L2-flushed) timing
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.inductor_cache"
kore_export_rigor_env

RUN_ID="${KORE_RUN_ID:-$(kore_new_run_id e2e-14b)}"
mkdir -p runs/e2e/logs
LOG="runs/e2e/logs/e2e_${RUN_ID}.log"
COMMAND=("$PY" scripts/run_campaign.py \
  --model Qwen/Qwen3-14B \
  --tasks rmsnorm_aiter,gemm_bf16,genv_softmax_bf16 \
  --full-ft --use-hf --teacher claude \
  --n-repair 4 --n-parents 3 --k 4 --wins-gens 2 \
  --n-agentic 4 --max-tool-turns 4 \
  --sft-total 1500 --dpo-rounds 2 --dagger-n 4 \
  --grpo-steps 4 --eval-budget 3 --eval-n 20 \
  --retention-epsilon 0.15 \
  --data-root data/e2e14b \
  --midtrain-out runs/e2e/midtrain \
  --sft-out runs/e2e/sft \
  --dpo-out runs/e2e/dpo \
  --grpo-out runs/e2e/grpo \
  --soup-out runs/e2e/soup)
set +e
kore_owned_run "$PY" "$REPO_ROOT" "$RUNTIME" "$RUN_ID" "e2e-14b" "$LOG" \
  "${COMMAND[@]}"
rc=$?
set -e
if (( rc != 0 )); then
  echo "[run_e2e_14b] campaign failed rc=$rc (run_id=$RUN_ID)" >&2
  exit "$rc"
fi
kore_verify "$PY" "$REPO_ROOT" campaign \
  --repo "$REPO_ROOT" \
  --data-root data/e2e14b \
  --required-stages midtrain,datagen,agentic,build,sft,dpo,grpo,soup,eval
echo "[run_e2e_14b] strict completion verified (run_id=$RUN_ID)"
