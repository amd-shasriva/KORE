#!/usr/bin/env bash
# Resilient GRPO launcher for a HEAVILY SHARED node.
#
# The node's GPUs are intermittently saturated by other users' root-owned kernel
# eval jobs (VRAM spikes to ~full on random GPUs). A single fixed-GPU launch OOMs
# whenever a spike lands on one of our GPUs during model/ref/replica load. This
# wrapper dynamically selects the GPUs that are FREE right now, pins the run to
# them, and RETRIES across transient spikes until a load window succeeds (after
# which the run holds its memory and is stable).
#
# Usage: scripts/run_grpo_resilient.sh <grpo_launch_config.json> [max_tries] [min_gpus]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/ops_runtime.sh
source "$SCRIPT_DIR/lib/ops_runtime.sh"
kore_deprecated_guard \
  "scripts/run_grpo_resilient.sh" \
  "submit GRPO through the site scheduler with an explicit GPU allocation" \
  "scripts/run_grpo_resilient.sh <config.json> [max_tries] [min_gpus] [--dry-run]" \
  "$@"

CONFIG="${1:?usage: run_grpo_resilient.sh <config.json> [max_tries] [min_gpus]}"
MAX_TRIES="${2:-15}"
MIN_GPUS="${3:-4}"
FREE_GIB="${FREE_GIB:-60}"          # a GPU counts as "free" if used < this many GiB
WANT_GPUS="${WANT_GPUS:-6}"          # use at most this many GPUs
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
PY="$(kore_resolve_python "$REPO_ROOT")"
RUNTIME="$(kore_private_runtime)"
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
kore_require_commands rocm-smi awk paste seq od stat
[[ -f "$CONFIG" ]] || { echo "ERROR: missing GRPO config: $CONFIG" >&2; exit 2; }
[[ "$MAX_TRIES" =~ ^[1-9][0-9]*$ && "$MIN_GPUS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: max_tries and min_gpus must be positive integers" >&2
  exit 2
}
kore_export_rigor_env
RUN_ID="${KORE_RUN_ID:-$(kore_new_run_id grpo-resilient)}"
mkdir -p runs/grpo_logs
LOG="runs/grpo_logs/${RUN_ID}.log"

free_gpus() {
  # Print comma-separated physical GPU ids whose used VRAM < FREE_GIB, capped to
  # WANT_GPUS. Parse the id from the "GPU[N]" prefix (rocm-smi prints Total and
  # "Total Used Memory" lines per GPU; we key off the Used line).
  rocm-smi --showmeminfo vram 2>/dev/null \
    | awk -v thr="$FREE_GIB" '
        /Total Used Memory/ { id=$1; gsub(/[^0-9]/,"",id); used=$NF/1073741824;
                              if (used+0 < thr+0) print id }
      ' \
    | head -n "$WANT_GPUS" | paste -sd, -
}

for try in $(seq 1 "$MAX_TRIES"); do
  IDS="$(free_gpus)"
  N=$(printf '%s' "$IDS" | tr ',' '\n' | grep -c .)
  echo "[resilient] try $try/$MAX_TRIES @ $(date +%T): free GPUs=[$IDS] (n=$N, need>=$MIN_GPUS)"
  if [ "$N" -lt "$MIN_GPUS" ]; then
    echo "[resilient] not enough free GPUs; waiting 45s for a window..."
    sleep 45; continue
  fi
  echo "[resilient] launching GRPO pinned to GPUs $IDS ..."
  set +e
  kore_owned_run "$PY" "$REPO_ROOT" "$RUNTIME" "$RUN_ID" "grpo-resilient" "$LOG" \
    env GPU_IDS="$IDS" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 NCCL_DEBUG=WARN \
    bash "$REPO_ROOT/scripts/launch_distributed.sh" grpo "$CONFIG"
  rc=$?
  set -e
  if [ "$rc" -eq 0 ] && kore_verify "$PY" "$REPO_ROOT" grpo-config \
      --repo "$REPO_ROOT" --config "$CONFIG"; then
    echo "[resilient] strict GRPO completion verified on try $try."
    exit 0
  fi
  echo "[resilient] GRPO failed or artifacts were incomplete (rc=$rc) on try $try; retrying in 30s."
  sleep 30
done
echo "[resilient] exhausted $MAX_TRIES tries without a clean load window."
exit 6
