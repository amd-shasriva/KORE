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
set -uo pipefail

CONFIG="${1:?usage: run_grpo_resilient.sh <config.json> [max_tries] [min_gpus]}"
MAX_TRIES="${2:-15}"
MIN_GPUS="${3:-4}"
FREE_GIB="${FREE_GIB:-60}"          # a GPU counts as "free" if used < this many GiB
WANT_GPUS="${WANT_GPUS:-6}"          # use at most this many GPUs
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/shasriva/kore-venv/bin:$PATH"

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
  GPU_IDS="$IDS" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 NCCL_DEBUG=WARN \
    bash "$REPO_ROOT/scripts/launch_distributed.sh" grpo "$CONFIG"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[resilient] GRPO exited 0 (success) on try $try."
    exit 0
  fi
  echo "[resilient] GRPO failed (rc=$rc) on try $try - likely a transient VRAM spike on a pinned GPU. Retrying in 30s."
  pkill -9 -f "python -u -m kore.policy.grpo" 2>/dev/null || true
  sleep 30
done
echo "[resilient] exhausted $MAX_TRIES tries without a clean load window."
exit 1
