#!/bin/bash
# Two-node split datagen worker: run_campaign datagen over a FIXED task list
# (one disjoint half of the remaining uncovered genb_ tasks), re-running the
# shard-resumable pass until every task in the list has repair+groups+wins, then
# exit. Lets b05-1 and b05-2 grind disjoint halves in parallel with ZERO
# duplicated teacher calls (the throughput bottleneck).
#
# Usage: datagen_half.sh <half_tasks_file> <data_root> <gpu_ids> [workers]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/ops_runtime.sh
source "$SCRIPT_DIR/lib/ops_runtime.sh"
kore_deprecated_guard \
  "scripts/datagen_half.sh" \
  "use scripts/spur_submit_datagen.sh, which creates immutable disjoint array manifests" \
  "scripts/datagen_half.sh <tasks_file> <data_root> <gpu_ids> [workers] [--dry-run]" \
  "$@"
HALF="${1:?half tasks file}"
DATA_ROOT="${2:?data root}"
GPUS="${3:?gpu ids}"
WORKERS="${4:-48}"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$(kore_resolve_python "$REPO")"
RUNTIME="$(kore_private_runtime)"
RUN_ID="${KORE_RUN_ID:-$(kore_new_run_id datagen-half)}"
MAX_PASSES="${KORE_DATAGEN_MAX_PASSES:-40}"
STATE_DIR="$RUNTIME/runs/$RUN_ID"
mkdir -m 0700 -p -- "$STATE_DIR"
SNAPSHOT_TASKS="$STATE_DIR/tasks.txt"
REMAINING_TASKS="$STATE_DIR/remaining.txt"
LOG="$REPO/runs/datagen_half_${RUN_ID}.log"
cd "$REPO"
kore_secure_source_env "$REPO/.env.local"
export PATH="$(dirname "$PY"):$PATH"
[[ ! -L "$HALF" && -f "$HALF" && "$(stat -c '%u' -- "$HALF")" == "$(id -u)" ]] || {
  echo "ERROR: task list must be an owned regular non-symlink file: $HALF" >&2
  exit 74
}
cp -- "$HALF" "$SNAPSHOT_TASKS"
chmod 0600 -- "$SNAPSHOT_TASKS"
PYTHONPATH="$REPO" "$PY" -m kore.ops verify task-set \
  --tasks-file "$SNAPSHOT_TASKS"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ && "$MAX_PASSES" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: workers and pass limit must be positive integers" >&2
  exit 2
}
# FSDP-style: never let a stale mask pin every worker to one GPU (workers re-pin
# HIP per-worker internally from --gpu-ids).
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export PYTHONPATH="$REPO"
kore_export_rigor_env

remaining() {
  local summary
  summary="$("$PY" scripts/_kf_verify.py "$DATA_ROOT" 1 \
    --tasks "$(cat "$SNAPSHOT_TASKS")" \
    --cleanup-out "$REMAINING_TASKS" --json)" || return
  "$PY" -c 'import json,sys; print(json.loads(sys.stdin.read())["remaining_undone"])' \
    <<<"$summary"
}

echo "[datagen_half] START run_id=$RUN_ID data_root=$DATA_ROOT gpus=$GPUS workers=$WORKERS $(date)"
for i in $(seq 1 "$MAX_PASSES"); do
  R="$(remaining)"
  echo "[datagen_half] pass=$i remaining=$R $(date)"
  if [ "$R" = "0" ]; then
    "$PY" scripts/_kf_verify.py "$DATA_ROOT" 1 \
      --tasks "$(cat "$SNAPSHOT_TASKS")" \
      --cleanup-out "$REMAINING_TASKS" --require-complete
    echo "[datagen_half] COMPLETE (strict verifier passed) $(date)"
    exit 0
  fi
  TASKS="$(cat "$REMAINING_TASKS")"
  [[ -n "$TASKS" ]] || {
    echo "ERROR: incomplete status without remaining task IDs" >&2
    exit 4
  }
  set +e
  kore_owned_run "$PY" "$REPO" "$RUNTIME" "$RUN_ID" "datagen-half-child" "$LOG" \
    env PYTHONPATH="$REPO" "$PY" scripts/run_campaign.py \
      --model Qwen/Qwen3-14B --stages datagen \
      --data-root "$DATA_ROOT" --teacher claude --datagen-workers "$WORKERS" \
      --gpu-ids "$GPUS" --tasks "$TASKS"
  rc=$?
  set -e
  (( rc == 0 )) || echo "[datagen_half] WARN pass=$i failed rc=$rc" >&2
  sleep 15
done
echo "[datagen_half] GIVEUP after $MAX_PASSES passes; work remains" >&2
exit 6
