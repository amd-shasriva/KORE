#!/usr/bin/env bash
# Development-only compatibility supervisor for the retired b05 factory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/ops_runtime.sh
source "$SCRIPT_DIR/lib/ops_runtime.sh"
kore_deprecated_guard \
  "scripts/factory_supervise.sh" \
  "use scripts/spur_supervise_datagen.py; it partitions immutable run shards and verifies every wave" \
  "bash scripts/factory_supervise.sh [--dry-run]" \
  "$@"

REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$(kore_resolve_python "$REPO")"
RUNTIME="$(kore_private_runtime)"
DATA_ROOT="${KORE_DATA_ROOT:-data/b05factory}"
PEER="${FACTORY_PEER:-cv350-tnndh2-b05-1.tnn.dcgpu}"
GPUS_FIXED="${FACTORY_GPUS:-}"
MAX_GPUS="${FACTORY_MAX_GPUS:-6}"
UTIL_MAX="${FACTORY_UTIL_MAX:-30}"
VRAM_MAX_GB="${FACTORY_VRAM_MAX_GB:-12}"
WORKERS="${FACTORY_WORKERS:-48}"
MAX_ATTEMPTS="${FACTORY_MAX_ATTEMPTS:-24}"
MAX_STALLS="${FACTORY_MAX_STALLS:-2}"
TARGET="${FACTORY_WINS_TARGET:-3}"
RUN_ID="${KORE_RUN_ID:-$(kore_new_run_id legacy-factory)}"
STATE_DIR="$RUNTIME/runs/$RUN_ID"
LOCK_DIR="$RUNTIME/legacy-factory.lock"

cd "$REPO"
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
kore_require_commands rocm-smi rsync ssh od stat
kore_secure_source_env "$REPO/.env.local"
kore_export_rigor_env
for value in "$MAX_GPUS" "$WORKERS" "$MAX_ATTEMPTS" "$MAX_STALLS" "$TARGET"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: factory numeric settings must be positive integers" >&2
    exit 2
  }
done
if ! mkdir -m 0700 -- "$LOCK_DIR" 2>/dev/null; then
  echo "ERROR: another legacy factory supervisor owns $LOCK_DIR" >&2
  exit 73
fi
trap 'rmdir -- "$LOCK_DIR" 2>/dev/null || true' EXIT
mkdir -m 0700 -p -- "$STATE_DIR" runs/factory_logs
TASKS_FILE="$STATE_DIR/tasks.txt"
CLEANUP_FILE="$STATE_DIR/remaining.txt"
SNAPSHOT="$STATE_DIR/task-set.json"
LOG="runs/factory_logs/${RUN_ID}.log"

PYTHONPATH="$REPO" "$PY" - "$RUNTIME" "$RUN_ID" "$TASKS_FILE" <<'PY'
import sys
from pathlib import Path
from kore.ops.runtime import SecureRuntime
from kore.tasks.registry import train_tasks

runtime_path, run_id, tasks_path = sys.argv[1:]
task_ids = sorted(
    task.task_id for task in train_tasks() if task.task_id.startswith("genb_")
)
if not task_ids:
    raise SystemExit("registry returned no genb_ train tasks")
identity = SecureRuntime(runtime_path).store_task_set(
    Path("runs") / run_id / "task-set.json", task_ids
)
Path(tasks_path).write_text(",".join(identity.task_ids))
Path(tasks_path).chmod(0o600)
print(f"TASK_SET count={identity.count} sha256={identity.sha256}")
PY

pick_gpus() {
  if [[ -n "$GPUS_FIXED" ]]; then
    printf '%s\n' "$GPUS_FIXED"
    return
  fi
  SFT_UTIL_MAX="$UTIL_MAX" SFT_VRAM_MAX_GB="$VRAM_MAX_GB" GATE_NGPU="$MAX_GPUS" \
    "$PY" scripts/gpu_pick_hip.py 2>/dev/null | cut -f1
}

verify_remaining() {
  "$PY" scripts/_kf_verify.py "$DATA_ROOT" "$TARGET" \
    --tasks "$(cat "$TASKS_FILE")" \
    --cleanup-out "$CLEANUP_FILE" \
    --json
}

echo "FACTORY_SUPERVISOR run_id=$RUN_ID attempts=$MAX_ATTEMPTS $(date)"
previous_remaining=-1
stalls=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  summary="$(verify_remaining)" || {
    echo "ERROR: strict factory verifier failed" >&2
    exit 4
  }
  remaining="$("$PY" -c 'import json,sys; print(json.loads(sys.stdin.read())["remaining_undone"])' <<<"$summary")"
  if (( remaining == 0 )); then
    "$PY" scripts/_kf_verify.py "$DATA_ROOT" "$TARGET" \
      --tasks "$(cat "$TASKS_FILE")" --cleanup-out "$CLEANUP_FILE" \
      --require-complete
    if ! rsync -a --timeout=120 "$DATA_ROOT"/ \
        "$PEER:$REPO/data/b05factory_synced/"; then
      echo "ERROR: final site-specific sync failed" >&2
      exit 7
    fi
    echo "FACTORY COMPLETE: strict verifier passed run_id=$RUN_ID"
    exit 0
  fi
  if (( previous_remaining >= 0 && remaining >= previous_remaining )); then
    stalls=$((stalls + 1))
  else
    stalls=0
  fi
  if (( stalls >= MAX_STALLS )); then
    echo "ERROR: no durable progress for $stalls attempts" >&2
    exit 5
  fi
  previous_remaining="$remaining"
  SEL="$(pick_gpus)"
  if [[ -z "$SEL" ]]; then
    echo "WARN: no verified-idle GPUs on attempt $attempt" >&2
    continue
  fi
  TASKS="$(cat "$CLEANUP_FILE")"
  [[ -n "$TASKS" ]] || {
    echo "ERROR: verifier reported remaining work but emitted no task IDs" >&2
    exit 4
  }
  echo "ALERT legacy factory attempt=$attempt remaining=$remaining gpus=[$SEL]"
  set +e
  kore_owned_run "$PY" "$REPO" "$RUNTIME" "$RUN_ID" "legacy-factory-child" "$LOG" \
    env PYTHONPATH="$REPO" KORE_VERIFIED_CORRECTNESS=1 \
    KORE_COMPILE_BASELINE=1 KORE_BENCH_COLD=1 KORE_SHAPE_AUGMENT=1 \
    "$PY" scripts/run_campaign.py --model Qwen/Qwen3-14B --stages datagen \
      --data-root "$DATA_ROOT" --teacher claude --datagen-workers "$WORKERS" \
      --gpu-ids "$SEL" --tasks "$TASKS"
  rc=$?
  set -e
  if (( rc != 0 )); then
    echo "WARN: owned datagen attempt failed rc=$rc" >&2
  fi
  rsync -a --timeout=120 "$DATA_ROOT"/ \
    "$PEER:$REPO/data/b05factory_synced/" || echo "WARN: intermediate sync failed" >&2
done
echo "ERROR: maximum factory attempts reached before strict completion" >&2
exit 6
