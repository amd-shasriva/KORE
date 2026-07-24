#!/bin/bash
# _kf_worker.sh KIND TASKS_FILE WORKERS SESSION
#   KIND        = deepen | base
#   TASKS_FILE  = path to a comma-list of task ids (read at launch time)
#   WORKERS     = worker processes (GPU-pinned round-robin over all 8 GPUs)
#   SESSION     = tmux session name (also the log basename under runs/)
#
# Launches the requested resume-safe driver in a detached tmux session so it
# survives ssh disconnects. deepen -> deepen_wins.py (wins/ only); base ->
# complete_base.py (repair/ + groups/ only). The two never touch each other's
# shard kinds, so a node can run both at once.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/ops_runtime.sh
source "$SCRIPT_DIR/lib/ops_runtime.sh"
kore_deprecated_guard \
  "scripts/_kf_worker.sh" \
  "use scripts/spur_submit_datagen.sh array workers; they own immutable task manifests and scheduler process groups" \
  "scripts/_kf_worker.sh <deepen|base> <tasks_file> <workers> <session> [--dry-run]" \
  "$@"
KIND="${1:?kind}"
TASKS_FILE="${2:?tasks file}"
WORKERS="${3:?workers}"
SESSION="${4:?session}"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$(kore_resolve_python "$REPO")"
RUNTIME="$(kore_private_runtime)"
DATA="${KORE_DATA_ROOT:-data/b05factory}"
GPUS="${KORE_GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${KORE_RUN_ID:-$(kore_new_run_id "$SESSION")}"
STATE_DIR="$RUNTIME/runs/$RUN_ID"
SNAPSHOT_TASKS="$STATE_DIR/tasks.txt"
ENV_FILE="$STATE_DIR/environment.sh"
LOG="$REPO/runs/$SESSION.log"
[[ "$KIND" == "deepen" || "$KIND" == "base" ]] || {
  echo "ERROR: KIND must be deepen or base" >&2
  exit 2
}
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: workers must be a positive integer" >&2
  exit 2
}
[[ "$SESSION" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || {
  echo "ERROR: unsafe tmux session name: $SESSION" >&2
  exit 2
}
[[ ! -L "$TASKS_FILE" && -f "$TASKS_FILE" && "$(stat -c '%u' -- "$TASKS_FILE")" == "$(id -u)" ]] || {
  echo "ERROR: task list must be an owned regular non-symlink file: $TASKS_FILE" >&2
  exit 74
}
cd "$REPO"
mkdir -p runs
mkdir -m 0700 -p -- "$STATE_DIR"
cp -- "$TASKS_FILE" "$SNAPSHOT_TASKS"
chmod 0600 -- "$SNAPSHOT_TASKS"
kore_secure_source_env "$REPO/.env.local"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO"
kore_export_rigor_env
export KORE_WINS_PMC="${KORE_WINS_PMC:-0}"
export -p > "$ENV_FILE"
chmod 0600 -- "$ENV_FILE"
kore_require_commands tmux od stat

if [ ! -s "$SNAPSHOT_TASKS" ]; then
  echo "[_kf_worker] $SESSION: empty task list - nothing assigned"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  pane_dead="$(tmux display-message -p -t "$SESSION" '#{pane_dead}' 2>/dev/null || echo unknown)"
  echo "[_kf_worker] $SESSION: existing session is unowned/stale (pane_dead=$pane_dead); refusing to replace it" >&2
  exit 73
fi

if [ "$KIND" = "deepen" ]; then
  DRIVER=(scripts/deepen_wins.py --data-root "$DATA" --tasks "$(cat "$SNAPSHOT_TASKS")" \
    --gpu-ids "$GPUS" --workers "$WORKERS" --target 3 --gens 8)
  VERIFY=(task-shards --data-root "$DATA" --tasks-file "$SNAPSHOT_TASKS" \
    --target-wins 3 --kinds wins)
else
  DRIVER=(scripts/complete_base.py --data-root "$DATA" --tasks "$(cat "$SNAPSHOT_TASKS")" \
    --gpu-ids "$GPUS" --workers "$WORKERS" --n-repair 50 --n-parents 20 --k 6)
  VERIFY=(task-shards --data-root "$DATA" --tasks-file "$SNAPSHOT_TASKS" \
    --target-wins 0 --kinds repair,groups)
fi

tmux new-session -d -s "$SESSION" \
  bash -c '
    set -euo pipefail
    source "$1"
    shift
    python="$1"; repo="$2"; runtime="$3"; run_id="$4"; name="$5"; log="$6"
    shift 6
    driver_count="$1"
    shift
    driver=("${@:1:driver_count}")
    shift "$driver_count"
    "$python" -m kore.ops run \
      --runtime-dir "$runtime" --run-id "$run_id" --name "$name" \
      --cwd "$repo" --log "$log" -- \
      "$python" "${driver[@]}"
    "$python" -m kore.ops verify "$@"
  ' -- "$ENV_FILE" "$PY" "$REPO" "$RUNTIME" "$RUN_ID" "$SESSION" "$LOG" \
  "${#DRIVER[@]}" "${DRIVER[@]}" "${VERIFY[@]}"
sleep 2
if tmux has-session -t "$SESSION" 2>/dev/null; then
  task_count="$(awk -F, '{print NF}' "$SNAPSHOT_TASKS")"
  echo "[_kf_worker] $SESSION: launched run_id=$RUN_ID ($KIND, workers=$WORKERS, tasks=$task_count)"
else
  state="runs/$RUN_ID/$SESSION.json"
  if PYTHONPATH="$REPO" "$PY" -m kore.ops status \
      --runtime-dir "$RUNTIME" --state "$state" >/dev/null 2>&1; then
    echo "[_kf_worker] $SESSION: process is running without tmux visibility" >&2
    exit 74
  fi
  rc="$("$PY" - "$RUNTIME/$state" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("returncode", 1))
except Exception:
    print(1)
PY
)"
  if [[ "$rc" == "0" ]]; then
    echo "[_kf_worker] $SESSION: completed before launch check"
    exit 0
  fi
  echo "[_kf_worker] $SESSION: FAILED to launch or exited rc=$rc" >&2
  exit 1
fi
