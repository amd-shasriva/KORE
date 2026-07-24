#!/usr/bin/env bash
# Development-only durable wrapper for the deprecated 14B conductor.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/ops_runtime.sh
source "$SCRIPT_DIR/lib/ops_runtime.sh"
kore_deprecated_guard \
  "scripts/tmux_campaign.sh" \
  "use scheduler-native jobs; production datagen is managed by scripts/spur_supervise_datagen.py" \
  "bash scripts/tmux_campaign.sh [--status|--dry-run]" \
  "$@"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$(kore_resolve_python "$REPO_ROOT")"
RUNTIME="$(kore_private_runtime)"
BASE_SESSION="${KORE_TMUX:-kore14b}"
RUN_ID="${KORE_RUN_ID:-$(kore_new_run_id tmux-14b)}"
SESSION="${BASE_SESSION}-${RUN_ID: -8}"
LOGDIR="$REPO_ROOT/runs/full/logs"
STATE_DIR="$RUNTIME/runs/$RUN_ID/tmux"
ENV_FILE="$STATE_DIR/environment.sh"
ACTIVE_REL="active/tmux-campaign.json"
RESULT_REL="runs/$RUN_ID/tmux/result.json"

export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export KORE_ALLOW_DEPRECATED_DEV=1
kore_require_commands tmux od stat

read_active() {
  PYTHONPATH="$REPO_ROOT" "$PY" -c '
import sys
from pathlib import Path
from kore.ops.runtime import SecureRuntime
value = SecureRuntime(sys.argv[1], create=False).read_json(
    Path("active") / "tmux-campaign.json"
)
for key in ("session", "run_id", "log_path", "result_state"):
    print(value.get(key, ""))
' "$RUNTIME"
}

if [[ "${1:-}" == "--status" ]]; then
  if ! active_text="$(read_active 2>/dev/null)"; then
    echo "[tmux] no owned campaign state."
    exit 3
  fi
  mapfile -t active <<<"$active_text"
  if [[ "${#active[@]}" -lt 4 ]]; then
    echo "[tmux] malformed owned campaign state." >&2
    exit 4
  fi
  active_session="${active[0]}"
  active_run="${active[1]}"
  active_log="${active[2]}"
  result_state="${active[3]}"
  if tmux has-session -t "$active_session" 2>/dev/null; then
    echo "[tmux] owned session '$active_session' is RUNNING run_id=$active_run."
    echo "[tmux] log: $active_log"
    exit 0
  fi
  if ! result="$(
    PYTHONPATH="$REPO_ROOT" "$PY" -c '
import sys
from kore.ops.runtime import SecureRuntime
print(SecureRuntime(sys.argv[1], create=False).read_json(sys.argv[2])["returncode"])
' "$RUNTIME" "$result_state" 2>/dev/null
  )"; then
    echo "[tmux] session ended without an owned result (stale or failed startup)." >&2
    exit 4
  fi
  echo "[tmux] session ended run_id=$active_run rc=$result log=$active_log"
  exit "$result"
fi

mkdir -p "$LOGDIR"
mkdir -m 0700 -p -- "$STATE_DIR"
kore_secure_source_env "$REPO_ROOT/.env.local"
export -p > "$ENV_FILE"
chmod 0600 -- "$ENV_FILE"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/campaign_${TS}_${RUN_ID}.log"

PYTHONPATH="$REPO_ROOT" "$PY" -c '
import sys
from pathlib import Path
from kore.ops.runtime import SecureRuntime
runtime, run_id, session, log_path, result_state = sys.argv[1:]
SecureRuntime(runtime).write_json(Path("active") / "tmux-campaign.json", {
    "schema": 1,
    "run_id": run_id,
    "session": session,
    "log_path": log_path,
    "result_state": result_state,
})
' "$RUNTIME" "$RUN_ID" "$SESSION" "$LOG" "$RESULT_REL"

if tmux has-session -t "$BASE_SESSION" 2>/dev/null; then
  echo "[tmux] warning: ignoring old unowned fixed session '$BASE_SESSION'; it will not block this run." >&2
fi
echo "[tmux] starting owned session '$SESSION' run_id=$RUN_ID"
echo "[tmux] log: $LOG"

RESULT_WRITER='
import sys
from kore.ops.runtime import SecureRuntime
SecureRuntime(sys.argv[1]).write_json(sys.argv[2], {
    "schema": 1,
    "returncode": int(sys.argv[3]),
})
'
tmux new-session -d -s "$SESSION" \
  bash -c '
    set -o pipefail
    source "$1"
    bash "$2" 2>&1 | tee -a "$3"
    rc=${PIPESTATUS[0]}
    PYTHONPATH="$4" "$5" -c "$8" "$6" "$7" "$rc"
    exit "$rc"
  ' -- "$ENV_FILE" "$SCRIPT_DIR/run_conductor_14b.sh" "$LOG" \
  "$REPO_ROOT" "$PY" "$RUNTIME" "$RESULT_REL" "$RESULT_WRITER"

sleep 1
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux] launched OK."
  echo "   tmux attach -t $SESSION"
  echo "   tail -f $LOG"
  echo "   bash scripts/tmux_campaign.sh --status"
  exit 0
fi
echo "[tmux] session ended during startup; reading owned result." >&2
exec bash "$0" --status
