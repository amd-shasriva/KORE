#!/usr/bin/env bash
# Launch the KORE 14B campaign inside a DURABLE tmux session with full logging.
#
# Why tmux: it detaches the training process from your SSH connection, so the run
# survives a dropped/closed terminal for the life of the node reservation. Files
# persist under your account, and the campaign is manifest-resumable, so if the
# reservation ends you just re-reserve and re-run this script — it resumes.
#
# Usage:
#   bash scripts/tmux_campaign.sh            # start (or report an existing run)
#   tmux attach -t kore14b                   # watch live  (Ctrl-b then d to detach)
#   tail -f runs/full/logs/campaign_*.log    # or follow the log file
#   bash scripts/tmux_campaign.sh --status   # quick status without attaching
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${KORE_TMUX:-kore14b}"
LOGDIR="$REPO_ROOT/runs/full/logs"
mkdir -p "$LOGDIR"

if [ "${1:-}" = "--status" ]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[tmux] session '$SESSION' is RUNNING."
    latest="$(ls -t "$LOGDIR"/campaign_*.log 2>/dev/null | head -1 || true)"
    [ -n "$latest" ] && { echo "[tmux] latest log: $latest"; echo "----- tail -----"; tail -n 15 "$latest"; }
  else
    echo "[tmux] no session '$SESSION' running."
  fi
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux] session '$SESSION' already exists — NOT starting a second run (GPU contention)."
  echo "[tmux] attach:  tmux attach -t $SESSION"
  echo "[tmux] status:  bash scripts/tmux_campaign.sh --status"
  exit 0
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/campaign_$TS.log"
echo "[tmux] starting campaign in session '$SESSION'"
echo "[tmux] log: $LOG"

# Run the launcher, tee to the log, keep the pane open afterwards so the exit
# code is visible on re-attach. PIPESTATUS[0] = the launcher's real exit code.
tmux new-session -d -s "$SESSION" \
  "bash '$SCRIPT_DIR/run_conductor_14b.sh' 2>&1 | tee '$LOG'; \
   rc=\${PIPESTATUS[0]}; echo; echo \"[tmux] campaign exited rc=\$rc\"; exec bash"

sleep 1
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux] launched OK. Next:"
  echo "   tmux attach -t $SESSION                 # watch live (Ctrl-b d to detach)"
  echo "   tail -f $LOG                            # follow the log"
  echo "   bash scripts/tmux_campaign.sh --status  # quick status"
  echo "[tmux] if the reservation ends: re-reserve, then re-run this script to resume."
else
  echo "[tmux] ERROR: session did not start; check $LOG"
  exit 1
fi
