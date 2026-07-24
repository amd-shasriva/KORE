#!/bin/bash
# two_node_maximize.sh - drive BOTH b05 nodes to a COMPLETE, DEEPENED dataset,
# end-to-end and unattended. Run inside a tmux session on b05-2 (the survivor).
#
# Strategy (disjoint, resume-safe, self-consolidating):
#   1. Partition every undone train task into two cost-balanced, DISJOINT halves
#      (A = this node / b05-2, B = peer / b05-1) via _kf_split.py.
#   2. Snapshot-sync code + the canonical dataset to the peer so it works from the
#      current state (no duplicated, already-finished work).
#   3. Each node runs deepen (wins->3) + base (repair/groups) on ITS half, each
#      saturating its own 8 GPUs, in tmux, resume-safe.
#   4. Every cycle: pull the peer's finished shards back (rsync --update; safe
#      because the halves are disjoint) so losing the peer never costs finished
#      work; log progress.
#   5. When both nodes go idle: final consolidate, verify, and run a cleanup pass
#      on b05-2 for ANY straggler (covers peer loss / partials) until 100%.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/ops_runtime.sh
source "$SCRIPT_DIR/lib/ops_runtime.sh"
kore_deprecated_guard \
  "scripts/two_node_maximize.sh" \
  "use scripts/spur_supervise_datagen.py; SSH/tmux b05 orchestration is retired" \
  "bash scripts/two_node_maximize.sh [--dry-run]" \
  "$@"

PEER="${KORE_PEER:-cv350-tnndh2-b05-1.tnn.dcgpu}"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$(kore_resolve_python "$REPO")"
RUNTIME="$(kore_private_runtime)"
DATA="${KORE_DATA_ROOT:-data/b05factory}"
TARGET="${KORE_WINS_TARGET:-3}"
WD="${KORE_DEEPEN_WORKERS:-80}"
WB="${KORE_BASE_WORKERS:-32}"
CYCLE="${KORE_MONITOR_CADENCE:-300}"
MAXCYC="${KORE_MONITOR_MAX_CYCLES:-288}"
RUN_ID="${KORE_RUN_ID:-$(kore_new_run_id two-node)}"
STATE_DIR="$RUNTIME/runs/$RUN_ID/two-node"
SESSION_DEEP_A="kf-deepA-${RUN_ID: -8}"
SESSION_BASE_A="kf-baseA-${RUN_ID: -8}"
SESSION_DEEP_B="kf-deepB-${RUN_ID: -8}"
SESSION_BASE_B="kf-baseB-${RUN_ID: -8}"
mkdir -m 0700 -p -- "$STATE_DIR"
cd "$REPO"
mkdir -p runs
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export KORE_RUN_ID="$RUN_ID"
kore_secure_source_env "$REPO/.env.local"
kore_require_commands ssh rsync tmux timeout od stat
LOG="runs/two_node_maximize_${RUN_ID}.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
peer_up() { timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=8 "$PEER" 'echo ok' 2>/dev/null | grep -q ok; }
local_jobs() {
  local count=0
  tmux has-session -t "$SESSION_DEEP_A" 2>/dev/null && count=$((count + 1))
  tmux has-session -t "$SESSION_BASE_A" 2>/dev/null && count=$((count + 1))
  echo "$count"
}
peer_jobs() {
  if ! peer_up; then
    echo 0
    return
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$PEER" \
    "n=0; tmux has-session -t '$SESSION_DEEP_B' 2>/dev/null && n=\$((n+1)); tmux has-session -t '$SESSION_BASE_B' 2>/dev/null && n=\$((n+1)); echo \$n"
}
# Disjoint halves => pull only the peer's newer shards; never clobber our own half.
consolidate() {
  if peer_up; then
    rsync -a --update "$PEER:$REPO/$DATA/" "$DATA/" &&
      say "consolidated peer->self"
  fi
}

say "==================== START two_node_maximize run_id=$RUN_ID $(date) ===================="

# 1. partition
PYTHONPATH="$REPO" "$PY" scripts/_kf_split.py "$DATA" "$TARGET" \
  --out-dir "$STATE_DIR" | tee -a "$LOG" || { say "SPLIT FAILED"; exit 1; }

PEER_OK=0; peer_up && PEER_OK=1

# 2. sync code + dataset + task lists to peer
if [ "$PEER_OK" = 1 ]; then
  say "syncing code + dataset to peer (few min over 10G subnet)..."
  rsync -a --exclude __pycache__ --exclude '*.pyc' scripts/ "$PEER:$REPO/scripts/" 2>/dev/null
  rsync -a --exclude __pycache__ --exclude '*.pyc' kore/ "$PEER:$REPO/kore/" 2>/dev/null
  rsync -a "$DATA/" "$PEER:$REPO/$DATA/" 2>/dev/null
  ssh "$PEER" "mkdir -p '$STATE_DIR' && chmod 0700 '$STATE_DIR'"
  rsync -a "$STATE_DIR/half_B.txt" "$STATE_DIR/base_B.txt" "$PEER:$STATE_DIR/"
  say "sync done"
else
  say "PEER DOWN at start -> single-node mode: b05-2 absorbs everything"
  cat "$STATE_DIR/half_A.txt" "$STATE_DIR/half_B.txt" | tr , '\n' | grep . | sort -u | paste -sd, - > "$STATE_DIR/half_A.new"
  mv "$STATE_DIR/half_A.new" "$STATE_DIR/half_A.txt"
  cat "$STATE_DIR/base_A.txt" "$STATE_DIR/base_B.txt" | tr , '\n' | grep . | sort -u | paste -sd, - > "$STATE_DIR/base_A.new"
  mv "$STATE_DIR/base_A.new" "$STATE_DIR/base_A.txt"
fi

# 3. launch workers (each node saturates its own 8 GPUs)
bash scripts/_kf_worker.sh deepen "$STATE_DIR/half_A.txt" "$WD" "$SESSION_DEEP_A" | tee -a "$LOG"
bash scripts/_kf_worker.sh base   "$STATE_DIR/base_A.txt" "$WB" "$SESSION_BASE_A" | tee -a "$LOG"
if [ "$PEER_OK" = 1 ]; then
  ssh "$PEER" "KORE_ALLOW_DEPRECATED_DEV=1 KORE_RUN_ID='$RUN_ID' bash '$REPO/scripts/_kf_worker.sh' deepen '$STATE_DIR/half_B.txt' '$WD' '$SESSION_DEEP_B'" 2>&1 | tee -a "$LOG"
  ssh "$PEER" "KORE_ALLOW_DEPRECATED_DEV=1 KORE_RUN_ID='$RUN_ID' bash '$REPO/scripts/_kf_worker.sh' base '$STATE_DIR/base_B.txt' '$WB' '$SESSION_BASE_B'" 2>&1 | tee -a "$LOG"
fi
say "workers launched (WD=$WD WB=$WB per node)"

# 4. monitor + periodic consolidation
finished=0
for cyc in $(seq 1 "$MAXCYC"); do
  sleep "$CYCLE"
  RL=$(local_jobs); RP=$(peer_jobs)
  DA=$(grep -c '\[deepen w' "runs/$SESSION_DEEP_A.log" 2>/dev/null || true)
  say "cycle $cyc: local_jobs=$RL peer_jobs=$RP deepA_completions=$DA"
  consolidate
  if [ "$RL" = 0 ] && [ "$RP" = 0 ]; then
    say "both nodes idle -> finishing"
    finished=1
    break
  fi
done
if [ "$finished" != 1 ]; then
  say "GIVEUP: workers still active at monitor deadline"
  exit 6
fi

# 5. final consolidate + verify + straggler cleanup on b05-2
consolidate
say "verifying..."
CLEANUP="$STATE_DIR/cleanup.txt"
PYTHONPATH="$REPO" "$PY" scripts/_kf_verify.py "$DATA" "$TARGET" \
  --cleanup-out "$CLEANUP" | tee -a "$LOG"
if [ -s "$CLEANUP" ] && [ "$(tr , '\n' <"$CLEANUP" | grep -c .)" != 0 ]; then
  N=$(tr , '\n' <"$CLEANUP" | grep -c .)
  say "cleanup pass on b05-2 for $N straggler task(s)"
  bash scripts/_kf_worker.sh deepen "$CLEANUP" "$WD" "$SESSION_DEEP_A-cleanup" | tee -a "$LOG"
  bash scripts/_kf_worker.sh base "$CLEANUP" "$WB" "$SESSION_BASE_A-cleanup" | tee -a "$LOG"
  cleanup_done=0
  for cyc in $(seq 1 "$MAXCYC"); do
    jobs=0
    tmux has-session -t "$SESSION_DEEP_A-cleanup" 2>/dev/null && jobs=$((jobs + 1))
    tmux has-session -t "$SESSION_BASE_A-cleanup" 2>/dev/null && jobs=$((jobs + 1))
    if [ "$jobs" = 0 ]; then cleanup_done=1; break; fi
    sleep "$CYCLE"
  done
  [ "$cleanup_done" = 1 ] || { say "GIVEUP: cleanup deadline reached"; exit 6; }
fi
PYTHONPATH="$REPO" "$PY" scripts/_kf_verify.py "$DATA" "$TARGET" \
  --cleanup-out "$CLEANUP" --require-complete | tee -a "$LOG"
say "==================== two_node_maximize VERIFIED COMPLETE $(date) ===================="
