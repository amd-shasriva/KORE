#!/bin/bash
# PHASE-2 (chained): additive GOLD-WINS deepening to TARGET distinct wins/task,
# across BOTH nodes with ZERO wasted effort and ZERO win-loss risk.
#
# Runs AFTER the repair/groups/wins split-datagen finishes on both nodes (they are
# GPU-bound; overlapping only contends). Then:
#   1. consolidate the FULL breadth dataset onto BOTH nodes (each ends holding every
#      wins shard, so the deepener reads + preserves existing wins everywhere);
#   2. split genb_ TRAIN tasks into DISJOINT halves - b05-1 deepens A, b05-2 deepens B
#      - so no shard is written by both nodes (no merge conflict, nothing lost);
#   3. deepen in parallel (b05-1: 8 GPUs; b05-2: 8 GPUs - user holds the full box);
#   4. one-way DISJOINT merge of each node's deepened half -> canonical b05-2.
# The deepener itself (scripts/deepen_wins.py) is additive + resume-safe: existing
# wins are never lost or regenerated, tasks already at target cost zero teacher calls.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/ops_runtime.sh
source "$SCRIPT_DIR/lib/ops_runtime.sh"
kore_deprecated_guard \
  "scripts/wins_deepen_supervise.sh" \
  "use scripts/spur_supervise_datagen.py; the b05 SSH deepening topology is retired" \
  "bash scripts/wins_deepen_supervise.sh [--dry-run]" \
  "$@"

REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$(kore_resolve_python "$REPO")"
RUNTIME="$(kore_private_runtime)"
PEER="${WINS_PEER:-cv350-tnndh2-b05-2.tnn.dcgpu}"
DR="${KORE_DATA_ROOT:-data/b05factory}"
TARGET="${WINS_TARGET:-3}"
GENS="${WINS_GENS:-8}"
WORKERS="${WINS_WORKERS:-48}"
MAX_WAIT_CYCLES="${WINS_MAX_WAIT_CYCLES:-120}"
WAIT_SECONDS="${WINS_WAIT_SECONDS:-120}"
LOCAL_GPUS="${WINS_LOCAL_GPUS:-0,1,2,3,4,5,6,7}"
PEER_GPUS="${WINS_PEER_GPUS:-0,1,2,3,4,5,6,7}"
RUN_ID="${KORE_RUN_ID:-$(kore_new_run_id wins-deepen)}"
STATE_DIR="$RUNTIME/runs/$RUN_ID/wins-deepen"
SESSION_PEER="deepenB-${RUN_ID: -8}"
LOG="runs/wins_deepen_${RUN_ID}.log"
mkdir -m 0700 -p -- "$STATE_DIR"
cd "$REPO"
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export KORE_RUN_ID="$RUN_ID"
kore_require_commands ssh rsync tmux od stat
kore_secure_source_env "$REPO/.env.local"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
kore_export_rigor_env
for value in "$TARGET" "$GENS" "$WORKERS" "$MAX_WAIT_CYCLES" "$WAIT_SECONDS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: deepening settings must be positive integers" >&2
    exit 2
  }
done

log() { echo "[wins_deepen] $* $(date)" | tee -a "$LOG"; }

# Do not infer readiness from process-list substrings. The base dataset itself is
# authoritative, and an incomplete input is a hard failure.
BASE_CLEANUP="$STATE_DIR/base-incomplete.txt"
if ! "$PY" scripts/_kf_verify.py "$DR" 1 \
    --cleanup-out "$BASE_CLEANUP" --require-complete; then
  log "INCOMPLETE: base datagen verifier failed; refusing deepening"
  exit 4
fi

log "consolidating verified base dataset across the configured peer"
rsync -a --ignore-existing --timeout=900 "$REPO/$DR/" "$PEER:$REPO/$DR/"
rsync -a --ignore-existing --timeout=900 "$PEER:$REPO/$DR/" "$REPO/$DR/"

PYTHONPATH="$REPO" "$PY" - "$STATE_DIR" <<'PY'
import json
import os
from pathlib import Path
import sys
from kore.ops.runtime import task_set_identity
from kore.tasks.registry import train_tasks

root = Path(sys.argv[1])
tasks = sorted(t.task_id for t in train_tasks() if t.task_id.startswith("genb_"))
identity = task_set_identity(tasks)
a, b = identity.task_ids[0::2], identity.task_ids[1::2]
for name, values in (("deepen_A.txt", a), ("deepen_B.txt", b)):
    path = root / name
    path.write_text(",".join(values))
    path.chmod(0o600)
(root / "task-set.json").write_text(json.dumps({
    "schema": 1,
    "count": identity.count,
    "sha256": identity.sha256,
    "task_ids": list(identity.task_ids),
}, sort_keys=True) + "\n")
(root / "task-set.json").chmod(0o600)
print(f"deepen split: A={len(a)} B={len(b)} count={identity.count} sha256={identity.sha256}")
PY

ssh "$PEER" "mkdir -p '$STATE_DIR' && chmod 0700 '$STATE_DIR'"
rsync -a "$STATE_DIR/deepen_B.txt" "$PEER:$STATE_DIR/"
ssh "$PEER" \
  "KORE_ALLOW_DEPRECATED_DEV=1 KORE_RUN_ID='$RUN_ID' KORE_GPU_IDS='$PEER_GPUS' bash '$REPO/scripts/_kf_worker.sh' deepen '$STATE_DIR/deepen_B.txt' '$WORKERS' '$SESSION_PEER'"

log "running local owned deepener"
set +e
kore_owned_run "$PY" "$REPO" "$RUNTIME" "$RUN_ID" "deepen-A" "$LOG" \
  env PYTHONPATH="$REPO" "$PY" scripts/deepen_wins.py \
    --data-root "$DR" --tasks "$(cat "$STATE_DIR/deepen_A.txt")" \
    --gpu-ids "$LOCAL_GPUS" --workers "$WORKERS" --target "$TARGET" --gens "$GENS"
local_rc=$?
set -e
if (( local_rc != 0 )); then
  log "FAILED: local deepener rc=$local_rc"
  exit "$local_rc"
fi
kore_verify "$PY" "$REPO" task-shards \
  --data-root "$DR" --tasks-file "$STATE_DIR/deepen_A.txt" \
  --target-wins "$TARGET" --kinds wins

peer_done=0
for cycle in $(seq 1 "$MAX_WAIT_CYCLES"); do
  if ! ssh "$PEER" "tmux has-session -t '$SESSION_PEER'" 2>/dev/null; then
    peer_done=1
    break
  fi
  log "peer deepener active cycle=$cycle"
  sleep "$WAIT_SECONDS"
done
if (( peer_done != 1 )); then
  log "GIVEUP: peer session remained active through bounded wait"
  exit 6
fi
ssh "$PEER" \
  "cd '$REPO' && PYTHONPATH='$REPO' '$PY' -m kore.ops verify task-shards --data-root '$DR' --tasks-file '$STATE_DIR/deepen_B.txt' --target-wins '$TARGET' --kinds wins"

# Each half has one writer. Pull B's newer shards, then push A's newer shards.
rsync -a --update --timeout=900 "$PEER:$REPO/$DR/wins/" "$REPO/$DR/wins/"
rsync -a --update --timeout=900 "$REPO/$DR/wins/" "$PEER:$REPO/$DR/wins/"
FINAL_CLEANUP="$STATE_DIR/final-incomplete.txt"
"$PY" scripts/_kf_verify.py "$DR" "$TARGET" \
  --cleanup-out "$FINAL_CLEANUP" --require-complete
ssh "$PEER" \
  "cd '$REPO' && PYTHONPATH='$REPO' '$PY' scripts/_kf_verify.py '$DR' '$TARGET' --cleanup-out '$STATE_DIR/peer-final-incomplete.txt' --require-complete"
log "VERIFIED COMPLETE on both configured nodes run_id=$RUN_ID"
