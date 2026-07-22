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
#   3. deepen in parallel (b05-1: 8 GPUs; b05-2: free co-tenant-safe GPUs);
#   4. one-way DISJOINT merge of each node's deepened half -> canonical b05-2.
# The deepener itself (scripts/deepen_wins.py) is additive + resume-safe: existing
# wins are never lost or regenerated, tasks already at target cost zero teacher calls.
set -u
REPO=/home/shasriva/Kore-RL/KORE; cd "$REPO" || exit 1
VENV=/home/shasriva/kore-venv/bin/python
PEER=cv350-tnndh2-b05-2.tnn.dcgpu
DR=data/b05factory
TARGET="${WINS_TARGET:-3}"; GENS="${WINS_GENS:-8}"; WORKERS="${WINS_WORKERS:-48}"
set -a; [ -f .env.local ] && . .env.local; set +a
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export KORE_VERIFIED_CORRECTNESS=1 KORE_COMPILE_BASELINE=1 KORE_BENCH_COLD=1 KORE_SHAPE_AUGMENT=1 PYTHONPATH=.

log(){ echo "[wins_deepen] $* $(date)"; }

# 1) wait for split datagen on BOTH nodes ([d]atagen bracket-trick: never self-match).
log "waiting for split datagen to finish on both nodes"
while true; do
  l=$(pgrep -f '[d]atagen_half.sh' | wc -l)
  p=$(ssh -o BatchMode=yes "$PEER" "pgrep -f '[d]atagen_half.sh' | wc -l" 2>/dev/null | tail -1); p=${p:-1}
  log "datagen running? local=$l peer=$p"
  { [ "$l" = "0" ] && [ "$p" = "0" ]; } && break
  sleep 180
done
log "split datagen COMPLETE on both nodes"

# 2) consolidate: pull the peer's full dataset onto b05-1 (which holds only half_A).
#    b05-2 already holds everything via the running merge loop, so after this BOTH
#    nodes hold every wins shard.
log "consolidating full dataset b05-2 -> b05-1"
rsync -a --timeout=900 "$PEER":"$REPO/$DR/" "$REPO/$DR/" 2>/dev/null && log "consolidate OK" || log "consolidate WARN"

# 3) disjoint split of genb_ TRAIN tasks (held-out already excluded by train_tasks).
PYTHONPATH=. "$VENV" - <<'PY'
from kore.tasks.registry import train_tasks
ts=sorted(t.task_id for t in train_tasks() if t.task_id.startswith("genb_"))
A=ts[0::2]; B=ts[1::2]
open("/tmp/deepen_A.txt","w").write(",".join(A))
open("/tmp/deepen_B.txt","w").write(",".join(B))
print("deepen split: A(b05-1)=%d  B(b05-2)=%d"%(len(A),len(B)))
PY
# files-from list for the disjoint one-way merge (b05-1 owns only its A wins shards)
: > /tmp/deepen_A_wins.txt
for t in $(tr ',' ' ' < /tmp/deepen_A.txt); do echo "$DR/wins/$t.jsonl" >> /tmp/deepen_A_wins.txt; done
log "split: A(b05-1)=$(tr ',' '\n' </tmp/deepen_A.txt|grep -c .)  B(b05-2)=$(tr ',' '\n' </tmp/deepen_B.txt|grep -c .)"

# 4) deploy deepener + B-list to peer; launch peer deepener on its free GPUs.
scp -o BatchMode=yes scripts/deepen_wins.py "$PEER":"$REPO/scripts/deepen_wins.py" >/dev/null 2>&1
scp -o BatchMode=yes /tmp/deepen_B.txt "$PEER":/tmp/deepen_B.txt >/dev/null 2>&1
PEER_GPUS=$(ssh -o BatchMode=yes "$PEER" "cd $REPO && SFT_UTIL_MAX=30 SFT_VRAM_MAX_GB=40 GATE_NGPU=6 PYTHONPATH=. $VENV scripts/gpu_pick_hip.py 2>/dev/null | cut -f1" 2>/dev/null | tail -1)
PEER_GPUS=${PEER_GPUS:-1,3,2,0,5,7}
log "launching b05-2 deepener (deepen_B) on GPUs=$PEER_GPUS"
ssh -o BatchMode=yes "$PEER" "cd $REPO && setsid nohup env PYTHONPATH=. $VENV scripts/deepen_wins.py --data-root $DR --tasks \"\$(cat /tmp/deepen_B.txt)\" --gpu-ids $PEER_GPUS --workers $WORKERS --target $TARGET --gens $GENS > runs/deepen_B_b05-2.log 2>&1 < /dev/null & echo peer_deepen_pid=\$!" 2>/dev/null | grep -i pid= || true

# 5) run b05-1 deepener (deepen_A) in the foreground on all 8 GPUs.
log "running b05-1 deepener (deepen_A) on 8 GPUs"
"$VENV" scripts/deepen_wins.py --data-root "$DR" --tasks "$(cat /tmp/deepen_A.txt)" \
  --gpu-ids 0,1,2,3,4,5,6,7 --workers "$WORKERS" --target "$TARGET" --gens "$GENS"
log "b05-1 deepener done; waiting for b05-2 deepener"
while [ "$(ssh -o BatchMode=yes "$PEER" "pgrep -f '[d]eepen_wins.py' | wc -l" 2>/dev/null | tail -1)" != "0" ]; do sleep 120; done
log "b05-2 deepener done"

# 6) one-way DISJOINT merge: b05-1's deepened A shards -> b05-2 (b05-2 keeps its own
#    deepened B). Disjoint task sets => no shard written twice => nothing lost.
log "merging b05-1 deepened A-wins -> canonical b05-2"
rsync -a --timeout=900 --files-from=/tmp/deepen_A_wins.txt "$REPO/" "$PEER":"$REPO/" 2>/dev/null && log "merge OK" || log "merge WARN"
log "ALL DONE (canonical full+deepened dataset on b05-2:$DR)"
