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
set -u

PEER=cv350-tnndh2-b05-1.tnn.dcgpu
REPO=/home/shasriva/Kore-RL/KORE
VENV=/home/shasriva/kore-venv/bin/python
DATA=data/b05factory
TARGET=3
WD=80           # deepen workers per node
WB=32           # base workers per node
CYCLE=300       # monitor cadence (seconds)
MAXCYC=288      # safety cap (~24h)

cd "$REPO" || exit 1
mkdir -p runs
LOG=runs/two_node_maximize.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
peer_up() { timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=8 "$PEER" 'echo ok' 2>/dev/null | grep -q ok; }
local_jobs() { tmux ls 2>/dev/null | grep -cE '^(kf_deepA|kf_baseA):'; }
peer_jobs() { peer_up && ssh -o BatchMode=yes -o ConnectTimeout=8 "$PEER" "tmux ls 2>/dev/null | grep -cE '^(kf_deepB|kf_baseB):'" 2>/dev/null || echo 0; }
# Disjoint halves => pull only the peer's newer shards; never clobber our own half.
consolidate() { if peer_up; then rsync -a --update "$PEER:$REPO/$DATA/" "$DATA/" 2>/dev/null && say "consolidated peer->self"; fi; }

say "==================== START two_node_maximize $(date) ===================="

# 0. clean slate
for s in deepen base kf_deepA kf_baseA; do tmux kill-session -t "$s" 2>/dev/null; done
if peer_up; then ssh "$PEER" "for s in deepen base kf_deepB kf_baseB; do tmux kill-session -t \$s 2>/dev/null; done" 2>/dev/null; fi
sleep 2

# 1. partition
PYTHONPATH=. "$VENV" scripts/_kf_split.py "$DATA" "$TARGET" | tee -a "$LOG" || { say "SPLIT FAILED"; exit 1; }

PEER_OK=0; peer_up && PEER_OK=1

# 2. sync code + dataset + task lists to peer
if [ "$PEER_OK" = 1 ]; then
  say "syncing code + dataset to peer (few min over 10G subnet)..."
  rsync -a --exclude __pycache__ --exclude '*.pyc' scripts/ "$PEER:$REPO/scripts/" 2>/dev/null
  rsync -a --exclude __pycache__ --exclude '*.pyc' kore/ "$PEER:$REPO/kore/" 2>/dev/null
  rsync -a "$DATA/" "$PEER:$REPO/$DATA/" 2>/dev/null
  rsync -a /tmp/half_B.txt /tmp/base_B.txt "$PEER:/tmp/" 2>/dev/null
  say "sync done"
else
  say "PEER DOWN at start -> single-node mode: b05-2 absorbs everything"
  cat /tmp/half_A.txt /tmp/half_B.txt | tr , '\n' | grep . | sort -u | paste -sd, - > /tmp/half_A.txt
  cat /tmp/base_A.txt /tmp/base_B.txt | tr , '\n' | grep . | sort -u | paste -sd, - > /tmp/base_A.txt
fi

# 3. launch workers (each node saturates its own 8 GPUs)
bash scripts/_kf_worker.sh deepen /tmp/half_A.txt "$WD" kf_deepA | tee -a "$LOG"
bash scripts/_kf_worker.sh base   /tmp/base_A.txt "$WB" kf_baseA | tee -a "$LOG"
if [ "$PEER_OK" = 1 ]; then
  ssh "$PEER" "bash $REPO/scripts/_kf_worker.sh deepen /tmp/half_B.txt $WD kf_deepB" 2>&1 | tee -a "$LOG"
  ssh "$PEER" "bash $REPO/scripts/_kf_worker.sh base   /tmp/base_B.txt $WB kf_baseB" 2>&1 | tee -a "$LOG"
fi
say "workers launched (WD=$WD WB=$WB per node)"

# 4. monitor + periodic consolidation
for cyc in $(seq 1 "$MAXCYC"); do
  sleep "$CYCLE"
  RL=$(local_jobs); RP=$(peer_jobs)
  DA=$(grep -c '\[deepen w' runs/kf_deepA.log 2>/dev/null || echo 0)
  say "cycle $cyc: local_jobs=$RL peer_jobs=$RP deepA_completions=$DA"
  consolidate
  if [ "$RL" = 0 ] && [ "$RP" = 0 ]; then say "both nodes idle -> finishing"; break; fi
done

# 5. final consolidate + verify + straggler cleanup on b05-2
consolidate
say "verifying..."
PYTHONPATH=. "$VENV" scripts/_kf_verify.py "$DATA" "$TARGET" | tee -a "$LOG"
if [ -s /tmp/cleanup.txt ] && [ "$(tr , '\n' </tmp/cleanup.txt | grep -c .)" != 0 ]; then
  N=$(tr , '\n' </tmp/cleanup.txt | grep -c .)
  say "cleanup pass on b05-2 for $N straggler task(s)"
  bash scripts/_kf_worker.sh deepen /tmp/cleanup.txt "$WD" kf_deepA | tee -a "$LOG"
  bash scripts/_kf_worker.sh base   /tmp/cleanup.txt "$WB" kf_baseA | tee -a "$LOG"
  while [ "$(local_jobs)" != 0 ]; do sleep "$CYCLE"; say "cleanup running... ($(local_jobs) jobs)"; done
  PYTHONPATH=. "$VENV" scripts/_kf_verify.py "$DATA" "$TARGET" | tee -a "$LOG"
fi
say "==================== two_node_maximize COMPLETE $(date) ===================="
