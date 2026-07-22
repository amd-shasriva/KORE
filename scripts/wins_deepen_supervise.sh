#!/bin/bash
# PHASE-2 (chained): additive GOLD-WINS deepening to TARGET distinct wins/task.
#
# Runs AFTER the repair/groups/wins split-datagen finishes on BOTH nodes -- they are
# GPU-bound, so overlapping would only contend (same total time, more fragility).
# Then it consolidates the FULL breadth dataset onto b05-1 (all 8 GPUs, no co-tenant
# sharing), runs the resume-safe / no-loss deepener, and merges the new wins back to
# b05-2 (the canonical accumulation point). genb_ TRAIN tasks only (held-out excluded).
set -u
REPO=/home/shasriva/Kore-RL/KORE; cd "$REPO" || exit 1
VENV=/home/shasriva/kore-venv/bin/python
PEER=cv350-tnndh2-b05-2.tnn.dcgpu
DATA_ROOT=data/b05factory
TARGET="${WINS_TARGET:-3}"; GENS="${WINS_GENS:-8}"; WORKERS="${WINS_WORKERS:-48}"
set -a; [ -f .env.local ] && . .env.local; set +a
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export KORE_VERIFIED_CORRECTNESS=1 KORE_COMPILE_BASELINE=1 KORE_BENCH_COLD=1 KORE_SHAPE_AUGMENT=1 PYTHONPATH=.

# 1) wait for split datagen to finish on BOTH nodes. [d]atagen bracket-trick so the
#    pgrep/ssh command never matches ITSELF (which would make the count never hit 0).
echo "[wins_deepen] waiting for split datagen to finish on both nodes $(date)"
while true; do
  local_run=$(pgrep -f '[d]atagen_half.sh' | wc -l)
  peer_run=$(ssh -o BatchMode=yes "$PEER" "pgrep -f '[d]atagen_half.sh' | wc -l" 2>/dev/null | tail -1)
  peer_run=${peer_run:-1}
  echo "[wins_deepen] datagen running? local=$local_run peer=$peer_run $(date)"
  { [ "$local_run" = "0" ] && [ "$peer_run" = "0" ]; } && break
  sleep 180
done
echo "[wins_deepen] split datagen COMPLETE on both nodes $(date)"

# 2) consolidate the FULL dataset onto b05-1 (it holds only half_A; pull the rest -
#    795 originals + half_B - from the canonical peer). Additive rsync; never deletes.
echo "[wins_deepen] consolidating full dataset b05-2 -> b05-1 $(date)"
rsync -a --timeout=600 "$PEER":"$REPO/$DATA_ROOT/" "$REPO/$DATA_ROOT/" 2>/dev/null \
  && echo "[wins_deepen] consolidate OK $(date)" || echo "[wins_deepen] consolidate WARN $(date)"

# 3) deepen genb_ TRAIN wins additively on b05-1's 8 GPUs.
TASKS=$(PYTHONPATH=. "$VENV" -c "from kore.tasks.registry import train_tasks; print(','.join(t.task_id for t in train_tasks() if t.task_id.startswith('genb_')))")
NT=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c .)
echo "[wins_deepen] deepening $NT genb_ train tasks -> target=$TARGET (gens=$GENS, 8 GPUs) $(date)"
"$VENV" scripts/deepen_wins.py --data-root "$DATA_ROOT" --tasks "$TASKS" \
  --gpu-ids 0,1,2,3,4,5,6,7 --workers "$WORKERS" --target "$TARGET" --gens "$GENS"

# 4) merge the deepened wins back to the canonical peer (disjoint: only wins/ changed).
echo "[wins_deepen] merging deepened wins b05-1 -> b05-2 $(date)"
rsync -a --timeout=600 "$REPO/$DATA_ROOT/wins/" "$PEER":"$REPO/$DATA_ROOT/wins/" 2>/dev/null \
  && echo "[wins_deepen] merge OK $(date)" || echo "[wins_deepen] merge WARN $(date)"
echo "[wins_deepen] ALL DONE $(date)"
