#!/bin/bash
# b05-2 DATA-FACTORY supervisor: keep the (resumable) breadth datagen alive and
# periodically push verified records back to b05-1 for the 32B run.
#
# Co-tenant contract: pinned to idle GPUs (3,4,6), NEVER touches the 4 serving
# containers, NEVER touches b05-1's 14B run. Datagen is shard-resumable, so a
# relaunch continues where it stopped.
set -u
REPO=/home/shasriva/Kore-RL/KORE
VENV=/home/shasriva/kore-venv/bin/python
DATA_ROOT=data/b05factory
PEER=cv350-tnndh2-b05-1.tnn.dcgpu
GPUS="${FACTORY_GPUS:-3,4,6}"
WORKERS="${FACTORY_WORKERS:-16}"
SYNC_EVERY=600
cd "$REPO" || exit 1
mkdir -p runs/factory_logs

# BREADTH-ONLY: this factory exists to produce the NEW genb_* op-class families
# (the frontier expansion for the 32B). The 280 base tasks already have data
# (data/full14b, ~152k records the 14B trained on), so we spend 100% of factory
# compute on breadth. The genb_ set is read live from the registry so every newly
# authored+materialized family is picked up automatically on the next relaunch.
# Empty (=> full registry) only if the query fails (fail-safe).
TASKS=$(PYTHONPATH=. "$VENV" -c "
from kore.tasks.registry import train_tasks
ids=sorted(t.task_id for t in train_tasks() if t.task_id.startswith('genb_'))
print(','.join(ids))
" 2>/dev/null)
TASKS_ARG=""; [ -n "$TASKS" ] && TASKS_ARG="--tasks $TASKS"
NBREADTH=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c . )
echo "FACTORY_SUPERVISOR start gpus=$GPUS workers=$WORKERS breadth_only_tasks=$NBREADTH $(date)"
while true; do
  if ! pgrep -u shasriva -f "run_campaign.py.*${DATA_ROOT}" >/dev/null 2>&1; then
    TS=$(date +%Y%m%d_%H%M%S)
    echo "ALERT relaunch datagen (resumable, breadth-only) -> factory_${TS}.log $(date)"
    setsid nohup env KORE_VERIFIED_CORRECTNESS=1 KORE_COMPILE_BASELINE=1 \
      KORE_BENCH_COLD=1 KORE_SHAPE_AUGMENT=1 PYTHONPATH=. \
      "$VENV" scripts/run_campaign.py --model Qwen/Qwen3-14B --stages datagen \
      --data-root "$DATA_ROOT" --teacher claude --datagen-workers "$WORKERS" \
      --gpu-ids "$GPUS" $TASKS_ARG > "runs/factory_logs/factory_${TS}.log" 2>&1 &
    sleep 30
  fi
  # Push verified breadth data back to b05-1 (fail-safe; never blocks the loop).
  rsync -a --timeout=120 "$DATA_ROOT"/ \
    "$PEER":/home/shasriva/Kore-RL/KORE/data/b05factory_synced/ 2>/dev/null \
    && echo "SYNC ok $(date)" || echo "SYNC skipped $(date)"
  sleep "$SYNC_EVERY"
done
