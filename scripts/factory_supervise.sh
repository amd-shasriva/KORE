#!/bin/bash
# b05-2 DATA-FACTORY supervisor: keep the (resumable) breadth datagen alive and
# periodically push verified records back to b05-1 for the 32B run.
#
# Co-tenant contract: each relaunch uses only currently-IDLE GPUs, auto-picked as
# HIP indices (rocm-smi's physical order != HIP order on this node - see
# scripts/gpu_pick_hip.py), capped by FACTORY_MAX_GPUS to leave headroom for the
# serving containers; NEVER touches b05-1's run. Datagen is shard-resumable, so a
# relaunch continues where it stopped.
set -u
REPO=/home/shasriva/Kore-RL/KORE
VENV=/home/shasriva/kore-venv/bin/python
DATA_ROOT=data/b05factory
PEER=cv350-tnndh2-b05-1.tnn.dcgpu
GPUS_FIXED="${FACTORY_GPUS:-}"          # optional fixed HIP set; empty => dynamic idle pick
GPUS_FALLBACK="3,4,6"                    # HIP ids (== physical 1,7,6) if the picker returns nothing
MAX_GPUS="${FACTORY_MAX_GPUS:-6}"        # cap: leave headroom for the serving containers
UTIL_MAX="${FACTORY_UTIL_MAX:-30}"
VRAM_MAX_GB="${FACTORY_VRAM_MAX_GB:-12}"
WORKERS="${FACTORY_WORKERS:-48}"         # teacher-bound concurrency (feeds the extra GPUs)
SYNC_EVERY=600
cd "$REPO" || exit 1
mkdir -p runs/factory_logs

# HIP indices of the currently-idle physical GPUs (correct physical->HIP mapping),
# capped at MAX_GPUS. Fixed override wins; fall back to the static set on any failure.
pick_gpus() {
  if [ -n "$GPUS_FIXED" ]; then echo "$GPUS_FIXED"; return; fi
  local sel
  sel=$(SFT_UTIL_MAX="$UTIL_MAX" SFT_VRAM_MAX_GB="$VRAM_MAX_GB" GATE_NGPU="$MAX_GPUS" \
        "$VENV" scripts/gpu_pick_hip.py 2>/dev/null | cut -f1)
  [ -n "$sel" ] && echo "$sel" || echo "$GPUS_FALLBACK"
}

# BREADTH-ONLY + FRONTIER-FIRST: target only genb_* op-class tasks that do NOT yet
# have verified group data, so a relaunch never re-walks completed tasks (those just
# replay-cache-hit and burn no-op cycles - GPUs idle, no new records). Recomputed
# each relaunch so covered tasks drop out automatically; once every task is covered
# it falls back to the full genb_ set (a depth pass). Empty only if the query fails.
compute_todo() {
  PYTHONPATH=. "$VENV" -c "
import os, glob
from kore.tasks.registry import train_tasks
ids=sorted(t.task_id for t in train_tasks() if t.task_id.startswith('genb_'))
covered={os.path.basename(f)[:-6] for f in glob.glob('${DATA_ROOT}/groups/*.jsonl') if os.path.getsize(f)>0}
todo=[i for i in ids if i not in covered]
print(','.join(todo if todo else ids))
" 2>/dev/null
}
echo "FACTORY_SUPERVISOR start (dynamic idle GPUs cap=${MAX_GPUS}, frontier-first) workers=$WORKERS $(date)"
while true; do
  if ! pgrep -u shasriva -f "run_campaign.py.*${DATA_ROOT}" >/dev/null 2>&1; then
    SEL=$(pick_gpus)
    TASKS=$(compute_todo)
    TASKS_ARG=""; [ -n "$TASKS" ] && TASKS_ARG="--tasks $TASKS"
    NTODO=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c .)
    TS=$(date +%Y%m%d_%H%M%S)
    echo "ALERT relaunch datagen (frontier-first) gpus=[${SEL}] workers=${WORKERS} uncovered_tasks=${NTODO} -> factory_${TS}.log $(date)"
    setsid nohup env KORE_VERIFIED_CORRECTNESS=1 KORE_COMPILE_BASELINE=1 \
      KORE_BENCH_COLD=1 KORE_SHAPE_AUGMENT=1 PYTHONPATH=. \
      "$VENV" scripts/run_campaign.py --model Qwen/Qwen3-14B --stages datagen \
      --data-root "$DATA_ROOT" --teacher claude --datagen-workers "$WORKERS" \
      --gpu-ids "$SEL" $TASKS_ARG > "runs/factory_logs/factory_${TS}.log" 2>&1 &
    sleep 30
  fi
  # Push verified breadth data back to b05-1 (fail-safe; never blocks the loop).
  rsync -a --timeout=120 "$DATA_ROOT"/ \
    "$PEER":/home/shasriva/Kore-RL/KORE/data/b05factory_synced/ 2>/dev/null \
    && echo "SYNC ok $(date)" || echo "SYNC skipped $(date)"
  sleep "$SYNC_EVERY"
done
