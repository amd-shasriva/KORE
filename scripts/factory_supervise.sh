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

# BREADTH-ONLY + FRONTIER-FIRST + COMPLETE: a genb_ task is DONE only when ALL THREE
# datagen shards (repair, groups, wins) exist, so a relaunch never re-walks finished
# tasks yet no task is ever left partially covered (the old groups-only test could
# strand a task that had groups but not wins). File-existence (not size) is used so a
# kind that legitimately yields zero records still counts as attempted and cannot loop
# forever. Emits 'ALLDONE' when every task is complete (-> supervisor exits cleanly),
# the CSV todo otherwise, and NOTHING on a registry error (-> caller retries, so an
# error is never mistaken for completion).
compute_todo() {
  PYTHONPATH=. "$VENV" -c "
import os, sys
from kore.tasks.registry import train_tasks
ids=sorted(t.task_id for t in train_tasks() if t.task_id.startswith('genb_'))
if not ids: sys.exit(3)
done=lambda tid: all(os.path.exists('${DATA_ROOT}/%s/%s.jsonl' % (k, tid)) for k in ('repair','groups','wins'))
todo=[i for i in ids if not done(i)]
print(','.join(todo) if todo else 'ALLDONE')
" 2>/dev/null
}
echo "FACTORY_SUPERVISOR start (dynamic idle GPUs cap=${MAX_GPUS}, frontier-first) workers=$WORKERS $(date)"
while true; do
  if ! pgrep -u shasriva -f "run_campaign.py.*${DATA_ROOT}" >/dev/null 2>&1; then
    TASKS=$(compute_todo)
    if [ "$TASKS" = "ALLDONE" ]; then
      # Every genb_ task has repair+groups+wins: do a final sync, then STOP cleanly.
      rsync -a --timeout=120 "$DATA_ROOT"/ \
        "$PEER":/home/shasriva/Kore-RL/KORE/data/b05factory_synced/ 2>/dev/null \
        && echo "SYNC ok (final) $(date)" || echo "SYNC skipped (final) $(date)"
      echo "FACTORY COMPLETE: all genb_ tasks have repair+groups+wins - exiting. $(date)"
      exit 0
    elif [ -z "$TASKS" ]; then
      # compute_todo failed (registry/query error): never mistake this for completion.
      echo "WARN compute_todo returned empty (query error) - retry next cycle $(date)"
    else
      SEL=$(pick_gpus)
      NTODO=$(printf '%s' "$TASKS" | tr ',' '\n' | grep -c .)
      TS=$(date +%Y%m%d_%H%M%S)
      echo "ALERT relaunch datagen (frontier-first) gpus=[${SEL}] workers=${WORKERS} uncovered_tasks=${NTODO} -> factory_${TS}.log $(date)"
      setsid nohup env KORE_VERIFIED_CORRECTNESS=1 KORE_COMPILE_BASELINE=1 \
        KORE_BENCH_COLD=1 KORE_SHAPE_AUGMENT=1 PYTHONPATH=. \
        "$VENV" scripts/run_campaign.py --model Qwen/Qwen3-14B --stages datagen \
        --data-root "$DATA_ROOT" --teacher claude --datagen-workers "$WORKERS" \
        --gpu-ids "$SEL" --tasks "$TASKS" > "runs/factory_logs/factory_${TS}.log" 2>&1 &
      sleep 30
    fi
  fi
  # Push verified breadth data back to b05-1 (fail-safe; never blocks the loop).
  rsync -a --timeout=120 "$DATA_ROOT"/ \
    "$PEER":/home/shasriva/Kore-RL/KORE/data/b05factory_synced/ 2>/dev/null \
    && echo "SYNC ok $(date)" || echo "SYNC skipped $(date)"
  sleep "$SYNC_EVERY"
done
