#!/bin/bash
# _kf_worker.sh KIND TASKS_FILE WORKERS SESSION
#   KIND        = deepen | base
#   TASKS_FILE  = path to a comma-list of task ids (read at launch time)
#   WORKERS     = worker processes (GPU-pinned round-robin over all 8 GPUs)
#   SESSION     = tmux session name (also the log basename under runs/)
#
# Launches the requested resume-safe driver in a detached tmux session so it
# survives ssh disconnects. deepen -> deepen_wins.py (wins/ only); base ->
# complete_base.py (repair/ + groups/ only). The two never touch each other's
# shard kinds, so a node can run both at once.
set -u
KIND="${1:?kind}"; TASKS_FILE="${2:?tasks file}"; WORKERS="${3:?workers}"; SESSION="${4:?session}"
REPO=/home/shasriva/Kore-RL/KORE
VENV=/home/shasriva/kore-venv/bin/python
DATA=data/b05factory
GPUS=0,1,2,3,4,5,6,7
cd "$REPO" || exit 1
mkdir -p runs

if [ ! -s "$TASKS_FILE" ]; then
  echo "[_kf_worker] $SESSION: empty task list ($TASKS_FILE) - nothing to do"
  exit 0
fi

tmux kill-session -t "$SESSION" 2>/dev/null; sleep 1

COMMON="cd $REPO && set -a; [ -f .env.local ] && . .env.local; set +a; \
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; \
export KORE_VERIFIED_CORRECTNESS=1 KORE_COMPILE_BASELINE=1 KORE_BENCH_COLD=1 KORE_SHAPE_AUGMENT=1 KORE_WINS_PMC=0 PYTHONPATH=."

if [ "$KIND" = deepen ]; then
  CMD="$COMMON; exec $VENV scripts/deepen_wins.py --data-root $DATA --tasks \$(cat $TASKS_FILE) --gpu-ids $GPUS --workers $WORKERS --target 3 --gens 8 > runs/$SESSION.log 2>&1"
else
  CMD="$COMMON; exec $VENV scripts/complete_base.py --data-root $DATA --tasks \$(cat $TASKS_FILE) --gpu-ids $GPUS --workers $WORKERS --n-repair 50 --n-parents 20 --k 6 > runs/$SESSION.log 2>&1"
fi

tmux new-session -d -s "$SESSION" "$CMD"
sleep 2
if tmux ls 2>/dev/null | grep -q "^$SESSION:"; then
  echo "[_kf_worker] $SESSION: launched ($KIND, workers=$WORKERS, tasks=$(tr , '\n' <"$TASKS_FILE" | grep -c .))"
else
  echo "[_kf_worker] $SESSION: FAILED to launch"
  exit 1
fi
