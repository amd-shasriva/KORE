#!/bin/bash
# Two-node split datagen worker: run_campaign datagen over a FIXED task list
# (one disjoint half of the remaining uncovered genb_ tasks), re-running the
# shard-resumable pass until every task in the list has repair+groups+wins, then
# exit. Lets b05-1 and b05-2 grind disjoint halves in parallel with ZERO
# duplicated teacher calls (the throughput bottleneck).
#
# Usage: datagen_half.sh <half_tasks_file> <data_root> <gpu_ids> [workers]
set -u
REPO=/home/shasriva/Kore-RL/KORE
VENV=/home/shasriva/kore-venv/bin/python
HALF="${1:?half tasks file}"; DATA_ROOT="${2:?data root}"; GPUS="${3:?gpu ids}"; WORKERS="${4:-48}"
cd "$REPO" || exit 1
set -a; [ -f .env.local ] && . .env.local; set +a          # teacher creds
# FSDP-style: never let a stale mask pin every worker to one GPU (workers re-pin
# HIP per-worker internally from --gpu-ids).
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export KORE_VERIFIED_CORRECTNESS=1 KORE_COMPILE_BASELINE=1 KORE_BENCH_COLD=1 KORE_SHAPE_AUGMENT=1 PYTHONPATH=.

remaining() {
  PYTHONPATH=. "$VENV" - "$HALF" "$DATA_ROOT" <<'PY'
import os, sys
half, root = sys.argv[1], sys.argv[2]
ts = [t for t in open(half).read().strip().split(",") if t]
u = [t for t in ts if not all(
        os.path.exists("%s/%s/%s.jsonl" % (root, k, t)) for k in ("repair", "groups", "wins"))]
print(len(u))
PY
}

echo "[datagen_half] START half=$HALF data_root=$DATA_ROOT gpus=$GPUS workers=$WORKERS $(date)"
for i in $(seq 1 40); do
  R=$(remaining 2>/dev/null)
  echo "[datagen_half] pass=$i remaining=$R $(date)"
  [ "$R" = "0" ] && { echo "[datagen_half] COMPLETE (all tasks covered) $(date)"; break; }
  "$VENV" scripts/run_campaign.py --model Qwen/Qwen3-14B --stages datagen \
    --data-root "$DATA_ROOT" --teacher claude --datagen-workers "$WORKERS" \
    --gpu-ids "$GPUS" --tasks "$(cat "$HALF")"
  sleep 15
done
echo "[datagen_half] EXIT $(date)"
