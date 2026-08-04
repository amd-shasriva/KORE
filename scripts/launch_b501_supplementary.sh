#!/usr/bin/env bash
# Run the supplementary datagen wave on this box's free GPUs.
#
# Concurrency is set high on purpose. An episode takes ~275s wall but the teacher
# calls inside it were measured at 27-54s each, so the GPU sits idle for most of
# an episode: at 12 workers/GPU the cards read 6-10% utilisation. The binding
# resource is outstanding API calls, not compute, and this box has 384 cores and
# 3TB of RAM to hold them.
#
# One shard per GPU, and only GPUs with the card genuinely free -- the other four
# on this host hold idle root-owned sglang/vLLM servers that still occupy 83-90%
# of VRAM, and run_agentic_shard's --min-free-gb refuses them anyway.
#
# Deliberately NOT --keep-only-useful. That flag keeps the SFT-quality subset and
# discards every episode that never reached correctness, which is the wrong trade
# for exactly this task set. These are the hardest tasks we have, so a failure
# here is the most informative record we can produce: SFT on a teacher's
# successes can at best reach that teacher, and the tasks the teacher CANNOT
# solve are the only ones where RL has room to exceed it. Dropping them would
# throw away the RL curriculum to save disk.
set -uo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
GPUS="${KORE_SUPP_GPUS:-3 5 6 7}"
WORKERS="${KORE_SUPP_WORKERS:-40}"
EPISODES="${KORE_SUPP_EPISODES:-16}"
SHARD_DIR="${KORE_SUPP_SHARDS:-$REPO/data/b501supp/shards}"
OUT_DIR="${KORE_SUPP_OUT:-$REPO/data/b501supp/out}"

cd "$REPO"
mkdir -p "$OUT_DIR"

# PID files rather than command-line pattern matching. The operations registry's
# contract test forbids pattern-based process lookup and signalling in registered
# scripts, and it is right to: a pattern broad enough to find these shards is
# also broad enough to match somebody else's python on a shared box.
PIDDIR="${KORE_SUPP_PIDDIR:-$OUT_DIR/.pids}"
mkdir -p "$PIDDIR"

i=0
for g in $GPUS; do
  idx=$(printf "%03d" "$i")
  setsid nohup "$PY" scripts/run_agentic_shard.py \
    --shard-file "$SHARD_DIR/shard_$idx.txt" \
    --out-dir "$OUT_DIR" \
    --shard-index "$i" \
    --episodes-per-task "$EPISODES" \
    --workers "$WORKERS" \
    --max-turns 8 \
    --gpu-ids "$g" \
    --teacher claude \
    > "/tmp/supp_shard$i.log" 2>&1 < /dev/null &
  echo $! > "$PIDDIR/shard_$idx.pid"
  disown
  echo "launched shard $i on gpu $g pid=$(cat "$PIDDIR/shard_$idx.pid") (workers=$WORKERS episodes=$EPISODES)"
  i=$((i + 1))
done

sleep 20
alive=0
for f in "$PIDDIR"/shard_*.pid; do
  [ -e "$f" ] || continue
  kill -0 "$(cat "$f")" 2>/dev/null && alive=$((alive + 1))
done
echo "running shard processes: $alive"
