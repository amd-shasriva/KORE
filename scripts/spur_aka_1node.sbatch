#!/bin/bash
# AgentKernelArena on a GPU node: discover | baseline | run.
#
# AKA ships a Docker runner and this cluster has no container runtime at all,
# so we drive the tasks' declared commands directly. They are plain argv, so the
# image was a reproducibility convenience rather than a dependency.
#
#   sbatch scripts/spur_aka_1node.sbatch <discover|baseline|run> [TYPES] [MODEL] [LIMIT] [ARM]
#SBATCH --job-name=kore-aka
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/aka-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/aka-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
source /home/shasriva/kore-venv/bin/activate
export PYTHONPATH="$REPO"
export HF_HOME=/home/shasriva/.cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

MODE="${1:-discover}"
TYPES="${2:-triton2triton}"     # '-' for all
MODEL="${3:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
LIMIT="${4:-0}"
ARM="${5:-kore}"
[ "$TYPES" = "-" ] && TYPES=""

# Sixth arg is an explicit output dir, and it is what makes a long arena run
# finishable. The full arena is 402 tasks with a 900s per-task ceiling, so it
# cannot fit in this script's 8h limit; it has to cross several allocations,
# resuming from the partial ledger each time. Keying the default on SLURM_JOB_ID
# gives every resubmission a fresh empty directory, which silently restarts the
# run from task 1 forever. Pass a stable path for any run you expect to resume.
OUT="${6:-$REPO/runs/aka_${SLURM_JOB_ID:-manual}}"
mkdir -p "$OUT"     # workers write their own logs here before Python starts
echo "host=$(hostname) mode=$MODE types=${TYPES:-ALL} model=$MODEL limit=$LIMIT arm=$ARM"
echo "arena=$HOME/third_party/AgentKernelArena  out=$OUT"

# The tasks' harnesses import torch/triton from OUR venv, so a missing GPU here
# shows up as every task failing correctness rather than as an obvious error.
VIS="$("$PY" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
echo "visible GPUs: $VIS"
[ "$VIS" = "0" ] && { echo "FATAL: no GPUs visible; every task would score 0"; exit 2; }

common_args() {
    echo "--arena-root $HOME/third_party/AgentKernelArena --gpu-arch gfx950 \
${TYPES:+--types $TYPES} --out $OUT --arm $ARM --model $MODEL --limit $LIMIT"
}

# 'run' fans out one worker per GPU; the other modes are single-process.
#
# transformers loads with device_map="auto" and a 30B in bf16 is ~60GB against
# 192GB of HBM, so a single process puts the whole model on GPU 0 and leaves seven
# idle. Each worker therefore gets exactly one GPU via HIP_VISIBLE_DEVICES and its
# own disjoint slice of tasks. Full copies rather than a shared engine: the model
# fits with room to spare, and independent processes cannot stall each other.
if [ "$MODE" != "run" ] || [ "$VIS" -lt 2 ]; then
    # shellcheck disable=SC2086
    "$PY" scripts/run_agent_kernel_arena.py "$MODE" $(common_args)
    rc=$?
else
    echo "fanning out $VIS workers, one per GPU"
    pids=""
    for i in $(seq 0 $((VIS - 1))); do
        # shellcheck disable=SC2086
        HIP_VISIBLE_DEVICES="$i" CUDA_VISIBLE_DEVICES="$i" \
            "$PY" scripts/run_agent_kernel_arena.py run $(common_args) \
            --num-shards "$VIS" --shard "$i" \
            > "$OUT/shard${i}.log" 2>&1 &
        pids="$pids $!"
        echo "  shard $i -> GPU $i (pid $!)"
    done
    rc=0
    for p in $pids; do wait "$p" || rc=1; done
    echo "all shards exited (rc=$rc); merging ledgers"
    # Merge regardless: a shard that died still wrote every task it finished, and
    # the merge is what turns those ledgers into the completion artifact.
    # shellcheck disable=SC2086
    "$PY" scripts/run_agent_kernel_arena.py merge $(common_args) || rc=1
fi
echo "aka rc=$rc"
exit $rc
