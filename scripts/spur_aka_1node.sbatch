#!/bin/bash
# AgentKernelArena on a GPU node: discover | baseline | run.
#
# AKA ships a Docker runner and this cluster has no container runtime at all,
# so we drive the tasks' declared commands directly. They are plain argv, so the
# image was a reproducibility convenience rather than a dependency.
#
#   sbatch scripts/spur_aka_1node.sbatch <discover|baseline|run> [TYPES] [MODEL] [LIMIT] [ARM]
#SBATCH --job-name=kore-aka
# The account MUST match the QoS. amd-general paired with a burst or primus QoS
# is a phantom association: the controller accepts the submission and then never
# dispatches it, so the job simply pends forever with no error to read. This file
# carried amd-general/amd-general-qos, which is why nothing submitted through it
# ever landed. amd-primus/amd-primus-qos is the pair the v5 SFT job runs under.
#SBATCH --account=amd-primus
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-primus-qos
#SBATCH --nodes=1
# gpu:8, not gpu:mi355x:8 -- the typed form does not dispatch on this controller
# (JobLaunchFailure, 0 of 1 nodes confirmed). Measured 0/3 vs 2/3; see the full
# explanation in scripts/spur_sft_1node.sbatch.
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
# The partition's MaxTime is UNLIMITED and the nodes are held by reservation for
# days, so an 8h limit bought nothing and cost a great deal: the arena runs 12
# agentic rollouts concurrently per shard, and the wall clock cut those batches
# in half. Every restart threw away twelve nearly-finished rollouts per shard,
# reloaded a 30B model across 16 checkpoint shards, and depended on the
# supervisor noticing and resubmitting -- which is how v4 lost eight hours.
# 3-00:00:00 rather than 3-23:00:00: this is the limit amd-primus-qos is known to
# accept, because the v5 SFT job is running under it right now. The longer value
# was never tested under this QoS, and an unschedulable walltime fails the same
# silent way the wrong account did.
#SBATCH --time=3-00:00:00
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
# own vLLM engine, and keeps several tasks in flight so that engine batches: one
# task at a time made every generate call a batch of one, so all eight GPUs were
# busy and none was saturated.
# own disjoint slice of tasks. Full copies rather than a shared engine: the model
# fits with room to spare, and independent processes cannot stall each other.
# baseline fans out for the same reason run does: it must cover all 402 tasks to
# be a usable control, and single-process that does not fit in one allocation.
case "$MODE" in run|baseline) FANOUT=1 ;; *) FANOUT=0 ;; esac
if [ "$FANOUT" = "0" ] || [ "$VIS" -lt 2 ]; then
    # shellcheck disable=SC2086
    "$PY" scripts/run_agent_kernel_arena.py "$MODE" $(common_args)
    rc=$?
else
    # The allocation already exports a visibility mask naming our GPUs, and
    # kore.policy.serve rejects two masks outright rather than intersecting them
    # ("double masks are forbidden") -- so narrow the inherited mask to one entry
    # per worker instead of adding HIP_VISIBLE_DEVICES on top of ROCR_VISIBLE_DEVICES.
    # The ordinals come from the mask rather than from a 0..N-1 count, so each
    # worker lands on a GPU we were actually given.
    ALLOC="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-}}"
    [ -z "$ALLOC" ] && ALLOC="$(seq -s, 0 $((VIS - 1)))"
    IFS=',' read -r -a GPULIST <<< "$ALLOC"
    NSHARD="${#GPULIST[@]}"
    echo "fanning out $NSHARD workers over allocated GPUs: $ALLOC"
    pids=""
    for i in $(seq 0 $((NSHARD - 1))); do
        # shellcheck disable=SC2086
        env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
            ROCR_VISIBLE_DEVICES="${GPULIST[$i]}" \
            "$PY" scripts/run_agent_kernel_arena.py "$MODE" $(common_args) \
            --num-shards "$NSHARD" --shard "$i" \
            --task-concurrency "${AKA_TASK_CONCURRENCY:-4}" \
            > "$OUT/${MODE}_shard${i}.log" 2>&1 &
        pids="$pids $!"
        echo "  shard $i -> GPU ${GPULIST[$i]} (pid $!)"
    done
    rc=0
    for p in $pids; do wait "$p" || rc=1; done
    echo "all shards exited (rc=$rc); merging ledgers"
    # Merge regardless: a shard that died still wrote every task it finished, and
    # the merge is what turns those ledgers into the completion artifact.
    case "$MODE" in baseline) MERGE=baseline-merge ;; *) MERGE=merge ;; esac
    # shellcheck disable=SC2086
    "$PY" scripts/run_agent_kernel_arena.py "$MERGE" $(common_args) || rc=1
fi
echo "aka rc=$rc"
exit $rc
