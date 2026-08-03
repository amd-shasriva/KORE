#!/bin/bash
# Measured before/after for the pool-admission gate, on one node, in one job.
#
# The claim under test is that the campaign's 70% never-correct rate is caused by
# pool tasks whose oracle depends on weights the candidate kernel is never handed.
# The experiment is an A/B on MATCHED SAMPLES drawn from the same materialized
# pool, with identical code, identical teacher and identical turn budget:
#
#   BEFORE  tasks sampled from the pre-gate index (pool.jsonl.prefilter)
#   AFTER   tasks sampled from the post-gate index (pool.jsonl)
#
# Nothing is pruned: both arms resolve through the same complete `tasks/` tree, so
# the only variable is which tasks the plan contains. That is precisely the fix.
#SBATCH --job-name=kore-poolgate
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=/shared_nfs/shasriva/kore/runs/poolgate-%j.out
#SBATCH --error=/shared_nfs/shasriva/kore/runs/poolgate-%j.err

set -uo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
OUT_ROOT="${1:-/shared_nfs/shasriva/kore/runs/poolgate}"
TASKS_PER_ARM="${2:-60}"
ARM_SECONDS="${3:-2400}"
WORKERS="${4:-32}"

cd "$REPO"
source /home/shasriva/kore-venv/bin/activate
source "$REPO/scripts/lib/ops_runtime.sh"
kore_secure_source_env "$REPO/.env.local"

export PYTHONPATH=.:/home/shasriva/Kore-RL/KORE/repos/aiter
export AITER_USE_SYSTEM_TRITON=1
export KORE_VERIFIED_CORRECTNESS=1
export KORE_COMPILE_BASELINE=1
export KORE_BENCH_COLD=1
export KORE_SHAPE_AUGMENT=1
export KORE_TIMING_LOCK=1
export GPU_TARGET=gfx950
export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

LOCAL_ROOT="/mnt/m2m_nobackup/${USER}/kore/poolgate-${SLURM_JOB_ID}"
export TMPDIR="$LOCAL_ROOT/tmp"
export KORE_COMPILE_CACHE_DIR="$LOCAL_ROOT/compile-cache"
export KORE_RUNS_DIR="$LOCAL_ROOT/runs"
mkdir -p "$TMPDIR" "$KORE_COMPILE_CACHE_DIR" "$KORE_RUNS_DIR" "$OUT_ROOT"
trap 'rm -rf -- "$LOCAL_ROOT"' EXIT

echo "[poolgate] START job=$SLURM_JOB_ID host=$(hostname) head=$(git rev-parse --short HEAD)"
VISIBLE_GPUS="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
echo "[poolgate] gpus=$VISIBLE_GPUS tasks_per_arm=$TASKS_PER_ARM arm_seconds=$ARM_SECONDS"
[[ "$VISIBLE_GPUS" == "8" ]] || { echo "[poolgate] FATAL expected 8 GPUs" >&2; exit 2; }

# --- 0. the oracle really does depend on state the kernel never sees --------- #
echo "[poolgate] === STEP 0: is the pool oracle a function of its declared inputs? ==="
"$PY" scripts/audit_pool_answerability.py \
    --sample 120 --json-out "$OUT_ROOT/answerability.json" || exit 3

# --- 1. apply the gate to the index already on disk ------------------------- #
echo "[poolgate] === STEP 1: re-screen the shipped index ==="
if [[ -f data/task_pool/pool.jsonl.prefilter ]]; then
    echo "[poolgate] already re-screened; reusing existing split"
else
    "$PY" scripts/build_task_pool.py --rescreen-index \
        --workers 120 --json-out "$OUT_ROOT/rescreen.json" || exit 4
fi

# --- 2. matched task lists, one per arm ------------------------------------ #
echo "[poolgate] === STEP 2: build matched task lists ==="
"$PY" scripts/plan_pool_gate_ab.py \
    --out-dir "$OUT_ROOT" --tasks-per-arm "$TASKS_PER_ARM" || exit 5

# --- 3. run both arms through the unmodified datagen path ------------------- #
for ARM in before after; do
    echo "[poolgate] === STEP 3: arm=$ARM ==="
    mkdir -p "$OUT_ROOT/$ARM"
    "$PY" scripts/run_agentic_shard.py \
        --shard-file "$OUT_ROOT/shard_${ARM}.txt" \
        --out-dir "$OUT_ROOT/$ARM" \
        --shard-index 0 \
        --episodes-per-task 1 \
        --workers "$WORKERS" \
        --max-turns 8 \
        --gpu-ids 0,1,2,3,4,5,6,7 \
        --min-free-gb 20 \
        --seconds "$ARM_SECONDS"
    echo "[poolgate] arm=$ARM exit=$?"
done

# --- 4. the number ---------------------------------------------------------- #
echo "[poolgate] === STEP 4: before/after ==="
"$PY" scripts/report_pool_gate_ab.py --out-dir "$OUT_ROOT" \
    --json-out "$OUT_ROOT/ab_result.json"

echo "[poolgate] END job=$SLURM_JOB_ID"
