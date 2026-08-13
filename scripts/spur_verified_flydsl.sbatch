#!/bin/bash
# Verified FlyDSL seeding on one full node. Generation talks to the gateway and
# verification compiles and runs each candidate, so this needs real GPUs.
#
# Every task is ledgered as it finishes, so losing the allocation costs only the
# tasks in flight; a resubmission skips what already passed.
#SBATCH --job-name=kore-flyseed
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
# gpu:8, not gpu:mi355x:8 -- the typed form does not dispatch on this controller
# (JobLaunchFailure, 0 of 1 nodes confirmed). Measured 0/3 vs 2/3; see the full
# explanation in scripts/spur_sft_1node.sbatch.
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/vflydsl-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/vflydsl-%j.err

set -euo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
TASK_LIST="${1:?arg1 must be the task list}"
OUT_ROOT="${2:-$REPO/data/registry_flydsl_frontier}"
ATTEMPTS="${3:-3}"
WORKERS="${4:-8}"

cd "$REPO"
mkdir -p runs
source /home/shasriva/kore-venv/bin/activate
source "$REPO/scripts/lib/ops_runtime.sh"
kore_secure_source_env "$REPO/.env.local"

# Same GPU discipline as the datagen element: fail closed on a partial
# allocation, then drop SPUR's physical mask so each verifier child can apply
# its own logical one without hipErrorNoDevice.
VISIBLE_GPUS="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$VISIBLE_GPUS" != "8" ]]; then
    echo "[flyseed] FATAL expected 8 GPUs, found $VISIBLE_GPUS" >&2
    exit 2
fi
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

export PYTHONPATH=.:/home/shasriva/Kore-RL/KORE/repos/aiter
export AITER_USE_SYSTEM_TRITON=1
export GPU_TARGET=gfx950
export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MAX_JOBS=2

# Extended thinking must stay off here. With it on, opus-5 spent the whole
# output budget reasoning and returned zero characters on all 96 calls of the
# previous run; off, the same prompts answer in 20-40s. The seeder sets this
# itself as well, so a stale environment cannot resurrect the empty run.
export KORE_TEACHER_THINKING=off
export KORE_TEACHER_MAX_TOKENS=16384

LOCAL_ROOT="/mnt/m2m_nobackup/${USER}/kore/${SLURM_JOB_ID}"
export TMPDIR="$LOCAL_ROOT/tmp"
export KORE_COMPILE_CACHE_DIR="$LOCAL_ROOT/compile-cache"
mkdir -p "$TMPDIR" "$KORE_COMPILE_CACHE_DIR"
trap 'rm -rf -- "$LOCAL_ROOT"' EXIT

echo "[flyseed] START job=$SLURM_JOB_ID host=$(hostname) list=$TASK_LIST attempts=$ATTEMPTS"
"$PY" scripts/verified_flydsl_seed.py \
    --task-list "$TASK_LIST" \
    --out "$OUT_ROOT" \
    --source-root kore/tasks \
    --attempts "$ATTEMPTS" \
    --workers "$WORKERS"
echo "[flyseed] END job=$SLURM_JOB_ID status=$?"
