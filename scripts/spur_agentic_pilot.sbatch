#!/bin/bash
# One node, one measurement: how many agentic episodes per hour does a node do,
# and at what worker count. Nothing is written into the dataset - the shards this
# produces live under runs/pilot_agentic_<jobid>/ and exist to be measured.
#
# Runs a worker sweep against disjoint task slices, so no sweep point inherits a
# warm replay cache from the point before it. KORE_RUNS_DIR points at node-local
# scratch for the same reason: a task benched by an earlier campaign would replay
# instead of running and the measurement would be a lie.
#SBATCH --job-name=kore-agentic-pilot
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/agentic-pilot-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/agentic-pilot-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
source /home/shasriva/kore-venv/bin/activate

WORKERS="${1:-8,16,32,64}"
ROUNDS="${2:-2}"
MAX_TURNS="${3:-8}"

OUT="$REPO/runs/pilot_agentic_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"

source "$REPO/scripts/lib/ops_runtime.sh"
kore_secure_source_env "$REPO/.env.local"

# aiter supplies the production vendor baselines the speedups are measured
# against; without it the whole point of the measurement is gone.
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

LOCAL_ROOT="/mnt/m2m_nobackup/${USER}/kore/pilot-${SLURM_JOB_ID:-manual}"
export TMPDIR="$LOCAL_ROOT/tmp"
export KORE_COMPILE_CACHE_DIR="$LOCAL_ROOT/compile-cache"
export KORE_RUNS_DIR="$LOCAL_ROOT/runs"
mkdir -p "$TMPDIR" "$KORE_COMPILE_CACHE_DIR" "$KORE_RUNS_DIR"
trap 'rm -rf -- "$LOCAL_ROOT"' EXIT

VISIBLE_GPUS="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
echo "[pilot] host=$(hostname) gpus=$VISIBLE_GPUS head=$(git rev-parse --short HEAD)"
if [[ "$VISIBLE_GPUS" != "8" ]]; then
    echo "[pilot] FATAL expected 8 GPUs, found $VISIBLE_GPUS" >&2
    exit 2
fi

"$PY" scripts/pilot_agentic_throughput.py \
    --workers "$WORKERS" \
    --rounds "$ROUNDS" \
    --max-turns "$MAX_TURNS" \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    --out-dir "$OUT" \
    --report "$OUT/report.json" \
    --point-timeout-seconds 2700
STATUS=$?

echo "[pilot] END status=$STATUS report=$OUT/report.json"
exit "$STATUS"
