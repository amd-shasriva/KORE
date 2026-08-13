#!/bin/bash
#SBATCH --job-name=kore-midtrain
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
# gpu:8, not gpu:mi355x:8 -- the typed form does not dispatch on this controller
# (JobLaunchFailure, 0 of 1 nodes confirmed). Measured 0/3 vs 2/3; see the full
# explanation in scripts/spur_sft_1node.sbatch.
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=23:00:00
# NOTE: --requeue is deliberately ABSENT. On this SPUR controller a requeued
# job trips JobHoldMaxRequeue on its first requeue and is held permanently,
# so any transient NODE_FAIL became an unrecoverable hold. Recovery now lives
# in scripts/spur_pipeline_driver.sh, which resubmits and lets the trainer
# resume from its last complete checkpoint.
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/midtrain-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/midtrain-%j.err
set -euo pipefail
REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
cd "$REPO"; mkdir -p runs
source /home/shasriva/kore-venv/bin/activate
set -a; source "$REPO/.env.local" 2>/dev/null || true; set +a

# Fail closed BEFORE the physical masks are dropped: SPUR partial-GPU allocations
# mount an unusable device subset, and once ROCR_VISIBLE_DEVICES is unset torch
# reports every device on the node, so the shortfall becomes invisible and the run
# would silently midtrain on fewer than eight MI355X devices.
VISIBLE_GPUS="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$VISIBLE_GPUS" != "8" ]]; then
    echo "[midtrain-1node] FATAL expected 8 GPUs, found $VISIBLE_GPUS" >&2
    exit 2
fi

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/home/shasriva/.cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Compile scratch. Triton and Inductor default to the compute node's local /tmp,
# which is small and shared: job 28059 died on all 8 ranks within seconds with
# "LLVM ERROR: IO failure on output stream: No space left on device" while /home
# had 1.1 TB free. Node-local is faster when it has room, so prefer it and fall
# back to shared home, and say which was chosen so a repeat is diagnosable.
KORE_SCRATCH_MIN_GB="${KORE_SCRATCH_MIN_GB:-60}"
_local_free_gb() { df -BG --output=avail "${1:-/tmp}" 2>/dev/null | tail -1 | tr -dc '0-9'; }
_avail="$(_local_free_gb /tmp)"; _avail="${_avail:-0}"
if (( _avail >= KORE_SCRATCH_MIN_GB )); then
    KORE_SCRATCH="/tmp/kore-${SLURM_JOB_ID:-$$}"
    echo "[midtrain-1node] compile scratch: node-local /tmp (${_avail}G free)"
else
    KORE_SCRATCH="$REPO/runs/.scratch/${SLURM_JOB_ID:-$$}"
    echo "[midtrain-1node] compile scratch: shared home (node /tmp only ${_avail}G < ${KORE_SCRATCH_MIN_GB}G)"
fi
mkdir -p "$KORE_SCRATCH"/{tmp,triton,inductor}
export TMPDIR="$KORE_SCRATCH/tmp"
export TRITON_CACHE_DIR="$KORE_SCRATCH/triton"
export TORCHINDUCTOR_CACHE_DIR="$KORE_SCRATCH/inductor"
trap 'rm -rf "$KORE_SCRATCH"' EXIT

echo "[midtrain-1node] host=$(hostname) gpus=$VISIBLE_GPUS"
CFG="${1:-$REPO/data/b05factory/launch/midtrain_8gpu.json}"
echo "[midtrain-1node] cfg=$CFG"
RC=0
bash scripts/launch_distributed.sh midtrain "$CFG" || RC=$?
echo "MIDTRAIN_RC=$RC"
exit "$RC"
