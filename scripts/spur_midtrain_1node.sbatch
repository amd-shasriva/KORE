#!/bin/bash
#SBATCH --job-name=kore-midtrain
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi355x:8
#SBATCH --cpus-per-task=128
#SBATCH --time=23:00:00
#SBATCH --requeue
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
echo "[midtrain-1node] host=$(hostname) gpus=$VISIBLE_GPUS"
CFG="${1:-$REPO/data/b05factory/launch/midtrain_8gpu.json}"
echo "[midtrain-1node] cfg=$CFG"
RC=0
bash scripts/launch_distributed.sh midtrain "$CFG" || RC=$?
echo "MIDTRAIN_RC=$RC"
exit "$RC"
