#!/bin/bash
#SBATCH --job-name=kore-midtrain
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-burst-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi355x:8
#SBATCH --cpus-per-task=128
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/midtrain-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/midtrain-%j.err
set -uo pipefail
REPO=/home/shasriva/Kore-RL/KORE
cd "$REPO"; mkdir -p runs
source /home/shasriva/kore-venv/bin/activate
[ -f "$REPO/.env.local" ] && { set -a; source "$REPO/.env.local"; set +a; }
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/home/shasriva/.cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
echo "[midtrain-1node] host=$(hostname) gpus=$($PY -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null)"
CFG="${1:-$REPO/data/b05factory/launch/midtrain_8gpu.json}"
echo "[midtrain-1node] cfg=$CFG"
bash scripts/launch_distributed.sh midtrain "$CFG"
echo "MIDTRAIN_RC=$?"
