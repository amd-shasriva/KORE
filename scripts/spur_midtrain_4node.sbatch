#!/bin/bash
# KORE 14B full-FT midtrain — multi-node FSDP (ZeRO-3) across 4 nodes x 8 GPUs.
# SPUR (not stock Slurm): scontrol has NO 'hostnames'; use SLURM_JOB_NODELIST
# (already comma-separated REAL hostnames) + SLURM_NODEID (0..N-1) for rank.
#SBATCH --job-name=kore-midtrain
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=mi355x:8
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/midtrain-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/midtrain-%j.err
set -euo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
CFG="${1:-$REPO/data/b05factory/launch/midtrain_32gpu.json}"
NNODES="${SLURM_NNODES:?not in an sbatch alloc}"
NPROC_GLOBAL=$(( NNODES * 8 ))
# First host in the (uncompressed, comma-sep) SPUR nodelist is the rendezvous master.
MASTER_ADDR="$(printf '%s' "$SLURM_JOB_NODELIST" | cut -d, -f1)"
MASTER_PORT=29517
echo "[midtrain] nodes=$NNODES world=$NPROC_GLOBAL master=$MASTER_ADDR:$MASTER_PORT nodelist=$SLURM_JOB_NODELIST cfg=$CFG"

# One task per node; each spawns 8 local ranks via accelerate. SLURM_NODEID -> machine_rank.
srun --nodes="$NNODES" --ntasks="$NNODES" bash -c '
  set -euo pipefail
  cd "'"$REPO"'"
  source /home/shasriva/kore-venv/bin/activate
  set -a; source "'"$REPO"'/.env.local" 2>/dev/null || true; set +a
  # Full-node GRES: expose all 8 render devices to accelerate (it sets per-worker masks).
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  # ---- OFFLINE (air-gapped): never touch the hub ----
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
  export HF_HOME=/home/shasriva/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
  # ---- RCCL over Mellanox IB (mlx5_0). Pin the fabric, disable PCIe P2P fallback flakiness. ----
  export NCCL_SOCKET_IFNAME=^lo,docker0
  export NCCL_IB_HCA=mlx5
  export NCCL_NET_GDR_LEVEL=3
  export NCCL_DEBUG=WARN
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export FSDP_CPU_RAM_EFFICIENT_LOADING=1
  echo "[node $SLURM_NODEID/$SLURM_NNODES] host=$(hostname) -> machine_rank=$SLURM_NODEID"
  exec accelerate launch \
    --config_file "'"$REPO"'/configs/accelerate_fsdp.yaml" \
    --num_machines "'"$NNODES"'" \
    --num_processes "'"$NPROC_GLOBAL"'" \
    --machine_rank "$SLURM_NODEID" \
    --main_process_ip "'"$MASTER_ADDR"'" \
    --main_process_port "'"$MASTER_PORT"'" \
    --same_network \
    -m kore.policy.midtrain "'"$CFG"'"
'
