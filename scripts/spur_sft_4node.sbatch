#!/bin/bash
# KORE 30B MoE STAGE 1 (SFT) - multi-node FSDP full-shard (ZeRO-3), 4 nodes x 8 GPUs.
#
#   sbatch scripts/spur_sft_4node.sbatch [CONFIG] [FROM_STAGE] [OUTPUT_DIR]
#
# Same config-resolution contract as spur_sft_1node.sbatch: this launcher takes
# the SHIPPED config and resolves it itself via spur_resolve_launch_config.py,
# writing runs/sft-$SLURM_JOB_ID.resolved.json as the audit trail. Pass '-' for
# FROM_STAGE or OUTPUT_DIR to use the config's own value.
#
# WHY THIS EXISTS: the 1-node variant was written when the measured 14B run fit
# one 23h allocation. The v4 mixture is ~2.6B tokens against v3's 333M, which is
# roughly 26h on one node -- past the wall clock and with no margin for the
# checkpoint I/O. Four nodes brings it to ~7h.
#
# SPUR IS NOT STOCK SLURM, and two of its differences matter here:
#
#   1. `scontrol` has no `hostnames` subcommand. SLURM_JOB_NODELIST is already a
#      comma-separated list of REAL hostnames, so the rendezvous master is just
#      its first field. Do not try to expand it.
#   2. A requeued job trips JobHoldMaxRequeue on its FIRST requeue and is held
#      permanently, so `--requeue` turns any transient NODE_FAIL into an
#      unrecoverable hold. It is deliberately absent, exactly as in the midtrain
#      multi-node launchers. Recovery is a manual resubmit, which resumes from
#      the newest complete checkpoint.
#SBATCH --job-name=kore-sft4
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=mi355x:8
#SBATCH --cpus-per-task=128
#SBATCH --time=23:00:00
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/sft4-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/sft4-%j.err
set -euo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"

CFG="${1:-$REPO/configs/sft_coder30b_a3b.json}"
FROM_STAGE="${2:--}"
OUT_DIR="${3:--}"
[ "$CFG" = "-" ] && CFG="$REPO/configs/sft_coder30b_a3b.json"

NNODES="${SLURM_NNODES:?not in an sbatch alloc}"
NPROC_GLOBAL=$(( NNODES * 8 ))
MASTER_ADDR="$(printf '%s' "$SLURM_JOB_NODELIST" | cut -d, -f1)"
MASTER_PORT=29519

echo "[sft-4node] START job=${SLURM_JOB_ID:-none} nodes=$NNODES world=$NPROC_GLOBAL"
echo "[sft-4node] master=$MASTER_ADDR:$MASTER_PORT nodelist=$SLURM_JOB_NODELIST"

# Resolve ONCE on the batch host, before any rank loads 61GB of weights. Every
# node then reads the same file off the shared filesystem, so the eight ranks
# cannot disagree about the recipe -- resolving per-node would let a mid-flight
# config edit produce ranks training different things.
RESOLVED="$REPO/runs/sft4-${SLURM_JOB_ID:-manual}.resolved.json"
"$PY" scripts/spur_resolve_launch_config.py \
    --stage sft --config "$CFG" --out "$RESOLVED" \
    --from-stage "$FROM_STAGE" --output-dir "$OUT_DIR"
echo "[sft-4node] resolved -> $RESOLVED"

RC=0
srun --nodes="$NNODES" --ntasks="$NNODES" bash -c '
  set -euo pipefail
  cd "'"$REPO"'"
  source /home/shasriva/kore-venv/bin/activate
  set -a; source "'"$REPO"'/.env.local" 2>/dev/null || true; set +a

  # Full-node GRES: accelerate sets per-worker masks itself, so any inherited
  # visibility mask would leave 7 of 8 cards unused on every node.
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
  export HF_HOME=/home/shasriva/.cache/huggingface
  export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

  # RCCL over Mellanox IB. Pinning the fabric matters more at 32 ranks than at 8:
  # the all-gather for a 30B MoE crosses nodes every layer, and letting RCCL fall
  # back to sockets turns a compute-bound step into a network-bound one.
  export NCCL_SOCKET_IFNAME=^lo,docker0
  export NCCL_IB_HCA=mlx5
  export NCCL_NET_GDR_LEVEL=3
  export NCCL_DEBUG=WARN
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  # Load the checkpoint on rank 0 only and broadcast, instead of 32 ranks each
  # pulling 61GB off the same NFS mount.
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
    -m kore.policy.sft "'"$RESOLVED"'"
' || RC=$?

echo "SFT_RC=$RC"
echo "[sft-4node] END job=${SLURM_JOB_ID:-none} status=$RC"
exit "$RC"
