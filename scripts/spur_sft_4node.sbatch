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
# --ntasks is deliberately NOT set here. On this SPUR controller an sbatch with
# --ntasks=N runs the batch BODY on all N tasks, not just the batch host, and
# each copy then calls `srun --nodes=4` against an allocation it does not own --
# which fails with "job <id> not running on this node" for every node the copy
# was not placed on. The task fan-out belongs on the srun line below, where it
# is a request against this job's own allocation.
#SBATCH --nodes=4
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
# Derived from the job id, not fixed. A hard-coded port collides with any earlier
# attempt whose ranks have not fully torn down -- accelerate then refuses to start
# with "another process is utilizing it", which is what killed the first three
# smoke runs. The range avoids the ephemeral ports the kernel hands out.
MASTER_PORT=$(( 29000 + (${SLURM_JOB_ID:-0} % 900) ))

# SPUR runs this batch body once PER ALLOCATED NODE, even with --ntasks unset.
# Measured: a --nodes=4 job printed START four times, and each copy then raced to
# drive its own srun over the whole allocation. Exactly one copy may proceed.
#
# mkdir is the arbiter for the same reason it guards the resolve: atomic on NFS,
# unlike a test-then-write. The winner runs the job; the others exit 0 so they do
# not fail the array. Placed before any output so the log has one START.
DRIVER_LOCK="$REPO/runs/sft4-${SLURM_JOB_ID:-manual}.driver.lock"
if ! mkdir "$DRIVER_LOCK" 2>/dev/null; then
    echo "[sft-4node] not the driver copy on $(hostname); exiting cleanly"
    exit 0
fi
trap 'rmdir "$DRIVER_LOCK" 2>/dev/null || true' EXIT

echo "[sft-4node] START job=${SLURM_JOB_ID:-none} nodes=$NNODES world=$NPROC_GLOBAL"
echo "[sft-4node] master=$MASTER_ADDR:$MASTER_PORT nodelist=$SLURM_JOB_NODELIST"

# Resolve ONCE, before any rank loads 61GB of weights, so all 32 ranks read one
# file off the shared filesystem and cannot disagree about the recipe.
#
# "Once" needs enforcing: under --ntasks=4 SPUR runs this batch body on EVERY
# task, not just the batch host, so the naive version raced -- two tasks resolved
# concurrently, each writing <out>.partial.<pid> and renaming, and the loser died
# with FileNotFoundError when its partial had already been moved. mkdir is the
# atomic primitive here (open(O_CREAT|O_EXCL) semantics on every POSIX
# filesystem, NFS included, unlike test -e followed by touch). The winner
# resolves; the losers wait for the file rather than racing it.
RESOLVED="$REPO/runs/sft4-${SLURM_JOB_ID:-manual}.resolved.json"
LOCKDIR="$RESOLVED.lock"
if mkdir "$LOCKDIR" 2>/dev/null; then
    "$PY" scripts/spur_resolve_launch_config.py \
        --stage sft --config "$CFG" --out "$RESOLVED" \
        --from-stage "$FROM_STAGE" --output-dir "$OUT_DIR"
    echo "[sft-4node] resolved -> $RESOLVED"
else
    echo "[sft-4node] another task is resolving; waiting for $RESOLVED"
    for _ in $(seq 1 120); do
        [ -s "$RESOLVED" ] && break
        sleep 1
    done
    [ -s "$RESOLVED" ] || { echo "[sft-4node] FATAL resolve never appeared" >&2; exit 1; }
fi

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
