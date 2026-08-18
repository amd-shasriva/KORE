#!/bin/bash
# Per-node entrypoint for scripts/spur_grpo_2node.sbatch. ONE copy runs on each
# node, started by a single `srun --ntasks-per-node=1` from the batch script.
#
# Everything here mirrors scripts/spur_grpo_1node.sbatch's environment, because
# the recipe's honesty depends on it (vendor baselines, adversarial correctness,
# the inert-feature audit). The additions are the multi-node rendezvous and the
# socket-interface pins.
#
# Usage: spur_grpo_2node_inner.sh <resolved-config.json>
set -uo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
RESOLVED="${1:?usage: spur_grpo_2node_inner.sh <resolved-config.json>}"
NODEID="${SLURM_NODEID:-0}"
NNODES="${SLURM_JOB_NUM_NODES:-${KORE_NUM_MACHINES:-2}}"

# Per-node log FILE. SPUR does not flush batch stdout until the job ends, so a
# 2-node run watched only through --output looks identical whether it is training
# or wedged at the rendezvous. Two nodes also interleave into one stream, which
# makes a per-rank failure hard to attribute.
LOGDIR="$REPO/runs/grpo2n-${SLURM_JOB_ID:-manual}"
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/node${NODEID}.log") 2>&1
echo "===== node $NODEID/$NNODES start $(date -Is) host=$(hostname) ====="

cd "$REPO" || exit 2
source /home/shasriva/kore-venv/bin/activate
source "$REPO/scripts/lib/ops_runtime.sh"
# The gateway credential lives in .env.local on a shared multi-user cluster, so
# refuse a symlink, a foreign owner, or group/other-writable modes before sourcing.
kore_secure_source_env "$REPO/.env.local"

# scripts/launch_distributed.sh execs BARE `accelerate`, so the venv must be on PATH.
export PATH="/home/shasriva/kore-venv/bin:$PATH"
# aiter (the production vendor baselines) is a source checkout at repos/aiter and is
# NOT pip-installed. Without it the vendor baseline silently falls back to torch,
# every measured speedup inflates, and the RL reward rewards the wrong thing.
export PYTHONPATH="$REPO:$REPO/repos/aiter"
export AITER_USE_SYSTEM_TRITON=1
export GPU_TARGET=gfx950
# ---- OFFLINE (air-gapped): never touch the hub ----
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/home/shasriva/.cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# Full-node GRES bind-mounts all eight render devices; drop SPUR's physical list so
# accelerate assigns one logical device per rank. grpo.py maps each rank's
# LOCAL_RANK back through CUDA_VISIBLE_DEVICES to the absolute id its bench
# subprocess needs. Fail closed BEFORE the masks are dropped, because once they are
# unset torch reports every device and a partial allocation becomes invisible.
VISIBLE_GPUS="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
if [[ "$VISIBLE_GPUS" != "8" ]]; then
    echo "[grpo-2node/node$NODEID] FATAL expected 8 GPUs, found $VISIBLE_GPUS" >&2
    exit 2
fi

# ---- interconnect ---- #
# No RDMA on these nodes: one Ethernet interface, which is why every run logs
# "NCCL WARN Could not find any local path from gpu N to net". Pin the socket
# transport rather than letting it probe for IB and fail.
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${KORE_NCCL_IFNAME:-ens3}"
# GLOO NEEDS ITS OWN VARIABLE, and this one is load-bearing rather than tidy.
# NCCL_SOCKET_IFNAME does not cover gloo, and gloo picks its address from the
# hostname - which on these nodes resolves to LOOPBACK. Measured (probe job 24399):
# ranks on the second node advertised 127.0.0.1 and every peer connection failed
# with "SO_ERROR: Connection refused, remote=[127.0.0.1]", killing the job in 27
# seconds. GRPO's per-step _all_gather_object rides gloo, so this pin is required
# for training, not just for rendezvous.
export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export TP_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
echo "[grpo-2node/node$NODEID] iface=$NCCL_SOCKET_IFNAME addr=$(ip -4 -o addr show "$NCCL_SOCKET_IFNAME" 2>/dev/null | awk '{print $4}') master=${KORE_MAIN_IP:-unset}:${KORE_MAIN_PORT:-unset}"

# ---- node-local compile/temp state ---- #
# Kernel compilation is the hot path and an NFS-backed TMPDIR contends across
# ranks. /tmp is not reliably writable on every node here, so prefer the node-local
# NVMe (28 TB, 24 TB free) and fall back only if it is missing.
LOCAL_ROOT="/mnt/m2m_nobackup/${USER}/kore/grpo2n-${SLURM_JOB_ID:-manual}-n${NODEID}"
if ! mkdir -p "$LOCAL_ROOT/tmp" 2>/dev/null; then
    LOCAL_ROOT="$REPO/runs/.scratch/grpo2n-${SLURM_JOB_ID:-manual}-n${NODEID}"
    mkdir -p "$LOCAL_ROOT/tmp" || { echo "[grpo-2node/node$NODEID] FATAL no writable scratch" >&2; exit 2; }
    echo "[grpo-2node/node$NODEID] scratch: shared home fallback ($LOCAL_ROOT)"
else
    echo "[grpo-2node/node$NODEID] scratch: node-local NVMe ($LOCAL_ROOT)"
fi
mkdir -p "$LOCAL_ROOT/triton" "$LOCAL_ROOT/inductor" "$LOCAL_ROOT/compile-cache"
export TMPDIR="$LOCAL_ROOT/tmp"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/inductor"
export KORE_COMPILE_CACHE_DIR="$LOCAL_ROOT/compile-cache"
trap 'rm -rf -- "$LOCAL_ROOT"' EXIT

export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- reward integrity (identical to the 1-node launcher) ---- #
# The adversarial correctness battery so no kernel lucky-passes the fixed seeds,
# and a torch.compile-fused baseline so a "speedup" is honest.
export KORE_VERIFIED_CORRECTNESS="${KORE_VERIFIED_CORRECTNESS:-1}"
export KORE_COMPILE_BASELINE="${KORE_COMPILE_BASELINE:-1}"
export KORE_USE_VENDOR_BASELINE="${KORE_USE_VENDOR_BASELINE:-1}"
# Stays 0 for RL: augmented shapes widen per-bench variance, and the GRPO advantage
# is computed from those speedups, so the noise degrades the learning signal.
export KORE_SHAPE_AUGMENT="${KORE_SHAPE_AUGMENT:-0}"
# Per-GPU exclusive lock around TIMED regions only, so ranks sharing a physical
# device cannot inflate each other's CV (compiles still overlap).
export KORE_TIMING_LOCK="${KORE_TIMING_LOCK:-1}"
# Fail closed on a requested-but-inert capability before 16 ranks load the weights.
export KORE_GRPO_INERT_FEATURES="${KORE_GRPO_INERT_FEATURES:-error}"
# report, NEVER strict: no measured peak profile exists for this workload.
export KORE_RESOURCE_PREFLIGHT=report
# See the long note in scripts/spur_grpo_1node.sbatch: the held-out lane's "code
# identity" digest is just the git HEAD, so any commit trips it - including the
# config-only commit that took the 1-node run down mid-campaign. The digest that
# actually covers shape-derivation code ('policy engine') is separate and stays
# enforced.
export KORE_CODE_IDENTITY="${KORE_CODE_IDENTITY:-736ba9327ad8c45fc9f3ee5ebe2a9e8dffbfe58e}"

# ---- launch ---- #
# num_processes is the GLOBAL rank count; machine_rank is this node's index.
export KORE_NUM_MACHINES="$NNODES"
export KORE_MACHINE_RANK="$NODEID"
: "${KORE_MAIN_IP:?KORE_MAIN_IP must be set by the batch script (literal IP)}"
export KORE_ACCEL_CONFIG="${KORE_ACCEL_CONFIG:-$REPO/configs/accelerate_fsdp_grpo_2node.yaml}"

echo "[grpo-2node/node$NODEID] launching $(( NNODES * VISIBLE_GPUS )) ranks"
bash "$REPO/scripts/launch_distributed.sh" grpo "$RESOLVED" \
    --nproc "$(( NNODES * VISIBLE_GPUS ))"
rc=$?
echo "[grpo-2node/node$NODEID] launch rc=$rc"
exit "$rc"
