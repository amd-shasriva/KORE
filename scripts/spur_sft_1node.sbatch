#!/bin/bash
# KORE 14B full-FT STAGE 1 (SFT) - single-node FSDP full-shard (ZeRO-3), 8x MI355X.
#
#   sbatch scripts/spur_sft_1node.sbatch [CONFIG] [FROM_STAGE] [OUTPUT_DIR]
#
#   CONFIG      shipped stage config      default configs/sft_14b_full.json
#   FROM_STAGE  checkpoint to train       default runs/midtrain_14b_frontier
#   OUTPUT_DIR  where this stage writes   default runs/sft_14b_frontier
#
# CONFIG RESOLUTION CONTRACT (identical for spur_sft/spur_dpo/spur_grpo):
# this launcher takes the SHIPPED config and does the override ITSELF via
# scripts/spur_resolve_launch_config.py; it does NOT expect a hand-made
# *.resolved.json. The shipped configs name the RAW base (Qwen/Qwen3-14B)
# because they double as the reference recipe, and the real pipeline trains the
# previous stage's output. Resolving inside the job means (a) one paste-able
# submit command per stage, (b) a requeued child re-derives byte-identically
# instead of depending on a file a human remembered to regenerate, and (c) the
# handoff is CHECKED (the previous stage must have consolidated weights) before
# eight ranks load 55 GiB each. The resolved copy is kept at
# runs/sft-$SLURM_JOB_ID.resolved.json as the audit trail. Pass '-' for
# FROM_STAGE or OUTPUT_DIR to use the config's own value verbatim.
#
# A stale `model_revision` is left in the config on purpose: model_spec ignores
# a Hub revision for a local directory (and logs that it did), so it stays as
# the record of which base this lineage started from.
#
# SINGLE NODE, no multi-node variant: measured at 1,677 optimizer steps and
# 6.5-8.5 h at max_seq_length 17408 on 8 ranks (docs/SFT_READINESS.md), which
# fits one 23 h allocation with room for the checkpoint I/O. midtrain needed
# 3/4/8-node variants because it trains 3 epochs over a far larger token
# budget; SFT does not, and a single node is the topology that is actually
# verified end to end.
#
# RESUME: sft.py calls latest_checkpoint(output_dir) itself and passes it to
# trainer.train(resume_from_checkpoint=...), and that helper walks candidates
# newest-first past a half-written checkpoint. So resumability needs only a
# STABLE output_dir across requeues, which the positional args guarantee.
#SBATCH --job-name=kore-sft
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi355x:8
#SBATCH --cpus-per-task=128
#SBATCH --time=23:00:00
# NOTE: --requeue is deliberately ABSENT. On this SPUR controller a requeued
# job trips JobHoldMaxRequeue on its first requeue and is held permanently,
# so any transient NODE_FAIL became an unrecoverable hold. Recovery now lives
# in scripts/spur_pipeline_driver.sh, which resubmits and lets the trainer
# resume from its last complete checkpoint.
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/sft-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/sft-%j.err
set -euo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
# Positional, not --export: SPUR's sbatch does not reliably propagate arbitrary
# --export KEY=VALUE into a requeued child, but positional args are part of the
# job specification and survive requeue.
CFG="${1:-$REPO/configs/sft_14b_full.json}"
FROM_STAGE="${2:-runs/midtrain_14b_frontier}"
OUT_DIR="${3:-runs/sft_14b_frontier}"

cd "$REPO"
mkdir -p runs
command -v scontrol >/dev/null || { echo "[sft-1node] missing required command: scontrol" >&2; exit 2; }
command -v setsid >/dev/null || { echo "[sft-1node] missing required command: setsid" >&2; exit 2; }

source /home/shasriva/kore-venv/bin/activate
# The gateway credential lives in .env.local on a shared multi-user cluster, so
# refuse a symlink, a foreign owner, or group/other-writable modes before sourcing.
source "$REPO/scripts/lib/ops_runtime.sh"
kore_secure_source_env "$REPO/.env.local"

# Fail closed BEFORE the physical masks are dropped: SPUR partial-GPU allocations
# mount an unusable device subset, and once ROCR_VISIBLE_DEVICES is unset torch
# reports every device on the node, so the shortfall becomes invisible and the
# run would silently train a 14B on fewer than eight MI355X devices.
VISIBLE_GPUS="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$VISIBLE_GPUS" != "8" ]]; then
    echo "[sft-1node] FATAL expected 8 GPUs, found $VISIBLE_GPUS" >&2
    exit 2
fi

# Full-node GRES bind-mounts all eight render devices; drop SPUR's physical list
# so accelerate's multi_gpu_launcher can assign one logical device per rank.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

# scripts/launch_distributed.sh execs BARE `accelerate` with no interpreter path,
# so the venv must be on PATH (activate does this; the explicit export makes it
# independent of activate's behaviour under a non-login shell).
export PATH="/home/shasriva/kore-venv/bin:$PATH"
export PYTHONPATH="$REPO"
# ---- OFFLINE (air-gapped): never touch the hub ----
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/home/shasriva/.cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Intra-node RCCL only (one node, no IB fabric to pin).
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Compile scratch. Triton and Inductor default to the compute node's local /tmp,
# which is small and shared: mid-train job 28059 died on all 8 ranks within
# seconds with "LLVM ERROR: IO failure on output stream: No space left on
# device" while /home had 1.1 TB free. Node-local is faster when it has room, so
# prefer it and fall back to shared home, logging which so a repeat is
# diagnosable.
KORE_SCRATCH_MIN_GB="${KORE_SCRATCH_MIN_GB:-60}"
_avail="$(df -BG --output=avail /tmp 2>/dev/null | tail -1 | tr -dc '0-9')"; _avail="${_avail:-0}"
if [ "$_avail" -ge "$KORE_SCRATCH_MIN_GB" ]; then
    KORE_SCRATCH="/tmp/kore-${SLURM_JOB_ID:-$$}"
    echo "[sft-1node] compile scratch: node-local /tmp (${_avail}G free)"
else
    KORE_SCRATCH="$REPO/runs/.scratch/${SLURM_JOB_ID:-$$}"
    echo "[sft-1node] compile scratch: shared home (node /tmp only ${_avail}G < ${KORE_SCRATCH_MIN_GB}G)"
fi
mkdir -p "$KORE_SCRATCH/tmp" "$KORE_SCRATCH/triton" "$KORE_SCRATCH/inductor"
export TMPDIR="$KORE_SCRATCH/tmp"
export TRITON_CACHE_DIR="$KORE_SCRATCH/triton"
export TORCHINDUCTOR_CACHE_DIR="$KORE_SCRATCH/inductor"
trap 'rm -rf "$KORE_SCRATCH"' EXIT

export FSDP_CPU_RAM_EFFICIENT_LOADING=1
# report, NEVER strict: there is no measured peak profile for this workload, and
# preflight cannot join DRM cards to HIP ordinals without KORE_HIP_INVENTORY_JSON,
# so strict mode would refuse to start a run that is otherwise fine. Exported
# AFTER .env.local so the launch decides this, not the shared env file.
export KORE_RESOURCE_PREFLIGHT=report

echo "[sft-1node] START job=${SLURM_JOB_ID:-none} host=$(hostname) gpus=$VISIBLE_GPUS"
echo "[sft-1node] HEAD=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "[sft-1node] cfg=$CFG from=$FROM_STAGE out=$OUT_DIR"

# SPUR's CLI talks to http://localhost:6817 unless SPUR_CONTROLLER_ADDR says
# otherwise, and spurctld does NOT run on every node (this cluster's login node
# has none). If the submitting environment did not export it, the timed
# self-requeue below cannot reach the controller - so report that NOW instead of
# discovering it at hour 22 with the allocation about to expire.
CONTROLLER_PING="$(scontrol ping 2>&1 || true)"
if grep -q 'is UP' <<<"$CONTROLLER_PING"; then
    echo "[sft-1node] controller reachable: $CONTROLLER_PING"
else
    echo "[sft-1node] WARNING controller unreachable from this node, so the timed" \
         "self-requeue will fail and the job will simply end at its time limit" \
         "(a manual resubmit still resumes from the newest checkpoint). Fix:" \
         "submit with SPUR_CONTROLLER_ADDR exported, or set" \
         "KORE_SPUR_CONTROLLER_ADDR. Detail: $CONTROLLER_PING" >&2
fi

RESOLVED="$REPO/runs/sft-${SLURM_JOB_ID:-manual}.resolved.json"
"$PY" scripts/spur_resolve_launch_config.py \
    --stage sft --config "$CFG" --out "$RESOLVED" \
    --from-stage "$FROM_STAGE" --output-dir "$OUT_DIR"

# Report the resume point BEFORE the 8-rank load, so the log says up front
# whether this child continues a run or starts one.
"$PY" - "$RESOLVED" <<'PY'
import json, sys
from pathlib import Path
from kore.policy.configs import latest_checkpoint
out = json.loads(Path(sys.argv[1]).read_text())["output_dir"]
print(f"[sft-1node] resume_from={latest_checkpoint(out)} output_dir={out}")
PY

CHILD_PID=""
REQUEUE_TIMER_PID=""

terminate_child() {
    if [[ -n "$CHILD_PID" ]]; then
        kill -TERM -- "-$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
}
handle_termination() {
    trap - USR1 TERM INT
    terminate_child
    echo "[sft-1node] signal received; the newest complete checkpoint is safe; exiting for requeue"
    exit 143
}
# SPUR's scontrol exits 0 even when it cannot reach the controller (verified:
# `scontrol ping` prints "failed to connect to spurctld" and still returns 0), so
# the exit status alone cannot confirm the requeue landed. Exiting 0 on an
# unlanded requeue would retire the job with training unfinished.
_requeue_attempt() {
    local out rc
    out="$(scontrol requeue "$SLURM_JOB_ID" 2>&1)"
    rc=$?
    if ((rc == 0)) && ! grep -qiE 'error|failed|unavailable|no leader|refused' <<<"$out"; then
        return 0
    fi
    printf '%s\n' "$out" >&2
    return 1
}
requeue_self() {
    _requeue_attempt && return 0
    # Retry against an explicit controller: the CLI's localhost default is wrong
    # on any node that does not itself run spurctld. Subshell so the override
    # cannot leak into the rest of the job.
    if [[ -n "${KORE_SPUR_CONTROLLER_ADDR:-}" ]]; then
        ( export SPUR_CONTROLLER_ADDR="$KORE_SPUR_CONTROLLER_ADDR"; _requeue_attempt ) \
            && return 0
    fi
    return 1
}
handle_requeue_warning() {
    trap - USR1 TERM INT
    echo "[sft-1node] USR1 time-limit warning; draining and requesting requeue"
    terminate_child
    if requeue_self; then exit 0; fi
    echo "[sft-1node] FATAL explicit requeue failed" >&2
    exit 143
}
cleanup() {
    if [[ -n "$REQUEUE_TIMER_PID" ]]; then kill "$REQUEUE_TIMER_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap handle_termination TERM INT
trap handle_requeue_warning USR1

# SPUR's sbatch does not implement Slurm's --signal option, and Slurm does NOT
# auto-requeue a job that merely hits its time limit (only preemption / node
# failure). Generate the warning locally at 22h30m of the 23h limit so the run
# drains and EXPLICITLY requeues itself with 30 min to spare.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    (
        sleep "${KORE_REQUEUE_AFTER_SECONDS:-81000}"
        kill -USR1 "$$"
    ) &
    REQUEUE_TIMER_PID="$!"
fi

# setsid puts accelerate + its 8 workers in their own process group, so the
# drain above can signal the whole tree with one kill.
RC=0
setsid bash scripts/launch_distributed.sh sft "$RESOLVED" &
CHILD_PID="$!"
if wait "$CHILD_PID"; then
    echo "[sft-1node] sft completed"
else
    RC=$?
    echo "[sft-1node] sft failed rc=$RC"
fi

echo "SFT_RC=$RC"
echo "[sft-1node] END job=${SLURM_JOB_ID:-none} status=$RC"
exit "$RC"
