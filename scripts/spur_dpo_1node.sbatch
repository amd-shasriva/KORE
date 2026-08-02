#!/bin/bash
# KORE 14B full-FT STAGE 2 (DPO) - single-node FSDP full-shard (ZeRO-3), 8x MI355X.
#
#   sbatch scripts/spur_dpo_1node.sbatch [CONFIG] [FROM_STAGE] [OUTPUT_DIR]
#
#   CONFIG      shipped stage config      default configs/dpo_14b_full.json
#   FROM_STAGE  checkpoint to train       default runs/sft_14b_frontier
#   OUTPUT_DIR  where this stage writes   default runs/dpo_14b_frontier
#
# CONFIG RESOLUTION CONTRACT: same as spur_sft_1node.sbatch - this launcher
# takes the SHIPPED config and does the model_id/output_dir override itself via
# scripts/spur_resolve_launch_config.py rather than expecting a hand-made
# *.resolved.json, so one submit command is the whole stage and a requeued child
# re-derives the same config. The resolved copy is kept at
# runs/dpo-$SLURM_JOB_ID.resolved.json. Pass '-' to keep a config value verbatim.
#
# DPO's reference policy is the FROZEN model it starts from: DPOConfig.ref_model_id
# defaults to None, which dpo.py resolves to model_id. Overriding model_id to the
# SFT output therefore also anchors the reference to the SFT policy, which is the
# intended recipe - nothing extra to pass.
#
# THIS LAUNCHER RUNS ONE NON-ITERATIVE DPO PASS. scripts/run_campaign.py defaults
# --dpo-rounds to 2 and would instead run the iterative on-policy DAgger loop
# (relabel from the current checkpoint -> aggregate -> retrain against a REFRESHED
# frozen reference). Those are different Stage-2 recipes; this one is the single
# pass over the pre-built preference corpus. Round N of the iterative recipe is
# just this launcher re-invoked with FROM_STAGE set to round N-1's output.
#
# SINGLE NODE, no multi-node variant: 96,675 preference pairs, but the corpus
# carries per-pair `weight` and kore.policy.dpo.apply_pref_weights consumes it by
# DETERMINISTIC MULTIPLICITY (weight>=1 -> round(weight) copies, 0<weight<1 -> kept
# with probability weight, weight<=0 -> dropped). Measured on the real corpus that
# is 96,675 -> 167,054 effective rows, a 1.728x multiplier, and it is ON by default
# (KORE_PREF_TRAIN_WEIGHTING=0 disables it). At an effective batch of 128 pairs/step
# (2 x 8 x 8 ranks) for 1 epoch that is 1,305 optimizer steps, NOT the 755 the
# unweighted count implies. Each step is heavier than SFT's (chosen+rejected
# concatenated, plus a no-grad reference forward). Measured 9.3 s/step at 2 ranks,
# seq ~2.2k, micro-batch 1 -> 1,559 tok/s/GPU; the corpus is 673.8M tokens/epoch
# (chosen+rejected, weighted), so 8 ranks project to roughly 13-19 h. That fits ONE
# 23 h allocation, but with as little as 4 h of margin at the pessimistic end, so
# the self-requeue below is load-bearing rather than a formality. midtrain's
# multi-node variants exist for a much larger token budget that DPO does not have.
#
# RESUME: dpo.py calls latest_checkpoint(output_dir) and passes it to
# trainer.train(resume_from_checkpoint=...); that helper walks candidates
# newest-first past a half-written checkpoint. Resumability therefore needs only
# a STABLE output_dir across requeues, which the positional args guarantee.
#SBATCH --job-name=kore-dpo
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
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/dpo-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/dpo-%j.err
set -euo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
# Positional, not --export: SPUR's sbatch does not reliably propagate arbitrary
# --export KEY=VALUE into a requeued child; positional args survive requeue.
CFG="${1:-$REPO/configs/dpo_14b_full.json}"
FROM_STAGE="${2:-runs/sft_14b_frontier}"
OUT_DIR="${3:-runs/dpo_14b_frontier}"

cd "$REPO"
mkdir -p runs
command -v scontrol >/dev/null || { echo "[dpo-1node] missing required command: scontrol" >&2; exit 2; }
command -v setsid >/dev/null || { echo "[dpo-1node] missing required command: setsid" >&2; exit 2; }

source /home/shasriva/kore-venv/bin/activate
# The gateway credential lives in .env.local on a shared multi-user cluster, so
# refuse a symlink, a foreign owner, or group/other-writable modes before sourcing.
source "$REPO/scripts/lib/ops_runtime.sh"
kore_secure_source_env "$REPO/.env.local"

# Fail closed BEFORE the physical masks are dropped: a SPUR partial-GPU allocation
# mounts an unusable device subset, and once ROCR_VISIBLE_DEVICES is unset torch
# reports every device on the node, so the shortfall becomes invisible.
VISIBLE_GPUS="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$VISIBLE_GPUS" != "8" ]]; then
    echo "[dpo-1node] FATAL expected 8 GPUs, found $VISIBLE_GPUS" >&2
    exit 2
fi

# Full-node GRES bind-mounts all eight render devices; drop SPUR's physical list
# so accelerate's multi_gpu_launcher can assign one logical device per rank.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

# scripts/launch_distributed.sh execs BARE `accelerate` with no interpreter path,
# so the venv must be on PATH.
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
    echo "[dpo-1node] compile scratch: node-local /tmp (${_avail}G free)"
else
    KORE_SCRATCH="$REPO/runs/.scratch/${SLURM_JOB_ID:-$$}"
    echo "[dpo-1node] compile scratch: shared home (node /tmp only ${_avail}G < ${KORE_SCRATCH_MIN_GB}G)"
fi
mkdir -p "$KORE_SCRATCH/tmp" "$KORE_SCRATCH/triton" "$KORE_SCRATCH/inductor"
export TMPDIR="$KORE_SCRATCH/tmp"
export TRITON_CACHE_DIR="$KORE_SCRATCH/triton"
export TORCHINDUCTOR_CACHE_DIR="$KORE_SCRATCH/inductor"
trap 'rm -rf "$KORE_SCRATCH"' EXIT

export FSDP_CPU_RAM_EFFICIENT_LOADING=1
# report, NEVER strict: no measured peak profile exists for this workload, so
# strict mode would refuse to start. Exported AFTER .env.local deliberately.
export KORE_RESOURCE_PREFLIGHT=report

echo "[dpo-1node] START job=${SLURM_JOB_ID:-none} host=$(hostname) gpus=$VISIBLE_GPUS"
echo "[dpo-1node] HEAD=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "[dpo-1node] cfg=$CFG from=$FROM_STAGE out=$OUT_DIR"

# SPUR's CLI talks to http://localhost:6817 unless SPUR_CONTROLLER_ADDR says
# otherwise, and spurctld does NOT run on every node (this cluster's login node
# has none). If the submitting environment did not export it, the timed
# self-requeue below cannot reach the controller - report that NOW rather than
# discovering it at hour 22 with the allocation about to expire.
CONTROLLER_PING="$(scontrol ping 2>&1 || true)"
if grep -q 'is UP' <<<"$CONTROLLER_PING"; then
    echo "[dpo-1node] controller reachable: $CONTROLLER_PING"
else
    echo "[dpo-1node] WARNING controller unreachable from this node, so the timed" \
         "self-requeue will fail and the job will simply end at its time limit" \
         "(a manual resubmit still resumes from the newest checkpoint). Fix:" \
         "submit with SPUR_CONTROLLER_ADDR exported, or set" \
         "KORE_SPUR_CONTROLLER_ADDR. Detail: $CONTROLLER_PING" >&2
fi

RESOLVED="$REPO/runs/dpo-${SLURM_JOB_ID:-manual}.resolved.json"
"$PY" scripts/spur_resolve_launch_config.py \
    --stage dpo --config "$CFG" --out "$RESOLVED" \
    --from-stage "$FROM_STAGE" --output-dir "$OUT_DIR"

# Report the resume point BEFORE the 8-rank load, so the log says up front
# whether this child continues a run or starts one.
"$PY" - "$RESOLVED" <<'PY'
import json, sys
from pathlib import Path
from kore.policy.configs import latest_checkpoint
out = json.loads(Path(sys.argv[1]).read_text())["output_dir"]
print(f"[dpo-1node] resume_from={latest_checkpoint(out)} output_dir={out}")
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
    echo "[dpo-1node] signal received; the newest complete checkpoint is safe; exiting for requeue"
    exit 143
}
# SPUR's scontrol exits 0 even when it cannot reach the controller, so the exit
# status alone cannot confirm the requeue landed; exiting 0 on an unlanded
# requeue would retire the job with training unfinished.
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
    echo "[dpo-1node] USR1 time-limit warning; draining and requesting requeue"
    terminate_child
    if requeue_self; then exit 0; fi
    echo "[dpo-1node] FATAL explicit requeue failed" >&2
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
# failure). Warn locally at 22h30m of the 23h limit so the run drains and
# EXPLICITLY requeues itself with 30 min to spare.
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
setsid bash scripts/launch_distributed.sh dpo "$RESOLVED" &
CHILD_PID="$!"
if wait "$CHILD_PID"; then
    echo "[dpo-1node] dpo completed"
else
    RC=$?
    echo "[dpo-1node] dpo failed rc=$RC"
fi

echo "DPO_RC=$RC"
echo "[dpo-1node] END job=${SLURM_JOB_ID:-none} status=$RC"
exit "$RC"
