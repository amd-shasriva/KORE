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
# gpu:8, NOT gpu:mi355x:8. The typed form does not dispatch on this controller: it is
# accepted, the controller picks a node, and the node never confirms, leaving the job
# in JobLaunchFailure (dispatch confirmation failed: 0 of 1 nodes confirmed) forever.
# Measured head to head, three trials each, identical in every other respect:
#
#   --gres=gpu:mi355x:8   0 of 3 dispatched
#   --gres=gpu:8          2 of 3 dispatched
#   --gpus-per-node=8     0 of 3 dispatched
#
# The untyped form is also what other users' working jobs on this cluster request --
# a running burst job shows TresPerNode=gpu:8/node while ours showed
# gpu:mi355x:8/node. SPUR advertises each node's GPUs as eight separate
# `gpu:mi355x:1` entries rather than one `gpu:mi355x:8`, which is the likely reason a
# typed request for 8 cannot be satisfied. Every node in this partition is MI355X, so
# naming the model buys nothing anyway, and the script already verifies it received
# eight usable devices before training -- which is the check that actually matters.
#SBATCH --gres=gpu:8
# 32, not 128. This is the constraint that actually kept the run out of the cluster.
#
# Measured after job 7872 sat PENDING for 8h54m: FORTY-TWO jobs submitted after it
# were already running, and the discriminator is visible in their requests --
#
#   job 9008 zgong     gpu:mi355x:8  cpus=1    running
#   job 9071 zihaoan2  gpu:8         cpus=1    running
#   job 8858 oshkarav  gpu:8         cpus=1    running
#   job 8895 yiqian    gpu:8         cpus=32   running
#   job 8037 chuschen  (no gpu)      cpus=236  running
#
# Jobs wanting 1-32 CPUs are placed continuously; ours wanting 128 never was. An
# earlier revision of this file argued 128 was safe because three sampled "mix" nodes
# each had ~234 of 236 CPUs free. That sample was far too small: there are now 59 mix
# nodes, and jobs like chuschen's take all 236 CPUs, so free-CPU count is genuinely
# contended and a 128-CPU request excludes most of the cluster.
#
# 32 is ample for what this run actually does with CPUs. Step time is dominated by the
# MoE expert loop on the GPUs, and per-step host work is loading two pre-tokenised
# rows per rank and padding them. The only CPU-heavy phase is TRL's one-time dataset
# map, which takes ~35s at 32 procs and merely takes a few minutes at fewer -- paid
# once against a run measured in tens of hours.
#
# dataloader_num_workers and dataset_num_proc are lowered alongside it, so 8 rank
# processes plus their workers do not oversubscribe the smaller allocation.
#SBATCH --cpus-per-task=32
# 7 days, not 23 hours. The 23h figure was assumed, not measured against the
# controller: `scontrol show partition amd-spur` reports MaxTime=UNLIMITED, and a
# probe job submitted with --time=7-00:00:00 was accepted and ran. This matters
# because one epoch is 1,609 steps and the honest walltime estimate is 26-36h, so a
# 23h limit GUARANTEED at least one interruption for a run that can otherwise finish
# in a single allocation. Every interruption costs a full model reload plus a
# dataloader skip-ahead, and the resume path at 30B has never been exercised
# end-to-end, so the cheapest way to de-risk it is to not need it.
#
# This is also the closest thing available to the week-long node reservation this run
# would ideally have: `scontrol create reservation` requires root or sudo/wheel
# membership, so a real reservation has to come from the SPUR team via ticket.
#SBATCH --time=7-00:00:00
# NOTE: --requeue is deliberately ABSENT. On this SPUR controller a requeued
# job trips JobHoldMaxRequeue on its first requeue and is held permanently,
# so any transient NODE_FAIL became an unrecoverable hold. Recovery now lives
# in scripts/spur_pipeline_driver.sh, which resubmits and lets the trainer
# resume from its last complete checkpoint.
# --exclusive is deliberately ABSENT, and this is a correction of an earlier
# assumption rather than a relaxation of standards.
#
# The job still requests --gres=gpu:8, i.e. ALL EIGHT GPUs on the node. That is what
# makes it a full node for training: no other GPU job can be placed there, and no
# other process can touch a device we hold. --exclusive adds only one further demand,
# that not even a single unrelated CPU process coexists, and on this cluster that one
# extra demand is the difference between running and not running.
#
# Measured, on a job submitted at 01:29:30 that was still PENDING at 05:11 while NINE
# jobs submitted 2.0-3.6 hours LATER all started ahead of it:
#
#   job 7163 yquan     submitted 03:28  gpu:8/node  cpus=1  exclusive=no  -> RUNNING
#   job 7388 olehtika  submitted 03:46  gpu:8/node  cpus=1  exclusive=no  -> RUNNING
#   job 7509 yihanwan  submitted 04:03  gpu:8/node  cpus=1  exclusive=no  -> RUNNING
#   job 7727 anamital  submitted 05:05  gpu:8/node  cpus=1  exclusive=no  -> RUNNING
#   job 7750 jiaolyu   submitted 05:08  gpu:8/node  cpus=1  exclusive=no  -> RUNNING
#
# Every one of them asked for the full complement of GPUs, so this is not a case of
# small jobs slipping through gaps. This scheduler places what fits rather than
# strictly honouring submit order, and an --exclusive request only fits a COMPLETELY
# idle node, of which there were 0-2 at any moment while 18 nodes sat drained for a
# "Bundle 2 upgrade".
#
# The original justification for --exclusive was that a co-tenant might exhaust host
# memory during the ~61GB rank-0 FULL_STATE_DICT gather at each checkpoint. That was
# never measured, and it is wrong: these nodes have RealMemory=2817843 (2.75 TB) with
# ~2.7 TB free, and CPUTot=236 of which a typical "mix" node has ONE OR TWO
# allocated. Our peak host footprint is roughly 2% of the machine. The risk being
# avoided did not exist; the cost of avoiding it was the run not starting.
#
# What still protects the run: --cpus-per-task=128 is a guaranteed allocation (a mix
# node typically has ~234 free, so it fits comfortably), and the GPU-count guard
# below fails closed if fewer than eight usable devices are visible.
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

# Counting devices is not enough: they also have to be EMPTY. A previous tenant can
# leave memory allocated on a card after its job ends -- this project measured ~270GB
# still held on one -- and the only symptom is that training wedges after the model
# load, which the supervisor currently notices via a 45-minute stall timer. Now that
# re-entering the queue costs hours rather than minutes, spending 45 minutes to
# discover a dirty node is far too expensive.
#
# Checked here, before any weights are read, so a bad node costs seconds. Exit 3 is
# distinct from the count failure above so the supervisor can tell "this NODE is bad,
# exclude it and retry" apart from "the CODE is broken, stop retrying".
BAD_GPU="$("$PY" - <<'PY'
import torch
bad = []
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    # 90%: leaves room for driver/context overhead while still catching a leak
    # measured in tens of gigabytes, which is what actually wedges a run.
    if free < 0.90 * total:
        bad.append(f"gpu{i} {(total-free)/2**30:.1f}GiB already allocated "
                   f"({100*free/total:.0f}% free)")
print("; ".join(bad))
PY
)"
if [[ -n "$BAD_GPU" ]]; then
    echo "[sft-1node] BAD NODE $(hostname): $BAD_GPU" >&2
    echo "KORE_BAD_NODE=$(hostname)"
    echo "[sft-1node] a prior tenant left memory on this card; excluding and retrying" >&2
    exit 3
fi
echo "[sft-1node] all 8 GPUs present and empty"

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

# SINGLE-TRAINER LOCK.
#
# We keep a job queued in BOTH the amd-general-qos and amd-primus-qos pools, because
# both are full and whichever frees first should be taken -- general has 6 rotating
# slots and primus 14, so holding one place in each roughly doubles the chance of
# catching an early finish. The hazard that creates is two jobs starting and both
# training into the same output_dir, which would interleave checkpoint writes and
# destroy the only thing that makes preemption survivable.
#
# So: the first job to start claims a lock and trains; a later starter finds the lock
# held by a live job, prints KORE_LOCK_HELD=<jobid> and exits 0 without touching
# anything. mkdir is the primitive because it is atomic even on NFS -- test-then-write
# is not. A lock whose holder is no longer in the queue is stale (the holder was
# preempted or killed) and is taken over, which is what makes this safe across
# restarts rather than a one-shot.
LOCK_DIR=""
_train_lock() {
    local out lock holder state
    out="$("$PY" -c "
import json,sys;print(json.loads(open(sys.argv[1]).read())['output_dir'])" "$RESOLVED" 2>/dev/null)"
    [[ -z "$out" ]] && return 0     # cannot resolve: do not block the run
    lock="$out/.kore_train.lock"
    mkdir -p "$out" 2>/dev/null || true
    if mkdir "$lock" 2>/dev/null; then
        echo "${SLURM_JOB_ID:-manual}" > "$lock/jobid"
        LOCK_DIR="$lock"
        echo "[sft-1node] acquired the training lock"
        return 0
    fi
    holder="$(cat "$lock/jobid" 2>/dev/null || true)"
    if [[ -n "$holder" && "$holder" != "${SLURM_JOB_ID:-}" ]]; then
        state="$(squeue -j "$holder" -h -o '%T' 2>/dev/null | head -1 || true)"
        if [[ "$state" == "RUNNING" ]]; then
            echo "KORE_LOCK_HELD=$holder"
            echo "[sft-1node] job $holder is already training; exiting so we do not" \
                 "write into the same output_dir" >&2
            exit 0
        fi
    fi
    # Stale: the recorded holder is gone. Claim it.
    echo "${SLURM_JOB_ID:-manual}" > "$lock/jobid"
    LOCK_DIR="$lock"
    echo "[sft-1node] took over a stale training lock from '${holder:-unknown}'"
}
_release_lock() {
    # Only the holder removes it, so a loser exiting cannot free the winner's lock.
    if [[ -n "$LOCK_DIR" && -d "$LOCK_DIR" ]]; then
        [[ "$(cat "$LOCK_DIR/jobid" 2>/dev/null || true)" == "${SLURM_JOB_ID:-manual}" ]] \
            && rm -rf "$LOCK_DIR"
    fi
}
_train_lock

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
# Submit a FRESH job rather than requeueing this one, and prefer it over
# requeue_self at the time limit.
#
# The header above already refuses `#SBATCH --requeue` because a requeued job on
# this controller trips JobHoldMaxRequeue on its first requeue and is held
# permanently. `scontrol requeue` goes through the same path, so the old time-limit
# handler was reaching for the mechanism the header exists to avoid: at 22h30m it
# would have requeued itself and been held, stalling the run silently with every
# checkpoint intact and nothing running. A fresh submission cannot be held that way.
#
# Correctness rests on the job being reproducible from its positional arguments,
# which is already the design (see the note above CFG): output_dir is stable across
# submissions, and the trainer resumes from the newest complete checkpoint via
# latest_checkpoint(). A new job id means a new output file, which is the only
# observable difference.
#
# GENERATION is carried as a fourth positional argument and capped, so a run that
# somehow fails immediately on startup cannot resubmit itself forever.
GENERATION="${4:-0}"
KORE_MAX_GENERATIONS="${KORE_MAX_GENERATIONS:-12}"
resubmit_self() {
    local out rc next
    next=$((GENERATION + 1))
    if ((next > KORE_MAX_GENERATIONS)); then
        echo "[sft-1node] refusing to resubmit: generation $next exceeds cap " \
             "$KORE_MAX_GENERATIONS. If training is genuinely unfinished, raise " \
             "KORE_MAX_GENERATIONS; if it is looping, fix the startup failure." >&2
        return 1
    fi
    # Preserve the QoS the job is ACTUALLY running under, which is not necessarily
    # the one in the header. amd-general-qos is capped at 8 nodes team-wide and runs
    # deep queues, so a launch may legitimately be submitted with
    # `--qos=amd-burst-qos` to get a node now; without this, a preempted burst run
    # would resubmit itself into the general queue and sit behind everyone else,
    # turning a 45-minute interruption into a day.
    # `|| true` is load-bearing, not defensive noise. This script runs under
    # `set -euo pipefail`, and a command substitution whose pipeline exits non-zero
    # makes the ASSIGNMENT non-zero, which kills the script -- silently, with an
    # empty stderr. That is exactly how job 6520 died: it read TimeLimit with the
    # same idiom, SPUR's scontrol does not emit that field, grep found nothing and
    # returned 1. Any grep over optional output needs this.
    local qos_arg=()
    local qos
    qos="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
           | tr ' ' '\n' | grep -m1 '^QOS=' | cut -d= -f2 || true)"
    [[ -z "$qos" ]] && qos="${KORE_SFT_QOS:-}"
    [[ -n "$qos" ]] && qos_arg=(--qos="$qos")
    out="$(sbatch "${qos_arg[@]}" "$REPO/scripts/spur_sft_1node.sbatch" \
                  "$CFG" "$FROM_STAGE" "$OUT_DIR" "$next" 2>&1)"
    rc=$?
    if ((rc == 0)) && grep -qE 'Submitted batch job [0-9]+' <<<"$out"; then
        echo "[sft-1node] resubmitted as generation $next: $out"
        return 0
    fi
    printf '[sft-1node] resubmit failed: %s\n' "$out" >&2
    return 1
}
handle_requeue_warning() {
    trap - USR1 TERM INT
    echo "[sft-1node] USR1 time-limit warning; draining and resubmitting" \
         "(generation ${GENERATION})"
    terminate_child
    # Resubmit first. Requeue is kept only as a fallback for the case where the
    # controller accepts a requeue but refuses a new submission (a full queue),
    # since a held job is still recoverable by hand and a lost run is not.
    if resubmit_self; then exit 0; fi
    echo "[sft-1node] resubmit failed; falling back to requeue" >&2
    if requeue_self; then exit 0; fi
    echo "[sft-1node] FATAL neither resubmit nor requeue landed" >&2
    exit 143
}
cleanup() {
    if [[ -n "$REQUEUE_TIMER_PID" ]]; then kill "$REQUEUE_TIMER_PID" 2>/dev/null || true; fi
    # Release the training lock however we exit -- completion, preemption, or crash.
    # A lock left behind is not fatal (the next job sees its holder is gone from the
    # queue and takes it over) but releasing it makes the common case clean.
    _release_lock
}
trap cleanup EXIT
trap handle_termination TERM INT
trap handle_requeue_warning USR1

# SPUR's sbatch does not implement Slurm's --signal option, and Slurm does NOT
# auto-requeue a job that merely hits its time limit (only preemption / node
# failure), so the warning has to be generated locally.
#
# Derived from the job's ACTUAL TimeLimit rather than hardcoded. The old default was
# 81000s (22h30m), tuned to a 23h limit; with the limit now 7 days that timer would
# have drained a perfectly healthy run at 22h30m and resubmitted it for no reason --
# the exact interruption the longer walltime exists to avoid. Falls back to the old
# constant only if TimeLimit cannot be read.
#
# The margin is 60 minutes, not 30: a checkpoint write is ~488GB and takes 4-8
# minutes against NFS, and the drain has to finish one before the limit lands.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    _drain_after="${KORE_REQUEUE_AFTER_SECONDS:-}"
    _tl=""
    if [[ -z "$_drain_after" ]]; then
        # Try to read the real limit, but do NOT depend on it. SPUR's scontrol (a
        # reimplementation, Version=0.8.0) does not emit TimeLimit at all, and
        # squeue's %l is empty for a job that has left the queue, so every read here
        # is best-effort and every one needs `|| true` -- see the note in
        # resubmit_self for how the missing `|| true` killed job 6520.
        _tl="$(squeue -j "$SLURM_JOB_ID" -h -o '%l' 2>/dev/null | tr -d ' \r' || true)"
        if [[ -z "$_tl" || "$_tl" == "UNLIMITED" ]]; then
            _tl="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
                   | tr ' ' '\n' | grep -m1 '^TimeLimit=' | cut -d= -f2 || true)"
        fi
        # [DD-]HH:MM:SS
        if [[ "$_tl" =~ ^(([0-9]+)-)?([0-9]+):([0-9]+):([0-9]+)$ ]]; then
            _limit=$(( ${BASH_REMATCH[2]:-0} * 86400 + 10#${BASH_REMATCH[3]} * 3600 \
                       + 10#${BASH_REMATCH[4]} * 60 + 10#${BASH_REMATCH[5]} ))
            (( _limit > 3600 )) && _drain_after=$(( _limit - 3600 ))
        fi
    fi
    # Falls back to the HEADER's walltime minus an hour, not to the old 22h30m
    # constant. That constant was tuned for a 23h limit and would drain a healthy
    # 7-day run at 22h30m -- the precise interruption the long walltime exists to
    # prevent. Keep KORE_SFT_WALL_SECONDS in step with the #SBATCH --time above.
    _drain_after="${_drain_after:-$(( ${KORE_SFT_WALL_SECONDS:-604800} - 3600 ))}"
    echo "[sft-1node] will drain and resubmit after ${_drain_after}s" \
         "(read TimeLimit='${_tl:-unavailable}'; SPUR often does not report it)"
    (
        sleep "$_drain_after"
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
