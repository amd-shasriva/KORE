#!/bin/bash
# KORE 14B full-FT STAGE 3 (multi-turn GRPO) - single-node FSDP SHARD_GRAD_OP
# (ZeRO-2), 8x MI355X. Online RL: the loop rolls out against the verified
# KoreEnv, so this job both TRAINS and COMPILES/BENCHES kernels on the same node.
#
#   sbatch scripts/spur_grpo_1node.sbatch [CONFIG] [FROM_STAGE] [OUTPUT_DIR] [DATA_ROOT]
#
#   CONFIG      shipped stage config      default configs/grpo_14b_full.json
#   FROM_STAGE  checkpoint to train       default runs/dpo_14b_frontier
#   OUTPUT_DIR  where this stage writes   default runs/grpo_14b_frontier
#   DATA_ROOT   co-evolution archive root default data/b05factory
#
# CONFIG RESOLUTION CONTRACT: same as spur_sft/spur_dpo_1node.sbatch - the
# launcher takes the SHIPPED config and does the override itself via
# scripts/spur_resolve_launch_config.py; it does NOT expect a hand-made
# *.resolved.json. The resolved copy is kept at
# runs/grpo-$SLURM_JOB_ID.resolved.json. Pass '-' to keep a config value verbatim.
#
# DATA_ROOT matters because kore.policy.grpo._build_opus_scores derives the
# co-evolution archive root from dirname(coevolve_distill_path), so that
# directory decides which corpus the regret-vs-Opus curriculum mines.
#
# The shipped config used to say data/full14b/coevolve_wins.jsonl and this
# rewrite was the only thing preventing a run from mining the wrong archive
# (measured: data/full14b yields 51 Opus-scored tasks, data/b05factory yields
# 1157). The config is now correct at rest, so on a default launch the resolver
# reports "no overrides" for both co-evolution keys -- which is the point: the
# recipe and the launcher agree, and you can see that they agree.
#
# The rewrite STAYS rather than being deleted as redundant, for two reasons.
# It is a real capability, not a workaround: DATA_ROOT is a positional argument,
# so an operator can point one job at data/b05factory_synced (557 scored tasks)
# or any other archive without editing a tracked config. And it keeps the two
# co-evolution paths re-rooted TOGETHER, which matters because build_opus_scores
# treats a present, non-empty opus_scores.json cache as authoritative and skips
# the archive scan entirely -- moving one path without the other would let a
# cache from the old root silently win. tests/test_grpo_recipe_honesty.py pins
# this default equal to the config's own root so the two cannot drift apart
# again in silence.
#
# WHY ONE NODE (and no multi-node variant): GRPO does NOT need a separate
# generation replica job. Its rollout replica is IN-PROCESS and PER-RANK -
# grpo.py loads a plain full-weight model on each rank and syncs the policy into
# it every step, so generation is collective-free and local (that is exactly why
# the stage uses SHARD_GRAD_OP: params stay replicated between forwards). Each
# rank also pins its compile/bench subprocess to its OWN physical GPU via
# LOCAL_RANK, so 8 ranks == 8 bench GPUs on one node with no contention. Adding
# nodes would put the per-step _all_gather_object and the weight sync on the IB
# fabric without reducing the dominant per-rank memory (a 14B stays replicated
# under ZeRO-2), so it is not clearly a win and it is not verified.
#
# EXPECT MULTIPLE JOBS: total_steps is 2000 with per-step rollouts and verified
# benching, which no 23 h allocation will finish. The self-requeue below drains
# and resubmits; grpo.py's own _discover_grpo_resume picks the newest COMPLETE
# checkpoint, broadcasts it so every rank agrees, and FAILS CLOSED if checkpoint
# dirs exist but none is resumable - it will never silently restart at step 0.
#SBATCH --job-name=kore-grpo
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
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/grpo-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/grpo-%j.err
set -euo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
# Positional, not --export: SPUR's sbatch does not reliably propagate arbitrary
# --export KEY=VALUE into a requeued child; positional args survive requeue.
CFG="${1:-$REPO/configs/grpo_14b_full.json}"
FROM_STAGE="${2:-runs/dpo_14b_frontier}"
OUT_DIR="${3:-runs/grpo_14b_frontier}"
DATA_ROOT="${4:-data/b05factory}"

# BORROWED ALLOCATION. This script normally owns its Slurm job, so it is free to
# requeue itself and to rewrite its own ExcNodeList. Both are wrong when it runs
# as a step inside SOMEONE ELSE'S allocation (srun --overlap --jobid=THEIRS),
# where SLURM_JOB_ID names a job we must never touch: the dirty-GPU guard below
# would requeue the lender's job, and the drain timer would tear it down at the
# 22h30m fallback. KORE_BORROWED_ALLOCATION=1 runs the training and nothing else
# - no scontrol writes, no drain, no self-requeue.
BORROWED="${KORE_BORROWED_ALLOCATION:-0}"
if [[ "$BORROWED" == 1 ]]; then
    # Artifacts land under the lender's uid, and this tree's owner has to be able
    # to resume from the checkpoints they produce, so keep the group write bit.
    umask 002
    echo "[grpo-1node] BORROWED allocation ${SLURM_JOB_ID:-none}:" \
         "will not requeue, drain, or otherwise modify this job"
fi

cd "$REPO"
mkdir -p runs
command -v scontrol >/dev/null || { echo "[grpo-1node] missing required command: scontrol" >&2; exit 2; }
command -v setsid >/dev/null || { echo "[grpo-1node] missing required command: setsid" >&2; exit 2; }

source /home/shasriva/kore-venv/bin/activate
# The gateway credential lives in .env.local on a shared multi-user cluster, so
# refuse a symlink, a foreign owner, or group/other-writable modes before sourcing.
source "$REPO/scripts/lib/ops_runtime.sh"
# A borrowed allocation reads this tree as a foreign uid, so the ownership check
# fails by design (and mode 600 makes the file unreadable anyway). Skip the file
# instead of relaxing the check, which would mean handing the gateway credential
# to the lender: GRPO does not need it. The only RL-relevant key it carries,
# KORE_USE_VENDOR_BASELINE, is exported with a default below, and AMD_LLM_API_KEY
# reaches kore_env only through the ambient-secret deny-list that strips it from
# the bench subprocess.
if [[ "$BORROWED" == 1 ]]; then
    echo "[grpo-1node] borrowed: skipping .env.local (its teacher credential is unused by GRPO)"
else
    kore_secure_source_env "$REPO/.env.local"
fi

# Fail closed BEFORE the physical masks are dropped: a SPUR partial-GPU allocation
# mounts an unusable device subset, and once ROCR_VISIBLE_DEVICES is unset torch
# reports every device on the node, so the shortfall becomes invisible.
VISIBLE_GPUS="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$VISIBLE_GPUS" != "8" ]]; then
    echo "[grpo-1node] FATAL expected 8 GPUs, found $VISIBLE_GPUS" >&2
    exit 2
fi

# Refuse a node whose GPUs are already occupied, BEFORE loading 30B of weights.
#
# Job 29554 died on crsuse2-m2m-192 with:
#   HIP out of memory ... GPU 0 has a total capacity of 287.98 GiB of which
#   0 bytes is free. Of the allocated memory 26.76 GiB is allocated by PyTorch
# at grpo.py:3751, placing the full-weight rollout replica. Our own process held
# 27 GiB while the device had nothing left, so ~260 GiB belonged to something
# else. This job asks for --exclusive and was granted gpu:8/node, and this
# controller exposes no OverSubscribe field to confirm exclusivity, so the two
# candidates are a stale process left by a previously cancelled job or an
# exclusivity request that was not enforced. Both look identical from inside.
#
# The cost of guessing is an entire allocation: the failure surfaced ~15 minutes
# in, after both models had started loading. This check costs one already-paid
# torch import and turns that into a few seconds, and it PRINTS what is holding
# the memory, so the next occurrence identifies the cause instead of repeating
# the ambiguity. Exiting non-zero simply lets the watchdog place the job
# elsewhere, which is the correct response to a bad node either way.
GPU_DIRTY_GB="${KORE_GPU_DIRTY_GB:-8}"
DIRTY="$("$PY" - "$GPU_DIRTY_GB" <<'PY'
import sys
import torch
limit = float(sys.argv[1])
bad = []
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    used = (total - free) / 2**30
    if used > limit:
        bad.append(f"gpu{i}={used:.1f}GiB")
print(" ".join(bad))
PY
)"
if [[ -n "$DIRTY" ]]; then
    echo "[grpo-1node] FATAL node $(hostname) has occupied GPUs before we start: $DIRTY" >&2
    echo "[grpo-1node] expected every device idle under --exclusive; who holds it:" >&2
    rocm-smi --showpids 2>&1 | head -20 >&2
    rocm-smi --showmeminfo vram 2>&1 | grep -iE "GPU|used" | head -20 >&2
    printf '%s %s\n' "$(date +%s)" "$(hostname -s)" \
        >> "${KORE_DIRTY_NODES:-/home/shasriva/.kore_dirty_nodes}" 2>/dev/null || true

    # A borrowed allocation gets the diagnosis above and nothing else. Excluding
    # the node and requeueing would be applied to the lender's job, which is both
    # destructive and useless: their allocation is pinned to this node, so the
    # requeue cannot move us off it.
    if [[ "$BORROWED" == 1 ]]; then
        echo "[grpo-1node] borrowed: not excluding or requeueing the lender's job" >&2
        exit 2
    fi

    # Do not die on a bad node: add it to THIS job's own exclusion list and
    # requeue. Exiting instead needs an external resubmitter, and a dirty node is
    # the scheduler's next free candidate every time, so the job would be handed
    # the same one repeatedly. Self-excluding makes each cycle strictly better
    # informed than the last with nothing watching over it.
    #
    # Capped, because the failure this cannot distinguish is "the whole partition
    # is busy": past the cap, excluding more nodes only makes the job
    # unschedulable, and a held job is worse than a failed one to diagnose.
    EXC_CAP="${KORE_EXCLUDE_CAP:-12}"
    PRIOR="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
             | tr ' ' '\n' | sed -n 's/^ExcNodeList=//p' | grep -v '^(null)$' || true)"
    COUNT=0
    [[ -n "$PRIOR" ]] && COUNT=$(( $(tr ',' '\n' <<<"$PRIOR" | grep -c .) ))
    if (( COUNT < EXC_CAP )); then
        NEXT="$(hostname -s)"
        [[ -n "$PRIOR" ]] && NEXT="$PRIOR,$NEXT"
        if scontrol update jobid="$SLURM_JOB_ID" ExcNodeList="$NEXT" 2>&1; then
            echo "[grpo-1node] excluded $(hostname -s) (now $((COUNT + 1))/$EXC_CAP); requeueing" >&2
            # requeue_self() is defined much further down, so inline it here. As
            # noted there, SPUR's scontrol exits 0 even when it cannot reach the
            # controller, so the output has to be inspected rather than the status.
            RQ="$(scontrol requeue "$SLURM_JOB_ID" 2>&1)"
            if ! grep -qiE 'error|failed|unavailable|no leader|refused' <<<"$RQ"; then
                exit 0
            fi
            printf '%s\n' "$RQ" >&2
        fi
        echo "[grpo-1node] requeue failed; exiting so the node is released" >&2
    else
        echo "[grpo-1node] FATAL $COUNT nodes already excluded; the partition is busy, "\
             "not broken. Not shrinking the pool further." >&2
    fi
    exit 2
fi
echo "[grpo-1node] all 8 GPUs idle (< ${GPU_DIRTY_GB} GiB used); node is clean"

# Full-node GRES bind-mounts all eight render devices; drop SPUR's physical list
# so accelerate assigns one logical device per rank. grpo.py then maps each
# rank's LOCAL_RANK back through the accelerate-set CUDA_VISIBLE_DEVICES to the
# absolute id its bench subprocess needs.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

# scripts/launch_distributed.sh execs BARE `accelerate` with no interpreter path,
# so the venv must be on PATH.
export PATH="/home/shasriva/kore-venv/bin:$PATH"
# aiter (the production vendor baselines) is a source checkout at repos/aiter and
# is NOT pip-installed, so it must be on PYTHONPATH. Without it the vendor
# baseline silently falls back to torch, every measured speedup inflates, and the
# RL reward rewards the wrong thing. AITER_USE_SYSTEM_TRITON=1 lets aiter run on
# the system triton (its gluon kernels are gfx1250-only, irrelevant to gfx950).
export PYTHONPATH="$REPO:$REPO/repos/aiter"
export AITER_USE_SYSTEM_TRITON=1
export GPU_TARGET=gfx950
# ---- OFFLINE (air-gapped): never touch the hub ----
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/home/shasriva/.cache/huggingface
# This cache is mode 700 under one user's home, so a borrowed run cannot read it,
# and transformers wants to create paths under HF_HOME even when it fetches
# nothing. Give the lender its own cache: offline mode plus an absolute model
# path means nothing is downloaded into it, it only has to be writable.
if [[ "$BORROWED" == 1 ]]; then
    for _hf in "/mnt/m2m_nobackup/${USER}/kore/hf" "${TMPDIR:-/tmp}/kore-hf-${USER}"; do
        if mkdir -p "$_hf" 2>/dev/null; then export HF_HOME="$_hf"; break; fi
    done
    echo "[grpo-1node] borrowed: HF_HOME=$HF_HOME"
fi
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
# FREE SPACE IS NOT PERMISSION, and picking on space alone cost a whole
# allocation. Job 24351 read "/tmp (85G free)", selected it, and died one line
# later on `mkdir: cannot create directory '/tmp/kore-24351': Permission
# denied` -- under `set -e` that is an immediate exit 1, so the watchdog
# resubmitted into the same wall every 90 s and RL made no progress at all.
# On these nodes /tmp has room but is not writable by the job user.
#
# So CLAIM the directory rather than predicting that a claim would work, and
# take the first candidate that both fits and actually accepts a mkdir:
#   1. /mnt/m2m_nobackup - node-local NVMe, 28T with 24T free, and already the
#      launcher's choice for LOCAL_ROOT below, so it is proven writable here.
#   2. /tmp             - the historical choice; kept for nodes where it works.
#   3. shared /home     - last resort. It is NFS, it contends across ranks, and
#      it is at 100% capacity cluster-wide, so it is a correctness fallback
#      rather than a good one.
_scratch_avail_gb() {
    local gb
    gb="$(df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9')"
    echo "${gb:-0}"
}
KORE_SCRATCH=""
for _cand in "/mnt/m2m_nobackup/${USER}/kore-scratch" "/tmp" "$REPO/runs/.scratch"; do
    _avail="$(_scratch_avail_gb "$(dirname "$_cand")")"
    if [ "$_avail" -lt "$KORE_SCRATCH_MIN_GB" ]; then
        echo "[grpo-1node] scratch candidate $_cand: only ${_avail}G < ${KORE_SCRATCH_MIN_GB}G, skipping"
        continue
    fi
    _try="$_cand/kore-${SLURM_JOB_ID:-$$}"
    if mkdir -p "$_try/tmp" "$_try/triton" "$_try/inductor" 2>/dev/null \
       && [ -w "$_try/tmp" ]; then
        KORE_SCRATCH="$_try"
        echo "[grpo-1node] compile scratch: $KORE_SCRATCH (${_avail}G free, writable)"
        break
    fi
    echo "[grpo-1node] scratch candidate $_cand: ${_avail}G free but NOT writable, skipping"
done
if [ -z "$KORE_SCRATCH" ]; then
    echo "[grpo-1node] FATAL no writable compile scratch among" \
         "/mnt/m2m_nobackup, /tmp, $REPO/runs/.scratch" >&2
    exit 2
fi
export TMPDIR="$KORE_SCRATCH/tmp"
export TRITON_CACHE_DIR="$KORE_SCRATCH/triton"
export TORCHINDUCTOR_CACHE_DIR="$KORE_SCRATCH/inductor"
trap 'rm -rf "$KORE_SCRATCH"' EXIT

export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Reward integrity, matching every other production launcher in this repo
# (spur_datagen_array / spur_stage_array / spur_gpu_smoke all set both to 1):
# the adversarial correctness battery so no kernel lucky-passes the fixed seeds,
# and a torch.compile-fused baseline so a "speedup" is honest. Overridable from
# .env.local or the submit environment.
export KORE_VERIFIED_CORRECTNESS="${KORE_VERIFIED_CORRECTNESS:-1}"
export KORE_COMPILE_BASELINE="${KORE_COMPILE_BASELINE:-1}"
export KORE_USE_VENDOR_BASELINE="${KORE_USE_VENDOR_BASELINE:-1}"
# KORE_SHAPE_AUGMENT stays at its RL default of 0: augmented shapes widen
# per-bench variance, and the GRPO advantage is computed from those speedups, so
# the extra noise degrades the learning signal instead of the reward's honesty
# (the same argument reverify makes for its timing-measurement stage).
export KORE_SHAPE_AUGMENT="${KORE_SHAPE_AUGMENT:-0}"
# Per-GPU exclusive lock around TIMED regions only, so ranks that land on the
# same physical device cannot inflate each other's CV (compiles still overlap).
export KORE_TIMING_LOCK="${KORE_TIMING_LOCK:-1}"
# Recipe integrity. kore.policy.capabilities.log_capability_audit runs at
# train_grpo startup and reports every requested-but-inert capability; 'error'
# promotes that report to a hard failure before the 8-rank weight load. This is
# the fail-closed lane for a config that CANNOT set strict_feature_validation:
# the strict profile bans use_search, value_prefilter, coevolve_mint,
# search_bnb, coevolve_regret_vs_opus and distillation outright, so turning it
# on here would delete the recipe rather than verify it. The audit checks
# artifacts too, so this also refuses to burn an allocation training with a
# silently-absent KL retention anchor because the SFT stage has not landed yet.
# Overridable for a deliberate degraded run.
export KORE_GRPO_INERT_FEATURES="${KORE_GRPO_INERT_FEATURES:-error}"
# report, NEVER strict: no measured peak profile exists for this workload, so
# strict mode would refuse to start. Exported AFTER .env.local deliberately.
export KORE_RESOURCE_PREFLIGHT=report

# ---- frozen shape-manifest code identity ---- #
# The held-out shape lane records a "code identity" digest that is simply the
# repository git HEAD, so ANY commit trips it -- including a commit that cannot
# possibly move a task shape. That is not hypothetical: it took this run down.
#
#   Job 23935 trained happily at HEAD=736ba9327. Commit 51c337c1 then landed,
#   whose entire content is +118 lines of configs/grpo_coder30b_a3b_trloo_burst.json
#   (verified: `git diff --stat 736ba9327 51c337c13` is that one file). The next
#   requeue read the new HEAD, failed the digest, and EVERY subsequent GRPO job
#   died on rank 0 before loading a single weight, in a resubmit loop.
#
# shape_policy.py names the sanctioned recoveries: re-freeze the lane, or pin
# KORE_CODE_IDENTITY to the frozen value to assert this commit moved no shape.
# Re-freezing is the wrong one -- it changes the hidden shapes and voids every
# comparison against the runs already completed on this lane, which is the whole
# point of freezing it. So pin, and pin DELIBERATELY:
#
# This is safe because the digest that actually covers shape-derivation code is
# a SEPARATE one ('policy engine'), it is still enforced on every startup, and it
# PASSED here. The pin silences the git-HEAD proxy, not the real guard.
#
# The cost is honest: while pinned, the proxy no longer notices commits at all.
# Refresh it deliberately whenever the held-out lane is legitimately re-frozen,
# and let the 'policy engine' digest be what catches a real shape move.
export KORE_CODE_IDENTITY="${KORE_CODE_IDENTITY:-736ba9327ad8c45fc9f3ee5ebe2a9e8dffbfe58e}"

# Node-local compile/temp state: kernel compilation is the hot path here and an
# NFS-backed TMPDIR both contends across ranks and is where the per-GPU timing
# lock lives. Only checkpoints and logs go to shared /home.
LOCAL_ROOT="/mnt/m2m_nobackup/${USER}/kore/grpo-${SLURM_JOB_ID:-manual}"
# Same lesson as the scratch selection above: an unguarded mkdir here is another
# single point of failure for the whole allocation, so fall back to the scratch
# root already proven writable instead of exiting on it.
if ! mkdir -p "$LOCAL_ROOT/tmp" "$LOCAL_ROOT/compile-cache" 2>/dev/null; then
    echo "[grpo-1node] WARNING $LOCAL_ROOT not writable; using $KORE_SCRATCH instead" >&2
    LOCAL_ROOT="$KORE_SCRATCH"
    mkdir -p "$LOCAL_ROOT/tmp" "$LOCAL_ROOT/compile-cache"
fi
export TMPDIR="$LOCAL_ROOT/tmp"
export KORE_COMPILE_CACHE_DIR="$LOCAL_ROOT/compile-cache"

echo "[grpo-1node] START job=${SLURM_JOB_ID:-none} host=$(hostname) gpus=$VISIBLE_GPUS"
echo "[grpo-1node] HEAD=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "[grpo-1node] cfg=$CFG from=$FROM_STAGE out=$OUT_DIR data_root=$DATA_ROOT"

# SPUR's CLI talks to http://localhost:6817 unless SPUR_CONTROLLER_ADDR says
# otherwise, and spurctld does NOT run on every node (this cluster's login node
# has none). A 2000-step RL run crosses many allocations, so an unreachable
# controller here means the chain stops after ONE - report it up front.
CONTROLLER_PING="$(scontrol ping 2>&1 || true)"
if grep -q 'is UP' <<<"$CONTROLLER_PING"; then
    echo "[grpo-1node] controller reachable: $CONTROLLER_PING"
else
    echo "[grpo-1node] WARNING controller unreachable from this node, so the timed" \
         "self-requeue will fail and this job will end at its time limit without" \
         "continuing (a manual resubmit still resumes from the newest complete" \
         "checkpoint). Fix: submit with SPUR_CONTROLLER_ADDR exported, or set" \
         "KORE_SPUR_CONTROLLER_ADDR. Detail: $CONTROLLER_PING" >&2
fi

RESOLVED="$REPO/runs/grpo-${SLURM_JOB_ID:-manual}.resolved.json"
# A borrowed allocation runs under the lender's uid, which has no write access to
# this tree, so the resolved config cannot live in it. OUT_DIR is the right home:
# the run must be able to write there anyway or it could not save a checkpoint,
# and keeping the config beside the checkpoints it produced is what a later
# resume wants to read.
if [[ "$BORROWED" == 1 ]]; then
    mkdir -p "$OUT_DIR" || { echo "[grpo-1node] FATAL borrowed run cannot write OUT_DIR=$OUT_DIR" >&2; exit 2; }
    RESOLVED="$OUT_DIR/grpo-${SLURM_JOB_ID:-manual}.resolved.json"
fi
"$PY" scripts/spur_resolve_launch_config.py \
    --stage grpo --config "$CFG" --out "$RESOLVED" \
    --from-stage "$FROM_STAGE" --output-dir "$OUT_DIR" --data-root "$DATA_ROOT"

# Report the resume point BEFORE the 8-rank load. GRPO validates a checkpoint
# against the file manifest its writer recorded, so this uses grpo.py's own
# discovery rather than the Trainer-shaped latest_checkpoint helper.
"$PY" - "$RESOLVED" <<'PY'
import json, sys
from pathlib import Path
from kore.policy.grpo import _find_grpo_resume_checkpoint
cfg = json.loads(Path(sys.argv[1]).read_text())
found = _find_grpo_resume_checkpoint(type("C", (), {"output_dir": cfg["output_dir"]}))
if found is None:
    print(f"[grpo-1node] resume_from=None output_dir={cfg['output_dir']} (fresh run)")
else:
    print(f"[grpo-1node] resume_from={found[0]} "
          f"start_step={found[1].get('global_step')} total_steps={cfg.get('total_steps')}")
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
    echo "[grpo-1node] signal received; the newest complete checkpoint is safe; exiting for requeue"
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
    # Single choke point for every requeue path, so a borrowed allocation is
    # refused here even if a new caller is added later.
    if [[ "$BORROWED" == 1 ]]; then
        echo "[grpo-1node] borrowed: refusing to requeue a job this run does not own" >&2
        return 1
    fi
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
    echo "[grpo-1node] USR1 time-limit warning; draining and requesting requeue"
    terminate_child
    if requeue_self; then exit 0; fi
    echo "[grpo-1node] FATAL explicit requeue failed" >&2
    exit 143
}
cleanup() {
    if [[ -n "$REQUEUE_TIMER_PID" ]]; then kill "$REQUEUE_TIMER_PID" 2>/dev/null || true; fi
    rm -rf -- "$LOCAL_ROOT"
}
trap cleanup EXIT
trap handle_termination TERM INT
trap handle_requeue_warning USR1

# SPUR's sbatch does not implement Slurm's --signal option, and Slurm does NOT
# auto-requeue a job that merely hits its time limit (only preemption / node
# failure). Warn locally at 22h30m of the 23h limit so the run drains and
# EXPLICITLY requeues itself with 30 min to spare - which is how a 2000-step RL
# run crosses many allocations.

# Slurm duration ([DD-]HH:MM:SS, [DD-]HH:MM, MM:SS, UNLIMITED) -> seconds, 0 if
# it cannot be read. Used to size the drain against the allocation we actually
# got rather than against a constant.
_slurm_secs() {
    local t="${1:-}" d=0 s=0
    [[ -z "$t" || "$t" == UNLIMITED || "$t" == "?" || "$t" == "N/A" ]] && { echo 0; return; }
    if [[ "$t" == *-* ]]; then d="${t%%-*}"; t="${t#*-}"; fi
    local IFS=:; local -a p; read -ra p <<<"$t"
    case ${#p[@]} in
        3) s=$(( 10#${p[0]} * 3600 + 10#${p[1]} * 60 + 10#${p[2]} )) ;;
        2) s=$(( 10#${p[0]} * 60 + 10#${p[1]} )) ;;
        1) s=$(( 10#${p[0]} )) ;;
        *) s=0 ;;
    esac
    echo $(( d * 86400 + s ))
}

# The drain wants to fire just before OUR wall, so derive it from the allocation
# instead of trusting a constant. A hardcoded 39600 (tuned for burst's 12h
# ceiling) was carried onto a 7-day primus allocation, which would have requeued
# a healthy run after 11 hours and sent it back to the queue for the remaining
# six days. A constant cannot be right for both queues; the time limit can.
# KORE_REQUEUE_AFTER_SECONDS still wins when it is SHORTER, so an explicit early
# drain is still possible, but it can no longer silently truncate a long one.
if [[ -n "${SLURM_JOB_ID:-}" && "$BORROWED" != 1 ]]; then
    REQUEUE_MARGIN="${KORE_REQUEUE_MARGIN_SECONDS:-1800}"
    LIMIT_S="$(_slurm_secs "$(squeue -h -j "$SLURM_JOB_ID" -o '%l' 2>/dev/null | tr -d '[:space:]')")"
    DERIVED=0
    (( LIMIT_S > REQUEUE_MARGIN )) && DERIVED=$(( LIMIT_S - REQUEUE_MARGIN ))
    REQUEUE_AFTER="${KORE_REQUEUE_AFTER_SECONDS:-0}"
    # Take the derived value when the env is unset, and when the env asks for a
    # drain LONGER than the allocation can honour. An env that asks for a
    # shorter one is an explicit early drain and is kept, which is what the
    # comment above promises. Written as `DERIVED > REQUEUE_AFTER` this was the
    # max of the two, so `KORE_REQUEUE_AFTER_SECONDS=3600` against a 23h wall
    # was silently replaced by 81000 and the early drain never happened.
    if (( REQUEUE_AFTER <= 0 || DERIVED < REQUEUE_AFTER )); then
        REQUEUE_AFTER="$DERIVED"
    fi
    (( REQUEUE_AFTER > 0 )) || REQUEUE_AFTER=81000        # limit unreadable
    echo "[grpo-1node] self-requeue drain in ${REQUEUE_AFTER}s" \
         "(time limit ${LIMIT_S}s, margin ${REQUEUE_MARGIN}s," \
         "env ${KORE_REQUEUE_AFTER_SECONDS:-unset})"
    (
        sleep "$REQUEUE_AFTER"
        kill -USR1 "$$"
    ) &
    REQUEUE_TIMER_PID="$!"
elif [[ "$BORROWED" == 1 ]]; then
    # The drain exists to hand a wall-limited allocation back cleanly. In a
    # borrowed one it would only kill the lender's job, and its own wall is
    # theirs to manage, so there is nothing here to drain against.
    echo "[grpo-1node] borrowed: drain timer not armed (the allocation's wall belongs to its owner)"
fi

# setsid puts accelerate + its 8 workers (and their bench subprocesses) in their
# own process group, so the drain above can signal the whole tree with one kill.
RC=0
setsid bash scripts/launch_distributed.sh grpo "$RESOLVED" &
CHILD_PID="$!"
if wait "$CHILD_PID"; then
    echo "[grpo-1node] grpo completed"
else
    RC=$?
    echo "[grpo-1node] grpo failed rc=$RC"
fi

echo "GRPO_RC=$RC"
echo "[grpo-1node] END job=${SLURM_JOB_ID:-none} status=$RC"
exit "$RC"
