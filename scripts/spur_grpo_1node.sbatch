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
#SBATCH --gres=gpu:mi355x:8
#SBATCH --cpus-per-task=128
#SBATCH --time=23:00:00
#SBATCH --requeue
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

cd "$REPO"
mkdir -p runs
command -v scontrol >/dev/null || { echo "[grpo-1node] missing required command: scontrol" >&2; exit 2; }
command -v setsid >/dev/null || { echo "[grpo-1node] missing required command: setsid" >&2; exit 2; }

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
    echo "[grpo-1node] FATAL expected 8 GPUs, found $VISIBLE_GPUS" >&2
    exit 2
fi

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
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Intra-node RCCL only (one node, no IB fabric to pin).
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
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

# Node-local compile/temp state: kernel compilation is the hot path here and an
# NFS-backed TMPDIR both contends across ranks and is where the per-GPU timing
# lock lives. Only checkpoints and logs go to shared /home.
LOCAL_ROOT="/mnt/m2m_nobackup/${USER}/kore/grpo-${SLURM_JOB_ID:-manual}"
export TMPDIR="$LOCAL_ROOT/tmp"
export KORE_COMPILE_CACHE_DIR="$LOCAL_ROOT/compile-cache"
mkdir -p "$TMPDIR" "$KORE_COMPILE_CACHE_DIR"

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
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    (
        sleep "${KORE_REQUEUE_AFTER_SECONDS:-81000}"
        kill -USR1 "$$"
    ) &
    REQUEUE_TIMER_PID="$!"
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
