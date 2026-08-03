#!/bin/bash
# One SPUR array element = one node's shard of the agentic datagen campaign.
#
# Every element owns a disjoint task list and its own output JSONL, so nodes on
# the shared volume cannot collide. Requeue is safe and expected: run_node_shard
# checkpoints each finished episode and resumes by skipping the work items its
# own shard already holds, so a preempted node loses at most the episodes that
# were in flight.
#SBATCH --job-name=kore-agentic
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/agentic-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/agentic-%j.err

set -uo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"

# SPUR's sbatch does not reliably propagate `--export KEY=VALUE` into array
# children or requeued elements, so every knob arrives positionally.
SHARD_DIR="${1:?arg1 must be the absolute shard-plan directory}"
OUT_DIR="${2:?arg2 must be the absolute output directory}"
EPISODES="${3:-6}"
WORKERS="${4:-48}"
MAX_TURNS="${5:-8}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:?must be submitted as a Slurm array}"

cd "$REPO"
source /home/shasriva/kore-venv/bin/activate
source "$REPO/scripts/lib/ops_runtime.sh"
kore_secure_source_env "$REPO/.env.local"

printf -v IDX "%03d" "$ARRAY_ID"
SHARD_FILE="$SHARD_DIR/shard_${IDX}.txt"
[[ -f "$SHARD_FILE" ]] || { echo "missing shard plan: $SHARD_FILE" >&2; exit 2; }

# The plan pins the commit it was built from. A node running different code would
# write records under a contract the plan does not describe.
"$PY" - "$SHARD_DIR/manifest.json" "$REPO" "$OUT_DIR" "$ARRAY_ID" <<'PY'
import json, subprocess, sys
from pathlib import Path
manifest_path, repo, out_dir, array_id = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text())
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
errors = []
if manifest.get("repo_commit") != head:
    errors.append(f"manifest commit {manifest.get('repo_commit')} != checkout {head}")
if Path(manifest.get("out_dir", "")).resolve() != Path(out_dir).resolve():
    errors.append("manifest out_dir does not match job argument")
if not 0 <= int(array_id) < int(manifest.get("n_shards", 0)):
    errors.append("array index is outside manifest shard range")
if errors:
    raise SystemExit("FATAL manifest preflight: " + "; ".join(errors))
PY
[[ $? -eq 0 ]] || exit 2

export PYTHONPATH=.:/home/shasriva/Kore-RL/KORE/repos/aiter
export AITER_USE_SYSTEM_TRITON=1
export KORE_VERIFIED_CORRECTNESS=1
export KORE_COMPILE_BASELINE=1
export KORE_BENCH_COLD=1
export KORE_SHAPE_AUGMENT=1
export KORE_TIMING_LOCK=1
export GPU_TARGET=gfx950
export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

# Compile/temp/replay state is node-local: only the trajectory shards belong on
# the shared volume, so a preemption never leaves scratch behind on /home.
LOCAL_ROOT="/mnt/m2m_nobackup/${USER}/kore/agentic-${SLURM_JOB_ID}"
export TMPDIR="$LOCAL_ROOT/tmp"
export KORE_COMPILE_CACHE_DIR="$LOCAL_ROOT/compile-cache"
export KORE_RUNS_DIR="$LOCAL_ROOT/runs"
mkdir -p "$TMPDIR" "$KORE_COMPILE_CACHE_DIR" "$KORE_RUNS_DIR"

CHILD_PID=""
REQUEUE_TIMER_PID=""

cleanup_local() {
    [[ -n "$REQUEUE_TIMER_PID" ]] && kill "$REQUEUE_TIMER_PID" 2>/dev/null
    rm -rf -- "$LOCAL_ROOT"
}

terminate_child() {
    if [[ -n "$CHILD_PID" ]]; then
        kill -TERM -- "-$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
}

handle_termination() {
    trap - USR1 TERM INT
    terminate_child
    echo "[agentic] signal received; finished episodes are durable; exiting for requeue"
    exit 143
}

# SPUR's scontrol exits 0 even when it cannot reach the controller, so the exit
# status alone cannot confirm a requeue landed; exiting 0 on an unlanded requeue
# would retire the element with its shard unfinished.
requeue_self() {
    local out rc
    out="$(scontrol requeue "$SLURM_JOB_ID" 2>&1)"
    rc=$?
    if ((rc == 0)) && ! grep -qiE 'error|failed|unavailable|no leader|refused' <<<"$out"; then
        return 0
    fi
    printf '%s\n' "$out" >&2
    return 1
}

handle_requeue_warning() {
    trap - USR1 TERM INT
    echo "[agentic] time-limit warning; draining and requesting requeue"
    terminate_child
    if requeue_self; then exit 0; fi
    echo "[agentic] FATAL explicit requeue failed" >&2
    exit 143
}

trap cleanup_local EXIT
trap handle_termination TERM INT
trap handle_requeue_warning USR1

# SPUR does not implement Slurm's --signal, so generate the drain warning here.
(
    sleep "${KORE_REQUEUE_AFTER_SECONDS:-27600}"
    kill -USR1 "$$"
) &
REQUEUE_TIMER_PID="$!"

VISIBLE_GPUS="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
echo "[agentic] START job=$SLURM_JOB_ID shard=$IDX host=$(hostname) gpus=$VISIBLE_GPUS head=$(git rev-parse --short HEAD)"
if [[ "$VISIBLE_GPUS" != "8" ]]; then
    echo "[agentic] FATAL expected 8 GPUs, found $VISIBLE_GPUS" >&2
    exit 2
fi

# Stop the pool a few minutes before the drain signal so in-flight episodes end
# on their own rather than being killed with their work unwritten.
setsid "$PY" scripts/run_agentic_shard.py \
    --shard-file "$SHARD_FILE" \
    --out-dir "$OUT_DIR" \
    --shard-index "$ARRAY_ID" \
    --episodes-per-task "$EPISODES" \
    --workers "$WORKERS" \
    --max-turns "$MAX_TURNS" \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    --min-free-gb "${KORE_AGENTIC_MIN_FREE_GB:-150}" \
    --seconds "${KORE_AGENTIC_RUN_SECONDS:-26400}" &
CHILD_PID="$!"
wait "$CHILD_PID"
STATUS=$?

echo "[agentic] END job=$SLURM_JOB_ID shard=$IDX status=$STATUS"
exit "$STATUS"
