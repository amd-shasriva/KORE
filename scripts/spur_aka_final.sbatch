#!/bin/bash
# Final, measured AgentKernelArena sweep: one allocation, one baseline, N arms.
#
# WHY THIS EXISTS SEPARATELY FROM spur_aka_1node.sbatch
#
# The 2026-08-10 numbers (base 416 tasks, opus 413 tasks) measured correctness
# and nothing else: baseline_seconds was null on every single row, and a speedup
# survived on 22/416 and 49/413 -- only the suites that compute their own ratio
# in-process. The denominators were never produced, and because the workspaces
# are deleted after scoring they cannot be recovered. Those files are archived
# under runs/aka_archive_correctness_only/ and are not comparable to anything
# this script writes.
#
# Reproducing that mistake takes one wrong invocation, so the correct sequence
# is encoded here rather than left to the operator:
#
#   phase 0  preflight   -- GPUs present; every API arm proven live
#   phase 1  baseline    -- time each task's shipped kernel  -> baseline_results.json
#   phase 2  arm base    -- Qwen3-Coder-30B-A3B-Instruct     -> results_base.json
#   phase 3  arm opus    -- claude-opus-5 via the gateway    -> results_opus.json
#
# ONE ALLOCATION, ONE OUTPUT DIR, ON PURPOSE
#
# baseline_results.json is arm-independent: it is the reference kernel's latency,
# which does not depend on who wrote the challenger. Every arm in the same --out
# therefore divides by the *same* denominator, so in any arm-vs-arm comparison
# the baseline cancels exactly. Splitting the arms across jobs would give each
# its own baseline measured on a different node, and the arm-vs-arm ratio would
# then carry two nodes' worth of clock and thermal noise that no amount of
# averaging removes. Keeping them in one allocation also costs one queue wait
# instead of three, which on a partition with ~380 pending jobs is the
# difference between days and weeks.
#
# The v5 arm is deliberately the same script into the same --out once SFT ends:
#   KORE_AKA_ARMS=v5 KORE_AKA_V5_MODEL=/path/to/final sbatch ... scripts/spur_aka_final.sbatch
# It then divides by the identical denominators base and opus used.
#
# RESUMPTION IS FREE
#
# Every phase is idempotent. `baseline` skips task ids already present in any
# baseline*.jsonl ledger; `run` skips ids already in any results_<arm>*.jsonl.
# Re-running this script against a populated --out resumes; it does not restart.
# That is what makes --requeue safe after a node failure.
#
#SBATCH --job-name=kore-aka-final
# amd-primus with amd-primus-qos. The account MUST match the QoS: amd-general
# paired with a burst/primus QoS is a phantom association that the controller
# accepts at submit and then never dispatches. scripts/ensure_sft.sh passes the
# same pair explicitly; it is repeated here so a bare `sbatch` of this file is
# also correct rather than silently unschedulable forever.
#SBATCH --account=amd-primus
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-primus-qos
#SBATCH --nodes=1
# gpu:8, not gpu:mi355x:8 -- the typed form is accepted, picks a node, and never
# confirms (JobLaunchFailure). Measured 0/3 vs 2/3 dispatched.
#SBATCH --gres=gpu:8
# 32, not 128: a larger request blocked dispatch on this controller.
#SBATCH --cpus-per-task=32
# 3 days matches the walltime this QoS is known to accept (the v5 SFT job runs
# at exactly this limit). The sweep is expected to need well under one day; the
# margin is for a slow arm, not a plan.
#SBATCH --time=3-00:00:00
# Without --exclusive, burst placed jobs on nodes already holding 207-288 GiB of
# someone else's VRAM. A timing run on a shared GPU is not a measurement.
#SBATCH --exclusive
# Node failure should cost the tasks in flight, not the sweep. Safe only because
# every phase resumes from its ledger.
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/aka-final-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/aka-final-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO" || exit 2
# shellcheck disable=SC1091
source /home/shasriva/kore-venv/bin/activate
export PYTHONPATH="$REPO"
export HF_HOME=/home/shasriva/.cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# The base arm's weights are already in the local cache and the hub is not
# reachable in a way we want to depend on. The gateway used by API arms is a
# different endpoint and is unaffected by these.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

ARMS="${KORE_AKA_ARMS:-base opus}"
OUT="${KORE_AKA_OUT:-$REPO/runs/aka_v5_final}"
TYPES="${KORE_AKA_TYPES:-}"
LIMIT="${KORE_AKA_LIMIT:-0}"
ARENA="${KORE_AKA_ARENA_ROOT:-$HOME/third_party/AgentKernelArena}"
ARCH="${KORE_AKA_GPU_ARCH:-gfx950}"
BASE_MODEL="${KORE_AKA_BASE_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
OPUS_MODEL="${KORE_AKA_OPUS_MODEL:-claude-opus-5}"
V5_MODEL="${KORE_AKA_V5_MODEL:-}"
mkdir -p "$OUT" "$REPO/runs"

echo "=== kore-aka-final job=${SLURM_JOB_ID:-manual} host=$(hostname) ==="
echo "arms=$ARMS  out=$OUT  arena=$ARENA  arch=$ARCH  types=${TYPES:-ALL}  limit=$LIMIT"
echo "started=$(date -Is)"

model_for_arm() {
    case "$1" in
        base) echo "$BASE_MODEL" ;;
        opus) echo "$OPUS_MODEL" ;;
        v5)   echo "$V5_MODEL" ;;
        *)    echo "" ;;
    esac
}

# Matches _API_MODEL_PREFIXES in scripts/run_agent_kernel_arena.py. Kept in sync
# by tests/test_aka_final_launcher.py, which reads both files.
is_api_model() {
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
        claude-*|anthropic/*|gpt-*|opus*|sonnet*) return 0 ;;
        *) return 1 ;;
    esac
}

# ---- phase 0: preflight ---------------------------------------------------
# The tasks' harnesses import torch from OUR venv, so a node with no visible GPU
# does not raise -- every task simply fails correctness, and the sweep reads as a
# catastrophic model regression.
VIS="$("$PY" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
echo "visible GPUs: $VIS"
if [ "$VIS" = "0" ]; then
    echo "FATAL: no GPUs visible; every task would score 0"
    exit 2
fi

RESOLVED_ARMS=""
SKIPPED_ARMS=""
for arm in $ARMS; do
    m="$(model_for_arm "$arm")"
    if [ -z "$m" ]; then
        echo "FATAL: arm '$arm' has no model. Set KORE_AKA_$(echo "$arm" | tr '[:lower:]' '[:upper:]')_MODEL."
        exit 2
    fi
    if is_api_model "$m"; then
        echo "--- preflight $arm ($m) ---"
        "$PY" scripts/aka_gateway_preflight.py --model "$m"
        prc=$?
        if [ "$prc" != "0" ]; then
            # Do not throw away the allocation. A scarce node that can still
            # produce the baseline and every local arm is worth far more than a
            # clean exit, and the missing arm resumes into this same directory
            # later. The non-zero exit at the end is what makes the gap loud.
            echo "WARNING: arm '$arm' failed preflight (rc=$prc); it will be SKIPPED."
            SKIPPED_ARMS="$SKIPPED_ARMS $arm"
            continue
        fi
    else
        # A served arm gets a REAL check, not a reassuring message.
        #
        # This branch used to print "local checkpoint, no gateway needed" for
        # whatever string it was handed, without looking at it. On 2026-08-17 the
        # base arm was the launcher's default -- a Hub repo id, not a path -- and
        # a repo id with no pinned revision cannot be loaded at all. Preflight
        # announced it ready, the baseline ran for four hours, and then all 8
        # shards of the arm died two seconds in. check-model resolves the model
        # exactly the way the run will, and imports no GPU framework to do it.
        echo "--- preflight $arm ($m) ---"
        "$PY" scripts/run_agent_kernel_arena.py check-model --model "$m" \
              ${KORE_AKA_BASE_REVISION:+--revision "$KORE_AKA_BASE_REVISION"}
        mrc=$?
        if [ "$mrc" != "0" ]; then
            # Same reasoning as a failed gateway preflight: keep the allocation
            # for the baseline and any arm that can still run, and make the gap
            # loud at the end rather than silently shipping a missing arm.
            echo "WARNING: arm '$arm' failed model preflight (rc=$mrc); it will be SKIPPED."
            SKIPPED_ARMS="$SKIPPED_ARMS $arm"
            continue
        fi
    fi
    RESOLVED_ARMS="$RESOLVED_ARMS $arm"
done

common_args() {
    # shellcheck disable=SC2086
    echo "--arena-root $ARENA --gpu-arch $ARCH ${TYPES:+--types $TYPES} \
--out $OUT --limit $LIMIT"
}

# Fan one worker out per allocated GPU, then merge the shard ledgers.
#
# transformers loads with device_map="auto", so a single process puts the whole
# 30B on GPU 0 and idles seven. Each worker instead owns one GPU and a disjoint
# slice of tasks, with several tasks in flight so vLLM actually batches.
#
# The allocation already exports a visibility mask, and kore.policy.serve refuses
# two masks outright, so we narrow the inherited mask rather than stacking
# HIP_VISIBLE_DEVICES on top of it. Ordinals come from the mask, not from a
# 0..N-1 count, so every worker lands on a GPU we were actually given.
phase() {
    local mode="$1" arm="$2" model="$3" conc="$4"
    local label="$mode${arm:+_$arm}"
    echo ""
    echo "=== phase $label  model=${model:-n/a}  concurrency=$conc  $(date -Is) ==="

    local ALLOC="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-}}"
    [ -z "$ALLOC" ] && ALLOC="$(seq -s, 0 $((VIS - 1)))"
    IFS=',' read -r -a GPULIST <<< "$ALLOC"
    local NSHARD="${#GPULIST[@]}"

    local armflag=""
    [ -n "$arm" ] && armflag="--arm $arm"
    local modelflag=""
    [ -n "$model" ] && modelflag="--model $model"

    if [ "$NSHARD" -lt 2 ]; then
        # shellcheck disable=SC2086
        "$PY" scripts/run_agent_kernel_arena.py "$mode" $(common_args) $armflag $modelflag
        return $?
    fi

    local pids="" rc=0
    for i in $(seq 0 $((NSHARD - 1))); do
        # shellcheck disable=SC2086
        env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
            ROCR_VISIBLE_DEVICES="${GPULIST[$i]}" \
            "$PY" scripts/run_agent_kernel_arena.py "$mode" $(common_args) \
            $armflag $modelflag \
            --num-shards "$NSHARD" --shard "$i" \
            --task-concurrency "$conc" \
            > "$OUT/${label}_shard${i}.log" 2>&1 &
        pids="$pids $!"
        echo "  shard $i -> GPU ${GPULIST[$i]} (pid $!)"
    done
    for p in $pids; do wait "$p" || rc=1; done
    echo "phase $label: all shards exited (rc=$rc); merging"

    # Merge regardless of rc: a shard that died still wrote every task it
    # finished, and the merge is what turns those ledgers into the artifact.
    local merge="merge"
    [ "$mode" = "baseline" ] && merge="baseline-merge"
    # shellcheck disable=SC2086
    "$PY" scripts/run_agent_kernel_arena.py "$merge" $(common_args) $armflag $modelflag || rc=1
    return $rc
}

# ---- phase 1: baseline ----------------------------------------------------
# Unconditional and first. It is idempotent, so on a resume it costs a directory
# scan; and every arm below refuses to start without it (exit 2), which is the
# guard that makes a repeat of the correctness-only run impossible.
RC=0
phase baseline "" "" "${KORE_AKA_BASELINE_CONCURRENCY:-4}" || RC=1

# ---- phases 2..N: one per arm --------------------------------------------
# An arm that scored nothing at all is a different failure from an arm that
# scored badly, and only the first one is a bug in the run. Job 13907 logged
# "all shards exited (rc=1); merging" and "merge: no ledgers found", then moved
# on to the next arm as though a phase had completed -- which is how a dead
# control arm stays invisible until someone reads the artifact list. Count the
# ledger rows and name the arm dead when there are none.
DEAD_ARMS=""
for arm in $RESOLVED_ARMS; do
    m="$(model_for_arm "$arm")"
    conc="${AKA_TASK_CONCURRENCY:-4}"
    # An API arm's concurrency is a request rate, not a GPU occupancy: 8 workers
    # x 4 in flight is 32 simultaneous gateway calls. The teacher retries 429s
    # with bounded backoff, so the run survives either way, but backoff spent
    # waiting is allocation time burned. Halved by default.
    is_api_model "$m" && conc="${KORE_AKA_OPUS_CONCURRENCY:-2}"
    phase run "$arm" "$m" "$conc" || RC=1

    rows=$(cat "$OUT"/results_"$arm"*.jsonl 2>/dev/null | wc -l)
    if [ "${rows:-0}" -eq 0 ]; then
        echo ""
        echo "########################################################################"
        echo "# ARM '$arm' PRODUCED ZERO SCORED TASKS."
        echo "# model=$m"
        echo "# Every shard exited without writing a ledger row, so there is no"
        echo "# results_$arm.json to compare against any other arm. This is a"
        echo "# failed run, not a bad score. First error from shard 0:"
        sed -n '/Traceback/,$p' "$OUT/run_${arm}_shard0.log" 2>/dev/null \
            | tail -12 | sed 's/^/#   /'
        echo "########################################################################"
        DEAD_ARMS="$DEAD_ARMS $arm"
        RC=1
    else
        echo "arm '$arm': $rows scored task row(s) on disk"
    fi
done

# ---- report ---------------------------------------------------------------
echo ""
echo "=== artifacts in $OUT  $(date -Is) ==="
ls -1 "$OUT"/baseline_results.json "$OUT"/results_*.json 2>/dev/null || true
"$PY" - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
b = out / "baseline_results.json"
nb = 0
if b.exists():
    try:
        nb = len(json.loads(b.read_text()).get("results", []))
    except Exception:
        nb = -1
print(f"baseline rows: {nb}")
for f in sorted(out.glob("results_*.json")):
    try:
        d = json.loads(f.read_text())
    except Exception:
        print(f"{f.name}: unreadable"); continue
    rows = d.get("results", [])
    s = d.get("summary", {}) or {}
    timed = sum(1 for r in rows if r.get("speedup") is not None)
    print(f"{f.name}: {len(rows)} rows, {timed} with a measured speedup, "
          f"coverage={s.get('speedup_coverage')}")
PYEOF

if [ -n "$SKIPPED_ARMS" ]; then
    echo "INCOMPLETE: arms skipped after failed preflight:$SKIPPED_ARMS"
    echo "  fix the gateway, then resubmit with KORE_AKA_ARMS=\"$(echo "$SKIPPED_ARMS" | xargs)\" and the same KORE_AKA_OUT=$OUT"
    RC=1
fi
if [ -n "$DEAD_ARMS" ]; then
    echo "INCOMPLETE: arms that ran but scored nothing:$DEAD_ARMS"
    echo "  resubmit with KORE_AKA_ARMS=\"$(echo "$DEAD_ARMS" | xargs)\" and the same KORE_AKA_OUT=$OUT"
    echo "  the baseline and any healthy arm in that directory are reused, not rerun"
    RC=1
fi
echo "aka-final rc=$RC"
exit $RC
