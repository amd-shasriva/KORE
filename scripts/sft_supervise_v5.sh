#!/usr/bin/env bash
# Start the v5 SFT run under supervision on burst nodes, and keep it alive.
#
# One entry point rather than a remembered command line, because the details are not
# guessable and getting one wrong is expensive:
#
#   * QoS is amd-burst-qos, NOT amd-general-qos. General is capped at 8 nodes
#     team-wide, all eight are held by other users, and our job sat SIXTH in that
#     queue behind five single-node jobs -- with the running jobs' limits, that is
#     potentially a day before it starts. Burst reaches all 282 nodes and starts now.
#     The cost is preemptibility, which is what this supervisor is for.
#
#   * KORE_OUTPUT_DIR must match the config's output_dir. The supervisor uses it as
#     the authoritative "this run finished" test, so a mismatch means it can never
#     tell success from preemption and would resubmit a completed run.
#
#   * MIN_PROGRESS_SECS / MAX_FAST_FAILURES stop a crash-loop. A job that dies
#     seconds after starting is a bug, not a preemption; job 6520 died in 12s on a
#     shell error, and a supervisor without this retries the same bug 40 times.
#
# Safe to re-run: it adopts an in-flight kore-sft job instead of submitting a second
# one, because two runs sharing output_dir would interleave checkpoint writes and
# destroy the thing that makes preemption survivable.
set -uo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
cd "$REPO" || exit 1

export SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR:-http://crs-m2m-cpu-spur-005:6817}"
export KORE_OUTPUT_DIR="${KORE_OUTPUT_DIR:-$(
    /home/shasriva/kore-venv/bin/python -c \
      "import json;print(json.load(open('configs/sft_coder30b_a3b.json'))['output_dir'])")}"

# Preemption on burst is expected, so poll often enough to lose minutes rather than
# hours, and allow generously many resubmissions: at ~30h of training in 45-minute
# worst-case losses, the budget has to absorb a lot of churn.
export POLL_SECS="${POLL_SECS:-120}"
export MAX_RESUBMITS="${MAX_RESUBMITS:-60}"
export MIN_PROGRESS_SECS="${MIN_PROGRESS_SECS:-600}"
export MAX_FAST_FAILURES="${MAX_FAST_FAILURES:-3}"
# A 30B load plus a dataset map can legitimately look quiet for a while, and killing
# a healthy run for being slow is worse than waiting. 45 min.
export STALL_SECS="${STALL_SECS:-2700}"
export KORE_SFT_QOS="amd-burst-qos"

LOG="$REPO/runs/supervise_v5_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$REPO/runs"

echo "[supervise] repo=$REPO"
echo "[supervise] output_dir=$KORE_OUTPUT_DIR"
echo "[supervise] log=$LOG"

exec bash scripts/watch_and_resume.sh sft \
    sbatch --qos=amd-burst-qos "$REPO/scripts/spur_sft_1node.sbatch" \
    configs/sft_coder30b_a3b.json - - \
    >>"$LOG" 2>&1
