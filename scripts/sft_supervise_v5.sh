#!/usr/bin/env bash
# Start the v5 SFT run under supervision on burst nodes, and keep it alive.
#
# One entry point rather than a remembered command line, because the details are not
# guessable and getting one wrong is expensive:
#
#   * account amd-primus WITH qos amd-primus-qos, because that is the pairing that
#     DEMONSTRABLY schedules. The run landed there (job 9229 on crsuse2-m2m-037) after
#     burst never did: even a 1-GPU, 1-CPU, 5-minute burst probe stayed PENDING while
#     113 other burst jobs ran, so the amd-burst association is present but not
#     functioning for placement. Recovery must target a pool we have actually started
#     in, or a preemption would resubmit into a queue that never runs.
#
#     primus is also a GUARANTEED pool, so unlike burst the run cannot be preempted --
#     which is a better outcome than the burst plan this file previously described.
#
#   * HISTORICAL, for context: account amd-burst WITH qos amd-burst-qos. Both parts matter: burst QoS paired
#     with the amd-general account was a PHANTOM association -- accepted by the
#     controller and never scheduled, because zero of the running burst jobs used that
#     account. A job of ours sat PENDING with the meaningless Reason=None for nine
#     hours while 42 jobs submitted later ran. Ops added this user to the amd-burst
#     account on 2026-08-13, which is what makes the pairing real.
#
#     Why burst rather than the guaranteed pools: amd-general-qos is capped at 8 nodes
#     and amd-primus-qos at 16, both were full, and both are dominated by holds that do
#     not rotate (general had a 15-day job and two unlimited; primus had two 30-day,
#     three 14-day and two unlimited). Measured churn was one opening every ~4h per
#     pool with 7 competitors ahead, i.e. ~32h. Burst is using 113 of ~282 nodes, so
#     the cap no longer binds and only real free hardware does.
#
#     Burst is preemptible, which is the trade and the reason this supervisor exists.
#     amd-general and amd-primus jobs stay queued as backups; the training lock in the
#     launcher guarantees that whichever starts first is the only one that trains.
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
# High, because dispatch failures consume attempts without ever starting the job.
# GPU dispatch on this cluster was observed failing roughly half the time, so a
# budget sized only for preemptions would be exhausted before training began. The
# guard against a genuine crash-loop is MAX_FAST_FAILURES, which only counts jobs
# that actually STARTED and then died quickly -- a failed dispatch cannot trip it,
# so raising this does not weaken that protection.
export MAX_RESUBMITS="${MAX_RESUBMITS:-300}"
export MIN_PROGRESS_SECS="${MIN_PROGRESS_SECS:-600}"
export MAX_FAST_FAILURES="${MAX_FAST_FAILURES:-3}"
# A 30B load plus a dataset map can legitimately look quiet for a while, and killing
# a healthy run for being slow is worse than waiting. 45 min.
export STALL_SECS="${STALL_SECS:-2700}"
# 0 = never cancel a merely-waiting job. Queue position is by submit time at equal
# priority, so resubmitting a pending job sends it to the back of the queue -- on a
# cluster with zero fully-idle nodes that is strictly worse than waiting. The run
# wants a whole node (--exclusive), so a long pending wait is expected and correct.
export STUCK_PENDING_SECS="${STUCK_PENDING_SECS:-0}"
export KORE_SFT_QOS="amd-primus-qos"

LOG="$REPO/runs/supervise_v5_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$REPO/runs"

echo "[supervise] repo=$REPO"
echo "[supervise] output_dir=$KORE_OUTPUT_DIR"
echo "[supervise] log=$LOG"

exec bash scripts/watch_and_resume.sh sft \
    sbatch --account=amd-primus --qos=amd-primus-qos "$REPO/scripts/spur_sft_1node.sbatch" \
    configs/sft_coder30b_a3b.json - - \
    >>"$LOG" 2>&1
