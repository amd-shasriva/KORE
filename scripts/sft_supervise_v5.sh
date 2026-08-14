#!/usr/bin/env bash
# Start the v5 SFT run under supervision on primus, and keep it alive.
#
# One entry point rather than a remembered command line, because the details are not
# guessable and getting one wrong is expensive:
#
#   * account amd-primus WITH qos amd-primus-qos, and nothing else. ONE job, queued,
#     waiting for a real allocation. The account must match the QoS; pairing
#     amd-general with amd-burst-qos is a phantom association that the controller
#     accepts and never schedules.
#
#     This settles a question that was reopened twice, so the reasoning is worth
#     keeping. Burst looked attractive because it placed jobs in under a minute while
#     the guaranteed pools were hours deep. That speed was an illusion: burst is the
#     lowest-priority tier, and the partition runs 135 alloc / 56 mix / 36
#     down-or-drained / 2 idle. Since the job needs --exclusive (without it we land on
#     nodes whose eight GPUs already hold 207-288 GiB of another tenant's memory), it
#     can only be placed on a WHOLLY idle node -- and on a cluster this saturated,
#     healthy nodes never sit idle, because they are taken the moment they free. The
#     idle ones are idle because they are broken.
#
#     Measured: six consecutive burst placements died on their node -- ...301, ...297,
#     ...296, ...331, ...291, ...317, four still DOWN afterwards -- one of them before
#     the job produced a single line of output. Total progress banked: none. Fast
#     placement onto hardware that dies is slower than waiting for hardware that works.
#
#     primus is also GUARANTEED, so unlike burst the run cannot be preempted once it
#     starts, which is what a multi-day run needs.
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
# 0 = NEVER cancel a running job. This supervisor can now only ever RESUBMIT a job
# that has already left the queue; it cannot end one itself.
#
# It was 2700 (45 min), on the theory that a job whose log has gone silent that long is
# wedged rather than working. The theory is sound and the threshold is generous -- the
# live run touches its log every 11-56s, a ~48x margin -- but the asymmetry of the bet
# is wrong for a 25-hour unattended run. A hung job costs the hours until a human looks
# at it. A wrongly cancelled job destroys every completed step that is not yet in a
# checkpoint, and the things that could legitimately go quiet for 45 minutes are exactly
# the things this run does: a sharded save at 30B, a held-out eval over 899 rows across
# 8 groups, an NFS hiccup.
#
# What is given up: nothing recovers a job that is alive but hung. That is a failure a
# human notices from the step counter not advancing, and it has never been observed on
# this run. What is kept: recovery from the failure that HAS been observed repeatedly,
# a node dying, which takes the job out of the queue and is handled by resubmission.
export STALL_SECS="${STALL_SECS:-0}"
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
