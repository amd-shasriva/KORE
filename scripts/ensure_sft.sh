#!/bin/bash
# Make sure the v5 SFT run is alive. Safe to run repeatedly, designed for cron.
#
# This exists because the layer above the training job died and took the run with
# it. Job 9229 was killed externally at 20:31:28 after 154 clean steps; the
# supervisor process died in the same window, so nothing resubmitted, and the run
# sat idle for 45 minutes until a human noticed. watch_and_resume.sh covers a job
# that fails. It cannot cover its own death, and setsid/nohup do not help when
# whatever killed the job reaches the login node too.
#
# So the top of the tree has to be owned by something the machine restarts rather
# than something a session owns. That is cron, which is also how
# scripts/ensure_loops.sh solved the same problem after a login-node reboot took
# out four loops at once and 745 gated tasks sat unharvested for six hours.
#
# Idempotent by construction: it starts only what is missing, and does nothing at
# all once training has genuinely finished.
#
#   scripts/ensure_sft.sh
#
# Stop everything permanently by creating the stop file:
#
#   touch runs/.sft_stopped
#
set -uo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
cd "$REPO" || exit 1
LOG="$REPO/runs/ensure_sft.log"
STOP="$REPO/runs/.sft_stopped"
ATTEMPTS="$REPO/runs/.sft_submit_attempts"
MAX_ATTEMPTS="${KORE_MAX_SUBMIT_ATTEMPTS:-60}"
mkdir -p "$REPO/runs"

say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$LOG"; }

# cron gives a minimal environment: no login shell, so no controller address and
# no venv on PATH. Every scheduler call below fails silently without this, which
# would make this guard look healthy while doing nothing.
[ -z "${SPUR_CONTROLLER_ADDR:-}" ] && [ -r /etc/profile.d/spur.sh ] && . /etc/profile.d/spur.sh
export SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR:-http://crs-m2m-cpu-spur-005:6817}"
export PATH="/home/shasriva/kore-venv/bin:$PATH"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"

# Only one of these may run at a time. The checks below are read-then-act, so two
# copies racing both conclude nothing is running and both submit. The lock is the
# part that cannot be raced.
exec 9>"$REPO/runs/.ensure_sft.lock"
if ! flock -n 9; then
    exit 0        # a concurrent guard holds it; silence is correct here
fi

if [ -f "$STOP" ]; then
    exit 0        # deliberately halted by a human; say nothing, every 10 minutes
fi

OUT_DIR="$("$PY" -c "
import json;print(json.load(open('configs/sft_coder30b_a3b.json'))['output_dir'])" 2>/dev/null)"
[ -z "$OUT_DIR" ] && { say "cannot read output_dir from config; doing nothing"; exit 1; }

# ---- 1. Has the training genuinely FINISHED? ---------------------------------
# Ask the work, not the scheduler. A consolidated model at the top of output_dir
# (rather than inside a checkpoint-* subdirectory) is what the trainer writes when
# it is done, and it names the run, so it cannot be satisfied by an earlier one.
# Getting this wrong in the optimistic direction means resubmitting a finished run
# forever; getting it wrong in the pessimistic direction means abandoning a live
# one, so it is checked narrowly.
if [ -s "$OUT_DIR/config.json" ] && ls "$OUT_DIR"/*.safetensors >/dev/null 2>&1; then
    if [ ! -f "$REPO/runs/.sft_complete_announced" ]; then
        say "training COMPLETE: consolidated model present in $OUT_DIR. Nothing further to do."
        touch "$REPO/runs/.sft_complete_announced"
    fi
    exit 0
fi
if grep -qs "SFT_RC=0" "$REPO"/runs/sft-*.out 2>/dev/null; then
    if [ ! -f "$REPO/runs/.sft_complete_announced" ]; then
        say "training COMPLETE: a job reported SFT_RC=0. Nothing further to do."
        touch "$REPO/runs/.sft_complete_announced"
    fi
    exit 0
fi

# ---- 2. Is a training job alive? --------------------------------------------
# ONE job, on amd-primus, and any state of it counts as covered -- including a long
# QOSGrpNodeLimit wait. That is a deliberate reversal of the previous rule, which
# treated a cap-blocked pending job as "not covered" and submitted to amd-burst
# alongside it.
#
# The old rule was written when burst was placing jobs in under a minute and the
# guaranteed pools were hours deep, so a queue position there was worth little. Burst
# is now actively harmful: it is the lowest-priority tier on a partition running 135
# alloc / 56 mix / 36 down-or-drained / 2 idle, and --exclusive can only place on a
# WHOLLY idle node. On a cluster this saturated, healthy nodes never sit idle -- they
# are taken instantly -- so the idle ones are idle because they are broken. Six burst
# placements in a row died that way (...301, ...297, ...296, ...331, ...291, ...317,
# four of them still DOWN afterwards), one of them before our code produced a single
# line of output. Zero progress was banked.
#
# Waiting in the primus queue for a real allocation is therefore the faster route to a
# trained model, even though it starts slower. Treating the wait as covered is what
# keeps this guard from piling on duplicates while we wait.
mapfile -t OUR_JOBS < <(squeue -u "${USER:-shasriva}" -h -o "%i %j %T %q" 2>/dev/null \
                        | awk '$2=="kore-sft"')
RUNNING_LINE="$(printf '%s\n' "${OUR_JOBS[@]:-}" | awk '$3=="RUNNING"{print; exit}')"
ANY_PENDING="$(printf '%s\n' "${OUR_JOBS[@]:-}" | awk 'NF{print; exit}')"
JOB_LINE="${RUNNING_LINE:-$ANY_PENDING}"
JOB_ID="$(awk '{print $1}' <<<"$JOB_LINE")"
JOB_STATE="$(awk '{print $3}' <<<"$JOB_LINE")"

# Distinguish "the controller did not answer" from "there is no job". Submitting
# during a controller outage burns the attempt budget for nothing.
if [ -z "$JOB_LINE" ] && ! squeue -u "${USER:-shasriva}" -h -o "%i" >/dev/null 2>&1; then
    say "controller unreachable; waiting rather than submitting"
    exit 0
fi

# A node failure is the failure mode actually observed here, twice: job 10849 died
# with NODE_FAIL/NodeDown and job 9229 was killed with no error, no traceback and no
# launcher epilogue, which is what a node going away looks like from inside the job.
# Nothing about it is recoverable by waiting, and it leaves the training lock behind
# holding a job id that no longer exists. The launcher takes over a lock whose holder
# has left the queue, so this only needs to be visible, not repaired.
if [ -z "$RUNNING_LINE" ] && [ -d "$OUT_DIR/.kore_train.lock" ]; then
    holder="$(cat "$OUT_DIR/.kore_train.lock/jobid" 2>/dev/null || true)"
    if [ -n "$holder" ] && ! squeue -j "$holder" -h -o "%T" >/dev/null 2>&1; then
        say "note: stale training lock from job $holder (gone from the queue); the launcher will take it over"
    fi
fi

# ---- 3. Ensure a supervisor is watching whatever is alive --------------------
supervisor_alive() { pgrep -f "watch_and_resume.sh sft" >/dev/null; }

start_supervisor() {
    # 9>&- closes the lock fd in the child. Without it the long-lived supervisor
    # inherits this guard's lock and holds it for days, after which every later
    # invocation decides a concurrent guard is running and does nothing, which is
    # far worse than the duplication the lock prevents.
    setsid nohup bash "$REPO/scripts/sft_supervise_v5.sh" \
        >> "$REPO/runs/cron_sft_supervise.log" 2>&1 9>&- &
    sleep 5
}

if [ -n "$JOB_ID" ]; then
    if supervisor_alive; then
        exit 0    # healthy: job alive, supervisor watching. The common case, silent.
    fi
    say "job $JOB_ID is $JOB_STATE but no supervisor was running; starting one (it adopts, never duplicates)"
    start_supervisor
    supervisor_alive && say "supervisor started and adopted job $JOB_ID" \
                     || say "WARNING supervisor failed to start"
    exit 0
fi

# ---- 4. No job at all: submit, within a bounded budget ----------------------
n="$(cat "$ATTEMPTS" 2>/dev/null || echo 0)"
case "$n" in ''|*[!0-9]*) n=0 ;; esac
if [ "$n" -ge "$MAX_ATTEMPTS" ]; then
    say "no job running, but $n submissions already made (cap $MAX_ATTEMPTS). Refusing to submit again; investigate rather than loop. Clear runs/.sft_submit_attempts to resume."
    exit 1
fi
echo $((n + 1)) > "$ATTEMPTS"

# No node exclusion list, deliberately. An earlier version harvested dirty and failed
# nodes and passed them as --exclude, and that was the wrong instinct on this cluster.
# Nodes here are cycled and repaired continuously: crsuse2-m2m-296 failed a job, went
# DOWN, and was back and running our next job cleanly about fifteen minutes later. The
# partition also runs at roughly four idle nodes out of 228, so refusing capacity that
# has already been fixed is the most expensive mistake available.
#
# Nothing is lost by dropping it. --exclusive is what actually keeps us off occupied
# GPUs, at the source, by only ever placing us on a wholly free node. The hygiene check
# in the launcher remains as the backstop for leaked memory, and it costs ten seconds
# to reject a bad node and try again. Retrying is cheaper than remembering.

# amd-primus, and only amd-primus. Note the account must match the QoS: pairing
# amd-general with amd-burst-qos is a phantom association that the controller accepts
# and never schedules.
say "no training job found; submitting on amd-primus (attempt $((n + 1)) of $MAX_ATTEMPTS)"
out="$(sbatch --account=amd-primus --qos=amd-primus-qos \
       "$REPO/scripts/spur_sft_1node.sbatch" \
       configs/sft_coder30b_a3b.json - - 2>&1)"
if grep -qE 'Submitted batch job [0-9]+' <<<"$out"; then
    say "submitted: $out"
    supervisor_alive || start_supervisor
else
    say "submit FAILED: $out"
fi
exit 0
