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
# Any state that means the job still exists counts. Treating only RUNNING as
# alive would submit a duplicate against a job that is merely PENDING or
# CONFIGURING, and two trainers sharing output_dir interleave checkpoint writes.
JOB_LINE="$(squeue -u "${USER:-shasriva}" -h -o "%i %j %T" 2>/dev/null \
            | awk '$2=="kore-sft"{print; exit}')"
JOB_ID="$(awk '{print $1}' <<<"$JOB_LINE")"
JOB_STATE="$(awk '{print $3}' <<<"$JOB_LINE")"

# Distinguish "the controller did not answer" from "there is no job". Submitting
# during a controller outage burns the attempt budget for nothing.
if [ -z "$JOB_LINE" ] && ! squeue -u "${USER:-shasriva}" -h -o "%i" >/dev/null 2>&1; then
    say "controller unreachable; waiting rather than submitting"
    exit 0
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

# amd-burst first: it is the pairing that actually places this job. amd-primus and
# amd-general are both capped and were full every time this was needed, and
# amd-general+amd-burst-qos is a phantom association that is accepted and never
# scheduled. The supervisor keeps whatever lands.
say "no training job found; submitting on amd-burst (attempt $((n + 1)) of $MAX_ATTEMPTS)"
out="$(sbatch --account=amd-burst --qos=amd-burst-qos \
       "$REPO/scripts/spur_sft_1node.sbatch" \
       configs/sft_coder30b_a3b.json - - 2>&1)"
if grep -qE 'Submitted batch job [0-9]+' <<<"$out"; then
    say "submitted: $out"
    supervisor_alive || start_supervisor
else
    say "submit FAILED: $out"
fi
exit 0
