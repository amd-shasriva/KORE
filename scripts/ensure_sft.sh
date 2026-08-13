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

# ---- 2. Is a training job alive, and is it actually going to RUN? -----------
# "A job exists" is not the same as "the run is covered". amd-primus-qos and
# amd-general-qos are capped at 16 and 8 nodes team-wide and are routinely full, so
# a job of ours can sit at QOSGrpNodeLimit for many hours. Treating that as covered
# is how the run stayed down: two cap-blocked jobs were queued, no job was running,
# and this guard stayed silent because it found a "kore-sft" job in the list.
#
# So: a RUNNING job means covered. Otherwise, a pending job only counts as covered
# if it is on a pool with headroom (amd-burst), because that is the one that
# actually gets placed. A duplicate submission is safe regardless -- the launcher's
# training lock guarantees exactly one job trains, and any later starter exits
# after printing KORE_LOCK_HELD -- so the cost of acting is a wasted queue slot,
# while the cost of not acting is the run sitting idle.
mapfile -t OUR_JOBS < <(squeue -u "${USER:-shasriva}" -h -o "%i %j %T %q" 2>/dev/null \
                        | awk '$2=="kore-sft"')
RUNNING_LINE="$(printf '%s\n' "${OUR_JOBS[@]:-}" | awk '$3=="RUNNING"{print; exit}')"
BURST_PENDING="$(printf '%s\n' "${OUR_JOBS[@]:-}" | awk '$4=="amd-burst-qos"{print; exit}')"
JOB_LINE="${RUNNING_LINE:-$BURST_PENDING}"
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

# Harvest dirty nodes before resubmitting. The launcher's GPU-hygiene check exits 3
# after printing KORE_BAD_NODE=<host> when it finds a card already occupied, and this
# guard is usually the thing that submitted the job that just died, so if it ignores
# that signal it will keep re-rolling the same saturated pool. It did: 10903 and 10923
# were both submitted from here and both landed on full nodes within seconds.
#
# The list is a FILE rather than a shell variable because the supervisor's in-memory
# copy is lost every time it restarts, and this guard exists precisely to restart it.
# Both writers append here and both readers pass it as --exclude, so a node found
# dirty by either one stays excluded for everyone across restarts and reboots.
BAD_NODES="$REPO/runs/.sft_bad_nodes"
for log in $(ls -t "$REPO"/runs/sft-*.err 2>/dev/null | head -12); do
    grep -hoE 'KORE_BAD_NODE=[^ ]+' "$log" 2>/dev/null | cut -d= -f2
done | sort -u >> "$BAD_NODES".tmp 2>/dev/null || true

# Also harvest nodes that died UNDER us. A dirty node reports itself, because the
# hygiene check runs and prints KORE_BAD_NODE before exiting 3. A node that fails
# outright cannot report anything -- it is gone -- so nothing would ever exclude it
# and the scheduler is free to hand us the same dying hardware repeatedly. It did:
# 10849 on ...301, 10886 on ...297, 10942 on ...296, three adjacent nodes, with 296
# left in state DOWN afterwards. Ask Slurm which of our jobs ended in NODE_FAIL and
# treat those nodes exactly like dirty ones.
for jid in $(ls -t "$REPO"/runs/sft-*.err 2>/dev/null | head -15 \
             | sed 's#.*/sft-\([0-9]\+\)\.err#\1#'); do
    info="$(scontrol show job "$jid" 2>/dev/null | tr ' ' '\n')"
    case "$(printf '%s' "$info" | grep -m1 '^JobState=')" in
        JobState=NODE_FAIL|JobState=BOOT_FAIL)
            printf '%s' "$info" | grep -m1 '^NodeList=' | cut -d= -f2 \
                | xargs -r scontrol show hostnames 2>/dev/null ;;
    esac
done | grep -E '^[a-z0-9-]+$' | sort -u >> "$BAD_NODES".tmp 2>/dev/null || true
if [ -s "$BAD_NODES".tmp ]; then
    cat "$BAD_NODES" "$BAD_NODES".tmp 2>/dev/null | sort -u | grep -E '^[a-z0-9-]+$' > "$BAD_NODES".new || true
    mv -f "$BAD_NODES".new "$BAD_NODES" 2>/dev/null || true
fi
rm -f "$BAD_NODES".tmp

# Bounded, because an exclude list that only grows eventually excludes the cluster and
# the run stops being schedulable for a reason nobody is looking at. Nodes are also
# repaired -- 18 sat drained for a "Bundle 2 upgrade" and came back -- so a permanent
# blacklist would keep discarding capacity that is healthy again. Keep the 40 most
# recent findings and let anything older be retried; the hygiene check is the backstop
# that makes retrying a once-bad node safe.
if [ "$(wc -l < "$BAD_NODES" 2>/dev/null || echo 0)" -gt 40 ]; then
    tail -40 "$BAD_NODES" > "$BAD_NODES".new && mv -f "$BAD_NODES".new "$BAD_NODES"
fi
EXCLUDE="$(paste -sd, "$BAD_NODES" 2>/dev/null || true)"

# amd-burst first: it is the pairing that actually places this job. amd-primus and
# amd-general are both capped and were full every time this was needed, and
# amd-general+amd-burst-qos is a phantom association that is accepted and never
# scheduled. The supervisor keeps whatever lands.
say "no training job found; submitting on amd-burst (attempt $((n + 1)) of $MAX_ATTEMPTS)${EXCLUDE:+, excluding dirty nodes: $EXCLUDE}"
out="$(sbatch --account=amd-burst --qos=amd-burst-qos \
       ${EXCLUDE:+--exclude="$EXCLUDE"} \
       "$REPO/scripts/spur_sft_1node.sbatch" \
       configs/sft_coder30b_a3b.json - - 2>&1)"
if grep -qE 'Submitted batch job [0-9]+' <<<"$out"; then
    say "submitted: $out"
    supervisor_alive || start_supervisor
else
    say "submit FAILED: $out"
fi
exit 0
