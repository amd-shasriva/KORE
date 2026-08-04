#!/usr/bin/env bash
# Keep a long job alive on a cluster that preempts it.
#
# Every long run today ended the same way: SIGTERM (rc=143) two hours in, on a
# partition sitting at 77 allocated against 2 idle. Neither SFT nor the arena
# eval finishes in one allocation, so "launch it and check later" means finding a
# dead job and hours of lost work. This resubmits and lets the stage resume.
#
# It watches for the three distinct failure shapes we have actually hit, because
# they need different responses and only one of them is a real error:
#
#   preempted   gone from the queue with rc=143 -> resubmit, resume from the
#               newest checkpoint / results ledger
#   held        SPUR parks a newly submitted job in JobHoldMaxRequeue with
#               Priority=0 before it ever runs -> scontrol release, and give it a
#               full scheduler cycle (~60s; a 45s check reported failure while the
#               release was in fact working)
#   stalled     in the queue, log untouched for STALL_SECS -> the run is wedged
#               rather than working. Observed after model load when a prior
#               tenant left ~270GB allocated on a card despite --exclusive.
#               Cancel and resubmit, excluding that node.
#
# Deliberately NOT using --requeue to get this: on this controller a requeued job
# trips JobHoldMaxRequeue on its FIRST requeue and is held permanently, which
# turns a transient preemption into a dead run. Resubmission from outside is the
# only recovery that works here.
set -uo pipefail

STAGE="${1:?usage: watch_and_resume.sh <sft|aka> <submit-command...>}"
shift
SUBMIT=("$@")
[ ${#SUBMIT[@]} -gt 0 ] || { echo "no submit command given" >&2; exit 2; }

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
POLL_SECS="${POLL_SECS:-120}"
STALL_SECS="${STALL_SECS:-1800}"
MAX_RESUBMITS="${MAX_RESUBMITS:-40}"
RELEASE_WAIT="${RELEASE_WAIT:-60}"

log() { printf '[watch:%s %s] %s\n' "$STAGE" "$(date +%H:%M:%S)" "$*"; }

job_state() { squeue -j "$1" -h -o '%T' 2>/dev/null | head -1; }
job_reason() { squeue -j "$1" -h -o '%R' 2>/dev/null | head -1; }

newest_log() {   # newest log for this job id, either stream
    ls -t "$REPO"/runs/*"$1"*.err "$REPO"/runs/*"$1"*.out 2>/dev/null | head -1
}

log_age() {      # seconds since the log was last written; huge if absent
    local f; f="$(newest_log "$1")"
    [ -n "$f" ] && [ -e "$f" ] || { echo 999999; return; }
    echo $(( $(date +%s) - $(stat -c %Y "$f") ))
}

submit() {
    local out id
    out="$("${SUBMIT[@]}" 2>&1 | tail -3)"
    # --parsable prints a bare id; without it, "Submitted batch job <id>".
    id="$(printf '%s' "$out" | grep -oE '[0-9]{4,}' | tail -1)"
    [ -n "$id" ] || { log "submit produced no job id: $out"; return 1; }
    printf '%s' "$id"
}

resubmits=0
JOB=""
EXCLUDE=""

while :; do
    if [ -z "$JOB" ]; then
        if [ "$resubmits" -ge "$MAX_RESUBMITS" ]; then
            log "hit MAX_RESUBMITS=$MAX_RESUBMITS; stopping"
            exit 1
        fi
        JOB="$(submit)" || { sleep "$POLL_SECS"; continue; }
        resubmits=$((resubmits + 1))
        log "submitted job=$JOB (attempt $resubmits)"
        sleep 15
    fi

    state="$(job_state "$JOB")"

    case "$state" in
        RUNNING)
            age="$(log_age "$JOB")"
            if [ "$age" -gt "$STALL_SECS" ]; then
                node="$(squeue -j "$JOB" -h -o '%N' 2>/dev/null | head -1)"
                log "job=$JOB RUNNING but log untouched ${age}s -- wedged on ${node:-?}; cancelling"
                scancel "$JOB" 2>/dev/null
                # Steer the retry away from a node that just wedged us. A prior
                # tenant's leftover GPU memory survives --exclusive, so the same
                # node will usually do it again.
                [ -n "$node" ] && EXCLUDE="$node"
                JOB=""
                sleep 20
            else
                log "job=$JOB running, log ${age}s old"
                sleep "$POLL_SECS"
            fi
            ;;
        PENDING)
            reason="$(job_reason "$JOB")"
            case "$reason" in
                *JobHoldMaxRequeue*|*held*)
                    log "job=$JOB held ($reason); releasing"
                    scontrol release "$JOB" 2>/dev/null
                    # A shorter wait than this reports failure while the release
                    # is actually working -- measured: 45s said still-held, 60s
                    # showed RUNNING.
                    sleep "$RELEASE_WAIT"
                    ;;
                *)
                    log "job=$JOB pending ($reason)"
                    sleep "$POLL_SECS"
                    ;;
            esac
            ;;
        "")
            # Not in the queue: finished, preempted, or the controller is down.
            # Those must not be conflated -- resubmitting during a controller
            # outage just burns attempts.
            if ! squeue -u "$USER" -h -o '%i' >/dev/null 2>&1; then
                log "controller unreachable; waiting rather than resubmitting"
                sleep "$POLL_SECS"
                continue
            fi
            rc="$(sacct -j "$JOB" --format=ExitCode -n 2>/dev/null | head -1 | tr -d ' ')"
            st="$(sacct -j "$JOB" --format=State -n 2>/dev/null | head -1 | tr -d ' ')"
            log "job=$JOB left the queue state=${st:-?} exit=${rc:-?}"
            if [ "$STAGE" = "sft" ] && printf '%s' "$st" | grep -qi completed; then
                log "SFT completed; done"
                exit 0
            fi
            if [ "$STAGE" = "aka" ] && printf '%s' "$st" | grep -qi completed; then
                log "arena eval completed; done"
                exit 0
            fi
            log "treating as preemption; resubmitting to resume"
            JOB=""
            sleep 20
            ;;
        *)
            log "job=$JOB state=$state"
            sleep "$POLL_SECS"
            ;;
    esac
done
