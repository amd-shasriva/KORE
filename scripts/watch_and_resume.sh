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
    # EXCLUDE is applied here. It used to be assigned when a node wedged us and then
    # never read, so the retry could land straight back on the bad node -- which is
    # the specific thing it exists to prevent, since a prior tenant's leftover GPU
    # memory survives --exclusive.
    if [ -n "$EXCLUDE" ]; then
        log "excluding node(s): $EXCLUDE"
        out="$("${SUBMIT[@]}" --exclude="$EXCLUDE" 2>&1 | tail -3)"
    else
        out="$("${SUBMIT[@]}" 2>&1 | tail -3)"
    fi
    # --parsable prints a bare id; without it, "Submitted batch job <id>".
    id="$(printf '%s' "$out" | grep -oE '[0-9]{4,}' | tail -1)"
    [ -n "$id" ] || { log "submit produced no job id: $out"; return 1; }
    printf '%s' "$id"
}

# Did the TRAINING finish, as opposed to the job merely leaving the queue?
#
# This must not rest on Slurm accounting alone. The launcher prints SFT_RC=0 only
# after the trainer returns and the model is saved, so it is the one signal that
# means "the work is done" rather than "the scheduler stopped telling us about it".
# Without an authoritative completion test, a supervisor whose accounting lookup
# comes back empty will happily resubmit a FINISHED run until MAX_RESUBMITS, each
# time reloading 61GB and retraining from the last checkpoint -- days of wasted GPU
# on a job that already succeeded, unattended.
run_completed() {
    # Run-SPECIFIC and authoritative: a finished run leaves a consolidated model at
    # the top of output_dir, not just checkpoint-* subdirectories. Checked first
    # because output_dir names the run (…_v5), so it cannot be satisfied by some
    # earlier run's artifacts.
    if [ -n "${KORE_OUTPUT_DIR:-}" ] \
        && [ -s "$KORE_OUTPUT_DIR/config.json" ] \
        && ls "$KORE_OUTPUT_DIR"/*.safetensors >/dev/null 2>&1; then
        return 0
    fi
    # Then the launcher's own sentinel, but ONLY for jobs this supervisor started.
    # Globbing every runs/sft-*.out would let a PREVIOUS run's success declare this
    # one finished before it has trained a single step -- harmless today because no
    # such log exists, and a silent no-op launch the first time one does.
    local j
    for j in $SEEN_JOBS; do
        grep -qs "SFT_RC=0" "$REPO/runs/sft-${j}.out" 2>/dev/null && return 0
    done
    return 1
}

# Terminal state, preferring scontrol. On this controller sacct can come back empty
# while `scontrol show job` still reports JobState and ExitCode, and treating an
# empty answer as "preempted" is what turns a crash-loop into 40 resubmissions.
terminal_state() {
    local st
    st="$(scontrol show job "$1" 2>/dev/null | tr ' ' '\n' \
          | grep -m1 '^JobState=' | cut -d= -f2 || true)"
    [ -z "$st" ] && st="$(sacct -j "$1" --format=State -n 2>/dev/null | head -1 | tr -d ' ' || true)"
    printf '%s' "$st"
}
terminal_exit() {
    local rc
    rc="$(scontrol show job "$1" 2>/dev/null | tr ' ' '\n' \
          | grep -m1 '^ExitCode=' | cut -d= -f2 || true)"
    [ -z "$rc" ] && rc="$(sacct -j "$1" --format=ExitCode -n 2>/dev/null | head -1 | tr -d ' ' || true)"
    printf '%s' "$rc"
}

resubmits=0
JOB=""
# Seeded from the persisted list so a restarted supervisor does not have to
# rediscover, one wasted submission at a time, every dirty node it already found.
EXCLUDE="$(paste -sd, "$REPO/runs/.sft_bad_nodes" 2>/dev/null || true)"
LAST_START=""
consecutive_fast_failures=0
launch_failures=0
PENDING_SINCE=""
#: Job ids this supervisor has started or adopted, so run_completed cannot be
#: satisfied by an unrelated run's log.
SEEN_JOBS=""

# Adopt a job that is already in flight rather than starting a second one. Two
# concurrent runs would share output_dir and interleave checkpoint writes into each
# other, which corrupts the only thing that makes preemption survivable. This makes
# the supervisor safe to start after a manual launch, which is how it will usually
# be used.
existing="$(squeue -u "${USER:-$(id -un)}" -h -o '%i %j %T' 2>/dev/null \
            | awk -v n="kore-${STAGE}" '$2==n && ($3=="RUNNING" || $3=="PENDING"){print $1}' \
            | tail -1 || true)"
if [ -n "$existing" ]; then
    JOB="$existing"
    SEEN_JOBS="$SEEN_JOBS $existing"
    LAST_START="$(date +%s)"
    log "adopting in-flight job=$JOB rather than submitting a duplicate"
fi

while :; do
    if [ -z "$JOB" ]; then
        if run_completed; then
            log "training already reported completion; nothing to supervise"
            exit 0
        fi
        if [ "$resubmits" -ge "$MAX_RESUBMITS" ]; then
            log "hit MAX_RESUBMITS=$MAX_RESUBMITS; stopping"
            exit 1
        fi
        JOB="$(submit)" || { sleep "$POLL_SECS"; continue; }
        resubmits=$((resubmits + 1))
        SEEN_JOBS="$SEEN_JOBS $JOB"
        LAST_START="$(date +%s)"
        log "submitted job=$JOB (attempt $resubmits)"
        sleep 15
    fi

    state="$(job_state "$JOB")"
    [ -z "$state" ] && state="$(scontrol show job "$JOB" 2>/dev/null | tr ' ' '\n' \
        | grep -m1 '^JobState=' | cut -d= -f2 | grep -E 'RUNNING|PENDING' || true)"

    case "$state" in
        RUNNING)
            launch_failures=0
            PENDING_SINCE=""
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
                *JobLaunchFailure*|*dispatch*)
                    # The controller accepted the job, tried to dispatch it, and got
                    # no confirmation from the node. This does NOT recover on its own
                    # -- job 6530 sat in it unchanged across three poll cycles, and
                    # this repo has seen a submission wedge here before. `scontrol
                    # release` is the wrong tool (nothing is held); the job has to be
                    # cancelled and a fresh one submitted.
                    launch_failures=$((launch_failures + 1))
                    log "job=$JOB failed to dispatch ($reason);" \
                        "cancelling and resubmitting (launch failure $launch_failures)"
                    scancel "$JOB" 2>/dev/null
                    JOB=""
                    # Retry FAST, with only a mild escalation and a low cap.
                    #
                    # This deliberately does not back off far, because the failure is
                    # random rather than persistent. Measured against identical
                    # single-purpose probes: `--gres=gpu:mi355x:8` alone both
                    # succeeded and failed minutes apart, and gres+cpu128 succeeded
                    # while gres+cpu32, gres+exclusive and gres+time7d all failed --
                    # no flag combination predicts it, and roughly half of GPU
                    # dispatches were failing. Each attempt is therefore an
                    # independent coin flip, so the expected wait is minimised by
                    # flipping often; escalating to 15 minutes would just add idle
                    # time to a 50/50 draw. Attempts are cheap: a failed dispatch
                    # never starts the job, so it costs nothing but a queue slot.
                    backoff=$(( 30 + launch_failures * 15 ))
                    (( backoff > 120 )) && backoff=120
                    log "backing off ${backoff}s before the next attempt"
                    sleep "$backoff"
                    ;;
                *)
                    # A job can sit in PENDING with Reason=None and StartTime=N/A
                    # indefinitely -- job 6684 did it for 1h48m while other users'
                    # jobs started around it. The scheduler is not saying it cannot
                    # run us; it is saying nothing at all, and waiting on that is
                    # indistinguishable from being wedged. Fresh submissions DO get
                    # picked up (6620 started on its second attempt), so past a
                    # threshold, resubmitting is strictly better than waiting.
                    #
                    # The threshold has to exceed a legitimate queue wait for an
                    # exclusive whole-node request, hence 25 minutes rather than a
                    # couple of poll cycles.
                    pending_secs=$(( $(date +%s) - ${PENDING_SINCE:-$(date +%s)} ))
                    if [ -z "${PENDING_SINCE:-}" ]; then
                        PENDING_SINCE="$(date +%s)"
                        pending_secs=0
                    fi
                    # DISABLED BY DEFAULT (STUCK_PENDING_SECS=0), and that is the
                    # important part. Queue position is by submit time at equal
                    # priority, so cancelling and resubmitting a merely-waiting job
                    # sends it to the BACK of the queue. On a full cluster that is
                    # strictly worse than waiting, however long the wait looks.
                    #
                    # My first version of this fired after 25 minutes, on the theory
                    # that job 6684 sitting at PENDING(None) for 1h48m was wedged. It
                    # was not wedged, it was starved: the request is --exclusive, and a
                    # whole-node request cannot start while zero nodes are fully idle,
                    # which was exactly the cluster state (137 allocated, 54 partially
                    # allocated, 0 idle). Resubmitting would have surrendered a
                    # position earned at 01:29 for nothing.
                    #
                    # Set STUCK_PENDING_SECS explicitly if you ever have evidence of a
                    # genuinely wedged pending job -- but note that JobLaunchFailure,
                    # the one wedge actually observed here, is handled above and does
                    # need a resubmit because such a job holds no useful position.
                    if [ "${STUCK_PENDING_SECS:-0}" -le 0 ]; then
                        log "job=$JOB pending ($reason) ${pending_secs}s;" \
                            "holding queue position"
                        sleep "$POLL_SECS"
                    elif printf '%s' "$reason" | grep -q QOSGrpNodeLimit; then
                        # A full QoS cap is a real capacity answer, not a wedge.
                        log "job=$JOB pending on a full QoS cap (${pending_secs}s);" \
                            "holding queue position"
                        sleep "$POLL_SECS"
                    elif [ "$pending_secs" -gt "$STUCK_PENDING_SECS" ]; then
                        log "job=$JOB has been PENDING ($reason) for ${pending_secs}s" \
                            "with no start time -- cancelling and resubmitting"
                        scancel "$JOB" 2>/dev/null
                        JOB=""
                        PENDING_SINCE=""
                        sleep 20
                    else
                        log "job=$JOB pending ($reason) for ${pending_secs}s"
                        sleep "$POLL_SECS"
                    fi
                    ;;
            esac
            ;;
        "")
            # Not in the queue: finished, preempted, or the controller is down.
            # Those must not be conflated -- resubmitting during a controller
            # outage just burns attempts.
            if ! squeue -u "${USER:-${LOGNAME:-$(id -un)}}" -h -o '%i' >/dev/null 2>&1; then
                log "controller unreachable; waiting rather than resubmitting"
                sleep "$POLL_SECS"
                continue
            fi
            rc="$(terminal_exit "$JOB")"
            st="$(terminal_state "$JOB")"
            log "job=$JOB left the queue state=${st:-?} exit=${rc:-?}"
            # A JOB THAT CYCLED TO A NEW GENERATION is not a failure and not a
            # completion. At its walltime boundary the launcher submits a fresh job
            # and exits 0 WITHOUT ever reaching the SFT_RC=0 sentinel (that line is
            # only reached after the trainer itself returns), so to Slurm this job's
            # exit is indistinguishable from a genuinely finished one -- both report
            # JobState=COMPLETED. Checked before both the run_completed and the
            # "completed" fallback below, or the fallback would declare the whole
            # stage done and stop supervising the generation that was just
            # submitted, orphaning it mid-training. Mirrors the KORE_LOCK_HELD
            # handling: follow the new job rather than resubmitting or exiting.
            resubmitted="$(grep -m1 -oE 'SFT_RESUBMITTED=[0-9]+' "$(newest_log "$JOB")" \
                           2>/dev/null | cut -d= -f2 || true)"
            if [ -n "$resubmitted" ]; then
                log "job=$JOB reached its walltime and resubmitted as $resubmitted;" \
                    "following $resubmitted"
                JOB="$resubmitted"
                SEEN_JOBS="$SEEN_JOBS $resubmitted"
                PENDING_SINCE=""
                sleep "$POLL_SECS"
                continue
            fi
            # Ask the WORK whether it is done before asking the scheduler. This is
            # the check that makes unattended operation safe.
            if run_completed; then
                log "training reported SFT_RC=0 (or a consolidated model exists); done"
                exit 0
            fi
            if printf '%s' "$st" | grep -qi completed; then
                log "$STAGE completed; done"
                exit 0
            fi
            # A job that FAILED immediately is a bug, not a preemption. Resubmitting
            # a crash on a loop burns the attempt budget and hides the error, so
            # require that the run actually got somewhere before treating it as
            # interrupted. Job 6520 died in 12 seconds on a shell bug; a supervisor
            # without this would have retried it 40 times.
            # A DIRTY NODE is not a code bug, and must not consume the crash-loop
            # budget. The launcher exits 3 after printing KORE_BAD_NODE=<host> when a
            # previous tenant left memory allocated on a card (measured once at
            # ~270GB), which would otherwise wedge training after the model load.
            # Land on three such nodes in a row and a fast-failure counter would halt
            # the supervisor over a cluster-hygiene problem. Instead: exclude that
            # node, retry immediately, and do not count it.
            # A LOCK LOSER is not a failure. We keep a job queued in both the general
            # and primus pools so whichever frees first is taken; when the second one
            # eventually starts it finds the first already training, prints
            # KORE_LOCK_HELD=<jobid> and exits 0 without touching anything. Resubmitting
            # it would be pointless churn, and treating it as a completion would abandon
            # the run. Follow the winner instead.
            heldby="$(grep -m1 -oE 'KORE_LOCK_HELD=[0-9]+' "$(newest_log "$JOB")" 2>/dev/null \
                      | cut -d= -f2 || true)"
            if [ -n "$heldby" ]; then
                log "job=$JOB exited because job $heldby already holds the training lock;" \
                    "following $heldby instead"
                JOB="$heldby"
                SEEN_JOBS="$SEEN_JOBS $heldby"
                PENDING_SINCE=""
                sleep "$POLL_SECS"
                continue
            fi

            badnode="$(grep -m1 -oE 'KORE_BAD_NODE=[^ ]+' "$(newest_log "$JOB")" 2>/dev/null \
                       | cut -d= -f2 || true)"
            if [ -n "$badnode" ]; then
                # Persist as well as remember. An in-memory list dies with this
                # process, and the cron guard restarts this process, so a node found
                # dirty here would otherwise be retried freely a few minutes later.
                # The guard reads the same file and passes the same --exclude.
                printf '%s\n' "$badnode" >> "$REPO/runs/.sft_bad_nodes" 2>/dev/null || true
                sort -u -o "$REPO/runs/.sft_bad_nodes" "$REPO/runs/.sft_bad_nodes" 2>/dev/null || true
                EXCLUDE="$(paste -sd, "$REPO/runs/.sft_bad_nodes" 2>/dev/null \
                           || printf '%s' "${EXCLUDE:+$EXCLUDE,}$badnode")"
                log "job=$JOB landed on a dirty node ($badnode) with memory already" \
                    "allocated on a GPU; excluding it and resubmitting." \
                    "excluded so far: $EXCLUDE"
                JOB=""
                PENDING_SINCE=""
                sleep 15
                continue
            fi

            short_run=0
            if [ -n "$LAST_START" ]; then
                elapsed=$(( $(date +%s) - LAST_START ))
                [ "$elapsed" -lt "${MIN_PROGRESS_SECS:-300}" ] && short_run=1
            fi
            if printf '%s' "$st" | grep -qiE 'fail|cancel' && [ "$short_run" = 1 ]; then
                consecutive_fast_failures=$((consecutive_fast_failures + 1))
                log "job=$JOB ended ${elapsed:-?}s after starting with state=$st" \
                    "-- that is a crash, not a preemption" \
                    "(consecutive: $consecutive_fast_failures)"
                if [ "$consecutive_fast_failures" -ge "${MAX_FAST_FAILURES:-3}" ]; then
                    log "STOPPING: $consecutive_fast_failures consecutive fast failures." \
                        "Fix the error rather than retrying. Last log:"
                    tail -25 "$(newest_log "$JOB")" 2>/dev/null | sed 's/^/    /'
                    exit 1
                fi
            else
                consecutive_fast_failures=0
            fi
            log "treating as interruption; resubmitting to resume"
            JOB=""
            sleep 20
            ;;
        *)
            log "job=$JOB state=$state"
            sleep "$POLL_SECS"
            ;;
    esac
done
