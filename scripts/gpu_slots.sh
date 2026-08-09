# Shared GPU-slot accounting for the unattended loops. Source, don't execute.
#
# This scheduler allows a small number of concurrent GPU jobs per user (4 here).
# Submitting past the cap does not queue politely: the extra job is accepted, then
# fails to launch, is requeued, and after a few rounds lands in
# JobHoldMaxRequeue -- where it still counts against the cap. So a loop that
# resubmits on a timer digs its own hole. Held jobs pile up, and eventually
# nothing launches at all while `sinfo` still reports dozens of idle nodes, which
# reads as a full cluster rather than as self-inflicted queue debt.
#
# Two rules follow, and both loops need them:
#   purge_held   drop held jobs, because they hold slots while doing no work
#   gpu_free     ask before submitting, instead of submitting and hoping

# The ceiling is a QoS group limit, not a per-user job limit, which is why it
# looked like four: amd-general-qos caps the whole QoS at 8 nodes shared across
# users, and 6 were already taken. The idle nodes sinfo reports belong to other
# QoS pools and are unreachable from ours, so "47 nodes idle" and "cannot launch"
# were both true at once.
#
# amd-burst-qos is the large pool (90+ nodes) and caps each user at 8 running
# jobs. Every job here is requeue-safe and ledgered, so preemptible capacity is
# the right trade: a preempted shard resumes instead of being lost.
GPU_JOB_CAP="${GPU_JOB_CAP:-8}"

#: Passed to every sbatch. A command-line --qos overrides the script's own
#: directive, so the workhorse jobs move pools without editing 33 sbatch files.
KORE_QOS="${KORE_QOS:-amd-burst-qos}"
QOS_ARG="${QOS_ARG:---qos=$KORE_QOS}"

# Names that must never be purged or counted as expendable: losing training
# progress to make room for a gate would be a bad trade.
PROTECTED="${PROTECTED:-kore-sft}"

# $USER is not set in a cron environment, and these scripts run under set -u,
# so referencing it directly aborted the harvest mid-run.
KORE_USER="${USER:-${LOGNAME:-$(id -un)}}"
_squeue() { squeue -u "$KORE_USER" -h "$@" 2>/dev/null; }

# Held jobs occupy the cap without running. Nothing recovers them -- the
# scheduler has already given up -- so the only useful action is to remove them
# and let the loop resubmit deliberately.
# Recover held jobs rather than destroy them.
#
# The cycle a job actually goes through here is: (Priority) -> the scheduler picks
# a node -> the launch fails on that node -> requeue -> after a few rounds,
# JobHoldMaxRequeue. Cancelling at that point threw the job away, and the loop then
# resubmitted it to the back of a 25-deep queue. Nine miners died that way in one
# afternoon while burst was starting a job a minute.
#
# `scontrol release` un-holds the job so it queues again. Only a job this has
# already failed to rescue is cancelled, so a genuinely broken submission still
# cannot accumulate.
#: Reasons that mean the job is stuck rather than merely waiting its turn.
#:
#: "hold" alone missed JobLaunchFailure, which is the state a job sits in
#: between the launch failing on a node and the scheduler giving up on it. A
#: gate wedged there held one of the eight shared general-QoS nodes while doing
#: nothing, and because the reason never contained the word "hold" nothing here
#: looked at it. Waiting for it to age into JobHoldMaxRequeue works eventually
#: and costs a general slot the whole time.
_STUCK_REASON="${_STUCK_REASON:-hold|launchfail|joblaunchfailure}"
_HOLD_STATE="${_HOLD_STATE:-/tmp/kore_held_seen}"
purge_held() {
    local n_rel=0 n_kill=0 j seen
    for j in $(_squeue -t PD -o "%i %R" | grep -iE "$_STUCK_REASON" | awk '{print $1}'); do
        seen=$(grep -c "^$j\$" "$_HOLD_STATE" 2>/dev/null || echo 0)
        if [ "$seen" -ge 3 ]; then
            scancel "$j" 2>/dev/null && n_kill=$((n_kill + 1))
        else
            echo "$j" >> "$_HOLD_STATE"
            scontrol release "$j" >/dev/null 2>&1 && n_rel=$((n_rel + 1))
        fi
    done
    [ "$n_rel" -gt 0 ] && echo "released $n_rel held job(s) back to the queue"
    [ "$n_kill" -gt 0 ] && echo "cancelled $n_kill job(s) held repeatedly"
    return 0
}

# Every job I hold, running or pending. The cap counts jobs rather than GPUs, and
# a CPU-only job occupies one just as a training job does -- measured by adding a
# CPU-only seeding job and watching the next GPU submission fail to launch.
gpu_used() { _squeue -o "%i" | wc -l; }

# How many more GPU jobs may be submitted right now.
gpu_free() {
    local used free
    used=$(gpu_used)
    free=$(( GPU_JOB_CAP - used ))
    [ "$free" -lt 0 ] && free=0
    echo "$free"
}

# True when at least one slot is available.
have_slot() { [ "$(gpu_free)" -gt 0 ]; }

#: Which QoS to submit against, for one kind of job.
#:
#: general is a small shared pool -- 8 nodes across everyone using it -- and
#: burst is large but heavily oversubscribed: 114 running against 35 pending
#: when this was written. That is not a queue you wait in. A gate submitted to
#: burst sat for an hour behind other people's jobs while 15 nodes stood idle
#: and every other job I had was running happily on general.
#:
#: So take a general slot when the pool has room and I am not already holding
#: my share, and fall back to burst otherwise. The share is per kind of job and
#: passed in, because mining must not be able to spend the whole allowance: the
#: gate is what turns seeds into mineable tasks, so starving it stops the next
#: batch of training rows entirely, and mining a smaller set faster does not
#: make up for that.
GENERAL_QOS_CAP="${GENERAL_QOS_CAP:-8}"

pick_qos() {
    local prefix="$1" max="$2" used free mine
    used=$(squeue -t R -h -o "%q %D" 2>/dev/null |
           awk '$1=="amd-general-qos"{s+=$2} END{print s+0}')
    free=$(( GENERAL_QOS_CAP - used ))
    # Count queued as well as running: a submission that has not started yet
    # still intends to occupy a slot, and ignoring it lets one pass overshoot
    # the cap several times over before the first job appears as running.
    mine=$(_squeue -t R,PD -o "%q %j" |
           awk -v p="^$prefix" '$1=="amd-general-qos" && $2 ~ p {n++} END{print n+0}')
    if [ "$free" -gt 0 ] && [ "$mine" -lt "$max" ]; then
        echo "--qos=amd-general-qos"
    else
        echo "$QOS_ARG"
    fi
}

# The node hold to submit against, when one exists.
#
# The arena is the one thing that must not lose its node. It cannot finish inside
# an 8h allocation, so it crosses several, and on a cluster with zero idle nodes
# the gap between one job ending and its successor being submitted is enough for
# someone else to take the machine -- which is how a 413-task sweep ends up
# waiting hours between rollovers. A reservation on the node it already occupies
# closes that gap without taking anything from anyone else.
#
# A reservation on this controller excludes every job that does not ask for it,
# so the hold is only useful if the submission also requests it. That cuts both
# ways: requesting it restricts the job to reserved nodes, so this belongs on the
# arena (which wants exactly its own node back) and NOT on mining (which should
# land anywhere free).
#
# Empty when the reservation is absent or expired, so this fails open -- a lapsed
# hold degrades to ordinary scheduling instead of blocking every submission.
KORE_RESERVATION="${KORE_RESERVATION:-kore_hold}"
res_arg() {
    if scontrol show reservation 2>/dev/null |
       grep -q "ReservationName=${KORE_RESERVATION}$"; then
        echo "--reservation=${KORE_RESERVATION}"
    fi
}
