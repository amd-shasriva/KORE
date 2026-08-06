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

_squeue() { squeue -u "$USER" -h "$@" 2>/dev/null; }

# Held jobs occupy the cap without running. Nothing recovers them -- the
# scheduler has already given up -- so the only useful action is to remove them
# and let the loop resubmit deliberately.
# Only genuinely held jobs. JobLaunchFailure looks similar but is transient --
# the scheduler retries it, and jobs do recover: a gate sat in JobLaunchFailure
# and then ran for half an hour. Cancelling on sight made this loop destroy its
# own work, submitting a datagen sweep and killing it seconds later, forever.
purge_held() {
    local n=0 j
    for j in $(_squeue -t PD -o "%i %R" | grep -i "hold" | awk '{print $1}'); do
        scancel "$j" 2>/dev/null && n=$((n + 1))
    done
    [ "$n" -gt 0 ] && echo "purged $n held job(s)"
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
