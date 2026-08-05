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

GPU_JOB_CAP="${GPU_JOB_CAP:-4}"

# Names that must never be purged or counted as expendable: losing training
# progress to make room for a gate would be a bad trade.
PROTECTED="${PROTECTED:-kore-sft}"

_squeue() { squeue -u "$USER" -h "$@" 2>/dev/null; }

# Held jobs occupy the cap without running. Nothing recovers them -- the
# scheduler has already given up -- so the only useful action is to remove them
# and let the loop resubmit deliberately.
purge_held() {
    local n=0 j
    for j in $(_squeue -t PD -o "%i %R" | grep -i "hold" | awk '{print $1}'); do
        scancel "$j" 2>/dev/null && n=$((n + 1))
    done
    # A job stuck in JobLaunchFailure is on its way to being held; drop it now
    # rather than after it has burned its requeues.
    for j in $(_squeue -t PD -o "%i %R" | grep -i "launchfailure" | awk '{print $1}'); do
        scancel "$j" 2>/dev/null && n=$((n + 1))
    done
    [ "$n" -gt 0 ] && echo "purged $n held/failing job(s)"
    return 0
}

# GPU jobs currently held by me, running or pending.
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
