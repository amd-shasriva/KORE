#!/bin/bash
# Keep several datagen streams staffed at once, within the QoS job cap.
#
# One stream is not enough for the dataset we need. Mining only pool-HIP grows HIP
# volume but leaves two gaps that no amount of it closes:
#
#   pool-Triton   the 13,570 KernelBook tasks have never been mined for Triton, so
#                 all Triton data is registry ops. Mining the same task ids we mine
#                 for HIP is also the only way translation pairs ever appear: the
#                 reshaper emits a dialect-to-dialect row only where one op won in
#                 both backends.
#   registry-HIP  the 171 unmined registry tasks. HIP wins today are 7 elementwise
#                 ops while hip2hip scores 38%, so this is the quality gap rather
#                 than the volume one.
#
# Streams are declared with a share of the available slots rather than a fixed
# count, so the split holds whether the cap grants us three slots or eight.
#
#   scripts/staff_datagen.sh
set -uo pipefail

REPO=/home/shasriva/Kore-RL/KORE
cd "$REPO" || exit 1
[ -z "${SPUR_CONTROLLER_ADDR:-}" ] && [ -r /etc/profile.d/spur.sh ] && . /etc/profile.d/spur.sh
# shellcheck disable=SC1091
. "$REPO/scripts/gpu_slots.sh"

LOG="$REPO/runs/staff_datagen.log"
say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$LOG"; }

#: name | shard dir | data root | wanted elements | job names that serve it
#:
#: The name list matters: jobs submitted before per-stream naming existed are still
#: called kore-factory and are still mining pool-HIP. Not counting them made the
#: stream look understaffed, and the top-up landed a second worker on a shard
#: another job was already grinding through.
#: Wanted counts sum to one less than the job cap, leaving a slot for the arena --
#: which the supervisor owns, not this script. Sizing them to the cap instead left
#: the arena unable to launch; sizing them below it left slots idle, which on a pool
#: this contended is the more expensive mistake.
STREAMS="${DATAGEN_STREAMS:-\
poolhip:runs/shards_hippool:data/v5hippool:4:kore-mine-poolhip+kore-factory \
pooltriton:runs/shards_pooltriton:data/v5pooltriton:2:kore-mine-pooltriton \
hipreg:runs/shards_hipreg:data/v5hip:1:kore-mine-hipreg}"

# Count by job name, which the scheduler knows for a job the moment it is
# submitted. Counting by reading each job's log missed every pending job, because
# a job that has not started has written nothing -- so a stream already fully
# queued looked empty and got submitted again on the next pass.
#
# Queue position is expensive here: burst runs 35+ jobs deep, so a job cancelled
# and resubmitted goes to the back. Nothing in this script cancels; it only tops a
# stream up, and only when the scheduler says that stream is genuinely short.
staffed_for() { _squeue -t R,PD -n "kore-mine-$1" -o "%i" | wc -l; }

# A shard manifest records the commit it was partitioned at, and the worker refuses
# to mine a shard whose code has moved -- a deliberate guard, but it means every
# commit invalidates every manifest. Left unhandled, submissions die instantly with
# NonZeroExitCode and the stream looks like it is merely waiting in the queue: that
# is exactly how pool-Triton produced nothing for an entire afternoon while three
# fixes landed on top of its manifest.
# Delegate to Python. This was shell, and the shell was the bug: reading the
# manifest with a here-document nested inside a command substitution mis-parsed
# twice -- once shifting fields so n_shards came back wrong, once failing outright
# with "here-document delimited by end-of-file" -- and both times rebuilt a
# seven-shard layout as one shard, making every array index above 0 illegal.
# scripts/refresh_shards.py parses JSON with a JSON parser and refuses to
# re-partition on values that cannot be right.
refresh_if_stale() {
    "${KORE_PY:-/home/shasriva/kore-venv/bin/python}" \
        "$REPO/scripts/refresh_shards.py" "$1" >> "$LOG" 2>&1
}

#: Shard indices a stream already has covered, read from each worker's own startup
#: line. The scheduler cannot answer this -- it reports "?" for the array index and
#: exposes neither ArrayTaskId nor the submit command -- so the worker's log is the
#: only source of truth for which shard a job is actually on.
covered_shards() {
    local names="$1" j nm
    for j in $(_squeue -t R,PD -o "%i"); do
        nm=$(_squeue -j "$j" -o "%j" | head -1)
        case " $names " in *" $nm "*) ;; *) continue ;; esac
        grep -aoE "START job=$j array=[0-9]+ .*shard=[0-9]+" \
            "$REPO/runs/spur-$j.out" 2>/dev/null | tail -1 |
            grep -oE "shard=[0-9]+" | cut -d= -f2 | sed 's/^0*//;s/^$/0/'
    done | sort -un
}

for spec in $STREAMS; do
    IFS=: read -r name dir root want names <<< "$spec"
    names="${names//+/ }"
    [ -d "$REPO/$dir" ] || { say "$name: no shard dir $dir; skipping"; continue; }
    refresh_if_stale "$dir"

    nsh=$("${KORE_PY:-/home/shasriva/kore-venv/bin/python}" -c "
import json;print(json.load(open('$REPO/$dir/manifest.json')).get('n_shards',0))" 2>/dev/null)
    case "$nsh" in ''|*[!0-9]*) say "$name: n_shards unreadable; skipping"; continue ;; esac
    [ "$nsh" -lt 1 ] && { say "$name: n_shards=$nsh; skipping"; continue; }

    have=$(_squeue -t R,PD -n "${names%% *}" -o "%i" | wc -l)
    for extra in ${names#* }; do
        [ "$extra" = "$names" ] && break
        have=$(( have + $(_squeue -t R,PD -n "$extra" -o "%i" | wc -l) ))
    done
    free=$(gpu_free)
    need=$(( want - have )); [ "$want" -gt "$nsh" ] && need=$(( nsh - have ))
    [ "$need" -gt "$free" ] && need=$free
    [ "$need" -le 0 ] && continue

    # Pick shard indices nobody is on. Submitting by count assumed shards 0..have-1
    # were covered, which is false as soon as any job sits on a different index --
    # two workers then raced through the same 922 tasks while four shards went
    # untouched.
    busy=" $(covered_shards "$names" | tr '\n' ' ') "
    picked=""
    i=0
    while [ "$i" -lt "$nsh" ] && [ "$(printf '%s' "$picked" | wc -w)" -lt "$need" ]; do
        case "$busy" in *" $i "*) ;; *) picked="$picked $i" ;; esac
        i=$(( i + 1 ))
    done
    [ -z "${picked// /}" ] && { say "$name: all $nsh shard(s) already covered"; continue; }

    say "$name: $have/$want staffed (shards busy:${busy%  }), $free free -> adding shard(s)$picked"
    for idx in $picked; do
        # One element per submission: a range would re-queue indices that are
        # already covered, and there is no way to express a gap in an array range.
        # shellcheck disable=SC2086
        sbatch $QOS_ARG --job-name="kore-mine-$name" --array="$idx-$idx" \
            scripts/spur_datagen_array.sbatch \
            "$REPO/$dir" "$REPO/$root" 3 run >> "$LOG" 2>&1
        sleep 3
    done
done
exit 0
