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

# One staffing pass at a time, across every caller.
#
# Deciding to submit is a read-then-write against the queue: gpu_free() and
# covered_shards() ask the scheduler what exists, and the submission happens
# afterwards. Nothing makes that atomic, so concurrent passes all read the same
# answer and all act on it. Ten copies of the pipeline loop did exactly that --
# each logged "1/3 staffed, 2 free -> adding shard(s) 0 2" in the same second,
# each submitted shard 0, slept 3s, submitted shard 2 -- and twenty jobs landed
# against an eight-job cap, on two shards, in three seconds.
#
# Holding the lock for the whole pass makes a second pass observe the first
# one's submissions rather than race them. Skipping rather than waiting is
# deliberate: a queued pass would submit against a queue snapshot that is
# already stale by the time it wakes, which is the bug again with extra steps.
exec 9>"$REPO/runs/.staff_datagen.lock"
if ! flock -n 9; then
    say "another staffing pass holds the lock; skipping this one"
    exit 0
fi

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
#: The default must match what ensure_loops.sh configures, because this script
#: is also run by hand -- and when it was left pointing at pool-HIP, a manual
#: pass silently staffed four miners onto the stream that had just been retired,
#: against shards whose ledger we were deliberately no longer growing.
STREAMS="${DATAGEN_STREAMS:-\
frontier:runs/shards_frontier:data/v5frontier:6:kore-mine-frontier \
pooltriton:runs/shards_pooltriton:data/v5pooltriton:0:kore-mine-pooltriton \
poolhip:runs/shards_hippool:data/v5hippool:0:kore-mine-poolhip+kore-factory \
hipreg:runs/shards_hipreg:data/v5hip:0:kore-mine-hipreg}"

# Count by job name, which the scheduler knows for a job the moment it is
# submitted. Counting by reading each job's log missed every pending job, because
# a job that has not started has written nothing -- so a stream already fully
# queued looked empty and got submitted again on the next pass.
#
# Queue position is expensive here: burst runs 35+ jobs deep, so a job cancelled
# and resubmitted goes to the back. Nothing in this script cancels; it only tops a
# stream up, and only when the scheduler says that stream is genuinely short.
staffed_for() { _squeue -t R,PD -n "kore-mine-$1" -o "%i" | wc -l; }

# Submit to the pool that can actually start the job.
#
# Every mining submission went to burst, because that is the default in
# gpu_slots.sh and nothing here overrode it. Burst is the large pool but it is
# persistently saturated -- 125 nodes running against 55 queued -- so a
# replacement worker for a stream that had just lost its node sat queued for
# hours, and the stream stayed dead the whole time. Both pool-Triton and
# registry-HIP died that way twice in one night.
#
# amd-general-qos is small, 8 nodes shared with every other user, and it is the
# only other pool this account may submit to (amd-primus-qos is refused
# outright: "not permitted for user under account amd-general"). So prefer it
# when it has headroom, and fall back to burst when it does not -- a queued
# burst job is still better than no job.
# Two limits, not one. Free capacity says whether a job *can* start there; the
# self-imposed cap says whether it *should*. General is 8 nodes for all 363
# users on this filesystem, and the two arena arms already sit there because the
# eval is the one thing whose progress has to be observable. Letting mining take
# every slot that happens to be free would leave nothing for anyone else, so
# mining is held to three and the remainder falls back to burst.
GENERAL_QOS_CAP="${GENERAL_QOS_CAP:-8}"
GENERAL_MINE_MAX="${GENERAL_MINE_MAX:-3}"
pick_qos() {
    local used free mine
    used=$(squeue -t R -h -o "%q %D" 2>/dev/null |
           awk '$1=="amd-general-qos"{s+=$2} END{print s+0}')
    free=$(( GENERAL_QOS_CAP - used ))
    # Count queued as well as running: a submission that has not started yet
    # still intends to occupy a slot, and ignoring it lets one pass overshoot
    # the cap several times over before the first job appears as running.
    mine=$(_squeue -t R,PD -o "%q %j" |
           awk '$1=="amd-general-qos" && $2 ~ /^kore-mine-/ {n++} END{print n+0}')
    if [ "$free" -gt 0 ] && [ "$mine" -lt "$GENERAL_MINE_MAX" ]; then
        echo "--qos=amd-general-qos"
    else
        echo "$QOS_ARG"
    fi
}

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

#: Shard indices a stream already has covered.
#:
#: Two sources, because neither alone is complete. A running worker prints the shard
#: it took, which is authoritative -- the scheduler cannot answer this at all, it
#: reports "?" for the array index and exposes neither ArrayTaskId nor the submit
#: command. But a job submitted seconds ago has printed nothing yet, and treating it
#: as uncovered puts the next pass on the same shard: that is exactly how two
#: pool-Triton workers both ended up on shard 000. So the index is also recorded at
#: submit time and believed for any job still in the queue.
_CLAIMS="${_CLAIMS:-$REPO/runs/shard_claims.tsv}"

claim_shard() { printf '%s\t%s\n' "$1" "$2" >> "$_CLAIMS"; }

covered_shards() {
    local names="$1" j nm live
    live=" $(_squeue -t R,PD -o "%i" | tr '\n' ' ') "
    {
        for j in $(_squeue -t R,PD -o "%i"); do
            nm=$(_squeue -j "$j" -o "%j" | head -1)
            case " $names " in *" $nm "*) ;; *) continue ;; esac
            grep -aoE "START job=$j array=[0-9]+ .*shard=[0-9]+" \
                "$REPO/runs/spur-$j.out" 2>/dev/null | tail -1 |
                grep -oE "shard=[0-9]+" | cut -d= -f2 | sed 's/^0*//;s/^$/0/'
        done
        # Claims made at submit time, kept only while the job is still queued so a
        # finished job stops reserving its shard.
        if [ -f "$_CLAIMS" ]; then
            while IFS=$'\t' read -r cj cs; do
                case "$live" in *" $cj "*) ;; *) continue ;; esac
                nm=$(_squeue -j "$cj" -o "%j" | head -1)
                case " $names " in *" $nm "*) echo "$cs" ;; esac
            done < "$_CLAIMS"
        fi
    } | sort -un
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
        qos_arg="$(pick_qos)"
        # shellcheck disable=SC2086
        out=$(sbatch $qos_arg --job-name="kore-mine-$name" --array="$idx-$idx" \
            scripts/spur_datagen_array.sbatch \
            "$REPO/$dir" "$REPO/$root" 3 run 2>&1)
        say "  shard $idx via ${qos_arg#--qos=}"
        echo "$out" >> "$LOG"
        jid=$(printf '%s' "$out" | grep -oE '[0-9]+$' | tail -1)
        # Claim the index immediately: the job will not print its own shard for a
        # minute or more, and until then the next pass would pick it again.
        [ -n "$jid" ] && claim_shard "$jid" "$idx"
        sleep 3
    done
done
exit 0
