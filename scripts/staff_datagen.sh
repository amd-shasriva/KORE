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

#: name | shard dir | data root | wanted elements
STREAMS="${DATAGEN_STREAMS:-\
poolhip:runs/shards_hippool:data/v5hippool:3 \
pooltriton:runs/shards_pooltriton:data/v5pooltriton:3 \
hipreg:runs/shards_hipreg:data/v5hip:1}"

# Count by job name, which the scheduler knows for a job the moment it is
# submitted. Counting by reading each job's log missed every pending job, because
# a job that has not started has written nothing -- so a stream already fully
# queued looked empty and got submitted again on the next pass.
#
# Queue position is expensive here: burst runs 35+ jobs deep, so a job cancelled
# and resubmitted goes to the back. Nothing in this script cancels; it only tops a
# stream up, and only when the scheduler says that stream is genuinely short.
staffed_for() { _squeue -t R,PD -n "kore-mine-$1" -o "%i" | wc -l; }

for spec in $STREAMS; do
    IFS=: read -r name dir root want <<< "$spec"
    [ -d "$REPO/$dir" ] || { say "$name: no shard dir $dir; skipping"; continue; }
    have=$(staffed_for "$name")
    free=$(gpu_free)
    need=$(( want - have ))
    [ "$need" -gt "$free" ] && need=$free
    [ "$need" -le 0 ] && continue

    # Submit the missing elements by index, so a top-up covers the shards that are
    # actually unstaffed instead of re-queuing shard 0 and leaving the tail idle.
    lo="$have"; hi=$(( have + need - 1 ))
    say "$name: $have/$want staffed, $free slot(s) free -> adding shards $lo-$hi"
    # shellcheck disable=SC2086
    sbatch $QOS_ARG --job-name="kore-mine-$name" --array="$lo-$hi" \
        scripts/spur_datagen_array.sbatch \
        "$REPO/$dir" "$REPO/$root" 3 run >> "$LOG" 2>&1
    sleep 5
done
exit 0
