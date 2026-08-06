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

# Every stream submits under the same job name, so per-stream counts come from the
# shard directory each running job was given rather than from the name.
running_for() {
    local dir="$1" n=0 j
    for j in $(_squeue -t R,PD -o "%i"); do
        grep -qs -- "$dir" "$REPO/runs/spur-$j.out" 2>/dev/null && n=$((n + 1))
    done
    echo "$n"
}

for spec in $STREAMS; do
    IFS=: read -r name dir root want <<< "$spec"
    [ -d "$REPO/$dir" ] || { say "$name: no shard dir $dir; skipping"; continue; }
    have=$(running_for "$dir")
    free=$(gpu_free)
    need=$(( want - have ))
    [ "$need" -gt "$free" ] && need=$free
    if [ "$need" -le 0 ]; then
        continue
    fi
    say "$name: $have/$want up, $free slot(s) free -> submitting $need"
    # shellcheck disable=SC2086
    sbatch $QOS_ARG --array=0-$((need - 1)) scripts/spur_datagen_array.sbatch \
        "$REPO/$dir" "$REPO/$root" 3 run >> "$LOG" 2>&1
    sleep 5
done
exit 0
