#!/bin/bash
# Keep the pool-HIP pipeline turning without a human in the loop.
#
# Three stages run at very different rates and only one needs a GPU:
#
#   seeding    teacher writes a naive HIP seed, ~19s each, 3,607 to do
#   gating     gfx950 says whether it compiles and is correct, minutes per batch
#   mining     datagen optimizes the gated seeds, hours per shard
#
# Left manual, the slow stage starves the fast one: seeds pile up ungated and
# nothing reaches datagen. This loop closes that cycle -- gate whatever is new,
# promote the passers, re-shard, keep the sweep staffed -- and is safe to run
# alongside the main supervisor, which does not know about HIP.
#
# Everything it calls is idempotent: seeding is ledgered, gating re-runs cheaply,
# promotion skips what it already copied, and datagen skips finished shards.
#
#   scripts/hip_pipeline_loop.sh
set -uo pipefail

REPO=/home/shasriva/Kore-RL/KORE
PY=/home/shasriva/kore-venv/bin/python
LOG="$REPO/runs/hip_pipeline.log"
PROMOTED="$REPO/data/pool_hip_ok"
GATE_EVERY="${GATE_EVERY:-40}"      # gate once this many new seeds have landed
SHARDS="${HIP_SHARDS:-4}"

# Two seed roots, gated the same way. pool_hip holds the parameter-free modules;
# pool_hip_f holds functionalized ones, whose weights arrive as trailing tensor
# arguments. That distinction matters when writing a seed and is invisible
# afterwards, so both roots share one gate, one harvest and one sweep.
ROOTS="${HIP_ROOTS:-data/pool_hip data/pool_hip_f}"

cd "$REPO" || exit 1
[ -z "${SPUR_CONTROLLER_ADDR:-}" ] && [ -r /etc/profile.d/spur.sh ] && . /etc/profile.d/spur.sh
. "$REPO/scripts/gpu_slots.sh"

say() { echo "[$(date -u '+%H:%M:%SZ')] $*" | tee -a "$LOG"; }

# "__hip*" matches both suffixes: __hip and the functionalized __hipf.
#
# Count each root separately. Appending the glob to a command substitution that
# emits several paths attaches it to the last one only, so this counted a single
# root: with 3,600 seeds on disk it reported 197, and the loop under-triggered
# gating for as long as that undercount stayed below the threshold.
n_seeds() {
    local n=0 r
    for r in $ROOTS; do
        n=$(( n + $(ls -d "$REPO/$r"/tasks/*__hip* 2>/dev/null | wc -l) ))
    done
    echo "$n"
}
n_promoted() { ls -d "$PROMOTED"/tasks/*__hip* 2>/dev/null | wc -l; }
queued() { squeue -u "$USER" -h -n "$1" 2>/dev/null | wc -l; }

say "=== hip pipeline loop start (pid $$) ==="
say "roots: $ROOTS"
last_gated=$(n_promoted)

while :; do
    # Held jobs hold a GPU slot and do no work, so clear them before deciding
    # anything. Left alone they accumulate until nothing can launch.
    purge_held | while read -r l; do say "$l"; done

    # Stop when seeding is done AND everything it produced has been gated.
    seeding_alive=0
    pgrep -f materialize_pool_hip >/dev/null && seeding_alive=1
    seeds=$(n_seeds); promoted=$(n_promoted)

    # --- gate, when enough new seeds have accumulated to be worth a node ------
    # Gating comes before mining when slots are scarce: nothing can be mined that
    # has not been gated, and gating a backlog turns one node into hundreds of
    # tasks, while a sweep over the handful already promoted re-walks the same few.
    ungated=$(( seeds - last_gated ))
    if [ "$ungated" -ge "$GATE_EVERY" ] || { [ "$seeding_alive" = "0" ] && [ "$ungated" -gt 0 ]; }; then
        if [ "$(queued kore-hipgate)" -eq 0 ] && have_slot; then
            say "gating: $seeds seeds on disk, $ungated since the last gate ($(gpu_free) slot(s) free)"
            for r in $ROOTS; do
                [ -d "$REPO/$r/tasks" ] || continue
                have_slot || { say "  no slot left; $r waits for the next pass"; break; }
                GATE_ROOT="$r" sbatch scripts/spur_gate_pool_hip.sbatch 2>&1 | tee -a "$LOG"
                sleep 5   # let the submission register before re-counting slots
            done
            last_gated=$seeds
        fi
    fi

    # --- promote whatever passed, and keep the sweep staffed -----------------
    # Harvest is cheap and idempotent, so run it whenever datagen has room
    # rather than trying to detect "new passers" separately.
    want=$(gpu_free)
    [ "$want" -gt "$SHARDS" ] && want=$SHARDS
    if [ "$(queued kore-factory)" -lt "$SHARDS" ] && [ "$promoted" -gt 0 ] \
       && [ "$want" -gt 0 ]; then
        say "harvest: $promoted promoted task(s), $want slot(s) free -> submitting"
        bash scripts/hip_pool_harvest.sh "$want" 2>&1 | tail -3 | tee -a "$LOG"
    fi

    if [ "$seeding_alive" = "0" ] && [ "$ungated" -le 0 ] && [ "$(queued kore-hipgate)" -eq 0 ]; then
        say "seeding finished, everything gated ($promoted promoted); loop exiting"
        exit 0
    fi
    sleep 300
done
