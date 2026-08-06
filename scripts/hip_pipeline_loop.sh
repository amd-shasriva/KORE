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

# Selector for the seeding sweep this loop keeps staffed. The parameter-free
# sweep is complete, so what remains is the functionalized set.
SEED_ARGS="${SEED_ARGS:---functionalize --skip-parameter-free}"

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
    #
    # Seeding runs as a compute job, not a local process, so liveness is a queue
    # question. Checking only for a local process would report seeding finished
    # the moment it moved off the login node, and the loop would exit while
    # thousands of tasks were still unseeded.
    seeding_alive=0
    pgrep -f materialize_pool_hip >/dev/null && seeding_alive=1
    [ "$(queued kore-seed)" -gt 0 ] && seeding_alive=1
    seeds=$(n_seeds); promoted=$(n_promoted)

    # --- mine first ----------------------------------------------------------
    # Slots go to mining before gating, which reverses the earlier rule. That rule
    # was right when 14 tasks were promoted and gating was what produced work; it
    # is wrong now that 1,600+ gated tasks are queued, far more than a night of
    # mining can consume. Gating more tasks onto that pile produces nothing until
    # the pile is drawn down, while every mining slot produces training data.
    want=$(gpu_free)
    [ "$want" -gt "$SHARDS" ] && want=$SHARDS
    if [ "$(queued kore-factory)" -lt "$SHARDS" ] && [ "$promoted" -gt 0 ] \
       && [ "$want" -gt 0 ]; then
        say "harvest: $promoted promoted task(s), $want slot(s) free -> submitting"
        bash scripts/hip_pool_harvest.sh "$want" 2>&1 | tail -3 | tee -a "$LOG"
    fi

    # --- keep the seeding sweep staffed --------------------------------------
    # Only when explicitly enabled. Seeding is currently ahead of gating -- 2,500
    # seeds are on disk undecided -- so another seeding job would occupy one of
    # only four slots to deepen a queue nothing is draining.
    if [ "$seeding_alive" = "0" ] && [ -n "$SEED_ARGS" ] \
       && [ ! -f "$REPO/runs/seeding.done" ] && have_slot; then
        say "seeding absent -> submitting ($SEED_ARGS)"
        # shellcheck disable=SC2086
        sbatch $QOS_ARG scripts/spur_seed_hip.sbatch $SEED_ARGS 2>&1 | tee -a "$LOG"
        seeding_alive=1
    fi

    # --- gate whatever slots remain ------------------------------------------
    # Each root gets its own gate job, named after the root and tracked
    # separately. A single shared gate slot meant that while the parameter-free
    # root was being gated -- hours of work -- the functionalized root could never
    # start, so the seeds that unlock most of the pool would have sat undecided
    # all night behind the ones that were already nearly done.
    ungated=$(( seeds - last_gated ))
    if [ "$ungated" -ge "$GATE_EVERY" ] || { [ "$seeding_alive" = "0" ] && [ "$ungated" -gt 0 ]; }; then
        for r in $ROOTS; do
            [ -d "$REPO/$r/tasks" ] || continue
            tag=$(basename "$r")
            n_root=$(ls -d "$REPO/$r"/tasks/*__hip* 2>/dev/null | wc -l)
            [ "$n_root" -eq 0 ] && continue
            [ "$(queued "kore-gate-$tag")" -gt 0 ] && continue
            have_slot || { say "  no slot for gate-$tag; waits for next pass"; break; }
            say "gating $r: $n_root seeds ($(gpu_free) slot(s) free)"
            GATE_ROOT="$r" sbatch $QOS_ARG --job-name="kore-gate-$tag" \
                scripts/spur_gate_pool_hip.sbatch 2>&1 | tee -a "$LOG"
            sleep 5   # let the submission register before re-counting slots
        done
        last_gated=$seeds
    fi

    # Only stop once seeding is genuinely finished -- marked by runs/seeding.done,
    # not merely by no seeding job being queued right now, which is also true
    # between a preemption and the next resubmission.
    if [ -f "$REPO/runs/seeding.done" ] && [ "$ungated" -le 0 ] \
       && [ "$(queued kore-gate-pool_hip)" -eq 0 ] \
       && [ "$(queued kore-gate-pool_hip_f)" -eq 0 ]; then
        say "seeding finished, everything gated ($promoted promoted); loop exiting"
        exit 0
    fi
    sleep 300
done
