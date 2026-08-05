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
SEEDS="$REPO/data/pool_hip"
PROMOTED="$REPO/data/pool_hip_ok"
GATE_EVERY="${GATE_EVERY:-40}"      # gate once this many new seeds have landed
SHARDS="${HIP_SHARDS:-4}"

cd "$REPO" || exit 1
[ -z "${SPUR_CONTROLLER_ADDR:-}" ] && [ -r /etc/profile.d/spur.sh ] && . /etc/profile.d/spur.sh

say() { echo "[$(date -u '+%H:%M:%SZ')] $*" | tee -a "$LOG"; }

n_seeds() { ls -d "$SEEDS"/tasks/*__hip 2>/dev/null | wc -l; }
n_promoted() { ls -d "$PROMOTED"/tasks/*__hip 2>/dev/null | wc -l; }
queued() { squeue -u "$USER" -h -n "$1" 2>/dev/null | wc -l; }

say "=== hip pipeline loop start (pid $$) ==="
last_gated=$(n_promoted)

while :; do
    # Stop when seeding is done AND everything it produced has been gated.
    seeding_alive=0
    pgrep -f materialize_pool_hip >/dev/null && seeding_alive=1
    seeds=$(n_seeds); promoted=$(n_promoted)

    # --- gate, when enough new seeds have accumulated to be worth a node ------
    ungated=$(( seeds - last_gated ))
    if [ "$ungated" -ge "$GATE_EVERY" ] || { [ "$seeding_alive" = "0" ] && [ "$ungated" -gt 0 ]; }; then
        if [ "$(queued kore-hipgate)" -eq 0 ]; then
            say "gating: $seeds seeds on disk, $ungated since the last gate"
            sbatch scripts/spur_gate_pool_hip.sbatch 2>&1 | tee -a "$LOG"
            last_gated=$seeds
        fi
    fi

    # --- promote whatever passed, and keep the sweep staffed -----------------
    # Harvest is cheap and idempotent, so run it whenever datagen has room
    # rather than trying to detect "new passers" separately.
    if [ "$(queued kore-factory)" -lt "$SHARDS" ] && [ "$promoted" -gt 0 ]; then
        say "harvest: $promoted promoted task(s), factory below $SHARDS -> submitting"
        bash scripts/hip_pool_harvest.sh "$SHARDS" 2>&1 | tail -3 | tee -a "$LOG"
    fi

    if [ "$seeding_alive" = "0" ] && [ "$ungated" -le 0 ] && [ "$(queued kore-hipgate)" -eq 0 ]; then
        say "seeding finished, everything gated ($promoted promoted); loop exiting"
        exit 0
    fi
    sleep 300
done
