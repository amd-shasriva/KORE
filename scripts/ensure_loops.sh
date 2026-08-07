#!/bin/bash
# Make sure the unattended loops are running. Safe to run repeatedly.
#
# The loops live on the login node, and a login node reboot took all of them out
# at once -- after which 745 already-gated tasks sat unharvested for six hours
# because nothing was alive to promote them. keepalive covers a loop that
# crashes; it cannot cover the machine going away underneath it. Cron can, so
# this is written to be idempotent and driven from @reboot plus a timer.
#
#   scripts/ensure_loops.sh
set -uo pipefail

REPO=/home/shasriva/Kore-RL/KORE
LOG="$REPO/runs/ensure_loops.log"
cd "$REPO" || exit 1
[ -z "${SPUR_CONTROLLER_ADDR:-}" ] && [ -r /etc/profile.d/spur.sh ] && . /etc/profile.d/spur.sh

say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$LOG"; }

# Only one of these may run at a time. The idempotency below is per-loop and
# depends on pgrep seeing a wrapper that a concurrent invocation has not started
# yet, so two copies racing each other both conclude nothing is running. A lock
# is the part that cannot be raced.
exec 9>"$REPO/runs/.ensure_loops.lock"
if ! flock -n 9; then
    say "another ensure_loops holds the lock; exiting"
    exit 0
fi

# Match the keepalive wrapper rather than the loop itself: the wrapper is what
# restarts the loop, so a loop running without its wrapper is still degraded.
running() { pgrep -f "keepalive.sh $1 " >/dev/null; }

start() {
    local name="$1"; shift
    running "$name" && return 0
    say "$name not running -> starting"
    # 9>&- closes the lock fd in the child. Without it the long-lived loop
    # inherits the descriptor and therefore the lock, so the lock outlives this
    # script by days and every later invocation decides another copy is already
    # running and does nothing -- which is worse than the duplication the lock
    # was added to prevent.
    setsid nohup env "$@" >/dev/null 2>&1 < /dev/null 9>&- &
    sleep 2
}

# Slot budget: 6 mining (3 pool-HIP + 2 pool-Triton + 1 registry-HIP) and 2
# arenas -- the v4 run and the untuned-base baseline that makes it
# interpretable. Wanting 7 mining left no room for the second arena, and the
# staffing loop reclaimed the slot within a minute of it being freed by hand.
#
# Configuration lives here, not in the caller, so a cron-triggered restart brings
# the loops back with the same settings a human start would give them. The cap is
# the measured four concurrent jobs.
start hip_pipeline \
    GPU_JOB_CAP=8 HIP_SHARDS=7 SEED_ARGS="" \
    DATAGEN_STREAMS="poolhip:runs/shards_hippool:data/v5hippool:3:kore-mine-poolhip+kore-factory pooltriton:runs/shards_pooltriton:data/v5pooltriton:2:kore-mine-pooltriton hipreg:runs/shards_hipreg:data/v5hip:1:kore-mine-hipreg" \
    HIP_ROOTS="" \
    bash "$REPO/scripts/keepalive.sh" hip_pipeline -- bash "$REPO/scripts/hip_pipeline_loop.sh"

start supervise \
    GPU_JOB_CAP=8 AKA_AFTER_SFT=1 AKA_ARM=v4 AKA_TASK_CONCURRENCY=12 \
    AKA_JOB_NAME=kore-aka \
    AKA_MODEL=/shared_nfs/shasriva/kore/runs/sft_v4 \
    bash "$REPO/scripts/keepalive.sh" supervise -- bash "$REPO/scripts/supervise.sh"

# The untuned base arm, which is what makes the v4 number mean anything. It had
# no supervisor at all: it was submitted by hand, and the one instance running
# watches only kore-aka, so when the base arena hit its 8h wall nothing brought
# it back. Same script, its own arm, queue name, ledger and log.
#
# DATAGEN_SHARDS stays unset so this instance never staffs mining, and SFT is
# already complete, so in practice it supervises exactly one thing.
start supervise_base \
    GPU_JOB_CAP=8 AKA_AFTER_SFT=1 AKA_ARM=base AKA_TASK_CONCURRENCY=12 \
    AKA_JOB_NAME=kore-aka-base \
    AKA_OUT="$REPO/runs/aka_base" \
    SUPERVISE_LOG="$REPO/runs/supervise_base.log" \
    AKA_MODEL=/home/shasriva/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120 \
    bash "$REPO/scripts/keepalive.sh" supervise_base -- bash "$REPO/scripts/supervise.sh"

exit 0
