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

# Match the keepalive wrapper rather than the loop itself: the wrapper is what
# restarts the loop, so a loop running without its wrapper is still degraded.
running() { pgrep -f "keepalive.sh $1 " >/dev/null; }

start() {
    local name="$1"; shift
    running "$name" && return 0
    say "$name not running -> starting"
    setsid nohup env "$@" >/dev/null 2>&1 < /dev/null &
    sleep 2
}

# Configuration lives here, not in the caller, so a cron-triggered restart brings
# the loops back with the same settings a human start would give them. The cap is
# the measured four concurrent jobs.
start hip_pipeline \
    GPU_JOB_CAP=4 HIP_SHARDS=2 SEED_ARGS="" \
    HIP_ROOTS="data/pool_hip_f data/pool_hip" \
    bash "$REPO/scripts/keepalive.sh" hip_pipeline -- bash "$REPO/scripts/hip_pipeline_loop.sh"

start supervise \
    GPU_JOB_CAP=4 AKA_AFTER_SFT=1 AKA_ARM=v4 \
    AKA_MODEL=/shared_nfs/shasriva/kore/runs/sft_v4 \
    bash "$REPO/scripts/keepalive.sh" supervise -- bash "$REPO/scripts/supervise.sh"

exit 0
