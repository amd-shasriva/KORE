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

# Pool-HIP is retired at 0 workers, and its share goes to the frontier set.
#
# It was mining breadth, not difficulty. Measured across its 6,457 gated tasks:
# 86% have a baseline under 100us, median 17us, so they are launch-bound and no
# amount of them teaches tiling, LDS staging or MFMA scheduling; attention is
# 1.3% of the pool, quantization 0.2%, MoE one task. Its wins reported a median
# 2.29x and a maximum of 3171x, which measures the torch baseline rather than
# the kernel.
#
# scripts/select_frontier_tasks.py ranks by what a win is worth instead, and the
# score histogram separates cleanly: 482 registry tasks -- 214 attention, 115
# MoE, 94 quant/fp8, 52 gemm, in fp8_e4m3fn/int4_w4a16/mxfp4/bf16 against vendor
# baselines -- of which 401 had never been mined at all.
#
# Pool-Triton is retired for the same reason and one more. Its own tasks are the
# same launch-bound modules -- median baseline 16us, 92% under 100us, zero
# attention and zero MoE across 6,064 rows -- and its second purpose no longer
# holds: translation pairs exist only where a task won in BOTH backends, and
# with pool-HIP stopped that overlap (109 tasks) cannot grow. Those pairs are
# already banked; the next node-hour spent there buys Triton coverage of 16us
# kernels.
#
# Its slot goes to frontier, which has Triton seeds for flash attention, MoE and
# fp8 -- Triton data that is hard rather than merely new.
#
# Slot budget: 6 mining (all frontier, one per shard) and 2
# arenas -- the v4 run and the untuned-base baseline that makes it
# interpretable. Wanting 7 mining left no room for the second arena, and the
# staffing loop reclaimed the slot within a minute of it being freed by hand.
#
# Configuration lives here, not in the caller, so a cron-triggered restart brings
# the loops back with the same settings a human start would give them. The cap is
# the measured four concurrent jobs.
# One pipeline for every dialect, replacing the HIP-only loop.
#
# The old loop drove seed->gate->mine for HIP alone; everything else was a thing
# somebody had to remember to start, which is how pool-Triton and registry-HIP
# each died twice in a night and how FlyDSL sat at zero rows while 45 arena tasks
# scored 1%. frontier_pipeline.sh keeps the HIP and FlyDSL materializers alive on
# the login node -- they are gateway-bound and would hold a node hostage to
# network latency inside an allocation -- gates whichever root accumulates seeds,
# refreshes the shard stamps, and staffs mining across every declared stream.
#
# GPU_JOB_CAP is 8 because 8 is where the scheduler actually stops, measured:
# with eight jobs running and two nodes idle inside my own reservation, the
# ninth submission went straight to JobLaunchFailure, requeued, and wedged in
# JobHoldMaxRequeue -- the signature gpu_slots.sh describes for submitting past
# the ceiling. Set to 10, every pass of every loop offered two jobs that could
# only ever wedge, and each one held a slot in the queue while doing nothing:
# four of them had been pending between six and nine hours, one submitted at
# 16:05, and they were what made the cap look full to the staffing loop.
#
# The stream split is 1 frontier-Triton to 2 frontier-twins. Triton already has
# 10k mined rows and is the dialect the corpus is thickest in; the twins had
# none at all, because until now nothing mined them -- the arena is 22% HIP and
# 25% FlyDSL and both were being scored against training data that did not
# exist. Streams are staffed in the order they appear here, and the first one
# takes what it can, so the ordering is the priority.
start frontier_pipeline \
    GPU_JOB_CAP=8 FRONTIER_FAMILIES="attention gemm quantization" \
    HIP_ROOT=data/pool_hip_frontier FLYDSL_ROOT=data/pool_flydsl \
    REG_HIP_ROOT=data/registry_hip_frontier \
    REG_FLYDSL_ROOT=data/registry_flydsl_frontier \
    TWIN_OK_ROOT=data/frontier_twins_ok \
    TWIN_DATA_ROOT=data/v5frontier_twins \
    TWIN_SHARD_DIR=runs/shards_frontier_twins \
    KORE_QOS=amd-burst-qos \
    DATAGEN_STREAMS="frontiertwins:runs/shards_frontier_twins:data/v5frontier_twins:3:kore-mine-frontiertwins poolflydsl:runs/shards_pool_flydsl:data/v5pool_flydsl:3:kore-mine-poolflydsl hipreg:runs/shards_hipreg:data/v5hip:3:kore-mine-hipreg poolhip:runs/shards_hippool:data/v5hippool:0:kore-mine-poolhip+kore-factory frontier:runs/shards_frontier:data/v5frontier:0:kore-mine-frontier pooltriton:runs/shards_pooltriton:data/v5pooltriton:0:kore-mine-pooltriton" \
    bash "$REPO/scripts/keepalive.sh" frontier_pipeline -- \
    bash "$REPO/scripts/frontier_pipeline.sh"

# hip_pipeline_loop.sh is superseded by frontier_pipeline.sh and no longer
# started. It is kept in the tree because its seed/gate/harvest sequencing is
# the thing frontier_pipeline generalises, and because pool-HIP can be revived
# from it if the frontier set is ever exhausted.

# Arenas go to amd-general-qos, mining stays on burst.
#
# Burst is the big pool but it is genuinely saturated -- 125 nodes running
# cluster-wide -- and a burst job simply waits, which for the eval means the
# thing we are actually trying to measure makes no progress. amd-general-qos
# caps the whole QoS at 8 nodes across all users and only 4 were taken, so a
# one-node arena fits there now and starts immediately. Mining is throughput
# work that can afford to queue; the eval is not.
start supervise \
    GPU_JOB_CAP=8 AKA_AFTER_SFT=1 AKA_ARM=v4 AKA_TASK_CONCURRENCY=12 \
    AKA_JOB_NAME=kore-aka KORE_QOS=amd-general-qos \
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
    AKA_JOB_NAME=kore-aka-base KORE_QOS=amd-general-qos \
    AKA_OUT="$REPO/runs/aka_base" \
    SUPERVISE_LOG="$REPO/runs/supervise_base.log" \
    AKA_MODEL=/home/shasriva/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120 \
    bash "$REPO/scripts/keepalive.sh" supervise_base -- bash "$REPO/scripts/supervise.sh"

exit 0
