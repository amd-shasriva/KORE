#!/bin/bash
# Keep the v4 SFT run and the full AgentKernelArena eval alive until they finish.
#
# Both jobs die to preemption on this cluster well before they can complete, and
# spurctld itself goes away periodically -- a scancel/sbatch issued at the wrong
# moment just errors out. So the loop treats "no scheduler" as a wait rather than
# a failure, and re-submits only when a job is genuinely absent from the queue.
#
#   scripts/supervise.sh [KILL_JOBIDS...]
#
# Any job ids passed as arguments are cancelled once on the first successful
# scheduler contact. That is how the older, wrongly-configured jobs get retired:
# 38327 was checkpointing every 200 steps, and 38326 was scoring 40 tasks of one
# category instead of the whole arena.
set -uo pipefail

REPO="/home/shasriva/Kore-RL/KORE"
cd "$REPO" || exit 1

# spurctld's address lives in SPUR_CONTROLLER_ADDR, exported by
# /etc/profile.d/spur.sh -- which only LOGIN shells read. Started as a plain
# `bash scripts/supervise.sh`, this script inherited no such variable, so every
# squeue/sbatch/scancel failed with "failed to connect to spurctld" and the loop
# read its own broken environment as a cluster-wide outage: it waited politely for
# 40+ minutes, submitted nothing, and retired nothing, while both jobs were in
# fact running the whole time. Source the profile rather than hardcode the address,
# so a controller move is picked up here too.
if [ -z "${SPUR_CONTROLLER_ADDR:-}" ] && [ -r /etc/profile.d/spur.sh ]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/spur.sh
fi

SFT_OUT="/shared_nfs/shasriva/kore/runs/sft_v4"
V3_MODEL="/shared_nfs/shasriva/kore/runs/sft_coder30b_a3b"
AKA_ARM="kore"
# One fixed directory for the arena, not one per job id. 402 tasks at up to 900s
# each cannot finish inside the eval's 8h allocation, so the run only completes by
# resuming from its partial ledger across several jobs -- which requires every
# attempt to land in the same place.
AKA_OUT="$REPO/runs/aka_full_${AKA_ARM}"
LOG="$REPO/runs/supervise.log"
RETIRE="${*:-}"

mkdir -p "$REPO/runs"

say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }

# Treat the scheduler as a resource that comes and goes. Returns the queue text
# on success so callers can inspect it without a second round trip.
queue() {
    local out
    out="$(squeue -u "$USER" 2>&1)"
    if echo "$out" | grep -q JOBID; then printf '%s' "$out"; return 0; fi
    return 1
}

# A 30B checkpoint with optimizer state is ~488GB, so extra copies are measured
# in half-terabytes. save_total_limit=1 already rotates them, but that only holds
# while the Trainer is alive to do it: a run killed mid-save, or two runs that
# briefly overlap on one output_dir, can strand a checkpoint that nothing will
# ever clean up. Keep the highest step and drop the rest.
prune_checkpoints() {
    local dir="$1" keep n
    [ -d "$dir" ] || return 0
    mapfile -t ck < <(ls -d "$dir"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | grep -E '^[0-9]+$' | sort -n)
    n=${#ck[@]}
    [ "$n" -le 1 ] && return 0
    keep="${ck[$((n-1))]}"
    for s in "${ck[@]}"; do
        [ "$s" = "$keep" ] && continue
        say "prune: removing stale $dir/checkpoint-$s (keeping $keep)"
        rm -rf "$dir/checkpoint-$s"
    done
}

sft_done() {
    # The Trainer writes the sharded weights and their index only after the last
    # step, so the index is the signal that the run finished rather than stopped.
    [ -f "$SFT_OUT/model.safetensors.index.json" ]
}

aka_done() {
    # run_agent_kernel_arena.py writes results_<arm>.json only after the last
    # task, so it -- not the per-task ledger, which exists from task one -- is the
    # signal that the arena is finished. Getting this wrong means resubmitting a
    # completed 402-task eval forever and holding a GPU node for nothing.
    [ -f "$AKA_OUT/results_${AKA_ARM}.json" ]
}

# Carry a ledger written under an older per-job-id directory into the stable one,
# so switching to a fixed output dir does not throw away tasks already scored.
# Only ever adds tasks: lines are keyed by task_id and the longer ledger wins.
adopt_stray_ledger() {
    local best="" bestn=0 n f
    mkdir -p "$AKA_OUT"
    for d in "$REPO"/runs/aka_*; do
        [ -d "$d" ] || continue
        [ "$d" = "$AKA_OUT" ] && continue
        f="$d/results_${AKA_ARM}.partial.jsonl"
        [ -f "$f" ] || continue
        n=$(wc -l < "$f" 2>/dev/null || echo 0)
        if [ "$n" -gt "$bestn" ]; then bestn=$n; best=$f; fi
    done
    [ -z "$best" ] && return 0
    local cur="$AKA_OUT/results_${AKA_ARM}.partial.jsonl"
    local curn=0
    [ -f "$cur" ] && curn=$(wc -l < "$cur" 2>/dev/null || echo 0)
    if [ "$bestn" -gt "$curn" ]; then
        say "adopting ledger with $bestn scored task(s) from $best (stable had $curn)"
        cp "$best" "$cur"
    fi
}

say "=== supervisor start (pid $$) ==="
say "sft_out=$SFT_OUT  retire='${RETIRE}'"
say "controller=${SPUR_CONTROLLER_ADDR:-<unset>}"

# Prove the scheduler is reachable at startup. Without this the two failure modes
# are indistinguishable in the log -- a real outage and a missing
# SPUR_CONTROLLER_ADDR both print "scheduler unreachable" forever -- and the second
# one silently does nothing while looking patient.
if queue >/dev/null; then
    say "scheduler reachable at startup"
elif [ -z "${SPUR_CONTROLLER_ADDR:-}" ]; then
    say "FATAL: SPUR_CONTROLLER_ADDR is unset and /etc/profile.d/spur.sh was not"
    say "       readable. Every submission would fail and this loop would wait"
    say "       forever without submitting anything. Refusing to start blind."
    exit 3
else
    say "WARNING: scheduler unreachable at startup but controller addr is set;"
    say "         treating as a real outage and waiting."
fi

[ -n "$RETIRE" ] && say "will retire on first scheduler contact: $RETIRE"

while :; do
    if ! q="$(queue)"; then
        say "scheduler unreachable; waiting"
        sleep 30
        continue
    fi

    # Retire on the first successful contact, however long that takes. This used
    # to be a bounded pre-loop that gave up after 60 minutes and logged nothing
    # while it waited: a scheduler outage longer than that left the superseded job
    # running, and the main loop then saw a job of the right name in the queue and
    # never replaced it. On this cluster spurctld has been down for 40+ minutes at
    # a stretch, so that ceiling was reachable.
    if [ -n "$RETIRE" ]; then
        for j in $RETIRE; do say "retiring superseded job $j"; scancel "$j" 2>&1 | tee -a "$LOG"; done
        RETIRE=""
        sleep 10
        q="$(queue)" || continue
    fi

    prune_checkpoints "$SFT_OUT"

    have_sft=0; have_aka=0
    echo "$q" | grep -q "kore-sft" && have_sft=1
    echo "$q" | grep -q "kore-aka" && have_aka=1

    # --- SFT ---------------------------------------------------------------
    if sft_done; then
        [ "$have_sft" = "0" ] && say "SFT COMPLETE (index written in $SFT_OUT)"
    elif [ "$have_sft" = "0" ]; then
        # An existing checkpoint means this is a resume, not a fresh start; the
        # launcher picks it up from output_dir on its own.
        ck="$(ls -d "$SFT_OUT"/checkpoint-* 2>/dev/null | wc -l)"
        say "SFT absent from queue -> submitting (existing checkpoints=$ck)"
        sbatch scripts/spur_sft_1node.sbatch \
            configs/sft_coder30b_a3b.json - "$SFT_OUT" 2>&1 | tee -a "$LOG"
    fi

    # --- AgentKernelArena: every task, every category ----------------------
    # types='-' means all of torch2hip/hip2hip/triton2triton and limit=0 means no
    # cap, so this is the whole arena rather than a sample of it.
    if aka_done; then
        [ "$have_aka" = "0" ] && say "AKA COMPLETE (summary written)"
    elif [ "$have_aka" = "0" ]; then
        adopt_stray_ledger
        scored=0
        [ -f "$AKA_OUT/results_${AKA_ARM}.partial.jsonl" ] && \
            scored=$(wc -l < "$AKA_OUT/results_${AKA_ARM}.partial.jsonl")
        say "AKA absent from queue -> submitting full arena (all types, no limit; $scored/402 already scored)"
        sbatch scripts/spur_aka_1node.sbatch run - "$V3_MODEL" 0 "$AKA_ARM" "$AKA_OUT" 2>&1 | tee -a "$LOG"
    fi

    if sft_done && aka_done; then
        say "=== both jobs complete; supervisor exiting ==="
        exit 0
    fi

    sleep 60
done
