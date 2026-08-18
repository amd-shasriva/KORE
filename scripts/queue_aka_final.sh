#!/usr/bin/env bash
# Queue the final measured arena sweep on amd-primus.
#
# The job just needs to be in the queue so it lands when a node frees.
#
# PREEMPTION, CORRECTED. This header used to claim "amd-spur reports
# PreemptMode=OFF, so nothing queued can evict anything running". As of
# 2026-08-18 `scontrol show partition amd-spur` reports PreemptMode=CANCEL with
# PriorityTier=100, so that reassurance was false. What actually keeps this
# submission from harming another job is narrower and still true: submitting adds
# a job to the queue and touches nothing else, and this script issues no scancel
# and no scontrol update. A queued job of ours cannot evict a running job of ours
# because they share one account and QOS at the same priority tier.
#
# The consequence to keep in mind is the reverse direction: a job WE submit on a
# preemptible QOS (amd-burst-qos) can itself be cancelled. That is survivable here
# only because every phase is idempotent and each task is ledgered as it finishes.
#
# This script is read-only with respect to every job it did not create: no
# scancel, no scontrol update, no write anywhere under the SFT output tree. It
# prints the queue before and after submitting so the SFT row is visibly
# unchanged.
#
# Note on SPUR: sbatch here is a reimplementation with a narrower flag set than
# Slurm's. It supports --account, --qos, --dependency, --export, --requeue,
# --exclusive and --hold; it does NOT implement --nice, and passing it aborts the
# submission outright.
set -uo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
export SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR:-http://crs-m2m-cpu-spur-005:6817}"

ARMS="${KORE_AKA_ARMS:-base opus}"
OUT="${KORE_AKA_OUT:-$REPO/runs/aka_v5_final}"
DRY="${KORE_AKA_DRY_RUN:-0}"
# A distinct name lets a second sweep queue alongside the first. The duplicate
# guard below keys on this name, so two sweeps with different names and
# different --out directories can wait in line together, while a second copy of
# the SAME sweep is still refused.
NAME="${KORE_AKA_JOB_NAME:-kore-aka-final}"
V5_MODEL="${KORE_AKA_V5_MODEL:-}"

# Which account/QOS to submit under, and why this is a knob rather than a constant.
#
# amd-primus-qos has a GROUP NODE CAP. On 2026-08-18 it was 16/16 with 32 jobs
# waiting on QOSGrpNodeLimit, ours 20th, and 15 of the 16 nodes were parked shells
# (sdc-hold14, ethany-hold, amc-hold, gc-reserve-primus, dev-node) idling for up to
# 4 days. Position in that queue does not improve when OUR job ends: the freed node
# goes to the head of the line, not back to us. amd-burst-qos had 98 nodes running
# and ZERO jobs blocked by a node cap.
#
# This user is associated with amd-burst, amd-general and amd-primus, so the burst
# route is available -- but ONLY as a matching pair. The controller rejects
# amd-primus + amd-burst-qos outright:
#   QOS 'amd-burst-qos' is not permitted for user 'shasriva' under account 'amd-primus'
# which is the same "account must match the QoS" rule spur_aka_final.sbatch already
# documents. _matching_qos below encodes it so a mismatch fails here, in a second,
# instead of after a queue wait.
#
# Burst is the preemptible tier and this partition reports PreemptMode=CANCEL, so a
# burst job CAN be killed. That is acceptable for this sweep specifically because
# every phase is idempotent and every task's result is written to a durable ledger
# as it completes: a preempted run resumes into the same --out and re-scores
# nothing. Observed burst jobs had been running 3-4 days, so it is not frequent.
ACCOUNT="${KORE_AKA_ACCOUNT:-amd-primus}"

_matching_qos() {
    case "$1" in
        amd-primus) echo "amd-primus-qos" ;;
        amd-burst)  echo "amd-burst-qos" ;;
        amd-general) echo "amd-general-qos" ;;
        *)          echo "" ;;
    esac
}

QOS="${KORE_AKA_QOS:-$(_matching_qos "$ACCOUNT")}"
if [ -z "$QOS" ]; then
    echo "unknown account '$ACCOUNT': no QOS mapping. Set KORE_AKA_QOS explicitly." >&2
    exit 2
fi
expected_qos="$(_matching_qos "$ACCOUNT")"
if [ -n "$expected_qos" ] && [ "$QOS" != "$expected_qos" ]; then
    echo "REFUSING: account '$ACCOUNT' with QOS '$QOS' is a cross-family pair." >&2
    echo "  The controller rejects these at submit; '$ACCOUNT' pairs with '$expected_qos'." >&2
    exit 2
fi

command -v squeue >/dev/null 2>&1 || { echo "no squeue on PATH" >&2; exit 2; }
command -v sbatch >/dev/null 2>&1 || { echo "no sbatch on PATH" >&2; exit 2; }

# Refuse to queue a second copy. Two sweeps sharing one --out would have their
# workers delete each other's task workspaces mid-evaluation.
existing="$(squeue -u "$USER" -h -o '%i %j %T' 2>/dev/null \
            | awk -v n="$NAME" '$2==n{print $1" "$3}')"
if [ -n "$existing" ]; then
    echo "already queued; not submitting a second '$NAME':"
    echo "$existing" | sed 's/^/  /'
    exit 0
fi

# Validate the v5 checkpoint HERE, at submit, not three days from now on a
# compute node. The arena would exit 2 on a missing model after the whole queue
# wait, and a scarce allocation spent discovering a typo is the most expensive
# way to find one.
case " $ARMS " in
    *" v5 "*)
        if [ -z "$V5_MODEL" ]; then
            echo "arms include v5 but KORE_AKA_V5_MODEL is unset" >&2
            exit 2
        fi
        if [ ! -f "$V5_MODEL/config.json" ]; then
            echo "KORE_AKA_V5_MODEL=$V5_MODEL has no config.json; that is not a" \
                 "loadable checkpoint" >&2
            exit 2
        fi
        if [ ! -f "$V5_MODEL/model.safetensors.index.json" ] \
           && ! ls "$V5_MODEL"/*.safetensors >/dev/null 2>&1; then
            echo "KORE_AKA_V5_MODEL=$V5_MODEL has no weights" >&2
            exit 2
        fi
        echo "v5 checkpoint verified: $V5_MODEL"
        ;;
esac

echo "--- queue before submit ---"
squeue -u "$USER" -o '%.9i %.16j %.9T %.12M %R' 2>/dev/null

mkdir -p "$OUT" "$REPO/runs"

# The knobs travel in the environment, NOT inside --export=ALL,NAME=VALUE.
# --export takes a comma-separated list and KORE_AKA_ARMS is "base opus" -- a
# value with a space in it. Embedding that in the list is exactly how an arm gets
# silently dropped from a multi-day sweep. --export=ALL propagates the submitting
# environment verbatim, spaces included.
export KORE_AKA_ARMS="$ARMS"
export KORE_AKA_OUT="$OUT"
[ -n "$V5_MODEL" ] && export KORE_AKA_V5_MODEL="$V5_MODEL"

# The account MUST match the QoS; validated at the top of this script. Passed
# explicitly as well as in the sbatch header (which defaults to primus) so these
# win over it, the way ensure_sft.sh does.
set -- --account="$ACCOUNT" --qos="$QOS" --export=ALL \
       --job-name="$NAME" "$REPO/scripts/spur_aka_final.sbatch"

echo "sbatch $*"
if [ "$DRY" = "1" ]; then
    echo "(dry run; nothing submitted)"
    exit 0
fi

out="$(sbatch "$@" 2>&1)"
rc=$?
echo "$out"
[ "$rc" != "0" ] && { echo "submit FAILED (rc=$rc)" >&2; exit "$rc"; }

jid="$(echo "$out" | grep -oE '[0-9]+' | tail -1)"
echo "queued arena sweep job ${jid:-?}  arms='$ARMS'  out=$OUT"

echo "--- queue after submit (the kore-sft row must be unchanged) ---"
squeue -u "$USER" -o '%.9i %.16j %.9T %.12M %R' 2>/dev/null
