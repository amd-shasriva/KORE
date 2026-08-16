#!/usr/bin/env bash
# Queue the final measured arena sweep on amd-primus.
#
# The job just needs to be in the queue so it lands when a node frees. It cannot
# affect the running SFT job: amd-spur reports PreemptMode=OFF, so nothing queued
# can evict anything running, and the SFT job runs to completion on its own node.
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

# The account MUST match the QoS. amd-general paired with a primus or burst QoS
# is a phantom association: accepted at submit, never dispatched. Passed
# explicitly as well as in the sbatch header, the way ensure_sft.sh does.
set -- --account=amd-primus --qos=amd-primus-qos --export=ALL \
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
