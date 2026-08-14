#!/usr/bin/env bash
# Queue the final measured arena sweep without endangering the running SFT job.
#
# THE HAZARD THIS SCRIPT EXISTS TO REMOVE
#
# amd-primus enforces a QoS group node limit. Job 11215 (v5 SFT) sat in
# QOSGrpNodeLimit for over seven hours before it landed, so the cap is real and
# it is close. A sweep that starts while SFT is still running would hold one of
# those node slots for up to three days. That cannot kill SFT -- the partition
# reports PreemptMode=OFF, so nothing queued can evict anything running -- but if
# SFT ever died and its supervisor resubmitted, the resubmission would then be
# the job blocked on QOSGrpNodeLimit, behind this sweep. Recovery of a multi-day
# training run would wait on a benchmark.
#
# So the sweep is submitted with a dependency on the live SFT job. It queues now
# and accrues its place, and it is *unable* to occupy a node while SFT holds one.
# The costs are asymmetric and that is the whole argument: delaying the benchmark
# costs a day, delaying SFT recovery costs the run.
#
# --nice covers the residual case. If SFT ends and its supervisor submits a
# successor, the dependency releases this sweep at roughly the same moment; the
# nice value keeps the successor ahead of it in the queue.
#
# To deliberately race SFT instead (not recommended), release the dependency:
#   scontrol update JobId=<id> Dependency=
#
# This script only ever reads SFT state. It contains no scancel, no scontrol
# update, and no write to the SFT output tree.
set -uo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
export SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR:-http://crs-m2m-cpu-spur-005:6817}"

ARMS="${KORE_AKA_ARMS:-base opus}"
OUT="${KORE_AKA_OUT:-$REPO/runs/aka_v5_final}"
NICE="${KORE_AKA_NICE:-500}"
SFT_NAME="${KORE_SFT_JOB_NAME:-kore-sft}"
DRY="${KORE_AKA_DRY_RUN:-0}"

command -v squeue >/dev/null 2>&1 || { echo "no squeue on PATH" >&2; exit 2; }

# Refuse to queue a second copy. Two sweeps sharing one --out would have their
# workers delete each other's task workspaces mid-evaluation.
existing="$(squeue -u "$USER" -h -o '%i %j %T' 2>/dev/null | awk '$2=="kore-aka-final"{print $1" "$3}')"
if [ -n "$existing" ]; then
    echo "already queued; not submitting a second sweep:"
    echo "$existing" | sed 's/^/  /'
    exit 0
fi

# Read-only look at SFT. Match on name so a resubmitted job is still found.
sft="$(squeue -u "$USER" -h -o '%i %j %T' 2>/dev/null | awk -v n="$SFT_NAME" '$2==n{print $1" "$3; exit}')"
dep=""
if [ -n "$sft" ]; then
    sft_id="${sft%% *}"
    sft_state="${sft##* }"
    echo "SFT job $sft_id is $sft_state -- gating this sweep behind it."
    dep="--dependency=afterany:$sft_id"
else
    echo "no $SFT_NAME job in the queue; submitting without a dependency."
fi

mkdir -p "$OUT" "$REPO/runs"

set -- --account=amd-primus --qos=amd-primus-qos --nice="$NICE"
[ -n "$dep" ] && set -- "$@" "$dep"
set -- "$@" \
    --export=ALL,KORE_AKA_ARMS="$ARMS",KORE_AKA_OUT="$OUT" \
    "$REPO/scripts/spur_aka_final.sbatch"

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
echo "queued arena sweep job $jid  arms='$ARMS'  out=$OUT"

# Prove, from the controller's own view, that SFT was untouched.
echo "--- SFT after submit (must be unchanged) ---"
squeue -u "$USER" -o '%.9i %.16j %.9T %.12M %R' 2>/dev/null
