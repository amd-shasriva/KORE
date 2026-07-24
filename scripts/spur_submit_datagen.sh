#!/bin/bash
# Partition current unfinished work and submit a preemption-safe SPUR burst array.
#
# Usage:
#   scripts/spur_submit_datagen.sh [SHARDS] [WAVE_NODES]
#
# Recommended ramp:
#   scripts/spur_submit_datagen.sh 16 4  # submit exactly four nodes
#   # After the wave ends, rerun: current-state partitioning removes completed work.
#
# Re-running is the cleanup/resume operation: partitioning reads the CURRENT shared
# dataset, so complete work disappears and only remaining gaps are resubmitted.
set -euo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
DATA_ROOT="${KORE_DATA_ROOT:-$REPO/data/b05factory}"
SHARDS="${1:-8}"
WAVE_NODES="${2:-2}"
TARGET="${KORE_WINS_TARGET:-3}"

if ! [[ "$SHARDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "SHARDS must be a positive integer" >&2
    exit 2
fi
if ! [[ "$WAVE_NODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "WAVE_NODES must be a positive integer" >&2
    exit 2
fi
if ((WAVE_NODES > SHARDS)); then
    WAVE_NODES="$SHARDS"
fi

cd "$REPO"
[[ -x "$PY" ]] || { echo "missing venv python: $PY" >&2; exit 2; }
[[ -s .env.local ]] || { echo "missing .env.local" >&2; exit 2; }
mkdir -p runs/spur_shards

# SPUR ignores Slurm's array throttle suffix (e.g. %4), so overlapping campaigns
# would launch every child and race on task shards. Refuse while any factory job
# owned by this user is active; submit only an explicit wave below.
if squeue -u "$USER" 2>/dev/null | awk \
    'NR > 1 && $3 ~ /^kore-fac/ && ($5 == "R" || $5 == "PD") {found=1} END {exit !found}'; then
    echo "active kore-factory jobs detected; refusing overlapping submission" >&2
    exit 3
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
SHARD_DIR="$REPO/runs/spur_shards/$RUN_ID"

PYTHONPATH=. "$PY" scripts/spur_partition.py \
    --data-root "$DATA_ROOT" \
    --out-dir "$SHARD_DIR" \
    --shards "$SHARDS" \
    --target "$TARGET" | tee "$SHARD_DIR.partition.log"

WORK="$("$PY" - "$SHARD_DIR/manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["n_work_items"])
PY
)"
if [[ "$WORK" == 0 ]]; then
    echo "Dataset is already complete; no jobs submitted."
    exit 0
fi

ARRAY="0-$((WAVE_NODES - 1))"
JOB_ID="$(sbatch --parsable --array="$ARRAY" scripts/spur_datagen_array.sbatch \
    "$SHARD_DIR" "$DATA_ROOT" "$TARGET")"
printf '%s\n' "$JOB_ID" > "$SHARD_DIR/job_id"

echo "SPUR_DATAGEN_SUBMITTED job_id=$JOB_ID array=$ARRAY work=$WORK shard_dir=$SHARD_DIR"
echo "SPUR ignores array throttles; this wave contains exactly $WAVE_NODES node(s)."
echo "Rerun this submit script after the wave ends to repartition remaining work."
echo "Monitor:"
echo "  squeue -j $JOB_ID"
echo "  $PY scripts/_kf_verify.py $DATA_ROOT $TARGET"
echo "  sacct -j $JOB_ID -l"
echo "  ls -lt $REPO/runs/spur-*.out"
