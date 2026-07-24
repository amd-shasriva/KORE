#!/bin/bash
# Partition current unfinished work and submit a preemption-safe SPUR burst array.
#
# Usage:
#   scripts/spur_submit_datagen.sh [SHARDS] [MAX_CONCURRENT]
#
# Recommended ramp:
#   scripts/spur_submit_datagen.sh 8 2   # first live throughput/API check
#   # After completion, rerun with 8 4 (then 16 8) if retries remain near zero.
#
# Re-running is the cleanup/resume operation: partitioning reads the CURRENT shared
# dataset, so complete work disappears and only remaining gaps are resubmitted.
set -euo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
DATA_ROOT="${KORE_DATA_ROOT:-$REPO/data/b05factory}"
SHARDS="${1:-8}"
MAX_CONCURRENT="${2:-2}"
TARGET="${KORE_WINS_TARGET:-3}"

if ! [[ "$SHARDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "SHARDS must be a positive integer" >&2
    exit 2
fi
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_CONCURRENT must be a positive integer" >&2
    exit 2
fi
if ((MAX_CONCURRENT > SHARDS)); then
    MAX_CONCURRENT="$SHARDS"
fi

cd "$REPO"
[[ -x "$PY" ]] || { echo "missing venv python: $PY" >&2; exit 2; }
[[ -s .env.local ]] || { echo "missing .env.local" >&2; exit 2; }
mkdir -p runs/spur_shards

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

ARRAY="0-$((SHARDS - 1))%${MAX_CONCURRENT}"
JOB_ID="$(sbatch --parsable --array="$ARRAY" scripts/spur_datagen_array.sbatch \
    "$SHARD_DIR" "$DATA_ROOT" "$TARGET")"
printf '%s\n' "$JOB_ID" > "$SHARD_DIR/job_id"

echo "SPUR_DATAGEN_SUBMITTED job_id=$JOB_ID array=$ARRAY work=$WORK shard_dir=$SHARD_DIR"
echo "Monitor:"
echo "  squeue -j $JOB_ID"
echo "  $PY scripts/_kf_verify.py $DATA_ROOT $TARGET"
echo "  sacct -j $JOB_ID -l"
echo "  ls -lt $REPO/runs/spur-*.out"
