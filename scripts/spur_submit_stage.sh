#!/bin/bash
# Partition + submit a FRONTIER STAGE (reverify|evolve) across up to 16 nodes:
# 8 DEDICATED (amd-general-qos, non-preemptible) + 8 BURST (amd-burst-qos, bonus).
# Both stages are resumable, so burst preemption/requeue is harmless.
#
# Usage: scripts/spur_submit_stage.sh <reverify|evolve> [SHARDS] [DED_NODES] [BURST_NODES]
# Re-running repartitions CURRENT remaining work (completed tasks disappear).
set -euo pipefail

STAGE="${1:?stage: reverify|evolve}"
SHARDS="${2:-16}"
DED_NODES="${3:-8}"
BURST_NODES="${4:-8}"

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
DATA_ROOT="${KORE_DATA_ROOT:-$REPO/data/b05factory}"
TARGET="${KORE_WINS_TARGET:-3}"
DED_QOS="${KORE_DED_QOS:-amd-general-qos}"
DED_ACCT="${KORE_DED_ACCT:-amd-general}"
BURST_QOS="${KORE_BURST_QOS:-amd-burst-qos}"
BURST_ACCT="${KORE_BURST_ACCT:-amd-general}"

case "$STAGE" in reverify|evolve) ;; *) echo "bad stage: $STAGE" >&2; exit 2 ;; esac
[[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "SHARDS must be positive int" >&2; exit 2; }
((DED_NODES + BURST_NODES <= SHARDS)) || { echo "DED+BURST must be <= SHARDS" >&2; exit 2; }

cd "$REPO"
[[ -x "$PY" ]] || { echo "missing venv python: $PY" >&2; exit 2; }
[[ -s .env.local ]] || { echo "missing .env.local" >&2; exit 2; }
mkdir -p runs/spur_shards
for cmd in flock git sbatch squeue; do
    command -v "$cmd" >/dev/null || { echo "missing required command: $cmd" >&2; exit 2; }
done

exec 9>runs/.spur_submit.lock
flock -n 9 || { echo "another SPUR submission is being prepared" >&2; exit 3; }

SOURCE_STATUS="$(git status --porcelain --untracked-files=all -- kore scripts tests)"
if [[ -n "$SOURCE_STATUS" ]]; then
    echo "uncommitted source/test changes; refusing deployment" >&2
    printf '%s\n' "$SOURCE_STATUS" >&2
    exit 3
fi

QUEUE_OUTPUT="$(squeue -u "${USER:?}")" || { echo "cannot query scheduler" >&2; exit 4; }
if awk 'NR>1 && $3 ~ /^kore-fron/ {f=1} END{exit !f}' <<<"$QUEUE_OUTPUT"; then
    echo "active kore-frontier jobs detected; refusing overlapping submission" >&2
    exit 3
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
SHARD_DIR="$REPO/runs/spur_shards/${STAGE}-$RUN_ID"

PYTHONPATH=. "$PY" scripts/spur_partition.py \
    --data-root "$DATA_ROOT" --out-dir "$SHARD_DIR" \
    --shards "$SHARDS" --target "$TARGET" --mode "$STAGE" | tee "$SHARD_DIR.partition.log"

WORK="$("$PY" - "$SHARD_DIR/manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["n_work_items"])
PY
)"
if [[ "$WORK" == 0 ]]; then
    echo "STAGE $STAGE already complete; no jobs submitted."
    exit 0
fi

# Dedicated wave (guaranteed nodes).
DED_ARRAY="0-$((DED_NODES - 1))"
DED_JOB="$(sbatch --parsable --array="$DED_ARRAY" \
    -A "$DED_ACCT" -q "$DED_QOS" \
    scripts/spur_stage_array.sbatch "$STAGE" "$SHARD_DIR" "$DATA_ROOT" "$TARGET")"
printf '%s\n' "$DED_JOB" > "$SHARD_DIR/job_id_dedicated"
echo "SPUR_STAGE_SUBMITTED stage=$STAGE qos=$DED_QOS job=$DED_JOB array=$DED_ARRAY shard_dir=$SHARD_DIR"

# Burst wave (best-effort bonus; do not fail the run if the account/qos is refused).
if ((BURST_NODES > 0)); then
    BURST_ARRAY="$DED_NODES-$((DED_NODES + BURST_NODES - 1))"
    if BURST_JOB="$(sbatch --parsable --array="$BURST_ARRAY" \
            -A "$BURST_ACCT" -q "$BURST_QOS" \
            scripts/spur_stage_array.sbatch "$STAGE" "$SHARD_DIR" "$DATA_ROOT" "$TARGET" 2>"$SHARD_DIR.burst.err")"; then
        printf '%s\n' "$BURST_JOB" > "$SHARD_DIR/job_id_burst"
        echo "SPUR_STAGE_SUBMITTED stage=$STAGE qos=$BURST_QOS job=$BURST_JOB array=$BURST_ARRAY (bonus)"
    else
        echo "BURST submission refused (bonus only); dedicated wave continues. Reason:" >&2
        cat "$SHARD_DIR.burst.err" >&2 || true
    fi
fi

echo "work=$WORK shard_dir=$SHARD_DIR"
echo "Monitor: $PY scripts/_kf_verify.py $DATA_ROOT $TARGET ; ls -lt $REPO/runs/spur-*-$STAGE.log"
