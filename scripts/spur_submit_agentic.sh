#!/bin/bash
# Partition the agentic datagen campaign and submit one SPUR job per node shard.
#
# Usage: scripts/spur_submit_agentic.sh [NODES] [EPISODES_PER_TASK] [WORKERS]
#
# Each shard goes out as its own single-element array rather than one wide array,
# because SPUR hands back PENDING(JobHoldMaxRequeue) spuriously and often: a wide
# array leaves individual elements wedged with no way to retry just those, while
# a per-shard submission can cancel and resubmit exactly the one that stuck. The
# manifest preflight in the sbatch still keys off the array index, so the shard
# identity is unchanged.
#
# Re-running is the resume operation: the partitioner drops tasks whose episode
# quota is already durable, so a second wave only covers what is left.
set -uo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
OUT_DIR="${KORE_AGENTIC_OUT:-$REPO/data/b05factory/agentic_mt}"
NODES="${1:-8}"
EPISODES="${2:-6}"
WORKERS="${3:-48}"
MAX_TURNS="${KORE_AGENTIC_MAX_TURNS:-8}"
MIN_FREE_GB="${KORE_AGENTIC_MIN_FREE_GB:-150}"
SUBMIT_ATTEMPTS="${KORE_AGENTIC_SUBMIT_ATTEMPTS:-8}"

export SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR:-http://crs-m2m-cpu-spur-005:6817}"

cd "$REPO"
[[ -x "$PY" ]] || { echo "missing venv python: $PY" >&2; exit 2; }
[[ -s .env.local ]] || { echo "missing .env.local" >&2; exit 2; }
if ! [[ "$NODES" =~ ^[1-9][0-9]*$ ]]; then echo "NODES must be a positive integer" >&2; exit 2; fi

# The shared volume is the one resource a runaway campaign can take from other
# users irreversibly. Refuse to start a wave that begins below the floor its own
# nodes would stop at.
FREE_GB="$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')"
if (( FREE_GB < MIN_FREE_GB )); then
    echo "FATAL /home has ${FREE_GB}G free, below the ${MIN_FREE_GB}G floor" >&2
    exit 4
fi
echo "[submit] /home free: ${FREE_GB}G (floor ${MIN_FREE_GB}G)"

# A manifest must describe exactly the committed source deployed to the nodes.
SOURCE_STATUS="$(git status --porcelain --untracked-files=all -- kore scripts tests)"
if [[ -n "$SOURCE_STATUS" ]]; then
    echo "uncommitted source/test changes detected; refusing deployment" >&2
    printf '%s\n' "$SOURCE_STATUS" >&2
    exit 3
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
SHARD_DIR="$REPO/runs/agentic_shards/$RUN_ID"
mkdir -p "$SHARD_DIR" "$OUT_DIR"

PYTHONPATH=. "$PY" scripts/agentic_partition.py \
    --out-dir "$OUT_DIR" \
    --shard-dir "$SHARD_DIR" \
    --shards "$NODES" \
    --episodes-per-task "$EPISODES" | tee "$SHARD_DIR/partition.log"

PLANNED="$("$PY" - "$SHARD_DIR/manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["n_tasks_planned"])
PY
)"
if [[ "$PLANNED" == "0" ]]; then
    echo "Every task already has its episode quota; nothing submitted."
    exit 0
fi

SUBMITTED=()
for ((shard = 0; shard < NODES; shard++)); do
    IDX="$(printf '%03d' "$shard")"
    if [[ ! -s "$SHARD_DIR/shard_${IDX}.txt" ]]; then
        echo "[submit] shard $IDX is empty; skipping"
        continue
    fi
    LANDED=0
    for ((attempt = 1; attempt <= SUBMIT_ATTEMPTS; attempt++)); do
        JOB="$(sbatch --parsable --array="${shard}-${shard}" \
            scripts/spur_agentic_saturate.sbatch \
            "$SHARD_DIR" "$OUT_DIR" "$EPISODES" "$WORKERS" "$MAX_TURNS" 2>&1)"
        if ! [[ "$JOB" =~ ^[0-9]+$ ]]; then
            echo "[submit] shard $IDX attempt $attempt: submit failed: $JOB"
            sleep 10
            continue
        fi
        for ((poll = 1; poll <= 12; poll++)); do
            sleep 10
            STATE="$(squeue -j "$JOB" -h -o '%t %R' 2>/dev/null)"
            case "$STATE" in
                R*)
                    echo "[submit] shard $IDX RUNNING job=$JOB ${STATE#R }"
                    SUBMITTED+=("$JOB")
                    LANDED=1
                    break
                    ;;
                *JobHoldMaxRequeue*)
                    echo "[submit] shard $IDX job=$JOB spurious hold; cancelling and retrying"
                    scancel "$JOB" 2>/dev/null || true
                    break
                    ;;
                "")
                    echo "[submit] shard $IDX job=$JOB left the queue before running"
                    break
                    ;;
            esac
        done
        ((LANDED)) && break
    done
    ((LANDED)) || echo "[submit] shard $IDX FAILED to land after $SUBMIT_ATTEMPTS attempts"
done

echo "AGENTIC_SUBMITTED shards=${#SUBMITTED[@]} jobs=${SUBMITTED[*]:-none}"
echo "  plan   : $SHARD_DIR"
echo "  output : $OUT_DIR"
echo "  monitor: $PY scripts/agentic_progress.py --out-dir $OUT_DIR"
