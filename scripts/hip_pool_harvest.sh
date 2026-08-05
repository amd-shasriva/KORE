#!/bin/bash
# Turn gated HIP seeds into a running datagen sweep.
#
# The pool HIP path has three stages and only the middle one needs a GPU:
#
#   materialize_pool_hip.py   teacher writes a naive HIP seed  (teacher-bound)
#   verify_pool_hip_seeds.py  gfx950 says whether it compiles and is correct
#   this script               keep the passers, shard them, submit datagen
#
# Seeds accumulate continuously, so this is written to be re-run: it gates only
# what has not been gated, promotes the passers into a task root datagen can
# resolve, and re-partitions the whole promoted set each time.
#
# A seed that fails the gate is discarded rather than mined. Datagen against a
# task whose own seed cannot clear its SNR threshold can only ever score zero,
# and it reports as a model error rather than as the broken task it is.
#
#   scripts/hip_pool_harvest.sh [SHARDS]
set -uo pipefail

REPO=/home/shasriva/Kore-RL/KORE
PY=/home/shasriva/kore-venv/bin/python
PROMOTED="$REPO/data/pool_hip_ok"
DATA_ROOT="$REPO/data/v5hippool"
SHARD_DIR="$REPO/runs/shards_hippool"
SHARDS="${1:-4}"

# Both seed roots promote into one task root. Whether a seed's weights were
# passed in as arguments is a property of how it was written, not of how it is
# mined, so downstream sees a single flat set of HIP tasks.
ROOTS="${HIP_ROOTS:-data/pool_hip data/pool_hip_f}"

cd "$REPO" || exit 1
. /etc/profile.d/spur.sh 2>/dev/null

n_seeds=$(ls -d $(for r in $ROOTS; do echo "$REPO/$r/tasks"; done)/*__hip* 2>/dev/null | wc -l)
echo "[harvest] seeds available: $n_seeds"
[ "$n_seeds" -eq 0 ] && { echo "[harvest] nothing to do"; exit 0; }

# --- promote whatever the last gate approved --------------------------------
mkdir -p "$PROMOTED/tasks"
promoted=0
for r in $ROOTS; do
    # Each root's verdicts live in their own report, named after the root by the
    # gate job, so one root's gate cannot mask another's.
    gj="$REPO/runs/$(basename "$r")_gate.json"
    [ -f "$gj" ] || continue
    got=$("$PY" - "$gj" "$REPO/$r" "$PROMOTED" <<'PY'
import json, shutil, sys
from pathlib import Path
gate, seeds, dst = (Path(p) for p in sys.argv[1:4])
if not gate.exists():
    print(0); raise SystemExit
rows = json.loads(gate.read_text()).get("rows", [])
n = 0
for r in rows:
    # Trust the driver's own vocabulary: it prints a Python bool, not "pass".
    if r.get("status") != "pass" and "allclose: true" not in (r.get("error") or "").lower():
        continue
    tid = r.get("task_id")
    src = seeds / "tasks" / tid
    out = dst / "tasks" / tid
    if src.is_dir() and not out.exists():
        shutil.copytree(src, out)
        n += 1
print(n)
PY
)
    echo "[harvest]   $r -> promoted ${got:-0}"
    promoted=$(( promoted + ${got:-0} ))
done
echo "[harvest] promoted $promoted newly-passing seed(s)"
total=$(ls -d "$PROMOTED"/tasks/*__hip* 2>/dev/null | wc -l)
echo "[harvest] promoted total: $total"
[ "$total" -eq 0 ] && { echo "[harvest] no gated seeds yet; run the gate first"; exit 0; }

# --- shard and submit -------------------------------------------------------
ls -d "$PROMOTED"/tasks/*__hip* 2>/dev/null | xargs -n1 basename > "$REPO/runs/hippool_tasks.txt"
# KORE_TASK_POOL points task resolution at the promoted root, so datagen can
# resolve these ids the same way it resolves the Triton pool.
export KORE_TASK_POOL="$PROMOTED"
PYTHONPATH="$REPO" "$PY" scripts/partition_any_tasks.py \
    --task-file "$REPO/runs/hippool_tasks.txt" \
    --out-dir "$SHARD_DIR" --data-root "$DATA_ROOT" \
    --shards "$SHARDS" --target 3 --skip-check 2>&1 | tail -4

running=$(squeue -u "$USER" -h -n kore-factory 2>/dev/null | wc -l)
echo "[harvest] kore-factory elements already up: $running"
sbatch --array=0-$((SHARDS - 1)) scripts/spur_datagen_array.sbatch \
    "$SHARD_DIR" "$DATA_ROOT" 3 run 2>&1 | tail -1
