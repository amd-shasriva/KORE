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
SHARDS="${1:-4}"

# Both seed roots promote into one task root. Whether a seed's weights were
# passed in as arguments is a property of how it was written, not of how it is
# mined, so downstream sees a single flat set of HIP tasks.
ROOTS="${HIP_ROOTS:-data/pool_hip data/pool_hip_f}"

# Every destination is overridable, because there is now more than one set of
# twins and they must not be pooled. Promoting the registry's frontier twins
# into the same root as the 6,457 pool twins would leave them 5% of a shard set
# and mine the launch-bound ones ~19 times out of 20 -- the dilution the
# frontier selection exists to prevent.
PROMOTED="${HIP_PROMOTED:-$REPO/data/pool_hip_ok}"
DATA_ROOT="${HIP_DATA_ROOT:-$REPO/data/v5hippool}"
SHARD_DIR="${HIP_SHARD_DIR:-$REPO/runs/shards_hippool}"

# Which twin suffixes to harvest. The default matched only *__hip*, so a FlyDSL
# twin could be gated and then was invisible to every stage after it: the
# promote loop skipped it, the id list omitted it, and nothing ever mined one.
TWIN_GLOBS="${HIP_TWIN_GLOBS:-*__hip*}"

# Optional: keep only twins whose *source* task is in this list. A root can
# contain work that predates a narrowing of its selection -- the registry roots
# hold 740 twins seeded before they were restricted to the frontier 482 -- and
# promoting those would put them back into the mix.
TASK_LIST="${HIP_TASK_LIST:-}"

#: Twin dirs under a root, across every configured suffix. A bare glob appended
#: to a loop binds to the last pattern only, which is the same failure the seed
#: count below already had to work around.
list_twins() {
    local root="$1" g
    for g in $TWIN_GLOBS; do
        ls -d "$root"/tasks/$g 2>/dev/null
    done
}

cd "$REPO" || exit 1
. /etc/profile.d/spur.sh 2>/dev/null
. "$REPO/scripts/gpu_slots.sh"

# Count each root separately; a glob appended to a multi-line command
# substitution binds to the last path only and silently counts one root.
n_seeds=0
for r in $ROOTS; do
    n_seeds=$(( n_seeds + $(list_twins "$REPO/$r" | wc -l) ))
done
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
    got=$("$PY" - "$gj" "$REPO/$r" "$PROMOTED" "$TASK_LIST" <<'PY'
import json, re, shutil, sys
from pathlib import Path
gate, seeds, dst = (Path(p) for p in sys.argv[1:4])
listing = sys.argv[4] if len(sys.argv) > 4 else ""
if not gate.exists():
    print(0); raise SystemExit
wanted = None
if listing:
    p = Path(listing)
    if p.is_file():
        wanted = {ln.split("#", 1)[0].strip()
                  for ln in p.read_text().splitlines() if ln.split("#", 1)[0].strip()}
rows = json.loads(gate.read_text()).get("rows", [])
n = 0
for r in rows:
    # Trust the driver's own vocabulary: it prints a Python bool, not "pass".
    if r.get("status") != "pass" and "allclose: true" not in (r.get("error") or "").lower():
        continue
    tid = r.get("task_id")
    if not tid:
        continue
    if wanted is not None:
        base = re.sub(r"(__hipf|__hip|__flydsl)$", "", tid)
        if base not in wanted:
            continue
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
total=$(list_twins "$PROMOTED" | wc -l)
echo "[harvest] promoted total: $total"
[ "$total" -eq 0 ] && { echo "[harvest] no gated seeds yet; run the gate first"; exit 0; }

# --- shard and submit -------------------------------------------------------
TASK_IDS="${HIP_TASK_IDS_OUT:-$REPO/runs/hippool_tasks.txt}"
list_twins "$PROMOTED" | xargs -n1 basename > "$TASK_IDS"
# KORE_TASK_POOL points task resolution at the promoted root, so datagen can
# resolve these ids the same way it resolves the Triton pool.
export KORE_TASK_POOL="$PROMOTED"
PYTHONPATH="$REPO" "$PY" scripts/partition_any_tasks.py \
    --task-file "$TASK_IDS" \
    --out-dir "$SHARD_DIR" --data-root "$DATA_ROOT" \
    --shards "$SHARDS" --target 3 --skip-check 2>&1 | tail -4

# Submitting is not this script's job when NO_SUBMIT is set. The staffing script
# owns slot decisions, and having two submitters means a stream gets queued twice
# -- which on a pool running 35 jobs deep costs real queue position, since the
# duplicate has to be cancelled and the survivor keeps waiting.
if [ -n "${NO_SUBMIT:-}" ]; then
    echo "[harvest] promoted and partitioned; submission left to staff_datagen.sh"
    exit 0
fi
running=$(squeue -u "${USER:-$(id -un)}" -h -n kore-factory 2>/dev/null | wc -l)
echo "[harvest] kore-factory elements already up: $running"
sbatch ${QOS_ARG:-} --array=0-$((SHARDS - 1)) scripts/spur_datagen_array.sbatch \
    "$SHARD_DIR" "$DATA_ROOT" "$TARGET" run 2>&1 | tail -1
