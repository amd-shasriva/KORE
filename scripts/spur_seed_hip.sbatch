#!/bin/bash
# Seed HIP twins on a compute node.
#
# Seeding is teacher latency plus a real CPU cost: admission instantiates every
# candidate module and runs it to check that supplying its weights from outside
# reproduces it. Across 13,570 pool modules with 16 workers that is not something
# to run on a shared login node, which everyone else needs interactive.
#
# No --gres: this never touches a GPU, so it does not spend one of the few
# concurrent GPU-job slots that gating and datagen compete for.
#
# Selector flags are passed as script arguments, not through --export. An
# --export list is comma-separated, so a value containing a space silently
# arrives empty -- which here meant the sweep ran with no selector at all and
# started seeding the wrong task set into the functionalized root.
#
#   sbatch scripts/spur_seed_hip.sbatch --functionalize --skip-parameter-free
#SBATCH --job-name=kore-seed
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/seed-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/seed-%j.err
cd /home/shasriva/Kore-RL/KORE
source /home/shasriva/kore-venv/bin/activate
export PYTHONPATH=/home/shasriva/Kore-RL/KORE

OUT="${SEED_OUT:-data/pool_hip_f}"
WORKERS="${SEED_WORKERS:-16}"
LIMIT="${SEED_LIMIT:-9000}"

echo "[seed] host=$(hostname) out=$OUT workers=$WORKERS args=$*"
if [ "$#" -eq 0 ]; then
    echo "[seed] refusing to run with no selector: an unselected sweep seeds the"
    echo "[seed] wrong task set into $OUT. Pass e.g. --functionalize." >&2
    exit 2
fi
# The attempt ledger in $OUT makes this restart-safe, so a requeue after
# preemption resumes the sweep instead of re-paying for seeds already written.
python scripts/materialize_pool_hip.py "$@" \
    --limit "$LIMIT" --workers "$WORKERS" --out "$OUT"
rc=$?

# Mark completion explicitly. The pipeline loop keeps this sweep staffed, so it
# needs to tell "finished the whole selection" apart from "not queued at this
# instant", which is also true between a preemption and the resubmission.
if [ "$rc" -eq 0 ]; then
    touch /home/shasriva/Kore-RL/KORE/runs/seeding.done
    echo "[seed] sweep complete; wrote runs/seeding.done"
fi
exit "$rc"
