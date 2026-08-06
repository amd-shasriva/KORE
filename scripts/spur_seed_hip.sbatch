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
#   SEED_ARGS="--functionalize --skip-parameter-free" sbatch scripts/spur_seed_hip.sbatch
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

echo "[seed] host=$(hostname) out=$OUT workers=$WORKERS args=${SEED_ARGS:-}"
# The attempt ledger in $OUT makes this restart-safe, so a requeue after
# preemption resumes the sweep instead of re-paying for seeds already written.
python scripts/materialize_pool_hip.py ${SEED_ARGS:-} \
    --limit "$LIMIT" --workers "$WORKERS" --out "$OUT"
