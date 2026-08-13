#!/bin/bash
# Gate one root of teacher-written HIP seeds on real gfx950.
#
# GATE_ROOT selects which root to gate, so the parameter-free seeds and the
# functionalized ones (whose weights arrive as trailing arguments) go through the
# same gate rather than needing a second script that could drift from this one.
#SBATCH --job-name=kore-hipgate
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
# Typed GRES. An untyped "gpu:8" request is accepted at submit time and then
# fails to bind on these nodes, so the job cycles through JobLaunchFailure into
# JobHoldMaxRequeue while the partition still shows dozens of idle nodes.
# gpu:8, not gpu:mi355x:8 -- the typed form does not dispatch on this controller
# (JobLaunchFailure, 0 of 1 nodes confirmed). Measured 0/3 vs 2/3; see the full
# explanation in scripts/spur_sft_1node.sbatch.
#SBATCH --gres=gpu:8
# The whole node's CPUs. Asking for a fraction alongside --exclusive is what this
# scheduler rejects at launch, and gating is hipcc-bound anyway, so the cores are
# the resource that actually makes it fast.
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/hipgate-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/hipgate-%j.err
cd /home/shasriva/Kore-RL/KORE
source /home/shasriva/kore-venv/bin/activate
export PYTHONPATH=/home/shasriva/Kore-RL/KORE
ROOT="${GATE_ROOT:-data/pool_hip}"
# Name the report after the root so gating a second root cannot overwrite the
# first one's verdicts, which harvest reads to decide what to promote.
TAG=$(basename "$ROOT")
echo "gating root=$ROOT -> runs/${TAG}_gate.json"
python scripts/verify_pool_hip_seeds.py --root "$ROOT" \
    --json "runs/${TAG}_gate.json" \
    --workers "${GATE_WORKERS:-8}" --resume
