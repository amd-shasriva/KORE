#!/bin/bash
# Prove the spec-synthesis task family on real gfx950, and measure whether
# Composable Kernel is even compilable on the nodes that generate episodes.
#
# Both halves have to run on a GPU node, and neither can be answered anywhere
# else: the login node runs no ROCm at all (no /opt/rocm, no hipcc), so "is CK
# available" is unanswerable there, and a task is only PROVEN once its oracle and
# timing protocol have run on the silicon the episodes will use.
#
# The CK probe is first and costs seconds. It is bundled here rather than given
# its own job because the QoS caps 6 concurrent nodes and a datagen campaign is
# already holding all of them; asking for a second node to run `ls` would be
# taking a node another agent needs.
#
#   sbatch scripts/spur_verify_spec_1node.sbatch
#SBATCH --job-name=kore-spec-verify
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
# gpu:8, not gpu:mi355x:8 -- the typed form does not dispatch on this controller
# (JobLaunchFailure, 0 of 1 nodes confirmed). Measured 0/3 vs 2/3; see the full
# explanation in scripts/spur_sft_1node.sbatch.
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/spec-verify-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/spec-verify-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
mkdir -p runs
source /home/shasriva/kore-venv/bin/activate
if [[ -f "$REPO/.env.local" ]]; then set -a; source "$REPO/.env.local"; set +a; fi

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 GPU_TARGET=gfx950
export KORE_COMPILE_BASELINE=1 KORE_BENCH_COLD=1

echo "[spec] HEAD=$(git rev-parse --short HEAD) host=$(hostname)"

echo "======================================================================"
echo "[1/3] Composable Kernel availability on an episode-generating node"
echo "======================================================================"
CK_JSON="$REPO/runs/ck_probe-${SLURM_JOB_ID:-local}.json"
bash scripts/probe_composable_kernel.sh "$CK_JSON"

echo "======================================================================"
echo "[2/3] spec-synthesis tasks: solvable AND unsolved by their stub"
echo "======================================================================"
# One logical device: the driver pins its own child, and the verification is
# sequential, so 8 visible devices would only add allocator noise to the timing.
HIP_VISIBLE_DEVICES=0 $PY scripts/verify_spec_tasks_e2e.py \
    --gpu 0 --json "$REPO/runs/spec_verification-${SLURM_JOB_ID:-local}.json"
rc=$?
echo "[spec] verify_spec_tasks_e2e rc=$rc"

echo "======================================================================"
echo "[3/3] the same tasks under the enumerated adversarial regimes"
echo "======================================================================"
# Correctness only: this pass exists to catch a spec whose prose admits a
# degenerate reading (a masked tail filled with the wrong identity, say), which
# the random battery can miss and the adversarial one is built to find.
HIP_VISIBLE_DEVICES=0 $PY scripts/verify_spec_tasks_e2e.py \
    --gpu 0 --no-bench --verified-correctness --skip-stub-check \
    --json "$REPO/runs/spec_verification_adversarial-${SLURM_JOB_ID:-local}.json"
arc=$?
echo "[spec] adversarial rc=$arc"

[[ "$rc" == "0" && "$arc" == "0" ]] && echo "[spec] ALL SPEC TASKS PROVEN" \
    || echo "[spec] FAILURES (verify=$rc adversarial=$arc)"
exit $(( rc || arc ))
