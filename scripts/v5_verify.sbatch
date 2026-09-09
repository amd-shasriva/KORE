#!/usr/bin/env bash
# Data verification on a compute node instead of the login node.
#
# The gates tokenize ~206k rows with the real tokenizer. On the login node that is
# one slow core against NFS and takes 10-15 minutes per pass; a compute node has 128
# and the tokenizer parallelises, so this is the difference between blocking a
# session and a background job. No GPU is requested -- nothing here needs one, and
# asking for eight would queue behind real training.
#SBATCH --job-name=kore-v5-verify
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
# amd-general-qos. This asked for amd-burst-qos to escape QOSGrpNodeLimit, on
# the reasoning that general is capped at 8 nodes for the team while burst
# reaches all 282. That reasoning was sound and the pairing does not work: one
# day later, docs/CLUSTER_OPERATIONS.md recorded amd-general + amd-burst-qos as
# accepted by sbatch and never scheduled -- Reason=None, StartTime=N/A for nine
# hours while 42 later-submitted jobs ran, and zero of 60 sampled running burst
# jobs used the amd-general account. Queueing on QOSGrpNodeLimit is a real wait
# that ends; not being a scheduling candidate is a wait that does not.
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/v5_verify_%j.out

set -uo pipefail
cd "${KORE_REPO:-/home/shasriva/Kore-RL/KORE}" || exit 1
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
echo "[v5-verify] node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} started=$(date -Is)"
bash scripts/v5_verify_all.sh
echo "[v5-verify] finished=$(date -Is)"
