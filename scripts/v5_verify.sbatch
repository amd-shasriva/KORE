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
# amd-burst-qos, not amd-general-qos. General is capped at 8 nodes for the whole
# team and was full, so this pended on QOSGrpNodeLimit indefinitely; burst reaches
# all 282 nodes. Burst jobs can be preempted, which is irrelevant here -- this is a
# read-only verification that costs an hour and can simply be resubmitted.
#SBATCH --qos=amd-burst-qos
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
