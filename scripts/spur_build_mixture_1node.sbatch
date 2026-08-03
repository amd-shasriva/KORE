#!/bin/bash
# Build the v2 SFT mixture on a COMPUTE node.
#
# Downloads and filters a HuggingFace dataset, so it needs network, disk and a
# real CPU budget -- none of which the login node can give it (80 users, load
# 150+, per-process memory cap). Note this is the one KORE job that must NOT run
# with HF_HUB_OFFLINE set, unlike the training launchers.
#
#   sbatch scripts/spur_build_mixture_1node.sbatch [--dry-run]
#SBATCH --job-name=kore-mixture
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/mixture-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/mixture-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
source /home/shasriva/kore-venv/bin/activate
export PYTHONPATH="$REPO"
export HF_HOME=/home/shasriva/.cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

echo "host=$(hostname)"
df -BG --output=avail "$HOME" | tail -1 | xargs echo "free:"

"$PY" scripts/build_sft_v2_mixture.py "$@"
rc=$?
echo "mixture rc=$rc"
exit $rc
