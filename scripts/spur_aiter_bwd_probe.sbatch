#!/bin/bash
# Measure which backward kernel AITER dispatches to on gfx950, and time it.
#
# Runs on a compute node rather than the login node for the usual two reasons:
# the login node has no GPU, and it carries ~80 users at load 150+ with a
# per-process memory cap. Takes ONE node and releases it as soon as the probe
# finishes -- the QoS caps the account at 6 concurrent nodes and datagen wants
# most of them.
#
#   sbatch scripts/spur_aiter_bwd_probe.sbatch [JSON_OUT] [--phase-b]
#SBATCH --job-name=kore-aiterbwd
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/aiterbwd-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/aiterbwd-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
source /home/shasriva/kore-venv/bin/activate
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# Pin to a single GPU: the probe is single-device and a stray multi-GPU init
# only adds noise to a timing measurement.
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export HIP_VISIBLE_DEVICES=0

JSON_OUT="${1:--}"; [ "$JSON_OUT" = "-" ] && JSON_OUT="$REPO/runs/aiter_bwd_dispatch_${SLURM_JOB_ID:-local}.json"
shift || true

echo "host=$(hostname) out=$JSON_OUT extra=$*"
rocm-smi --showproductname 2>/dev/null | head -20

"$PY" scripts/aiter_bwd_dispatch_probe.py --out "$JSON_OUT" "$@"
rc=$?
echo "aiter_bwd_probe rc=$rc"
exit $rc
