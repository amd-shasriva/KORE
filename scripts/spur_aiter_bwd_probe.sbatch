#!/bin/bash
# Measure which backward kernel AITER dispatches to on gfx950, and time it.
#
# Runs on a compute node rather than the login node for the usual two reasons:
# the login node has no GPU, and it carries ~80 users at load 150+ with a
# per-process memory cap.
#
# Deliberately asks for ONE GPU and does NOT ask for the node exclusively. The
# account's concurrent-node QoS is saturated by datagen, so an exclusive 8-GPU
# request sits in PENDING(QOSGrpNodeLimit) indefinitely behind ~30 queued jobs,
# and this probe is single-device anyway. The cost of that choice is that the
# node may be shared, so the run records its co-tenants: the Phase A dispatch
# observation is a logical fact and immune to neighbours, but any Phase B
# timing measured next to another job has to be read with that in mind.
#
#   sbatch scripts/spur_aiter_bwd_probe.sbatch [JSON_OUT] [--phase-b]
#SBATCH --job-name=kore-aiterbwd
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
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

# Record who else is on this node, so a noisy timing row can be recognised as
# noisy later instead of being trusted at face value.
echo "=== co-tenants on $(hostname) ==="
squeue -w "$(hostname)" -h -o "%.10i %.16j %.8u %.6D" 2>&1 | head -20
echo "=== end co-tenants ==="

"$PY" scripts/aiter_bwd_dispatch_probe.py --out "$JSON_OUT" "$@"
rc=$?
echo "aiter_bwd_probe rc=$rc"
exit $rc
