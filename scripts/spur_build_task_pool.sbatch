#!/bin/bash
# Build the external task pool on a COMPUTE node.
#
# Why this is a batch job and not a login-node command: it downloads two pinned
# HuggingFace datasets and then executes ~26,000 mined PyTorch modules to probe
# their shapes. The login node carries ~80 users at load 150+ and enforces a
# per-process memory cap, so the screening pass would either crawl or be reaped.
# Compute nodes have network, so the download works from here.
#
# CPU-only on purpose. The probe answers "does this module run, is it
# deterministic, and is its shape worth optimizing", none of which needs a GPU,
# and the GPUs are committed to training.
#
#   sbatch scripts/spur_build_task_pool.sbatch [OUT] [WORKERS] [SYNTH_COUNT]
#SBATCH --job-name=kore-taskpool
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/taskpool-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/taskpool-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
source /home/shasriva/kore-venv/bin/activate
export PYTHONPATH="$REPO"
export HF_HOME=/home/shasriva/.cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# The pinned datasets must be fetched, so this job -- unlike every training
# launcher here -- must not run offline.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

# Positional, matching the other launchers, because SPUR does not reliably
# propagate --export into a requeued child. Pass `-` to keep a default.
OUT="${1:--}";     [ "$OUT" = "-" ]     && OUT="$REPO/data/task_pool"
WORKERS="${2:--}"; [ "$WORKERS" = "-" ] && WORKERS=48
SYNTH="${3:--}";   [ "$SYNTH" = "-" ]   && SYNTH=9000

echo "host=$(hostname) out=$OUT workers=$WORKERS synthetic=$SYNTH"
df -BG --output=avail "$HOME" | tail -1 | xargs echo "free_before:"

"$PY" scripts/build_task_pool.py \
  --out "$OUT" \
  --workers "$WORKERS" \
  --synthetic-count "$SYNTH" \
  --materialize \
  --json-out "$REPO/runs/task_pool_report.json"
rc=$?

echo "task pool rc=$rc"
df -BG --output=avail "$HOME" | tail -1 | xargs echo "free_after:"
exit $rc
