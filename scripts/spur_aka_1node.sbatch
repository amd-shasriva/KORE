#!/bin/bash
# AgentKernelArena on a GPU node: discover | baseline | run.
#
# AKA ships a Docker runner and this cluster has no container runtime at all,
# so we drive the tasks' declared commands directly. They are plain argv, so the
# image was a reproducibility convenience rather than a dependency.
#
#   sbatch scripts/spur_aka_1node.sbatch <discover|baseline|run> [TYPES] [MODEL] [LIMIT] [ARM]
#SBATCH --job-name=kore-aka
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/aka-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/aka-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
source /home/shasriva/kore-venv/bin/activate
export PYTHONPATH="$REPO"
export HF_HOME=/home/shasriva/.cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

MODE="${1:-discover}"
TYPES="${2:-triton2triton}"     # '-' for all
MODEL="${3:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
LIMIT="${4:-0}"
ARM="${5:-kore}"
[ "$TYPES" = "-" ] && TYPES=""

OUT="$REPO/runs/aka_${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) mode=$MODE types=${TYPES:-ALL} model=$MODEL limit=$LIMIT arm=$ARM"
echo "arena=$HOME/third_party/AgentKernelArena  out=$OUT"

# The tasks' harnesses import torch/triton from OUR venv, so a missing GPU here
# shows up as every task failing correctness rather than as an obvious error.
VIS="$("$PY" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
echo "visible GPUs: $VIS"
[ "$VIS" = "0" ] && { echo "FATAL: no GPUs visible; every task would score 0"; exit 2; }

# shellcheck disable=SC2086
"$PY" scripts/run_agent_kernel_arena.py "$MODE" \
    --arena-root "$HOME/third_party/AgentKernelArena" \
    --gpu-arch gfx950 \
    ${TYPES:+--types $TYPES} \
    --out "$OUT" --arm "$ARM" --model "$MODEL" --limit "$LIMIT"
rc=$?
echo "aka rc=$rc"
exit $rc
