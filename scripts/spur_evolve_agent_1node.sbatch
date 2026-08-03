#!/bin/bash
# Evolve-agent (kore.search.evolve_agent) on one GPU node.
#
# The loop's mutation operator is a live model, so this needs a generator the
# node can reach: either a served policy (--teacher vllm) or the API teacher
# (--teacher claude). The verifier work is a bench subprocess per candidate,
# pinned by HIP_VISIBLE_DEVICES, so one process owns one GPU and parallelism
# comes from --shard/--shards across jobs rather than from threads.
#
#   sbatch scripts/spur_evolve_agent_1node.sbatch [TASKS] [TEACHER] [MODEL] [GENERATIONS] [BUDGET] [GPU]
#
# QoS caps concurrent nodes, and datagen normally holds most of them: take ONE
# node and release it.
#SBATCH --job-name=kore-evolve
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/evolve-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/evolve-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
source /home/shasriva/kore-venv/bin/activate
export PYTHONPATH="$REPO"
export HF_HOME=/home/shasriva/.cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

TASKS="${1:-}"                  # comma-separated task ids; empty = whole registry
TEACHER="${2:-vllm}"
MODEL="${3:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
GENERATIONS="${4:-8}"
BUDGET="${5:-400}"
GPU="${6:-0}"

OUT="$REPO/runs/evolve_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"
echo "host=$(hostname) teacher=$TEACHER model=$MODEL generations=$GENERATIONS budget=$BUDGET gpu=$GPU"
echo "tasks=${TASKS:-ALL}  out=$OUT"

# The bench subprocess imports torch/triton from OUR venv, so a node with no
# visible GPU shows up as every candidate failing correctness rather than as an
# obvious error - and an evolve run of all-incorrect candidates still "finishes".
VIS="$("$PY" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
echo "visible GPUs: $VIS"
[ "$VIS" = "0" ] && { echo "FATAL: no GPUs visible; every candidate would score 0"; exit 2; }

# /home is shared and near capacity; a run that fills it corrupts other jobs'
# checkpoints, so refuse to start rather than find out at the first write.
FREE_GB="$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')"
echo "free on /home: ${FREE_GB}G"
[ "${FREE_GB:-0}" -lt 20 ] && { echo "FATAL: <20G free on /home; refusing to write"; exit 3; }

# shellcheck disable=SC2086
"$PY" scripts/run_evolve_agent.py \
    ${TASKS:+--tasks "$TASKS"} \
    --out "$OUT" \
    --teacher "$TEACHER" --model "$MODEL" \
    --generations "$GENERATIONS" --budget "$BUDGET" \
    --gpu "$GPU" \
    --seconds 25200
rc=$?
echo "evolve rc=$rc"
exit $rc
