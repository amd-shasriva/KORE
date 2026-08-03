#!/bin/bash
# Run a data audit on a COMPUTE node.
#
# This exists as a batch job rather than a login-node command on purpose: the
# login node carries ~80 users at load 150+ and enforces a per-process memory
# cap, so a pass over a 630MB mixture either crawls or gets reaped. It also runs
# no spurctld, which is why every SPUR CLI call needs an explicit controller
# address.
#
#   sbatch scripts/spur_audit_1node.sbatch [DATASET] [JSON_OUT]
#SBATCH --job-name=kore-audit
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/audit-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/audit-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
source /home/shasriva/kore-venv/bin/activate
export PYTHONPATH="$REPO"
export HF_HOME=/home/shasriva/.cache/huggingface
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

DATASET="${1:--}"; [ "$DATASET" = "-" ] && DATASET="$REPO/data/b05factory/sft/multicap.jsonl"
JSON_OUT="${2:--}"; [ "$JSON_OUT" = "-" ] && JSON_OUT="$REPO/runs/decontam_audit.json"

echo "host=$(hostname) dataset=$DATASET"
"$PY" scripts/audit_decontamination.py "$DATASET" --json-out "$JSON_OUT"
rc=$?
echo "audit rc=$rc"
exit $rc
