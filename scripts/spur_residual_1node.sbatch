#!/bin/bash
# Verify or build an instruction-residual (chat-vector) transfer.
#
#   theta_out = theta_midtrain + (theta_instruct - theta_base)
#
# This runs under the scheduler rather than on the login node on purpose: it
# streams tens of GB through safetensors, and the login node routinely sits at
# load 150+ with 78 users, where even `import torch` fails to return.
#
# The work is pure tensor arithmetic, so this asks for CPUs and a single GPU
# rather than an exclusive 8-GPU node -- it schedules in a gap instead of
# queueing behind the training jobs.
#SBATCH --job-name=kore-residual
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/residual-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/residual-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
source /home/shasriva/kore-venv/bin/activate

# Positional, matching the other launchers, because SPUR does not reliably
# propagate --export into a requeued child. Pass `-` to keep a default.
#   sbatch spur_residual_1node.sbatch <verify|build> [TARGET] [OUT] [SCALE]
MODE="${1:-verify}"
ARG_TARGET="${2:--}"
ARG_OUT="${3:--}"
ARG_SCALE="${4:--}"

BASE_MODEL="Qwen/Qwen3-14B-Base"
INST_MODEL="Qwen/Qwen3-14B"
TARGET="${ARG_TARGET}"; [ "$TARGET" = "-" ] && TARGET="$REPO/runs/midtrain_14b_base"
OUT="${ARG_OUT}";       [ "$OUT" = "-" ]    && OUT="$REPO/runs/midtrain_14b_chatvec"
SCALE="${ARG_SCALE}";   [ "$SCALE" = "-" ]  && SCALE="1.0"

echo "host=$(hostname) mode=$MODE"
echo "base=$BASE_MODEL instruct=$INST_MODEL target=$TARGET out=$OUT scale=$SCALE"
df -BG --output=avail "$REPO" | tail -1 | xargs echo "free:"

if [ "$MODE" = "verify" ]; then
  # No --target: verification applies the residual to the BASE and requires the
  # INSTRUCT checkpoint back, bit for bit.
  "$PY" scripts/build_residual.py verify \
      --base "$BASE_MODEL" --instruct "$INST_MODEL"
  rc=$?
  echo "verify rc=$rc"
  exit $rc
fi

# Refuse to build on top of a target that is not finished: a half-written
# midtrain directory would produce a plausible-looking but wrong model.
if [ ! -f "$TARGET/model.safetensors.index.json" ]; then
  echo "REFUSING: $TARGET has no consolidated index yet"; exit 1
fi

"$PY" scripts/build_residual.py build \
    --base "$BASE_MODEL" --instruct "$INST_MODEL" \
    --target "$TARGET" --out "$OUT" --scale "$SCALE"
rc=$?
echo "build rc=$rc"
[ $rc -eq 0 ] && du -sh "$OUT"
exit $rc
