#!/bin/bash
# First KernelBench number KORE has ever produced.
#
# One node, ONE GPU used for measurement at a time, four phases:
#   0  materialize the real KernelBench Level 1/2 problems into runnable KORE
#      task directories (probes each problem's ABI on the GPU)
#   1  CONTROL: hand-written Triton kernels for three Level-1 problems, measured
#      through the same path. Without this a fast_1 of 0 cannot be distinguished
#      from a harness that is unable to admit any kernel at all.
#   2  generate one kernel per problem from the model under test (one model load,
#      batched decode, nothing measured while 14B of weights are resident)
#   3  measure every completion through the real KoreEnv (verified correctness +
#      cold-cache paired timing) and emit the field-standard fast_p report
#
# Phases 1 and 3 are deliberately serial on a single device: they compare
# timings, so nothing else may run on the card while it is benching.
#
# BASELINE: torch-eager, KernelBench's baseline - NOT the AITER/hipBLASLt
# production baseline the rest of KORE grades against. The two are not
# comparable and the emitted report labels which one it used.
#
# Everything lands under runs/kernelbench_<jobid>/. No training is launched and
# nothing is written into runs/midtrain_*.
#SBATCH --job-name=kore-kernelbench
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/kernelbench-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/kernelbench-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
source /home/shasriva/kore-venv/bin/activate

# Positional args beat env vars: SPUR does not reliably propagate
# `--export KEY=VALUE` into a requeued child (same reason the training
# launchers are positional). Pass `-` to keep a default.
#   sbatch spur_kernelbench_1node.sbatch [MODEL] [ARM] [OUT] [LEVELS] [REVISION]
ARG_MODEL="${1:--}"; ARG_ARM="${2:--}"; ARG_OUT="${3:--}"
ARG_LEVELS="${4:--}"; ARG_REV="${5:--}"
_or() { [ "$1" = "-" ] || [ -z "$1" ] && printf '%s' "$2" || printf '%s' "$1"; }

OUT="$(_or "$ARG_OUT" "${KORE_KB_OUT:-$REPO/runs/kernelbench_${SLURM_JOB_ID:-manual}}")"
MODEL="$(_or "$ARG_MODEL" "${KORE_KB_MODEL:-Qwen/Qwen3-14B}")"
ARM="$(_or "$ARG_ARM" "${KORE_KB_ARM:-qwen3-14b}")"
LEVELS="$(_or "$ARG_LEVELS" "${KORE_KB_LEVELS:-1,2}")"
# Pinned so the arm names an immutable revision rather than a moving tag.
REVISION="$(_or "$ARG_REV" "${KORE_KB_REVISION:-40c069824f4251a91eefaf281ebe4c544efd3e18}")"

KB_ROOT="${KORE_KB_ROOT:-$HOME/third_party/KernelBench}"
TASKS_DIR="${KORE_KB_TASKS_DIR:-$HOME/third_party/kernelbench_tasks}"
MAX_TOKENS="${KORE_KB_MAX_TOKENS:-4096}"
BATCH="${KORE_KB_BATCH:-8}"
BUDGET="${KORE_KB_BUDGET:-1}"
mkdir -p "$OUT"

export PYTHONPATH=. PYTHONUNBUFFERED=1 GPU_TARGET=gfx950
export HF_HUB_OFFLINE=1
# Publication-grade paired timing + the verified-correctness prongs, matching
# the champion gate. Neither is relaxed anywhere in this job.
export KORE_BENCH_COLD=1 KORE_VERIFIED_CORRECTNESS=1
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

# PRIVATE MIOpen databases. KoreEnv points the driver subprocess's HOME at
# TMPDIR, so MIOpen puts its user/find databases in a world-shared /tmp path.
# On a node where another user created that file first it is not writable by us,
# MIOpen raises miopenStatusInternalError, and EVERY convolution problem
# disappears from the benchmark - torch's own conv baseline cannot run, so the
# loss is silent and biased (it takes out precisely the hard problems). Observed
# on crsuse2-m2m-011: 8+ Level-1 conv problems skipped this way.
export MIOPEN_USER_DB_PATH="$OUT/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$OUT/miopen"
# Same failure mode for the shared Triton/inductor cache, which also defaults
# into a world-shared /tmp directory.
export KORE_COMPILE_CACHE_DIR="$OUT/compile_cache"
mkdir -p "$MIOPEN_USER_DB_PATH" "$KORE_COMPILE_CACHE_DIR"

echo "[kb] HEAD=$(git rev-parse --short HEAD) host=$(hostname) out=$OUT"
echo "[kb] model=$MODEL@$REVISION arm=$ARM levels=$LEVELS"

if [[ ! -d "$KB_ROOT/.git" ]]; then
  echo "[kb] cloning KernelBench -> $KB_ROOT"
  mkdir -p "$(dirname "$KB_ROOT")"
  git clone https://github.com/ScalingIntelligence/KernelBench.git "$KB_ROOT" || exit 1
fi
echo "[kb] KernelBench revision $(git -C "$KB_ROOT" rev-parse HEAD)"

rc=0
run() { echo; echo "=== $* ==="; "$@" || { echo "[kb] FAILED: $*"; rc=1; }; }

# ------------------------------------------------------------------ phase 0 --
run env HIP_VISIBLE_DEVICES=0 $PY scripts/run_kernelbench_amd.py materialize \
    --kernelbench-root "$KB_ROOT" --out "$TASKS_DIR" --levels "$LEVELS" \
    --gpu-target gfx950
cp -f "$TASKS_DIR/manifest.json" "$OUT/manifest.json" 2>/dev/null

# ------------------------------------------------------------------ phase 1 --
# Measurement-path control, on the same device the model arm will be measured on.
run $PY scripts/run_kernelbench_amd.py selftest \
    --tasks-dir "$TASKS_DIR" --out "$OUT/generations_control.jsonl"
if [[ -f "$OUT/generations_control.jsonl" ]]; then
  run env HIP_VISIBLE_DEVICES=0 $PY scripts/run_kernelbench_amd.py measure \
      --tasks-dir "$TASKS_DIR" --generations "$OUT/generations_control.jsonl" \
      --out "$OUT/report_control" --budget 1 --mode parallel --no-replay
fi

# ------------------------------------------------------------------ phase 2 --
# The 14B fits on one MI350X with room for a batched KV cache; device_map="auto"
# across 8 cards would only add pipeline hops.
run env HIP_VISIBLE_DEVICES=0 $PY scripts/run_kernelbench_amd.py generate \
    --tasks-dir "$TASKS_DIR" --out "$OUT/generations_$ARM.jsonl" \
    --model "$MODEL" --revision "$REVISION" --arm "$ARM" --levels "$LEVELS" \
    --dtype bfloat16 --max-tokens "$MAX_TOKENS" --batch-size "$BATCH" \
    --temperature 0.0

# ------------------------------------------------------------------ phase 3 --
if [[ -f "$OUT/generations_$ARM.jsonl" ]]; then
  run env HIP_VISIBLE_DEVICES=0 $PY scripts/run_kernelbench_amd.py measure \
      --tasks-dir "$TASKS_DIR" --generations "$OUT/generations_$ARM.jsonl" \
      --out "$OUT/report_$ARM" --budget "$BUDGET" --mode parallel --no-replay
fi

# ------------------------------------------------------------------ phase 4 --
# The frontier comparator, through the same prompt contract, parser, verifier,
# budget and timing-integrity gate, measured on the SAME idle device right after
# the model arm. Skipped without failing the job when the gateway is down.
if [[ "${KORE_KB_OPUS:-1}" == "1" ]]; then
  $PY scripts/run_kernelbench_amd.py generate-opus \
      --tasks-dir "$TASKS_DIR" --out "$OUT/generations_opus.jsonl" \
      --levels "$LEVELS" --max-tokens "${KORE_KB_OPUS_MAX_TOKENS:-8192}" \
      --temperature 0.0 --concurrency "${KORE_KB_OPUS_CONCURRENCY:-16}"
  opus_rc=$?
  if [[ "$opus_rc" == "3" ]]; then
    echo "[kb] Opus arm skipped: gateway unavailable"
  elif [[ "$opus_rc" != "0" ]]; then
    echo "[kb] FAILED: generate-opus (rc=$opus_rc)"; rc=1
  fi
  if [[ -f "$OUT/generations_opus.jsonl" ]]; then
    run env HIP_VISIBLE_DEVICES=0 $PY scripts/run_kernelbench_amd.py measure \
        --tasks-dir "$TASKS_DIR" --generations "$OUT/generations_opus.jsonl" \
        --out "$OUT/report_opus" --budget "$BUDGET" --mode parallel --no-replay
    if [[ -f "$OUT/report_opus.json" && -f "$OUT/report_$ARM.json" ]]; then
      run $PY scripts/run_kernelbench_amd.py headtohead \
          --a-generations "$OUT/generations_$ARM.jsonl" \
          --a-measures "$OUT/report_$ARM.json" \
          --b-generations "$OUT/generations_opus.jsonl" \
          --b-measures "$OUT/report_opus.json" \
          --out "$OUT/headtohead_${ARM}_vs_opus"
    fi
  fi
fi

echo; echo "[kb] artifacts:"; ls -la "$OUT"
if [[ "$rc" == "0" ]]; then echo "[kb] DONE (all phases succeeded)"; else echo "[kb] DONE WITH FAILURES rc=$rc"; fi
exit "$rc"
