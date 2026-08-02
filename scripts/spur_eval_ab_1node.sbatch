#!/bin/bash
# First real evaluation of the Stage-0 mid-train checkpoint against its own base.
#
# One node, one GPU used at a time, three phases:
#   A  per arm, ONE model load: generate a kernel per held-out task (greedy,
#      matched budget) + teacher-forced held-out LM loss + retention smoke
#   B  measure BOTH arms' kernels back to back on the SAME idle device through
#      the real KoreEnv verifier + cold-cache paired bench
#   C  paired comparison + markdown reports
#
# Phase B is deliberately serial on one device: the two arms are compared on
# timings, so nothing else may run on the card while it is benching.
#
# Everything lands under runs/eval_ab_<jobid>/. Nothing is written into
# runs/sft_* and no training is launched.
#SBATCH --job-name=kore-eval-ab
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=03:30:00
#SBATCH --exclusive
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/eval-ab-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/eval-ab-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
source /home/shasriva/kore-venv/bin/activate

# Positional args take precedence over the env vars, because SPUR does not
# reliably propagate `--export KEY=VALUE` into a requeued child -- the same
# reason the training launchers are positional. Pass `-` to keep a default.
#   sbatch spur_eval_ab_1node.sbatch [CAND_MODEL] [CAND_ARM] [OUT]
ARG_CAND="${1:--}"; ARG_ARM="${2:--}"; ARG_OUT="${3:--}"
_or() { [ "$1" = "-" ] || [ -z "$1" ] && printf '%s' "$2" || printf '%s' "$1"; }

OUT="$(_or "$ARG_OUT" "${KORE_EVAL_AB_OUT:-$REPO/runs/eval_ab_${SLURM_JOB_ID:-manual}}")"
mkdir -p "$OUT"

BASE_MODEL="${KORE_AB_BASE_MODEL:-Qwen/Qwen3-14B}"
BASE_REV="${KORE_AB_BASE_REVISION:-40c069824f4251a91eefaf281ebe4c544efd3e18}"
CAND_MODEL="$(_or "$ARG_CAND" "${KORE_AB_CAND_MODEL:-$REPO/runs/midtrain_14b_1ep}")"
# The candidate arm is named so an SFT comparison does not file itself under
# "midtrain"; every downstream artifact and report keys off this string.
CAND_ARM="$(_or "$ARG_ARM" "${KORE_AB_CAND_ARM:-midtrain}")"
MAX_TOKENS="${KORE_AB_MAX_TOKENS:-4096}"
BATCH="${KORE_AB_BATCH:-12}"
BUDGET="${KORE_AB_BUDGET:-1}"

export PYTHONPATH=. PYTHONUNBUFFERED=1 GPU_TARGET=gfx950
export HF_HUB_OFFLINE=1
# Publication-grade paired timing for the bench, matching the champion gate.
export KORE_BENCH_COLD=1 KORE_VERIFIED_CORRECTNESS=1
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

echo "[eval-ab] HEAD=$(git rev-parse --short HEAD) host=$(hostname) out=$OUT"
echo "[eval-ab] base=$BASE_MODEL@$BASE_REV candidate=$CAND_MODEL"
$PY - <<'PY'
import torch
print(f"[eval-ab] torch {torch.__version__} devices={torch.cuda.device_count()}")
if torch.cuda.device_count():
    p = torch.cuda.get_device_properties(0)
    print(f"[eval-ab] device0 {p.name} arch={getattr(p, 'gcnArchName', '?')} "
          f"mem={p.total_memory / 2**30:.0f}GiB")
PY

rc=0
run() { echo; echo "=== $* ==="; "$@" || { echo "[eval-ab] FAILED: $*"; rc=1; }; }

# ---------------------------------------------------------------- phase A ----
# One GPU per arm; the model is 14B so a single MI355X holds it with room for a
# batched KV cache. device_map="auto" over 8 cards would only add pipeline hops.
for spec in "base:$BASE_MODEL:$BASE_REV" "$CAND_ARM:$CAND_MODEL:"; do
  ARM="${spec%%:*}"; rest="${spec#*:}"; MODEL="${rest%:*}"; REV="${rest##*:}"
  REVARG=(); [[ -n "$REV" ]] && REVARG=(--revision "$REV")
  run env HIP_VISIBLE_DEVICES=0 $PY -m kore.eval.checkpoint_ab run-arm \
      --arm "$ARM" --outdir "$OUT" --backend hf-batch --model "$MODEL" \
      "${REVARG[@]}" --dtype bfloat16 --max-tokens "$MAX_TOKENS" \
      --temperature 0.0 --batch-size "$BATCH" --retention mmlu,humaneval
done

# ---------------------------------------------------------------- phase B ----
# Both arms on the SAME device, back to back, nothing else resident.
for ARM in base "$CAND_ARM"; do
  [[ -f "$OUT/generations_$ARM.jsonl" ]] || { echo "[eval-ab] no generations for $ARM"; rc=1; continue; }
  run env HIP_VISIBLE_DEVICES=0 $PY -m kore.eval.checkpoint_ab measure \
      "$OUT/generations_$ARM.jsonl" --out "$OUT/measures_$ARM.json" \
      --arm "$ARM" --budget "$BUDGET" --mode parallel
done

# ---------------------------------------------------------------- phase C ----
if [[ -f "$OUT/measures_$CAND_ARM.json" && -f "$OUT/measures_base.json" ]]; then
  run $PY -m kore.eval.checkpoint_ab compare \
      --candidate "$OUT/measures_$CAND_ARM.json" \
      --reference "$OUT/measures_base.json" --out "$OUT/report_kernel_ab"
fi
if [[ -f "$OUT/lm_scores_$CAND_ARM.json" && -f "$OUT/lm_scores_base.json" ]]; then
  run $PY -m kore.eval.heldout_lm \
      --candidate "$OUT/lm_scores_$CAND_ARM.json" \
      --reference "$OUT/lm_scores_base.json" --out "$OUT/report_heldout_lm"
fi

echo; echo "[eval-ab] artifacts:"; ls -la "$OUT"
if [[ "$rc" == "0" ]]; then echo "[eval-ab] DONE (all phases succeeded)"; else echo "[eval-ab] DONE WITH FAILURES rc=$rc"; fi
exit "$rc"
