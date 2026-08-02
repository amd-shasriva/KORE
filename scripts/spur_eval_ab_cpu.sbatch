#!/bin/bash
# CPU-only half of the mid-train A/B: everything that does not need a GPU.
#
# Why this exists: on 2026-08-02 every GPU-requesting submission on amd-spur was
# instantly requeued into JobHoldMaxRequeue while CPU-only submissions ran
# normally, so the model-side measurements were moved onto a CPU allocation and
# only the kernel verify/bench phase was left waiting on a GPU. A 2x64-core EPYC
# 9575F node sustains ~4.3 fp32 TFLOP/s at 64 threads with 2.7 TB of RAM, which
# is ample for teacher-forced forward passes over ~30k tokens and workable for a
# batched greedy decode.
#
# Produces, per arm: lm_scores_<arm>.json, generations_<arm>.jsonl,
# retention_<arm>.json. The GPU verify/bench phase then runs over the SAME
# generations via scripts/spur_eval_ab_1node.sbatch (phase B) or
# `python -m kore.eval.checkpoint_ab measure`.
#SBATCH --job-name=kore-eval-ab-cpu
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=700G
#SBATCH --time=07:00:00
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/eval-ab-cpu-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/eval-ab-cpu-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
source /home/shasriva/kore-venv/bin/activate

OUT="${KORE_EVAL_AB_OUT:-$REPO/runs/eval_ab_cpu_${SLURM_JOB_ID:-manual}}"
mkdir -p "$OUT"

BASE_MODEL="${KORE_AB_BASE_MODEL:-Qwen/Qwen3-14B}"
BASE_REV="${KORE_AB_BASE_REVISION:-40c069824f4251a91eefaf281ebe4c544efd3e18}"
CAND_MODEL="${KORE_AB_CAND_MODEL:-$REPO/runs/midtrain_14b_frontier}"
MAX_TOKENS="${KORE_AB_MAX_TOKENS:-3072}"
BATCH="${KORE_AB_BATCH:-34}"
DTYPE="${KORE_AB_DTYPE:-bfloat16}"

THREADS="${SLURM_CPUS_PER_TASK:-64}"
export PYTHONPATH=. PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"

echo "[eval-ab-cpu] HEAD=$(git rev-parse --short HEAD) host=$(hostname) threads=$THREADS out=$OUT"
echo "[eval-ab-cpu] base=$BASE_MODEL@$BASE_REV candidate=$CAND_MODEL dtype=$DTYPE"
free -g | head -2

rc=0
for spec in "base:$BASE_MODEL:$BASE_REV" "midtrain:$CAND_MODEL:"; do
  ARM="${spec%%:*}"; rest="${spec#*:}"; MODEL="${rest%:*}"; REV="${rest##*:}"
  REVARG=(); [[ -n "$REV" ]] && REVARG=(--revision "$REV")
  echo; echo "=== arm $ARM ($MODEL) ==="
  $PY -m kore.eval.checkpoint_ab run-arm \
      --arm "$ARM" --outdir "$OUT" --backend hf-batch --model "$MODEL" \
      "${REVARG[@]}" --dtype "$DTYPE" --device-map cpu \
      --max-tokens "$MAX_TOKENS" --temperature 0.0 --batch-size "$BATCH" \
      --retention mmlu,humaneval || { echo "[eval-ab-cpu] arm $ARM FAILED"; rc=1; }
done

if [[ -f "$OUT/lm_scores_midtrain.json" && -f "$OUT/lm_scores_base.json" ]]; then
  echo; echo "=== held-out LM comparison ==="
  $PY -m kore.eval.heldout_lm --candidate "$OUT/lm_scores_midtrain.json" \
      --reference "$OUT/lm_scores_base.json" --out "$OUT/report_heldout_lm" \
      || { echo "[eval-ab-cpu] LM compare FAILED"; rc=1; }
fi

echo; echo "[eval-ab-cpu] artifacts:"; ls -la "$OUT"
if [[ "$rc" == "0" ]]; then echo "[eval-ab-cpu] DONE"; else echo "[eval-ab-cpu] DONE WITH FAILURES rc=$rc"; fi
exit "$rc"
