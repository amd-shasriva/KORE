#!/bin/bash
# Dynamic, co-tenant-SAFE SFT-finish for the shared node b05-1.
#
# Goal: run ONLY the campaign's SFT stage (resume checkpoint-351 [0 steps] + the
# retention gate) and STOP before DPO, while (a) NEVER interrupting the serving
# containers / other users, and (b) maximally using the *idle* capacity.
#
# How it stays safe + dynamic:
#   * Each (re)launch it re-reads rocm-smi and picks the currently-IDLE GPUs
#     (util <= SFT_UTIL_MAX% AND vram_used <= SFT_VRAM_MAX_GB), leaving SFT_RESERVE
#     GPUs free as headroom for co-tenants that may wake up.
#   * It MASKS the campaign to exactly those GPUs via ROCR_VISIBLE_DEVICES +
#     HIP_VISIBLE_DEVICES, so every process it spawns (FSDP ranks AND the gate's
#     vLLM/HF) physically cannot see - let alone touch - the busy GPUs.
#   * --stages build,sft => finishes SFT + the (now fixed + score-cached) retention
#     gate, then exits BEFORE dpo/grpo/soup/eval.
#   * On any death (SIGKILL from contention, a container waking up, etc.) it waits,
#     re-picks idle GPUs, and RESUMES - the per-benchmark retention-score cache means
#     the ~1.75h gate continues from where it left off instead of restarting.
#
# Env knobs (all optional): SFT_UTIL_MAX=20 SFT_VRAM_MAX_GB=8 SFT_RESERVE=1
#   SFT_MIN_GPUS=1 SFT_MAX_GPUS=8 SFT_MAX_RETRIES=48 SFT_WAIT_S=120 SFT_COOLDOWN_S=60
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/ops_runtime.sh
source "$SCRIPT_DIR/lib/ops_runtime.sh"
kore_deprecated_guard \
  "scripts/sft_finish_dynamic.sh" \
  "submit SFT through the site scheduler with an explicit allocation, then run the strict retention gate" \
  "bash scripts/sft_finish_dynamic.sh [--dry-run]" \
  "$@"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$(kore_resolve_python "$REPO")"
RUNTIME="$(kore_private_runtime)"
cd "$REPO"
mkdir -p runs/full/logs

# CRITICAL: put the venv bin on PATH so launch_distributed.sh's bare `accelerate`
# resolves. Running "$VENV" (the venv python) directly does NOT activate the venv,
# so without this the FSDP launch dies with exit 127 (accelerate: not found).
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
kore_secure_source_env "$REPO/.env.local"
kore_require_commands rocm-smi od stat
kore_export_rigor_env
RUN_ID="${KORE_RUN_ID:-$(kore_new_run_id sft-finish)}"

UTIL_MAX="${SFT_UTIL_MAX:-20}"
VRAM_MAX_GB="${SFT_VRAM_MAX_GB:-8}"
RESERVE="${SFT_RESERVE:-1}"
MIN_GPUS="${SFT_MIN_GPUS:-1}"
MAX_GPUS="${SFT_MAX_GPUS:-8}"
MAX_RETRIES="${SFT_MAX_RETRIES:-48}"
WAIT_S="${SFT_WAIT_S:-120}"
COOLDOWN_S="${SFT_COOLDOWN_S:-60}"

# Emit the comma-list of currently-idle physical GPU indices. Robust against a
# co-tenant container whose VRAM/util momentarily dips: samples TWICE (3s apart)
# and takes the MAX util + MAX vram per GPU, so a GPU with a resident model (or any
# transient spike) is never mistaken for idle. Thresholds are strict (a loaded
# serving model holds tens of GB, far above VRAM_MAX).
pick_idle_gpus() {
  SFT_UTIL_MAX="$UTIL_MAX" SFT_VRAM_MAX_GB="$VRAM_MAX_GB" "$PY" - <<'PY'
import os, re, subprocess, time
util_max = float(os.environ.get("SFT_UTIL_MAX", "20"))
vram_max = float(os.environ.get("SFT_VRAM_MAX_GB", "8")) * 1e9
def smi(args):
    try:
        return subprocess.run(["rocm-smi", *args], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""
util, vram = {}, {}
for s in range(2):
    if s:
        time.sleep(3)
    u = smi(["--showuse"]); m = smi(["--showmeminfo", "vram"])
    for ln in u.splitlines():
        mm = re.search(r"GPU\[(\d+)\].*?GPU use \(%\):\s*(\d+)", ln)
        if mm:
            g = int(mm.group(1)); util[g] = max(util.get(g, 0.0), float(mm.group(2)))
    for ln in m.splitlines():
        mm = re.search(r"GPU\[(\d+)\].*?Used Memory \(B\):\s*(\d+)", ln)
        if mm:
            g = int(mm.group(1)); vram[g] = max(vram.get(g, 0.0), float(mm.group(2)))
idle = [g for g in sorted(util) if util.get(g, 100) <= util_max and vram.get(g, 9e12) <= vram_max]
print(",".join(map(str, idle)))
PY
}

echo "SFT_FINISH_SUPERVISOR start util_max=${UTIL_MAX}% vram_max=${VRAM_MAX_GB}GB reserve=${RESERVE} $(date)"
for attempt in $(seq 1 "$MAX_RETRIES"); do
  IDLE=$(pick_idle_gpus)
  if [ -z "$IDLE" ]; then
    echo "ALERT no idle GPUs right now - waiting ${WAIT_S}s [attempt $attempt] $(date)"
    sleep "$WAIT_S"; continue
  fi
  # leave RESERVE idle GPUs for co-tenants; use the rest (clamped to [MIN,MAX]).
  mapfile -t _IDLE < <(echo "$IDLE" | tr ',' '\n' | grep -c . >/dev/null; echo "$IDLE" | tr ',' '\n')
  NUM_IDLE=${#_IDLE[@]}
  USE=$((NUM_IDLE - RESERVE))
  [ "$USE" -lt "$MIN_GPUS" ] && USE="$MIN_GPUS"
  [ "$USE" -gt "$MAX_GPUS" ] && USE="$MAX_GPUS"
  [ "$USE" -gt "$NUM_IDLE" ] && USE="$NUM_IDLE"
  SEL=$(printf '%s\n' "${_IDLE[@]}" | head -n "$USE" | paste -sd,)
  N=$(echo "$SEL" | tr ',' '\n' | grep -c .)
  TS=$(date +%Y%m%d_%H%M%S)
  LOG="runs/full/logs/sft_finish_${TS}.log"
  echo "ALERT LAUNCH attempt=${attempt} idle=[${IDLE}] using=[${SEL}] n=${N} log=$(basename "$LOG") $(date)"

  # Isolation on this shared node via --gpu-ids=$SEL (PHYSICAL idle GPUs), the design's
  # shared-node pinning lever:
  #  * FSDP SFT resume: run_campaign forwards GPU_IDS -> accelerate --gpu_ids (physical,
  #    authoritative) so the sharded workers run only on the idle GPUs.
  #  * retention GATE: run_campaign now passes gpu_ids to load_generate ->
  #    configure_rocm_env pins HF device_map="auto" to the idle GPUs before HIP init.
  # No parent VISIBLE_DEVICES mask (it fights accelerate's own --gpu_ids remap); both
  # paths target $SEL, so neither can touch a busy container GPU.
  COMMAND=("$PY" scripts/run_campaign.py --model Qwen/Qwen3-14B --full-ft --use-hf \
      --teacher claude --adaptive-steps --stages build,sft --sft-total 13000 \
      --gpu-ids "$SEL" --datagen-workers 16 --ground-reasoning \
      --profile-reward 0.15 --data-root data/full14b \
      --midtrain-out runs/full/midtrain --sft-out runs/full/sft \
      --dpo-out runs/full/dpo --grpo-out runs/full/grpo --soup-out runs/full/soup)
  set +e
  kore_owned_run "$PY" "$REPO" "$RUNTIME" "$RUN_ID" "sft-finish" "$LOG" \
    env KORE_VERIFIED_CORRECTNESS=1 KORE_COMPILE_BASELINE=1 \
    KORE_BENCH_COLD=1 KORE_SHAPE_AUGMENT=1 PYTHONPATH="$REPO" \
    "${COMMAND[@]}"
  rc=$?
  set -e

  if [ "$rc" -eq 0 ] && kore_verify "$PY" "$REPO" campaign \
      --repo "$REPO" --data-root data/full14b --required-stages build,sft; then
    echo "ALERT SFT_FINISH_COMPLETE attempt=${attempt} (strict artifacts verified) $(date)"
    exit 0
  fi
  if grep -qiE "retention[^\n]*(fail|regress|below)|gate[^\n]*fail|hard.?stop" "$LOG"; then
    echo "ALERT SFT_FINISH_GATE_FAIL attempt=${attempt} rc=${rc} $(date)" >&2
    exit 2
  fi
  echo "ALERT SFT_FINISH_DIED attempt=${attempt} rc=${rc} - re-pick idle GPUs + resume (gate cache) in ${COOLDOWN_S}s $(date)"
  sleep "$COOLDOWN_S"
done
echo "ALERT SFT_FINISH_GIVEUP after ${MAX_RETRIES} attempts $(date)"
exit 6
