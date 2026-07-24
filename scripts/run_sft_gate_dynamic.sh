#!/bin/bash
# Dynamic, co-tenant-SAFE STANDALONE SFT gate: runs scripts/run_sft_gate.py (score
# base vs the finished SFT checkpoint + apply the retention gate, NO training, NO
# accelerate/FSDP) on currently-IDLE GPUs. The gate is GPU-aware (load_generate
# pins device_map to --gpu-ids), so it stays on the free GPUs; 2x 14B fits in a few
# GPUs. Re-picks idle GPUs + RESUMES via the per-benchmark score cache on any kill.
# On PASS it marks 'sft' done in the manifest and stops (never touches DPO).
#
# Env: GATE_NGPU=3 SFT_UTIL_MAX=20 SFT_VRAM_MAX_GB=8 SFT_MAX_RETRIES=48
#      GATE_GPUS=0,2,5  # optional: prefer this exact GPU set (still intersected with
#                       # the live-idle set for co-tenant safety - never stomps a busy GPU)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/ops_runtime.sh
source "$SCRIPT_DIR/lib/ops_runtime.sh"
kore_deprecated_guard \
  "scripts/run_sft_gate_dynamic.sh" \
  "run the retention gate in a scheduler allocation with explicit devices" \
  "bash scripts/run_sft_gate_dynamic.sh [--dry-run]" \
  "$@"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$(kore_resolve_python "$REPO")"
RUNTIME="$(kore_private_runtime)"
cd "$REPO"
mkdir -p runs/full/logs
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
kore_require_commands rocm-smi od stat
kore_export_rigor_env
RUN_ID="${KORE_RUN_ID:-$(kore_new_run_id sft-gate)}"

UTIL_MAX="${SFT_UTIL_MAX:-20}"
VRAM_MAX_GB="${SFT_VRAM_MAX_GB:-8}"
NGPU="${GATE_NGPU:-3}"           # 2x14B (base+candidate) fits comfortably; small footprint
GATE_GPUS="${GATE_GPUS:-}"       # optional preferred GPU set (intersected with idle)
EVAL_N="${KORE_EVAL_N:-300}"     # items/bench cap - matches run_campaign --eval-n (fast ~1.75h
                                 # gate); WITHOUT this the full 14k MMLU split makes it ~10h+
MAX_RETRIES="${SFT_MAX_RETRIES:-48}"
WAIT_S="${SFT_WAIT_S:-120}"
COOLDOWN_S="${SFT_COOLDOWN_S:-60}"

# Emits "<hip_csv>\t<phys_csv>" for idle GPUs. GATE_GPUS filters by PHYSICAL (rocm-smi)
# id; the emitted HIP ids are what HIP_VISIBLE_DEVICES needs (rocm-smi and HIP index
# orders DIFFER on this node - see scripts/gpu_pick_hip.py).
pick_idle_hip() {
  SFT_UTIL_MAX="$UTIL_MAX" SFT_VRAM_MAX_GB="$VRAM_MAX_GB" GATE_NGPU="$NGPU" \
  GATE_GPUS="$GATE_GPUS" "$PY" scripts/gpu_pick_hip.py 2>/dev/null
}

echo "SFT_GATE_SUPERVISOR start ngpu=${NGPU} util_max=${UTIL_MAX}% vram_max=${VRAM_MAX_GB}GB pref=[${GATE_GPUS:-any}] $(date)"
for attempt in $(seq 1 "$MAX_RETRIES"); do
  PICK=$(pick_idle_hip)
  HIP_SEL=$(printf '%s' "$PICK" | cut -f1)
  PHYS_SEL=$(printf '%s' "$PICK" | cut -f2)
  if [ -z "$HIP_SEL" ]; then
    echo "ALERT no idle GPUs (pref=[${GATE_GPUS:-any}]); waiting ${WAIT_S}s [attempt $attempt] $(date)"
    sleep "$WAIT_S"; continue
  fi
  TS=$(date +%Y%m%d_%H%M%S)
  LOG="runs/full/logs/sft_gate_${TS}.log"
  echo "ALERT GATE_LAUNCH attempt=${attempt} physical=[${PHYS_SEL}] hip=[${HIP_SEL}] log=$(basename "$LOG") $(date)"
  # Mask via HIP index in the ENV before python starts - the authoritative pin. torch
  # only ever sees these physical GPUs, so device_map="auto" cannot leak onto busy /
  # factory GPUs. HIP only (no ROCR) to avoid a broken composed remap. --gpu-ids ""
  # so run_sft_gate.py does not re-mask.
  set +e
  kore_owned_run "$PY" "$REPO" "$RUNTIME" "$RUN_ID" "sft-gate" "$LOG" \
    env HIP_VISIBLE_DEVICES="$HIP_SEL" PYTHONPATH="$REPO" KORE_EVAL_FULL=1 \
    KORE_EVAL_N="$EVAL_N" KORE_ALLOW_DEPRECATED_DEV=1 \
    "$PY" scripts/run_sft_gate.py \
      --base Qwen/Qwen3-14B --candidate runs/full/sft --gpu-ids "" --mark-done
  rc=$?
  set -e
  if [ "$rc" -eq 0 ] && kore_verify "$PY" "$REPO" sft-gate \
      --repo "$REPO" --manifest data/full14b/campaign_manifest.json \
      --candidate runs/full/sft; then
    echo "ALERT SFT_GATE_PASS attempt=${attempt} (strict artifacts verified) $(date)"
    exit 0
  fi
  if [ "$rc" -eq 1 ] || grep -q "RESULT: FAIL" "$LOG" 2>/dev/null; then
    echo "ALERT SFT_GATE_FAIL attempt=${attempt} rc=${rc} (real regression - inspect before proceeding) $(date)"
    exit 2
  fi
  echo "ALERT SFT_GATE_DIED attempt=${attempt} rc=${rc} - re-pick idle GPUs + resume (score cache) in ${COOLDOWN_S}s $(date)"
  sleep "$COOLDOWN_S"
done
echo "ALERT SFT_GATE_GIVEUP after ${MAX_RETRIES} attempts $(date)"
exit 6
