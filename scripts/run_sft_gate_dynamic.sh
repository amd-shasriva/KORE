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
set -u
REPO=/home/shasriva/Kore-RL/KORE
VENV=/home/shasriva/kore-venv/bin/python
cd "$REPO" || exit 1
mkdir -p runs/full/logs
export PATH="$(dirname "$VENV"):$PATH"

UTIL_MAX="${SFT_UTIL_MAX:-20}"
VRAM_MAX_GB="${SFT_VRAM_MAX_GB:-8}"
NGPU="${GATE_NGPU:-3}"           # 2x14B (base+candidate) fits comfortably; small footprint
GATE_GPUS="${GATE_GPUS:-}"       # optional preferred GPU set (intersected with idle)
MAX_RETRIES="${SFT_MAX_RETRIES:-48}"
WAIT_S="${SFT_WAIT_S:-120}"
COOLDOWN_S="${SFT_COOLDOWN_S:-60}"

# Emits "<hip_csv>\t<phys_csv>" for idle GPUs. GATE_GPUS filters by PHYSICAL (rocm-smi)
# id; the emitted HIP ids are what HIP_VISIBLE_DEVICES needs (rocm-smi and HIP index
# orders DIFFER on this node - see scripts/gpu_pick_hip.py).
pick_idle_hip() {
  SFT_UTIL_MAX="$UTIL_MAX" SFT_VRAM_MAX_GB="$VRAM_MAX_GB" GATE_NGPU="$NGPU" \
  GATE_GPUS="$GATE_GPUS" "$VENV" scripts/gpu_pick_hip.py 2>/dev/null
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
  HIP_VISIBLE_DEVICES="$HIP_SEL" \
  PYTHONPATH=. KORE_EVAL_FULL=1 "$VENV" scripts/run_sft_gate.py \
    --base Qwen/Qwen3-14B --candidate runs/full/sft --gpu-ids "" --mark-done \
    > "$LOG" 2>&1
  rc=$?
  if grep -q "RESULT: PASS" "$LOG" 2>/dev/null; then
    echo "ALERT SFT_GATE_PASS attempt=${attempt} rc=${rc} (marked sft done; stopped before DPO) $(date)"
    exit 0
  fi
  if grep -q "RESULT: FAIL" "$LOG" 2>/dev/null; then
    echo "ALERT SFT_GATE_FAIL attempt=${attempt} rc=${rc} (real regression - inspect before proceeding) $(date)"
    exit 2
  fi
  echo "ALERT SFT_GATE_DIED attempt=${attempt} rc=${rc} - re-pick idle GPUs + resume (score cache) in ${COOLDOWN_S}s $(date)"
  sleep "$COOLDOWN_S"
done
echo "ALERT SFT_GATE_GIVEUP after ${MAX_RETRIES} attempts $(date)"
