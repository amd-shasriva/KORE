#!/bin/bash
# Dynamic, co-tenant-SAFE STANDALONE SFT gate: runs scripts/run_sft_gate.py (score
# base vs the finished SFT checkpoint + apply the retention gate, NO training, NO
# accelerate/FSDP) on currently-IDLE GPUs. The gate is GPU-aware (load_generate
# pins device_map to --gpu-ids), so it stays on the free GPUs; 2x 14B fits in a few
# GPUs. Re-picks idle GPUs + RESUMES via the per-benchmark score cache on any kill.
# On PASS it marks 'sft' done in the manifest and stops (never touches DPO).
#
# Env: GATE_NGPU=3 SFT_UTIL_MAX=20 SFT_VRAM_MAX_GB=8 SFT_MAX_RETRIES=48
set -u
REPO=/home/shasriva/Kore-RL/KORE
VENV=/home/shasriva/kore-venv/bin/python
cd "$REPO" || exit 1
mkdir -p runs/full/logs
export PATH="$(dirname "$VENV"):$PATH"

UTIL_MAX="${SFT_UTIL_MAX:-20}"
VRAM_MAX_GB="${SFT_VRAM_MAX_GB:-8}"
NGPU="${GATE_NGPU:-3}"           # 2x14B (base+candidate) fits comfortably; small footprint
MAX_RETRIES="${SFT_MAX_RETRIES:-48}"
WAIT_S="${SFT_WAIT_S:-120}"
COOLDOWN_S="${SFT_COOLDOWN_S:-60}"

pick_idle_gpus() {
  SFT_UTIL_MAX="$UTIL_MAX" SFT_VRAM_MAX_GB="$VRAM_MAX_GB" "$VENV" - <<'PY'
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

echo "SFT_GATE_SUPERVISOR start ngpu=${NGPU} util_max=${UTIL_MAX}% vram_max=${VRAM_MAX_GB}GB $(date)"
for attempt in $(seq 1 "$MAX_RETRIES"); do
  IDLE=$(pick_idle_gpus)
  if [ -z "$IDLE" ]; then
    echo "ALERT no idle GPUs; waiting ${WAIT_S}s [attempt $attempt] $(date)"
    sleep "$WAIT_S"; continue
  fi
  SEL=$(echo "$IDLE" | tr ',' '\n' | head -n "$NGPU" | paste -sd,)
  TS=$(date +%Y%m%d_%H%M%S)
  LOG="runs/full/logs/sft_gate_${TS}.log"
  echo "ALERT GATE_LAUNCH attempt=${attempt} idle=[${IDLE}] using=[${SEL}] log=$(basename "$LOG") $(date)"
  PYTHONPATH=. KORE_EVAL_FULL=1 "$VENV" scripts/run_sft_gate.py \
    --base Qwen/Qwen3-14B --candidate runs/full/sft --gpu-ids "$SEL" --mark-done \
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
