#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# launch_verl.sh — provision an isolated ROCm verl + SGLang box and launch a
# KORE multi-turn GRPO run (verl backend).
#
# WHY A SEPARATE VENV/BOX: verl + SGLang pin a different torch than KORE's
# *verified* env (the verifier/reward path). They could not co-install here
# (torch 2.11 vs the ROCm 2.10 wheel). So the RL trainer runs in its own venv
# (.venv-verl) on a dedicated ROCm box, while the verified reward
# (kore.reward.compute_reward over KoreEnv) is invoked as verl's custom reward
# function from *this* repo. See docs/rl_server.md for the full architecture.
#
# This script is idempotent: it (1) creates .venv-verl, (2) installs the AMD
# ROCm vLLM/SGLang wheel + verl from the ROCm wheel index, (3) starts the
# SGLang rollout server, and (4) runs `python -m verl.trainer.main_ppo` with
# the KORE GRPO config + reward via kore.policy.grpo.
#
# Usage:
#   bash scripts/launch_verl.sh \
#       --model Qwen/Qwen3-32B \
#       --tasks rmsnorm_aiter,gemm_bf16 \
#       --out runs/grpo \
#       --backend verl \
#       [--steps 500] [--tp 4] [--traj 16] [--turns 4] [--lora] \
#       [--rocm 6.3] [--sglang-port 30000] [--no-install] [--dry-run]
# ---------------------------------------------------------------------------
set -euo pipefail

# --- repo root (this script lives in <repo>/scripts) -----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- defaults --------------------------------------------------------------
MODEL="Qwen/Qwen3-32B"
TASKS=""
OUT="runs/grpo"
BACKEND="verl"
STEPS=""
TP=4
TRAJ=16
TURNS=4
USE_LORA=0
ROCM="6.3"
SGLANG_PORT=30000
VENV=".venv-verl"
DO_INSTALL=1
DRY_RUN=0

# ROCm wheel indices (AMD-published). Pin ROCM to match your driver stack.
ROCM_TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM}"
# SGLang + vLLM ROCm wheels are published on the AMD infinity-hub / GH releases;
# adjust to the exact wheel your box is qualified for (see docs/rl_server.md).
SGLANG_ROCM_WHEEL="sglang[srt]"
VLLM_ROCM_WHEEL="vllm"

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

# --- arg parse -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2"; shift 2;;
    --tasks)       TASKS="$2"; shift 2;;
    --out)         OUT="$2"; shift 2;;
    --backend)     BACKEND="$2"; shift 2;;
    --steps)       STEPS="$2"; shift 2;;
    --tp)          TP="$2"; shift 2;;
    --traj)        TRAJ="$2"; shift 2;;
    --turns)       TURNS="$2"; shift 2;;
    --lora)        USE_LORA=1; shift;;
    --rocm)        ROCM="$2"; ROCM_TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM}"; shift 2;;
    --sglang-port) SGLANG_PORT="$2"; shift 2;;
    --venv)        VENV="$2"; shift 2;;
    --no-install)  DO_INSTALL=0; shift;;
    --dry-run)     DRY_RUN=1; shift;;
    -h|--help)     usage 0;;
    *) echo "unknown arg: $1" >&2; usage 1;;
  esac
done

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" -eq 0 ]]; then "$@"; fi
}

echo "=== KORE verl launcher ==="
echo "repo=${REPO_ROOT} model=${MODEL} tasks=${TASKS:-<all>} out=${OUT}"
echo "backend=${BACKEND} tp=${TP} traj=${TRAJ} turns=${TURNS} lora=${USE_LORA} rocm=${ROCM}"

# --- 1. isolated venv ------------------------------------------------------
if [[ ! -d "${VENV}" ]]; then
  run python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
run python -m pip install --upgrade pip setuptools wheel

# --- 2. ROCm torch + SGLang/vLLM + verl -----------------------------------
if [[ "${DO_INSTALL}" -eq 1 ]]; then
  # ROCm-built torch (must match the driver stack; NOT the verified-env pin).
  run python -m pip install --index-url "${ROCM_TORCH_INDEX}" torch
  # SGLang (rollout server) + vLLM ROCm wheels.
  run python -m pip install "${SGLANG_ROCM_WHEEL}"
  run python -m pip install "${VLLM_ROCM_WHEEL}"
  # verl (RL trainer). Prefer the ROCm extra when publishing supports it.
  run python -m pip install "verl"
  # KORE itself (the reward/env/tasks package) — importable so verl's custom
  # reward function (kore.policy.grpo.kore_verl_reward) can be resolved.
  run python -m pip install -e "${REPO_ROOT}"
fi

# --- 3. sanity: verl must be importable now --------------------------------
if [[ "${DRY_RUN}" -eq 0 ]]; then
  python -c "import verl, sglang; print('verl', verl.__version__ if hasattr(verl,'__version__') else 'ok')"
fi

# --- 4. start the SGLang rollout server ------------------------------------
# For async multi-turn rollout verl can manage SGLang workers itself; you may
# alternatively run a standalone server (useful for shared serving / debugging):
SGLANG_LOG="${OUT}/sglang_server.log"
mkdir -p "${OUT}"
echo "--- starting SGLang rollout server on :${SGLANG_PORT} (log: ${SGLANG_LOG}) ---"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --tp "${TP}" \
    --port "${SGLANG_PORT}" \
    --host 127.0.0.1 \
    >"${SGLANG_LOG}" 2>&1 &
  SGLANG_PID=$!
  trap 'kill "${SGLANG_PID}" 2>/dev/null || true' EXIT
  # wait until the server answers /health
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${SGLANG_PORT}/health" >/dev/null 2>&1; then
      echo "SGLang server is up."; break
    fi
    sleep 5
  done
fi

# --- 5. run verl GRPO via KORE's config builder ----------------------------
# kore.policy.grpo._train_grpo_verl builds the verl config from GRPOConfig,
# materializes the dataset + resolved config, and invokes main_ppo. Driving it
# through KORE keeps the exact recipe (adv_estimator=grpo, rollout.n=traj,
# multi_turn + max_assistant_turns=turns, use_kl_loss + kl_loss_coef, Clip-Higher,
# SGLang rollout, tensor_parallel_size, LoRA/full-FT, verified reward fn).
LORA_KW=""
if [[ "${USE_LORA}" -eq 1 ]]; then LORA_KW="use_lora=True"; else LORA_KW="use_lora=False"; fi
STEPS_KW=""
if [[ -n "${STEPS}" ]]; then STEPS_KW="cfg.total_steps=${STEPS}"; fi

echo "--- launching verl GRPO (backend=${BACKEND}) ---"
PYPROG=$(cat <<PY
import os
from kore.policy.configs import GRPOConfig
from kore.policy.grpo import train_grpo

tasks = os.environ.get("KORE_TASKS", "").strip()
tasks = [t for t in tasks.split(",") if t] or None
cfg = GRPOConfig(
    model_id=os.environ["KORE_MODEL"],
    output_dir=os.environ["KORE_OUT"],
    tensor_parallel_size=int(os.environ["KORE_TP"]),
    num_trajectories=int(os.environ["KORE_TRAJ"]),
    num_turns=int(os.environ["KORE_TURNS"]),
    use_lora=os.environ["KORE_LORA"] == "1",
)
steps = os.environ.get("KORE_STEPS", "")
if steps:
    cfg.total_steps = int(steps)
print(train_grpo(cfg, tasks=tasks, backend=os.environ["KORE_BACKEND"]))
PY
)

if [[ "${DRY_RUN}" -eq 0 ]]; then
  KORE_MODEL="${MODEL}" KORE_TASKS="${TASKS}" KORE_OUT="${OUT}" \
  KORE_TP="${TP}" KORE_TRAJ="${TRAJ}" KORE_TURNS="${TURNS}" \
  KORE_LORA="${USE_LORA}" KORE_STEPS="${STEPS}" KORE_BACKEND="${BACKEND}" \
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python -c "${PYPROG}"
else
  echo "[dry-run] would run KORE train_grpo(backend=${BACKEND}) for model=${MODEL}"
fi

echo "=== done -> ${OUT} ==="
