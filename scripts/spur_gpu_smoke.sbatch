#!/bin/bash
# One-node GPU smoke for the integration-all merge on real gfx950 (MI355X).
# Validates: (1) KoreEnv label-minter build + 5-stage correctness + AITER baseline
# + lexicographic reward (smoke_env.py); (2) genb_* breadth seeds compile+correct+
# baseline (verify_breadth.py, sampled); (3) the accelerator-gated FSDP/bf16 tests
# that are skipped on CPU. Non-destructive. Submit from YOUR computer via spur.
#SBATCH --job-name=kore-gpu-smoke
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
# gpu:8, not gpu:mi355x:8 -- the typed form does not dispatch on this controller
# (JobLaunchFailure, 0 of 1 nodes confirmed). Measured 0/3 vs 2/3; see the full
# explanation in scripts/spur_sft_1node.sbatch.
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/gpu-smoke-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/gpu-smoke-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
mkdir -p runs
source /home/shasriva/kore-venv/bin/activate
if [[ -f "$REPO/.env.local" ]]; then set -a; source "$REPO/.env.local"; set +a; fi

# Full-node GRES exposes all 8 render devices; drop physical masks before KORE
# pins one logical HIP device per verifier child (same as the datagen sbatch).
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export PYTHONPATH=. PYTHONUNBUFFERED=1 GPU_TARGET=gfx950
export KORE_VERIFIED_CORRECTNESS=1 KORE_COMPILE_BASELINE=1 KORE_BENCH_COLD=1 KORE_SHAPE_AUGMENT=1

echo "[smoke] HEAD=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD) host=$(hostname)"
VIS="$($PY -c 'import torch; print(torch.cuda.device_count())')"
echo "[smoke] torch sees $VIS GPUs"
[[ "$VIS" == "8" ]] || { echo "[smoke] FATAL expected 8 GPUs"; exit 2; }

rc=0

echo "======================================================================"
echo "[smoke 1/3] KoreEnv verifier + reward on representative tasks"
echo "======================================================================"
# rmsnorm (norm/vendor), gemm (compute-bound), a genb breadth task, a MoE task.
SMOKE_TASKS="rmsnorm_aiter gemm_bf16 genb_gemm_bf16_bias_gelu_bf16 genb_moe_block_silu_bf16"
$PY scripts/smoke_env.py $SMOKE_TASKS || { echo "[smoke] smoke_env FAILED"; rc=1; }

echo "======================================================================"
echo "[smoke 2/3] genb_* breadth seed compile+correct+baseline (sampled: shard 0/16)"
echo "======================================================================"
HIP_VISIBLE_DEVICES=0 KORE_PY="$PY" $PY scripts/verify_breadth.py --nshards 16 --shard 0 \
    || { echo "[smoke] verify_breadth FAILED"; rc=1; }

echo "======================================================================"
echo "[smoke 3/3] accelerator-gated tests (FSDP plugin + bf16 TrainingArguments)"
echo "======================================================================"
# One GPU visible so accelerate/transformers can actually construct on device.
HIP_VISIBLE_DEVICES=0 $PY -m pytest \
    tests/test_grpo_fsdp.py::test_build_fsdp_plugin_full_shard \
    tests/test_grpo_fsdp.py::test_build_fsdp_plugin_autodetects_llama_layer \
    tests/test_distributed.py::test_training_arguments_accept_fsdp_kwargs \
    -q -p no:cacheprovider -o addopts= || { echo "[smoke] gated tests FAILED"; rc=1; }

echo "======================================================================"
if [[ "$rc" == "0" ]]; then echo "[smoke] ALL GPU SMOKE CHECKS PASSED"; else echo "[smoke] SMOKE FAILURES (rc=$rc)"; fi
exit "$rc"
