#!/bin/bash
# Prove SHARDED_STATE_DICT save+resume on real GPUs before betting the v5 SFT run on it.
#
# Why this exists: the run's recovery story depends on resuming from a sharded
# checkpoint, and that had never been executed. The format it replaced saved
# perfectly well and could not be loaded back -- all eight ranks died with SIGBUS on
# a consolidated 244 GB optimizer -- so "the save worked" proves nothing. This
# submits the smoke test that actually resumes.
#
# Deliberately small and cheap so it can run on whatever is free while the real job
# waits in the amd-primus queue: 15 minutes, and it does NOT ask for --exclusive or
# all 8 GPUs, because a tiny random model needs neither and asking would put it in
# the same queue as the job it is meant to de-risk.
#
#   sbatch --account=amd-burst --qos=amd-burst-qos scripts/spur_smoke_sharded_resume.sbatch
#SBATCH --job-name=kore-smoke-resume
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --open-mode=append
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/smoke-resume-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/smoke-resume-%j.err
set -euo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
cd "$REPO"
export PATH="/home/shasriva/kore-venv/bin:$PATH"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/shasriva/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

echo "[smoke] host=$(hostname) job=${SLURM_JOB_ID:-none}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

# Model is randomly initialised, so a co-tenant's leftover VRAM is survivable here in
# a way it is not for a 61 GB load; report it rather than refusing the node.
python - <<'PY'
import torch
n = torch.cuda.device_count()
print(f"[smoke] visible GPUs: {n}")
for i in range(n):
    free, total = torch.cuda.mem_get_info(i)
    print(f"[smoke]   gpu{i}: {free/2**30:.0f}/{total/2**30:.0f} GiB free")
PY

# Keep the checkpoint on shared storage so its layout can be inspected afterwards.
export SMOKE_DIR=/shared_nfs/shasriva/kore/runs/smoke_sharded_resume_${SLURM_JOB_ID:-0}
export SMOKE_KEEP=1

accelerate launch --num_processes 2 --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no \
    "$REPO/scripts/smoke_sharded_resume.py"
rc=$?
echo "[smoke] SMOKE_RC=$rc"
exit $rc
