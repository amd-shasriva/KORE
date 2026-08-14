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
# NOTE: do NOT unset ROCR_VISIBLE_DEVICES here. spur_sft_1node.sbatch does, and that is
# correct THERE because it holds the whole node and every render device is bind-mounted,
# so dropping SPUR's list lets accelerate assign one logical device per rank. On a
# partial allocation the list is the only thing mapping our two cards, and clearing it
# left torch with 0 visible GPUs (measured, job 11376).

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

# Use the run's OWN accelerate config, overriding only the process count. Passing bare
# flags instead left distributed_type unset, so no rendezvous was configured and the
# trainer died on "environment variable RANK expected, but not set" (job 11376). Using
# the real file also means this test exercises the same FSDP wiring as the SFT run.
accelerate launch --config_file "$REPO/configs/accelerate_fsdp.yaml" \
    --num_processes 2 --num_machines 1 \
    "$REPO/scripts/smoke_sharded_resume.py"
rc=$?
echo "[smoke] SMOKE_RC=$rc"
exit $rc
