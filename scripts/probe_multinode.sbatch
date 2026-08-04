#!/bin/bash
# What does SPUR actually give a multi-node batch job?
#
# Two behaviours differ from stock Slurm and both change how a distributed
# launcher must be written, so they are worth establishing by measurement rather
# than by reading someone's launcher and assuming:
#
#   1. How many times is the batch BODY executed -- once on the batch host, or
#      once per allocated node? Observed: per node, which means a naive
#      `srun` inside the body is issued N times over the same allocation.
#   2. Is SLURM_NODEID distinct per body copy? If it is, the body IS the
#      per-node worker and accelerate can take machine_rank=$SLURM_NODEID with
#      no srun at all, which is far simpler than fighting behaviour 1.
#
# Prints one line per body execution. Run it, read the output, then write the
# launcher to match what it says.
#SBATCH --job-name=kore-probe
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=4
#SBATCH --gres=gpu:mi355x:8
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=/shared_nfs/shasriva/kore/probe-%j.out
#SBATCH --error=/shared_nfs/shasriva/kore/probe-%j.err

echo "BODY host=$(hostname) NODEID=${SLURM_NODEID:-unset} PROCID=${SLURM_PROCID:-unset} NNODES=${SLURM_NNODES:-unset} NODELIST=${SLURM_JOB_NODELIST:-unset}"

# Does srun work from this copy, and what does it see?
srun --nodes="${SLURM_NNODES:-1}" --ntasks="${SLURM_NNODES:-1}" bash -c \
    'echo "SRUN host=$(hostname) NODEID=${SLURM_NODEID:-unset} PROCID=${SLURM_PROCID:-unset}"' \
    2>&1 | head -8
echo "SRUN_RC=$?"
