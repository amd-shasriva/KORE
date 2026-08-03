#!/bin/bash
# Verify the HIP C++ task family on a real gfx950 compute node.
#
# The local dev host has 8x MI350X, so the HIP work was developed and measured
# there. This exists because the cluster is where datagen actually runs, and the
# login node is a CPU node with no hipcc at all -- so "HIP compiles here" is not
# evidence that "HIP compiles where the episodes are generated". If hipcc is
# absent on a compute node, all 20 HIP tasks fail there; they fail as INFRA errors
# rather than as model errors (kore/env/hip_toolchain.py), but they still produce
# no data, and that is worth knowing before a campaign is planned around them.
#
#   export SPUR_CONTROLLER_ADDR="http://crs-m2m-cpu-spur-005:6817"
#   sbatch scripts/spur_hip_verify_1node.sbatch [TASKS]
#
# Takes ONE node and releases it. Default is a 3-task sample (one per family plus
# both low-precision tasks); pass '-' for the whole HIP family.
#SBATCH --job-name=kore-hipverify
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/hipverify-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/hipverify-%j.err

set -uo pipefail
REPO="/home/shasriva/Kore-RL/KORE"
PY="/home/shasriva/kore-venv/bin/python"
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
# The venv bin must be on PATH for ninja; kore.env.hip_toolchain also re-adds it
# inside the driver subprocess, but making it explicit here keeps the failure
# legible if that ever regresses.
export PATH="/home/shasriva/kore-venv/bin:$PATH"

TASKS="${1:-hip_silu_bf16,hip_softmax_rows_bf16,hip_quant_fp8_pertoken_fp8_e4m3fn,hip_dequant_mxfp4_mxfp4}"
[ "$TASKS" = "-" ] && TASKS=""

echo "host=$(hostname)  date=$(date -u +%FT%TZ)"
echo "=== toolchain ==="
"$PY" -c "
from kore.env import hip_toolchain as h
s = h.probe_toolchain()
print('available :', s.available)
print('hipcc     :', s.hipcc)
print('ninja     :', s.ninja)
print('torch_hip :', s.torch_hip_version)
print('rocm_home :', s.rocm_home)
print('missing   :', s.missing, s.detail)
print('hipcc ver :', h.hipcc_version())
"
echo "=== device ==="
"$PY" -c "
import torch
print('torch', torch.__version__, 'hip', torch.version.hip)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_properties(i).gcnArchName)
"

echo "=== seed compile + SNR (op level) ==="
"$PY" scripts/verify_hip_seeds.py --gpu 0 --adversarial \
    --json "$REPO/runs/hipverify-${SLURM_JOB_ID:-manual}-seeds.json"
SEED_RC=$?

echo "=== end-to-end through KoreEnv ==="
if [ -n "$TASKS" ]; then
  "$PY" scripts/verify_hip_tasks_e2e.py --gpu 0 --verified-correctness \
      --tasks "$TASKS" \
      --json "$REPO/runs/hipverify-${SLURM_JOB_ID:-manual}-e2e.json"
else
  "$PY" scripts/verify_hip_tasks_e2e.py --gpu 0 --verified-correctness \
      --json "$REPO/runs/hipverify-${SLURM_JOB_ID:-manual}-e2e.json"
fi
E2E_RC=$?

echo "seed_rc=$SEED_RC e2e_rc=$E2E_RC"
exit $(( SEED_RC != 0 || E2E_RC != 0 ))
