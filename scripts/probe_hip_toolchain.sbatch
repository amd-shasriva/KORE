#!/bin/bash
# Why can torch2hip not build, when hip2hip builds fine on the same node?
#
# The baseline control settled that this is not a model problem: all 57
# torch2hip tasks fail to compile even with their own shipped reference kernel,
# while all 32 hip2hip tasks compile. The two use different toolchains --
# hip2hip runs the task's own eval_tools/compile.py, torch2hip goes through
# torch.utils.cpp_extension, which needs ROCM_HOME and a hipcc on PATH. On the
# login node both are unset, but that node has no GPU and proves nothing.
#
# This answers it on the hardware the eval actually runs on, and tries the
# smallest real build, because "hipcc exists" and "an extension links" are
# different claims.
#SBATCH --job-name=kore-hipprobe
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/hipprobe-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/hipprobe-%j.err

set -uo pipefail
PY=/home/shasriva/kore-venv/bin/python
echo "host=$(hostname)"
echo "--- environment as the eval sees it ---"
for v in ROCM_HOME ROCM_PATH HIP_PATH PATH; do
    echo "  $v=${!v:-<unset>}"
done
echo "--- compilers on PATH ---"
for c in hipcc amdclang++ clang++ hipconfig ninja; do
    printf '  %-12s %s\n' "$c" "$(command -v $c || echo MISSING)"
done
echo "--- rocm install roots present on this node ---"
ls -d /opt/rocm* 2>/dev/null || echo "  no /opt/rocm*"
echo "--- what torch thinks ---"
"$PY" - <<'PY'
import torch, shutil
from torch.utils import cpp_extension as e
print("  torch", torch.__version__, "hip", torch.version.hip)
print("  cpp_extension.ROCM_HOME:", getattr(e, "ROCM_HOME", None))
print("  IS_HIP_EXTENSION:", getattr(e, "IS_HIP_EXTENSION", None))
print("  torch lib dir:", torch.utils.cmake_prefix_path)
PY
echo "--- smallest real HIP extension build (the actual question) ---"
"$PY" - <<'PY'
import tempfile, pathlib, traceback
from torch.utils.cpp_extension import load
src = '''
#include <torch/extension.h>
torch::Tensor forward(torch::Tensor x) { return x * 2; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
'''
d = pathlib.Path(tempfile.mkdtemp())
f = d / "probe.hip"
f.write_text(src)
try:
    m = load(name="kore_hip_probe", sources=[str(f)], verbose=True,
             build_directory=str(d))
    import torch
    print("  BUILD OK ->", m.forward(torch.ones(4, device="cuda")).sum().item())
except Exception:
    print("  BUILD FAILED:")
    traceback.print_exc()
PY
echo "done"
