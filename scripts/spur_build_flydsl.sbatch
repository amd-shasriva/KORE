#!/bin/bash
# Build LLVM/MLIR and install FlyDSL into the KORE venv.
#
# FlyDSL is not optional scenery: 101 of the arena's 402 tasks are FlyDSL
# (triton2flydsl 51, torch2flydsl 45, flydsl2flydsl 5), and the package is an
# MLIR dialect, not pure Python -- `pip install` stops at
# "MLIR_PATH not set and could not be auto-detected". Without it those tasks can
# only be passed by writing Triton into a file named kernel.py and letting the
# correctness check pass, which is what the model is doing today: 35 of 51
# triton2flydsl "correct" with no FlyDSL involved and no speedup measured.
#
# It also gates datagen. We cannot generate verified FlyDSL training data without
# a toolchain to verify against, and unverified kernel data is precisely what the
# rest of this pipeline refuses to produce.
#
# Build products land on /shared_nfs, not /home: an LLVM build tree is tens of
# gigabytes and /home is a shared volume that has sat near full.
#
#   sbatch scripts/spur_build_flydsl.sbatch
#SBATCH --job-name=kore-flydsl
#SBATCH --account=amd-general
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-general-qos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/flydsl-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/flydsl-%j.err

set -uo pipefail
SRC=/home/shasriva/third_party/flydsl
PY=/home/shasriva/kore-venv/bin/python
PIP=/home/shasriva/kore-venv/bin/pip
export TMPDIR=/shared_nfs/shasriva/tmp
mkdir -p "$TMPDIR"

echo "[flydsl] host=$(hostname) cores=$(nproc) start=$(date -u +%H:%M:%S)"
cd "$SRC" || exit 2

if "$PY" -c "import flydsl" 2>/dev/null; then
    echo "[flydsl] already importable; nothing to do"
    exit 0
fi

# The project ships its own LLVM bootstrap and pins the revision it needs, so use
# it rather than guessing a system LLVM: MLIR's C++ API is unstable across
# releases and a mismatched one fails deep in the build.
if [ -x scripts/build_llvm.sh ] || [ -f scripts/build_llvm.sh ]; then
    echo "[flydsl] building LLVM/MLIR (this is the long part)"
    bash scripts/build_llvm.sh 2>&1 | tail -40
    rc=$?
    echo "[flydsl] build_llvm rc=$rc"
else
    echo "[flydsl] FATAL no scripts/build_llvm.sh; cannot bootstrap MLIR" >&2
    exit 3
fi

echo "[flydsl] installing package"
"$PIP" install -e . --no-build-isolation 2>&1 | tail -20
rc=$?

"$PY" -c "import flydsl; print('[flydsl] import OK')" 2>&1 | tail -3
echo "[flydsl] done rc=$rc end=$(date -u +%H:%M:%S)"
exit $rc
