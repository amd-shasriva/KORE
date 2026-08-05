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
# LLVM+MLIR from source is hours even on 128 cores, and the run is worthless if
# it is killed at 95%: the install prefix is written last.
#SBATCH --time=16:00:00
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/flydsl-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/flydsl-%j.err

set -uo pipefail
SRC=/home/shasriva/third_party/flydsl
VENV=/home/shasriva/kore-venv
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
export TMPDIR=/shared_nfs/shasriva/tmp
mkdir -p "$TMPDIR"

# The build shells out to bare `python3` (to locate nanobind's cmake package) and
# to pip. Without the venv first on PATH those resolve to the system interpreter,
# where nanobind is absent and PEP 668 refuses installs -- which is how the first
# attempt died with "externally-managed-environment" after having already cloned
# 2.3GB of LLVM, a failure that looks like an LLVM problem and is not.
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"

# nanobind is a hard build dependency: scripts/build_llvm.sh asks python3 for its
# cmake directory and feeds that to the MLIR configure.
"$PIP" install -q nanobind 2>&1 | tail -2
"$PY" -c "import nanobind, os; print('nanobind cmake:', os.path.dirname(nanobind.__file__) + '/cmake')" \
    || { echo "[flydsl] FATAL nanobind unavailable; the MLIR configure cannot run" >&2; exit 4; }

echo "[flydsl] host=$(hostname) cores=$(nproc) start=$(date -u +%H:%M:%S)"
cd "$SRC" || exit 2

if "$PY" -c "import flydsl" 2>/dev/null; then
    echo "[flydsl] already importable; nothing to do"
    exit 0
fi

# The project ships its own LLVM bootstrap and pins the revision it needs, so use
# it rather than guessing a system LLVM: MLIR's C++ API is unstable across
# releases and a mismatched one fails deep in the build.
MLIR_INSTALL=/home/shasriva/third_party/llvm-project/mlir_install
if [ -d "$MLIR_INSTALL/lib/cmake/mlir" ]; then
    # Already built by an earlier attempt. Rebuilding costs hours and cannot
    # improve on a complete install, and this job has failed twice *after* this
    # point -- repeating the expensive half to reach the same broken half is the
    # slowest possible way to debug it.
    echo "[flydsl] reusing existing MLIR install at $MLIR_INSTALL"
else
    echo "[flydsl] building LLVM/MLIR (this is the long part)"
    bash scripts/build_llvm.sh 2>&1 | tail -40
    echo "[flydsl] build_llvm rc=$?"
fi
export MLIR_PATH="$MLIR_INSTALL"
echo "[flydsl] MLIR_PATH=$MLIR_PATH"

# The two prior failures were both "a command build.sh calls is not on PATH",
# reported only as exit 127 from three layers up inside pip. Run it directly with
# tracing first so the missing command is named instead of inferred.
echo "[flydsl] === tracing scripts/build.sh directly ==="
bash -x scripts/build.sh 2>&1 | tail -45
echo "[flydsl] build.sh rc=$?"

echo "[flydsl] installing package"
"$PIP" install -e . --no-build-isolation 2>&1 | tail -20
rc=$?

"$PY" -c "import flydsl; print('[flydsl] import OK')" 2>&1 | tail -3
echo "[flydsl] done rc=$rc end=$(date -u +%H:%M:%S)"
exit $rc
