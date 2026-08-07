#!/usr/bin/env python
"""Prove the arena's toolchain works on gfx950, one check per known failure mode.

Two arena categories scored zero on every task for reasons that had nothing to do
with the model, and neither was visible from the ledger:

  torch2flydsl   75 of 80 candidates compiled, then every shape failed with
                 "aiter gluon kernels require triton>=3.6.0, found 3.5.1". Two
                 triton distributions were installed -- pytorch-triton-rocm 3.5.1
                 and triton-rocm 3.6.0 -- and both dist-infos claimed ownership of
                 site-packages/triton, so the stray 3.5.1 had overwritten the
                 3.6.0 that torch itself pins (torch requires triton-rocm==3.6.0).

  torch2hip      0 of 62 candidates compiled. The task driver reports
                 "[ERROR] Compilation failed:" followed only by hipify's green
                 "Successfully preprocessed all matching files", because it prints
                 the wrong stream; torch's cpp_extension had raised
                 CalledProcessError from ninja and the actual compiler diagnostic
                 was never surfaced. The build directory is removed afterwards, so
                 the error cannot be recovered after the fact either.

Each check is independent and reports its own verdict, so a partial toolchain
gives a partial answer rather than one opaque failure. Run inside a GPU
allocation -- several checks need a real device and /opt/rocm, neither of which a
login node has.

    python scripts/verify_arena_toolchain.py
    python scripts/verify_arena_toolchain.py --keep-build-dir /tmp/hipcheck
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap

MIN_TRITON = "3.6.0"

# A HIP extension small enough that any failure is the toolchain's, not the
# kernel's. torch2hip candidates are compiled through exactly this path
# (torch.utils.cpp_extension.load), so a failure here reproduces the category's
# 0-of-62 result without needing a model in the loop.
_HIP_SRC = textwrap.dedent(
    """
    #include <torch/extension.h>
    #include <hip/hip_runtime.h>

    __global__ void add_one_kernel(const float* in, float* out, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = in[i] + 1.0f;
    }

    torch::Tensor add_one(torch::Tensor x) {
        auto out = torch::empty_like(x);
        int n = x.numel();
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(add_one_kernel, dim3(blocks), dim3(threads), 0, 0,
                           x.data_ptr<float>(), out.data_ptr<float>(), n);
        return out;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("add_one", &add_one, "add one");
    }
    """
).strip()

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""),
          flush=True)


def check_triton_version() -> None:
    """The gate that failed every torch2flydsl task."""
    try:
        import triton
        from packaging.version import Version
    except Exception as exc:  # noqa: BLE001 - absence is itself the verdict
        record("triton importable", False, f"{type(exc).__name__}: {exc}")
        return
    v = triton.__version__
    ok = Version(v.split("+")[0]) >= Version(MIN_TRITON)
    record("triton >= " + MIN_TRITON, ok, f"found {v}")


def check_single_triton_dist() -> None:
    """Two distributions owning site-packages/triton is what caused the clobber.

    Version alone cannot catch this: the losing distribution leaves its dist-info
    behind, so pip reports the version it believes is installed while the files on
    disk belong to the other one.
    """
    import importlib.metadata as md

    owners = []
    for dist in md.distributions():
        name = dist.metadata["Name"] or ""
        if "triton" not in name.lower():
            continue
        files = dist.files or []
        if any(str(f) == "triton/__init__.py" for f in files):
            owners.append(f"{name}=={dist.version}")
    record("exactly one triton distribution owns triton/", len(owners) == 1,
           ", ".join(owners) or "none found")


def check_torch_rocm() -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        record("torch importable", False, f"{type(exc).__name__}: {exc}")
        return
    hip = getattr(torch.version, "hip", None)
    record("torch is a ROCm build", bool(hip), f"torch {torch.__version__} hip {hip}")
    n = torch.cuda.device_count()
    record("GPUs visible", n > 0, f"{n} device(s)")
    if n:
        record("device is gfx950",
               "gfx950" in torch.cuda.get_device_properties(0).gcnArchName,
               torch.cuda.get_device_properties(0).gcnArchName)


def check_aiter_gluon() -> None:
    """torch2flydsl correctness runs through aiter; the gluon import gates it."""
    root = os.path.expanduser("~/third_party/aiter")
    if not os.path.isdir(root):
        record("aiter checkout present", False, root)
        return
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import aiter.ops.triton.gluon  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        record("aiter gluon imports", False, f"{type(exc).__name__}: {exc}")
        return
    record("aiter gluon imports", True)


def check_triton_jit() -> None:
    """Compiling is the part that matters; importing triton proves nothing."""
    try:
        import torch
        import triton
        import triton.language as tl
    except Exception as exc:  # noqa: BLE001
        record("triton kernel compiles", False, f"import failed: {exc}")
        return
    if not torch.cuda.is_available():
        record("triton kernel compiles", False, "no GPU visible")
        return

    @triton.jit
    def _add1(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        tl.store(y_ptr + off, tl.load(x_ptr + off, mask=mask) + 1.0, mask=mask)

    try:
        x = torch.ones(1024, device="cuda")
        y = torch.empty_like(x)
        _add1[(1,)](x, y, x.numel(), BLOCK=1024)
        torch.cuda.synchronize()
        ok = bool(torch.allclose(y, x + 1))
        record("triton kernel compiles and is correct", ok,
               "" if ok else "numerics wrong")
    except Exception as exc:  # noqa: BLE001
        record("triton kernel compiles and is correct", False,
               f"{type(exc).__name__}: {str(exc)[:300]}")


def check_arch_env() -> None:
    """The three variables AKA's setup_rocm_env exports and we did not.

    Without PYTORCH_ROCM_ARCH, torch falls back to a built-in arch list, and
    gfx950 is new enough that being on that list is not something to assume.
    """
    have = {v: os.environ.get(v, "") for v in
            ("PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS")}
    ok = all(have.values()) and len(set(have.values())) == 1
    record("ROCm arch env set and consistent", ok,
           ", ".join(f"{k}={v or '<unset>'}" for k, v in have.items()))


def _build_one(ext: str, keep: str | None) -> tuple[bool, str, str]:
    """Build the same HIP source under one file extension. Returns (ok, kind, msg).

    The extension is the whole experiment. torch decides whether to run the HIP
    toolchain by looking at the source's suffix -- ``.hip`` goes through hipcc with
    ROCm includes, while ``.cpp`` is compiled by plain c++ with only
    -D__HIP_PLATFORM_AMD__ defined and no -I for the ROCm headers. That is why a
    file containing <hip/hip_runtime.h> fails as .cpp and can succeed as .hip, and
    why testing only one of them tells you nothing about the other.
    """
    import torch
    from torch.utils.cpp_extension import load

    root = keep or tempfile.mkdtemp(prefix="kore_hipcheck_")
    build_dir = pathlib.Path(root) / ext.lstrip(".")
    build_dir.mkdir(parents=True, exist_ok=True)
    src = build_dir / f"add_one{ext}"
    src.write_text(_HIP_SRC)
    name = f"kore_hipcheck_{ext.lstrip('.')}"
    try:
        mod = load(name=name, sources=[str(src)],
                   build_directory=str(build_dir), verbose=False)
        if torch.cuda.is_available():
            x = torch.ones(64, device="cuda")
            if not bool(torch.allclose(mod.add_one(x), x + 1)):
                return False, "numerics wrong", ""
            return True, "built and correct on device", ""
        return True, "built (no GPU to run it)", ""
    except Exception as exc:  # noqa: BLE001 - the message IS the finding
        return False, type(exc).__name__, str(exc)


def check_hip_extension(keep: str | None) -> None:
    """Reproduce the torch2hip compile path under both source extensions.

    torch's cpp_extension raises a CalledProcessError whose message carries the
    ninja log; the task driver discards it and prints hipify's success line
    instead. Surfacing it is the point of this check -- "compilation failed" on
    its own is unactionable, and it cost 62 of 62 torch2hip candidates.
    """
    try:
        import torch  # noqa: F401
        from torch.utils.cpp_extension import load  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        record("HIP extension builds", False, f"import failed: {exc}")
        return

    for ext in (".hip", ".cpp"):
        ok, kind, msg = _build_one(ext, keep)
        record(f"HIP extension builds from {ext}", ok, kind)
        if not ok and msg:
            print(f"\n  ---- diagnostic for {ext} "
                  "(the part the task driver drops) ----", flush=True)
            for line in msg.splitlines()[:40]:
                print(f"    {line}", flush=True)
            print("", flush=True)


def check_hipcc() -> None:
    exe = None
    for cand in ("hipcc", "/opt/rocm/bin/hipcc"):
        if subprocess.run(["bash", "-lc", f"command -v {cand}"],
                          capture_output=True, text=True).returncode == 0:
            exe = cand
            break
    if not exe:
        record("hipcc on PATH", False, "not found (expected on a GPU node)")
        return
    out = subprocess.run(["bash", "-lc", f"{exe} --version"],
                         capture_output=True, text=True)
    first = (out.stdout or out.stderr).splitlines()[:1]
    record("hipcc on PATH", True, first[0] if first else exe)


def check_ninja() -> None:
    out = subprocess.run(["bash", "-lc", "ninja --version"],
                         capture_output=True, text=True)
    record("ninja available", out.returncode == 0,
           (out.stdout or out.stderr).strip()[:60])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-build-dir", default=None,
                    help="build the HIP extension here and leave it in place")
    args = ap.parse_args()

    print("=== arena toolchain verification ===", flush=True)
    check_triton_version()
    check_single_triton_dist()
    check_torch_rocm()
    check_arch_env()
    check_hipcc()
    check_ninja()
    check_triton_jit()
    check_aiter_gluon()
    check_hip_extension(args.keep_build_dir)

    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n=== {len(_results) - len(failed)}/{len(_results)} checks passed ===",
          flush=True)
    if failed:
        print("failed: " + ", ".join(failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
