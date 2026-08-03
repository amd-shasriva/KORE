"""HIP C++ backend support for the KORE task ABI.

Why this module exists
----------------------
Every task in the registry was ``backend: triton``, and the evaluation path was
Triton-only in five concrete, separately-fatal ways.  Each was measured on an
8x MI350X (gfx950) host, not inferred:

1. **The candidate had nowhere to land.**  :mod:`kore.env.kore_env` wrote the
   candidate to a hard-coded ``kernel.py``.  ``torch.utils.cpp_extension.load``
   dispatches the compiler off the file *extension*, so HIP C++ in a ``.py``
   file cannot be compiled at all.  :func:`candidate_filename` derives the name
   from the task's declared backend instead.

2. **Task assets were filtered to ``*.py``.**  A HIP task ships a ``.hip``
   baseline next to its Python oracle; the glob dropped it before the workdir
   was staged.

3. **Compiles took 115s instead of 15s.**  With ``PYTORCH_ROCM_ARCH`` unset,
   hipcc fat-binaries the kernel for every architecture the local ROCm supports.
   Measured on a trivial elementwise kernel: 114.6s unpinned, 15.4s pinned to
   ``gfx950`` -- a 7.5x difference that turns a normal compile into a timeout.
   :func:`compile_environment` pins it from the task's own ``gpu_target``.

4. **ninja was invisible, and its absence was blamed on the kernel.**
   ``torch.utils.cpp_extension`` shells out to ``ninja``; the console script
   lives in the venv's ``bin/``, which is *not* on ``PATH`` when the interpreter
   is invoked by absolute path (how datagen runs it).  The resulting
   ``RuntimeError: Ninja is required to load C++ extensions`` carries a
   traceback, so the environment's classifier read it as a *candidate compile
   failure*.  Every HIP episode would have scored zero and been recorded as the
   model's fault.  :data:`TOOLCHAIN_ABSENCE_PATTERN` makes toolchain absence an
   infra error, and :func:`compile_environment` puts ninja on ``PATH``.

5. **Concurrent builds could have swapped binaries.**  ``load(name=X)`` builds
   in ``TORCH_EXTENSIONS_DIR/X``.  Two workers compiling *different* candidates
   under the same name share that directory, so worker A can import worker B's
   ``.so`` -- a silent correctness corruption, the worst possible failure here.
   :func:`extension_name` derives the name from the SHA-256 of the source, so
   distinct sources can never collide and identical sources reuse the cache.

Nothing in this module weakens a gate.  It is imported by the *driver*
subprocess as well as the environment, so it stays free of heavy imports at
module scope: ``torch`` is imported inside the functions that need it.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

TRITON_BACKEND = "triton"
HIP_BACKEND = "hip"

#: Candidate artifact name per declared backend.  ``triton`` keeps ``kernel.py``
#: verbatim: 1,334 existing task drivers read that exact filename, so the Triton
#: path must be byte-identical to what it was.
CANDIDATE_FILENAMES: Mapping[str, str] = {
    TRITON_BACKEND: "kernel.py",
    HIP_BACKEND: "kernel.hip",
}
SUPPORTED_BACKENDS: frozenset[str] = frozenset(CANDIDATE_FILENAMES)

#: Source language per backend, for the language-aware reward-hack scan.
SOURCE_LANGUAGES: Mapping[str, str] = {
    TRITON_BACKEND: "python",
    HIP_BACKEND: "cpp",
}

#: Extensions staged into an evaluation workdir alongside ``*.py``.  Deliberately
#: a closed list: a task directory is copied into a location where a candidate
#: runs, so "everything in the directory" would be an unbounded surface.
STAGED_SUFFIXES: tuple[str, ...] = (".py", ".hip", ".cpp", ".cu", ".h", ".hpp", ".cuh")

#: Emitted by :func:`load_hip_candidate` when the HIP toolchain itself is
#: unusable.  The environment matches this to attribute the failure to infra
#: rather than to the candidate (see :data:`TOOLCHAIN_ABSENCE_PATTERN`).
TOOLCHAIN_MARKER = "KORE_HIP_TOOLCHAIN_UNAVAILABLE"

#: Infra-error evidence for a *toolchain* fault, never a kernel defect.  A
#: missing ninja/hipcc produces a Python traceback, which the environment's
#: compile-error branch would otherwise charge to the candidate.
TOOLCHAIN_ABSENCE_PATTERN = re.compile(
    TOOLCHAIN_MARKER
    + r"|Ninja is required to load C\+\+ extensions"
    + r"|No such file or directory: '(?:ninja|hipcc)'"
    + r"|(?:ninja|hipcc): (?:command )?not found"
    + r"|ROCM_HOME|Error checking compiler version for",
)

DEFAULT_EXTENSION_CACHE = "/tmp/kore_compile_cache"
#: Parallel hipcc jobs per driver subprocess.  Datagen runs up to 64 workers on
#: one node; an unbounded ``MAX_JOBS`` would fork-bomb the box and make every
#: measured time meaningless.
DEFAULT_MAX_JOBS = "4"


class HipToolchainError(RuntimeError):
    """The HIP toolchain is unusable, so no verdict can be produced."""


def normalize_backend(backend: Any) -> str:
    value = str(backend or "").strip().lower()
    return value or TRITON_BACKEND


def is_hip_backend(backend: Any) -> bool:
    return normalize_backend(backend) == HIP_BACKEND


def candidate_filename(backend: Any) -> str:
    """Candidate artifact name for ``backend`` (fail-closed on the unknown).

    An unrecognised backend raises rather than defaulting to ``kernel.py``: a
    silent default would stage a candidate the driver cannot compile and report
    it as the model's compile failure.
    """
    value = normalize_backend(backend)
    try:
        return CANDIDATE_FILENAMES[value]
    except KeyError:
        raise HipToolchainError(
            f"unsupported task backend {value!r}; known backends: "
            + ", ".join(sorted(SUPPORTED_BACKENDS))
        ) from None


def source_language(backend: Any) -> str:
    """Source language for ``backend``, for the language-aware hack scan."""
    return SOURCE_LANGUAGES.get(normalize_backend(backend), "python")


def candidate_filename_for_task(task: Any) -> str:
    return candidate_filename(getattr(task, "backend", TRITON_BACKEND))


def gpu_arch(gpu_target: Any) -> str:
    """Bare architecture from a target string.

    ROCm reports ``gfx950:sramecc+:xnack-``; ``PYTORCH_ROCM_ARCH`` wants the
    bare ``gfx950``, and passing the feature-decorated form makes hipcc fail.
    """
    value = str(gpu_target or "").strip()
    return value.split(":", 1)[0] if value else ""


def detected_gpu_arch() -> str:
    """Architecture of the visible device, or ``""`` when there is none.

    The fallback for a driver run without a declared target.  Getting this wrong
    costs a 7.5x compile slowdown rather than a wrong answer, so it degrades
    quietly.
    """
    try:
        import torch  # noqa: PLC0415 - deliberately lazy

        if not torch.cuda.is_available():
            return ""
        return gpu_arch(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    except Exception:  # noqa: BLE001
        return ""


@dataclass(frozen=True)
class ToolchainStatus:
    """What the HIP toolchain can actually do on this host."""

    available: bool
    hipcc: Optional[str] = None
    ninja: Optional[str] = None
    torch_hip_version: Optional[str] = None
    rocm_home: Optional[str] = None
    missing: tuple[str, ...] = ()
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "hipcc": self.hipcc,
            "ninja": self.ninja,
            "torch_hip_version": self.torch_hip_version,
            "rocm_home": self.rocm_home,
            "missing": list(self.missing),
            "detail": self.detail,
        }


def script_dirs() -> tuple[str, ...]:
    """Directories that hold this interpreter's console scripts.

    ``Path(sys.executable).resolve()`` is deliberately NOT used: in this venv
    ``bin/python`` symlinks out to ``/usr/local/bin/python3.10``, so resolving it
    lands outside the venv and loses ``bin/ninja`` entirely.  ``sys.prefix`` and
    the *unresolved* executable path both stay inside it.
    """
    out: list[str] = []
    for candidate in (
        Path(sys.executable).parent,
        Path(sys.prefix) / "bin",
        Path(sysconfig.get_path("scripts") or ""),
    ):
        text = str(candidate)
        if text and text not in out and candidate.is_dir():
            out.append(text)
    return tuple(out)


def _which(name: str) -> Optional[str]:
    """Locate ``name``, including the interpreter's own script directories.

    ``shutil.which`` alone misses the venv console scripts when the interpreter
    was invoked by absolute path, which is exactly how datagen starts drivers.
    """
    found = shutil.which(name)
    if found:
        return found
    for directory in script_dirs():
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


@lru_cache(maxsize=1)
def probe_toolchain() -> ToolchainStatus:
    """Detect whether HIP C++ can be compiled here.  Never raises."""
    missing: list[str] = []
    details: list[str] = []

    ninja = _which("ninja")
    if not ninja:
        missing.append("ninja")
        details.append("ninja executable not found (pip install ninja)")

    torch_hip: Optional[str] = None
    rocm_home: Optional[str] = None
    try:
        import torch  # noqa: PLC0415 - deliberately lazy
        from torch.utils import cpp_extension

        torch_hip = getattr(torch.version, "hip", None)
        rocm_home = getattr(cpp_extension, "ROCM_HOME", None)
        if not torch_hip:
            missing.append("torch-hip")
            details.append("this torch build has no HIP runtime (torch.version.hip is unset)")
    except Exception as exc:  # noqa: BLE001 - absence is a status, not a crash
        missing.append("torch")
        details.append(f"torch/cpp_extension unimportable: {type(exc).__name__}: {exc}")

    hipcc = _which("hipcc")
    if not hipcc and rocm_home:
        candidate = Path(str(rocm_home)) / "bin" / "hipcc"
        if candidate.is_file():
            hipcc = str(candidate)
    if not hipcc:
        missing.append("hipcc")
        details.append("hipcc not found on PATH or under ROCM_HOME")

    return ToolchainStatus(
        available=not missing,
        hipcc=hipcc,
        ninja=ninja,
        torch_hip_version=torch_hip,
        rocm_home=str(rocm_home) if rocm_home else None,
        missing=tuple(missing),
        detail="; ".join(details),
    )


def hipcc_version(status: Optional[ToolchainStatus] = None, timeout: int = 30) -> str:
    """``hipcc --version`` first line, or ``""`` when it cannot be read."""
    status = status or probe_toolchain()
    if not status.hipcc:
        return ""
    try:
        proc = subprocess.run(
            [status.hipcc, "--version"], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in (proc.stdout or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def extension_cache_root(env: Optional[Mapping[str, str]] = None) -> str:
    src = env if env is not None else os.environ
    root = str(src.get("KORE_COMPILE_CACHE_DIR") or DEFAULT_EXTENSION_CACHE)
    return os.path.join(root, "hip_ext")


def compile_environment(
    env: Optional[dict] = None,
    gpu_target: Any = None,
) -> dict:
    """Augment ``env`` so a driver subprocess can compile HIP for ``gpu_target``.

    Uses ``setdefault`` throughout so an explicit parent setting always wins;
    the only unconditional edit is prepending the interpreter's ``bin/`` to
    ``PATH``, which is what makes ninja reachable.
    """
    env = dict(os.environ if env is None else env)

    path = env.get("PATH") or os.defpath
    entries = path.split(os.pathsep)
    missing = [d for d in script_dirs() if d not in entries]
    if missing:
        env["PATH"] = os.pathsep.join([*missing, path])

    arch = gpu_arch(gpu_target) or gpu_arch(env.get("PYTORCH_ROCM_ARCH"))
    if arch:
        # The single highest-impact setting here: 15.4s vs 114.6s per compile.
        env.setdefault("PYTORCH_ROCM_ARCH", arch)

    cache = extension_cache_root(env)
    env.setdefault("TORCH_EXTENSIONS_DIR", cache)
    env.setdefault("MAX_JOBS", DEFAULT_MAX_JOBS)
    try:
        os.makedirs(cache, exist_ok=True)
    except OSError:
        # A read-only cache location is not fatal; torch falls back and the
        # compile merely loses its warm-start.
        pass
    return env


def extension_name(source: str) -> str:
    """Collision-free extension name derived from the source itself.

    Content-addressed on purpose: ``load(name=N)`` builds in
    ``TORCH_EXTENSIONS_DIR/N``, so a name shared by two different sources lets
    one worker import another's binary.  Hashing the source makes that
    impossible while still letting identical sources hit the warm cache.
    """
    digest = hashlib.sha256(source.encode("utf-8", errors="replace")).hexdigest()
    return f"kore_hip_{digest[:32]}"


def _require_toolchain() -> ToolchainStatus:
    status = probe_toolchain()
    if not status.available:
        message = (
            f"{TOOLCHAIN_MARKER}: missing {', '.join(status.missing)}"
            f"{'; ' + status.detail if status.detail else ''}"
        )
        print(message, flush=True)
        raise HipToolchainError(message)
    return status


def compile_hip_source(
    source_path: str | os.PathLike,
    *,
    gpu_target: Any = None,
    extra_cflags: Optional[list[str]] = None,
):
    """JIT-compile a HIP source file and return the loaded extension module.

    Raises :class:`HipToolchainError` when the *toolchain* is unusable (an infra
    fault) and lets ``torch``'s own ``RuntimeError`` propagate when hipcc
    rejects the source (a candidate fault).  Keeping those two apart is the
    whole point: conflating them is what reports a broken node as a 97% model
    error rate.
    """
    _require_toolchain()
    from torch.utils.cpp_extension import load  # noqa: PLC0415 - lazy

    path = Path(source_path).resolve()
    if not path.is_file():
        raise HipToolchainError(f"{TOOLCHAIN_MARKER}: no HIP source at {path}")
    source = path.read_text(errors="replace")

    name = extension_name(source)
    # KoreEnv._env already prepared the parent environment; re-derive the values
    # a hand-run driver would otherwise be missing, and fall back to the visible
    # device's own architecture so an un-pinned run still compiles in ~15s.
    arch = gpu_arch(gpu_target) or os.environ.get("PYTORCH_ROCM_ARCH") or detected_gpu_arch()
    if arch:
        os.environ["PYTORCH_ROCM_ARCH"] = gpu_arch(arch)
    path = os.environ.get("PATH") or os.defpath
    missing = [d for d in script_dirs() if d not in path.split(os.pathsep)]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, path])
    os.environ.setdefault("MAX_JOBS", DEFAULT_MAX_JOBS)

    build_dir = Path(extension_cache_root()) / name
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", extension_cache_root())

    # Build from a copy inside the (shared, content-addressed) build directory so
    # a throwaway workdir being deleted cannot invalidate the cached artifact.
    staged = build_dir / path.name
    if not staged.is_file() or staged.read_text(errors="replace") != source:
        staged.write_text(source)

    return load(
        name=name,
        sources=[str(staged)],
        build_directory=str(build_dir),
        extra_cflags=extra_cflags or [],
        verbose=False,
    )


def load_hip_candidate(task_dir: str | os.PathLike, entry: str, *, gpu_target: Any = None):
    """Compile the staged HIP candidate and return its ``entry`` symbol.

    The returned callable comes from a *cached* module, matching the Triton
    loader's contract: a stateful kernel's globals must persist from the timed
    loop into the post-timing re-verification, or the invocation-count timing
    hack becomes invisible.
    """
    path = Path(task_dir) / CANDIDATE_FILENAMES[HIP_BACKEND]
    module = compile_hip_source(path, gpu_target=gpu_target)
    if not hasattr(module, entry):
        raise AttributeError(
            f"HIP candidate {path.name} exports no {entry!r} symbol; a HIP task "
            f"must bind it with PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) "
            f"{{ m.def(\"{entry}\", ...); }}"
        )
    return getattr(module, entry)


__all__ = [
    "CANDIDATE_FILENAMES",
    "DEFAULT_MAX_JOBS",
    "HIP_BACKEND",
    "HipToolchainError",
    "SOURCE_LANGUAGES",
    "STAGED_SUFFIXES",
    "SUPPORTED_BACKENDS",
    "TOOLCHAIN_ABSENCE_PATTERN",
    "TOOLCHAIN_MARKER",
    "TRITON_BACKEND",
    "ToolchainStatus",
    "candidate_filename",
    "candidate_filename_for_task",
    "compile_environment",
    "compile_hip_source",
    "extension_cache_root",
    "extension_name",
    "gpu_arch",
    "hipcc_version",
    "is_hip_backend",
    "load_hip_candidate",
    "normalize_backend",
    "probe_toolchain",
    "source_language",
]
