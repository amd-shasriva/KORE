"""Strict candidate environment construction."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Optional

from kore.sandbox.errors import PolicyViolation


# Only host runtime plumbing needed to locate Python/compiler/ROCm libraries is
# inherited. Candidate controls and private paths are assigned below.
INHERITED_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "ROCM_PATH",
        "HIP_PATH",
    }
)

CANDIDATE_ENV_ALLOWLIST = INHERITED_ENV_ALLOWLIST | frozenset(
    {
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONSAFEPATH",
        "HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "MIOPEN_USER_DB_PATH",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_TARGET",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
        "MAX_JOBS",
    }
)

OVERRIDABLE_ENV_KEYS = frozenset(
    {
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
        "MAX_JOBS",
    }
)

_FORBIDDEN_NAME = re.compile(
    r"(?:"
    r"API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|"
    r"^AWS_|^AZURE_|^GOOGLE_|^GITHUB_|^OPENAI_|^ANTHROPIC_|"
    r"(?:^|_)PROXY$|^NO_PROXY$|"
    r"^SLURM|^SSH|^LD_PRELOAD$|^PYTHONUSERBASE$"
    r")",
    re.IGNORECASE,
)


def _private_directory(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise PolicyViolation(f"candidate private directory cannot be a symlink: {name}")
    os.chmod(path, 0o700)
    return path


def assert_safe_candidate_environment(environment: Mapping[str, str]) -> None:
    """Reject inherited or injected variables outside the explicit contract."""

    for key, value in environment.items():
        if key not in CANDIDATE_ENV_ALLOWLIST:
            raise PolicyViolation(f"candidate environment key is not allowlisted: {key}")
        if _FORBIDDEN_NAME.search(key):
            raise PolicyViolation(f"forbidden candidate environment key: {key}")
        if not isinstance(value, str) or "\x00" in value:
            raise PolicyViolation(f"invalid candidate environment value for {key}")


def build_candidate_environment(
    *,
    base_environment: Mapping[str, str],
    private_root: Path,
    project_root: Path,
    gpu_target: str,
    gpu: Optional[str] = None,
    rocm_path: Optional[str] = None,
    overrides: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Build a fresh, allowlisted environment for trusted candidate processes.

    This does not turn the subprocess into an untrusted-code sandbox. It merely
    prevents ambient credentials and scheduler/session controls from crossing
    the repository boundary.
    """

    private_root = Path(private_root)
    if not private_root.is_absolute():
        raise PolicyViolation("candidate private root must be absolute")
    project_root = Path(project_root)
    if not project_root.is_absolute():
        raise PolicyViolation("candidate project root must be absolute")
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if private_root.is_symlink():
        raise PolicyViolation("candidate private root cannot be a symlink")
    os.chmod(private_root, 0o700)

    home = _private_directory(private_root, "home")
    tmp = _private_directory(private_root, "tmp")
    cache = _private_directory(private_root, "cache")
    triton_cache = _private_directory(cache, "triton")
    inductor_cache = _private_directory(cache, "torchinductor")
    extensions_cache = _private_directory(cache, "torch-extensions")
    miopen_cache = _private_directory(cache, "miopen")

    env = {
        key: str(base_environment[key])
        for key in INHERITED_ENV_ALLOWLIST
        if key in base_environment
    }
    env.update(
        {
            "PYTHONPATH": str(project_root),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONSAFEPATH": "1",
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "TMP": str(tmp),
            "TEMP": str(tmp),
            "XDG_CACHE_HOME": str(cache),
            "TRITON_CACHE_DIR": str(triton_cache),
            "TORCHINDUCTOR_CACHE_DIR": str(inductor_cache),
            "TORCH_EXTENSIONS_DIR": str(extensions_cache),
            "MIOPEN_USER_DB_PATH": str(miopen_cache),
            "GPU_TARGET": str(gpu_target),
            "OPENBLAS_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "TORCHINDUCTOR_COMPILE_THREADS": "4",
            "MAX_JOBS": "4",
        }
    )
    if rocm_path:
        env["ROCM_PATH"] = str(rocm_path)
    if gpu is not None:
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        env["HIP_VISIBLE_DEVICES"] = str(base_environment.get("HIP_VISIBLE_DEVICES", "0"))

    if overrides:
        for key, value in overrides.items():
            if key not in OVERRIDABLE_ENV_KEYS:
                raise PolicyViolation(f"candidate override is not permitted: {key}")
            env[key] = str(value)

    # ROCR visibility is deliberately never inherited. Combining an outer ROCR
    # mask with a per-evaluation HIP mask can hide every device.
    env.pop("ROCR_VISIBLE_DEVICES", None)
    assert_safe_candidate_environment(env)
    return env
