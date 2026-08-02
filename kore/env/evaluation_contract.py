"""Exact replay identity for a KoreEnv evaluation request.

The replay key must describe what was *requested*, not infer capability from an
``Observation`` after the fact.  This module keeps that provenance construction
small, deterministic, and independent of the replay storage implementation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


EVALUATION_CONTRACT_VERSION = 3
"""Wire-format version; core content hashes independently bind evaluator semantics."""

_TRUTHY = {"1", "true", "yes", "on"}
_PREFLIGHT_IDENTITY_ENV = "KORE_PREFLIGHT_RUNTIME_IDENTITY"
PREFLIGHT_IDENTITY_ENV = _PREFLIGHT_IDENTITY_ENV
PREFLIGHT_IDENTITY_BUNDLE_KIND = "kore-preflight-identity-bundle"
"""One artifact holding one identity per validated GPU.

``gpu_target`` and ``selected_gpu`` are per-evaluation values, and each rank of a
distributed run benches on its own physical GPU, so a single flat identity
exported for a whole multi-GPU run would be wrong for every rank but one. A
bundle lets one exported value be correct everywhere: each consumer selects the
entry matching the GPU and architecture only it knows, and a rank whose GPU was
never validated selects nothing rather than somebody else's evidence.
"""
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CORE_CODE_PATHS: tuple[tuple[str, Path], ...] = tuple(
    (
        str(path.relative_to(_PROJECT_ROOT)),
        path,
    )
    for path in sorted(
        {
            path
            for directory in (
                _PROJECT_ROOT / "kore",
                # The roofline/physical model decides the speed-of-light integrity
                # verdict, so its content is part of what produced an observation.
                _PROJECT_ROOT / "kore" / "analysis",
                _PROJECT_ROOT / "kore" / "env",
                _PROJECT_ROOT / "kore" / "reward",
                _PROJECT_ROOT / "kore" / "tasks",
                _PROJECT_ROOT / "kore" / "tasks" / "breadth",
                _PROJECT_ROOT / "kore" / "verify",
                _PROJECT_ROOT / "kore" / "verifier",
                # The rocprofv3 CSV parser is the sole source of the counters that
                # become profile_efficiency.
                _PROJECT_ROOT / "kore" / "verifier" / "parsers",
            )
            for path in directory.glob("*.py")
        },
        key=lambda item: str(item),
    )
)
_REPLAY_DISABLED_ANNOUNCED: set[str] = set()
_BOOT_IDENTITY_CACHE: list[str] = []
_FINGERPRINT_LOCK = threading.RLock()
_FILE_FINGERPRINT_CACHE: dict[
    str, tuple[tuple[int, int, int, int, int], dict[str, Any]]
] = {}
_CODE_SET_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_COMPILER_VERSION_CACHE: dict[
    tuple[str, tuple[int, int, int, int, int]], dict[str, Any]
] = {}


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _json_scalar(value: Any) -> bool | int | float | str | None:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("evaluation context contains a non-finite number")
        return value
    raise TypeError(f"evaluation context scalar is not JSON-compatible: {type(value)!r}")


def _cfg(config: Any, name: str, default: Any = None) -> bool | int | float | str | None:
    return _json_scalar(getattr(config, name, default))


def _env_truthy(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUTHY


def _correctness_trials() -> int | str:
    raw = os.environ.get("KORE_CORRECTNESS_TRIALS", "5").strip()
    try:
        return int(raw)
    except ValueError:
        # Preserve an invalid operator value in the identity. The driver may fail,
        # but that failure cannot alias a valid trial count.
        return raw


def _finite_json_copy(value: Any, *, label: str) -> Any:
    """Canonical JSON round-trip that rejects NaN and both infinities recursively."""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON data") from exc


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _file_fingerprint(path: Path) -> dict[str, Any]:
    """Hash one file with a stat-validated, replacement-safe process cache."""
    cache_key = str(path.absolute())
    try:
        before = path.stat()
    except FileNotFoundError:
        return {"state": "missing", "sha256": None}
    except OSError as exc:
        return {"state": "unreadable", "sha256": None, "errno": exc.errno}

    before_sig = _stat_signature(before)
    with _FINGERPRINT_LOCK:
        cached = _FILE_FINGERPRINT_CACHE.get(cache_key)
        if cached is not None and cached[0] == before_sig:
            return dict(cached[1])

    digest = hashlib.sha256()
    try:
        with path.open("rb") as f:
            opened = os.fstat(f.fileno())
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        after = path.stat()
    except FileNotFoundError:
        return {"state": "unstable", "sha256": None}
    except OSError as exc:
        return {"state": "unreadable", "sha256": None, "errno": exc.errno}

    opened_sig = _stat_signature(opened)
    after_sig = _stat_signature(after)
    if before_sig != opened_sig or opened_sig != after_sig:
        return {"state": "unstable", "sha256": digest.hexdigest()}
    result = {
        "state": "present",
        "sha256": digest.hexdigest(),
        "size": opened.st_size,
    }
    with _FINGERPRINT_LOCK:
        _FILE_FINGERPRINT_CACHE[cache_key] = (after_sig, dict(result))
    return result


def _task_files(task: Any) -> tuple[dict[str, Any], bool]:
    raw_dir = getattr(task, "dir", None)
    task_dir = Path(raw_dir) if raw_dir is not None else None
    required: dict[str, Any] = {}
    for name in ("task.yaml", "reference.py", "driver.py"):
        required[name] = (
            _file_fingerprint(task_dir / name)
            if task_dir is not None
            else {"state": "missing", "sha256": None}
        )

    staged_python: dict[str, Any] = {}
    listing_stable = True
    if task_dir is not None:
        try:
            paths = sorted(task_dir.glob("*.py"), key=lambda p: p.name)
            staged_python = {p.name: _file_fingerprint(p) for p in paths}
        except OSError:
            listing_stable = False

    fingerprints = {"required": required, "staged_python": staged_python}
    states = [
        fp.get("state")
        for group in fingerprints.values()
        for fp in group.values()
    ]
    stable = listing_stable and all(state not in {"unstable", "unreadable"} for state in states)
    return fingerprints, stable


def _path_cache_token(path: Path) -> tuple[Any, ...]:
    try:
        return ("present", *_stat_signature(path.stat()))
    except FileNotFoundError:
        return ("missing",)
    except OSError as exc:
        return ("unreadable", exc.errno)


def _fingerprint_code_paths(
    paths: Optional[Iterable[tuple[str, Path]]] = None,
) -> dict[str, Any]:
    """Aggregate content identity for evaluator/core-driver semantics.

    The aggregate cache key includes inode, size, mtime, and ctime for every
    module. A replacement or metadata/content update therefore forces a rehash;
    unchanged modules use the validated per-file hash cache above.
    """
    selected = tuple(paths if paths is not None else _CORE_CODE_PATHS)
    cache_key = tuple(
        (label, str(path.absolute()), _path_cache_token(path))
        for label, path in selected
    )
    with _FINGERPRINT_LOCK:
        cached = _CODE_SET_CACHE.get(cache_key)
        if cached is not None:
            return {
                **cached,
                "files": {
                    label: dict(fingerprint)
                    for label, fingerprint in cached["files"].items()
                },
            }

    files = {label: _file_fingerprint(path) for label, path in selected}
    stable = bool(files) and all(
        fingerprint.get("state") == "present" for fingerprint in files.values()
    )
    digest_payload = {
        label: {
            "state": fingerprint.get("state"),
            "sha256": fingerprint.get("sha256"),
            "size": fingerprint.get("size"),
        }
        for label, fingerprint in files.items()
    }
    digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    result = {
        "state": "stable" if stable else "unstable",
        "sha256": digest,
        "files": files,
    }
    if stable:
        with _FINGERPRINT_LOCK:
            _CODE_SET_CACHE[cache_key] = result
    return {
        **result,
        "files": {
            label: dict(fingerprint)
            for label, fingerprint in result["files"].items()
        },
    }


def _compiler_identity(command: Optional[str]) -> dict[str, Any]:
    if not command:
        return {"state": "not-configured", "command": None}
    try:
        argv = shlex.split(command)
    except ValueError:
        return {"state": "unknown", "command": command, "reason": "invalid command"}
    if not argv:
        return {"state": "not-configured", "command": command}
    executable = shutil.which(argv[0])
    if executable is None:
        candidate = Path(argv[0])
        executable = str(candidate) if candidate.is_file() else None
    if executable is None:
        return {"state": "not-installed", "command": command}

    path = Path(executable)
    try:
        signature = _stat_signature(path.stat())
    except OSError as exc:
        return {
            "state": "unknown",
            "command": command,
            "reason": f"stat errno={exc.errno}",
        }
    key = (str(path.absolute()), signature)
    with _FINGERPRINT_LOCK:
        cached = _COMPILER_VERSION_CACHE.get(key)
        if cached is not None:
            return dict(cached)

    try:
        proc = subprocess.run(
            [str(path), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3,
            check=False,
        )
        lines = [" ".join(line.split()) for line in proc.stdout.splitlines() if line.strip()]
        if proc.returncode != 0 or not lines:
            raise RuntimeError(f"--version exited {proc.returncode}")
        result = {
            "state": "present",
            "command": command,
            "executable": str(path.absolute()),
            "version": "\n".join(lines[:4]),
            "binary": _file_fingerprint(path),
        }
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return {
            "state": "unknown",
            "command": command,
            "executable": str(path.absolute()),
            "reason": type(exc).__name__,
        }
    with _FINGERPRINT_LOCK:
        _COMPILER_VERSION_CACHE[key] = dict(result)
    return result


def _module_origin_identity(module_name: str) -> tuple[dict[str, Any], bool]:
    """Locate and hash a package entrypoint without importing the package."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError) as exc:
        return {"state": "unknown", "reason": type(exc).__name__}, False
    if spec is None:
        return {"state": "not-installed"}, True
    if spec.origin in (None, "built-in", "frozen"):
        locations = (
            sorted(str(path) for path in spec.submodule_search_locations)
            if spec.submodule_search_locations
            else []
        )
        return {
            "state": "namespace" if spec.origin is None else str(spec.origin),
            "locations": locations,
        }, True
    path = Path(spec.origin)
    fingerprint = _file_fingerprint(path)
    return {
        "state": fingerprint.get("state"),
        "origin": str(path.absolute()),
        "fingerprint": fingerprint,
    }, fingerprint.get("state") == "present"


def _package_versions() -> tuple[dict[str, Any], bool]:
    packages: dict[str, Any] = {}
    stable = True
    for logical_name, candidates in {
        "torch": ("torch",),
        "triton": ("triton", "pytorch-triton-rocm"),
        "aiter": ("aiter", "aiter-rocm"),
    }.items():
        installed = None
        for distribution in candidates:
            try:
                installed = {
                    "state": "present",
                    "distribution": distribution,
                    "version": importlib_metadata.version(distribution),
                }
                break
            except importlib_metadata.PackageNotFoundError:
                continue
            except Exception as exc:  # noqa: BLE001 - metadata backends vary
                installed = {
                    "state": "unknown",
                    "distribution": distribution,
                    "reason": type(exc).__name__,
                }
                stable = False
                break
        package = installed or {
            "state": "not-installed",
            "distributions": list(candidates),
        }
        module, module_stable = _module_origin_identity(logical_name)
        package["module"] = module
        packages[logical_name] = package
        stable = stable and module_stable
    return packages, stable


def _rocm_version_identity(rocm_path: Path) -> tuple[dict[str, Any], bool]:
    candidates = (
        rocm_path / ".info" / "version",
        rocm_path / ".info" / "version-dev",
        rocm_path / "share" / "rocm" / ".info" / "version",
    )
    for path in candidates:
        fingerprint = _file_fingerprint(path)
        state = fingerprint.get("state")
        if state == "missing":
            continue
        if state != "present":
            return {
                "state": "unknown",
                "root": str(rocm_path),
                "file": str(path),
                "fingerprint": fingerprint,
            }, False
        try:
            text = path.read_text(errors="replace")[:4096].strip()
        except OSError as exc:
            return {
                "state": "unknown",
                "root": str(rocm_path),
                "file": str(path),
                "reason": f"read errno={exc.errno}",
            }, False
        after = _file_fingerprint(path)
        if after != fingerprint:
            return {
                "state": "unstable",
                "root": str(rocm_path),
                "file": str(path),
            }, False
        return {
            "state": "present",
            "root": str(rocm_path),
            "file": str(path),
            "version": text,
            "fingerprint": fingerprint,
        }, True
    return {"state": "not-installed", "root": str(rocm_path)}, True


def _toolchain_fingerprint(config: Any) -> dict[str, Any]:
    packages, packages_stable = _package_versions()
    rocm_path = Path(
        str(getattr(config, "rocm_path", None) or os.environ.get("ROCM_PATH", "/opt/rocm"))
    )
    rocm, rocm_stable = _rocm_version_identity(rocm_path)
    cc = os.environ.get("CC") or sysconfig.get_config_var("CC")
    cxx = os.environ.get("CXX") or sysconfig.get_config_var("CXX")
    hipcc_path = rocm_path / "bin" / "hipcc"
    compilers = {
        "cc": _compiler_identity(str(cc) if cc else None),
        "cxx": _compiler_identity(str(cxx) if cxx else None),
        "hipcc": _compiler_identity(str(hipcc_path)),
    }
    compilers_stable = all(
        compiler.get("state") != "unknown" for compiler in compilers.values()
    )
    payload = {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "build": sys.version,
            "abi": sysconfig.get_config_var("SOABI"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "rocm": rocm,
        "compilers": compilers,
    }
    payload = _finite_json_copy(payload, label="toolchain identity")
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    stable = packages_stable and rocm_stable and compilers_stable
    return {
        "state": "stable" if stable else "unstable",
        "sha256": digest,
        **payload,
    }


def boot_identity() -> str:
    """This boot's kernel-generated id, or ``""`` when it cannot be read.

    A preflight identity binds to it because HIP device enumeration is stable
    for the life of a boot but may be renumbered across boots. The value cannot
    change while this process lives, so it is read once.
    """
    with _FINGERPRINT_LOCK:
        if _BOOT_IDENTITY_CACHE:
            return _BOOT_IDENTITY_CACHE[0]
    try:
        value = _BOOT_ID_PATH.read_text(errors="replace").strip()
    except OSError:
        value = ""
    with _FINGERPRINT_LOCK:
        if not _BOOT_IDENTITY_CACHE and value:
            _BOOT_IDENTITY_CACHE.append(value)
    return value


def core_code_digest() -> dict[str, Any]:
    """PUBLIC: aggregate content identity of the verdict-bearing core modules."""
    details = _fingerprint_code_paths()
    return {"state": details.get("state"), "sha256": details.get("sha256")}


def toolchain_digest(config: Any) -> dict[str, Any]:
    """PUBLIC: aggregate identity of compilers, ROCm, and GPU packages."""
    fingerprint = _toolchain_fingerprint(config)
    return {"state": fingerprint.get("state"), "sha256": fingerprint.get("sha256")}


def _select_bundled_identity(
    value: Mapping[str, Any],
    *,
    effective_gpu_target: str,
    expected_gpu: Any,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Pick this evaluation's identity out of a multi-GPU bundle.

    Returns ``(identity, None)`` on an unambiguous match. Two entries claiming
    the same GPU and architecture are not evidence about which one describes it,
    so an ambiguous bundle is refused rather than resolved by position.
    """
    entries = value.get("identities")
    if not isinstance(entries, list):
        return None, "bundle has no identity list"
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("hardware"), dict)
        and str(entry["hardware"].get("selected_gpu")) == str(expected_gpu)
        and entry["hardware"].get("gpu_target") == effective_gpu_target
    ]
    if not matches:
        return None, "bundle validated no identity for this GPU and architecture"
    if len(matches) > 1:
        return None, "bundle is ambiguous for this GPU and architecture"
    return matches[0], None


def _binding_drift(
    identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    toolchain_sha256: Optional[str],
    core_code_sha256: Optional[str],
) -> Optional[str]:
    """Re-verify, against live state, every fact the identity claims it was validated against.

    This is what stops the evidence from outliving what it attests to: a
    preflight that validated one toolchain, one evaluator revision, or one boot
    must not authorize replaying measurements taken under another.

    An identity that declares nothing makes the weaker, unbound claim it always
    made. An identity that declares a binding this build cannot check is
    *refused*, so no identity can look stronger by naming evidence nobody
    verifies.
    """
    declared = identity.get("bindings")
    if declared is None:
        return None
    if not isinstance(declared, list) or not all(
        isinstance(name, str) for name in declared
    ):
        return "bindings must be a list of names"
    live = {
        "boot_id": boot_identity() or None,
        "toolchain_sha256": toolchain_sha256,
        "core_code_sha256": core_code_sha256,
    }
    for name in declared:
        if name not in live:
            return f"unverifiable binding {name!r}"
        current = live[name]
        if not current:
            return f"binding {name!r} cannot be checked in this process"
        if runtime.get(name) != current:
            return f"{name} changed since the preflight"
    return None


def _announce_replay_disabled() -> None:
    """Say once, per process, that replay caching cannot engage.

    An evaluation is only replayable against an identity proving the hardware and
    runtime were validated and found stable *before* the run. Without one the
    cache fails closed and every candidate -- including one already measured --
    costs a fresh GPU evaluation. That is safe and slow, and the slowness should
    be attributable, so name the producer that would fix it.
    """
    with _FINGERPRINT_LOCK:
        if "no_identity" in _REPLAY_DISABLED_ANNOUNCED:
            return
        _REPLAY_DISABLED_ANNOUNCED.add("no_identity")
    import logging

    logging.getLogger("kore.env").info(
        "replay caching disabled: no preflight runtime identity supplied "
        "(argument or %s), so no evaluation is cacheable and replay_hits will "
        "stay at 0. Every candidate costs a fresh GPU evaluation. This is "
        "fail-closed by design -- an observation may only be replayed against "
        "proof that the hardware and runtime were validated as stable. Run "
        "'python -m kore.env.preflight_identity --export' and export the "
        "result to establish that proof.",
        _PREFLIGHT_IDENTITY_ENV,
    )


def _validated_preflight_identity(
    identity: Optional[Mapping[str, Any]],
    *,
    effective_gpu_target: str,
    gpu_selection: Mapping[str, Any],
    toolchain_sha256: Optional[str] = None,
    core_code_sha256: Optional[str] = None,
) -> tuple[dict[str, Any], bool]:
    source = "argument"
    value: Any = identity
    if value is None:
        raw = os.environ.get(_PREFLIGHT_IDENTITY_ENV)
        if raw:
            source = _PREFLIGHT_IDENTITY_ENV
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                return {"state": "invalid", "source": source, "reason": "invalid JSON"}, False
    if value is None:
        # Fail-closed is correct, but silence here is not: the caller sees
        # replay_hits stay at 0 forever and cannot tell "no duplicates" apart
        # from "caching never engaged".
        _announce_replay_disabled()
        return {"state": "unknown", "source": "none"}, False
    try:
        value = _finite_json_copy(value, label="preflight runtime identity")
    except ValueError:
        return {"state": "invalid", "source": source, "reason": "non-finite/non-JSON"}, False
    if not isinstance(value, dict):
        return {"state": "invalid", "source": source, "reason": "not an object"}, False
    expected_gpu = gpu_selection.get("selected_gpu")
    if value.get("kind") == PREFLIGHT_IDENTITY_BUNDLE_KIND:
        selected, bundle_reason = _select_bundled_identity(
            value,
            effective_gpu_target=effective_gpu_target,
            expected_gpu=expected_gpu,
        )
        if selected is None:
            return {
                "state": "invalid",
                "source": source,
                "reason": bundle_reason,
            }, False
        # Only the selected entry continues, so one rank's evidence never
        # becomes part of another rank's replay key.
        source = f"{source}[bundle]"
        value = selected
    if value.get("identity_version") != 1:
        return {"state": "invalid", "source": source, "reason": "identity_version"}, False
    if value.get("validated") is not True or value.get("stable") is not True:
        return {"state": "invalid", "source": source, "reason": "not validated/stable"}, False
    hardware = value.get("hardware")
    if not isinstance(hardware, dict):
        return {"state": "invalid", "source": source, "reason": "missing hardware"}, False
    runtime = value.get("runtime")
    if not isinstance(runtime, dict) or not runtime:
        return {"state": "invalid", "source": source, "reason": "missing runtime"}, False
    hardware_id = hardware.get("id")
    target = hardware.get("gpu_target")
    selected_gpu = hardware.get("selected_gpu")
    if not isinstance(hardware_id, str) or not hardware_id.strip():
        return {"state": "invalid", "source": source, "reason": "hardware id"}, False
    if target != effective_gpu_target:
        return {"state": "invalid", "source": source, "reason": "gpu target mismatch"}, False
    if str(selected_gpu) != str(expected_gpu):
        return {"state": "invalid", "source": source, "reason": "selected GPU mismatch"}, False
    drift = _binding_drift(
        value,
        runtime,
        toolchain_sha256=toolchain_sha256,
        core_code_sha256=core_code_sha256,
    )
    if drift is not None:
        return {"state": "invalid", "source": source, "reason": drift}, False
    digest = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "state": "validated",
        "source": source,
        "sha256": digest,
        # Records how strong this authorization actually was, and keeps two
        # otherwise-identical identities that verified different amounts of
        # evidence from sharing one replay key.
        "bindings_verified": sorted(value.get("bindings") or ()),
        "identity": value,
    }, True


def _clear_fingerprint_caches() -> None:
    """Test/support hook for environments that replace a toolchain in-process."""
    with _FINGERPRINT_LOCK:
        _FILE_FINGERPRINT_CACHE.clear()
        _CODE_SET_CACHE.clear()
        _COMPILER_VERSION_CACHE.clear()


def _shape_payload(shapes: Iterable[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for shape in shapes:
        dims = getattr(shape, "dims", {}) or {}
        payload.append(
            {
                "name": str(getattr(shape, "name", "")),
                # Shape dimensions are defined as str->int. A sorted pair list
                # makes mapping insertion order irrelevant while preserving exact
                # names, values, shape order, and duplicate requests.
                "dims": [
                    [str(key), int(value)]
                    for key, value in sorted(dims.items(), key=lambda item: str(item[0]))
                ],
            }
        )
    return payload


def build_evaluation_contract(
    *,
    task: Any,
    shapes: Iterable[Any],
    do_bench: bool,
    config: Any,
    snr_threshold: float,
    correctness_timeout: int,
    bench_timeout: int,
    gpu_selection: Optional[Mapping[str, Any]] = None,
    runtime_identity: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the complete, JSON-compatible identity of one evaluation request."""
    files, files_stable = _task_files(task)
    shape_augment_cfg = bool(getattr(config, "shape_augment", False))
    shape_augment_env = os.environ.get("KORE_SHAPE_AUGMENT", "0") == "1"
    verified_correctness = os.environ.get("KORE_VERIFIED_CORRECTNESS") == "1"
    compile_baseline = _env_truthy("KORE_COMPILE_BASELINE")
    cold_cache = os.environ.get("KORE_BENCH_COLD", "1") != "0"
    # Mirrors kore.tasks._genops._use_vendor_baseline (default ON). It switches
    # baseline_fn between the AITER/hipBLASLt production kernel and torch, which
    # changes baseline_ms and therefore every speedup.
    use_vendor_baseline = _env_truthy("KORE_USE_VENDOR_BASELINE", "1")

    raw = getattr(task, "raw", {}) or {}
    baseline_tier = raw.get("baseline_tier") if isinstance(raw, Mapping) else None
    architecture = str(
        getattr(task, "gpu_target", None) or getattr(config, "gpu_target", None) or ""
    )
    try:
        selection = _finite_json_copy(
            gpu_selection or {"state": "unknown"},
            label="GPU selection",
        )
    except ValueError:
        selection = {"state": "invalid", "reason": "non-finite/non-JSON"}
    selection_valid = (
        isinstance(selection, dict)
        and selection.get("state") == "selected"
        and isinstance(selection.get("selected_gpu"), str)
        and bool(selection.get("selected_gpu"))
        and selection.get("effective_gpu_target") == architecture
    )
    core_details = _fingerprint_code_paths()
    core_code = {
        "state": core_details.get("state"),
        "sha256": core_details.get("sha256"),
        "file_count": len(core_details.get("files", {})),
    }
    toolchain = _toolchain_fingerprint(config)
    preflight, preflight_valid = _validated_preflight_identity(
        runtime_identity,
        effective_gpu_target=architecture,
        gpu_selection=selection if isinstance(selection, dict) else {},
        # The live values the identity's declared bindings are re-checked
        # against, taken from the same fingerprints this contract records.
        toolchain_sha256=toolchain.get("sha256"),
        core_code_sha256=core_code.get("sha256"),
    )
    runtime_stable = (
        core_code.get("state") == "stable"
        and toolchain.get("state") == "stable"
        and selection_valid
        and preflight_valid
    )

    contract = {
        "contract_version": EVALUATION_CONTRACT_VERSION,
        "cacheable_context": files_stable and runtime_stable,
        "request": {
            "capabilities": {
                "correctness": True,
                "timing": bool(do_bench),
                # ``step(full_validation=...)`` maps directly to ``do_bench``.
                "full_validation": bool(do_bench),
            },
            "shapes": _shape_payload(shapes),
        },
        "task": {
            "task_id": str(getattr(task, "task_id", "")),
            "operation": _json_scalar(getattr(task, "operation", None)),
            "dtype": _json_scalar(getattr(task, "dtype", None)),
            "backend": _json_scalar(getattr(task, "backend", None)),
            "architecture": architecture,
            "files": files,
        },
        "baseline": {
            "declared_mode": _json_scalar(getattr(task, "comparison_baseline", None)),
            "tier": _json_scalar(baseline_tier),
            "compile_baseline": compile_baseline,
            "use_vendor_baseline": use_vendor_baseline,
            "fp8_encoding_override": (
                os.environ.get("KORE_FP8_ENCODING", "").strip().lower() or None
            ),
        },
        "verification": {
            "snr_threshold": _json_scalar(float(snr_threshold)),
            "atol": _cfg(config, "atol"),
            "rtol": _cfg(config, "rtol"),
            "correctness_trials": _correctness_trials(),
            "verified_correctness": verified_correctness,
            "determinism_check": bool(
                getattr(config, "verifier_determinism_check", False)
            ),
            "determinism_snr_tol_db": _cfg(config, "determinism_snr_tol_db"),
        },
        "benchmark": {
            "cold_cache": cold_cache,
            "warmup_iters": _cfg(config, "warmup_iters"),
            "bench_iters": _cfg(config, "bench_iters"),
            "min_variance_runs": _cfg(config, "min_variance_runs"),
            "max_variance_runs": _cfg(config, "max_variance_runs"),
            "cv_threshold_pct": _cfg(config, "cv_threshold_pct"),
            "batch_bench_enabled": not _env_truthy("KORE_NO_BENCH_BOTH"),
            "timing_lock_enabled": (
                os.environ.get("KORE_TIMING_LOCK", "1").strip().lower()
                not in {"0", "false", "no"}
            ),
            "profile_reward_weight": _cfg(config, "profile_reward_weight", 0.0),
        },
        "shape_augmentation": {
            "config_enabled": shape_augment_cfg,
            "environment_requested": shape_augment_env,
            "max_shapes": _cfg(config, "shape_augment_max"),
        },
        "rigor": {
            "level": (
                "strong"
                if (
                    verified_correctness
                    and compile_baseline
                    and cold_cache
                    and (shape_augment_cfg or shape_augment_env)
                )
                else "custom"
            ),
            "verified_correctness": verified_correctness,
            "compile_baseline": compile_baseline,
            "cold_cache": cold_cache,
            "shape_augmentation": shape_augment_cfg or shape_augment_env,
        },
        "execution": {
            "correctness_timeout_s": int(correctness_timeout),
            "bench_timeout_s": int(bench_timeout),
        },
        "runtime": {
            "effective_gpu_target": architecture,
            "gpu_selection": selection,
            "preflight_identity": preflight,
            "toolchain": toolchain,
            "core_code": core_code,
        },
    }
    return _finite_json_copy(contract, label="evaluation contract")


def contract_is_cacheable(contract: Mapping[str, Any]) -> bool:
    """False when task, evaluator, hardware, or toolchain identity is untrusted."""
    return contract.get("cacheable_context") is True


def observation_satisfies_contract(obs: Any, contract: Mapping[str, Any]) -> bool:
    """Reject a capability claim that the Observation itself cannot support.

    Compile failures and incorrect terminal verdicts are reusable under their
    exact request context. A successful verdict must prove every requested shape;
    a successful timed verdict must additionally contain candidate and baseline
    timing for every requested shape.
    """
    if bool(getattr(obs, "infra_error", False)):
        return False
    for name in (
        "snr_db",
        "wall_ms",
        "baseline_ms",
        "cv_pct",
        "profile_efficiency",
    ):
        value = getattr(obs, name, None)
        if value is not None and not _is_finite_number(value):
            return False
    for name in ("snr_by_shape", "wall_by_shape", "baseline_by_shape"):
        values = getattr(obs, name, {}) or {}
        if not isinstance(values, dict) or any(
            not _is_finite_number(value)
            for value in values.values()
        ):
            return False
    if not bool(getattr(obs, "validation_passed", False)):
        return True

    request = contract.get("request", {})
    shapes = request.get("shapes", [])
    expected = [shape.get("name") for shape in shapes if isinstance(shape, Mapping)]
    expected_names = set(expected)
    if len(expected) != len(expected_names):
        # Observation maps are keyed by shape name and cannot prove two distinct
        # requests that reuse a name. Run them, but do not cache a successful claim.
        return False
    snr_by_shape = getattr(obs, "snr_by_shape", {}) or {}
    if set(snr_by_shape) != expected_names:
        return False

    capabilities = request.get("capabilities", {})
    if capabilities.get("timing") is True:
        wall = getattr(obs, "wall_by_shape", {}) or {}
        baseline = getattr(obs, "baseline_by_shape", {}) or {}
        if set(wall) != expected_names or set(baseline) != expected_names:
            return False
        try:
            for name in expected_names:
                candidate_ms = float(wall[name])
                baseline_ms = float(baseline[name])
                if (not math.isfinite(candidate_ms) or not math.isfinite(baseline_ms)
                        or candidate_ms <= 0.0 or baseline_ms <= 0.0):
                    return False
        except (KeyError, OverflowError, TypeError, ValueError):
            return False
    return True


__all__ = [
    "EVALUATION_CONTRACT_VERSION",
    "PREFLIGHT_IDENTITY_BUNDLE_KIND",
    "PREFLIGHT_IDENTITY_ENV",
    "boot_identity",
    "build_evaluation_contract",
    "contract_is_cacheable",
    "core_code_digest",
    "observation_satisfies_contract",
    "toolchain_digest",
]
