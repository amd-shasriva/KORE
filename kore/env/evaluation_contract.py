"""Exact replay identity for a KoreEnv evaluation request.

The replay key must describe what was *requested*, not infer capability from an
``Observation`` after the fact.  This module keeps that provenance construction
small, deterministic, and independent of the replay storage implementation.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


EVALUATION_CONTRACT_VERSION = 1
"""Bump whenever observation-producing evaluation semantics change."""

_TRUTHY = {"1", "true", "yes", "on"}


def _json_scalar(value: Any) -> bool | int | float | str | None:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    return str(value)


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


def _file_fingerprint(path: Path) -> dict[str, Any]:
    """Hash one file and flag a concurrent replacement/modification as unstable."""
    try:
        before = path.stat()
    except FileNotFoundError:
        return {"state": "missing", "sha256": None}
    except OSError as exc:
        return {"state": "unreadable", "sha256": None, "errno": exc.errno}

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

    opened_sig = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    before_sig = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_sig = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_sig != opened_sig or opened_sig != after_sig:
        return {"state": "unstable", "sha256": digest.hexdigest()}
    return {
        "state": "present",
        "sha256": digest.hexdigest(),
        "size": opened.st_size,
    }


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
) -> dict[str, Any]:
    """Build the complete, JSON-compatible identity of one evaluation request."""
    files, files_stable = _task_files(task)
    shape_augment_cfg = bool(getattr(config, "shape_augment", False))
    shape_augment_env = os.environ.get("KORE_SHAPE_AUGMENT", "0") == "1"
    verified_correctness = os.environ.get("KORE_VERIFIED_CORRECTNESS") == "1"
    compile_baseline = _env_truthy("KORE_COMPILE_BASELINE")
    cold_cache = os.environ.get("KORE_BENCH_COLD", "1") != "0"

    raw = getattr(task, "raw", {}) or {}
    baseline_tier = raw.get("baseline_tier") if isinstance(raw, Mapping) else None
    architecture = getattr(task, "gpu_target", None) or getattr(config, "gpu_target", None)

    return {
        "contract_version": EVALUATION_CONTRACT_VERSION,
        "cacheable_context": files_stable,
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
            "architecture": _json_scalar(architecture),
            "files": files,
        },
        "baseline": {
            "declared_mode": _json_scalar(getattr(task, "comparison_baseline", None)),
            "tier": _json_scalar(baseline_tier),
            "compile_baseline": compile_baseline,
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
        "runtime": {
            "correctness_timeout_s": int(correctness_timeout),
            "bench_timeout_s": int(bench_timeout),
        },
    }


def contract_is_cacheable(contract: Mapping[str, Any]) -> bool:
    """False when task files changed or became unreadable during fingerprinting."""
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
        except (KeyError, TypeError, ValueError):
            return False
    return True


__all__ = [
    "EVALUATION_CONTRACT_VERSION",
    "build_evaluation_contract",
    "contract_is_cacheable",
    "observation_satisfies_contract",
]
