"""KoreEnv: the verified evaluation environment.

Wraps the KernelForge verifier contract into a task-bound ``step(source)`` call
that returns a reward :class:`Observation`. Hardening (see audits):

* **No verdict forgery.** The candidate ``kernel.py`` is imported by the driver
  and could print fake ``SNR:``/``median_ms:`` lines. We parse the *last* match
  (the driver prints its verdict after calling the candidate) AND the anti-hack
  scanner rejects any candidate that prints a verdict literal.
* **Execution boundary.** Each eval runs in a throwaway workdir; the copied task
  sources (incl. reference.py oracle) are made read-only so a kernel can't
  corrupt them. The default backend is the in-process ``trusted-code-only``
  subprocess path: its own session with a process limit; on timeout the whole
  process group is killed (no leaked grandchildren / GPU holders). These controls
  do not isolate hostile same-UID code. An OPTIONAL, config/env-GATED sandbox
  broker path (default OFF) routes execution through an approved external broker
  with a signed verdict; when no broker is configured the verifier behaves
  exactly as the default subprocess path.
* **Infra vs kernel.** Timeouts, OOM-kills, segfaults, and missing-dependency
  imports are classified as ``infra_error`` - never cached, never fed to the
  policy as a kernel-correctness signal.
* **Trustworthy timing.** Timing is cold-cache (the driver L2-flushes between
  timed iters) and each (shape, impl) is benched several times; the coefficient of
  variation is recorded and high-variance speedups are damped. Candidate correctness
  is re-verified AFTER the timed loop, so a kernel that is correct while checked but
  garbage while timed (a stateful invocation-count hack) is caught and rejected.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Optional

from kore.config import CONFIG
from kore.env.evaluation_contract import (
    build_evaluation_contract,
    contract_is_cacheable,
    observation_satisfies_contract,
)
from kore.env.replay import ReplayCache
from kore.obs import get_logger
from kore.policy.budget import (
    BudgetLedgerV1,
    EvaluationWork,
    charge_evaluation_work,
    check_evaluation_budget,
)
from kore.env import hip_toolchain as _hip
from kore.reward.reward import Observation, scan_for_hacks
from kore.reward.reward import _worst_speedup
from kore.reward.stats import cv_pct as _cv_pct
from kore.reward.stats import median as _median
from kore.reward.stats import paired_timing_stats as _paired_timing_stats
from kore.reward.stats import publication_admission_error as _publication_admission_error
from kore.tasks._genops import (
    DRIVER_CAPABILITY_PROTOCOL,
    DRIVER_PROTOCOL_ID,
    PUBLICATION_GUARANTEES,
)
from kore.tasks.base import Shape, Task
from kore.verify.production import (
    DEFAULT_MAX_ELEMENTS,
    RUNNER_SHIM_NAME,
    RUNNER_SHIM_SOURCE,
    MetamorphicPlan,
    OracleReport,
    ProngStatus,
    build_oracle_report,
    expected_output_elements,
    generic_adversarial_families,
    metamorphic_plan_for_task,
    parse_metamorphic_report,
    sanitize_detail,
    select_metamorphic_shape,
    shape_spec,
    task_output_op_class,
)

_LOG = get_logger("env")


# --------------------------------------------------------------------------- #
# OPTIONAL sandbox/isolation backend.
#
# The fail-closed broker/isolation execution boundary lives in ``kore.sandbox``.
# That package IS present in this deployment, and ``kore.config`` already imports
# ``kore.sandbox.config`` at module top level, so ``_SANDBOX_AVAILABLE`` is in
# practice always True here. Consequences worth stating plainly:
#
# * every :class:`KoreEnv` constructs a ``TrustedSubprocessController`` (the
#   in-process, broker-free controller) plus its policy;
# * the EXECUTION gate (``self._sandbox_enabled``) is still default OFF, so that
#   controller is never actually invoked - ``_exec`` runs the plain subprocess
#   path and none of the sandbox policy checks apply.
#
# The guarded import is kept because it is the only thing that keeps this module
# importable in a stripped deployment (absence => "sandbox unavailable" and the
# gate can never turn on), not because the package is expected to be missing.
# --------------------------------------------------------------------------- #
_SANDBOX_AVAILABLE = False
try:  # pragma: no cover - exercised only where kore.sandbox is deployed
    from kore.sandbox.config import SandboxConfig  # noqa: F401
    from kore.sandbox.controller import (  # noqa: F401
        IsolationController,
        create_isolation_controller,
    )
    from kore.sandbox.environment import build_candidate_environment  # noqa: F401
    from kore.sandbox.errors import PolicyViolation, SandboxError  # noqa: F401
    from kore.sandbox.models import (  # noqa: F401
        ExecutionKind,
        ExecutionStatus,
        SandboxRequest,
        SandboxResponse,
    )
    from kore.sandbox.signing import VerdictSignatureVerifier  # noqa: F401

    _SANDBOX_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure => sandbox unavailable
    SandboxConfig = None  # type: ignore[assignment]
    IsolationController = None  # type: ignore[assignment]
    create_isolation_controller = None  # type: ignore[assignment]
    build_candidate_environment = None  # type: ignore[assignment]
    PolicyViolation = None  # type: ignore[assignment]
    SandboxError = None  # type: ignore[assignment]
    ExecutionKind = None  # type: ignore[assignment]
    ExecutionStatus = None  # type: ignore[assignment]
    SandboxRequest = None  # type: ignore[assignment]
    SandboxResponse = None  # type: ignore[assignment]
    VerdictSignatureVerifier = None  # type: ignore[assignment]


def _sandbox_requested(config) -> bool:
    """True only when the sandbox execution path is explicitly opted into.

    Default OFF. The gate is honored ONLY if ``kore.sandbox`` is importable; when
    the broker backend is absent this always returns False and the default
    frontier+verifier subprocess path runs unchanged.
    """
    if not _SANDBOX_AVAILABLE:
        return False
    env_flag = os.environ.get("KORE_SANDBOX_ENABLED", "").strip().lower()
    if env_flag in ("1", "true", "yes", "on"):
        return True
    if env_flag in ("0", "false", "no", "off"):
        return False
    return bool(getattr(config, "sandbox_enabled", False))


_METAMORPHIC_ENV = "KORE_METAMORPHIC"
_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")


def _metamorphic_gate() -> tuple[bool, bool]:
    """``(enabled, described_by_the_evaluation_contract)`` for the 3rd prong.

    The metamorphic prong costs one extra candidate-only subprocess, so it rides
    the SAME operator switch as the driver's enumerated adversarial battery,
    ``KORE_VERIFIED_CORRECTNESS=1`` - one honest "verified correctness" gate for
    both deterministic value/structure prongs, and one that
    :func:`kore.env.evaluation_contract.build_evaluation_contract` already
    records, so a cached verdict can never have been produced under a different
    prong set.

    ``KORE_METAMORPHIC=0/1`` overrides it for A/B measurement and for disabling a
    single prong in an emergency. That override is NOT part of the contract, so
    whenever it disagrees with the gate the contract describes, this returns
    ``described=False`` and the caller must refuse to read or write the replay
    cache rather than let two different oracles share one cache key.
    """
    default_on = os.environ.get("KORE_VERIFIED_CORRECTNESS") == "1"
    raw = os.environ.get(_METAMORPHIC_ENV, "").strip().lower()
    if raw in _TRUTHY:
        return True, default_on
    if raw in _FALSY:
        return False, not default_on
    return default_on, True


def _ev(level: str, name: str, **fields) -> None:
    """Emit a structured event at an explicit level (JSONL always).

    ``KoreLogger.event`` hard-codes INFO; per-shape verifier detail must ride at
    DEBUG so it never spams INFO while a run is going, so we route through the
    logger's emit with the level we want but keep ``kind="event"`` for
    machine-readable JSONL. This is additive-only - pure observability.
    """
    _LOG._emit(level, name, fields, kind="event")


def _sha12(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8", "ignore")).hexdigest()[:12]


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


def _open_private_lockfile(path: Path) -> Optional[int]:
    """Open/create an owned, private, non-symlink lockfile; ``None`` if unsafe.

    The timing lockfile lives at a predictable name in a world-writable shared
    tmpdir on a multi-user cluster, so it gets the same discipline as
    :class:`kore.ops.runtime.SecureFileLock`: ``O_NOFOLLOW`` refuses a symlink
    planted at the path, ``O_CLOEXEC`` keeps the descriptor out of the candidate
    subprocess, ``fchmod`` makes it private, and the post-open ``fstat`` on the
    *descriptor we hold* proves it is a regular file owned by us (so a foreign uid
    cannot redirect our writes or hold our timing phase hostage).

    Deliberately inlined rather than importing ``kore.ops``: ``SecureFileLock`` is
    fail-CLOSED and raises :class:`~kore.ops.runtime.SecurityError`, while this lock
    is only a measurement-quality optimization and must degrade to unlocked timing
    instead of destroying an otherwise valid evaluation.
    """
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | _NOFOLLOW | _CLOEXEC, 0o600)
    except OSError:
        return None
    try:
        os.fchmod(fd, 0o600)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise OSError(f"timing lockfile is not an owned regular file: {path}")
    except OSError:
        os.close(fd)
        return None
    return fd


_SNR = re.compile(r"SNR:\s*([-\d.eE]+)")
_ALLCLOSE = re.compile(r"allclose:\s*(True|False)", re.IGNORECASE)
_MEDIAN = re.compile(r"median_ms:\s*([-\d.eE]+)")
# Which baseline the driver ACTUALLY timed. Every aiter_ref wrapper degrades to
# torch on an import/JIT failure and emits this sentinel; without parsing it a
# node where AITER fails to build still records baseline_kind="vendor" for
# numbers that came from eager torch -- i.e. a fabricated vendor-beating claim.
_BASELINE_IMPL = re.compile(r"^KORE_BASELINE_IMPL:\s*(\w+)\s*$", re.M)
# The driver names the enumerated regime it failed on, which is what lets the
# oracle report separate the adversarial prong from the random one (both share
# one ``allclose:`` verdict). A candidate cannot forge itself a PASS with this
# marker - printing it can only produce a rejection - so it is safe evidence.
_ADV_FAIL = re.compile(r"^ADVERSARIAL_(?:FAIL|ERROR)\[([^\]]{0,80})\]", re.MULTILINE)
_TIMING_PAIR = re.compile(r"^KORE_TIMING_PAIR:\s*(\{[^\n]+\})\s*$",
                          re.MULTILINE)
_DRIVER_CAPS = re.compile(r"^KORE_DRIVER_CAPABILITIES:\s*(\{[^\n]+\})\s*$",
                          re.MULTILINE)
_BATCH_CAPABILITIES = {
    "protocol": DRIVER_CAPABILITY_PROTOCOL,
    "protocol_id": DRIVER_PROTOCOL_ID,
    "performance_eligible": True,
    **PUBLICATION_GUARANTEES,
}
# Candidate import/compile failure (the kernel's fault).
_COMPILE_ERR = re.compile(
    r"(SyntaxError|CompilationError|triton\..*Error|IndentationError|"
    r"NameError|out of resource|OutOfResources|AssertionError)",
    re.IGNORECASE,
)
# Infrastructure failure (NOT the kernel's fault) - never cache, never train on.
_HIP_TOOLCHAIN_ERR = _hip.TOOLCHAIN_ABSENCE_PATTERN
_INFRA_ERR = re.compile(
    r"(hipError|HIP error|out of memory|hipErrorOutOfMemory|CUDA error|"
    r"no CUDA-capable|device-side assert|ECC|Xid|"
    r"ModuleNotFoundError:.*(torch|aiter|triton|rocm)|"
    r"ImportError:.*(torch|aiter|triton|rocm|libamdhip|librocm))",
    re.IGNORECASE,
)


def _last(pattern: re.Pattern, text: str):
    ms = list(pattern.finditer(text))
    return ms[-1] if ms else None


def _parse_driver_capabilities(text: str) -> dict:
    """Parse the strict, versioned driver handshake from subprocess output."""
    matches = list(_DRIVER_CAPS.finditer(text or ""))
    if len(matches) != 1:
        return {}
    try:
        caps = json.loads(matches[0].group(1))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(caps, dict) or not isinstance(caps.get("protocol"), int):
        return {}
    return caps


def _supports_batch_bench(caps: dict) -> bool:
    return all(caps.get(k) == v for k, v in _BATCH_CAPABILITIES.items())


def _task_source_files(task: Task) -> list[Path]:
    """Task assets staged into an evaluation workdir, in a stable order.

    Was ``dir.glob("*.py")``, which silently dropped a HIP task's ``.hip``
    baseline before the workdir was built.  The suffix list is closed
    (:data:`kore.env.hip_toolchain.STAGED_SUFFIXES`) rather than "everything in
    the directory": these files are copied next to a running candidate, so the
    set has to stay something a reviewer can enumerate.  The candidate artifact
    itself is never staged from the task directory -- it is written by the
    environment -- so a stray ``kernel.py``/``kernel.hip`` left in a task
    directory cannot shadow the candidate under evaluation.
    """
    task_dir = getattr(task, "dir", None)
    if task_dir is None:
        return []
    reserved = set(_hip.CANDIDATE_FILENAMES.values())
    out: list[Path] = []
    for path in sorted(Path(task_dir).iterdir()):
        if not path.is_file() or path.name in reserved:
            continue
        if path.suffix in _hip.STAGED_SUFFIXES:
            out.append(path)
    return out


def _parse_timing_pairs(block: str, expected_count: int) -> tuple[list[dict], Optional[str]]:
    """Parse and validate exact-count, balanced raw timing pairs."""
    matches = list(_TIMING_PAIR.finditer(block or ""))
    if len(matches) != expected_count:
        return [], (
            f"paired sample count {len(matches)} != requested {expected_count}")
    pairs: list[dict] = []
    for expected_index, match in enumerate(matches):
        try:
            pair = json.loads(match.group(1))
        except (TypeError, ValueError, json.JSONDecodeError):
            return [], f"pair {expected_index} is not valid JSON"
        if not isinstance(pair, dict) or pair.get("pair") != expected_index:
            return [], f"pair index mismatch at {expected_index}"
        order = pair.get("order")
        if order not in ("AB", "BA"):
            return [], f"pair {expected_index} has invalid order {order!r}"
        try:
            cand = float(pair["candidate_ms"])
            base = float(pair["baseline_ms"])
            ratio = float(pair["ratio"])
            log_su = float(pair["log_speedup"])
        except (KeyError, TypeError, ValueError):
            return [], f"pair {expected_index} has missing/invalid numeric fields"
        if not all(math.isfinite(v) for v in (cand, base, ratio, log_su)):
            return [], f"pair {expected_index} contains non-finite values"
        if not (cand > 0.0 and base > 0.0 and ratio > 0.0):
            return [], f"pair {expected_index} contains non-positive values"
        expected_ratio = base / cand
        if not math.isclose(ratio, expected_ratio, rel_tol=1e-9, abs_tol=1e-12):
            return [], f"pair {expected_index} ratio does not match raw times"
        if not math.isclose(log_su, math.log(expected_ratio),
                            rel_tol=1e-9, abs_tol=1e-12):
            return [], f"pair {expected_index} log speedup does not match ratio"
        pairs.append({
            "pair": expected_index, "order": order,
            "candidate_ms": cand, "baseline_ms": base,
            "ratio": expected_ratio, "log_speedup": math.log(expected_ratio),
        })
    orders = [p["order"] for p in pairs]
    if any(a == b for a, b in zip(orders, orders[1:])):
        return [], "pair order is not alternating AB/BA"
    if abs(orders.count("AB") - orders.count("BA")) > 1:
        return [], "pair order is not balanced AB/BA"
    return pairs, None


def _cold_cache_timing(env: Mapping[str, Any], caps: Mapping[str, Any]) -> bool:
    """Whether the timed subprocess really flushed L2 between timed iterations.

    Two independent facts have to hold, and both are known here: the driver we timed
    with advertised the KORE timing protocol (its ``_time_fn``/``_time_fn_value``
    helpers are what own the L2 flush), and ``KORE_BENCH_COLD`` was not disabled in
    the exact environment mapping we handed that subprocess - which is not always
    ``os.environ``, because the sandbox path rebuilds the child environment from an
    allowlist. The default-ON reading mirrors the driver's own.
    """
    try:
        protocol = int(caps.get("protocol") or 0)
    except (TypeError, ValueError):
        return False
    if protocol < DRIVER_CAPABILITY_PROTOCOL:
        return False
    return str(env.get("KORE_BENCH_COLD", "1")) != "0"


def _noise_demoted_timing(obs: Observation) -> bool:
    """True for a correct observation whose timing was demoted by measurement noise.

    ``timing_pair_count`` is set only by the paired publication path, so a
    ``screening`` grade carrying one can only have come from the noise demotion in
    :meth:`KoreEnv._run` (operator-forced screening reports protocol 0 and no pair
    count). That verdict is a property of the run's noise rather than of the
    ``(task, source, contract)`` identity, so it must never be replayed: a later
    quiet measurement of the same kernel deserves its speed credit.
    """
    return (getattr(obs, "timing_grade", None) == "screening"
            and getattr(obs, "timing_pair_count", None) is not None)


def _timing_completeness_error(expected_names, candidate, baseline) -> Optional[str]:
    """Return why per-shape timing is incomplete, else ``None``."""
    expected_list = list(expected_names)
    expected = set(expected_list)
    if len(expected) != len(expected_list):
        return "requested shape names are not unique"
    for label, values in (("candidate", candidate), ("baseline", baseline)):
        keys = set(values)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            return f"{label} timing keys mismatch (missing={missing}, extra={extra})"
        for name, value in values.items():
            try:
                valid = math.isfinite(float(value)) and float(value) > 0.0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                return f"{label} timing for {name!r} is not finite and positive"
    return None


def _preexec():  # pragma: no cover - runs in child only
    # NB: session is created via Popen(start_new_session=True); do NOT setsid
    # again here (would EPERM).
    #
    # Do NOT *lower* RLIMIT_NPROC. It is PER-UID (it counts EVERY process/thread the
    # user owns, not just this child), so a small per-subprocess soft cap throttles
    # the entire user. Under concurrent datagen (32 workers spawn thousands of
    # torch/OpenBLAS threads) an old 512 cap made OpenBLAS `blas_thread_init` fail
    # and `import numpy` die inside the driver, so EVERY eval falsely reported
    # compiled=False -> 100% silent datagen failure on a busy node. Raise the soft
    # limit to the hard cap; runaway containment is the timeout + killpg in _exec
    # and the system hard limit, not a per-child nproc cap.
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    except (ValueError, OSError):
        pass
    # Deliberately NOT setting RLIMIT_AS - ROCm/HIP reserve huge virtual address
    # space and an AS cap breaks legitimate GPU kernels.


class KoreEnv:
    """Task-bound verified environment. One per task; call ``step`` per candidate."""

    def __init__(self, task: Task, config=CONFIG, use_replay: bool = True,
                 correctness_timeout: int = 300, bench_timeout: int = 300,
                 gpu: Optional[str] = None,
                 isolation_controller: Optional["IsolationController"] = None,
                 sandbox_config: Optional["SandboxConfig"] = None,
                 verdict_verifier: Optional["VerdictSignatureVerifier"] = None,
                 runtime_identity: Optional[Mapping[str, Any]] = None,
                 budget_ledger: Optional[BudgetLedgerV1] = None):
        self.task = task
        self.cfg = config
        self.correctness_timeout = correctness_timeout
        self.bench_timeout = bench_timeout
        self.use_replay = use_replay
        # Physical GPU for the compile/bench SUBPROCESS (HIP_VISIBLE_DEVICES).
        # Under distributed GRPO every rank must bench on its OWN GPU; otherwise
        # all ranks default to GPU 0, contend/OOM there, one stalls, and the
        # cross-rank all_gather deadlocks. None => inherit/legacy default "0".
        self._gpu = gpu

        # ---- sandbox gate (default OFF) --------------------------------- #
        # The fail-closed broker/isolation path is opt-in. When it is not
        # explicitly enabled (the default, and the only possibility when
        # kore.sandbox is not deployed) the environment runs the standard
        # subprocess + paired-timing path with NONE of the sandbox policy
        # checks, exactly as the frontier+verifier build does.
        self._sandbox_enabled = _sandbox_requested(config) or (
            isolation_controller is not None
            or sandbox_config is not None
            or verdict_verifier is not None
        )
        if self._sandbox_enabled and not _SANDBOX_AVAILABLE:
            raise RuntimeError(
                "sandbox execution requested but kore.sandbox is not available")
        self.isolation_policy = None
        self.isolation_controller = None
        self.sandbox_config = None
        self._last_execution_status = None
        self._active_source: Optional[str] = None
        self._active_task: Optional[Task] = None
        self._task_descriptor_cache: dict[str, dict] = {}
        # Which correctness prongs actually ran for the most recent candidate,
        # and the honest statistical false-accept bound behind the verdict.
        self._last_oracle_report: Optional[OracleReport] = None
        # The in-process "trusted-code-only" isolation controller is ALWAYS
        # constructed when kore.sandbox is available: it needs no external broker,
        # provides ambient-secret scrubbing + a trusted backend label, and is the
        # safe default. Only broker-backed STRONG isolation (execution routing) is
        # gated by self._sandbox_enabled below.
        if _SANDBOX_AVAILABLE:
            self.sandbox_config = (
                sandbox_config
                or getattr(config, "sandbox", None)
                or SandboxConfig()
            )
            self.isolation_policy = self.sandbox_config.policy()
            self.isolation_controller = (
                isolation_controller
                or create_isolation_controller(
                    self.sandbox_config,
                    verifier=verdict_verifier,
                )
            )
            if self.isolation_controller.policy != self.isolation_policy:
                raise PolicyViolation(
                    "isolation controller policy does not match KoreEnv policy")

        # A preflight-produced, validated identity for the selected physical GPU.
        # Without it replay fails closed (evaluation still runs normally).
        self._runtime_identity = (
            runtime_identity
            if runtime_identity is not None
            else getattr(config, "runtime_identity", None)
        )
        # Set from the bench subprocess's KORE_BASELINE_IMPL sentinel; reset per
        # evaluation so a stale value can never be attributed to a later one.
        self._last_baseline_impl: Optional[str] = None
        # Compute ledger for the five evaluation counters, or None for an
        # unbudgeted environment. Passed EXPLICITLY rather than derived from
        # ``config``: the ledger lives on the GRPOConfig, while every KoreEnv on
        # the rollout path is handed ``kore.config.CONFIG`` (or a copy of it), so a
        # config-derived ledger would find nothing exactly where it matters and
        # the counters would stay zero while looking wired.
        self._budget_ledger = budget_ledger
        # Wall time the rocprofv3 passes of the most recent evaluation spent, so
        # profiler seconds are attributed separately from verifier seconds instead
        # of being inferred from a count.
        self._last_profiler_seconds = 0.0
        self._cache_obj = ReplayCache(self.cfg.runs_dir / f"replay_{task.task_id}.jsonl") \
            if use_replay else None

    @property
    def _snr_threshold(self) -> float:
        return self._snr_threshold_for(self.task)

    def _snr_threshold_for(self, task: Task) -> float:
        t = getattr(task, "snr_threshold", None)
        return float(t) if t else self.cfg.snr_threshold_for(task.dtype)

    @property
    def last_profiler_seconds(self) -> float:
        """PUBLIC: wall seconds the last evaluation spent inside the profiler.

        The rocprofv3 candidate+reference passes run INSIDE one ``evaluate`` call,
        so a caller timing that call cannot separate profiler time from verifier
        time on its own. This is the measured carve-out
        (:mod:`kore.eval.champion` reads it to attribute ``profiler_gpu_seconds``
        without double-counting the interval), and it is ``0.0`` whenever the
        profiler did not run: bonus off, replay hit, or an evaluation that ended
        before the timed tier.
        """
        return self._last_profiler_seconds

    @property
    def last_execution_status(self):
        """Typed status from the most recent sandbox-controlled subprocess.

        ``None`` on the default (non-sandbox) path, which does not produce a
        typed execution status.
        """
        return self._last_execution_status

    @property
    def last_oracle_report(self) -> Optional[OracleReport]:
        """PUBLIC: what the correctness oracle actually checked, last evaluation.

        Carries one :class:`~kore.verify.production.ProngStatus` per prong -
        random, adversarial, metamorphic, determinism - saying whether it passed,
        failed, was gated off, or does not apply to this task, together with the
        exact evidence source for each, and the ``(1-p)**m`` false-accept bound
        over the random prong's element comparisons.

        ``None`` when no oracle verdict was reached for the last call: nothing
        evaluated yet, a replay-cache hit, or an evaluation that ended before the
        prongs could report (hack rejection, compile failure, or an infra failure
        during the per-shape correctness runs). It is deliberately NOT stored on
        :class:`~kore.reward.reward.Observation`: that object is the replay-cached
        payload, and a report describes one physical run.
        """
        return self._last_oracle_report

    # ------------------------------------------------------------------ #
    def step(self, source: str, full_validation: bool = True,
             multi_shape: bool = True) -> Observation:
        return self.evaluate(self.task, source, shapes=self._shapes(multi_shape),
                             do_bench=full_validation)

    def _shapes(self, multi_shape: bool) -> list[Shape]:
        shapes = self.task.shapes or [Shape("default", {})]
        if multi_shape:
            # data-scale: optionally expand to a diverse shape set (shape-robust RL).
            if getattr(self.cfg, "shape_augment", False):
                from kore.tasks.augment import augment_shapes
                aug = augment_shapes(
                    shapes,
                    task=self.task,
                    max_shapes=int(getattr(self.cfg, "shape_augment_max", 6)),
                )
                if aug:
                    return aug
            return shapes
        primary = self.task.shape("primary") or self.task.shape("minimal") or shapes[0]
        return [primary]

    # ------------------------------------------------------------------ #
    def evaluate(self, task: Task, source: str, shapes: Optional[list[Shape]] = None,
                 do_bench: bool = True) -> Observation:
        # Resolve the request exactly once. The concrete ordered shape list is a
        # first-class part of replay identity; ``None`` can never alias a later
        # task/augmentation change.
        shapes = list(shapes or task.shapes or [Shape("default", {})])
        source_sha = _sha12(source)
        n_shapes = len(shapes)
        # A report describes one physical run; a replayed or short-circuited
        # verdict must not appear to carry fresh oracle evidence, and no later
        # evaluation may inherit this one's profiler attribution.
        self._last_oracle_report = None
        self._last_profiler_seconds = 0.0
        _ev("INFO", "eval_start", task=task.task_id, n_shapes=n_shapes,
            source_sha=source_sha, do_bench=do_bench)

        # Sandbox-only source-budget gate (skipped entirely on the default path).
        if self._sandbox_enabled:
            source_bytes = len(source.encode("utf-8"))
            if source_bytes > self.isolation_policy.budget.max_source_bytes:
                self._last_execution_status = ExecutionStatus.POLICY_VIOLATION
                return Observation(
                    compiled=False,
                    dtype=task.dtype,
                    validation_passed=False,
                    infra_error=True,
                    error_text=(
                        f"sandbox policy: candidate source is {source_bytes} bytes; "
                        f"limit is {self.isolation_policy.budget.max_source_bytes}"
                    ),
                )

        hack = scan_for_hacks(source, _hip.source_language(task.backend))
        if hack:
            _ev("WARN", "eval_hack", task=task.task_id, reason=hack, source_sha=source_sha)
            return Observation(compiled=False, dtype=task.dtype, flagged_hack=True,
                               hack_reason=hack, error_text=f"reward-hack: {hack}")

        contract = build_evaluation_contract(
            task=task,
            shapes=shapes,
            do_bench=do_bench,
            config=self.cfg,
            snr_threshold=self._snr_threshold_for(task),
            correctness_timeout=self.correctness_timeout,
            bench_timeout=self.bench_timeout,
            gpu_selection=self._gpu_selection(task),
            runtime_identity=self._runtime_identity,
        )
        # The metamorphic override is outside the contract, so a run that uses it
        # is not describable by the replay key: fail closed to no caching rather
        # than let a 3-prong and a 4-prong verdict share one entry.
        _, gate_in_contract = _metamorphic_gate()
        if not gate_in_contract:
            _ev("WARN", "eval_replay_disabled", task=task.task_id,
                reason=f"{_METAMORPHIC_ENV} overrides the contract-recorded "
                       "verified-correctness gate")
        replay_ready = contract_is_cacheable(contract) and gate_in_contract
        if self.use_replay and self._cache_obj is not None and replay_ready:
            cached = self._cache_obj.get(task.task_id, source, context=contract)
            if cached is not None:
                _LOG.debug("cache hit", task=task.task_id, source_sha=source_sha,
                           compiled=cached.compiled, correct=cached.validation_passed)
                # This is the ONLY place a replay hit is observable, so it is the
                # only place ``replay_hits`` can be charged. A hit ran nothing, so
                # it is charged as a hit and nothing else.
                charge_evaluation_work(
                    self._budget_ledger, EvaluationWork(replay_hits=1))
                self._log_eval_done(task, cached, cached=True)
                return cached

        # Pre-flight BEFORE anything launches: charging afterwards still binds a
        # limit, but only once the GPU time is already gone. A ``correctness_calls``
        # limit of 0 must refuse to start this evaluation, not report it.
        check_evaluation_budget(
            self._budget_ledger, self._planned_work(do_bench=do_bench))

        workdir = Path(tempfile.mkdtemp(prefix=f"kore_{task.task_id}_"))
        previous_source, previous_task = self._active_source, self._active_task
        self._active_source, self._active_task = source, task
        started = time.perf_counter()
        try:
            obs = self._run(task, source, shapes, workdir, do_bench)
        finally:
            elapsed = time.perf_counter() - started
            self._active_source, self._active_task = previous_source, previous_task
            shutil.rmtree(workdir, ignore_errors=True)

        # Charge what physically happened, splitting the ONE measured interval into
        # the profiler passes ``_run`` timed and the verifier work that is the rest.
        # Splitting (rather than timing the profiler separately) is what keeps the
        # two second-counters from double-counting the same wall time.
        profiler_seconds = min(self._last_profiler_seconds, elapsed)
        charge_evaluation_work(self._budget_ledger, EvaluationWork.from_observation(
            obs,
            verifier_seconds=max(0.0, elapsed - profiler_seconds),
            profiler_seconds=profiler_seconds,
        ))

        # Only cache DETERMINISTIC terminal verdicts - never transient infra errors,
        # and never a timing verdict that only this run's measurement noise produced.
        cacheable = (obs.compiled or obs.error_text) and not obs.infra_error
        cacheable = cacheable and not _noise_demoted_timing(obs)
        cacheable = cacheable and observation_satisfies_contract(obs, contract)
        if (self.use_replay and self._cache_obj is not None and replay_ready
                and cacheable):
            # A task/config/env mutation during a long GPU evaluation must not
            # label the resulting observation with stale pre-run provenance.
            final_contract = build_evaluation_contract(
                task=task,
                shapes=shapes,
                do_bench=do_bench,
                config=self.cfg,
                snr_threshold=self._snr_threshold_for(task),
                correctness_timeout=self.correctness_timeout,
                bench_timeout=self.bench_timeout,
                gpu_selection=self._gpu_selection(task),
                runtime_identity=self._runtime_identity,
            )
            if final_contract == contract and contract_is_cacheable(final_contract):
                self._cache_obj.put(task.task_id, source, obs, context=contract)
        self._log_eval_done(task, obs, cached=False)
        return obs

    def _planned_work(self, *, do_bench: bool) -> EvaluationWork:
        """The work an evaluation is ABOUT to do, for the pre-flight check.

        The CALL counts are exactly known before launching. The SECONDS are not,
        but they are known to be positive: a launched evaluation always spends
        verifier wall time, and it spends profiler wall time whenever the counter
        bonus is on for a timed run. So the claim here is the smallest amount the
        ledger can register - enough to refuse a launch under an EXHAUSTED (in
        particular zero) seconds budget, and small enough that at any nonzero
        remaining budget the float addition absorbs it and the pre-flight cannot
        invent a duration it has no way to predict.
        """
        floor = math.ulp(0.0)
        profiling = do_bench and getattr(self.cfg, "profile_reward_weight", 0.0) > 0.0
        return EvaluationWork(
            correctness_calls=1,
            fresh_timed_calls=1 if do_bench else 0,
            verifier_gpu_seconds=floor,
            profiler_gpu_seconds=floor if profiling else 0.0,
        )

    def _log_eval_done(self, task: Task, obs: Observation, cached: bool) -> None:
        """Final per-candidate verdict at INFO (structured), covering every path."""
        _ev("INFO", "eval_done", task=task.task_id, compiled=obs.compiled,
            correct=obs.validation_passed, snr_min=obs.snr_db,
            worst_speedup=_worst_speedup(obs), cv_pct=obs.cv_pct,
            infra_error=obs.infra_error, cached=cached)

    # ------------------------------------------------------------------ #
    def _gpu_selection(self, task: Optional[Task] = None) -> dict[str, Any]:
        """Exact visibility mapping used by the evaluator subprocess.

        This is pure environment bookkeeping: it never imports torch/HIP or
        initializes a GPU in the parent process.
        """
        active_task = task or self.task
        target = str(
            getattr(active_task, "gpu_target", None) or self.cfg.gpu_target
        )
        names = (
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        )
        parent = {name: os.environ.get(name) for name in names}
        child = dict(parent)
        if self._gpu is not None:
            selected = str(self._gpu)
            child["ROCR_VISIBLE_DEVICES"] = None
            child["HIP_VISIBLE_DEVICES"] = selected
            child["CUDA_VISIBLE_DEVICES"] = selected
            mode = "explicit-physical"
        else:
            child["HIP_VISIBLE_DEVICES"] = (
                parent["HIP_VISIBLE_DEVICES"]
                if parent["HIP_VISIBLE_DEVICES"] is not None
                else "0"
            )
            selected = str(child["HIP_VISIBLE_DEVICES"]).split(",")[0].strip()
            mode = "inherited"
        return {
            "state": "selected",
            "mode": mode,
            "selected_gpu": selected,
            "parent_visibility": parent,
            "child_visibility": child,
            "effective_gpu_target": target,
        }

    def _env(self, task: Optional[Task] = None,
             private_root: Optional[Path] = None) -> dict:
        """Environment for a candidate-bearing subprocess.

        Default path: the standard allowlist-augmented copy of ``os.environ``
        (frontier behavior). When the sandbox gate is on, delegates to the
        sandbox's fresh-allowlisted ``build_candidate_environment``.
        """
        if self._sandbox_enabled:
            root = private_root or (
                Path(tempfile.gettempdir()) / f"kore_env_{os.getpid()}_{id(self):x}"
            )
            active_task = task or self._active_task or self.task
            selection = self._gpu_selection(active_task)
            selected_gpu = (
                str(selection["child_visibility"]["HIP_VISIBLE_DEVICES"])
                if self._gpu is not None
                else None
            )
            return build_candidate_environment(
                base_environment=os.environ,
                private_root=Path(root),
                project_root=Path(__file__).resolve().parents[2],
                gpu_target=str(selection["effective_gpu_target"]),
                gpu=selected_gpu,
                rocm_path=getattr(self.cfg, "rocm_path", None),
            )

        env = os.environ.copy()
        # Trusted-code-only ambient-secret scrub (default isolation posture): the
        # candidate/driver subprocess runs untrusted-authored kernel code, so never
        # leak host credentials or process-injection vectors into it. Drop known
        # sensitive vars + credential/token patterns, while KEEPING vars the ROCm
        # driver actually needs (PATH, LD_LIBRARY_PATH, HOME/GPU/cache are set below).
        _AMBIENT_SECRET_DENY = {
            "ANTHROPIC_API_KEY", "AMD_LLM_API_KEY", "OPENAI_API_KEY", "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "HTTPS_PROXY", "HTTP_PROXY",
            "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy", "NO_PROXY",
            "SSH_AUTH_SOCK", "SSH_AGENT_PID", "LD_PRELOAD", "PYTHONUSERBASE",
            "PYTHONSTARTUP", "SLURM_JOB_ID",
        }
        for _sk in list(env):
            if (_sk in _AMBIENT_SECRET_DENY
                    or _sk.startswith(("AWS_", "GOOGLE_", "AZURE_", "GCP_", "SLURM_"))
                    or _sk.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))):
                env.pop(_sk, None)
        # Repo root (the parent of the kore/ package). Prepended to PYTHONPATH so the
        # compile/bench driver subprocess can ``import kore.*`` (e.g. _genops.driver_main).
        project_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        active_task = task or self.task
        selection = self._gpu_selection(active_task)
        if self._gpu is not None:
            # ABSOLUTE physical GPU id for the compile/bench subprocess. Set BOTH
            # HIP_ and CUDA_VISIBLE_DEVICES to it (and drop any inherited list) so the
            # subprocess sees exactly this one physical GPU as its device 0 - no
            # double-remap from a restricted parent visible-device list.
            env.pop("ROCR_VISIBLE_DEVICES", None)
            # str(): subprocess env values MUST be strings - an int gpu id (e.g.
            # KoreEnv(gpu=5)) would make subprocess.Popen raise inside os.fsencode.
            env["HIP_VISIBLE_DEVICES"] = selection["child_visibility"]["HIP_VISIBLE_DEVICES"]
            env["CUDA_VISIBLE_DEVICES"] = selection["child_visibility"]["CUDA_VISIBLE_DEVICES"]
        else:
            env["HIP_VISIBLE_DEVICES"] = selection["child_visibility"]["HIP_VISIBLE_DEVICES"]
        # Prefer the TASK's declared arch over the global default so the driver
        # subprocess compiles/benches + selects the fp8 encoding for the arch the
        # task actually targets (a gfx950 task must not be built as gfx942/FNUZ).
        env["GPU_TARGET"] = selection["effective_gpu_target"]
        env["HOME"] = str(Path(env.get("TMPDIR", "/tmp")))
        # Shared, persistent Triton/inductor compile caches (audit R2 perf M3). Pinned
        # to a STABLE dir -- NOT the per-eval HOME/TMPDIR above -- so the FIRST worker to
        # compile a given kernel warms the cache for ALL 64 workers and every future
        # eval + restart, turning the cold-compile bulk of the ~35s/eval into a one-time
        # cost. Triton/inductor handle concurrent cache access (atomic writes + locks).
        # Overridable via KORE_COMPILE_CACHE_DIR. setdefault so an explicit parent env
        # wins. Compiled code is deterministic, so caching never changes measured timing.
        _cache_root = env.get("KORE_COMPILE_CACHE_DIR") or "/tmp/kore_compile_cache"
        env.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, "triton"))
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_cache_root, "inductor"))
        # Cap CPU BLAS/OMP threads in the driver. By default OpenBLAS spawns one
        # thread PER CORE (96 here); across 32 concurrent datagen workers that is a
        # thread explosion that both wastes CPU and pushes the per-UID thread count
        # sky-high. The driver's numpy use is tiny (output comparison) and the real
        # work is on the GPU, so a few threads is plenty. Defense-in-depth alongside
        # the RLIMIT_NPROC fix in _preexec.
        for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS"):
            env.setdefault(_v, "4")
        # Cap the per-driver torch-inductor compile-worker pool (audit R2 perf): each
        # eval is its own driver subprocess, and inductor's default pool is
        # ~min(32, cores/2) workers PER driver -- with many concurrent reverify/datagen
        # workers that is a thread explosion that oversubscribes the box (400+ procs on
        # 384 cores) and SLOWS every eval via CPU contention. A small fixed pool keeps
        # total processes ~= worker_count x few, so cores feed compiles instead of
        # thrashing on context switches. Overridable via the env.
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
        env.setdefault("MAX_JOBS", "4")   # ninja/C++ ext build parallelism per driver
        # HIP C++ tasks compile through torch.utils.cpp_extension, which needs
        # three things this env did not previously provide: ninja on PATH (the
        # console script lives beside sys.executable and is NOT on PATH when the
        # interpreter is invoked by absolute path), PYTORCH_ROCM_ARCH pinned to
        # the task's own target (measured: 15.4s vs 114.6s per compile), and a
        # STABLE extension cache -- HOME is redirected to the per-eval TMPDIR
        # above, and torch's default cache lives under HOME, so every HIP eval
        # would otherwise pay a full cold compile.
        if _hip.is_hip_backend(getattr(active_task, "backend", None)):
            env = _hip.compile_environment(
                env, selection["effective_gpu_target"] or active_task.gpu_target)
        return env

    def _exec(self, cmd, workdir, env, timeout):
        """Execute ``cmd``. Default path: own session, killpg on timeout.

        When the sandbox gate is on, execution is routed through the configured
        fail-closed isolation controller instead. Returns
        ``(returncode, combined_output, timed_out)``.
        """
        if self._sandbox_enabled:
            return self._exec_sandbox(cmd, workdir, env, timeout)
        p = subprocess.Popen(cmd, cwd=str(workdir), env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, start_new_session=True,
                             preexec_fn=_preexec)
        try:
            out, err = p.communicate(timeout=timeout)
            return p.returncode, (out or "") + "\n" + (err or ""), False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                out, err = p.communicate(timeout=10)
            except Exception:
                out, err = "", ""
            return -9, (out or "") + "\n" + (err or ""), True

    def _exec_sandbox(self, cmd, workdir, env, timeout):
        """Execute through the configured isolation controller (gated path)."""
        task = self._active_task or self.task
        source = self._active_source
        if source is None:
            try:
                source = (Path(workdir)
                          / _hip.candidate_filename_for_task(task)).read_text()
            except OSError:
                source = ""
        try:
            request = SandboxRequest.create(
                task_id=task.task_id,
                task_descriptor=self._task_descriptor(task),
                source=source,
                policy=self.isolation_policy,
                toolchain_descriptor={
                    "python_implementation": platform.python_implementation(),
                    "python_version": platform.python_version(),
                    "python_executable": Path(sys.executable).name,
                    "rocm_path": str(getattr(self.cfg, "rocm_path", "")),
                    "packages": {
                        name: _distribution_version(name)
                        for name in ("kore", "torch", "triton")
                    },
                },
                runtime_descriptor={
                    "system": platform.system(),
                    "kernel_release": platform.release(),
                    "machine": platform.machine(),
                    "gpu_target": (
                        getattr(task, "gpu_target", None)
                        or getattr(self.cfg, "gpu_target", "gfx950")
                    ),
                    "gpu": str(self._gpu) if self._gpu is not None else "inherited-or-0",
                    "backend": self.isolation_controller.backend_label,
                },
                execution_kind=ExecutionKind.LEGACY_PYTHON,
                argv=tuple(str(part) for part in cmd),
                working_directory=str(workdir),
                environment=env,
                timeout_seconds=min(
                    float(timeout),
                    self.isolation_policy.budget.wall_time_seconds,
                ),
            )
        except (SandboxError, TypeError, ValueError) as exc:
            self._last_execution_status = ExecutionStatus.POLICY_VIOLATION
            return 126, f"sandbox policy: {exc}", False

        try:
            response = self.isolation_controller.execute(request)
        except Exception as exc:  # noqa: BLE001 - isolation failures must fail closed
            self._last_execution_status = ExecutionStatus.INFRA_ERROR
            return 125, f"sandbox controller failure: {exc}", False
        if not isinstance(response, SandboxResponse):
            self._last_execution_status = ExecutionStatus.INVALID_VERDICT
            return 125, "sandbox controller returned an invalid response", False
        self._last_execution_status = response.status
        out = response.stdout or ""
        err = response.stderr or ""
        if response.verdict.message:
            err = f"{err}\n[sandbox:{response.status.value}] {response.verdict.message}"
        returncode = response.verdict.exit_code
        if returncode is None:
            if response.status is ExecutionStatus.OK:
                returncode = 0
            elif response.status is ExecutionStatus.TIMEOUT:
                returncode = -9
            elif response.status is ExecutionStatus.POLICY_VIOLATION:
                returncode = 126
            else:
                returncode = 125
        return returncode, out + "\n" + err, response.status is ExecutionStatus.TIMEOUT

    def _task_descriptor(self, task: Task) -> dict:
        cache_key = str(getattr(task, "task_id", "unknown"))
        cached = self._task_descriptor_cache.get(cache_key)
        if cached is not None:
            return cached
        files: dict[str, str] = {}
        # Every file that will be STAGED, not just the Python ones: a HIP task's
        # ``.hip`` baseline is part of its identity, and a descriptor blind to it
        # would let an edited oracle reuse another oracle's cached verdict.
        for path in _task_source_files(task):
            try:
                files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                files[path.name] = "unreadable"
        descriptor = {
            "task_id": cache_key,
            "dtype": str(getattr(task, "dtype", "")),
            "backend": _hip.normalize_backend(getattr(task, "backend", "")),
            "gpu_target": str(getattr(task, "gpu_target", "")),
            "shapes": [
                {
                    "name": str(getattr(shape, "name", "")),
                    "dims": dict(getattr(shape, "dims", {})),
                }
                for shape in (getattr(task, "shapes", None) or [])
            ],
            "task_files": files,
        }
        self._task_descriptor_cache[cache_key] = descriptor
        return descriptor

    def _classify(self, out: str, returncode: int, timed_out: bool):
        """-> ('ok'|'compile'|'infra', message)."""
        # Sandbox-typed status classification (gated path only).
        if self._sandbox_enabled:
            status = self._last_execution_status
            if timed_out:
                return "infra", "timeout"
            if status in {
                ExecutionStatus.INFRA_ERROR,
                ExecutionStatus.POLICY_VIOLATION,
                ExecutionStatus.GPU_FAULT,
                ExecutionStatus.GPU_QUARANTINED,
                ExecutionStatus.BROKER_UNAVAILABLE,
                ExecutionStatus.UNSUPPORTED_ISOLATION,
                ExecutionStatus.INVALID_VERDICT,
            }:
                return "infra", f"{status.value}: {_tail(out)}"
            if status is ExecutionStatus.CANDIDATE_ERROR:
                return "compile", _tail(out)
        if timed_out:
            return "infra", "timeout"
        # A missing ninja/hipcc raises a Python RuntimeError with a traceback, so
        # the compile branch below would charge a broken node to the candidate and
        # report a node-wide outage as a near-100% model error rate. Toolchain
        # absence is checked FIRST and is always infra.
        if _HIP_TOOLCHAIN_ERR.search(out):
            return "infra", f"hip toolchain unavailable: {_tail(out)}"
        if _INFRA_ERR.search(out):
            return "infra", _tail(out)
        if returncode < 0 or returncode == 137:  # signal / OOM-kill
            return "infra", f"process killed (rc={returncode}); {_tail(out)}"
        if returncode != 0:
            if _COMPILE_ERR.search(out) or "Traceback" in out:
                return "compile", _tail(out)
            return "compile", _tail(out)
        return "ok", ""

    def _run(self, task: Task, source: str, shapes: list[Shape], workdir: Path,
             do_bench: bool) -> Observation:
        # Clear per-evaluation runtime evidence so a previous candidate's baseline
        # identity - or profiler attribution - can never be attributed to this one.
        self._last_baseline_impl = None
        self._last_profiler_seconds = 0.0
        # stage isolated sources; make the oracle/driver READ-ONLY so a kernel
        # cannot corrupt reference.py for future evals.
        task_sources = _task_source_files(task)
        # Sandbox-only task-source budget gate (skipped on the default path).
        if self._sandbox_enabled:
            task_bytes = sum(p.stat().st_size for p in task_sources)
            if task_bytes > self.isolation_policy.budget.max_task_bytes:
                self._last_execution_status = ExecutionStatus.POLICY_VIOLATION
                return Observation(
                    compiled=False,
                    dtype=task.dtype,
                    validation_passed=False,
                    infra_error=True,
                    error_text=(
                        f"sandbox policy: task sources are {task_bytes} bytes; "
                        f"limit is {self.isolation_policy.budget.max_task_bytes}"
                    ),
                )
        for p in task_sources:
            dst = workdir / p.name
            shutil.copy(p, dst)
            os.chmod(dst, 0o444)
        candidate_name = _hip.candidate_filename_for_task(task)
        (workdir / candidate_name).write_text(source)
        os.chmod(workdir / candidate_name, 0o444)
        driver = workdir / "driver.py"
        env = (self._env(task=task, private_root=workdir / ".sandbox")
               if self._sandbox_enabled else self._env(task))

        requested_names = [sh.name for sh in shapes]
        snr_by_shape: dict[str, float] = {}
        compiled = True
        validation_passed = True
        last_err: Optional[str] = None
        adv_failures: list[str] = []

        for i, sh in enumerate(shapes):
            t_sh = time.perf_counter()
            # First shape pays the Triton JIT compile cost; .timer records it.
            with _LOG.timer("verify_exec", task=task.task_id, shape=sh.name, first=(i == 0)):
                rc, out, timed = self._exec([sys.executable, str(driver), *sh.as_args()],
                                            workdir, env, self.correctness_timeout)
            took_s = round(time.perf_counter() - t_sh, 3)
            kind, msg = self._classify(out, rc, timed)
            _snr_m = _last(_SNR, out)
            _ac_m = _last(_ALLCLOSE, out)
            _ev("DEBUG", "verify_shape", task=task.task_id, shape=sh.name, kind=kind,
                snr_db=(float(_snr_m.group(1)) if _snr_m else None),
                allclose=(_ac_m.group(1).lower() == "true" if _ac_m else None),
                rc=rc, took_s=took_s)
            if kind == "infra":
                return Observation(compiled=True, dtype=task.dtype, validation_passed=False,
                                   infra_error=True, error_text=f"infra: {msg}")
            if kind == "compile":
                return Observation(compiled=False, dtype=task.dtype, validation_passed=False,
                                   error_text=msg)
            # rc==0: parse the driver-owned verdict (LAST match beats candidate forgery)
            m = _last(_SNR, out)
            ac = _last(_ALLCLOSE, out)
            if m:
                snr_by_shape[sh.name] = float(m.group(1))
            if ac and ac.group(1).lower() == "false":
                validation_passed = False
            if not m and not ac:
                validation_passed = False
                last_err = _tail(out)
            # Which enumerated regime (if any) the adversarial prong rejected, so
            # the oracle report can attribute the driver's single verdict.
            adv_failures.extend(
                f"{sh.name}:{sanitize_detail(hit.group(1), 40)}"
                for hit in _ADV_FAIL.finditer(out))

        # Verifier stricter correctness gate: validation passed AND the benchmarked
        # shape set EXACTLY covers the requested shapes (no dupes, no missing/extra)
        # AND every per-shape SNR clears the per-task threshold.
        thr = self._snr_threshold_for(task)
        requested_set = set(requested_names)
        correct = (
            validation_passed
            and len(requested_set) == len(requested_names)
            and set(snr_by_shape) == requested_set
            and all(v >= thr for v in snr_by_shape.values())
        )

        # The driver publishes ONE ``allclose:`` per shape covering both the
        # reseeded random trials and (when gated on) the enumerated adversarial
        # regimes, so attribute the two prongs before anything downstream mutates
        # ``correct``.
        random_ok = correct
        determinism_ran = False
        determinism_ok = False

        # Anti-hack determinism re-check: re-run the primary shape once and require
        # a stable verdict, so a kernel cannot be rewarded for passing the SNR gate
        # by luck (partly-random output). One extra exec, only when already correct.
        if correct and getattr(self.cfg, "verifier_determinism_check", False):
            sh0 = shapes[0]
            rc2, out2, timed2 = self._exec([sys.executable, str(driver), *sh0.as_args()],
                                           workdir, env, self.correctness_timeout)
            kind2, _ = self._classify(out2, rc2, timed2)
            snr2 = None
            # A transient INFRA error (timeout/OOM/HIP flake) on the re-run is NOT
            # evidence the kernel is non-deterministic - treat it as inconclusive and
            # keep the (already-verified) correct verdict, so a one-off flake can
            # never cache a correct kernel as incorrect (preserves infra-vs-kernel).
            if kind2 == "infra":
                _ev("DEBUG", "verify_determinism", task=task.task_id, shape=sh0.name,
                    inconclusive=True, reason="infra error on re-run")
                stable, reason = True, ""
            else:
                m2, ac2 = _last(_SNR, out2), _last(_ALLCLOSE, out2)
                snr2 = float(m2.group(1)) if m2 else None
                ac2_false = bool(ac2 and ac2.group(1).lower() == "false")
                ok2 = (kind2 == "ok" and not ac2_false
                       and ((snr2 is not None and snr2 >= thr)
                            or bool(ac2 and ac2.group(1).lower() == "true")))
                tol = float(getattr(self.cfg, "determinism_snr_tol_db", 10.0))
                stable, reason = _determinism_stable(snr_by_shape.get(sh0.name), snr2, ok2, tol)
            _ev("DEBUG", "verify_determinism", task=task.task_id, shape=sh0.name,
                snr1=snr_by_shape.get(sh0.name), snr2=snr2, stable=stable)
            determinism_ran, determinism_ok = True, bool(stable)
            if not stable:
                _ev("WARN", "eval_nondeterministic", task=task.task_id,
                    source_sha=_sha12(source), reason=reason)
                correct = False
                last_err = reason

        # ---- prong 3: metamorphic (candidate-only structural identities) ---- #
        # Deterministic algebraic relations the TRUE operator satisfies for any
        # input (permutation equivariance / locality / reshape for elementwise;
        # order-invariance + row independence for reductions). They need no
        # second oracle evaluation, and they reject a class the random prong
        # structurally cannot see: a kernel that is right on every sampled value
        # but wrong about which elements may influence which outputs.
        plan = metamorphic_plan_for_task(task)
        gate_on, _ = _metamorphic_gate()
        meta_payload: Optional[dict] = None
        if not gate_on:
            metamorphic = ProngStatus(
                "metamorphic", "metamorphic", "off",
                f"gated off: set KORE_VERIFIED_CORRECTNESS=1 (or "
                f"{_METAMORPHIC_ENV}=1) to run it",
                plan.reason)
        elif not plan.applicable:
            metamorphic = ProngStatus(
                "metamorphic", "metamorphic", "not-applicable",
                "kore.verify.production.metamorphic_plan_for_task", plan.reason)
        elif not correct:
            metamorphic = ProngStatus(
                "metamorphic", "metamorphic", "off",
                "not run: the candidate was already rejected by an earlier prong",
                plan.reason)
        else:
            metamorphic, meta_infra, meta_payload = self._metamorphic_prong(
                task, plan, shapes, workdir, env)
            if meta_infra is not None:
                # Fail-CLOSED: a prong that was supposed to run and could not is
                # never a pass. There is no trustworthy correctness evidence for
                # this candidate, so the evaluation is inconclusive (infra) -
                # never cached, never fed to the policy as a kernel signal.
                _ev("WARN", "eval_metamorphic_inconclusive", task=task.task_id,
                    source_sha=_sha12(source), reason=_tail(meta_infra, 300))
                obs = Observation(
                    compiled=True, dtype=task.dtype, validation_passed=False,
                    infra_error=True,
                    error_text=f"infra: metamorphic prong inconclusive: {meta_infra}")
                self._publish_oracle_report(
                    task, shapes, verified=False, random_ok=random_ok,
                    adv_failures=adv_failures, metamorphic=metamorphic,
                    determinism_ran=determinism_ran, determinism_ok=determinism_ok,
                    do_bench=False, meta_payload=meta_payload,
                    detail="evaluation abandoned: metamorphic prong inconclusive")
                return obs
            if metamorphic.state == "fail":
                _ev("WARN", "eval_metamorphic_violation", task=task.task_id,
                    source_sha=_sha12(source), detail=_tail(metamorphic.detail, 300))
                correct = False
                last_err = f"metamorphic violation: {metamorphic.detail}"

        self._publish_oracle_report(
            task, shapes, verified=correct, random_ok=random_ok,
            adv_failures=adv_failures, metamorphic=metamorphic,
            determinism_ran=determinism_ran, determinism_ok=determinism_ok,
            do_bench=do_bench, meta_payload=meta_payload)

        obs = Observation(
            compiled=compiled, dtype=task.dtype,
            snr_by_shape=snr_by_shape,
            snr_db=min(snr_by_shape.values()) if snr_by_shape else None,
            validation_passed=correct, error_text=last_err if not correct else None,
            requested_shapes=list(requested_names),
            timing_requested=bool(correct and do_bench),
        )
        if not (correct and do_bench):
            return obs

        caps = self._driver_capabilities(driver, workdir, env)
        publication_capable = _supports_batch_bench(caps)
        force_screening = os.environ.get(
            "KORE_NO_BENCH_BOTH", "").strip().lower() in ("1", "true", "yes")
        obs.timing_protocol = caps.get("protocol_id") or "unknown"
        obs.timing_protocol_version = caps.get("protocol")
        obs.timing_guarantees = {
            k: bool(caps.get(k, False)) for k in PUBLICATION_GUARANTEES
        }
        obs.performance_eligible = publication_capable and not force_screening
        if not publication_capable and not force_screening:
            obs.timing_grade = "ineligible"
            reason = caps.get("ineligible_reason") or (
                "driver lacks the complete paired-v2 publication guarantees")
            obs.error_text = f"performance-ineligible: {reason}"
            _ev("WARN", "eval_performance_ineligible", task=task.task_id,
                source_sha=_sha12(source), reason=reason,
                protocol=obs.timing_protocol_version)
            return obs

        wall_by_shape: dict[str, float] = {}
        base_by_shape: dict[str, float] = {}
        candidate_cvs: list[float] = []
        baseline_cvs: list[float] = []
        ratio_cvs: list[float] = []
        ci_widths: list[float] = []
        admission_errors: list[str] = []
        if publication_capable and not force_screening:
            obs.timing_grade = "publication"
            obs.timing_pair_count = max(1, int(self.cfg.max_variance_runs))
            per_shape, poisoned = self._bench_all(
                driver, shapes, workdir, env, snr_threshold=thr)
            if poisoned:
                _ev("WARN", "eval_bench_hack", task=task.task_id, source_sha=_sha12(source),
                    reason="post-timing correctness failed (bench-time reward hack)")
                return Observation(compiled=False, dtype=task.dtype, validation_passed=False,
                                   flagged_hack=True, hack_reason="bench-time output mismatch",
                                   error_text="reward-hack: kernel incorrect under timing")
            for sh in shapes:
                pairs = per_shape.get(sh.name, [])
                if not pairs:
                    continue
                cand_s = [p["candidate_ms"] for p in pairs]
                ref_s = [p["baseline_ms"] for p in pairs]
                try:
                    stats = _paired_timing_stats(
                        cand_s, ref_s,
                        noise_floor_pct=float(getattr(
                            self.cfg, "noise_floor_pct", 2.0)),
                        z=float(getattr(self.cfg, "paired_confidence_z", 1.96)),
                    )
                except ValueError as exc:
                    admission_errors.append(f"{sh.name}: {exc}")
                    continue
                obs.candidate_samples_by_shape[sh.name] = cand_s
                obs.baseline_samples_by_shape[sh.name] = ref_s
                obs.paired_ratio_samples_by_shape[sh.name] = stats["paired_ratios"]
                obs.paired_log_speedup_samples_by_shape[sh.name] = \
                    stats["paired_log_speedups"]
                obs.candidate_cv_by_shape[sh.name] = stats["candidate_cv_pct"]
                obs.baseline_cv_by_shape[sh.name] = stats["baseline_cv_pct"]
                obs.paired_ratio_cv_by_shape[sh.name] = stats["paired_ratio_cv_pct"]
                obs.paired_log_ci_by_shape[sh.name] = [
                    stats["log_ci_lo"], stats["log_ci_hi"]]
                obs.timing_classification_by_shape[sh.name] = stats["classification"]
                wall_by_shape[sh.name] = _median(cand_s)
                base_by_shape[sh.name] = _median(ref_s)
                candidate_cvs.append(stats["candidate_cv_pct"])
                baseline_cvs.append(stats["baseline_cv_pct"])
                ratio_cvs.append(stats["paired_ratio_cv_pct"])
                ci_widths.append(stats["ci_half_width_pct"])
                err = _publication_admission_error(
                    stats,
                    min_pairs=max(2, int(self.cfg.min_variance_runs)),
                    candidate_cv_threshold_pct=float(self.cfg.cv_threshold_pct),
                    baseline_cv_threshold_pct=float(getattr(
                        self.cfg, "baseline_cv_threshold_pct",
                        self.cfg.cv_threshold_pct)),
                    paired_ratio_cv_threshold_pct=float(getattr(
                        self.cfg, "paired_ratio_cv_threshold_pct",
                        self.cfg.cv_threshold_pct)),
                    paired_ci_threshold_pct=float(getattr(
                        self.cfg, "paired_ci_threshold_pct",
                        self.cfg.cv_threshold_pct)),
                )
                if err:
                    admission_errors.append(f"{sh.name}: {err}")
        else:
            # Explicit operator-requested screening only.  This path is useful
            # for debugging/parity but can never earn vendor-grade speed credit.
            obs.timing_protocol = "legacy-unpaired-v0"
            obs.timing_protocol_version = 0
            obs.timing_guarantees = {}
            obs.timing_grade = "screening"
            obs.performance_eligible = False
            for sh in shapes:
                cand, cand_cv, poisoned = self._bench_multi(
                    driver, sh, "candidate", workdir, env, snr_threshold=thr)
                # Anti-hack: candidate bench re-verifies correctness AFTER timing. A False
                # post-timing verdict => correct during checks but garbage while timed
                # (invocation-count hack) -> reject the whole eval, never reward it.
                if poisoned:
                    _ev("WARN", "eval_bench_hack", task=task.task_id, shape=sh.name,
                        source_sha=_sha12(source),
                        reason="post-timing correctness failed (bench-time reward hack)")
                    return Observation(compiled=False, dtype=task.dtype, validation_passed=False,
                                       flagged_hack=True, hack_reason="bench-time output mismatch",
                                       error_text="reward-hack: kernel incorrect under timing")
                ref = self._bench_multi(
                    driver, sh, "reference", workdir, env, snr_threshold=thr)[0]
                if cand is not None:
                    wall_by_shape[sh.name] = cand
                    candidate_cvs.append(cand_cv)
                if ref is not None:
                    base_by_shape[sh.name] = ref

        # Retain raw/summary evidence even when admission subsequently fails.
        obs.wall_by_shape = wall_by_shape
        obs.baseline_by_shape = base_by_shape
        # Runtime baseline identity, so a silent AITER->torch degradation cannot
        # be recorded as a vendor-beating win.
        obs.baseline_impl = self._last_baseline_impl
        obs.cv_pct = max(candidate_cvs) if candidate_cvs else None
        obs.baseline_cv_pct = max(baseline_cvs) if baseline_cvs else None
        obs.paired_ratio_cv_pct = max(ratio_cvs) if ratio_cvs else None
        obs.paired_ci_half_width_pct = max(ci_widths) if ci_widths else None
        if wall_by_shape:
            obs.wall_ms = max(wall_by_shape.values())
        if base_by_shape:
            obs.baseline_ms = max(base_by_shape.values())
        # Physics-integrity provenance for the speed-of-light gate: the HBM branch of
        # the floor is sound only if L2 was flushed between timed iterations, so the
        # flag must reflect the configuration the timing subprocess actually ran under.
        obs.cold_cache_verified = bool(wall_by_shape) and _cold_cache_timing(env, caps)

        timing_error = _timing_completeness_error(
            requested_names, wall_by_shape, base_by_shape)
        if timing_error:
            # No usable measurement came back for some requested shape - the bench
            # subprocess was killed/timed out, or the driver broke the pair protocol.
            # There is no timing evidence to judge, so this stays an infra failure.
            _ev("WARN", "eval_bench_incomplete", task=task.task_id,
                source_sha=_sha12(source), reason=timing_error)
            obs.timing_grade = "rejected"
            obs.performance_eligible = False
            obs.infra_error = True
            obs.error_text = f"infra: timing admission failed: {timing_error}"
            return obs
        if admission_errors:
            # Complete candidate+baseline timing exists for every requested shape; it
            # merely missed the CV/CI admission gates. That is MEASUREMENT NOISE, not
            # broken infrastructure, and the kernel already cleared every correctness
            # check including the post-timing re-verification. Flagging it infra_error
            # made kore.policy.grpo drop the turn from the training batch entirely,
            # throwing away a verified-correct signal because the node was busy.
            # Demote the timing instead: the reward ladder's ``correct_screening``
            # tier banks the correctness credit and grants no speed credit, which is
            # the honest verdict for a correct-but-unmeasurable candidate.
            reason = "; ".join(admission_errors)
            _ev("WARN", "eval_timing_unadmitted", task=task.task_id,
                source_sha=_sha12(source), reason=reason, cv_pct=obs.cv_pct,
                paired_ratio_cv_pct=obs.paired_ratio_cv_pct,
                paired_ci_half_width_pct=obs.paired_ci_half_width_pct)
            obs.timing_grade = "screening"
            obs.performance_eligible = False
            obs.error_text = f"timing not admitted (measurement noise): {reason}"
            # The P5 bonus below is reachable only from the timed tier, so spending
            # two more rocprofv3 runs here would buy nothing.
            return obs

        # P5 (flagship novelty): dense hardware-counter efficiency, baseline-relative.
        # Feature-flagged (profile_reward_weight>0) and fully fail-safe: any profiler
        # hiccup leaves profile_efficiency=None and never affects the correctness/
        # speedup verdict. Collected once on the primary shape only (rocprof is slow).
        if getattr(self.cfg, "profile_reward_weight", 0.0) > 0.0:
            # Measured, not inferred: the two rocprofv3 passes are the profiler's
            # entire GPU cost, and ``last_profiler_seconds`` is how the caller
            # charges them to ``profiler_gpu_seconds`` instead of to the verifier.
            profile_started = time.perf_counter()
            try:
                obs.profile_efficiency = self._collect_profile(driver, shapes[0], workdir, env)
            except Exception as e:  # pragma: no cover - GPU/rocprof only
                _ev("DEBUG", "profile_error", task=task.task_id, error=str(e)[:200])
                obs.profile_efficiency = None
            finally:
                self._last_profiler_seconds = time.perf_counter() - profile_started
            # Two rocprofv3 runs produced a number; the reward's P5 gate additionally
            # needs the empirical evidence backing counter shaping to exist and pass.
            passed, fingerprint = self._profile_evidence(task, obs.profile_efficiency)
            obs.profile_evidence_passed = passed
            obs.profile_evidence_fingerprint = fingerprint
            _ev("DEBUG", "profile_evidence", task=task.task_id, passed=passed,
                fingerprint=fingerprint, efficiency=obs.profile_efficiency)
        return obs

    # ------------------------------------------------------------------ #
    # Correctness oracle: the metamorphic prong + the consumer-visible report
    # ------------------------------------------------------------------ #
    def _metamorphic_prong(self, task: Task, plan: MetamorphicPlan,
                           shapes: list[Shape], workdir: Path,
                           env: dict) -> tuple[ProngStatus, Optional[str], Optional[dict]]:
        """Run the structural prong in its own subprocess in the staged workdir.

        Returns ``(status, inconclusive_reason, diagnostic_payload)``. The
        verdict is read ONLY from the ``allclose:`` / ``SNR:`` literals the
        runner prints after the candidate has finished: those are the same
        literals the task driver uses, so ``kore.reward.scan_for_hacks`` already
        rejects any candidate whose source contains them, and the last-match rule
        already beats an in-run print. The ``KORE_METAMORPHIC:`` JSON is
        diagnostic and is discarded whenever it disagrees with that verdict, so
        forging it can never turn a rejection into an acceptance.

        ``inconclusive_reason`` is non-``None`` exactly when the prong was meant
        to run and produced no verdict; the caller must then refuse to call the
        candidate verified.
        """
        name, kind = "metamorphic", "metamorphic"
        dims, declared, note = select_metamorphic_shape(
            shapes, max_elements=DEFAULT_MAX_ELEMENTS)
        if dims is None:
            # A property of the task's declared shapes, not of this run: report
            # it as "does not apply" rather than failing an honest candidate.
            return (ProngStatus(name, kind, "not-applicable",
                                "kore.verify.production.select_metamorphic_shape",
                                note),
                    None, None)

        shim = workdir / RUNNER_SHIM_NAME
        try:
            # Staged read-only like driver.py/reference.py. Re-staging tolerates
            # the read-only mode a previous run (or a task source of the same
            # name) left behind, so a second call in one workdir is not an error.
            if shim.exists():
                os.chmod(shim, 0o644)
            shim.write_text(RUNNER_SHIM_SOURCE)
            os.chmod(shim, 0o444)
        except OSError as exc:
            reason = f"could not stage {RUNNER_SHIM_NAME}: {exc}"
            return ProngStatus(name, kind, "inconclusive", "staging", reason), reason, None

        spec = shape_spec(dims)
        declared_spec = shape_spec(declared or dims)
        cmd = [sys.executable, str(shim),
               "--shape", spec,
               "--op-class", plan.op_class,
               "--source-family", plan.source_family,
               "--dtype", plan.dtype,
               "--shape-declared", declared_spec]
        t0 = time.perf_counter()
        with _LOG.timer("verify_metamorphic", task=task.task_id, shape=spec):
            rc, out, timed = self._exec(cmd, workdir, env, self.correctness_timeout)
        took_s = round(time.perf_counter() - t0, 3)
        run_kind, msg = self._classify(out, rc, timed)
        evidence = (
            f"kore.verify.runner subprocess in the staged workdir, shape {spec}"
            + (f" (stands in for {declared_spec})" if spec != declared_spec else "")
            + f", op_class={plan.op_class}, relations={len(plan.relations)}; verdict "
              "from the driver-protocol allclose:/SNR: literals (last match)")

        if run_kind == "infra":
            reason = f"runner hit an infrastructure failure: {_tail(msg, 200)}"
            return ProngStatus(name, kind, "inconclusive", evidence, reason), reason, None
        ac = _last(_ALLCLOSE, out)
        if rc != 0 or ac is None:
            reason = (f"runner published no verdict (rc={rc}, "
                      f"allclose_line={'yes' if ac else 'no'}): {_tail(out, 300)}")
            return ProngStatus(name, kind, "inconclusive", evidence, reason), reason, None

        passed = ac.group(1).lower() == "true"
        payload = parse_metamorphic_report(out)
        if payload is not None and (payload.get("state") != "verdict"
                                    or bool(payload.get("verified")) is not passed):
            _ev("WARN", "metamorphic_payload_inconsistent", task=task.task_id,
                verdict=passed)
            payload = None
        detail = ""
        if payload is not None:
            failures = payload.get("failures") or []
            detail = "; ".join(str(f) for f in failures)[:400]
            # The reported false-accept bound models the output extent from the
            # shape; if the candidate's own output disagrees with that model the
            # bound would be wrong, so mark it unusable instead of publishing it.
            observed = payload.get("output_elements")
            expected = expected_output_elements(plan.op_class, dims)
            payload["element_model_ok"] = (
                observed is None or expected is None or int(observed) == int(expected))
        if not detail and not passed:
            detail = _tail(out, 300)
        _ev("DEBUG", "verify_metamorphic", task=task.task_id, shape=spec,
            op_class=plan.op_class, passed=passed, took_s=took_s,
            relations=len(plan.relations))
        return (ProngStatus(name, kind, "pass" if passed else "fail", evidence,
                            sanitize_detail(detail, 400) if detail else ""),
                None, payload)

    def _publish_oracle_report(self, task: Task, shapes: list[Shape], *,
                               verified: bool, random_ok: bool,
                               adv_failures: list[str], metamorphic: ProngStatus,
                               determinism_ran: bool, determinism_ok: bool,
                               do_bench: bool,
                               meta_payload: Optional[dict] = None,
                               detail: str = "") -> OracleReport:
        """Record which prongs ran, and the honest false-accept bound.

        Published on :attr:`last_oracle_report` and as one structured JSONL
        event, because a four-prong claim is only meaningful if a consumer can
        see which prongs were live for a given candidate and how large the
        residual statistical risk behind the verdict is.
        """
        n_shapes = len(shapes)
        gate_on = os.environ.get("KORE_VERIFIED_CORRECTNESS") == "1"

        # -- random ------------------------------------------------------- #
        random_evidence = (
            f"driver stdout (allclose:/SNR:, last match): reseeded random trials "
            f"per shape against the fp32 reference, {n_shapes} shape(s), every "
            f"per-shape SNR vs the task threshold")
        if random_ok:
            random_state, random_detail = "pass", ""
        elif adv_failures:
            # One combined verdict per shape: an adversarial regime failed, so the
            # random trials' own contribution is not separable from this output.
            random_state = "inconclusive"
            random_detail = ("not separable: the driver publishes one allclose: "
                             "verdict per shape and an adversarial regime failed")
        else:
            random_state, random_detail = "fail", "no adversarial regime failed"

        # -- adversarial --------------------------------------------------- #
        family = ""
        raw = getattr(task, "raw", None)
        if isinstance(raw, Mapping):
            family = str(raw.get("op_family") or "")
        battery = (meta_payload or {}).get("adversarial_battery")
        if battery is None:
            battery = ("generic" if family in generic_adversarial_families()
                       else "unknown")
        adv_evidence = (
            "driver stdout: the enumerated adversarial fills share the same "
            f"allclose: verdict; ADVERSARIAL_FAIL[...] names a failing regime "
            f"(battery={battery})")
        if not gate_on:
            adversarial = ProngStatus(
                "adversarial", "adversarial", "off",
                "gated off: set KORE_VERIFIED_CORRECTNESS=1 to run the driver's "
                "enumerated adversarial battery", "")
        elif adv_failures:
            adversarial = ProngStatus(
                "adversarial", "adversarial", "fail", adv_evidence,
                "regimes: " + ", ".join(adv_failures[:8]))
        elif battery == "unknown":
            adversarial = ProngStatus(
                "adversarial", "adversarial", "unknown", adv_evidence,
                "the environment cannot observe whether this reference supplies "
                "an adversarial battery to the driver")
        elif battery == "none":
            adversarial = ProngStatus(
                "adversarial", "adversarial", "not-applicable", adv_evidence,
                "the driver has no adversarial battery for this reference")
        else:
            adversarial = ProngStatus(
                "adversarial", "adversarial", "pass" if random_ok else "fail",
                adv_evidence, "" if random_ok else "no regime named a failure")

        # -- determinism ---------------------------------------------------- #
        det_evidence = (
            "environment: identical-input correctness re-run compared under "
            "determinism_snr_tol_db"
            + (" (plus the driver's post-timing re-verification on the cached "
               "candidate module)" if do_bench else ""))
        if not getattr(self.cfg, "verifier_determinism_check", False):
            determinism = ProngStatus(
                "determinism", "determinism", "off", det_evidence,
                "CONFIG.verifier_determinism_check is off")
        elif not determinism_ran:
            determinism = ProngStatus(
                "determinism", "determinism", "off", det_evidence,
                "not run: the candidate was already rejected by an earlier prong")
        else:
            determinism = ProngStatus(
                "determinism", "determinism",
                "pass" if determinism_ok else "fail", det_evidence, "")

        op_class = task_output_op_class(task)
        if meta_payload is not None and meta_payload.get("element_model_ok") is False:
            op_class = "generic"  # observed output extent contradicts the model
        extra: dict[str, Any] = {}
        if meta_payload is not None:
            extra["metamorphic"] = {
                k: meta_payload.get(k)
                for k in ("op_class", "shape_used", "relations", "worst_rel_err",
                          "worst_snr_db", "metamorphic_rtol",
                          "metamorphic_snr_db_min", "candidate_calls")
            }
        report = build_oracle_report(
            task_id=task.task_id, verified=verified,
            prongs=(ProngStatus("random", "random", random_state,
                                random_evidence, random_detail),
                    adversarial, metamorphic, determinism),
            op_class=op_class,
            shape_dims=[dict(sh.dims) for sh in shapes],
            detail=detail, extra=extra)
        self._last_oracle_report = report
        # One line per candidate, so the long static ``bound_basis`` prose rides
        # only when it is explaining why there is NO number; everything needed to
        # reconstruct the bound (p, m, trials) is on the line either way.
        payload = {k: v for k, v in report.to_dict().items()
                   if k not in ("prongs", "bound_basis", "extra")}
        if report.false_accept_bound is None:
            payload["bound_basis"] = report.bound_basis
        _ev("INFO", "oracle_report", task=task.task_id, **payload,
            prong_states={p.name: p.state for p in report.prongs})
        return report

    def _profile_evidence(self, task: Task,
                          efficiency: Optional[float]) -> tuple[bool, Optional[str]]:
        """Validated held-out evidence backing a counter-efficiency bonus, if any.

        The P5 profile bonus is empirical shaping, so it is admitted only when BOTH
        halves exist for THIS observation: a usable efficiency in [0, 1] from the
        rocprofv3 passes, and preregistered evidence for this task's operator family
        that passes :meth:`kore.reward.shaping.FamilyShapingEvidence.passes` under the
        same fingerprinted physical model. The evidence artifact and its expected
        fingerprint must be configured explicitly
        (``physics_shaping_evidence_path`` / ``..._fingerprint``); with none
        configured this returns ``(False, None)`` and the bonus stays withheld.

        Imported lazily - the physics bridge pulls in ``kore.analysis.roofline``,
        which an evaluation that is not profiling has no reason to load.
        """
        if not (isinstance(efficiency, (int, float))
                and not isinstance(efficiency, bool)
                and math.isfinite(float(efficiency))
                and 0.0 <= float(efficiency) <= 1.0):
            return False, None
        if not (getattr(self.cfg, "physics_shaping_evidence_path", None)
                and getattr(self.cfg, "physics_shaping_evidence_fingerprint", None)):
            return False, None
        try:
            from kore.reward.physics import model_from_config
            from kore.reward.shaping import evidence_for_task

            model = model_from_config(self.cfg)
            evidence = evidence_for_task(task, self.cfg, model.fingerprint)
        except Exception as exc:  # noqa: BLE001 - unresolvable evidence => no bonus
            _ev("DEBUG", "profile_evidence_unavailable", task=task.task_id,
                error=str(exc)[:200])
            return False, None
        if evidence is None or not evidence.report_fingerprint:
            return False, None
        return True, str(evidence.report_fingerprint)

    def collect_counters(self, source: str, shape: Optional["Shape"] = None) -> Optional[dict]:
        """PUBLIC: rocprofv3 PMC counters for a kernel (Pillar 4 grounded reasoning).

        Stages an isolated workdir (like ``evaluate``), profiles the CANDIDATE on one
        shape (``primary`` by default), and returns aggregated ``{counter: value}`` or
        ``None`` if the profiler is unavailable / fails. Fully fail-safe (never raises)
        so grounded-reasoning datagen degrades gracefully to the templated path.
        """
        import glob as _glob
        import tempfile as _tmp
        try:
            from kore.verifier.parsers.rocprofv3 import parse_rocprofv3_csv
            from kore.verifier.pmc import COUNTER_SETS
            try:
                from kore.verifier.pmc import counter_passes
                passes = counter_passes("grounding")   # real gfx950/gfx942 BW/L2/occupancy set
            except Exception:  # noqa: BLE001 - older pmc: single-pass fallback
                passes = [COUNTER_SETS["full"]]
        except Exception:  # noqa: BLE001
            return None
        # Sandbox-only source-budget gate (skipped on the default path).
        if self._sandbox_enabled and (
                len(source.encode("utf-8")) > self.isolation_policy.budget.max_source_bytes):
            self._last_execution_status = ExecutionStatus.POLICY_VIOLATION
            return None
        sh = shape or self.task.shape("primary") or self.task.shape("minimal") or (
            self.task.shapes[0] if self.task.shapes else Shape("default", {}))
        workdir = Path(tempfile.mkdtemp(prefix=f"pmc_{self.task.task_id}_"))
        previous_source, previous_task = self._active_source, self._active_task
        self._active_source, self._active_task = source, self.task
        try:
            task_sources = _task_source_files(self.task)
            if self._sandbox_enabled and (
                    sum(p.stat().st_size for p in task_sources)
                    > self.isolation_policy.budget.max_task_bytes):
                self._last_execution_status = ExecutionStatus.POLICY_VIOLATION
                return None
            for p in task_sources:
                dst = workdir / p.name
                shutil.copy(p, dst)
                os.chmod(dst, 0o444)
            candidate_name = _hip.candidate_filename_for_task(self.task)
            (workdir / candidate_name).write_text(source)
            os.chmod(workdir / candidate_name, 0o444)
            driver = workdir / "driver.py"
            env = (self._env(private_root=workdir / ".sandbox")
                   if self._sandbox_enabled else self._env())
            agg: dict = {}
            # The grounding set spans SQ+GRBM+TCC and cannot be one --pmc pass, so run
            # one rocprofv3 invocation per pass and merge the disjoint counter dicts.
            for pcounters in passes:
                outdir = _tmp.mkdtemp(prefix="pmc_cand_", dir=str(workdir))
                cmd = ["rocprofv3", "--pmc", *pcounters, "-d", outdir,
                       "--output-format", "csv", "--", sys.executable, str(driver),
                       "--bench-mode", "--impl", "candidate", "--warmup", "2", "--iters", "3",
                       *sh.as_args()]
                rc, out, timed = self._exec(cmd, workdir, env, self.bench_timeout)
                if timed or rc != 0:
                    continue  # a failed pass never aborts grounding; keep what we got
                csvs = _glob.glob(os.path.join(outdir, "**", "*counter_collection.csv"),
                                  recursive=True) or [
                    c for c in _glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True)
                    if "agent_info" not in os.path.basename(c)]
                for c in csvs:
                    try:
                        for k in parse_rocprofv3_csv(c):
                            for name, val in k.counters.items():
                                agg[name] = agg.get(name, 0) + int(val)
                            # Capture resource fields (VGPR/LDS/warps) so grounded
                            # reasoning + roofline can compute occupancy.
                            for attr in ("vgpr_count", "lds_bytes", "num_warps"):
                                v = getattr(k, attr, None)
                                if v is not None and attr not in agg:
                                    agg[attr] = v
                    except Exception:  # noqa: BLE001
                        pass
            return agg or None
        except Exception:  # noqa: BLE001
            return None
        finally:
            self._active_source, self._active_task = previous_source, previous_task
            shutil.rmtree(workdir, ignore_errors=True)

    def collect_kernel_trace(self, source: str,
                             shape: Optional["Shape"] = None) -> Optional[list]:
        """PUBLIC: per-dispatch kernel durations for a candidate (coverage signal).

        Stages an isolated workdir exactly like :meth:`collect_counters`, runs the
        candidate under ``rocprofv3 --kernel-trace``, and returns the parsed
        :class:`~kore.verifier.parsers.rocprofv3.KernelDispatch` list, or ``None``
        if the profiler is unavailable or produced nothing usable.

        This answers a question the PMC path cannot: what SHARE of the region's GPU
        time the candidate's own kernels account for. See
        :mod:`kore.reward.coverage` -- a kernel accounting for 0.014% of GPU time
        has an end-to-end ceiling of 1.00014x no matter how fast it is, and a
        kernel accounting for 0.0% never ran at all.

        Deliberately additive and fail-safe. It touches no correctness or timing
        path: :meth:`evaluate` and :meth:`step` do not call it, a candidate's
        verdict does not depend on it, and every failure mode returns ``None``
        rather than a partial trace that would understate coverage.

        UNVALIDATED ON HARDWARE. The invocation mirrors the PMC path that IS
        hardware-proven, but rocprofv3's kernel-trace export filename and column
        names have not been confirmed on this ROCm build, so the parser accepts
        several naming variants and the whole method fails closed. Anything reading
        it must treat ``None`` as "no measurement" -- never as zero coverage --
        which is why ``profiling_reward_weight`` additionally requires a measured
        evidence receipt before it can shape a reward.
        """
        import glob as _glob
        import tempfile as _tmp
        # The WHOLE body is guarded, including the sandbox/shape resolution that
        # ``collect_counters`` performs before its own try. This method's promise is
        # that it never raises into a rollout, and "never" should not depend on the
        # env being fully constructed.
        try:
            from kore.verifier.parsers.rocprofv3 import parse_kernel_dispatches

            if self._sandbox_enabled and (
                    len(source.encode("utf-8"))
                    > self.isolation_policy.budget.max_source_bytes):
                self._last_execution_status = ExecutionStatus.POLICY_VIOLATION
                return None
            sh = shape or self.task.shape("primary") or self.task.shape("minimal") or (
                self.task.shapes[0] if self.task.shapes else Shape("default", {}))
        except Exception:  # noqa: BLE001
            return None
        workdir = Path(tempfile.mkdtemp(prefix=f"ktrace_{self.task.task_id}_"))
        previous_source, previous_task = self._active_source, self._active_task
        self._active_source, self._active_task = source, self.task
        try:
            for p in list(self.task.dir.glob("*.py")):
                dst = workdir / p.name
                shutil.copy(p, dst)
                os.chmod(dst, 0o444)
            (workdir / "kernel.py").write_text(source)
            os.chmod(workdir / "kernel.py", 0o444)
            driver = workdir / "driver.py"
            env = (self._env(private_root=workdir / ".sandbox")
                   if self._sandbox_enabled else self._env())
            outdir = _tmp.mkdtemp(prefix="ktrace_", dir=str(workdir))
            # --bench-mode + --impl candidate for the same reason the PMC path needs
            # them: drivers honor --impl only in bench mode, so without it the trace
            # would describe the correctness run instead of the benched candidate.
            cmd = ["rocprofv3", "--kernel-trace", "-d", outdir,
                   "--output-format", "csv", "--", sys.executable, str(driver),
                   "--bench-mode", "--impl", "candidate", "--warmup", "2",
                   "--iters", "3", *sh.as_args()]
            rc, out, timed = self._exec(cmd, workdir, env, self.bench_timeout)
            if timed or rc != 0:
                return None
            csvs = _glob.glob(os.path.join(outdir, "**", "*kernel_trace.csv"),
                              recursive=True) or [
                c for c in _glob.glob(os.path.join(outdir, "**", "*.csv"),
                                      recursive=True)
                if "agent_info" not in os.path.basename(c)]
            dispatches: list = []
            for c in csvs:
                try:
                    dispatches.extend(parse_kernel_dispatches(c))
                except Exception:  # noqa: BLE001 - a bad export is no measurement
                    pass
            return dispatches or None
        except Exception:  # noqa: BLE001
            return None
        finally:
            self._active_source, self._active_task = previous_source, previous_task
            shutil.rmtree(workdir, ignore_errors=True)

    def _collect_profile(self, driver: Path, sh: Shape, workdir: Path,
                         env: dict) -> Optional[float]:
        """rocprofv3 PMC on candidate + reference -> baseline-relative efficiency.

        Returns a score in [0,1] (see kore.reward.profile_reward) or None if the
        profiler is unavailable or produced no usable counters. Never raises to the
        caller path that matters (wrapped by the caller's try/except)."""
        import glob as _glob
        import tempfile as _tmp
        from kore.reward import profile_reward as _pr
        from kore.verifier.parsers.rocprofv3 import parse_rocprofv3_csv
        from kore.verifier.pmc import COUNTER_SETS

        counters = COUNTER_SETS["full"]

        def _counters_for(impl: str) -> Optional[dict]:
            outdir = _tmp.mkdtemp(prefix=f"pmc_{impl}_", dir=str(workdir))
            # --bench-mode is REQUIRED: drivers honor --impl (candidate vs reference)
            # ONLY in bench mode; without it both runs execute correctness on the
            # candidate -> identical work -> a degenerate ~1.0 profile score. Small
            # warmup/iters keep rocprof's multi-pass replay cheap.
            cmd = ["rocprofv3", "--pmc", *counters, "-d", outdir,
                   "--output-format", "csv", "--",
                   sys.executable, str(driver), "--bench-mode", "--impl", impl,
                   "--warmup", "2", "--iters", "3", *sh.as_args()]
            rc, out, timed = self._exec(cmd, workdir, env, self.bench_timeout)
            if timed or rc != 0:
                _ev("DEBUG", "profile_run", task=self.task.task_id, impl=impl,
                    ok=False, rc=rc)
                return None
            # rocprofv3 writes <pid>_counter_collection.csv (+ an agent_info.csv we
            # must ignore). Prefer the counter file; never parse agent_info.
            csvs = _glob.glob(os.path.join(outdir, "**", "*counter_collection.csv"),
                              recursive=True)
            if not csvs:
                csvs = [c for c in _glob.glob(os.path.join(outdir, "**", "*.csv"),
                                              recursive=True)
                        if "agent_info" not in os.path.basename(c)]
            kernels = []
            for c in csvs:
                try:
                    kernels.extend(parse_rocprofv3_csv(c))
                except Exception:
                    pass
            if not kernels:
                return None
            # Aggregate all dispatches for this impl (a kernel may launch several).
            agg: dict[str, int] = {}
            for k in kernels:
                for name, val in k.counters.items():
                    agg[name] = agg.get(name, 0) + int(val)
            return agg or None

        cand = _counters_for("candidate")
        ref = _counters_for("reference")
        if not cand or not ref:
            return None
        score = _pr.profile_efficiency_score(cand, ref)
        _ev("DEBUG", "profile_score", task=self.task.task_id, score=score)
        return score

    @contextmanager
    def _timing_lock(self):
        """Serialize the TIMING phase per physical GPU (advisory flock).

        Compilation + correctness are deterministic, so many workers can share a GPU
        for them (oversubscription uses the idle cores). But wall-clock TIMING needs
        the GPU to itself - concurrent kernels/L2-flushes inflate and destabilize the
        measurement (CV blows up). Workers pinned to the same physical GPU take an
        exclusive lock on ``<tmp>/kore_timing_gpu_<id>.uid<uid>.lock`` around timing
        only, so speedups stay clean while compiles keep running in parallel. Disable
        with KORE_TIMING_LOCK=0.

        The lockfile is opened through :func:`_open_private_lockfile` (O_NOFOLLOW,
        O_CLOEXEC, mode 0600, ownership re-checked on the fd) because the shared
        tmpdir is world-writable on a multi-user cluster. The name is uid-scoped so
        same-uid workers - the real deployment - still serialize on one lock per
        physical GPU, while a foreign uid squatting the path can neither be blocked
        by us nor block us. If the path cannot be opened safely we time WITHOUT the
        lock (noisier measurement, admission gates still apply) rather than fail the
        evaluation."""
        if os.environ.get("KORE_TIMING_LOCK", "1").strip().lower() in ("0", "false", "no"):
            yield
            return
        physid = str(self._gpu if self._gpu is not None
                     else os.environ.get("HIP_VISIBLE_DEVICES", "0")).split(",")[0].strip() or "0"
        lp = (Path(tempfile.gettempdir())
              / f"kore_timing_gpu_{physid}.uid{os.getuid()}.lock")
        fd = _open_private_lockfile(lp)
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError:
                os.close(fd)
                fd = None
        if fd is None:
            _ev("WARN", "timing_lock_unavailable", gpu=physid, path=str(lp))
            yield
            return
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _driver_capabilities(self, driver: Path, workdir: Path, env: dict) -> dict:
        """Probe and cache the driver's explicit versioned capability handshake."""
        cached = getattr(self, "_driver_caps_cache", None)
        if cached is not None:
            return cached
        timeout = min(max(int(self.correctness_timeout), 1), 30)
        rc, out, timed = self._exec(
            [sys.executable, str(driver), "--kore-driver-capabilities"],
            workdir, env, timeout)
        caps = _parse_driver_capabilities(out) if (not timed and rc == 0) else {}
        if not caps:
            caps = {
                "protocol": 0,
                "protocol_id": "unknown",
                "performance_eligible": False,
                "ineligible_reason": (
                    "driver did not advertise a recognized timing protocol"),
            }
        elif not _supports_batch_bench(caps):
            caps = dict(caps)
            caps["performance_eligible"] = False
            caps.setdefault(
                "ineligible_reason",
                "driver advertised only partial publication guarantees")
        self._driver_caps_cache = caps
        _ev("DEBUG", "driver_capabilities", task=self.task.task_id,
            protocol=caps.get("protocol"), batch=_supports_batch_bench(caps),
            probe_rc=rc, timed_out=timed)
        return caps

    @staticmethod
    def _shape_spec(sh: Shape) -> str:
        return ",".join(f"{k}={v}" for k, v in sh.dims.items()) if sh.dims else "default"

    def _bench_all(self, driver: Path, shapes, workdir: Path, env: dict,
                   snr_threshold: Optional[float] = None):
        """Time ALL shapes (candidate+reference, ``max_variance_runs`` repeats each) in
        ONE ``--bench-both --shapes`` process, under a SINGLE per-GPU timing-lock hold.

        Collapsing the per-shape spawns to one import means the exclusive (locked)
        window is ~one torch import + the tiny GPU timing, so oversubscribed workers
        barely wait -> max throughput with clean, contention-free measurements.
        Returns ``({shape_name: [validated_pair_dict, ...]}, poisoned)``."""
        n_max = max(1, self.cfg.max_variance_runs)
        specs = [self._shape_spec(sh) for sh in shapes]
        cmd = [sys.executable, str(driver), "--bench-both", "--shapes", ";".join(specs),
               "--warmup", str(self.cfg.warmup_iters), "--iters", str(self.cfg.bench_iters),
               "--repeat", str(n_max)]
        with self._timing_lock(), _LOG.timer("bench_all", task=self.task.task_id,
                                             n_shapes=len(shapes)):
            rc, out, timed = self._exec(cmd, workdir, env, self.bench_timeout)
        if timed or rc != 0:
            _ev("DEBUG", "bench_all", task=self.task.task_id, ok=False, rc=rc)
            return {}, False
        # Emitted once per bench process, before the per-shape blocks, so read it
        # off the whole output. This is the only runtime evidence of which
        # baseline was really timed.
        _impl = _last(_BASELINE_IMPL, out)
        self._last_baseline_impl = _impl.group(1) if _impl else None
        blocks = out.split("SHAPE_BEGIN")[1:]  # per-shape, in the order we passed them
        if len(blocks) != len(shapes):
            return {}, False
        threshold = self._snr_threshold if snr_threshold is None else snr_threshold
        result: dict[str, tuple] = {}
        for sh, spec, block in zip(shapes, specs, blocks):
            marker, _, _body = block.partition("\n")
            if marker.strip() != spec:
                return {}, False
            # The shared driver emits one late correctness verdict in EVERY
            # shape block.  Validate per block; looking only at the final global
            # verdict would let an earlier failing shape hide behind a later pass.
            ac = _last(_ALLCLOSE, block)
            snr = _last(_SNR, block)
            if ac is None and snr is None:
                return {}, False
            if (ac and ac.group(1).lower() == "false") or \
               (snr and float(snr.group(1)) < threshold):
                return {}, True
            pairs, pair_error = _parse_timing_pairs(block, n_max)
            if pair_error:
                _ev("DEBUG", "bench_all_pair_error", task=self.task.task_id,
                    shape=sh.name, error=pair_error)
                return {}, False
            result[sh.name] = pairs
        return result, False

    def _bench_multi(self, driver: Path, sh: Shape, impl: str, workdir: Path, env: dict,
                     snr_threshold: Optional[float] = None):
        """Bench a (shape, impl) ``min..max_variance_runs`` times; return
        (median-of-medians, CV%, poisoned).

        ``poisoned`` (candidate only) is True when the driver's POST-TIMING
        correctness re-verification failed - i.e. the kernel produced correct output
        for the correctness calls but garbage while being timed (the invocation-count
        timing hack). The timed window (warmup/iters) is RANDOMIZED per run so a
        stateful kernel cannot know which call indices are timed vs verified.
        """
        import random as _random
        threshold = self._snr_threshold if snr_threshold is None else snr_threshold
        samples: list[float] = []
        n_min = max(1, self.cfg.min_variance_runs)
        n_max = max(n_min, self.cfg.max_variance_runs)
        poisoned = False
        for i in range(n_max):
            # randomized timed window (defeats fixed-call-index bench sniffing)
            w = _random.randint(max(4, self.cfg.warmup_iters - 3), self.cfg.warmup_iters + 4)
            it = _random.randint(max(8, self.cfg.bench_iters - 5), self.cfg.bench_iters + 6)
            cmd = [sys.executable, str(driver), "--bench-mode", "--impl", impl,
                   "--warmup", str(w), "--iters", str(it), *sh.as_args()]
            with self._timing_lock(), _LOG.timer("bench_exec", task=self.task.task_id,
                                                 shape=sh.name, impl=impl, run=i):
                rc, out, timed = self._exec(cmd, workdir, env, self.bench_timeout)
            if timed or rc != 0:
                break
            # post-timing correctness verdict (candidate driver only): a False
            # allclose or a sub-threshold SNR AFTER the timed loop is a hack.
            if impl == "candidate":
                ac = _last(_ALLCLOSE, out)
                snr = _last(_SNR, out)
                if ac is None and snr is None:
                    samples = []
                    break
                if (ac and ac.group(1).lower() == "false") or \
                   (snr and float(snr.group(1)) < threshold):
                    poisoned = True
                    break
            m = _last(_MEDIAN, out)
            if m:
                samples.append(float(m.group(1)))
            if i + 1 >= n_min and len(samples) >= n_min and _cv_pct(samples) <= self.cfg.cv_threshold_pct:
                break
        if poisoned:
            return None, float("inf"), True
        if not samples:
            _ev("DEBUG", "bench_shape", task=self.task.task_id, shape=sh.name, impl=impl,
                median_ms=None, cv_pct=None, runs=0)
            return None, float("inf"), False
        med, cv = _median(samples), _cv_pct(samples)
        _ev("DEBUG", "bench_shape", task=self.task.task_id, shape=sh.name, impl=impl,
            median_ms=round(med, 4), cv_pct=round(cv, 3), runs=len(samples))
        return med, cv, False


def _determinism_stable(snr1: Optional[float], snr2: Optional[float],
                        ok2: bool, tol_db: float) -> tuple[bool, str]:
    """Anti-hack determinism verdict: is a second correctness run consistent?

    A kernel that passes the SNR gate by LUCK (partly random output) will fail or
    swing wildly on a re-run. Returns ``(stable, reason)``. Stable requires the
    re-run to still be correct AND its SNR to stay within ``tol_db`` of the first
    run. ``tol_db`` is generous enough to spare legitimate atomic-reduction jitter.
    """
    if not ok2:
        return False, "non-deterministic: 2nd correctness run failed the SNR gate"
    if snr1 is not None and snr2 is not None and abs(snr1 - snr2) > tol_db:
        return False, (f"non-deterministic: SNR drifted {abs(snr1 - snr2):.1f} dB "
                       f"(> {tol_db:.1f} dB) between identical runs")
    return True, ""


def _tail(s: str, n: int = 800) -> str:
    s = s.strip()
    return s[-n:] if len(s) > n else s


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"
