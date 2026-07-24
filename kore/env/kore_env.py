"""KoreEnv: the verified evaluation environment.

Wraps the KernelForge verifier contract into a task-bound ``step(source)`` call
that returns a reward :class:`Observation`. Hardening (see audits):

* **No verdict forgery.** The candidate ``kernel.py`` is imported by the driver
  and could print fake ``SNR:``/``median_ms:`` lines. We parse the *last* match
  (the driver prints its verdict after calling the candidate) AND the anti-hack
  scanner rejects any candidate that prints a verdict literal.
* **Execution boundary.** Each eval gets a private workdir/environment, bounded
  output, and process-group cleanup. The default backend is explicitly
  ``trusted-code-only``: these controls do not isolate hostile same-UID code.
  Production/untrusted policy requires an approved external broker and signed
  verdict; it never falls back to this subprocess path.
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
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Optional

from kore.config import CONFIG
from kore.env.evaluation_contract import (
    build_evaluation_contract,
    contract_is_cacheable,
    observation_satisfies_contract,
)
from kore.env.replay import ReplayCache
from kore.obs import get_logger
from kore.reward.reward import Observation, scan_for_hacks
from kore.reward.reward import _worst_speedup
from kore.reward.stats import cv_pct as _cv_pct
from kore.reward.stats import median as _median
from kore.sandbox.config import SandboxConfig
from kore.sandbox.controller import IsolationController, create_isolation_controller
from kore.sandbox.environment import build_candidate_environment
from kore.sandbox.errors import PolicyViolation, SandboxError
from kore.sandbox.models import (
    ExecutionKind,
    ExecutionStatus,
    SandboxRequest,
    SandboxResponse,
)
from kore.sandbox.signing import VerdictSignatureVerifier
from kore.tasks.base import Shape, Task

_LOG = get_logger("env")


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

_SNR = re.compile(r"SNR:\s*([-\d.eE]+)")
_ALLCLOSE = re.compile(r"allclose:\s*(True|False)", re.IGNORECASE)
_MEDIAN = re.compile(r"median_ms:\s*([-\d.eE]+)")
# Batched (--bench-both) per-impl medians: candidate + reference timed in ONE process.
_CAND_MED = re.compile(r"CAND_median_ms:\s*([-\d.eE]+)")
_REF_MED = re.compile(r"REF_median_ms:\s*([-\d.eE]+)")
# Candidate import/compile failure (the kernel's fault).
_COMPILE_ERR = re.compile(
    r"(SyntaxError|CompilationError|triton\..*Error|IndentationError|"
    r"NameError|out of resource|OutOfResources|AssertionError)",
    re.IGNORECASE,
)
# Infrastructure failure (NOT the kernel's fault) - never cache, never train on.
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


class KoreEnv:
    """Task-bound verified environment. One per task; call ``step`` per candidate."""

    def __init__(self, task: Task, config=CONFIG, use_replay: bool = True,
                 correctness_timeout: int = 300, bench_timeout: int = 300,
                 gpu: Optional[str] = None,
                 isolation_controller: Optional[IsolationController] = None,
                 sandbox_config: Optional[SandboxConfig] = None,
                 verdict_verifier: Optional[VerdictSignatureVerifier] = None):
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
            raise PolicyViolation("isolation controller policy does not match KoreEnv policy")
        self._last_execution_status: Optional[ExecutionStatus] = None
        self._active_source: Optional[str] = None
        self._active_task: Optional[Task] = None
        self._task_descriptor_cache: dict[str, dict] = {}
        self._cache_obj = ReplayCache(self.cfg.runs_dir / f"replay_{task.task_id}.jsonl") \
            if use_replay else None

    @property
    def _snr_threshold(self) -> float:
        return self._snr_threshold_for(self.task)

    def _snr_threshold_for(self, task: Task) -> float:
        t = getattr(task, "snr_threshold", None)
        return float(t) if t else self.cfg.snr_threshold_for(task.dtype)

    @property
    def last_execution_status(self) -> Optional[ExecutionStatus]:
        """Typed status from the most recent sandbox-controlled subprocess."""

        return self._last_execution_status

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
                aug = augment_shapes(shapes, max_shapes=int(getattr(
                    self.cfg, "shape_augment_max", 6)))
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
        _ev("INFO", "eval_start", task=task.task_id, n_shapes=n_shapes,
            source_sha=source_sha, do_bench=do_bench)

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

        hack = scan_for_hacks(source)
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
        )
        replay_ready = contract_is_cacheable(contract)
        if self.use_replay and self._cache_obj is not None and replay_ready:
            cached = self._cache_obj.get(task.task_id, source, context=contract)
            if cached is not None:
                _LOG.debug("cache hit", task=task.task_id, source_sha=source_sha,
                           compiled=cached.compiled, correct=cached.validation_passed)
                self._log_eval_done(task, cached, cached=True)
                return cached

        workdir = Path(tempfile.mkdtemp(prefix=f"kore_{task.task_id}_"))
        previous_source, previous_task = self._active_source, self._active_task
        self._active_source, self._active_task = source, task
        try:
            obs = self._run(task, source, shapes, workdir, do_bench)
        finally:
            self._active_source, self._active_task = previous_source, previous_task
            shutil.rmtree(workdir, ignore_errors=True)

        # Only cache DETERMINISTIC terminal verdicts - never transient infra errors.
        cacheable = (obs.compiled or obs.error_text) and not obs.infra_error
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
            )
            if final_contract == contract and contract_is_cacheable(final_contract):
                self._cache_obj.put(task.task_id, source, obs, context=contract)
        self._log_eval_done(task, obs, cached=False)
        return obs

    def _log_eval_done(self, task: Task, obs: Observation, cached: bool) -> None:
        """Final per-candidate verdict at INFO (structured), covering every path."""
        _ev("INFO", "eval_done", task=task.task_id, compiled=obs.compiled,
            correct=obs.validation_passed, snr_min=obs.snr_db,
            worst_speedup=_worst_speedup(obs), cv_pct=obs.cv_pct,
            infra_error=obs.infra_error, cached=cached)

    # ------------------------------------------------------------------ #
    def _env(
        self,
        private_root: Optional[Path] = None,
        task: Optional[Task] = None,
    ) -> dict:
        """Fresh allowlisted environment for a candidate-bearing subprocess."""

        root = private_root or (
            Path(tempfile.gettempdir()) / f"kore_env_{os.getpid()}_{id(self):x}"
        )
        active_task = task or self._active_task or self.task
        return build_candidate_environment(
            base_environment=os.environ,
            private_root=Path(root),
            project_root=Path(__file__).resolve().parents[2],
            gpu_target=(
                getattr(active_task, "gpu_target", None)
                or getattr(self.cfg, "gpu_target", "gfx950")
            ),
            gpu=(str(self._gpu) if self._gpu is not None else None),
            rocm_path=getattr(self.cfg, "rocm_path", None),
        )

    def _exec(self, cmd, workdir, env, timeout):
        """Execute through the configured isolation controller."""

        task = self._active_task or self.task
        source = self._active_source
        if source is None:
            try:
                source = (Path(workdir) / "kernel.py").read_text()
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
        task_dir = getattr(task, "dir", None)
        if task_dir is not None:
            for path in sorted(Path(task_dir).glob("*.py")):
                try:
                    files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError:
                    files[path.name] = "unreadable"
        descriptor = {
            "task_id": cache_key,
            "dtype": str(getattr(task, "dtype", "")),
            "gpu_target": str(getattr(task, "gpu_target", "")),
            "shapes": [
                {
                    "name": str(getattr(shape, "name", "")),
                    "dims": dict(getattr(shape, "dims", {})),
                }
                for shape in (getattr(task, "shapes", None) or [])
            ],
            "python_files": files,
        }
        self._task_descriptor_cache[cache_key] = descriptor
        return descriptor

    def _classify(self, out: str, returncode: int, timed_out: bool):
        """-> ('ok'|'compile'|'infra', message)."""
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
        # Stage private sources; make the oracle/driver read-only against
        # accidental mutation. This is not a same-UID filesystem security boundary.
        task_sources = list(task.dir.glob("*.py"))
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
        (workdir / "kernel.py").write_text(source)
        os.chmod(workdir / "kernel.py", 0o444)
        driver = workdir / "driver.py"
        env = self._env(workdir / ".sandbox", task=task)

        snr_by_shape: dict[str, float] = {}
        compiled = True
        validation_passed = True
        last_err: Optional[str] = None

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

        thr = self._snr_threshold_for(task)
        correct = validation_passed and bool(snr_by_shape) and all(v >= thr for v in snr_by_shape.values())

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
            if not stable:
                _ev("WARN", "eval_nondeterministic", task=task.task_id,
                    source_sha=_sha12(source), reason=reason)
                correct = False
                last_err = reason

        obs = Observation(
            compiled=compiled, dtype=task.dtype,
            snr_by_shape=snr_by_shape,
            snr_db=min(snr_by_shape.values()) if snr_by_shape else None,
            validation_passed=correct, error_text=last_err if not correct else None,
        )
        if not (correct and do_bench):
            return obs

        wall_by_shape: dict[str, float] = {}
        base_by_shape: dict[str, float] = {}
        cvs: list[float] = []
        if self._batch_bench_ok(driver):
            # Fast + accurate: ALL shapes timed (both impls, n_max repeats) in ONE
            # process under a single per-GPU timing-lock hold. Contention-fair ratio +
            # minimal exclusive window -> honest speedups at high throughput.
            per_shape, poisoned = self._bench_all(
                driver, shapes, workdir, env, snr_threshold=thr)
            if poisoned:
                _ev("WARN", "eval_bench_hack", task=task.task_id, source_sha=_sha12(source),
                    reason="post-timing correctness failed (bench-time reward hack)")
                return Observation(compiled=False, dtype=task.dtype, validation_passed=False,
                                   flagged_hack=True, hack_reason="bench-time output mismatch",
                                   error_text="reward-hack: kernel incorrect under timing")
            for sh in shapes:
                cand_s, ref_s = per_shape.get(sh.name, ([], []))
                if cand_s:
                    wall_by_shape[sh.name] = _median(cand_s)
                    cvs.append(_cv_pct(cand_s))
                if ref_s:
                    base_by_shape[sh.name] = _median(ref_s)
        else:
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
                    cvs.append(cand_cv)
                if ref is not None:
                    base_by_shape[sh.name] = ref
        obs.wall_by_shape = wall_by_shape
        obs.baseline_by_shape = base_by_shape
        obs.cv_pct = max(cvs) if cvs else None
        if wall_by_shape:
            obs.wall_ms = max(wall_by_shape.values())
        if base_by_shape:
            obs.baseline_ms = max(base_by_shape.values())

        # P5 (flagship novelty): dense hardware-counter efficiency, baseline-relative.
        # Feature-flagged (profile_reward_weight>0) and fully fail-safe: any profiler
        # hiccup leaves profile_efficiency=None and never affects the correctness/
        # speedup verdict. Collected once on the primary shape only (rocprof is slow).
        if getattr(self.cfg, "profile_reward_weight", 0.0) > 0.0:
            try:
                obs.profile_efficiency = self._collect_profile(driver, shapes[0], workdir, env)
            except Exception as e:  # pragma: no cover - GPU/rocprof only
                _ev("DEBUG", "profile_error", task=task.task_id, error=str(e)[:200])
                obs.profile_efficiency = None
        return obs

    def collect_counters(self, source: str, shape: Optional["Shape"] = None) -> Optional[dict]:
        """PUBLIC: rocprofv3 PMC counters for a kernel (Pillar 4 grounded reasoning).

        Stages a private workdir (like ``evaluate``), profiles the CANDIDATE on one
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
        if len(source.encode("utf-8")) > self.isolation_policy.budget.max_source_bytes:
            self._last_execution_status = ExecutionStatus.POLICY_VIOLATION
            return None
        sh = shape or self.task.shape("primary") or self.task.shape("minimal") or (
            self.task.shapes[0] if self.task.shapes else Shape("default", {}))
        workdir = Path(tempfile.mkdtemp(prefix=f"pmc_{self.task.task_id}_"))
        previous_source, previous_task = self._active_source, self._active_task
        self._active_source, self._active_task = source, self.task
        try:
            task_sources = list(self.task.dir.glob("*.py"))
            if sum(p.stat().st_size for p in task_sources) > self.isolation_policy.budget.max_task_bytes:
                self._last_execution_status = ExecutionStatus.POLICY_VIOLATION
                return None
            for p in task_sources:
                dst = workdir / p.name
                shutil.copy(p, dst)
                os.chmod(dst, 0o444)
            (workdir / "kernel.py").write_text(source)
            os.chmod(workdir / "kernel.py", 0o444)
            driver = workdir / "driver.py"
            env = self._env(workdir / ".sandbox")
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
        exclusive lock on ``/tmp/kore_timing_gpu_<id>.lock`` around timing only, so
        speedups stay clean while compiles keep running in parallel. Disable with
        KORE_TIMING_LOCK=0."""
        if os.environ.get("KORE_TIMING_LOCK", "1").strip().lower() in ("0", "false", "no"):
            yield
            return
        physid = str(self._gpu if self._gpu is not None
                     else os.environ.get("HIP_VISIBLE_DEVICES", "0")).split(",")[0].strip() or "0"
        lp = Path(tempfile.gettempdir()) / f"kore_timing_gpu_{physid}.lock"
        f = open(lp, "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(f, fcntl.LOCK_UN)
            finally:
                f.close()

    def _batch_bench_ok(self, driver: Path) -> bool:
        """True iff this task uses the shared genops driver (supports ``--bench-both``).

        Detected once by scanning the driver for ``driver_main`` (all generated
        ``gen_*``/``genv_*`` tasks route through it). Bespoke drivers fall back to the
        proven per-impl path. Set ``KORE_NO_BENCH_BOTH=1`` to force the legacy path
        (used for the head-to-head timing-parity validation)."""
        if os.environ.get("KORE_NO_BENCH_BOTH", "").strip().lower() in ("1", "true", "yes"):
            return False
        v = getattr(self, "_batch_ok_cache", None)
        if v is None:
            try:
                v = "driver_main" in Path(driver).read_text()
            except Exception:  # noqa: BLE001
                v = False
            self._batch_ok_cache = v
        return v

    def _bench_pair(self, driver: Path, sh: Shape, workdir: Path, env: dict,
                    snr_threshold: Optional[float] = None):
        """Time candidate AND reference in ONE ``--bench-both`` process (``max_variance_runs``
        in-process repeats). Returns ``(cand_samples, ref_samples, poisoned)``.

        Because both impls are timed back-to-back in the same process, external GPU
        load hits them equally -> the speedup RATIO stays fair under oversubscription,
        while collapsing ~10 per-shape subprocess spawns (2 impls x ~5 runs) into one.
        ``poisoned`` mirrors the per-impl path: a False/low post-timing candidate
        verdict is a bench-time reward hack."""
        n_max = max(1, self.cfg.max_variance_runs)
        cmd = [sys.executable, str(driver), "--bench-both",
               "--warmup", str(self.cfg.warmup_iters), "--iters", str(self.cfg.bench_iters),
               "--repeat", str(n_max), *sh.as_args()]
        with self._timing_lock(), _LOG.timer("bench_pair", task=self.task.task_id, shape=sh.name):
            rc, out, timed = self._exec(cmd, workdir, env, self.bench_timeout)
        if timed or rc != 0:
            _ev("DEBUG", "bench_pair", task=self.task.task_id, shape=sh.name, ok=False, rc=rc)
            return [], [], False
        ac = _last(_ALLCLOSE, out)
        snr = _last(_SNR, out)
        threshold = self._snr_threshold if snr_threshold is None else snr_threshold
        if (ac and ac.group(1).lower() == "false") or \
           (snr and float(snr.group(1)) < threshold):
            return None, None, True
        cand = [float(m.group(1)) for m in _CAND_MED.finditer(out)]
        ref = [float(m.group(1)) for m in _REF_MED.finditer(out)]
        _ev("DEBUG", "bench_pair", task=self.task.task_id, shape=sh.name,
            cand_runs=len(cand), ref_runs=len(ref),
            cand_med=round(_median(cand), 4) if cand else None,
            ref_med=round(_median(ref), 4) if ref else None)
        return cand, ref, False

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
        Returns ``({shape_name: (cand_samples, ref_samples)}, poisoned)``."""
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
        ac = _last(_ALLCLOSE, out)
        snr = _last(_SNR, out)
        threshold = self._snr_threshold if snr_threshold is None else snr_threshold
        if (ac and ac.group(1).lower() == "false") or \
           (snr and float(snr.group(1)) < threshold):
            return {}, True
        blocks = out.split("SHAPE_BEGIN")[1:]  # per-shape, in the order we passed them
        result: dict[str, tuple] = {}
        for sh, block in zip(shapes, blocks):
            cand = [float(m.group(1)) for m in _CAND_MED.finditer(block)]
            ref = [float(m.group(1)) for m in _REF_MED.finditer(block)]
            result[sh.name] = (cand, ref)
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
