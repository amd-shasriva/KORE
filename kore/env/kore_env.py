"""KoreEnv: the verified evaluation environment.

Wraps the KernelForge verifier contract into a task-bound ``step(source)`` call
that returns a reward :class:`Observation`. Hardening (see audits):

* **No verdict forgery.** The candidate ``kernel.py`` is imported by the driver
  and could print fake ``SNR:``/``median_ms:`` lines. We parse the *last* match
  (the driver prints its verdict after calling the candidate) AND the anti-hack
  scanner rejects any candidate that prints a verdict literal.
* **Isolation.** Each eval runs in a throwaway workdir; the copied task sources
  (incl. reference.py oracle) are made read-only so a kernel can't corrupt them.
  The subprocess runs in its own session with a process limit; on timeout the
  whole process group is killed (no leaked grandchildren / GPU holders).
* **Infra vs kernel.** Timeouts, OOM-kills, segfaults, and missing-dependency
  imports are classified as ``infra_error`` — never cached, never fed to the
  policy as a kernel-correctness signal.
* **Trustworthy timing.** Each (shape, impl) is benched several times; the
  coefficient of variation is recorded and high-variance speedups are damped.
"""

from __future__ import annotations

import hashlib
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from kore.config import CONFIG
from kore.env.replay import ReplayCache
from kore.obs import get_logger
from kore.reward.reward import Observation, scan_for_hacks
from kore.reward.reward import _worst_speedup as _worst_speedup
from kore.reward.stats import cv_pct as _cv_pct
from kore.reward.stats import median as _median
from kore.tasks.base import Shape, Task

_LOG = get_logger("env")


def _ev(level: str, name: str, **fields) -> None:
    """Emit a structured event at an explicit level (JSONL always).

    ``KoreLogger.event`` hard-codes INFO; per-shape verifier detail must ride at
    DEBUG so it never spams INFO while a run is going, so we route through the
    logger's emit with the level we want but keep ``kind="event"`` for
    machine-readable JSONL. This is additive-only — pure observability.
    """
    _LOG._emit(level, name, fields, kind="event")


def _sha12(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8", "ignore")).hexdigest()[:12]

_SNR = re.compile(r"SNR:\s*([-\d.eE]+)")
_ALLCLOSE = re.compile(r"allclose:\s*(True|False)", re.IGNORECASE)
_MEDIAN = re.compile(r"median_ms:\s*([-\d.eE]+)")
# Candidate import/compile failure (the kernel's fault).
_COMPILE_ERR = re.compile(
    r"(SyntaxError|CompilationError|triton\..*Error|IndentationError|"
    r"NameError|out of resource|OutOfResources|AssertionError)",
    re.IGNORECASE,
)
# Infrastructure failure (NOT the kernel's fault) — never cache, never train on.
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


def _preexec():  # pragma: no cover - runs in child only
    # NB: session is created via Popen(start_new_session=True); do NOT setsid
    # again here (would EPERM). Cap process count to contain fork-bombs.
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (512, 512))
    except (ValueError, OSError):
        pass
    # Deliberately NOT setting RLIMIT_AS — ROCm/HIP reserve huge virtual address
    # space and an AS cap breaks legitimate GPU kernels.


class KoreEnv:
    """Task-bound verified environment. One per task; call ``step`` per candidate."""

    def __init__(self, task: Task, config=CONFIG, use_replay: bool = True,
                 correctness_timeout: int = 300, bench_timeout: int = 300):
        self.task = task
        self.cfg = config
        self.correctness_timeout = correctness_timeout
        self.bench_timeout = bench_timeout
        self.use_replay = use_replay
        self._cache_obj = ReplayCache(self.cfg.runs_dir / f"replay_{task.task_id}.jsonl") \
            if use_replay else None

    @property
    def _snr_threshold(self) -> float:
        t = getattr(self.task, "snr_threshold", None)
        return float(t) if t else self.cfg.snr_threshold_for(self.task.dtype)

    # ------------------------------------------------------------------ #
    def step(self, source: str, full_validation: bool = True,
             multi_shape: bool = True) -> Observation:
        return self.evaluate(self.task, source, shapes=self._shapes(multi_shape),
                             do_bench=full_validation)

    def _shapes(self, multi_shape: bool) -> list[Shape]:
        shapes = self.task.shapes or [Shape("default", {})]
        if multi_shape:
            return shapes
        primary = self.task.shape("primary") or self.task.shape("minimal") or shapes[0]
        return [primary]

    # ------------------------------------------------------------------ #
    def evaluate(self, task: Task, source: str, shapes: Optional[list[Shape]] = None,
                 do_bench: bool = True) -> Observation:
        source_sha = _sha12(source)
        n_shapes = len(shapes or task.shapes or [Shape("default", {})])
        _ev("INFO", "eval_start", task=task.task_id, n_shapes=n_shapes,
            source_sha=source_sha, do_bench=do_bench)

        hack = scan_for_hacks(source)
        if hack:
            _ev("WARN", "eval_hack", task=task.task_id, reason=hack, source_sha=source_sha)
            return Observation(compiled=False, dtype=task.dtype, flagged_hack=True,
                               hack_reason=hack, error_text=f"reward-hack: {hack}")

        if self.use_replay and self._cache_obj is not None:
            cached = self._cache_obj.get(task.task_id, source)
            if cached is not None:
                _LOG.debug("cache hit", task=task.task_id, source_sha=source_sha,
                           compiled=cached.compiled, correct=cached.validation_passed)
                self._log_eval_done(task, cached, cached=True)
                return cached

        shapes = shapes or task.shapes or [Shape("default", {})]
        workdir = Path(tempfile.mkdtemp(prefix=f"kore_{task.task_id}_"))
        try:
            obs = self._run(task, source, shapes, workdir, do_bench)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        # Only cache DETERMINISTIC terminal verdicts — never transient infra errors.
        cacheable = (obs.compiled or obs.error_text) and not obs.infra_error
        if self.use_replay and self._cache_obj is not None and cacheable:
            self._cache_obj.put(task.task_id, source, obs)
        self._log_eval_done(task, obs, cached=False)
        return obs

    def _log_eval_done(self, task: Task, obs: Observation, cached: bool) -> None:
        """Final per-candidate verdict at INFO (structured), covering every path."""
        _ev("INFO", "eval_done", task=task.task_id, compiled=obs.compiled,
            correct=obs.validation_passed, snr_min=obs.snr_db,
            worst_speedup=_worst_speedup(obs), cv_pct=obs.cv_pct,
            infra_error=obs.infra_error, cached=cached)

    # ------------------------------------------------------------------ #
    def _env(self) -> dict:
        env = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[2])  # /root/Kore-rl/kore
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        env["HIP_VISIBLE_DEVICES"] = env.get("HIP_VISIBLE_DEVICES", "0")
        env["GPU_TARGET"] = self.cfg.gpu_target
        env["HOME"] = str(Path(env.get("TMPDIR", "/tmp")))
        return env

    def _exec(self, cmd, workdir, env, timeout):
        """Run cmd in its own session; kill the whole group on timeout.
        Returns (returncode, combined_output, timed_out)."""
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

    def _classify(self, out: str, returncode: int, timed_out: bool):
        """-> ('ok'|'compile'|'infra', message)."""
        if timed_out:
            return "infra", "timeout"
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
        # stage isolated sources; make the oracle/driver READ-ONLY so a kernel
        # cannot corrupt reference.py for future evals.
        for p in task.dir.glob("*.py"):
            dst = workdir / p.name
            shutil.copy(p, dst)
            os.chmod(dst, 0o444)
        (workdir / "kernel.py").write_text(source)
        os.chmod(workdir / "kernel.py", 0o444)
        driver = workdir / "driver.py"
        env = self._env()

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

        thr = self._snr_threshold
        correct = validation_passed and bool(snr_by_shape) and all(v >= thr for v in snr_by_shape.values())

        # Anti-hack determinism re-check: re-run the primary shape once and require
        # a stable verdict, so a kernel cannot be rewarded for passing the SNR gate
        # by luck (partly-random output). One extra exec, only when already correct.
        if correct and getattr(self.cfg, "verifier_determinism_check", False):
            sh0 = shapes[0]
            rc2, out2, timed2 = self._exec([sys.executable, str(driver), *sh0.as_args()],
                                           workdir, env, self.correctness_timeout)
            kind2, _ = self._classify(out2, rc2, timed2)
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
        for sh in shapes:
            cand, cand_cv = self._bench_multi(driver, sh, "candidate", workdir, env)
            ref, _ = self._bench_multi(driver, sh, "reference", workdir, env)
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
            cmd = ["rocprofv3", "--pmc", *counters, "-d", outdir,
                   "--output-format", "csv", "--",
                   sys.executable, str(driver), "--impl", impl, *sh.as_args()]
            rc, out, timed = self._exec(cmd, workdir, env, self.bench_timeout)
            if timed or rc != 0:
                _ev("DEBUG", "profile_run", task=self.task.task_id, impl=impl,
                    ok=False, rc=rc)
                return None
            csvs = _glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True)
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

    def _bench_multi(self, driver: Path, sh: Shape, impl: str, workdir: Path, env: dict):
        """Bench a (shape, impl) ``min..max_variance_runs`` times; return
        (median-of-medians, CV%). Extra runs are taken only if variance is high."""
        cmd = [sys.executable, str(driver), "--bench-mode", "--impl", impl, *sh.as_args()]
        samples: list[float] = []
        n_min = max(1, self.cfg.min_variance_runs)
        n_max = max(n_min, self.cfg.max_variance_runs)
        for i in range(n_max):
            with _LOG.timer("bench_exec", task=self.task.task_id, shape=sh.name,
                            impl=impl, run=i):
                rc, out, timed = self._exec(cmd, workdir, env, self.bench_timeout)
            if timed or rc != 0:
                break
            m = _last(_MEDIAN, out)
            if m:
                samples.append(float(m.group(1)))
            if i + 1 >= n_min and len(samples) >= n_min and _cv_pct(samples) <= self.cfg.cv_threshold_pct:
                break
        if not samples:
            _ev("DEBUG", "bench_shape", task=self.task.task_id, shape=sh.name, impl=impl,
                median_ms=None, cv_pct=None, runs=0)
            return None, float("inf")
        med, cv = _median(samples), _cv_pct(samples)
        _ev("DEBUG", "bench_shape", task=self.task.task_id, shape=sh.name, impl=impl,
            median_ms=round(med, 4), cv_pct=round(cv, 3), runs=len(samples))
        return med, cv


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
