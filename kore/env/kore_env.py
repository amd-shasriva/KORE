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
  imports are classified as ``infra_error`` - never cached, never fed to the
  policy as a kernel-correctness signal.
* **Trustworthy timing.** Each (shape, impl) is benched several times; the
  coefficient of variation is recorded and high-variance speedups are damped.
"""

from __future__ import annotations

import fcntl
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
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from kore.config import CONFIG
from kore.env.replay import ReplayCache
from kore.obs import get_logger
from kore.reward.reward import Observation, scan_for_hacks
from kore.reward.reward import _worst_speedup
from kore.reward.stats import cv_pct as _cv_pct
from kore.reward.stats import median as _median
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
                 gpu: Optional[str] = None):
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

        # Only cache DETERMINISTIC terminal verdicts - never transient infra errors.
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
        if self._gpu is not None:
            # ABSOLUTE physical GPU id for the compile/bench subprocess. Set BOTH
            # HIP_ and CUDA_VISIBLE_DEVICES to it (and drop any inherited list) so the
            # subprocess sees exactly this one physical GPU as its device 0 - no
            # double-remap from a restricted parent visible-device list.
            # str(): subprocess env values MUST be strings - an int gpu id (e.g.
            # KoreEnv(gpu=5)) would make subprocess.Popen raise inside os.fsencode.
            env["HIP_VISIBLE_DEVICES"] = str(self._gpu)
            env["CUDA_VISIBLE_DEVICES"] = str(self._gpu)
        else:
            env["HIP_VISIBLE_DEVICES"] = env.get("HIP_VISIBLE_DEVICES", "0")
        # Prefer the TASK's declared arch over the global default so the driver
        # subprocess compiles/benches + selects the fp8 encoding for the arch the
        # task actually targets (a gfx950 task must not be built as gfx942/FNUZ).
        env["GPU_TARGET"] = getattr(self.task, "gpu_target", None) or self.cfg.gpu_target
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
            per_shape, poisoned = self._bench_all(driver, shapes, workdir, env)
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
                cand, cand_cv, poisoned = self._bench_multi(driver, sh, "candidate", workdir, env)
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
                ref = self._bench_multi(driver, sh, "reference", workdir, env)[0]
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
                passes = counter_passes("grounding")   # real gfx942 BW/L2/occupancy set
            except Exception:  # noqa: BLE001 - older pmc: single-pass fallback
                passes = [COUNTER_SETS["full"]]
        except Exception:  # noqa: BLE001
            return None
        sh = shape or self.task.shape("primary") or self.task.shape("minimal") or (
            self.task.shapes[0] if self.task.shapes else Shape("default", {}))
        workdir = Path(tempfile.mkdtemp(prefix=f"pmc_{self.task.task_id}_"))
        try:
            for p in self.task.dir.glob("*.py"):
                dst = workdir / p.name
                shutil.copy(p, dst)
                os.chmod(dst, 0o444)
            (workdir / "kernel.py").write_text(source)
            os.chmod(workdir / "kernel.py", 0o444)
            driver = workdir / "driver.py"
            env = self._env()
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

    def _bench_pair(self, driver: Path, sh: Shape, workdir: Path, env: dict):
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
        if (ac and ac.group(1).lower() == "false") or \
           (snr and float(snr.group(1)) < self._snr_threshold):
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

    def _bench_all(self, driver: Path, shapes, workdir: Path, env: dict):
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
        if (ac and ac.group(1).lower() == "false") or \
           (snr and float(snr.group(1)) < self._snr_threshold):
            return {}, True
        blocks = out.split("SHAPE_BEGIN")[1:]  # per-shape, in the order we passed them
        result: dict[str, tuple] = {}
        for sh, block in zip(shapes, blocks):
            cand = [float(m.group(1)) for m in _CAND_MED.finditer(block)]
            ref = [float(m.group(1)) for m in _REF_MED.finditer(block)]
            result[sh.name] = (cand, ref)
        return result, False

    def _bench_multi(self, driver: Path, sh: Shape, impl: str, workdir: Path, env: dict):
        """Bench a (shape, impl) ``min..max_variance_runs`` times; return
        (median-of-medians, CV%, poisoned).

        ``poisoned`` (candidate only) is True when the driver's POST-TIMING
        correctness re-verification failed - i.e. the kernel produced correct output
        for the correctness calls but garbage while being timed (the invocation-count
        timing hack). The timed window (warmup/iters) is RANDOMIZED per run so a
        stateful kernel cannot know which call indices are timed vs verified.
        """
        import random as _random
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
                   (snr and float(snr.group(1)) < self._snr_threshold):
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
