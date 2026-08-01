"""Repository-wide pytest classification and the real-hardware GPU harness.

Tests are CPU-safe unless they explicitly opt into a resource marker. Keeping
this hook at the repository root applies the same rule to top-level tests and
to tests colocated under ``kore/``.

Two other things live here because they are shared across test modules:

* the **marker census** (:func:`marker_census`) - every marker carried by the
  collected items, counted BEFORE pytest applies the ``-m`` expression from
  ``addopts``. ``tests/test_marker_contract.py`` uses it to prove that each
  marker declared in ``pyproject.toml`` actually selects something, so a
  decorative marker cannot silently return.
* the **GPU harness** (:class:`GpuHarness`) - a staged task workdir plus the
  verifier's own ``KoreEnv._env``/``KoreEnv._exec`` subprocess boundary, so the
  ``gpu``-marked suites in ``tests/test_gpu_verifier.py`` and
  ``tests/test_gpu_timing_protocol.py`` exercise the production execution path
  rather than a mock of it. Nothing here imports torch or ``kore`` at collection
  time; the CPU suite pays no cost for it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import pytest

_RESOURCE_MARKERS = ("gpu", "release")

#: Marker name -> number of collected items carrying it, before ``-m`` deselection.
_MARKER_CENSUS: dict[str, int] = {}
#: Marker name -> repo-relative test modules that carry it.
_MARKER_MODULES: dict[str, set[str]] = {}
#: Marker names that appear together on a single item, as frozen pairs.
_MARKER_COMBINATIONS: set[frozenset] = set()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    """Give every non-optional test an explicit ``cpu`` group, then census markers.

    ``tryfirst`` matters: pytest's own ``-m`` handling deselects inside this same
    hook, so the census has to run before it to see the opt-in groups at all.
    """
    _MARKER_CENSUS.clear()
    _MARKER_MODULES.clear()
    _MARKER_COMBINATIONS.clear()
    root = Path(str(config.rootpath))
    for item in items:
        if not any(item.get_closest_marker(name) for name in _RESOURCE_MARKERS):
            item.add_marker(pytest.mark.cpu)
        try:
            module = Path(str(item.path)).relative_to(root).as_posix()
        except ValueError:
            module = Path(str(item.path)).name
        names = {marker.name for marker in item.iter_markers()}
        _MARKER_COMBINATIONS.add(frozenset(names))
        for name in names:
            _MARKER_CENSUS[name] = _MARKER_CENSUS.get(name, 0) + 1
            _MARKER_MODULES.setdefault(name, set()).add(module)


@pytest.fixture(scope="session")
def marker_census() -> dict[str, int]:
    """Markers carried by the collected items, before ``-m`` deselection."""
    return dict(_MARKER_CENSUS)


@pytest.fixture(scope="session")
def marker_modules() -> dict[str, set[str]]:
    """Marker name -> the repo-relative test modules that carry it."""
    return {name: set(paths) for name, paths in _MARKER_MODULES.items()}


@pytest.fixture(scope="session")
def marker_combinations() -> set[frozenset]:
    """Every distinct set of markers observed on a single collected item."""
    return {frozenset(names) for names in _MARKER_COMBINATIONS}


@pytest.fixture(autouse=True)
def _isolate_rigor_env():
    """Snapshot/restore RIGOR_ENV around every test.

    ``scripts.run_campaign`` calls ``set_rigorous_verification(True)`` during the
    datagen stage, which writes the KORE_* rigor vars into ``os.environ`` on
    purpose (production datagen subprocesses inherit them). Without isolation a
    campaign test leaks those vars into later tests and poisons the versioned
    generator-contract digest (``resolved_config_identity`` hashes RIGOR_ENV),
    e.g. breaking ``tests/test_parallel_datagen.py`` in a full-suite run.
    """
    import os
    try:
        from kore.data.verify_rigor import RIGOR_ENV
        keys = tuple(RIGOR_ENV)
    except Exception:  # noqa: BLE001 - never let isolation break collection
        keys = (
            "KORE_VERIFIED_CORRECTNESS",
            "KORE_COMPILE_BASELINE",
            "KORE_BENCH_COLD",
            "KORE_SHAPE_AUGMENT",
        )
    saved = {k: os.environ.get(k) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------------- #
# GPU harness (only ever constructed by ``gpu``-marked tests)
# --------------------------------------------------------------------------- #
#: Physical GPU the ``gpu`` suite is allowed to touch. A shared box hands the
#: other cards to other jobs, so the id is a single deliberate choice rather than
#: "whatever torch enumerates first"; override with ``KORE_TEST_GPU``.
DEFAULT_TEST_GPU = "6"
#: A second card on the same node, used only to prove the visible-device mapping
#: selects the card it was asked for. Override with ``KORE_TEST_GPU_ALT``.
DEFAULT_TEST_GPU_ALT = "7"

#: Task the GPU suite drives end to end. Deliberately a generated elementwise op:
#: its driver is the shared ``kore.tasks._genops`` publication driver (so the
#: paired-v2 protocol is under test), and it compiles in seconds.
GPU_TASK_ID = "gen_add_fp32"

_DEVICE_PROBE = """
import json, os, torch
devices = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    devices.append({
        "arch": str(p.gcnArchName),
        "uuid": str(getattr(p, "uuid", "")),
        "total_mb": int(p.total_memory) // (1024 * 1024),
    })
print("KORE_GPU_PROBE: " + json.dumps({
    "torch_version": torch.__version__,
    "hip_version": torch.version.hip,
    "device_count": torch.cuda.device_count(),
    "visibility": {
        name: os.environ.get(name)
        for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                     "ROCR_VISIBLE_DEVICES")
    },
    "gpu_target": os.environ.get("GPU_TARGET"),
    "devices": devices,
}))
"""

_AITER_PROBE = r"""
import io, json, sys, traceback

import torch

from kore.tasks import aiter_ref

report = {}
try:
    import aiter  # noqa: F401
    report["import_error"] = None
except Exception as exc:  # noqa: BLE001
    report["import_error"] = f"{type(exc).__name__}: {exc}"

x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
gated = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)


def _reference_rms_norm():
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return (y * weight.float()).to(x.dtype)


def _reference_silu_and_mul():
    half = gated.shape[-1] // 2
    a, b = gated[..., :half].float(), gated[..., half:].float()
    return ((a * torch.sigmoid(a)) * b).to(gated.dtype)


CASES = {
    "rms_norm": (
        "aiter_rms_norm",
        lambda: aiter_ref.aiter_rms_norm(x, weight, 1e-6),
        _reference_rms_norm,
    ),
    "silu_and_mul": (
        "aiter_silu_and_mul",
        lambda: aiter_ref.aiter_silu_and_mul(gated),
        _reference_silu_and_mul,
    ),
}

cases = {}
for op, (wrapper_name, call, oracle) in CASES.items():
    entry = {"wrapper": wrapper_name}
    try:
        aiter_ref._aiter_fn(op)
        entry["resolved"] = True
        entry["resolve_error"] = None
    except Exception as exc:  # noqa: BLE001
        entry["resolved"] = False
        entry["resolve_error"] = f"{type(exc).__name__}: {exc}"
    aiter_ref._MARKED_BASELINE.clear()
    captured, saved = io.StringIO(), sys.stderr
    sys.stderr = captured
    try:
        out = call()
    except Exception:  # noqa: BLE001
        out = None
        entry["call_error"] = traceback.format_exc()[-400:]
    else:
        entry["call_error"] = None
    finally:
        sys.stderr = saved
    entry["sentinels"] = [
        line.split(":", 1)[1].strip()
        for line in captured.getvalue().splitlines()
        if line.startswith("KORE_BASELINE_IMPL:")
    ]
    if out is None:
        entry["matches_oracle"] = None
    else:
        want = oracle()
        entry["matches_oracle"] = bool(
            torch.allclose(out.float(), want.float(), atol=1e-2, rtol=1e-2))
    cases[op] = entry

report["cases"] = cases
print("KORE_AITER_PROBE: " + json.dumps(report))
"""


def _probe_payload(prefix: str, text: str) -> Optional[dict]:
    for line in (text or "").splitlines():
        if line.startswith(prefix):
            try:
                return json.loads(line[len(prefix):])
            except ValueError:
                return None
    return None


class GpuHarness:
    """Drive the real verifier subprocess boundary for one task on one GPU.

    Everything routes through :class:`kore.env.kore_env.KoreEnv`'s own ``_env``
    and ``_exec`` so a GPU test measures the code the trainer runs. The staged
    workdirs are throwaway copies of the task sources (the same layout
    ``KoreEnv._run`` builds) so a driver can be invoked directly when a test
    needs to inspect raw protocol output.
    """

    #: Sentinel emitted once per bench process by ``kore.tasks.aiter_ref``.
    BASELINE_SENTINEL = "KORE_BASELINE_IMPL:"

    def __init__(self, gpu: str, alt_gpu: str) -> None:
        from kore.tasks.registry import get_task

        self.gpu = gpu
        self.alt_gpu = alt_gpu
        self.task = get_task(GPU_TASK_ID)
        #: Filled in by the ``gpu_harness`` fixture from the availability probe.
        self.probe: Optional[dict] = None
        self._workdirs: list[Path] = []
        self._device_probes: dict[str, Optional[dict]] = {}

    # -- construction ------------------------------------------------------ #
    def kore_env(self, gpu: Optional[str] = None, task: Any = None):
        """A replay-free :class:`KoreEnv` pinned to a physical GPU."""
        from kore.env.kore_env import KoreEnv

        return KoreEnv(task or self.task, use_replay=False, gpu=gpu or self.gpu)

    def task_by_id(self, task_id: str):
        from kore.tasks.registry import get_task

        return get_task(task_id)

    def shape(self, name: str):
        shape = self.task.shape(name)
        assert shape is not None, f"{self.task.task_id} has no shape {name!r}"
        return shape

    # -- staging + execution ----------------------------------------------- #
    def stage(self, source: str, task: Any = None) -> Path:
        """Copy a task's sources into a throwaway workdir carrying ``source``."""
        task = task or self.task
        workdir = Path(tempfile.mkdtemp(prefix=f"koregpu_{task.task_id}_"))
        for path in task.dir.glob("*.py"):
            shutil.copy(path, workdir / path.name)
        (workdir / "kernel.py").write_text(source)
        self._workdirs.append(workdir)
        return workdir

    def child_env(self, gpu: Optional[str] = None, task: Any = None,
                  **overrides: str) -> dict:
        """The exact environment ``KoreEnv`` hands a candidate subprocess."""
        task = task or self.task
        env = self.kore_env(gpu=gpu, task=task)._env(task)
        env.update({k: str(v) for k, v in overrides.items()})
        return env

    def exec(self, argv: list[str], workdir: Path, *, timeout: int = 600,
             env: Optional[dict] = None, gpu: Optional[str] = None,
             task: Any = None, **overrides: str):
        """Run ``argv`` through ``KoreEnv._exec``; returns ``(rc, output, timed_out)``."""
        task = task or self.task
        kore_env = self.kore_env(gpu=gpu, task=task)
        child = env if env is not None else self.child_env(
            gpu=gpu, task=task, **overrides)
        return kore_env._exec(argv, workdir, child, timeout)

    def run_driver(self, workdir: Path, *args: str, timeout: int = 600,
                   env: Optional[dict] = None, gpu: Optional[str] = None,
                   task: Any = None, **overrides: str):
        """Invoke the staged task driver with ``args``."""
        argv = [sys.executable, str(workdir / "driver.py"), *args]
        return self.exec(argv, workdir, timeout=timeout, env=env, gpu=gpu,
                         task=task, **overrides)

    def run_python(self, script: str, *, gpu: Optional[str] = None,
                   timeout: int = 600, env: Optional[dict] = None,
                   **overrides: str):
        """Run a probe script under the verifier's own child environment."""
        workdir = Path(tempfile.mkdtemp(prefix="koregpu_probe_"))
        self._workdirs.append(workdir)
        (workdir / "probe.py").write_text(script)
        argv = [sys.executable, str(workdir / "probe.py")]
        return self.exec(argv, workdir, timeout=timeout, env=env, gpu=gpu,
                         **overrides)

    def evaluate(self, source: str, shapes, *, do_bench: bool = True,
                 gpu: Optional[str] = None, task: Any = None):
        """Run one candidate through the full ``KoreEnv.evaluate`` path."""
        task = task or self.task
        return self.kore_env(gpu=gpu, task=task).evaluate(
            task, source, shapes=list(shapes), do_bench=do_bench)

    # -- probes ------------------------------------------------------------ #
    def device_probe(self, gpu: Optional[str] = None,
                     env: Optional[dict] = None) -> Optional[dict]:
        """What torch sees inside a child pinned to ``gpu``.

        Memoized per visibility mask: every probe is a fresh torch import, and
        several tests ask about the same mask.
        """
        key = gpu or self.gpu
        if env is None and key in self._device_probes:
            return self._device_probes[key]
        _rc, out, _timed = self.run_python(_DEVICE_PROBE, gpu=gpu, env=env,
                                           timeout=300)
        payload = _probe_payload("KORE_GPU_PROBE: ", out)
        if env is None:
            self._device_probes[key] = payload
        return payload

    def aiter_probe(self) -> tuple[Optional[dict], str]:
        """AITER availability + which baseline sentinel each wrapper emits."""
        _rc, out, _timed = self.run_python(_AITER_PROBE, timeout=900)
        return _probe_payload("KORE_AITER_PROBE: ", out), out

    def cleanup(self) -> None:
        while self._workdirs:
            shutil.rmtree(self._workdirs.pop(), ignore_errors=True)


@pytest.fixture(scope="session")
def gpu_id() -> str:
    """Physical GPU the ``gpu`` suite runs on (``KORE_TEST_GPU``)."""
    return (os.environ.get("KORE_TEST_GPU") or DEFAULT_TEST_GPU).strip()


@pytest.fixture(scope="session")
def alt_gpu_id() -> str:
    """Second physical GPU used only for visible-device mapping checks."""
    return (os.environ.get("KORE_TEST_GPU_ALT") or DEFAULT_TEST_GPU_ALT).strip()


@pytest.fixture(scope="session")
def gpu_harness(gpu_id: str, alt_gpu_id: str):
    """Session GPU harness; skips the whole ``gpu`` suite when no GPU answers.

    The availability probe runs in a CHILD process on purpose: the parent pytest
    process must never initialize HIP, both so a CPU-only collection stays clean
    and so the suite cannot accidentally hold a context on a card it was not
    given.
    """
    pytest.importorskip("torch", reason="the gpu suite needs torch + ROCm")
    try:
        harness = GpuHarness(gpu_id, alt_gpu_id)
    except Exception as exc:  # noqa: BLE001 - unbuildable task => nothing to test
        pytest.skip(f"cannot load task {GPU_TASK_ID!r} for the gpu suite: {exc}")
    probe = harness.device_probe()
    if not probe or int(probe.get("device_count") or 0) < 1:
        harness.cleanup()
        pytest.skip(
            f"no GPU visible to a child pinned to HIP_VISIBLE_DEVICES={gpu_id}; "
            "run the gpu suite on a ROCm node (see tests/README.md)")
    harness.probe = probe
    try:
        yield harness
    finally:
        harness.cleanup()


@pytest.fixture(scope="session")
def seed_observation(gpu_harness: GpuHarness):
    """One real ``evaluate`` of the task's seed kernel, shared by several tests.

    Session-scoped because it costs a compile plus the whole paired timing
    protocol; every assertion about the happy path reads this one measurement so
    the suite stays cheap on a shared node.
    """
    shapes = [gpu_harness.shape("minimal")]
    return gpu_harness.evaluate(gpu_harness.task.seed_source, shapes)
