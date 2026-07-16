"""Shared verifier contract for the DRAFT training-side (BACKWARD) tasks.

STAGING NOTE (safety): this module and every task under
``kore/tasks/_drafts/training/`` are STAGED, not live. The registry discovers
tasks via ``kore/tasks/*/task.yaml`` (ONE directory level, see
``registry._discover``), so a task nested at
``kore/tasks/_drafts/training/<id>/task.yaml`` is THREE levels deep and is NOT
auto-discovered -- no run/campaign can pick these up until a human promotes them.
Confirmed: ``registry.task_ids()`` contains none of the drafted ids.

Promotion (per task, after on-gfx950 verification): move
``kore/tasks/_drafts/training/<id>/`` to ``kore/tasks/<id>/`` AND copy this file
to ``kore/tasks/_training_common.py`` (each task's thin ``driver.py`` imports
``_training_common`` from its parent directory). See VERIFICATION_CHECKLIST.md.

Why a shared driver: a BACKWARD task returns SEVERAL gradient tensors (dQ/dK/dV,
dX/dgamma/dbeta, dX, or dgrad/wgrad), so the correctness gate must score EACH
gradient against the fp32 autograd oracle and report the WORST. That multi-output
scoring + the cold-cache bench + the post-timing anti-hack re-verification live
here once, instead of being duplicated in four per-task drivers.

reference.py contract (each drafted BACKWARD task implements):
    parse_shape(s) -> dict
    get_inputs(shape, device="cuda", seed=0, dtype=torch.bfloat16) -> tuple
        # forward inputs + any saved forward activations + the upstream grad,
        # in the POSITIONAL order the candidate entry expects.
    reference_grads(shape, inputs) -> tuple[Tensor, ...]
        # THE ORACLE: ground-truth gradients from torch AUTOGRAD on the fp32
        # forward, in GRAD_NAMES order. This is the correctness ground truth.
    candidate_grads(fn, shape, inputs) -> tuple[Tensor, ...]   # invoke candidate
    baseline_grads(shape, inputs) -> tuple[Tensor, ...]
        # perf-only bar: the framework fused autograd backward (NO AITER backward
        # kernel exists for these ops -- see VERIFICATION_CHECKLIST.md).
    ENTRY: str                      # candidate entry attr name
    GRAD_NAMES: tuple[str, ...]     # names of the returned gradients (for reports)
    TOL: dict[str, tuple[float,float]]   # per-grad (atol, rtol); DEFAULT_TOL if absent

torch is imported lazily inside the GPU paths so importing a reference (for the
CPU oracle / finite-difference sanity check) never needs a GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os

DEFAULT_TOL = (2e-2, 2e-2)


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def _snr_db(out, ref_out) -> float:
    o, r = out.float(), ref_out.float()
    noise = (o - r).norm().item()
    signal = r.norm().item()
    if noise == 0:
        return 999.0
    return 20.0 * math.log10(signal / noise) if signal > 0 else -999.0


def _tol_for(ref, name: str) -> tuple:
    tol = getattr(ref, "TOL", {}) or {}
    return tuple(tol.get(name, DEFAULT_TOL))


def _num_correct_trials() -> int:
    """KernelBench-fidelity: >=5 reseeded correctness trials (env-overridable)."""
    try:
        return max(5, int(os.environ.get("KORE_CORRECTNESS_TRIALS", "5")))
    except ValueError:
        return 5


def _bench_cold() -> bool:
    return os.environ.get("KORE_BENCH_COLD", "1") != "0"


_L2_SCRATCH = None


def _flush_l2(device: str = "cuda") -> None:
    """Evict the GPU last-level cache between timed iters (cold-cache, KernelBench)."""
    import torch

    global _L2_SCRATCH
    if _L2_SCRATCH is None:
        _L2_SCRATCH = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    _L2_SCRATCH.zero_()


def _time_fn(fn, warmup: int, iters: int) -> int:
    import torch

    cold = _bench_cold()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if cold:
            _flush_l2()
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(st, en))
    for t in times:
        print(f"wall_ms: {t:.4f}")
    print(f"median_ms: {times[len(times) // 2]:.4f}")
    return 0


def _as_tuple(x) -> tuple:
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _load_candidate(task_dir: str, entry: str):
    # Cache the module so a stateful kernel's globals persist from the bench timing
    # loop into the post-timing re-verification (anti invocation-count timing hack).
    if getattr(_load_candidate, "_mod", None) is None:
        path = os.path.join(task_dir, "kernel.py")
        spec = importlib.util.spec_from_file_location("candidate_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _load_candidate._mod = mod
    return getattr(_load_candidate._mod, entry)


# --------------------------------------------------------------------------- #
# Correctness (worst-of-all-gradients SNR) + bench
# --------------------------------------------------------------------------- #
def _run_correctness(ref, task_dir, shape) -> int:
    import torch

    fn = _load_candidate(task_dir, ref.ENTRY)
    names = tuple(getattr(ref, "GRAD_NAMES", ()))
    worst, maxd, ok = 999.0, 0.0, True
    for s in range(_num_correct_trials()):
        inputs = ref.get_inputs(shape, device="cuda", seed=s)
        refs = _as_tuple(ref.reference_grads(shape, inputs))
        try:
            outs = _as_tuple(ref.candidate_grads(fn, shape, inputs))
        except Exception as e:  # noqa: BLE001
            print("SNR: -999.00 dB"); print("allclose: False"); print("max_diff: inf")
            print(f"CANDIDATE_ERROR: {type(e).__name__}: {e}")
            return 0
        torch.cuda.synchronize()
        if len(outs) != len(refs):
            print("SNR: -999.00 dB"); print("allclose: False"); print("max_diff: inf")
            print(f"CANDIDATE_ERROR: returned {len(outs)} grads, expected {len(refs)}")
            return 0
        for i, (o, r) in enumerate(zip(outs, refs)):
            nm = names[i] if i < len(names) else f"grad{i}"
            atol, rtol = _tol_for(ref, nm)
            worst = min(worst, _snr_db(o, r))
            maxd = max(maxd, (o.float() - r.float()).abs().max().item())
            ok = ok and torch.allclose(o.float(), r.float(), atol=atol, rtol=rtol)
    print(f"SNR: {worst:.2f} dB"); print(f"allclose: {ok}"); print(f"max_diff: {maxd:.6f}")
    return 0


def _run_bench(ref, task_dir, shape, impl, warmup, iters) -> int:
    inputs = ref.get_inputs(shape, device="cuda", seed=0)
    if impl == "reference":
        fn = lambda: ref.baseline_grads(shape, inputs)     # perf-only framework backward
    elif impl == "torch":
        fn = lambda: ref.reference_grads(shape, inputs)     # fp32 autograd oracle
    else:
        cand = _load_candidate(task_dir, ref.ENTRY)
        fn = lambda: ref.candidate_grads(cand, shape, inputs)
    return _time_fn(fn, warmup, iters)


def driver_main(ref, task_dir: str, argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="default")
    p.add_argument("--mode", default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--bench-mode", action="store_true")
    p.add_argument("--impl", default="candidate", choices=["candidate", "reference", "torch"])
    a = p.parse_args(argv)
    shape = ref.parse_shape(a.shape)
    if a.bench_mode:
        rc = _run_bench(ref, task_dir, shape, a.impl, a.warmup, a.iters)
        # Post-timing anti-hack correctness re-verification on the cached candidate.
        if a.impl == "candidate":
            _run_correctness(ref, task_dir, shape)
        return rc
    return _run_correctness(ref, task_dir, shape)
