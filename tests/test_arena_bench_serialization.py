"""Timing must not overlap, even when everything else does.

Several tasks in flight is what makes vLLM batch, and generation, compilation and
correctness overlap harmlessly. Timing does not: a kernel benchmarked while other
kernels share the GPU reads slower than it is, so every speedup is biased downward
-- and the numbers still look plausible, which is what makes it dangerous.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kore.eval.agent_kernel_arena as aka  # noqa: E402


def test_a_benchmark_lock_exists_and_is_a_lock():
    assert hasattr(aka, "_BENCH_LOCK")
    assert isinstance(aka._BENCH_LOCK, type(threading.Lock()))


def test_the_lock_actually_excludes_concurrent_holders():
    """Pin the property rather than the identity: if the lock is ever replaced by
    something non-exclusive, speedups silently start reading low again."""
    overlap = []
    inside = []

    def worker():
        with aka._BENCH_LOCK:
            inside.append(1)
            overlap.append(len(inside))
            time.sleep(0.02)
            inside.pop()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlap == [1, 1, 1, 1], f"benchmarks overlapped: {overlap}"


def test_the_performance_command_is_run_under_the_lock():
    """Read the source: the guard has to wrap the timing subprocess, not merely
    exist somewhere in the module."""
    src = Path(aka.__file__).read_text()
    # Slice the whole performance block rather than a fixed character window. The
    # window was 900 chars and broke the moment a comment was added above the
    # lock, which is a property of the comment and not of the guard -- exactly the
    # kind of false failure that gets a real test deleted.
    body = src.split("if task.performance_command:", 1)[1]
    body = body.split("except Exception", 1)[0]
    assert "with _BENCH_LOCK:" in body, "the timing block does not take the lock"
    lock_at = body.index("with _BENCH_LOCK:")
    run_at = body.index("subprocess.run")
    assert lock_at < run_at, "the lock must be taken before the benchmark runs"
