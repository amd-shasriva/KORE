"""Kernel/GPU/ROCm QA + explanation SFT generation (Stage-1 ``kernel_qa``).

The kernel-QA slice teaches the model to *reason about* AMD GPU kernels in
natural language — occupancy, memory hierarchy, rocprof counters, numerics, and
HIP/Triton bug review — not just to emit ``FULL_KERNEL`` blocks. These rows are
grounded in the real task seeds/dtypes from ``kore.tasks`` and are produced by a
:class:`~kore.data.teacher.TeacherClient` (a ``StubTeacher`` in tests).

``generate_kernel_qa(tasks, teacher, n, seed)`` -> chat rows
``[{"messages": [...], "_source": "kernel_qa", "_style": ..., "_qa_type": ...}]``.

Both a THINKING exemplar style (answer reasons inside ``<think>...</think>``) and
a NO-THINK style (direct answer) are emitted so SFT sees both regimes, matching
the reasoning-base + on/off-thinking policy.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from typing import Any, Callable, Iterable, Optional

from kore.data.prompts import SYSTEM_PROMPT
from kore.data.teacher import TeacherClient
from kore.obs import get_logger

log = get_logger("data.gen_qa")

QA_SOURCE_TAG = "kernel_qa"

# The QA prompt families (see builders below).
KERNEL_QA_TYPES = (
    "occupancy_bound",
    "rocprof_counter",
    "hip_diff_bug",
    "fp32_accumulator",
    "wavefront_block",
    "mfma_tl_dot",
)

# rocprof / rocprofv3 hardware counters with a one-line meaning, used to ground
# the "what does this counter mean" QA type.
ROCPROF_COUNTERS: dict[str, str] = {
    "VALUUtilization": "fraction of active vector-ALU lanes per issued VALU instruction "
                       "(low => thread divergence or poor lane packing)",
    "LDSBankConflict": "cycles lost to LDS (shared memory) bank conflicts",
    "VALUBusy": "fraction of cycles the vector ALU was issuing instructions",
    "MemUnitBusy": "fraction of cycles the memory unit was busy (HBM/L2 pressure)",
    "FetchSize": "total bytes fetched from HBM into the cache hierarchy",
    "L2CacheHit": "fraction of memory requests served from L2",
    "GPUBusy": "fraction of the elapsed time the GPU was executing the kernel",
    "SALUBusy": "fraction of cycles the scalar ALU was busy (address/branch math)",
}

# A small, deliberately buggy HIP diff to ground the "review this diff" QA type.
_HIP_DIFF = """\
@@ kernel.hip
-  int idx = blockIdx.x * blockDim.x + threadIdx.x;
+  int idx = blockIdx.x * blockDim.x;                 // dropped threadIdx.x
   float acc = 0.0f;
   for (int k = 0; k < K; ++k)
     acc += a[idx * K + k] * b[k];
-  __syncthreads();
   out[idx] = acc;
"""


# --------------------------------------------------------------------------- #
# Task views (accept real Task objects OR lightweight dicts)
# --------------------------------------------------------------------------- #
def _get(task: Any, name: str, default: Any = None) -> Any:
    if isinstance(task, dict):
        return task.get(name, default)
    return getattr(task, name, default)


def _seed_snippet(task: Any, max_lines: int = 40) -> str:
    """Best-effort kernel-source snippet for a task, tolerating missing files."""
    src = _get(task, "seed_source")
    if src is None:
        try:  # real Task exposes ``seed_source`` as a property that reads a file
            src = task.seed_source  # type: ignore[attr-defined]
        except Exception:
            src = None
    if not isinstance(src, str) or not src.strip():
        # Fallback grounding snippet so offline/dict tasks still produce real rows.
        op = _get(task, "operation") or _get(task, "task_id") or "kernel"
        return (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            f"def _{op}_kernel(x_ptr, y_ptr, N, BLOCK_N: tl.constexpr):\n"
            "    offs = tl.arange(0, BLOCK_N)\n"
            "    mask = offs < N\n"
            "    x = tl.load(x_ptr + offs, mask=mask, other=0.0)\n"
            "    tl.store(y_ptr + offs, x, mask=mask)\n"
        )
    lines = src.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["    # ... (truncated) ..."]
    return "\n".join(lines)


def _view(task: Any) -> dict:
    gpu = _get(task, "gpu_target") or _get(task, "gpu", "gfx950")
    uarch = "CDNA4" if gpu == "gfx950" else ("CDNA3" if gpu in ("gfx942", "gfx90a") else "CDNA")
    return {
        "task_id": _get(task, "task_id", "unknown_task"),
        "operation": _get(task, "operation") or _get(task, "task_id", "kernel"),
        "dtype": _get(task, "dtype", "bf16"),
        "gpu": gpu,
        "uarch": uarch,   # microarch string derived from the arch (gfx950 -> CDNA4)
        "snippet": _seed_snippet(task),
    }


# --------------------------------------------------------------------------- #
# Prompt builders: (view, counter?) -> user question string
# --------------------------------------------------------------------------- #
def _q_occupancy_bound(v: dict, rng: random.Random) -> str:
    return (
        f"Below is the seed Triton kernel for task '{v['task_id']}' "
        f"({v['operation']}, {v['dtype']}) targeting {v['gpu']} ({v['uarch']}, wave64).\n\n"
        f"```python\n{v['snippet']}\n```\n\n"
        f"Explain why this kernel is likely occupancy-bound on {v['gpu']}, referencing "
        "VGPR/LDS pressure and the wave64 wavefront, and name the single change you "
        "would try first to raise occupancy."
    )


def _q_rocprof_counter(v: dict, rng: random.Random) -> str:
    counter = rng.choice(sorted(ROCPROF_COUNTERS))
    return (
        f"While profiling the '{v['task_id']}' Triton kernel on {v['gpu']} with "
        f"rocprofv3, the counter `{counter}` stands out. What does `{counter}` "
        "measure, what value would concern you, and what kernel change would you "
        "make in response?"
    )


def _q_hip_diff_bug(v: dict, rng: random.Random) -> str:
    return (
        f"Reviewing a HIP change to the '{v['operation']}' kernel on {v['gpu']}. "
        "Find the bug in this diff and explain the correct fix:\n\n"
        f"```diff\n{_HIP_DIFF}```"
    )


def _q_fp32_accumulator(v: dict, rng: random.Random) -> str:
    return (
        f"The '{v['task_id']}' kernel operates in {v['dtype']} but accumulates in "
        "fp32. Explain why the fp32 accumulator is required for correctness on "
        f"{v['gpu']}, and what SNR failure mode appears if you accumulate in "
        f"{v['dtype']} instead."
    )


def _q_wavefront_block(v: dict, rng: random.Random) -> str:
    return (
        f"On {v['gpu']} ({v['uarch']}) the wavefront is 64 lanes. For the '{v['operation']}' "
        "kernel, explain why BLOCK_M/BLOCK_N/BLOCK_K should be multiples of 64 and "
        "how a non-multiple (say 96) hurts MFMA utilization and masking."
    )


def _q_mfma_tl_dot(v: dict, rng: random.Random) -> str:
    return (
        f"For the '{v['task_id']}' matmul-style kernel, explain why you should use "
        "`tl.dot` instead of a hand-rolled scalar FMA inner loop so Triton emits "
        f"MFMA (matrix-core) instructions on {v['gpu']}, and how that interacts with "
        "the fp32 accumulator."
    )


# type -> (builder, style). Styles alternate so both regimes are represented.
_QA_BUILDERS: dict[str, tuple[Callable[[dict, random.Random], str], str]] = {
    "occupancy_bound": (_q_occupancy_bound, "think"),
    "rocprof_counter": (_q_rocprof_counter, "no_think"),
    "hip_diff_bug": (_q_hip_diff_bug, "think"),
    "fp32_accumulator": (_q_fp32_accumulator, "no_think"),
    "wavefront_block": (_q_wavefront_block, "think"),
    "mfma_tl_dot": (_q_mfma_tl_dot, "no_think"),
}

_STYLE_DIRECTIVE = {
    "think": (
        "\n\nReason step by step INSIDE a <think> ... </think> block, then give a "
        "concise final answer after it."
    ),
    "no_think": (
        "\n\nAnswer directly and concisely. Do NOT include a visible chain of "
        "thought."
    ),
}


def build_qa_messages(view: dict, qa_type: str, rng: random.Random) -> tuple[list[dict], str]:
    """Build the (messages, style) for one QA row (assistant turn not yet filled)."""
    builder, style = _QA_BUILDERS[qa_type]
    system = SYSTEM_PROMPT + _STYLE_DIRECTIVE[style]
    user = builder(view, rng)
    return (
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        style,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_kernel_qa(
    tasks: Iterable[Any],
    teacher: TeacherClient,
    n: int,
    seed: int = 0,
    qa_types: Optional[Iterable[str]] = None,
) -> list[dict]:
    """Generate ``n`` kernel/GPU/ROCm QA SFT rows grounded in ``tasks``.

    Each row is ``{"messages": [system, user, assistant], "_source": "kernel_qa",
    "_style": "think"|"no_think", "_qa_type": <type>, "_task_id": <id>}`` where
    the assistant turn is produced by ``teacher.generate``. Deterministic given
    ``seed`` and a deterministic teacher (e.g. ``StubTeacher``).

    Both thinking and no-think exemplar styles are emitted (the QA-type roster
    balances them), so the mixture teaches on/off-thinking behavior.
    """
    if n <= 0:
        return []
    with log.stage("generate_kernel_qa", n=n, seed=seed):
        views = [_view(t) for t in tasks]
        if not views:
            raise ValueError("generate_kernel_qa requires at least one task")

        types = list(qa_types) if qa_types else list(KERNEL_QA_TYPES)
        for t in types:
            if t not in _QA_BUILDERS:
                raise ValueError(f"unknown qa_type {t!r}; known: {tuple(_QA_BUILDERS)}")

        # Deterministic (task, type) plan, shuffled then cycled to length n.
        plan = [(vi, ty) for vi in range(len(views)) for ty in types]
        rng = random.Random(seed)
        rng.shuffle(plan)

        # Build a deterministic, over-provisioned plan (the buffer absorbs the rare
        # empty answer), construct all prompts up front (pure/deterministic given
        # ``seed``), then run the network-bound teacher calls CONCURRENTLY. QA rows
        # are independent, so a thread pool collapses a ~9h sequential pass into
        # minutes. Rows are assembled in plan order, so the output stays order-stable.
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        buf = int(n * 1.15) + 8
        specs: list[tuple] = []
        for p in range(buf):
            vi, qa_type = plan[p % len(plan)]
            view = views[vi]
            # Per-row RNG keeps counter choice etc. deterministic and decorrelated.
            row_rng = random.Random(f"{seed}:{p + 1}:{qa_type}")
            messages, style = build_qa_messages(view, qa_type, row_rng)
            specs.append((messages, style, qa_type, view["task_id"]))

        workers = max(1, int(os.environ.get("KORE_QA_WORKERS", "32") or 32))
        answers: list = [None] * len(specs)
        t_start = time.time()
        done = 0

        def _ask(k: int):
            return k, teacher.generate(specs[k][0])

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(_ask, k) for k in range(len(specs))]):
                k, ans = fut.result()
                answers[k] = ans
                done += 1
                if done % 50 == 0 or done == len(specs):
                    log.progress(done, len(specs), "kernel_qa", t_start=t_start)

        by_type: Counter = Counter()
        by_style: Counter = Counter()
        rows: list[dict] = []
        for k, (messages, style, qa_type, task_id) in enumerate(specs):
            if len(rows) >= n:
                break
            answer = answers[k]
            if not isinstance(answer, str) or not answer.strip():
                log.debug("empty QA answer; skipping", qa_type=qa_type, task=task_id)
                continue
            rows.append({
                "messages": messages + [{"role": "assistant", "content": answer}],
                "_source": QA_SOURCE_TAG,
                "_style": style,
                "_qa_type": qa_type,
                "_task_id": task_id,
            })
            by_type[qa_type] += 1
            by_style[style] += 1
        log.metric("qa_summary", n=len(rows), by_type=dict(by_type),
                   by_style=dict(by_style), workers=workers)
        return rows
