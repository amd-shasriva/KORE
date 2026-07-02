"""End-to-end gate: a microbench win must survive real serving (KORE.pdf Sec 4.7).

A kernel that is faster in the isolated verifier is only *useful* if the win
holds when the kernel is swapped into a production inference server (vLLM or
SGLang on ROCm) with NO accuracy regression. This module is the scaffold for
that gate.

It is intentionally IMPORT-GUARDED and does NOT require a served model to
import (so it is safe on a CPU box / in CI). The heavy dependencies (vllm /
sglang / a running server) are only touched inside the functions, which raise a
clear, actionable error when the environment is not provisioned.

How to actually run the gate (operator checklist):

  1. Build the winning kernel into a ROCm-compatible op and register it so the
     server picks it up. Two common routes:
       - vLLM: install a custom op / monkeypatch the target layer
         (e.g. the fused GEMM / attention path) before ``LLM(...)`` is created.
       - SGLang: register the kernel in the ROCm backend and launch the server
         with ``--attention-backend`` / custom-op flags pointing at it.
  2. Serve BOTH the baseline (stock kernel) and the candidate (winning kernel)
     with identical model, dtype, TP/PP layout, batch/seq settings, and seed.
  3. Measure throughput (tokens/s) under a fixed workload (see ``Workload``).
  4. Measure accuracy on a held-out eval set; require no regression beyond a
     tolerance (default: within 0.1% absolute of baseline).
  5. Gate: accept only if tokens/s improved AND accuracy did not regress.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Optional


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


VLLM_AVAILABLE = _module_available("vllm")
SGLANG_AVAILABLE = _module_available("sglang")


@dataclass
class Workload:
    """A fixed serving workload so baseline and candidate are compared fairly."""

    prompts: list = field(default_factory=list)
    max_new_tokens: int = 128
    num_requests: int = 256
    concurrency: int = 32
    seed: int = 0


@dataclass
class E2EResult:
    engine: str                       # "vllm" | "sglang"
    kind: str                         # "throughput" | "accuracy"
    baseline_value: Optional[float]
    candidate_value: Optional[float]
    unit: str
    passed: bool
    detail: str = ""

    @property
    def rel_change(self) -> Optional[float]:
        if not self.baseline_value:
            return None
        return (self.candidate_value - self.baseline_value) / self.baseline_value


class E2ENotProvisioned(RuntimeError):
    """Raised when a real serving stack is required but not available."""


def _require_engine(engine: str) -> None:
    engine = (engine or "").lower()
    if engine == "vllm" and not VLLM_AVAILABLE:
        raise E2ENotProvisioned(
            "vLLM is not installed. This E2E gate requires a served vLLM-ROCm "
            "model. Install vllm (ROCm build) and provide a served endpoint."
        )
    if engine == "sglang" and not SGLANG_AVAILABLE:
        raise E2ENotProvisioned(
            "SGLang is not installed. This E2E gate requires a served "
            "SGLang-ROCm model. Install sglang (ROCm build) and provide a "
            "served endpoint."
        )
    if engine not in ("vllm", "sglang"):
        raise ValueError(f"unknown engine {engine!r}; expected 'vllm' or 'sglang'")


def e2e_throughput(
    model: str,
    served_kernel: str,
    workload: Workload,
    engine: str = "vllm",
    baseline_tokens_per_s: Optional[float] = None,
) -> E2EResult:
    """Measure serving throughput (tokens/s) with the candidate kernel swapped in.

    Requires a provisioned vLLM/SGLang-ROCm stack; raises ``E2ENotProvisioned``
    otherwise. ``served_kernel`` is the kernel source (or a handle) to register
    before the engine loads. ``baseline_tokens_per_s`` lets a caller pass a
    previously-measured stock number for the gate comparison.
    """
    _require_engine(engine)
    # --- Real implementation (only reached on a provisioned box) ---------
    #   1. register ``served_kernel`` into the engine's op registry
    #   2. build the engine: LLM(model=model, dtype=..., tensor_parallel_size=...)
    #   3. replay ``workload`` and time the generation to get tokens/s
    #   4. compare against ``baseline_tokens_per_s``
    raise E2ENotProvisioned(
        f"e2e_throughput requires a served {engine} model (model={model!r}). "
        "This is a documented scaffold; wire in the engine on a ROCm serving box."
    )


def e2e_accuracy(
    model: str,
    served_kernel: str,
    workload: Workload,
    engine: str = "vllm",
    baseline_accuracy: Optional[float] = None,
    tol_abs: float = 1e-3,
) -> E2EResult:
    """Measure served-model accuracy with the candidate kernel; check no regression.

    Requires a provisioned serving stack; raises ``E2ENotProvisioned`` otherwise.
    The gate passes only if candidate accuracy is within ``tol_abs`` of the
    baseline (or higher).
    """
    _require_engine(engine)
    # --- Real implementation (only reached on a provisioned box) ---------
    #   1. register ``served_kernel``; build the engine as in e2e_throughput
    #   2. run the eval set; compute task accuracy
    #   3. passed = candidate_acc >= baseline_acc - tol_abs
    raise E2ENotProvisioned(
        f"e2e_accuracy requires a served {engine} model (model={model!r}). "
        "This is a documented scaffold; wire in the eval harness on a serving box."
    )


def e2e_gate(
    throughput: E2EResult,
    accuracy: E2EResult,
) -> dict:
    """Combine the two measurements into the final accept/reject.

    A microbench win is accepted only if tokens/s improved AND accuracy did not
    regress. This is PURE and testable given two ``E2EResult`` objects.
    """
    tput_ok = throughput.passed and (
        throughput.candidate_value is not None
        and throughput.baseline_value is not None
        and throughput.candidate_value > throughput.baseline_value
    )
    acc_ok = accuracy.passed
    return {
        "accept": bool(tput_ok and acc_ok),
        "throughput_improved": bool(tput_ok),
        "accuracy_held": bool(acc_ok),
        "throughput": throughput,
        "accuracy": accuracy,
    }


def _cli() -> int:
    print("KORE E2E gate (vLLM / SGLang on ROCm)")
    print(f"  vllm available:   {VLLM_AVAILABLE}")
    print(f"  sglang available: {SGLANG_AVAILABLE}")
    print(
        "\nThis entrypoint REQUIRES a served model on a ROCm serving box.\n"
        "It is a documented scaffold: provide --model and a winning kernel,\n"
        "then run e2e_throughput / e2e_accuracy against your served engine."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    raise SystemExit(_cli())
