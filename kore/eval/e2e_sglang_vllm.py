"""End-to-end gate: a microbench win must survive real serving (KORE.pdf Sec 4.7).

A kernel that is faster in the isolated verifier is only *useful* if the win
holds when the kernel is swapped into a production inference server (vLLM or
SGLang on ROCm) with NO accuracy regression. This module is the scaffold for
that gate.

It is intentionally IMPORT-GUARDED and does NOT require a served model to
import (so it is safe on a CPU box / in CI). The heavy dependencies (vllm /
sglang / a running server) are only touched inside the functions. When a backend
IS configured - an OpenAI-compatible ``base_url`` endpoint (served vLLM/SGLang)
or a ``model_generate`` callable - :func:`e2e_throughput` / :func:`e2e_accuracy`
run for real (replay the workload, time tokens/s, score accuracy). They raise
:class:`E2ENotProvisioned` ONLY when no backend is provided (with a message on
how to provision one). :func:`e2e_gate` stays pure and testable.

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

Provisioning a backend (the part that is NOT in this process)
-------------------------------------------------------------
``VLLM_AVAILABLE`` / ``SGLANG_AVAILABLE`` report whether the engine is importable
*here*, and both are expected to be ``False`` on the training box. They are
diagnostics, not preconditions: nothing in this module imports an engine. The
gate speaks HTTP to an OpenAI-compatible ``/v1/chat/completions`` endpoint, so
the server runs in its own environment - a container or a separate venv - and
the training stack never takes an engine's torch pin.

``docs/E2E_SERVING_GATE.md`` carries the verified, copy-pasteable procedure for
this node (ROCm 7.2.3, gfx950/MI350X, glibc 2.34), including why the prebuilt
vLLM-ROCm wheels cannot be installed here and which container serves instead.
The short version::

    # one GPU, its own userspace, nothing installed into the training venv
    docker run -d --name kore-e2e-sglang \\
      --device=/dev/kfd --device=/dev/dri/renderD168 \\
      --group-add video --group-add render \\
      --security-opt seccomp=unconfined --shm-size 32g \\
      -v <HF_MODEL_DIR>:/models/m:ro -p 127.0.0.1:30000:30000 \\
      <sglang-rocm-image> \\
      python -m sglang.launch_server --model-path /models/m \\
        --served-model-name <NAME> --host 0.0.0.0 --port 30000

Then point the gate at it - from ANY python that can reach the port::

    python -m kore.eval.e2e_sglang_vllm --engine sglang \\
      --base-url http://127.0.0.1:30000 --model <NAME> --requests 32

which measures throughput + accuracy and, when given a candidate endpoint or an
explicit baseline, prints the accept/reject and can write a JSON artifact.
"""

from __future__ import annotations

import importlib.util
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


VLLM_AVAILABLE = _module_available("vllm")
SGLANG_AVAILABLE = _module_available("sglang")

# The single model interface this gate needs: prompt -> completion string.
ModelGenerate = Callable[..., str]

# A tiny, self-contained accuracy workload (prompt + expected answer substring).
# Deliberately trivial + deterministic so a stub model can pass it on CPU and a
# real served model is exercised end-to-end without any external eval download.
_DEFAULT_E2E_TASKS: tuple[dict, ...] = (
    {"prompt": "What is 2 + 2? Reply with just the number.", "answer": "4"},
    {"prompt": "What is the capital of France? Answer in one word.", "answer": "Paris"},
    {"prompt": "Complete: the opposite of hot is ____ . One word.", "answer": "cold"},
    {"prompt": "How many days are in a week? Reply with just the number.", "answer": "7"},
)


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
    """Raised only when NO serving backend is configured (nothing to measure).

    This is the explicit "not wired up" path: neither an OpenAI-compatible
    ``base_url`` endpoint nor a ``model_generate`` callable was supplied. The
    message explains exactly how to provision one.
    """


_PROVISION_HELP = (
    "No serving backend configured for the E2E gate. Provide ONE of:\n"
    "  - base_url=... : a running OpenAI-compatible endpoint, e.g.\n"
    "      vLLM-ROCm:  python -m vllm.entrypoints.openai.api_server "
    "--model <MODEL> --port 8000   -> base_url='http://localhost:8000'\n"
    "      SGLang-ROCm: python -m sglang.launch_server --model-path <MODEL> "
    "--port 30000     -> base_url='http://localhost:30000'\n"
    "  - model_generate=fn : any callable fn(prompt, **kw) -> str "
    "(e.g. kore.policy.serve.VLLMPolicy(...).generate).\n"
    "Register the winning kernel into the engine BEFORE it loads, then run the "
    "baseline and candidate with identical model/dtype/TP/seed."
)


def _validate_engine(engine: str) -> str:
    e = (engine or "").lower()
    if e not in ("vllm", "sglang"):
        raise ValueError(f"unknown engine {engine!r}; expected 'vllm' or 'sglang'")
    return e


def _count_tokens_default(text: str) -> int:
    """Whitespace token count - a backend-agnostic tokens/s proxy.

    Pass ``count_tokens=`` (e.g. a real tokenizer's ``len(tok.encode(...))``) for
    exact accounting; the default keeps this offline/CPU-safe and deterministic.
    """
    return len(re.findall(r"\S+", text or ""))


def _openai_compatible_generate(
    base_url: str, model: str = "", api_key: Optional[str] = None
) -> ModelGenerate:
    """Build a ``model_generate`` that hits an OpenAI-compatible chat endpoint.

    Works against any vLLM/SGLang server started with the OpenAI API. Uses
    ``requests`` if present, else stdlib ``urllib`` - both imported lazily so a
    box without ``requests`` still imports this module fine.
    """
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"

    def generate(prompt: str, max_tokens: int = 128, temperature: float = 0.0, **kw) -> str:
        payload = {
            "model": model or "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = json.dumps(payload).encode("utf-8")
        timeout = kw.get("timeout", 120)
        try:
            import requests  # optional; lazy

            resp = requests.post(endpoint, data=data, headers=headers, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
        except ImportError:  # pragma: no cover - fallback path
            import urllib.request

            req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read().decode("utf-8"))
        message = body["choices"][0]["message"]
        # A reasoning model served with a reasoning parser (Qwen3 + SGLang's
        # ``--reasoning-parser qwen3``, DeepSeek-R1, ...) splits its output:
        # the chain of thought goes to ``reasoning_content`` and ``content`` is
        # left empty whenever the answer never arrived - e.g. the response hit
        # ``max_tokens`` mid-thought. Falling back keeps the tokens the server
        # really generated in the throughput count instead of silently scoring
        # that request as 0 tokens.
        return message.get("content") or message.get("reasoning_content") or ""

    return generate


def _resolve_generate(
    model_generate: Optional[ModelGenerate],
    base_url: Optional[str],
    api_key: Optional[str],
    model: str,
) -> ModelGenerate:
    """Return a usable ``model_generate`` or raise :class:`E2ENotProvisioned`."""
    if model_generate is not None:
        return model_generate
    if base_url:
        return _openai_compatible_generate(base_url, model, api_key)
    raise E2ENotProvisioned(_PROVISION_HELP)


def _workload_prompts(workload: Workload) -> list[str]:
    base = list(workload.prompts) if workload.prompts else [t["prompt"] for t in _DEFAULT_E2E_TASKS]
    if not base:
        base = ["Hello, world."]
    n = max(1, int(workload.num_requests))
    return [base[i % len(base)] for i in range(n)]


def _answer_matches(output: str, answer) -> bool:
    out = (output or "").lower()
    answers = answer if isinstance(answer, (list, tuple)) else [answer]
    return any(str(a).strip().lower() in out for a in answers)


def e2e_throughput(
    model: str = "",
    served_kernel: str = "",
    workload: Optional[Workload] = None,
    engine: str = "vllm",
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_generate: Optional[ModelGenerate] = None,
    baseline_tokens_per_s: Optional[float] = None,
    count_tokens: Optional[Callable[[str], int]] = None,
    role: str = "candidate",
) -> E2EResult:
    """Measure real end-to-end serving throughput (tokens/s) over a fixed workload.

    Runs for real whenever a backend is configured - either an OpenAI-compatible
    ``base_url`` (a served vLLM/SGLang-ROCm model, with ``served_kernel`` already
    registered into the engine) or a ``model_generate`` callable. It replays
    ``workload``, times generation, and divides generated tokens by wall time.
    Raises :class:`E2ENotProvisioned` ONLY when neither is provided.

    ``baseline_tokens_per_s`` (a previously-measured stock number) sets the pass
    threshold: ``passed`` iff we measured >0 tok/s and beat the baseline (if any).
    """
    engine = _validate_engine(engine)
    workload = workload or Workload()
    gen = _resolve_generate(model_generate, base_url, api_key, model)
    count = count_tokens or _count_tokens_default

    prompts = _workload_prompts(workload)
    total_tokens = 0
    start = time.perf_counter()
    for p in prompts:
        out = gen(p, max_tokens=workload.max_new_tokens, temperature=0.0)
        total_tokens += count(out)
    elapsed = time.perf_counter() - start
    tps = (total_tokens / elapsed) if elapsed > 0 else 0.0

    passed = tps > 0 and (baseline_tokens_per_s is None or tps > baseline_tokens_per_s)
    return E2EResult(
        engine=engine,
        kind="throughput",
        baseline_value=baseline_tokens_per_s,
        candidate_value=tps,
        unit="tokens/s",
        passed=bool(passed),
        detail=f"{role}: {total_tokens} tok / {elapsed:.4f}s over {len(prompts)} reqs",
    )


def e2e_accuracy(
    model: str = "",
    served_kernel: str = "",
    workload: Optional[Workload] = None,
    engine: str = "vllm",
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_generate: Optional[ModelGenerate] = None,
    baseline_accuracy: Optional[float] = None,
    tol_abs: float = 1e-3,
    tasks: Optional[list] = None,
    role: str = "candidate",
) -> E2EResult:
    """Measure served-model task accuracy end-to-end; check no regression.

    Runs for real against a configured backend (``base_url`` or ``model_generate``)
    over a small task workload (``tasks`` = list of ``{"prompt", "answer"}``;
    defaults to :data:`_DEFAULT_E2E_TASKS`). Raises :class:`E2ENotProvisioned`
    ONLY when no backend is provided.

    ``passed`` iff candidate accuracy is within ``tol_abs`` of ``baseline_accuracy``
    (or higher). With no baseline it reports the measured accuracy and passes.
    """
    engine = _validate_engine(engine)
    gen = _resolve_generate(model_generate, base_url, api_key, model)
    tasks = list(tasks) if tasks is not None else list(_DEFAULT_E2E_TASKS)
    max_tokens = workload.max_new_tokens if workload is not None else 32

    correct = 0
    for t in tasks:
        out = gen(t["prompt"], max_tokens=max_tokens, temperature=0.0)
        correct += int(_answer_matches(out, t["answer"]))
    n = len(tasks)
    acc = correct / n if n else 0.0

    passed = (baseline_accuracy is None) or (acc >= baseline_accuracy - tol_abs)
    return E2EResult(
        engine=engine,
        kind="accuracy",
        baseline_value=baseline_accuracy,
        candidate_value=acc,
        unit="accuracy",
        passed=bool(passed),
        detail=f"{role}: {correct}/{n} correct",
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


def e2e_measure(
    *,
    base_url: Optional[str] = None,
    model_generate: Optional[ModelGenerate] = None,
    model: str = "",
    api_key: Optional[str] = None,
    engine: str = "vllm",
    served_kernel: str = "",
    workload: Optional[Workload] = None,
    accuracy_workload: Optional[Workload] = None,
    tasks: Optional[list] = None,
    count_tokens: Optional[Callable[[str], int]] = None,
    baseline_tokens_per_s: Optional[float] = None,
    baseline_accuracy: Optional[float] = None,
    tol_abs: float = 1e-3,
    role: str = "candidate",
) -> tuple[E2EResult, E2EResult]:
    """Measure ``(throughput, accuracy)`` for ONE served endpoint.

    A convenience over calling :func:`e2e_throughput` and :func:`e2e_accuracy`
    with the same backend twice; it exists so the two halves of the gate cannot
    drift apart (same engine, same model, same client).

    ``accuracy_workload`` is separate from ``workload`` on purpose. Throughput
    wants a short, fixed generation length; accuracy wants enough room for the
    model to actually finish an answer, which on a reasoning model means far
    more tokens than a throughput sweep would use.
    """
    tput = e2e_throughput(
        model=model,
        served_kernel=served_kernel,
        workload=workload,
        engine=engine,
        base_url=base_url,
        api_key=api_key,
        model_generate=model_generate,
        baseline_tokens_per_s=baseline_tokens_per_s,
        count_tokens=count_tokens,
        role=role,
    )
    acc = e2e_accuracy(
        model=model,
        served_kernel=served_kernel,
        workload=accuracy_workload if accuracy_workload is not None else workload,
        engine=engine,
        base_url=base_url,
        api_key=api_key,
        model_generate=model_generate,
        baseline_accuracy=baseline_accuracy,
        tol_abs=tol_abs,
        tasks=tasks,
        role=role,
    )
    return tput, acc


def e2e_gate_endpoints(
    *,
    baseline_base_url: Optional[str] = None,
    candidate_base_url: Optional[str] = None,
    baseline_generate: Optional[ModelGenerate] = None,
    candidate_generate: Optional[ModelGenerate] = None,
    model: str = "",
    api_key: Optional[str] = None,
    engine: str = "vllm",
    served_kernel: str = "",
    workload: Optional[Workload] = None,
    accuracy_workload: Optional[Workload] = None,
    tasks: Optional[list] = None,
    count_tokens: Optional[Callable[[str], int]] = None,
    tol_abs: float = 1e-3,
) -> dict:
    """Run the whole gate: measure baseline, measure candidate, decide.

    This is the documented protocol in one call - two servers that differ ONLY
    in which kernel is registered, measured with an identical workload, then
    combined by :func:`e2e_gate`. The candidate's measurement carries the
    baseline's numbers as its thresholds, which is what makes
    ``throughput_improved`` / ``accuracy_held`` meaningful.

    Both endpoints are required: :func:`e2e_gate` cannot accept anything
    without a baseline to beat, so a one-sided run is a measurement, not a gate.
    """
    if not (baseline_base_url or baseline_generate):
        raise E2ENotProvisioned("no BASELINE backend configured.\n" + _PROVISION_HELP)
    if not (candidate_base_url or candidate_generate):
        raise E2ENotProvisioned("no CANDIDATE backend configured.\n" + _PROVISION_HELP)

    common = dict(
        model=model,
        api_key=api_key,
        engine=engine,
        served_kernel=served_kernel,
        workload=workload,
        accuracy_workload=accuracy_workload,
        tasks=tasks,
        count_tokens=count_tokens,
        tol_abs=tol_abs,
    )
    base_tput, base_acc = e2e_measure(
        base_url=baseline_base_url, model_generate=baseline_generate,
        role="baseline", **common,
    )
    cand_tput, cand_acc = e2e_measure(
        base_url=candidate_base_url, model_generate=candidate_generate,
        role="candidate",
        baseline_tokens_per_s=base_tput.candidate_value,
        baseline_accuracy=base_acc.candidate_value,
        **common,
    )
    return e2e_gate(cand_tput, cand_acc)


def gate_artifact(gate: dict, **extra) -> dict:
    """A JSON-serializable record of a gate decision, for archiving a run."""
    from dataclasses import asdict

    def _result(res: E2EResult) -> dict:
        out = asdict(res)
        out["rel_change"] = res.rel_change
        return out

    return {
        "schema": "kore.e2e-serving-gate.v1",
        "accept": gate["accept"],
        "throughput_improved": gate["throughput_improved"],
        "accuracy_held": gate["accuracy_held"],
        "throughput": _result(gate["throughput"]),
        "accuracy": _result(gate["accuracy"]),
        **extra,
    }


def _format(gate: dict) -> str:
    tput, acc = gate["throughput"], gate["accuracy"]
    rel = tput.rel_change
    rel_s = "n/a" if rel is None else f"{rel * 100:+.2f}%"
    return "\n".join([
        f"  throughput : baseline {tput.baseline_value} -> candidate "
        f"{tput.candidate_value:.2f} {tput.unit}  ({rel_s})",
        f"               {tput.detail}",
        f"  accuracy   : baseline {acc.baseline_value} -> candidate "
        f"{acc.candidate_value:.4f}  [{acc.detail}]",
        f"  throughput_improved={gate['throughput_improved']}  "
        f"accuracy_held={gate['accuracy_held']}",
        f"  DECISION   : {'ACCEPT' if gate['accept'] else 'REJECT'}",
    ])


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m kore.eval.e2e_sglang_vllm",
        description="Run KORE's end-to-end serving gate against a live "
                    "OpenAI-compatible endpoint (served vLLM/SGLang on ROCm). "
                    "With no arguments, prints backend availability and how to "
                    "provision one. See docs/E2E_SERVING_GATE.md.",
    )
    p.add_argument("--base-url", help="endpoint to measure (baseline role when "
                                      "--candidate-url is also given)")
    p.add_argument("--candidate-url", help="second endpoint, serving the winning "
                                           "kernel; enables the accept/reject")
    p.add_argument("--model", default="", help="served model name (OpenAI 'model' field)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--engine", default="vllm", choices=("vllm", "sglang"))
    p.add_argument("--served-kernel", default="", help="label for the kernel under test")
    p.add_argument("--requests", type=int, default=32, help="throughput requests")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="generation length for the throughput sweep")
    p.add_argument("--accuracy-max-new-tokens", type=int, default=512,
                   help="generation length for the accuracy tasks; a reasoning "
                        "model needs room to finish thinking AND answer")
    p.add_argument("--baseline-tokens-per-s", type=float, default=None,
                   help="a previously measured stock number, instead of "
                        "--candidate-url")
    p.add_argument("--baseline-accuracy", type=float, default=None)
    p.add_argument("--tol-abs", type=float, default=1e-3,
                   help="allowed absolute accuracy regression")
    p.add_argument("--json", dest="json_out", default=None,
                   help="write the decision to this path as a JSON artifact")
    return p


def _cli(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)

    print("KORE E2E gate (vLLM / SGLang on ROCm)")
    print(f"  vllm importable here:   {VLLM_AVAILABLE}")
    print(f"  sglang importable here: {SGLANG_AVAILABLE}")
    print("  (both False is normal - the gate talks HTTP to a server that "
          "lives in its own environment)")

    if not args.base_url:
        print("\n" + _PROVISION_HELP)
        return 0

    workload = Workload(num_requests=args.requests,
                        max_new_tokens=args.max_new_tokens)
    accuracy_workload = Workload(max_new_tokens=args.accuracy_max_new_tokens)
    common = dict(
        model=args.model, api_key=args.api_key, engine=args.engine,
        served_kernel=args.served_kernel, workload=workload,
        accuracy_workload=accuracy_workload, tol_abs=args.tol_abs,
    )

    if args.candidate_url:
        print(f"\nbaseline : {args.base_url}\ncandidate: {args.candidate_url}")
        gate = e2e_gate_endpoints(baseline_base_url=args.base_url,
                                  candidate_base_url=args.candidate_url, **common)
    else:
        print(f"\nendpoint : {args.base_url}")
        tput, acc = e2e_measure(
            base_url=args.base_url, role="candidate",
            baseline_tokens_per_s=args.baseline_tokens_per_s,
            baseline_accuracy=args.baseline_accuracy, **common,
        )
        gate = e2e_gate(tput, acc)
        if args.baseline_tokens_per_s is None:
            print("  NOTE: no baseline given, so this is a MEASUREMENT, not a "
                  "decision - e2e_gate cannot accept without something to beat.")

    print(_format(gate))

    if args.json_out:
        artifact = gate_artifact(
            gate,
            engine=args.engine, model=args.model,
            served_kernel=args.served_kernel, base_url=args.base_url,
            candidate_url=args.candidate_url,
            workload={"num_requests": workload.num_requests,
                      "max_new_tokens": workload.max_new_tokens,
                      "accuracy_max_new_tokens": accuracy_workload.max_new_tokens},
        )
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote artifact: {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    raise SystemExit(_cli())
