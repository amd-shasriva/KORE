"""The end-to-end serving gate, driven over real HTTP against a stub endpoint.

`kore/eval/e2e_sglang_vllm.py` never imports vLLM or SGLang: it speaks the
OpenAI `/v1/chat/completions` protocol to a server that lives in its own
environment (see `docs/E2E_SERVING_GATE.md`). That design is what makes the gate
testable without hardware — a stdlib `http.server` speaking the same protocol
exercises the *entire* client path (request encoding, response parsing, token
counting, answer matching, the accept/reject) with no GPU, no engine installed,
and no outbound network.

So these tests are not a mock of the gate; they are the gate, wired to a server
that happens to be twelve lines of Python. The only thing they cannot show is
that a real engine answers the same way, which is what the `gpu`-marked
`test_gpu_e2e_serving_gate.py` covers when an endpoint is configured.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kore.eval.e2e_sglang_vllm import (
    E2ENotProvisioned,
    E2EResult,
    Workload,
    e2e_accuracy,
    e2e_gate,
    e2e_gate_endpoints,
    e2e_measure,
    e2e_throughput,
    gate_artifact,
)

#: Correct answers for the module's built-in accuracy workload, keyed by a
#: substring of the prompt. A stub that knows these scores 1.0; one that does
#: not scores 0.0, which is how the accuracy half of the gate is proven to be
#: measuring something rather than always passing.
_ANSWERS = {
    "2 + 2": "4",
    "capital of France": "Paris",
    "opposite of hot": "cold",
    "days are in a week": "7",
}


def _oracle(prompt: str) -> str:
    for needle, answer in _ANSWERS.items():
        if needle in prompt:
            return f"The answer is {answer}."
    return "I do not know."


def _wrong(prompt: str) -> str:
    return "banana"


class _StubEndpoint:
    """A minimal OpenAI-compatible chat endpoint on a loopback port.

    ``responder`` maps a prompt to the assistant text. ``padding`` inflates the
    reply with filler words so a test can make one endpoint measurably more
    productive per request than another - the gate counts tokens, so that is how
    a "faster kernel" is simulated without a kernel.
    """

    def __init__(self, responder=_oracle, padding: int = 0, as_reasoning: bool = False):
        self.responder = responder
        self.padding = padding
        self.as_reasoning = as_reasoning
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):  # keep pytest output clean
                pass

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                stub.requests.append(payload)
                prompt = payload["messages"][-1]["content"]
                text = stub.responder(prompt)
                if stub.padding:
                    text = text + " " + " ".join(["filler"] * stub.padding)
                message = (
                    {"role": "assistant", "content": None, "reasoning_content": text}
                    if stub.as_reasoning
                    else {"role": "assistant", "content": text}
                )
                body = json.dumps(
                    {"choices": [{"message": message, "finish_reason": "stop"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_StubEndpoint":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"


@pytest.fixture
def endpoint():
    with _StubEndpoint() as stub:
        yield stub


def _tiny() -> Workload:
    return Workload(num_requests=4, max_new_tokens=16)


# --------------------------------------------------------------------------- #
# The not-provisioned path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fn", [e2e_throughput, e2e_accuracy])
def test_raises_only_when_no_backend_is_configured(fn) -> None:
    """No ``base_url`` and no ``model_generate`` is the one unmeasurable case."""
    with pytest.raises(E2ENotProvisioned) as excinfo:
        fn(model="m", workload=_tiny())
    message = str(excinfo.value)
    assert "base_url" in message and "model_generate" in message
    assert "sglang.launch_server" in message and "vllm.entrypoints" in message


def test_gate_endpoints_names_the_missing_side() -> None:
    """A one-sided run is a measurement, not a gate, and says which side is missing."""
    with _StubEndpoint() as stub:
        with pytest.raises(E2ENotProvisioned, match="BASELINE"):
            e2e_gate_endpoints(candidate_base_url=stub.base_url, workload=_tiny())
        with pytest.raises(E2ENotProvisioned, match="CANDIDATE"):
            e2e_gate_endpoints(baseline_base_url=stub.base_url, workload=_tiny())


def test_unknown_engine_is_rejected(endpoint) -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        e2e_throughput(engine="tensorrt", base_url=endpoint.base_url, workload=_tiny())


# --------------------------------------------------------------------------- #
# Throughput: a real measurement over the wire
# --------------------------------------------------------------------------- #
def test_throughput_measures_generated_tokens_over_http(endpoint) -> None:
    result = e2e_throughput(
        model="stub", engine="sglang", base_url=endpoint.base_url,
        workload=Workload(num_requests=6, max_new_tokens=16),
    )
    assert isinstance(result, E2EResult)
    assert result.engine == "sglang" and result.kind == "throughput"
    assert result.unit == "tokens/s"
    assert result.candidate_value > 0
    assert result.passed is True
    # Six requests really left the process, carrying the workload's settings.
    assert len(endpoint.requests) == 6
    assert {r["max_tokens"] for r in endpoint.requests} == {16}
    assert {r["model"] for r in endpoint.requests} == {"stub"}
    # "The answer is 4." -> 4 whitespace tokens per reply.
    assert "24 tok" in result.detail and "over 6 reqs" in result.detail


def test_throughput_fails_against_a_baseline_it_does_not_beat(endpoint) -> None:
    """``passed`` is a comparison, not a liveness check."""
    result = e2e_throughput(
        base_url=endpoint.base_url, workload=_tiny(),
        baseline_tokens_per_s=1e12,
    )
    assert result.candidate_value > 0
    assert result.passed is False
    assert result.rel_change is not None and result.rel_change < 0


def test_throughput_honours_a_custom_token_counter(endpoint) -> None:
    """The whitespace default is a proxy; a real tokenizer can be substituted."""
    from kore.eval.e2e_sglang_vllm import _DEFAULT_E2E_TASKS

    # Four requests cycle the four built-in prompts exactly once each.
    replies = [_oracle(task["prompt"]) for task in _DEFAULT_E2E_TASKS]
    result = e2e_throughput(
        base_url=endpoint.base_url, workload=_tiny(),
        count_tokens=len,
    )
    assert f"{sum(len(reply) for reply in replies)} tok" in result.detail
    default = e2e_throughput(base_url=endpoint.base_url, workload=_tiny())
    assert f"{4 * len(replies)} tok" in default.detail  # 4 whitespace tokens each


def test_reasoning_only_response_still_counts_its_tokens() -> None:
    """A server that splits out ``reasoning_content`` must not measure 0 tok/s.

    SGLang with ``--reasoning-parser qwen3`` returns ``content: null`` whenever
    the answer never arrived - a reply truncated at ``max_tokens`` mid-thought.
    Reading only ``content`` would score those very real generated tokens as
    zero and make a slow server look infinitely fast per token.
    """
    with _StubEndpoint(as_reasoning=True) as stub:
        result = e2e_throughput(base_url=stub.base_url, workload=_tiny())
    assert result.candidate_value > 0
    assert "16 tok" in result.detail


# --------------------------------------------------------------------------- #
# Accuracy: the built-in workload, scored
# --------------------------------------------------------------------------- #
def test_accuracy_scores_the_builtin_workload(endpoint) -> None:
    result = e2e_accuracy(model="stub", engine="vllm", base_url=endpoint.base_url)
    assert result.kind == "accuracy"
    assert result.candidate_value == 1.0
    assert result.detail.endswith("4/4 correct")
    assert result.passed is True


def test_accuracy_can_fail(endpoint) -> None:
    """A wrong model scores 0.0 - the scorer is not vacuously passing."""
    with _StubEndpoint(responder=_wrong) as stub:
        result = e2e_accuracy(base_url=stub.base_url, baseline_accuracy=1.0)
    assert result.candidate_value == 0.0
    assert result.passed is False


def test_accuracy_tolerance_is_absolute(endpoint) -> None:
    tasks = [{"prompt": p, "answer": a} for p, a in
             [("What is 2 + 2? Reply with just the number.", "4"),
              ("Unanswerable by this stub", "impossible")]]
    result = e2e_accuracy(base_url=endpoint.base_url, tasks=tasks,
                          baseline_accuracy=0.5, tol_abs=1e-3)
    assert result.candidate_value == 0.5
    assert result.passed is True
    tighter = e2e_accuracy(base_url=endpoint.base_url, tasks=tasks,
                           baseline_accuracy=0.75, tol_abs=1e-3)
    assert tighter.passed is False


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
def _result(kind: str, baseline, candidate, passed: bool) -> E2EResult:
    return E2EResult(engine="vllm", kind=kind, baseline_value=baseline,
                     candidate_value=candidate, unit=kind, passed=passed)


def test_gate_accepts_only_a_faster_kernel_that_held_accuracy() -> None:
    gate = e2e_gate(_result("throughput", 100.0, 120.0, True),
                    _result("accuracy", 0.80, 0.80, True))
    assert gate == {
        "accept": True, "throughput_improved": True, "accuracy_held": True,
        "throughput": gate["throughput"], "accuracy": gate["accuracy"],
    }


@pytest.mark.parametrize(
    "tput, acc, expected",
    [
        ((100.0, 120.0, True), (0.80, 0.60, False), "accuracy regressed"),
        ((100.0, 90.0, False), (0.80, 0.80, True), "throughput regressed"),
        ((None, 120.0, True), (0.80, 0.80, True), "no baseline to beat"),
    ],
)
def test_gate_rejects(tput, acc, expected) -> None:
    gate = e2e_gate(_result("throughput", *tput), _result("accuracy", *acc))
    assert gate["accept"] is False, expected


def test_gate_cannot_accept_without_a_baseline() -> None:
    """The whole point of the gate is a comparison; a lone number is not one."""
    gate = e2e_gate(_result("throughput", None, 999.0, True),
                    _result("accuracy", None, 1.0, True))
    assert gate["throughput_improved"] is False
    assert gate["accuracy_held"] is True
    assert gate["accept"] is False


# --------------------------------------------------------------------------- #
# The whole protocol, end to end over HTTP
# --------------------------------------------------------------------------- #
def test_gate_endpoints_accepts_a_measurably_more_productive_candidate() -> None:
    """Two live endpoints, measured identically, combined into one decision."""
    with _StubEndpoint() as baseline, _StubEndpoint(padding=20) as candidate:
        gate = e2e_gate_endpoints(
            baseline_base_url=baseline.base_url,
            candidate_base_url=candidate.base_url,
            model="stub", engine="sglang",
            workload=Workload(num_requests=8, max_new_tokens=16),
        )
    assert gate["accept"] is True
    assert gate["throughput_improved"] is True and gate["accuracy_held"] is True
    tput = gate["throughput"]
    assert tput.baseline_value > 0 and tput.candidate_value > tput.baseline_value
    assert gate["accuracy"].baseline_value == 1.0
    assert gate["accuracy"].candidate_value == 1.0


def test_gate_endpoints_rejects_a_candidate_that_broke_the_model() -> None:
    """Faster but wrong must not pass - that is the regression the gate exists for."""
    with _StubEndpoint() as baseline, \
            _StubEndpoint(responder=_wrong, padding=50) as candidate:
        gate = e2e_gate_endpoints(
            baseline_base_url=baseline.base_url,
            candidate_base_url=candidate.base_url,
            workload=Workload(num_requests=6, max_new_tokens=16),
        )
    assert gate["throughput_improved"] is True
    assert gate["accuracy_held"] is False
    assert gate["accept"] is False


def test_measure_uses_a_separate_generation_budget_for_accuracy() -> None:
    """Accuracy needs room to finish an answer; throughput wants a short sweep."""
    with _StubEndpoint() as stub:
        e2e_measure(
            base_url=stub.base_url,
            workload=Workload(num_requests=3, max_new_tokens=16),
            accuracy_workload=Workload(max_new_tokens=512),
        )
    budgets = [r["max_tokens"] for r in stub.requests]
    assert budgets[:3] == [16, 16, 16]
    assert set(budgets[3:]) == {512}


def test_gate_artifact_round_trips_through_json() -> None:
    """A gate decision must be archivable - no evaluation is real until it is."""
    with _StubEndpoint() as baseline, _StubEndpoint(padding=10) as candidate:
        gate = e2e_gate_endpoints(
            baseline_base_url=baseline.base_url,
            candidate_base_url=candidate.base_url,
            engine="sglang", model="stub",
            workload=Workload(num_requests=4, max_new_tokens=16),
        )
    artifact = gate_artifact(gate, engine="sglang", model="stub",
                             served_kernel="fused_add_rmsnorm_bf16")
    reloaded = json.loads(json.dumps(artifact))
    assert reloaded["schema"] == "kore.e2e-serving-gate.v1"
    assert reloaded["accept"] is True
    assert reloaded["served_kernel"] == "fused_add_rmsnorm_bf16"
    assert reloaded["throughput"]["unit"] == "tokens/s"
    assert reloaded["throughput"]["rel_change"] > 0
    assert reloaded["accuracy"]["candidate_value"] == 1.0


def test_model_generate_backend_bypasses_http_entirely() -> None:
    """The other provisioning route: any callable, no server at all."""
    calls: list[str] = []

    def generate(prompt: str, **_kw) -> str:
        calls.append(prompt)
        return _oracle(prompt)

    tput, acc = e2e_measure(model_generate=generate,
                            workload=Workload(num_requests=5, max_new_tokens=8))
    assert tput.candidate_value > 0
    assert acc.candidate_value == 1.0
    assert len(calls) == 9  # 5 throughput + 4 accuracy tasks


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_without_an_endpoint_explains_how_to_provision_one(capsys) -> None:
    from kore.eval.e2e_sglang_vllm import _cli

    assert _cli([]) == 0
    out = capsys.readouterr().out
    assert "sglang.launch_server" in out
    # Both flags are diagnostics, not preconditions; the CLI must say so rather
    # than leaving an operator to conclude the gate is unreachable.
    assert "importable here" in out


def test_cli_runs_the_gate_and_writes_an_artifact(tmp_path, capsys) -> None:
    from kore.eval.e2e_sglang_vllm import _cli

    out_path = tmp_path / "gate.json"
    with _StubEndpoint() as baseline, _StubEndpoint(padding=15) as candidate:
        rc = _cli([
            "--base-url", baseline.base_url,
            "--candidate-url", candidate.base_url,
            "--engine", "sglang", "--model", "stub",
            "--requests", "4", "--max-new-tokens", "16",
            "--accuracy-max-new-tokens", "64",
            "--json", str(out_path),
        ])
    assert rc == 0
    assert "DECISION   : ACCEPT" in capsys.readouterr().out
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["accept"] is True
    assert artifact["engine"] == "sglang"
    assert artifact["workload"] == {"num_requests": 4, "max_new_tokens": 16,
                                    "accuracy_max_new_tokens": 64}


def test_cli_says_a_lone_measurement_is_not_a_decision(capsys) -> None:
    from kore.eval.e2e_sglang_vllm import _cli

    with _StubEndpoint() as stub:
        rc = _cli(["--base-url", stub.base_url, "--requests", "3",
                   "--max-new-tokens", "8", "--accuracy-max-new-tokens", "32"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MEASUREMENT, not a" in out
    assert "DECISION   : REJECT" in out
