"""The serving gate against a REAL served model, opt-in via ``-m gpu``.

`tests/test_e2e_serving_gate.py` proves the gate's logic against a stdlib stub.
What it cannot prove is that a production engine on ROCm answers the same
protocol the client writes - that the OpenAI response shape matches, that a
reasoning model's output survives the token count, and that the numbers coming
back are a measurement of hardware rather than of a loopback socket. That is
this module.

It carries the `gpu` marker even though the pytest process never touches a GPU:
the endpoint it measures is a served model occupying an accelerator, so the test
belongs to the same opt-in group as the rest of the hardware suite, and
`tests/test_marker_contract.py` requires `gpu`-marked tests to live in a
`test_gpu*.py` module. No new marker is introduced.

Everything is configured by environment, and an unconfigured or unreachable
endpoint **skips with the reason** rather than failing - the endpoint is not
something CI can conjure. See `docs/E2E_SERVING_GATE.md` for the serve command.

    KORE_E2E_BASE_URL=http://127.0.0.1:30000 \
    KORE_E2E_MODEL=Qwen3-14B KORE_E2E_ENGINE=sglang \
    python -m pytest -m gpu tests/test_gpu_e2e_serving_gate.py
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

from kore.eval.e2e_sglang_vllm import (
    Workload,
    e2e_accuracy,
    e2e_gate,
    e2e_gate_endpoints,
    e2e_measure,
    e2e_throughput,
    gate_artifact,
)

pytestmark = pytest.mark.gpu

#: Kept small on purpose: this suite proves the path is real, not how fast the
#: node is. A throughput study uses the CLI with a proper request count.
REQUESTS = 4
MAX_NEW_TOKENS = 32
#: A reasoning model spends its first few hundred tokens thinking, so the
#: accuracy tasks need room to reach an answer at all.
ACCURACY_MAX_NEW_TOKENS = 512


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _reachable(base_url: str) -> str:
    """Empty string when the endpoint answers, else why it did not."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=10) as r:
            json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return ""


def _require(var: str) -> str:
    base_url = _env(var)
    if not base_url:
        pytest.skip(
            f"{var} is unset; the serving gate needs a live OpenAI-compatible "
            "endpoint (see docs/E2E_SERVING_GATE.md for the serve command)")
    why = _reachable(base_url)
    if why:
        pytest.skip(f"{var}={base_url} did not answer /v1/models - {why}")
    return base_url


@pytest.fixture(scope="module")
def base_url() -> str:
    return _require("KORE_E2E_BASE_URL")


@pytest.fixture(scope="module")
def model() -> str:
    return _env("KORE_E2E_MODEL")


@pytest.fixture(scope="module")
def engine() -> str:
    return _env("KORE_E2E_ENGINE", "sglang")


def test_endpoint_serves_the_model_the_gate_will_ask_for(base_url, model) -> None:
    """The OpenAI ``model`` field must name something the server actually has.

    A mismatch is the most common way a provisioned gate fails, and it fails as
    an HTTP error deep inside a timing loop rather than as anything readable.
    """
    with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=10) as r:
        served = [entry["id"] for entry in json.loads(r.read().decode("utf-8"))["data"]]
    assert served, f"{base_url} reports no served models"
    if model:
        assert model in served, f"KORE_E2E_MODEL={model!r} not in {served}"


def test_throughput_is_measured_from_the_live_endpoint(base_url, model, engine) -> None:
    result = e2e_throughput(
        model=model, engine=engine, base_url=base_url,
        workload=Workload(num_requests=REQUESTS, max_new_tokens=MAX_NEW_TOKENS),
        served_kernel=_env("KORE_E2E_KERNEL"),
    )
    assert result.kind == "throughput" and result.unit == "tokens/s"
    assert result.candidate_value > 0, (
        f"a live endpoint produced 0 tokens/s: {result.detail}")
    assert result.passed is True
    assert f"over {REQUESTS} reqs" in result.detail


def test_accuracy_scores_the_builtin_workload_on_the_live_model(
    base_url, model, engine,
) -> None:
    result = e2e_accuracy(
        model=model, engine=engine, base_url=base_url,
        workload=Workload(max_new_tokens=ACCURACY_MAX_NEW_TOKENS),
    )
    assert result.kind == "accuracy"
    assert 0.0 <= result.candidate_value <= 1.0
    assert result.detail.endswith("/4 correct")
    # An instruction-following model of any size should manage arithmetic and
    # world-capital trivia; a zero here means the served model is answering
    # nothing, which is a provisioning fault rather than a model result.
    assert result.candidate_value > 0.0, (
        f"the served model scored 0/4 on the built-in workload: {result.detail}")


def test_gate_reaches_a_decision_from_two_real_measurements(
    base_url, model, engine,
) -> None:
    """One endpoint, measured twice: the second run is gated on the first.

    With the same kernel on both sides the expected effect is zero, so this
    asserts the *mechanics* - that a decision is produced from measured numbers
    and that its parts agree with them - not which way it lands.
    """
    workload = Workload(num_requests=REQUESTS, max_new_tokens=MAX_NEW_TOKENS)
    accuracy_workload = Workload(max_new_tokens=ACCURACY_MAX_NEW_TOKENS)
    common = dict(model=model, engine=engine, workload=workload,
                  accuracy_workload=accuracy_workload)

    base_tput, base_acc = e2e_measure(base_url=base_url, role="baseline", **common)
    cand_tput, cand_acc = e2e_measure(
        base_url=base_url, role="candidate",
        baseline_tokens_per_s=base_tput.candidate_value,
        baseline_accuracy=base_acc.candidate_value, **common,
    )
    gate = e2e_gate(cand_tput, cand_acc)

    assert set(gate) == {"accept", "throughput_improved", "accuracy_held",
                         "throughput", "accuracy"}
    assert isinstance(gate["accept"], bool)
    assert gate["throughput_improved"] is (
        cand_tput.candidate_value > base_tput.candidate_value)
    assert gate["accuracy_held"] is (
        cand_acc.candidate_value >= base_acc.candidate_value - 1e-3)
    assert gate["accept"] is (gate["throughput_improved"] and gate["accuracy_held"])
    # Archivable, because a decision nobody can re-read is not a result.
    json.dumps(gate_artifact(gate, engine=engine, model=model, base_url=base_url))


def test_gate_compares_two_served_kernels(model, engine) -> None:
    """The real protocol: two servers differing only in the kernel registered."""
    baseline_url = _require("KORE_E2E_BASE_URL")
    candidate_url = _require("KORE_E2E_CANDIDATE_URL")
    gate = e2e_gate_endpoints(
        baseline_base_url=baseline_url, candidate_base_url=candidate_url,
        model=model, engine=engine, served_kernel=_env("KORE_E2E_KERNEL"),
        workload=Workload(num_requests=REQUESTS, max_new_tokens=MAX_NEW_TOKENS),
        accuracy_workload=Workload(max_new_tokens=ACCURACY_MAX_NEW_TOKENS),
    )
    tput, acc = gate["throughput"], gate["accuracy"]
    assert tput.baseline_value > 0 and tput.candidate_value > 0
    assert tput.rel_change is not None
    assert acc.baseline_value is not None
    assert gate["accept"] is (gate["throughput_improved"] and gate["accuracy_held"])
