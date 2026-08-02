"""The checkpoint A/B against real hardware and real weights, opt-in via ``-m gpu``.

`tests/test_checkpoint_ab.py` proves the harness logic against a stub env and a
stub arm. What a stub cannot prove is that the measurement half is wired to the
real thing: that :func:`kore.eval.checkpoint_ab.measure_arm` drives an actual
:class:`~kore.env.kore_env.KoreEnv`, that a kernel the verifier is known to
accept comes back as ACCEPTED through the replay path, and that
:func:`kore.eval.checkpoint_ab.load_hf_batch_generate` really produces both a
completion and a per-token likelihood from a 14B checkpoint. That is this module.

Every test SKIPS with a reason when its resource is absent, because neither an
accelerator nor a 55 GiB checkpoint is something CI can conjure. Per
`tests/test_marker_contract.py`, `gpu`-marked tests live in a `test_gpu*.py`
module; no new marker is introduced.

    # measurement path only (needs a GPU, no model)
    python -m pytest -m gpu tests/test_gpu_checkpoint_ab.py -q

    # add the model path (a local checkpoint dir or a cached Hub id)
    KORE_AB_TEST_MODEL=/path/to/runs/midtrain_14b_frontier \
    python -m pytest -m gpu tests/test_gpu_checkpoint_ab.py -q

    # add the endpoint path (see docs/E2E_SERVING_GATE.md for the serve command)
    KORE_AB_TEST_BASE_URL=http://127.0.0.1:30000 KORE_AB_TEST_SERVED_MODEL=Qwen3-14B \
    python -m pytest -m gpu tests/test_gpu_checkpoint_ab.py -q
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

from kore.eval import checkpoint_ab as ab
from kore.eval import heldout_lm as hl

pytestmark = pytest.mark.gpu

#: Kept to one task and a short budget: this suite proves the path is real, not
#: how good the model is. The measurement itself is the sbatch job's job.
MAX_TOKENS = 256


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _require_accelerator() -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep of the GPU path
        pytest.skip("torch is not installed")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("no ROCm/CUDA device visible to this process")


def _require_model() -> str:
    model = _env("KORE_AB_TEST_MODEL")
    if not model:
        pytest.skip("set KORE_AB_TEST_MODEL to a checkpoint dir or cached Hub id")
    return model


def _require_endpoint() -> tuple[str, str]:
    base_url = _env("KORE_AB_TEST_BASE_URL")
    if not base_url:
        pytest.skip("set KORE_AB_TEST_BASE_URL to an OpenAI-compatible endpoint")
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=10):
            pass
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"endpoint {base_url} unreachable: {exc}")
    return base_url, _env("KORE_AB_TEST_SERVED_MODEL")


@pytest.fixture(scope="module")
def task():
    scope = ab.generalization_scope()
    assert scope, "the generalization scope is empty"
    # A small, cheap held-out task whose committed seed is recorded PASS in
    # data/gfx950_task_verification.json, so a rejection here is the harness's
    # fault and not the kernel's.
    preferred = "genb_fx_reglu_act_fp16"
    return next((t for t in scope if t.task_id == preferred), scope[0])


def _seed_generation_rows(task, arm: str = "seed") -> list[dict]:
    """One generation row whose submitted source is the task's own seed kernel.

    The seed is the one input whose verdict is known independently (the corpus
    verification sweep accepts it), which makes it the right probe for "is the
    measurement path real" as distinct from "is the model any good".
    """
    from kore.policy.format import wrap_full_kernel

    def generate(messages, max_tokens: int = MAX_TOKENS, temperature: float = 0.0):
        return ("ANALYSIS\nreplay the verified seed\n\n"
                "PROPOSED_CHANGE\nnone\n\n" + wrap_full_kernel(task.seed_source))

    records = ab.generate_arm([task], generate, arm=arm, max_tokens=MAX_TOKENS)
    assert records[0].contract_ok, "the seed replay must parse as a FULL_KERNEL"
    return [r.to_dict() for r in records]


# --------------------------------------------------------------------------- #
# The measurement path (GPU, no model needed)
# --------------------------------------------------------------------------- #
def test_measure_arm_accepts_the_verified_seed_kernel_on_hardware(task):
    """The replay path, the real verifier, and a known-good kernel agree."""
    _require_accelerator()
    from kore.env.kore_env import KoreEnv

    rows = _seed_generation_rows(task)
    result = ab.measure_arm(rows, [task], arm="seed",
                            env_factory=lambda t: KoreEnv(t, use_replay=False),
                            budget=1, mode="parallel")
    summary = ab.arm_summary(result, rows, arm="seed")
    obs = result["observations"][0]
    if obs["infra_error"]:
        pytest.skip(f"infrastructure fault, not a kernel verdict: {obs['error_tail']}")
    counts = summary["counts"]
    assert counts["contract_ok"] == 1
    assert counts["compiled"] == 1, f"seed did not build: {obs['error_tail']}"
    assert counts["correct"] == 1, f"verifier rejected the seed: {obs['error_tail']}"
    assert obs["snr_db"] is not None
    # A speedup may or may not be admissible depending on the driver's timing
    # grade; correctness is the claim this test makes.
    assert summary["per_task"][0]["timing_grade"] in (
        "publication", "screening", "ineligible", "compat")


def test_a_nonsense_kernel_is_rejected_on_hardware(task):
    """The path can fail as well as pass, so a PASS above means something."""
    _require_accelerator()
    from kore.env.kore_env import KoreEnv

    def prose(messages, **kw):
        return "I would start by increasing the block size and re-benching."

    rows = [r.to_dict()
            for r in ab.generate_arm([task], prose, arm="prose",
                                     max_tokens=MAX_TOKENS)]
    result = ab.measure_arm(rows, [task], arm="prose",
                            env_factory=lambda t: KoreEnv(t, use_replay=False),
                            budget=1, mode="parallel")
    summary = ab.arm_summary(result, rows, arm="prose")
    assert summary["counts"]["contract_ok"] == 0
    assert summary["counts"]["compiled"] == 0
    assert summary["counts"]["correct"] == 0


def test_the_two_arms_produce_a_comparable_report_on_hardware(task):
    """A full A/B over one task: seed arm vs prose arm, through the real env."""
    _require_accelerator()
    from kore.env.kore_env import KoreEnv

    def factory(t):
        return KoreEnv(t, use_replay=False)

    seed_rows = _seed_generation_rows(task, arm="cand")
    good = ab.arm_summary(
        ab.measure_arm(seed_rows, [task], arm="cand", env_factory=factory),
        seed_rows, arm="cand")
    if good["per_task"][0]["infra_error"]:
        pytest.skip("infrastructure fault on the candidate arm")

    prose_rows = [r.to_dict() for r in ab.generate_arm(
        [task], lambda m, **kw: "no kernel here", arm="ref", max_tokens=MAX_TOKENS)]
    bad = ab.arm_summary(
        ab.measure_arm(prose_rows, [task], arm="ref", env_factory=factory),
        prose_rows, arm="ref")

    ab.assert_prompts_matched({"cand": seed_rows, "ref": prose_rows})
    report = ab.build_report(good, bad, scope=ab.scope_report([task]), n_boot=200)
    json.dumps(report)
    assert report["comparison"]["verdict"]["direction"] == "candidate_better"
    ab.format_report(report).encode("ascii")


# --------------------------------------------------------------------------- #
# The model path (needs real weights)
# --------------------------------------------------------------------------- #
def test_load_hf_batch_generate_produces_a_completion_and_a_likelihood(task):
    """One load serves generation and per-token loss, as the A/B runner assumes."""
    _require_accelerator()
    model = _require_model()
    backend = ab.load_hf_batch_generate(model, dtype="bfloat16")
    try:
        info = backend["info"]
        assert info["n_parameters"] > 0
        assert info["padding_side"] == "left", (
            "a batched decoder-only generate must left-pad or every short "
            "sequence continues from pad tokens")
        assert info["enable_thinking"] is False

        messages = ab.first_turn_messages(task)
        text = backend["generate"](messages, max_tokens=MAX_TOKENS, temperature=0.0)
        assert isinstance(text, str) and text.strip()

        batch = backend["generate_batch"]([messages, messages],
                                          max_tokens=32, temperature=0.0)
        assert len(batch) == 2
        assert batch[0] == batch[1], "greedy decoding must be deterministic"

        out = backend["nll"](task.seed_source)
        assert out["n_tokens"] > 50
        assert out["sum_nll_nats"] > 0.0
        bits = out["sum_nll_nats"] / out["n_tokens"] / hl.LN2
        assert 0.0 < bits < 17.0, (
            f"{bits:.2f} bits/token is at or above uniform over the vocabulary, "
            "which means the weights or the tokenizer are not the ones intended")
        assert out["tokens_sha"] == backend["nll"](task.seed_source)["tokens_sha"]
    finally:
        backend["close"]()


def test_generated_arm_records_survive_a_roundtrip_with_real_weights(tmp_path, task):
    _require_accelerator()
    model = _require_model()
    backend = ab.load_hf_batch_generate(model, dtype="bfloat16")
    try:
        records = ab.generate_arm(
            [task], backend["generate"], arm="real",
            generate_batch=backend["generate_batch"], batch_size=1,
            max_tokens=MAX_TOKENS)
    finally:
        backend["close"]()
    path = ab.write_generations(tmp_path / "g.jsonl", records, arm="real")
    meta, rows = ab.read_generations(path)
    assert meta["schema"] == ab.SCHEMA_GENERATIONS
    assert rows[0]["response"], "the arm produced no text at all"
    assert rows[0]["prompt_sha"] == ab.prompt_digest(ab.first_turn_messages(task))


# --------------------------------------------------------------------------- #
# The endpoint path (needs a served model)
# --------------------------------------------------------------------------- #
def test_endpoint_generate_speaks_the_message_list_contract(task):
    """The endpoint client must send system+task turns, not a flattened prompt."""
    base_url, served = _require_endpoint()
    generate = ab.endpoint_generate(base_url, served)
    messages = ab.first_turn_messages(task)
    assert [m["role"] for m in messages] == ["system", "user"]
    text = generate(messages, max_tokens=MAX_TOKENS, temperature=0.0)
    assert isinstance(text, str) and text.strip()


def test_endpoint_arm_generates_and_records(task):
    base_url, served = _require_endpoint()
    records = ab.generate_arm([task], ab.endpoint_generate(base_url, served),
                              arm="endpoint", max_tokens=MAX_TOKENS)
    assert len(records) == 1
    assert records[0].error is None, records[0].error
    assert records[0].response_chars > 0
