"""The held-out LM-loss harness, on CPU with a stub scorer.

``kore.eval.heldout_lm`` carries the highest-powered half of the mid-train
evaluation: single-shot kernel generation yields one bit per held-out task, while
teacher-forced loss over the same kernels yields tens of thousands of paired
per-token measurements. Because that is the number most likely to be quoted, the
guards around it need tests more than the arithmetic does:

  * a document set that includes a contaminated task must be REFUSED, not scored
    (the whole point of the scope is that leaked source cannot carry a held-out
    claim);
  * two arms that tokenize a document differently must be refused as unpaired,
    because a per-token delta across different tokenizations is not a difference
    in model quality;
  * the general-domain probe must stay clearly separated from the in-domain one -
    pooling "did it learn Triton" with "did it forget English" would hide exactly
    the trade-off the measurement exists to expose.
"""

from __future__ import annotations

import json
import math

import pytest

from kore.eval import heldout_lm as hl
from kore.eval.checkpoint_ab import generalization_scope


@pytest.fixture(scope="module")
def tasks():
    return generalization_scope()


@pytest.fixture(scope="module")
def seed_docs(tasks):
    return hl.heldout_documents(tasks[:6])


def constant_scorer(bits_per_token: float, *, tokens_per_char: float = 0.25,
                    tokens_sha: str = "fixed"):
    """A stub scorer with an exact, known bits/token and deterministic tokens."""

    def nll(text: str) -> dict:
        n = max(2, int(len(text) * tokens_per_char))
        return {"sum_nll_nats": bits_per_token * hl.LN2 * n, "n_tokens": n,
                "tokens_sha": f"{tokens_sha}:{len(text)}"}

    return nll


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def test_heldout_documents_are_the_task_seed_kernels(tasks):
    docs = hl.heldout_documents(tasks[:3])
    assert [d.task_id for d in docs] == [t.task_id for t in tasks[:3]]
    assert all(d.kind == "seed" for d in docs)
    for doc, task in zip(docs, tasks[:3]):
        assert doc.text == task.seed_source
        assert doc.chars > 100
        assert "triton" in doc.text.lower()


def test_reference_documents_are_the_torch_oracle(tasks):
    docs = hl.heldout_documents(tasks[:3], kinds=("reference",))
    assert docs and all(d.kind == "reference" for d in docs)
    assert all(d.text for d in docs)


def test_unknown_document_kind_is_rejected(tasks):
    with pytest.raises(ValueError, match="unknown document kind"):
        hl.heldout_documents(tasks[:1], kinds=("kernel_source",))


def test_document_dict_carries_a_digest_but_not_the_text(seed_docs):
    d = seed_docs[0].to_dict()
    assert "text" not in d, "an artifact must not inline every kernel source"
    assert len(d["text_sha"]) == 12 and d["chars"] > 0
    assert d["doc_id"].endswith(":seed")


def test_general_documents_are_separate_from_the_kernel_scope():
    docs = hl.general_documents()
    assert docs, "the bundled retention samples produced no general documents"
    assert all(d.kind == "general" for d in docs)
    assert all(d.task_id.startswith("general:") for d in docs)


# --------------------------------------------------------------------------- #
# Contamination guard
# --------------------------------------------------------------------------- #
def test_scope_documents_pass_the_contamination_assertion(tasks):
    info = hl.assert_documents_uncontaminated(hl.heldout_documents(tasks))
    assert info["n_tasks"] == len(tasks)
    assert info["taxonomy_digest"]


def test_a_contaminated_task_document_is_refused(tasks):
    from kore.tasks.registry import generalization_tasks, get_task, heldout_tasks

    contaminated = sorted({t.task_id for t in heldout_tasks()}
                          - {t.task_id for t in generalization_tasks()})
    assert contaminated, "no contaminated task to test the guard with"
    docs = hl.heldout_documents([get_task(contaminated[0])])
    with pytest.raises(AssertionError, match="contaminated"):
        hl.assert_documents_uncontaminated(docs)


def test_a_training_task_document_is_refused_too():
    from kore.tasks.registry import train_tasks

    docs = hl.heldout_documents([train_tasks()[0]])
    with pytest.raises(AssertionError):
        hl.assert_documents_uncontaminated(docs)


# --------------------------------------------------------------------------- #
# Scoring arithmetic
# --------------------------------------------------------------------------- #
def test_bits_per_token_is_the_nats_sum_over_tokens(seed_docs):
    rows = hl.score_documents(seed_docs, constant_scorer(1.25), arm="a")
    assert len(rows) == len(seed_docs)
    for row in rows:
        assert row["bits_per_token"] == pytest.approx(1.25)
        assert row["nats_per_token"] == pytest.approx(1.25 * hl.LN2)
        assert row["sum_nll_nats"] == pytest.approx(
            row["nats_per_token"] * row["n_tokens"])


def test_corpus_totals_are_token_weighted_not_document_averaged():
    rows = [
        {"n_tokens": 1000, "sum_nll_nats": 1000 * 1.0},
        {"n_tokens": 10, "sum_nll_nats": 10 * 5.0},
    ]
    tot = hl.corpus_totals(rows)
    assert tot["n_tokens"] == 1010
    assert tot["bits_per_token"] == pytest.approx((1000 + 50) / 1010 / hl.LN2)
    assert tot["perplexity"] == pytest.approx(math.exp(1050 / 1010))
    # A document-averaged number would have been (1.0 + 5.0)/2 = 3.0 nats.
    assert tot["sum_nll_nats"] / tot["n_tokens"] < 1.1


def test_scores_roundtrip_through_json(tmp_path, seed_docs):
    rows = hl.score_documents(seed_docs, constant_scorer(2.0), arm="a")
    path = hl.write_scores(tmp_path / "s.json", rows, arm="a", model={"id": "stub"})
    meta, back = hl.read_scores(path)
    assert meta["schema"] == hl.SCHEMA_SCORES and meta["arm"] == "a"
    assert meta["model"]["id"] == "stub"
    assert back == rows


# --------------------------------------------------------------------------- #
# The paired comparison
# --------------------------------------------------------------------------- #
def test_a_uniform_improvement_is_detected_with_the_right_sign(seed_docs):
    cand = hl.score_documents(seed_docs, constant_scorer(1.0), arm="cand")
    ref = hl.score_documents(seed_docs, constant_scorer(1.5), arm="ref")
    res = hl.compare_documents(cand, ref, name_a="cand", name_b="ref", n_boot=400)
    assert res["paired"]["effect_size"] == pytest.approx(-0.5)
    assert res["n_documents_improved"] == len(seed_docs)
    assert res["verdict"]["direction"] == "candidate_better"
    assert res["corpus"]["delta_bits_per_token"] == pytest.approx(-0.5)
    assert res["corpus"]["perplexity_ratio"] < 1.0


def test_a_regression_is_reported_as_such(seed_docs):
    cand = hl.score_documents(seed_docs, constant_scorer(2.0), arm="cand")
    ref = hl.score_documents(seed_docs, constant_scorer(1.5), arm="ref")
    res = hl.compare_documents(cand, ref, name_a="cand", name_b="ref", n_boot=400)
    assert res["paired"]["effect_size"] == pytest.approx(0.5)
    assert res["n_documents_improved"] == 0
    assert res["verdict"]["direction"].startswith("reference_better")


def test_identical_arms_are_a_tie_with_a_zero_effect(seed_docs):
    rows_a = hl.score_documents(seed_docs, constant_scorer(1.2), arm="a")
    rows_b = hl.score_documents(seed_docs, constant_scorer(1.2), arm="b")
    res = hl.compare_documents(rows_a, rows_b, name_a="a", name_b="b", n_boot=400)
    assert res["paired"]["effect_size"] == pytest.approx(0.0)
    assert res["verdict"]["direction"] == "tie"
    assert res["verdict"]["significant"] is False


def test_a_tokenizer_mismatch_refuses_a_per_token_comparison(seed_docs):
    cand = hl.score_documents(seed_docs, constant_scorer(1.0), arm="cand")
    ref = hl.score_documents(
        seed_docs, constant_scorer(1.5, tokens_per_char=0.30), arm="ref")
    with pytest.raises(AssertionError, match="tokenized these documents differently"):
        hl.compare_documents(cand, ref, name_a="cand", name_b="ref", n_boot=100)
    forced = hl.compare_documents(cand, ref, name_a="cand", name_b="ref",
                                  n_boot=100, require_matched_tokens=False)
    assert forced["matched_tokenization"] is False
    assert forced["n_mismatched_documents"] == len(seed_docs)


def test_a_differing_token_digest_is_caught_even_at_equal_length(seed_docs):
    cand = hl.score_documents(seed_docs, constant_scorer(1.0, tokens_sha="v1"),
                              arm="cand")
    ref = hl.score_documents(seed_docs, constant_scorer(1.0, tokens_sha="v2"),
                             arm="ref")
    with pytest.raises(AssertionError, match="tokenized"):
        hl.compare_documents(cand, ref, name_a="cand", name_b="ref", n_boot=100)


def test_disjoint_document_sets_are_an_error(seed_docs, tasks):
    cand = hl.score_documents(seed_docs, constant_scorer(1.0), arm="cand")
    other = hl.score_documents(hl.heldout_documents(tasks[6:10]),
                               constant_scorer(1.0), arm="ref")
    with pytest.raises(ValueError, match="no documents in common"):
        hl.compare_documents(cand, other, n_boot=100)


def test_only_the_shared_documents_are_compared(seed_docs):
    cand = hl.score_documents(seed_docs, constant_scorer(1.0), arm="cand")
    ref = hl.score_documents(seed_docs[:3], constant_scorer(1.5), arm="ref")
    res = hl.compare_documents(cand, ref, n_boot=200)
    assert res["n_documents"] == 3


def test_compare_by_kind_keeps_the_domains_apart(seed_docs):
    """In-domain gain and out-of-domain loss must not be averaged together."""
    general = hl.general_documents()
    docs = list(seed_docs) + general
    cand = hl.score_documents(docs, constant_scorer(1.0), arm="cand")
    # The reference is worse in-domain and better out-of-domain: pooling would
    # cancel, which is precisely the mistake this split prevents.
    ref = []
    for row, doc in zip(hl.score_documents(docs, constant_scorer(1.0), arm="ref"),
                        docs):
        shift = +0.5 if doc.kind == "seed" else -0.5
        ref.append({**row, "bits_per_token": row["bits_per_token"] + shift,
                    "sum_nll_nats": (row["bits_per_token"] + shift)
                    * hl.LN2 * row["n_tokens"]})
    res = hl.compare_by_kind(cand, ref, name_a="cand", name_b="ref", n_boot=300)
    assert set(res["by_kind"]) == {"seed", "general"}
    assert res["by_kind"]["seed"]["paired"]["effect_size"] == pytest.approx(-0.5)
    assert res["by_kind"]["general"]["paired"]["effect_size"] == pytest.approx(0.5)
    assert res["by_kind"]["seed"]["verdict"]["direction"] == "candidate_better"
    assert res["by_kind"]["general"]["verdict"]["direction"] == "reference_better"


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def test_lm_report_is_ascii_and_json_serializable(seed_docs):
    cand = hl.score_documents(seed_docs, constant_scorer(1.0), arm="cand")
    ref = hl.score_documents(seed_docs, constant_scorer(1.5), arm="ref")
    res = hl.compare_documents(cand, ref, name_a="cand", name_b="ref", n_boot=200)
    json.dumps(res)
    text = hl.format_lm_report(res)
    text.encode("ascii")
    assert "bits/token" in text and "Verdict" in text
    assert "Negative is better" in text


def test_by_kind_report_labels_the_general_probe_as_not_decontaminated(seed_docs):
    docs = list(seed_docs) + hl.general_documents()
    cand = hl.score_documents(docs, constant_scorer(1.0), arm="cand")
    ref = hl.score_documents(docs, constant_scorer(1.1), arm="ref")
    res = hl.compare_by_kind(cand, ref, name_a="cand", name_b="ref", n_boot=200)
    text = hl.format_by_kind_report(res)
    text.encode("ascii")
    assert "NOT decontaminated" in text
    assert "target domain" in text
