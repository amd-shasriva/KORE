"""Adversarial leakage, provenance, and frozen-input tests (offline only)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kore.data import decontam as dc
from kore.data.dedup import content_hash
from kore.data.midtrain_corpus import (
    _load_kernelbook_pairs,
    build_midtrain_corpus,
    chunk_text_tokens,
)
from kore.policy.configs import MidTrainConfig


HELDOUT_PAGED = """\
import triton
import triton.language as tl

@triton.jit
def paged_attn_decode(q_ptr, cache_ptr, table_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    slots = tl.load(table_ptr + pid)
    values = tl.load(cache_ptr + slots * BLOCK + tl.arange(0, BLOCK))
    score = tl.sum(values * tl.load(q_ptr + tl.arange(0, BLOCK)), axis=0)
    tl.store(out_ptr + pid, score)
"""


LEGIT_RMSNORM = """\
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    values = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    weights = tl.load(w_ptr + cols, mask=mask, other=0.0)
    variance = tl.sum(values * values, axis=0) / n_cols
    normalized = values * tl.rsqrt(variance + 1e-6)
    tl.store(out_ptr + row * n_cols + cols, normalized * weights, mask=mask)
"""


def _index() -> dc.HoldoutIndex:
    return dc.HoldoutIndex([
        dc.ReferenceDocument(
            "heldout:paged",
            HELDOUT_PAGED,
            family="paged_attention",
            source_id="eval-repo",
            lineage_id="eval-repo@abc",
        )
    ])


def _full_artifact() -> dict:
    benches = {}
    for name in dc.DEFAULT_BENCHMARKS:
        text = f"Full frozen benchmark prompt for {name}; unique row zero."
        benches[name] = {
            "dataset": f"org/{name}",
            "revision": f"commit-{name}-20260723",
            "split": "test",
            "license": "Apache-2.0",
            "records": [{
                "row_id": f"{name}-0",
                "text": text,
                "content_hash": content_hash(text),
            }],
        }
    return {
        "artifact_type": dc.FROZEN_BENCHMARK_ARTIFACT_TYPE,
        "schema_version": dc.FROZEN_BENCHMARK_SCHEMA_VERSION,
        "scope": "full",
        "benchmarks": benches,
    }


def _write_triton(path: Path, marker: str = "good_marker") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "import triton\nimport triton.language as tl\n\n"
        "@triton.jit\n"
        f"def {marker}(x, y, n: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offsets = pid * n + tl.arange(0, n)\n"
        "    values = tl.load(x + offsets)\n"
        "    tl.store(y + offsets, values + 1.0)\n"
    )


def test_generic_triton_boilerplate_is_not_leakage():
    # Imports, decorator, program_id/arange/load/store are common infrastructure,
    # not evidence that this unrelated train-family kernel copied paged attention.
    assert dc.analyze_text_contamination(LEGIT_RMSNORM, _index()) is None


def test_exact_ast_graph_and_embedded_descendants_drop():
    index = _index()
    exact = dc.analyze_text_contamination(HELDOUT_PAGED, index)
    assert exact and exact.reason == "exact_content"

    renamed = (
        HELDOUT_PAGED
        .replace("paged_attn_decode", "renamed_kernel")
        .replace("q_ptr", "query")
        .replace("cache_ptr", "storage")
        .replace("table_ptr", "mapping")
        .replace("out_ptr", "destination")
        .replace("slots", "indices")
        .replace("values", "payload")
        .replace("score", "result")
        .replace("as tl", "as lang")
        .replace("tl.", "lang.")
    )
    ast_match = dc.analyze_text_contamination(renamed, index)
    assert ast_match and ast_match.reason == "normalized_ast"

    tuned_constant = HELDOUT_PAGED.replace("slots * BLOCK", "(slots + 1) * BLOCK")
    graph_match = dc.analyze_text_contamination(tuned_constant, index)
    assert graph_match and graph_match.reason in {
        "semantic_graph", "directional_containment", "minhash_near_duplicate",
    }

    embedded = ("unrelated preface " * 500) + "\n" + HELDOUT_PAGED + \
        "\n" + ("unrelated appendix " * 500)
    containment = dc.analyze_text_contamination(embedded, index)
    assert containment and containment.reason == "directional_containment"
    assert containment.evidence["containment_denominator"] == "reference"


def test_declared_heldout_ancestry_always_drops_unrelated_text():
    index = _index()
    match = dc.analyze_text_contamination(
        LEGIT_RMSNORM,
        index,
        metadata={"root_content_hash": content_hash(HELDOUT_PAGED)},
    )
    assert match and match.reason == "heldout_lineage_descendant"


def test_family_source_and_time_holdouts_are_evidenced():
    policy = dc.HoldoutPolicy(
        families=frozenset({"mla"}),
        source_ids=frozenset({"reserved-source"}),
        lineage_ids=frozenset({"reserved-lineage"}),
        training_cutoff="2026-01-01T00:00:00Z",
    )
    index = dc.HoldoutIndex([], policy=policy)
    assert dc.analyze_text_contamination(
        LEGIT_RMSNORM, index, family="mla",
    ).reason == "heldout_family"
    assert dc.analyze_text_contamination(
        LEGIT_RMSNORM, index, metadata={"source_id": "reserved-source"},
    ).reason == "heldout_source"
    assert dc.analyze_text_contamination(
        LEGIT_RMSNORM, index, metadata={"lineage_id": "reserved-lineage"},
    ).reason == "heldout_source_lineage"
    timed = dc.analyze_text_contamination(
        LEGIT_RMSNORM,
        index,
        metadata={"source_timestamp": "2026-02-01T00:00:00Z"},
    )
    assert timed and timed.reason == "time_holdout"


def test_decontam_report_has_reason_and_non_text_evidence():
    rows = [
        {"text": LEGIT_RMSNORM, "source": "triton"},
        {"text": HELDOUT_PAGED, "source": "copied"},
    ]
    kept, stats = dc.decontaminate_corpus(rows, heldout_ngrams=_index())
    assert [row["source"] for row in kept] == ["triton"]
    assert stats["drop_reasons"] == {"exact_content": 1}
    assert stats["evidence"][0]["reference_id"] == "heldout:paged"
    assert "text" not in stats["evidence"][0]


def test_frozen_full_benchmark_artifact_validates_hashes_and_revisions(tmp_path):
    artifact = _full_artifact()
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(artifact))
    loaded = dc.load_frozen_benchmark_artifact(path)
    assert loaded.scope == "full"
    assert len(loaded.references) == len(dc.DEFAULT_BENCHMARKS)
    assert all(loaded.revisions[name] for name in dc.DEFAULT_BENCHMARKS)
    assert len(dc.eval_benchmark_texts(path)) == len(dc.DEFAULT_BENCHMARKS)

    bad = _full_artifact()
    bad["benchmarks"]["mmlu"]["records"][0]["content_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="content_hash mismatch"):
        dc.load_frozen_benchmark_artifact(bad)

    smoke = _full_artifact()
    smoke["scope"] = "smoke"
    with pytest.raises(ValueError, match="scope='full'"):
        dc.load_frozen_benchmark_artifact(smoke)


def test_production_requires_frozen_benchmark_artifact(tmp_path, monkeypatch):
    monkeypatch.delenv("KORE_DECONTAM_BENCHMARK_ARTIFACT", raising=False)
    config = MidTrainConfig(max_seq_length=64, general_replay_frac=0.0)
    with pytest.raises(FileNotFoundError, match="full frozen benchmark"):
        build_midtrain_corpus(
            tmp_path / "out.jsonl",
            config,
            source_roots=[],
            task_root=tmp_path / "tasks",
            development_mode=False,
        )


def test_development_benchmark_mode_is_explicit(monkeypatch):
    monkeypatch.delenv("KORE_DECONTAM_BENCHMARK_ARTIFACT", raising=False)
    with pytest.raises(FileNotFoundError):
        dc.eval_benchmark_texts()
    assert dc.eval_benchmark_texts(development_mode=True)


def test_source_lineages_drafts_and_unverified_roots_are_excluded(tmp_path):
    train = tmp_path / "train_repo"
    heldout = tmp_path / "heldout_repo"
    unverified = tmp_path / "unverified_repo"
    _write_triton(train / "src" / "good.py", "train_kernel")
    _write_triton(train / "_drafts" / "draft.py", "draft_secret")
    _write_triton(heldout / "src" / "reserved.py", "heldout_secret")
    _write_triton(unverified / "src" / "bad.py", "unverified_secret")

    catalog = {
        "schema_version": "1.0",
        "sources": [
            {
                "local_path": str(train),
                "repository_url": "https://example.test/train.git",
                "commit": "train-commit-1",
                "license": "MIT",
                "lineage_id": "train-lineage",
                "source_id": "train-source",
                "verified": True,
            },
            {
                "local_path": str(heldout),
                "repository_url": "https://example.test/heldout.git",
                "commit": "heldout-commit-1",
                "license": "Apache-2.0",
                "lineage_id": "heldout-lineage",
                "source_id": "heldout-source",
                "verified": True,
            },
        ],
        "datasets": [],
        "holdouts": {"lineage_ids": ["heldout-lineage"]},
    }
    config = MidTrainConfig(max_seq_length=512, general_replay_frac=0.0)
    output = tmp_path / "corpus.jsonl"
    report = build_midtrain_corpus(
        output,
        config,
        source_roots=[
            train,
            heldout,
            {
                "path": unverified,
                "repository_url": "https://example.test/unverified.git",
                "commit": "unverified-commit",
                "license": "MIT",
                "lineage_id": "unverified-lineage",
                "verified": False,
            },
        ],
        task_root=tmp_path / "empty_tasks",
        development_mode=True,
        source_metadata=catalog,
    )
    text = output.read_text()
    assert "train_kernel" in text
    assert "draft_secret" not in text
    assert "heldout_secret" not in text
    assert "unverified_secret" not in text
    assert report["n_excluded_unverified_sources"] == 1
    assert report["n_dropped_source_lineage"] >= 1

    rows = [json.loads(line) for line in text.splitlines()]
    metadata = rows[0]["source_metadata"]
    assert metadata["repository_url"] == "https://example.test/train.git"
    assert metadata["commit"] == "train-commit-1"
    assert metadata["license"] == "MIT"
    assert metadata["path"] == "src/good.py"
    assert metadata["row_id"]
    assert metadata["content_hash"].startswith("sha256:")


def test_heldout_task_root_never_derives_pair(tmp_path):
    task_root = tmp_path / "tasks"
    for task_id, secret in (
        ("rmsnorm_train", "TRAIN_ROOT_MARKER"),
        ("mla_decode_private", "HELDOUT_ROOT_MARKER"),
    ):
        _write_triton(task_root / task_id / "seed_triton.py", f"{task_id}_kernel")
        (task_root / task_id / "reference.py").write_text(
            f"import torch\n\n"
            f"def reference_{task_id}(x):\n"
            f"    value = x.float() + 1\n"
            f"    return value.to(x.dtype)  # {secret}\n"
        )
    config = MidTrainConfig(max_seq_length=512, general_replay_frac=0.0)
    output = tmp_path / "corpus.jsonl"
    build_midtrain_corpus(
        output,
        config,
        source_roots=[],
        task_root=task_root,
        development_mode=True,
    )
    text = output.read_text()
    assert "TRAIN_ROOT_MARKER" in text
    assert "HELDOUT_ROOT_MARKER" not in text
    assert "mla_decode_private" not in text


class _LexicalTokenizer:
    name_or_path = "tests/lexical"
    revision = "test-tokenizer-revision"

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return re.findall(r"[A-Za-z_]+|[^\sA-Za-z_]", text)


def test_token_chunk_admission_uses_supplied_tokenizer():
    text = "alpha beta gamma delta epsilon zeta eta theta iota"
    chunks = chunk_text_tokens(text, 3, _LexicalTokenizer())
    assert len(chunks) > 1
    assert all(len(_LexicalTokenizer.encode(chunk)) <= 3 for chunk in chunks)


def test_external_dataset_loader_passes_pin_and_samples_deterministically(monkeypatch):
    calls = []
    rows = [
        {
            "id": f"row-{index}",
            "python_code": f"def ref_{index}(x):\n    return x + {index}\n",
            "triton_code": (
                "import triton\nimport triton.language as tl\n"
                f"@triton.jit\ndef kernel_{index}(x):\n    return x + {index}\n"
            ),
        }
        for index in range(12)
    ]

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return list(reversed(rows))

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    first = _load_kernelbook_pairs(4, 10_000, revision="deadbeef1234", seed=7)
    second = _load_kernelbook_pairs(4, 10_000, revision="deadbeef1234", seed=7)
    assert first == second
    assert len(first) == 4
    assert all(call[1]["revision"] == "deadbeef1234" for call in calls)
    with pytest.raises(ValueError, match="pinned revision"):
        _load_kernelbook_pairs(1, 100, revision="main")


def test_replay_requests_replacements_after_dedup_underfill(tmp_path):
    repo = tmp_path / "repo"
    _write_triton(repo / "kernel.py")
    calls: dict[str, int] = {}

    def replay_loader(kind, n, seed=0, use_hf=False):
        del seed, use_hf
        calls[kind] = calls.get(kind, 0) + 1
        if calls[kind] == 1:
            # Every kind initially returns the same row. After the first kind,
            # cross-kind replay dedup must request replacements.
            values = ["shared duplicate"] * n
        else:
            values = [f"replacement {kind} {calls[kind]} {index}" for index in range(n)]
        return [{
            "messages": [
                {"role": "user", "content": value},
                {"role": "assistant", "content": f"answer for {value}"},
            ]
        } for value in values]

    report = build_midtrain_corpus(
        tmp_path / "corpus.jsonl",
        MidTrainConfig(max_seq_length=512, general_replay_frac=0.75),
        source_roots=[repo],
        task_root=tmp_path / "tasks",
        development_mode=True,
        replay_loader=replay_loader,
    )
    assert report["general_target"] >= 3
    assert report["replay_replacement_requests"] > 0
    assert report["general_underfill"] == 0
