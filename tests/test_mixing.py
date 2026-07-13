"""CPU-only tests for the multi-capability SFT mixture.

No GPU, no teacher model, no torch/transformers/vllm/network. HF loading is
never triggered (use_hf=False or monkeypatched) so everything runs offline.
"""

from __future__ import annotations

import random

import pytest

from kore.policy.configs import MultiCapSFTConfig, MidTrainConfig
from kore.data import general_replay
from kore.data.general_replay import (
    REPLAY_KINDS,
    load_bundled_samples,
    load_general_replay,
)
from kore.data.gen_qa import generate_kernel_qa, KERNEL_QA_TYPES
from kore.data.teacher import StubTeacher
from kore.data.mixing import (
    SOURCE_FRACTION_FIELDS,
    target_fractions,
    normalized_fractions,
    allocate_counts,
    build_multicap_sft,
    mixture_report,
    build_midtrain_corpus,
    dedup_rows,
    SOURCE_TAG_KEY,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _chat_rows(tag: str, n: int) -> list[dict]:
    return [
        {"messages": [
            {"role": "user", "content": f"{tag} question {i}"},
            {"role": "assistant", "content": f"{tag} answer {i}"},
        ]}
        for i in range(n)
    ]


def _all_sources(n: int = 500) -> dict[str, list]:
    return {key: _chat_rows(key, n) for key in SOURCE_FRACTION_FIELDS}


def _wellformed(row: dict) -> bool:
    if "messages" not in row or not row["messages"]:
        return False
    return all(
        isinstance(m, dict) and "role" in m and "content" in m
        for m in row["messages"]
    )


# --------------------------------------------------------------------------- #
# general_replay fallback
# --------------------------------------------------------------------------- #
def test_general_replay_fallback_loads_bundled_for_all_kinds():
    for kind in REPLAY_KINDS:
        bundled = load_bundled_samples(kind)
        assert len(bundled) >= 15, f"{kind} should bundle >=15 samples"
        rows = load_general_replay(kind, 10, seed=1, use_hf=False)
        assert len(rows) == 10
        for r in rows:
            assert _wellformed(r)
            assert r[SOURCE_TAG_KEY] == kind


def test_general_replay_oversamples_to_exact_n():
    # request more than the bundled pool -> deterministic oversample to exactly n
    n = 40
    rows = load_general_replay("math", n, seed=3, use_hf=False)
    assert len(rows) == n
    assert all(_wellformed(r) for r in rows)


def test_general_replay_determinism_same_seed():
    a = load_general_replay("chat", 12, seed=7, use_hf=False)
    b = load_general_replay("chat", 12, seed=7, use_hf=False)
    assert a == b
    c = load_general_replay("chat", 12, seed=8, use_hf=False)
    # different seed almost surely reorders/reselects
    assert a != c


def test_general_replay_unknown_kind_raises():
    with pytest.raises(ValueError):
        load_general_replay("not_a_kind", 5, use_hf=False)


def test_general_replay_hf_failure_falls_back(monkeypatch):
    # Force the HF path on, but make it fail -> must degrade to bundled samples.
    def _boom(kind, n, seed):
        raise RuntimeError("simulated offline")

    monkeypatch.setattr(general_replay, "_load_from_hf", _boom)
    rows = load_general_replay("code", 8, seed=0, use_hf=True)
    assert len(rows) == 8
    assert all(r[SOURCE_TAG_KEY] == "code" for r in rows)


def test_general_replay_use_hf_false_never_calls_hf(monkeypatch):
    def _boom(kind, n, seed):
        raise AssertionError("HF must not be called when use_hf=False")

    monkeypatch.setattr(general_replay, "_load_from_hf", _boom)
    rows = load_general_replay("tool_use", 5, seed=0, use_hf=False)
    assert len(rows) == 5


# --------------------------------------------------------------------------- #
# gen_qa
# --------------------------------------------------------------------------- #
def _qa_tasks() -> list[dict]:
    return [
        {"task_id": "gemm_fp8_a8w8", "operation": "gemm_fp8",
         "dtype": "fp8_e4m3fnuz", "gpu_target": "gfx942"},
        {"task_id": "rmsnorm_aiter", "operation": "rmsnorm",
         "dtype": "bf16", "gpu_target": "gfx942"},
    ]


def test_generate_kernel_qa_wellformed():
    rows = generate_kernel_qa(_qa_tasks(), StubTeacher(), 12, seed=0)
    assert len(rows) == 12
    for r in rows:
        assert _wellformed(r)
        assert r[SOURCE_TAG_KEY] == "kernel_qa"
        assert r["messages"][0]["role"] == "system"
        assert r["messages"][1]["role"] == "user"
        assert r["messages"][-1]["role"] == "assistant"
        assert r["messages"][-1]["content"].strip()
        assert r["_qa_type"] in KERNEL_QA_TYPES


def test_generate_kernel_qa_has_both_think_and_no_think():
    rows = generate_kernel_qa(_qa_tasks(), StubTeacher(), 12, seed=0)
    styles = {r["_style"] for r in rows}
    assert styles == {"think", "no_think"}


def test_generate_kernel_qa_is_grounded_in_task_dtype():
    # fp32_accumulator questions surface the task dtype string.
    rows = generate_kernel_qa(
        _qa_tasks(), StubTeacher(), 4, seed=0, qa_types=["fp32_accumulator"]
    )
    assert rows
    joined = " ".join(r["messages"][1]["content"] for r in rows)
    assert "fp8_e4m3fnuz" in joined or "bf16" in joined


def test_generate_kernel_qa_uses_teacher_answer():
    teacher = StubTeacher(fn=lambda msgs: "MY_CUSTOM_ANSWER")
    rows = generate_kernel_qa(_qa_tasks(), teacher, 3, seed=1)
    # Core invariant: every QA row carries the TEACHER's answer (not a template).
    assert all(r["messages"][-1]["content"] == "MY_CUSTOM_ANSWER" for r in rows)
    assert len(rows) == 3
    # gen_qa over-provisions a small buffer (n*1.15 + 8) to absorb empty teacher
    # answers, so it makes >= n concurrent teacher calls (exact count is an impl
    # detail; ~15% overhead at production scale). Just require at least n.
    assert len(teacher.calls) >= 3


def test_generate_kernel_qa_determinism():
    a = generate_kernel_qa(_qa_tasks(), StubTeacher(), 10, seed=5)
    b = generate_kernel_qa(_qa_tasks(), StubTeacher(), 10, seed=5)
    assert a == b


def test_generate_kernel_qa_empty_and_no_tasks():
    assert generate_kernel_qa(_qa_tasks(), StubTeacher(), 0, seed=0) == []
    with pytest.raises(ValueError):
        generate_kernel_qa([], StubTeacher(), 5, seed=0)


# --------------------------------------------------------------------------- #
# target fractions / allocation
# --------------------------------------------------------------------------- #
def test_target_fractions_sum_to_one():
    fr = target_fractions(MultiCapSFTConfig())
    assert set(fr) == set(SOURCE_FRACTION_FIELDS)
    assert abs(sum(fr.values()) - 1.0) < 1e-9


def test_allocate_counts_hits_target_within_tolerance():
    targets = target_fractions(MultiCapSFTConfig())
    available = {k: 100000 for k in targets}
    total = 1000
    counts = allocate_counts(available, targets, total)
    assert sum(counts.values()) == total
    for k, frac in targets.items():
        assert abs(counts[k] / total - frac) <= 0.01


def test_allocate_counts_short_source_renormalizes():
    targets = target_fractions(MultiCapSFTConfig())
    available = {k: 100000 for k in targets}
    available["kernel_qa"] = 5  # short: target*total would be 100
    total = 1000
    counts = allocate_counts(available, targets, total)
    # short source is fully consumed but not exceeded
    assert counts["kernel_qa"] == 5
    # deficit redistributed -> still allocate the full budget
    assert sum(counts.values()) == total
    # other sources grew above their nominal target counts
    assert counts["kernel_repair_opt"] > int(targets["kernel_repair_opt"] * total)


def test_allocate_counts_capped_by_capacity():
    targets = target_fractions(MultiCapSFTConfig())
    available = {k: 10 for k in targets}  # capacity 60 < total
    counts = allocate_counts(available, targets, total=1000)
    assert sum(counts.values()) == 60
    assert all(counts[k] <= available[k] for k in counts)


def test_normalized_fractions_all_zero_is_uniform():
    nf = normalized_fractions({"a": 0.0, "b": 0.0})
    assert abs(nf["a"] - 0.5) < 1e-9 and abs(nf["b"] - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# build_multicap_sft
# --------------------------------------------------------------------------- #
def test_build_multicap_sft_fractions_within_tolerance_and_sum_one():
    cfg = MultiCapSFTConfig()
    mix = build_multicap_sft(_all_sources(1000), cfg, total=1000, seed=0)
    rep = mixture_report(mix)
    assert rep["total"] == 1000
    assert abs(sum(rep["fractions"].values()) - 1.0) < 1e-9
    tgt = target_fractions(cfg)
    for k, frac in tgt.items():
        assert abs(rep["fractions"].get(k, 0.0) - frac) <= 0.02


def test_build_multicap_sft_tags_every_row():
    mix = build_multicap_sft(_all_sources(200), MultiCapSFTConfig(), total=100, seed=0)
    valid = set(SOURCE_FRACTION_FIELDS)
    assert all(row[SOURCE_TAG_KEY] in valid for row in mix)
    assert all(_wellformed(row) for row in mix)


def test_build_multicap_sft_short_source_behaves():
    sources = _all_sources(500)
    sources["kernel_qa"] = _chat_rows("kernel_qa", 3)
    mix = build_multicap_sft(sources, MultiCapSFTConfig(), total=200, seed=0)
    rep = mixture_report(mix)
    assert rep["counts"]["kernel_qa"] == 3  # capped at what's available
    assert rep["total"] == 200  # deficit redistributed to fill the budget


def test_build_multicap_sft_determinism():
    sources = _all_sources(400)
    a = build_multicap_sft(sources, MultiCapSFTConfig(), total=150, seed=42)
    b = build_multicap_sft(sources, MultiCapSFTConfig(), total=150, seed=42)
    assert a == b
    c = build_multicap_sft(sources, MultiCapSFTConfig(), total=150, seed=43)
    assert a != c  # different seed -> different shuffle/sample


def test_build_multicap_sft_dedups_within_source():
    # a source full of identical rows collapses to a single usable row
    dupes = [{"messages": [{"role": "user", "content": "same"},
                           {"role": "assistant", "content": "same"}]}
             for _ in range(50)]
    sources = _all_sources(500)
    sources["kernel_qa"] = dupes
    mix = build_multicap_sft(sources, MultiCapSFTConfig(), total=200, seed=0)
    kq = [r for r in mix if r[SOURCE_TAG_KEY] == "kernel_qa"]
    assert len(kq) == 1


def test_build_multicap_sft_ignores_unknown_source_keys():
    sources = _all_sources(200)
    sources["totally_unknown"] = _chat_rows("junk", 100)
    mix = build_multicap_sft(sources, MultiCapSFTConfig(), total=100, seed=0)
    assert all(row[SOURCE_TAG_KEY] != "totally_unknown" for row in mix)


# --------------------------------------------------------------------------- #
# mixture_report
# --------------------------------------------------------------------------- #
def test_mixture_report_counts_and_fractions():
    rows = (
        [{"messages": [], SOURCE_TAG_KEY: "a"} for _ in range(3)]
        + [{"messages": [], SOURCE_TAG_KEY: "b"} for _ in range(1)]
    )
    rep = mixture_report(rows)
    assert rep["total"] == 4
    assert rep["counts"] == {"a": 3, "b": 1}
    assert abs(rep["fractions"]["a"] - 0.75) < 1e-9
    assert abs(rep["fractions"]["b"] - 0.25) < 1e-9


def test_mixture_report_empty():
    rep = mixture_report([])
    assert rep == {"total": 0, "counts": {}, "fractions": {}}


def test_dedup_rows_content_hash_ignores_source_tag():
    r1 = {"messages": [{"role": "user", "content": "x"}], SOURCE_TAG_KEY: "a"}
    r2 = {"messages": [{"role": "user", "content": "x"}], SOURCE_TAG_KEY: "b"}
    out = dedup_rows([r1, r2])
    assert len(out) == 1  # same chat content -> deduped despite differing tag


# --------------------------------------------------------------------------- #
# build_midtrain_corpus
# --------------------------------------------------------------------------- #
def test_build_midtrain_corpus_general_fraction():
    cfg = MidTrainConfig()  # general_replay_frac = 0.15
    kernel = [f"kernel doc {i}" for i in range(100)]
    general = [f"general shard {i}" for i in range(100)]
    mix = build_midtrain_corpus(kernel, general, cfg, seed=0)
    rep = mixture_report(mix)
    frac = rep["fractions"].get("general_shard", 0.0)
    assert abs(frac - cfg.general_replay_frac) <= 0.03
    assert rep["counts"]["kernel_corpus"] == 100
    # docs normalized to dicts with a text field + source tag
    assert all(isinstance(d, dict) and SOURCE_TAG_KEY in d for d in mix)


def test_build_midtrain_corpus_caps_at_available_shards():
    cfg = MidTrainConfig()
    kernel = [f"k{i}" for i in range(100)]
    general = [f"g{i}" for i in range(2)]  # fewer than target
    mix = build_midtrain_corpus(kernel, general, cfg, seed=0)
    rep = mixture_report(mix)
    assert rep["counts"].get("general_shard", 0) == 2


def test_build_midtrain_corpus_determinism():
    cfg = MidTrainConfig()
    kernel = [{"text": f"k{i}"} for i in range(50)]
    general = [{"text": f"g{i}"} for i in range(50)]
    a = build_midtrain_corpus(kernel, general, cfg, seed=1)
    b = build_midtrain_corpus(kernel, general, cfg, seed=1)
    assert a == b
