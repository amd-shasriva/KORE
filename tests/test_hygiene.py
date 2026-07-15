"""Pillar 5 - near-dup dedup + eval decontamination."""

from __future__ import annotations

from kore.data.dedup import (
    dedup_near,
    jaccard,
    minhash_signature,
    structural_fingerprint,
)
from kore.data import decontam as D


K_A = "import triton.language as tl\ndef k(x):\n    # load it\n    a = tl.load(x)\n    return a + 1"
K_B = "import triton.language as tl\ndef k(y):\n\n    z = tl.load(y)   # renamed + comment\n    return z + 1"
K_C = "import triton.language as tl\ndef k(x):\n    a = tl.store(x)\n    return a + 1"


def test_structural_fingerprint_rename_comment_invariant():
    assert structural_fingerprint(K_A) == structural_fingerprint(K_B)   # type-2 clones
    assert structural_fingerprint(K_A) != structural_fingerprint(K_C)   # load != store


def test_structural_fingerprint_non_python_fallback():
    fp = structural_fingerprint("__global__ void k(){ int i = 0; }")
    assert fp.startswith("txt:")


def test_dedup_near_keeps_best_representative():
    items = [{"source": K_A, "sp": 1.1}, {"source": K_B, "sp": 2.5}, {"source": K_C, "sp": 1.0}]
    kept, stats = dedup_near(items, scorer=lambda d: d["sp"])
    assert stats["n_in"] == 3 and stats["n_kept"] == 2  # {A,B}->1 + C
    kept_a = [k for k in kept if structural_fingerprint(k["source"]) == structural_fingerprint(K_A)]
    assert kept_a and kept_a[0]["sp"] == 2.5  # kept the faster of the clones


def test_dedup_near_fuzzy_merge():
    near1 = "def k(x):\n    a = tl.load(x)\n    b = a * 2\n    return b + 1"
    near2 = "def k(x):\n    a = tl.load(x)\n    b = a * 2\n    return b + 2"  # 1 token diff
    s1, s2 = minhash_signature(near1), minhash_signature(near2)
    assert jaccard(s1, s2) > 0.5
    kept, stats = dedup_near([{"source": near1}, {"source": near2}], fuzzy_threshold=0.5)
    assert stats["n_kept"] == 1  # fuzzy-merged


def test_decontam_record_family_gate():
    # Core attention now TRAINS (product capability) -> NOT contaminated. Only the
    # held-out TASKS (paged-KV decode + MLA) are gated (registry HELDOUT_TASKS).
    assert not D.is_contaminated_record({"task_id": "flash_attn_prefill_bf16"})
    assert D.is_contaminated_record({"task_id": "paged_attn_decode_bf16"})
    assert D.is_contaminated_record({"task_id": "mla_decode_bf16"})
    assert not D.is_contaminated_record({"task_id": "gemm_bf16", "operation": "gemm"})
    clean, st = D.decontaminate_records(
        [{"task_id": "gemm_bf16"}, {"task_id": "paged_attn_decode_bf16"},
         {"task_id": "mla_decode_bf16"}, {"task_id": "flash_attn_decode_bf16"}])
    assert st["n_dropped_heldout"] == 2 and st["n_kept"] == 2  # paged+mla dropped; gemm+flash train


def test_decontam_text_ngram_containment():
    ref = "the quick brown fox jumps over the lazy dog while the cat sleeps"
    grams = D.ngram_set(ref, n=4)
    assert D.contaminated_by_text(ref, grams, n=4, threshold=0.5)
    assert not D.contaminated_by_text("completely unrelated tokens here now", grams, n=4, threshold=0.5)
    # empty held-out -> safe no-op
    assert not D.contaminated_by_text(ref, set(), n=4)


def test_decontaminate_corpus_safe_noop_without_heldout(monkeypatch):
    monkeypatch.setattr(D, "build_heldout_ngrams", lambda n=8, extra_sources=None: set())
    rows = [{"text": "anything"}, {"text": "else"}]
    clean, st = D.decontaminate_corpus(rows)
    assert st["n_dropped_contaminated"] == 0 and len(clean) == 2


def test_dedup_near_source_on_win_records_keeps_fastest():
    from kore.data.build_datasets import dedup_near_source
    from kore.data.schemas import WinRecord

    def _win(src, sp):
        return WinRecord(task_id="t", trajectory=[{"role": "assistant", "content": src}],
                         initial_wall_us=10.0, final_wall_us=10.0 / sp, speedup=sp,
                         final_source=src, snr_db=99.0, type="win", gpu="gfx942",
                         operation="gemm", arch="gfx942")

    recs = [_win(K_A, 1.1), _win(K_B, 3.0), _win(K_C, 1.0)]  # A,B clones
    kept = dedup_near_source(recs, per_fingerprint_cap=1)
    assert len(kept) == 2
    speeds = sorted(r.speedup for r in kept)
    assert speeds == [1.0, 3.0]  # kept the faster clone (3.0), dropped the 1.1
