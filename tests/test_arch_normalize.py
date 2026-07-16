"""Tests for the build-time legacy-arch normalizer (kore.data.arch_normalize).

Pre-gfx950 records baked gfx942 / CDNA3 / MI325X / MI300X (and the gfx942 fp8
encoding e4m3fnuz) into their training text. The build stage rewrites those to the
gfx950 / MI350X / OCP target on the way into training. These tests pin that
contract: the arch labels are always rewritten, the fp8 dtype is rewritten except
inside a two-encoding repair fix-lesson, and the pass is idempotent and
non-mutating.
"""
from kore.data.arch_normalize import normalize_obj, normalize_rows, normalize_text


def test_arch_labels_rewrite_to_gfx950_target():
    assert normalize_text("compile for gfx942") == "compile for gfx950"
    assert normalize_text("AMD Instinct MI325X") == "AMD Instinct MI350X"
    assert normalize_text("the MI300X board") == "the MI350X board"
    assert normalize_text("CDNA3 microarch") == "CDNA4 microarch"
    assert normalize_text("cdna3 lowercase") == "cdna4 lowercase"


def test_compound_system_prompt_phrase():
    out = normalize_text("kernels for AMD Instinct MI325X (CDNA3, gfx942).")
    assert "MI350X" in out and "CDNA4" in out and "gfx950" in out
    assert "MI325X" not in out and "CDNA3" not in out and "gfx942" not in out


def test_fnuz_fp8_text_left_verbatim():
    # FNUZ is the legacy gfx942 fp8 encoding; any text mentioning it is arch-specific
    # and left verbatim, because rewriting the dtype or the co-located arch label would
    # manufacture a false "gfx950 uses FNUZ" fact or corrupt a real fp8 kernel's dtype.
    assert normalize_text("XQ: [M, K] fp8 e4m3fnuz") == "XQ: [M, K] fp8 e4m3fnuz"
    assert normalize_text("e5m2fnuz accum on gfx942") == "e5m2fnuz accum on gfx942"


def test_fnuz_fix_lesson_and_arch_preserved_together():
    # A repair fix-lesson names FNUZ; the whole line (both encodings AND the co-located
    # arch label) stays verbatim, so the lesson is coherent and no false fact is made.
    lesson = "the fp8 was `e4m3fnuz` (FNUZ) instead of `e4m3fn` (OCP) on gfx942 / CDNA3"
    assert normalize_text(lesson) == lesson


def test_target_and_unrelated_text_untouched():
    assert normalize_text("already gfx950 / CDNA4 / MI350X") == (
        "already gfx950 / CDNA4 / MI350X")
    assert normalize_text("MI355X is already the target") == "MI355X is already the target"
    assert normalize_text("shape K=4096, tile 128") == "shape K=4096, tile 128"


def test_idempotent():
    src = "AMD Instinct MI325X (CDNA3, gfx942) fp8 e4m3fnuz"
    once = normalize_text(src)
    assert normalize_text(once) == once


def test_multi_arch_comparison_left_verbatim():
    # A deliberate legacy-vs-target comparison (e.g. the aiter_ref infra docstrings)
    # must be left verbatim, not collapsed into a self-contradiction. This is the
    # audit's "gfx950 uses FNUZ e4m3fn range 240" corruption case.
    src = ("gfx942 / CDNA3 (MI300X): FNUZ e4m3fnuz, range +/-240; "
           "gfx950 / CDNA4 (MI350X): OCP e4m3fn, range +/-448.")
    assert normalize_text(src) == src            # both arches named -> unchanged
    # single-arch stale text is still rewritten to the target
    assert normalize_text("optimize for gfx942 (CDNA3)") == "optimize for gfx950 (CDNA4)"


def test_normalize_obj_recurses_and_preserves_non_strings():
    obj = {
        "messages": [{"role": "system", "content": "target MI325X (gfx942)"}],
        "speedup": 1.42,
        "count": 304,          # a bare int, not an arch token: must be preserved
        "flags": [True, None],
    }
    out = normalize_obj(obj)
    assert out["messages"][0]["content"] == "target MI350X (gfx950)"
    assert out["messages"][0]["role"] == "system"
    assert out["speedup"] == 1.42 and out["count"] == 304
    assert out["flags"] == [True, None]


def test_normalize_rows_scrubs_sft_and_dpo_rows():
    sft = {"messages": [{"role": "user", "content": "kernel on gfx942 / CDNA3"}]}
    dpo = {
        "prompt": [{"role": "user", "content": "optimize for MI300X"}],
        "chosen": [{"role": "assistant", "content": "# gfx942-safe"}],
        "rejected": [{"role": "assistant", "content": "# CDNA3"}],
    }
    out = normalize_rows([sft, dpo])
    assert out[0]["messages"][0]["content"] == "kernel on gfx950 / CDNA4"
    assert out[1]["prompt"][0]["content"] == "optimize for MI350X"
    assert out[1]["chosen"][0]["content"] == "# gfx950-safe"
    assert out[1]["rejected"][0]["content"] == "# CDNA4"
