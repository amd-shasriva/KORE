"""CPU-only tests for the HipKittens SFT ingestion.

The tests that matter here are the ones that catch a SILENT wrong answer, not the
ones that check a row count. Three specific failures this module is capable of,
each of which would produce a corpus that looks fine:

  * a swizzle parsed as the identity because the upstream formula shape changed
    (this actually happened during development: ``st_16x128`` parenthesises its
    modulus as ``(16*128)``, and the first parser silently returned zero XOR
    terms, which would have taught an identity swizzle for fp8);
  * a kernel labelled with the wrong operation or the wrong schedule because a
    substring matched (``attn_bkwd_non_causal`` contains ``causal``);
  * HipKittens C++ leaking into the corpus under the ``FULL_KERNEL`` contract,
    which would train the model to emit code the eval harness cannot compile.

Most tests need a real checkout and skip without one, but the parser, gating and
contract tests run against synthetic fixtures so they hold in CI with no clone.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from kore.data import hipkittens as hk


# --------------------------------------------------------------------------- #
# Fixtures: a synthetic checkout, so the failure-mode tests need no clone
# --------------------------------------------------------------------------- #
_ST_SHAPE_OK = """
#pragma once
namespace kittens { namespace ducks { namespace st_shape {

struct st_16x32 {
    static constexpr int rows = 16;
    static constexpr int cols = 32;

    template<typename _T>
    static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 2 || sizeof(_T) == 4) {
            return 16;
        } else {
            static_assert(false, "Unsupported type");
        }
    }

    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        using T = _T;
        const uint32_t offset = sizeof(T)*(r*cols + c);
        if constexpr (sizeof(T) == 2) {
            const int swizzle = ((offset % 1024) >> 9) << 5;
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        } else if constexpr (sizeof(T) == 4) {
            return offset;
        } else {
            static_assert(false, "Unsupported type");
        }
    }
};

} } }
"""

# Same layout, but the modulus is written as a parenthesised product. A parser
# that only accepts a bare integer silently reports NO xor terms here.
_ST_SHAPE_PAREN_MOD = _ST_SHAPE_OK.replace(
    "((offset % 1024) >> 9) << 5", "((offset % (32*32)) >> 9) << 5"
)

# An XOR whose shape the parser does not recognise at all: it must raise rather
# than fall back to the identity.
_ST_SHAPE_UNPARSEABLE = _ST_SHAPE_OK.replace(
    "const int swizzle = ((offset % 1024) >> 9) << 5;",
    "const int swizzle = rotate_left(offset, 5) & 0x3ff;",
)

# A swizzle that is NOT a bijection: bit 5 is set unconditionally, so two
# distinct elements map to one address. Must be rejected.
_ST_SHAPE_NOT_BIJECTIVE = _ST_SHAPE_OK.replace(
    "const int swizzle = ((offset % 1024) >> 9) << 5;",
    "const int swizzle = ((offset % 1024) >> 0) << 5;",
)

_MIT = """MIT License

Copyright (c) 2024 HazyResearch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED.
"""

_PHASES = """Phase Assignment Results
============================================================

Phase 0: 2 threads - [0, 1]
Phase 1: 2 threads - [2, 3]

Total phases: 2
"""

_BANKS = """LDS Bank Detection Results (ds_read_b128)
======================================================================

Number of LDS banks: 64
"""


def _make_checkout(tmp_path: pathlib.Path, st_shape: str = _ST_SHAPE_OK,
                   license_text: str = _MIT) -> pathlib.Path:
    root = tmp_path / "HipKittens"
    (root / "include" / "cdna4" / "types" / "shared").mkdir(parents=True)
    (root / "include" / "cdna4" / "types" / "shared" / "st_shape.cuh").write_text(st_shape)
    (root / "LICENSE").write_text(license_text)
    ph = root / "analysis" / "paper_experiments" / "phases" / "ds_read_b128"
    ph.mkdir(parents=True)
    (ph / "phase_results.txt").write_text(_PHASES)
    (ph / "bank_results.txt").write_text(_BANKS)
    kd = root / "kernels" / "cdna4" / "gemm"
    kd.mkdir(parents=True)
    (kd / "k.cpp").write_text(
        "#define NUM_WARPS 8\n"
        "__global__ void micro_tk() {\n"
        "  int warp_row = warpid() / 4;\n"
        "  if (warp_row == 1) { __builtin_amdgcn_s_barrier(); }\n"
        "  for (int t = 0; t < N; ++t) {\n"
        "    __builtin_amdgcn_s_setprio(1);\n"
        "    mma_ABt(c, a, b, c);\n"
        "    __builtin_amdgcn_s_setprio(0);\n"
        "  }\n"
        "}\n"
    )
    return root


# --------------------------------------------------------------------------- #
# Locating / licensing the checkout
# --------------------------------------------------------------------------- #
def test_missing_checkout_raises_with_actionable_message(tmp_path):
    with pytest.raises(hk.HipKittensIngestError) as exc:
        hk.hk_root(tmp_path / "nope")
    assert "git clone" in str(exc.value)


def test_absent_license_refuses_to_ingest(tmp_path):
    """An unlicensed checkout must not become training data."""
    root = _make_checkout(tmp_path)
    (root / "LICENSE").unlink()
    with pytest.raises(hk.HipKittensIngestError, match="unlicensed"):
        hk.provenance(root)


def test_non_mit_license_refuses_to_ingest(tmp_path):
    """The licence is verified, not assumed from a constant in our own module."""
    root = _make_checkout(tmp_path, license_text="All rights reserved. Proprietary.")
    with pytest.raises(hk.HipKittensIngestError, match="MIT"):
        hk.provenance(root)


def test_provenance_records_attribution(tmp_path):
    prov = hk.provenance(_make_checkout(tmp_path))
    assert prov["license"] == "MIT"
    assert "HazyResearch" in prov["license_holder"]
    assert "arXiv:2511.08083" in prov["paper"]
    assert prov["license_file_sha256"]


# --------------------------------------------------------------------------- #
# Swizzle parsing: the silent-wrong-answer failure modes
# --------------------------------------------------------------------------- #
def test_parses_bare_integer_modulus(tmp_path):
    layouts = hk.parse_swizzles(_make_checkout(tmp_path))
    bf16 = [l for l in layouts if l.dtype_bytes == 2]
    assert len(bf16) == 1
    assert bf16[0].terms == ((1024, 9, 5),)
    assert bf16[0].is_bijection()


def test_parenthesised_modulus_is_not_silently_dropped(tmp_path):
    """`offset % (32*32)` must parse to 1024, not to "no swizzle".

    This is the regression that motivated the fail-loud design: an identity
    swizzle is a plausible-looking wrong answer that a row-count test cannot see.
    """
    root = _make_checkout(tmp_path, st_shape=_ST_SHAPE_PAREN_MOD)
    layouts = hk.parse_swizzles(root)
    bf16 = [l for l in layouts if l.dtype_bytes == 2]
    assert bf16[0].terms == ((1024, 9, 5),), "parenthesised modulus lost"
    assert not bf16[0].is_identity


def test_unrecognised_swizzle_shape_raises_instead_of_returning_identity(tmp_path):
    root = _make_checkout(tmp_path, st_shape=_ST_SHAPE_UNPARSEABLE)
    with pytest.raises(hk.HipKittensIngestError, match="contains '\\^' but no"):
        hk.parse_swizzles(root)


def test_non_bijective_swizzle_is_rejected(tmp_path):
    """A swizzle that aliases two elements corrupts data; refuse to teach it."""
    root = _make_checkout(tmp_path, st_shape=_ST_SHAPE_NOT_BIJECTIVE)
    with pytest.raises(hk.HipKittensIngestError, match="bijection"):
        hk.parse_swizzles(root)


def test_missing_st_shape_file_raises(tmp_path):
    root = _make_checkout(tmp_path)
    (root / "include" / "cdna4" / "types" / "shared" / "st_shape.cuh").unlink()
    with pytest.raises(hk.HipKittensIngestError, match="missing"):
        hk.parse_swizzles(root)


# --------------------------------------------------------------------------- #
# Phase model parsing
# --------------------------------------------------------------------------- #
def test_phase_model_parsed(tmp_path):
    models = hk.parse_phase_models(_make_checkout(tmp_path))
    assert len(models) == 1
    m = models[0]
    assert m.instruction == "ds_read_b128"
    assert m.num_banks == 64
    assert m.phases == ((0, 1), (2, 3))
    assert m.contiguous_phases is True


def test_ragged_phase_output_raises(tmp_path):
    """Phases of differing sizes mean the solver output format changed."""
    root = _make_checkout(tmp_path)
    p = (root / "analysis" / "paper_experiments" / "phases" / "ds_read_b128"
         / "phase_results.txt")
    p.write_text("Phase 0: 3 threads - [0, 1, 2]\nPhase 1: 1 threads - [3]\n")
    with pytest.raises(hk.HipKittensIngestError, match="differing sizes"):
        hk.parse_phase_models(root)


def test_missing_bank_count_raises(tmp_path):
    root = _make_checkout(tmp_path)
    p = (root / "analysis" / "paper_experiments" / "phases" / "ds_read_b128"
         / "bank_results.txt")
    p.write_text("no bank count here")
    with pytest.raises(hk.HipKittensIngestError, match="bank count"):
        hk.parse_phase_models(root)


# --------------------------------------------------------------------------- #
# Conflict model
# --------------------------------------------------------------------------- #
def test_conflict_degree_counts_only_within_a_phase():
    """Lanes in different phases share banks for free; same-phase lanes do not."""
    phases = ((0, 1), (2, 3))
    # Lanes 0 and 2 both start at bank 0 but are in DIFFERENT phases -> no conflict.
    assert hk._max_conflict_degree([0, 4, 0, 4], phases, 64, 1) == 1
    # Lanes 0 and 1 both at bank 0, SAME phase -> 2-way conflict.
    assert hk._max_conflict_degree([0, 0, 4, 8], phases, 64, 1) == 2


def test_wide_access_occupies_consecutive_banks():
    """A b128 lane claims 4 banks, so lanes one bank apart still overlap.

    The degree is the number of LANES contending for the worst bank (2 here),
    not the number of banks they happen to share.
    """
    phases = ((0, 1),)
    assert hk._max_conflict_degree([0, 4], phases, 64, 4) == 2
    # 4 banks apart: {0,1,2,3} vs {4,5,6,7} -> disjoint, conflict-free.
    assert hk._max_conflict_degree([0, 16], phases, 64, 4) == 1


def test_conflict_degree_wraps_at_bank_count():
    phases = ((0, 1),)
    # offset 0 -> bank 0; offset 256 -> bank 64 % 64 == 0. Same bank by wraparound.
    assert hk._max_conflict_degree([0, 256], phases, 64, 1) == 2


# --------------------------------------------------------------------------- #
# Kernel characterisation: the substring-match failure modes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filename,expected", [
    ("attn_bkwd_non_causal.cpp", "attention_backward"),
    ("attn_bkwd_causal.cpp", "attention_backward_causal"),
    ("attn_fwd_non_causal.cpp", "attention_forward"),
    ("attn_fwd_causal.cpp", "attention_forward_causal"),
    ("attn_bkwd_prep.cpp", "attention_backward_prep"),
])
def test_non_causal_is_not_read_as_causal(filename, expected):
    """"non_causal" CONTAINS "causal"; the naive pattern mislabels it."""
    op, _ = hk._detect_op(f"kernels/cdna4/attn/gqa_backwards/{filename}", "")
    assert op == expected


def test_filename_beats_directory():
    """A forward kernel living in a `_backwards/` directory is still forward."""
    op, _ = hk._detect_op(
        "kernels/cdna4/attn/gqa_causal_backwards/attn_fwd_causal.cpp", "")
    assert op == "attention_forward_causal"


@pytest.mark.parametrize("src,expected", [
    # Braced conditional barrier.
    ("#define NUM_WARPS 8\nif (warp_row == 1) { __builtin_amdgcn_s_barrier(); }",
     "8-wave ping-pong"),
    # Brace-LESS conditional barrier, as the MXFP8 kernels write it.
    ("#define NUM_WARPS 8\nif (warp_m == 1) __builtin_amdgcn_s_barrier();",
     "8-wave ping-pong"),
    # 8 waves but no conditional barrier: must NOT claim ping-pong.
    ("#define NUM_WARPS 8\n__builtin_amdgcn_s_barrier();",
     "8-wave (no stagger detected)"),
    # Producer/consumer roles win over the wave count.
    ("#define NUM_WARPS 8\nbool is_producer = (g == 0);\n"
     "if (warp_row == 1) { __builtin_amdgcn_s_barrier(); }",
     "wave-specialized producer/consumer"),
    # No wave count at all: unclassified, never guessed.
    ("__global__ void f() {}", "unclassified"),
])
def test_schedule_detection_is_evidence_based(src, expected):
    assert hk._detect_schedule(src, hk._detect_num_waves(src))[0] == expected


def test_interleave_needs_evidence_not_just_four_waves():
    plain = "#define NUM_WARPS 4\n__builtin_amdgcn_s_barrier();"
    assert hk._detect_schedule(plain, 4)[0] == "4-wave"
    many = "#define NUM_WARPS 4\n" + "__builtin_amdgcn_s_barrier();\n" * 20
    assert hk._detect_schedule(many, 4)[0] == "4-wave interleave"


@pytest.mark.parametrize("src,expected", [
    ("#define NUM_WARPS 8\n", 8),
    ("constexpr int WARPS_M = 2;\nconstexpr int WARPS_N = 4;\n"
     "#define NUM_WARPS (WARPS_M * WARPS_N)\n", 8),
    # The wave-specialized micros never define NUM_WARPS.
    ("#define NUM_PRODUCER_WORKERS (4)\nconstexpr int M_BLOCK = 2;\n"
     "#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)\n", 12),
    # NUM_THREADS in lanes, resolved through kittens::WARP_THREADS == 64.
    ("#define NUM_THREADS (kittens::WARP_THREADS * 4)\n", 4),
    # Unresolvable -> 0, never a guess.
    ("#define NUM_WARPS SOME_EXTERNAL_THING\n", 0),
])
def test_wave_count_resolution(src, expected):
    assert hk._detect_num_waves(src) == expected


def test_udna1_and_archive_are_excluded(tmp_path):
    """gfx1250 idioms do not transfer to gfx950, and archive/ is dead code."""
    root = _make_checkout(tmp_path)
    for rel in ("kernels/udna1/gemm/g.cpp",
                "kernels/cdna4/gemm/archive/old.cpp",
                "kernels/cdna4/gemm/timing/micro_01.cpp"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#define NUM_WARPS 8\n__global__ void f() {}\n")
    found = {k.rel_path for k in hk.discover_kernels(root)}
    assert found == {"kernels/cdna4/gemm/k.cpp"}, found


def test_helper_files_without_a_kernel_are_skipped(tmp_path):
    root = _make_checkout(tmp_path)
    (root / "kernels" / "cdna4" / "gemm" / "utils.cpp").write_text(
        "int helper() { return 0; }\n")
    assert all(not k.rel_path.endswith("utils.cpp") for k in hk.discover_kernels(root))


# --------------------------------------------------------------------------- #
# Row construction and the deployment contract
# --------------------------------------------------------------------------- #
def test_rows_never_use_the_full_kernel_contract(tmp_path):
    """The single most damaging thing this module could do.

    ``kore.policy.format.SYSTEM_PROMPT`` trains the model to emit a compilable
    ROCm/Triton kernel in a ``FULL_KERNEL:`` block. HipKittens kernels include
    ``kittens.cuh``, which does not exist in the eval harness, so an HK row
    wearing that contract would teach the model to emit code that cannot build --
    and it would look like clean data while doing it.
    """
    from kore.policy.format import SYSTEM_PROMPT

    rows, _ = hk.build_rows(_make_checkout(tmp_path))
    assert rows
    for row in rows:
        system = row["messages"][0]["content"]
        assert system != SYSTEM_PROMPT
        assert "FULL_KERNEL" not in system
        assert "FULL_KERNEL:" not in row["messages"][2]["content"]


def test_every_row_carries_mit_provenance(tmp_path):
    rows, _ = hk.build_rows(_make_checkout(tmp_path))
    for row in rows:
        prov = row["_provenance"]
        assert prov["license"] == "MIT"
        assert "HazyResearch" in prov["attribution"]
        assert prov["repository_url"] == hk.HK_REPO_URL
        assert row["_source"] == "hipkittens"
        assert row["_qa_type"].startswith("hk_")


def test_row_shape_matches_the_mixture_gate(tmp_path):
    """Rows must survive build_sft_v3_mixture.admit(): messages, roles, length."""
    rows, _ = hk.build_rows(_make_checkout(tmp_path))
    for row in rows:
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert all(m["content"].strip() for m in row["messages"])
        n_tokens = sum(len(m["content"]) for m in row["messages"]) / hk.CHARS_PER_TOKEN
        assert n_tokens <= 17408


def test_rows_are_deterministic(tmp_path):
    root = _make_checkout(tmp_path)
    a, _ = hk.build_rows(root, seed=0)
    b, _ = hk.build_rows(root, seed=0)
    assert [json.dumps(r, sort_keys=True) for r in a] == \
           [json.dumps(r, sort_keys=True) for r in b]


def test_unknown_family_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown families"):
        hk.build_rows(_make_checkout(tmp_path), families=["not_a_family"])


def test_near_duplicate_gate_drops_clones(tmp_path):
    """Two rows teaching the same lesson must not both survive."""
    base = {"messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": "u " * 40},
                         {"role": "assistant", "content": "a " * 200}],
            "_qa_type": "hk_x"}
    clone = json.loads(json.dumps(base))
    other = json.loads(json.dumps(base))
    other["messages"][2]["content"] = "completely different words here " * 60
    kept, dropped = hk._drop_near_duplicates([base, clone, other], threshold=0.75)
    assert len(kept) == 2 and len(dropped) == 1


def test_over_long_rows_are_dropped_not_truncated(tmp_path):
    """Truncation teaches an answer that stops mid-explanation."""
    rows, stats = hk.build_rows(_make_checkout(tmp_path), max_row_chars=200)
    assert stats["dropped_too_long"] > 0
    for row in rows:
        assert sum(len(m["content"]) for m in row["messages"]) <= 200


def test_no_fabricated_speedup_language(tmp_path):
    """Any multiplier in a row must be attributable.

    Guards the one thing that would make this corpus actively harmful: a
    confident performance claim we invented.
    """
    import re

    rows, _ = hk.build_rows(_make_checkout(tmp_path))
    attributions = ("HipKittens", "arXiv:2511.08083", "reported in", "Measured by",
                    "measured", "computed from")
    for row in rows:
        body = row["messages"][2]["content"]
        if re.search(r"\d+(?:\.\d+)?\s*x\b", body):
            assert any(a in body for a in attributions), (
                f"{row['_qa_type']} states a multiplier with no attribution")


# --------------------------------------------------------------------------- #
# Against the real checkout, when present
# --------------------------------------------------------------------------- #
def _real_root():
    try:
        return hk.hk_root()
    except hk.HipKittensIngestError:
        return None


real = pytest.mark.skipif(_real_root() is None,
                          reason="no HipKittens checkout; run scripts/build_hipkittens_sft.py")


@real
def test_real_swizzles_are_all_bijections():
    for lay in hk.parse_swizzles():
        assert lay.is_bijection(), lay.name


@real
def test_real_swizzles_are_conflict_free_under_the_measured_model():
    """The library's own claim, checked rather than repeated.

    Each non-identity swizzle must take its layout from a real conflict to
    conflict-free, using the authors' measured bank count and phase partition. If
    this fails, either our parse is wrong or upstream changed a formula -- both of
    which must stop the build rather than ship a wrong explanation.
    """
    models = {m.instruction: m for m in hk.parse_phase_models()}
    checked = 0
    for lay in hk.parse_swizzles():
        if lay.is_identity:
            continue
        rep = hk.bank_conflict_report(lay, models)
        assert rep is not None, lay.name
        assert rep.plain_max_conflict > 1, f"{lay.name}: nothing to fix?"
        assert rep.conflict_free, (
            f"{lay.name} @ {lay.dtype_bytes}B: swizzled conflict degree "
            f"{rep.swizzled_max_conflict}, expected 1")
        checked += 1
    assert checked >= 5, f"only checked {checked} swizzles"


@real
def test_real_lds_read_write_asymmetry_holds():
    """Reads and writes measure different bank counts; the corpus teaches this."""
    models = {m.instruction: m for m in hk.parse_phase_models()}
    assert models["ds_read_b64"].num_banks == 64
    assert models["ds_write_b64"].num_banks == 32
    assert models["ds_read_b128"].num_banks == 64
    assert not models["ds_read_b128"].contiguous_phases
    for m in models.values():
        flat = sorted(x for p in m.phases for x in p)
        assert flat == list(range(64)), f"{m.instruction} phases do not cover 64 lanes"


@real
def test_real_schedule_split_matches_the_paper():
    """Labels are derived from source evidence; they should reproduce the paper's
    split without having been told it: ping-pong for GEMM and attention forward,
    4-wave interleave for the imbalanced backward."""
    kernels = hk.discover_kernels()
    sched = {k.rel_path: k.schedule for k in kernels}
    fwd = [k for k in kernels if k.op.startswith("attention_forward")]
    bwd = [k for k in kernels
           if k.op in ("attention_backward", "attention_backward_causal")]
    gemm = [k for k in kernels if k.op.startswith("gemm")
            and "producer_consumer" not in k.rel_path
            and k.num_waves == 8]
    assert fwd and all(k.schedule == "8-wave ping-pong" for k in fwd), sched
    assert bwd and all(k.schedule == "4-wave interleave" for k in bwd), sched
    assert gemm and all(k.schedule == "8-wave ping-pong" for k in gemm), sched


@real
def test_real_build_is_clean_and_attributed():
    rows, stats = hk.build_rows()
    assert stats["rows"] > 0
    assert stats["license"] == "MIT"
    assert stats["swizzles_verified_conflict_free"] == len(stats["conflict_degrees"])
    assert len(stats["conflict_degrees"]) >= 5
    for row in rows:
        assert row["_provenance"]["commit"], "provenance lost the commit"
