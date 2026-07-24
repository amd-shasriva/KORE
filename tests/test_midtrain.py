"""CPU-only tests for KORE Stage-0 mid-train (corpus builder + campaign wiring).

No GPU / torch / transformers / trl are imported. We exercise the pure corpus
assembly (deterministic chunking, dedup, ~15% general-replay mix, per-source
counts) from a tmp source tree (and, when present, the real repo paths), and
confirm the campaign's ``_stage_midtrain`` wiring is import-correct on the
dry-run path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from kore.data.midtrain_corpus import (
    build_midtrain_corpus,
    chunk_text,
    discover_repo_roots,
)
from kore.policy.configs import MidTrainConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]  # <repo>/kore


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_config(max_seq_length: int = 64, general_replay_frac: float = 0.15) -> MidTrainConfig:
    return MidTrainConfig(max_seq_length=max_seq_length,
                          general_replay_frac=general_replay_frac)


def _dev_build(*args, **kwargs):
    """Explicitly labeled offline build (exact UTF-8 byte tokenizer + smoke refs)."""
    kwargs.setdefault("development_mode", True)
    return build_midtrain_corpus(*args, **kwargs)


_TRITON_SRC = '''\
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
'''

_HIP_SRC = '''\
#include <hip/hip_runtime.h>

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
'''

_DOC_SRC = '''\
# ROCm rocprof tuning guide

Use rocprofiler-compute to inspect occupancy and LDS bank conflicts on gfx942.
Tune BLOCK sizes and num_warps for the MI300 matmul/gemm kernels. On CDNA3 the
matrix cores prefer 16x16 and 32x32 MFMA tiles; keeping the fp32 accumulator in
registers and streaming the K dimension in BLOCK_K chunks avoids VGPR spills.

Measure the attained fraction of the HBM3 bandwidth roofline before optimizing:
memory-bound kernels benefit from wider vectorized loads and cache-modifier hints,
while compute-bound GEMMs want deeper software pipelining (num_stages) to hide the
global-load latency behind the MFMA issue. Watch the L2 hit-rate and the valu/mfma
busy counters in rocprofv3 to decide which bottleneck to attack next.
'''

_REF_SRC = '''\
import torch


def reference(x, w, eps=1e-6):
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * w
'''

_SEED_SRC = '''\
import triton
import triton.language as tl


# KORE-authored seed kernel carrying STALE pre-retarget arch labels
# (gfx942 / MI300X / CDNA3); the corpus builder must scrub these to the gfx950
# target because this code is genuinely gfx950 (portable Triton, re-verified).
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + row * N + offs, mask=offs < N).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    y = x * (1.0 / tl.sqrt(var + eps))
    tl.store(y_ptr + row * N + offs, y, mask=offs < N)
'''

# Genuinely gfx942 AMD ISA assembly (hand-written Tensile-style CustomKernel).
# EXTERNAL repo text: it is really compiled for gfx942 and MUST stay verbatim
# (its arch label is a true fact, not a stale KORE label to be rewritten).
_ASM_SRC = '''\
/* Custom Tensile GEMM CustomKernel (fp8 MFMA) */
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.text
.globl custom_gemm_f8_gfx942
.p2align 8
.type custom_gemm_f8_gfx942,@function
custom_gemm_f8_gfx942:
    s_load_dwordx4 s[0:3], s[4:5], 0x0
    v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[8:9], v[10:11], a[0:3]
    v_mfma_f32_16x16x32_fp8_fp8 a[4:7], v[12:13], v[14:15], a[4:7]
    ds_read_b128 v[16:19], v20 offset:0
    ds_write_b128 v24, v[16:19] offset:512
    s_waitcnt lgkmcnt(0)
    s_endpgm
'''

# ROCm/CDNA optimization doc as reStructuredText (Sphinx). EXTERNAL repo text
# that legitimately discusses gfx942 / MI300X / CDNA3 -> left verbatim.
_RST_SRC = '''\
CDNA ISA Optimization Guide
===========================

This guide covers optimizing GEMM and attention kernels on AMD Instinct GPUs.
On gfx942 (MI300X, CDNA3) the matrix cores execute MFMA instructions on a
64-lane wavefront; keep the fp32 accumulator resident in AccVGPRs and stream the
K dimension in BLOCK_K tiles through the LDS to avoid register spills. Measure
occupancy and LDS bank conflicts with rocprofiler before tuning num_warps and
num_stages for the software pipeline. Prefer wide ds_read_b128 loads and watch
the valu and mfma busy counters to locate the roofline bottleneck.
'''


def _make_tmp_sources(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny fake (repo_root, task_root) source tree. Returns both roots."""
    repo = tmp_path / "repos" / "FakeKernelRepo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "add_triton.py").write_text(_TRITON_SRC)
    # A duplicate triton file (identical body, different name) -> must dedup.
    (repo / "src" / "add_triton_copy.py").write_text(_TRITON_SRC)
    (repo / "src" / "saxpy.cu").write_text(_HIP_SRC)
    # AMD ISA assembly device code (.s) -> amd_asm source (external, verbatim).
    (repo / "src" / "custom_gemm_gfx942.s").write_text(_ASM_SRC)
    (repo / "docs" / "rocprof").mkdir(parents=True)
    (repo / "docs" / "rocprof" / "tuning_guide.md").write_text(_DOC_SRC)
    # A ROCm/CDNA optimization guide as .rst -> docs source (external, verbatim).
    (repo / "docs" / "rocm").mkdir(parents=True)
    (repo / "docs" / "rocm" / "cdna_isa_guide.rst").write_text(_RST_SRC)
    # An unrelated md (no ROCm/kernel keyword in path) -> excluded from docs.
    (repo / "docs" / "misc").mkdir(parents=True)
    (repo / "docs" / "misc" / "changelog.md").write_text("# Changelog\n\nRandom notes.\n")

    task_root = tmp_path / "tasks"
    for name in ("rmsnorm_fake", "gemm_fake"):
        d = task_root / name
        d.mkdir(parents=True)
        (d / "reference.py").write_text(_REF_SRC)
        (d / "seed_triton.py").write_text(_SEED_SRC)
        (d / "driver.py").write_text("import sys\nprint('driver', sys.argv)\n")
    return repo.parent, task_root


# --------------------------------------------------------------------------- #
# chunk_text unit
# --------------------------------------------------------------------------- #
def test_chunk_text_respects_budget_and_hard_splits():
    text = "\n".join(f"line-{i} " + "x" * 30 for i in range(50))
    budget = 40
    chunks = chunk_text(text, budget)
    assert chunks, "expected at least one chunk"
    assert all(len(c) <= budget for c in chunks)
    # A single over-long line is hard-split (never dropped).
    long_line = "z" * 205
    parts = chunk_text(long_line, budget)
    assert all(len(p) <= budget for p in parts)
    assert "".join(parts) == long_line


# --------------------------------------------------------------------------- #
# corpus builder - tmp source tree
# --------------------------------------------------------------------------- #
def test_corpus_wellformed_and_deterministic(tmp_path):
    repo_root, task_root = _make_tmp_sources(tmp_path)
    cfg = _make_config(max_seq_length=64)

    out1 = tmp_path / "c1.jsonl"
    out2 = tmp_path / "c2.jsonl"
    rep1 = _dev_build(out1, cfg, seed=0, source_roots=[repo_root],
                      task_root=task_root)
    rep2 = _dev_build(out2, cfg, seed=0, source_roots=[repo_root],
                      task_root=task_root)

    # Deterministic: byte-identical output for the same inputs/seed.
    assert out1.read_bytes() == out2.read_bytes()
    assert {k: v for k, v in rep1.items() if k != "out_path"} == \
           {k: v for k, v in rep2.items() if k != "out_path"}

    rows = [json.loads(ln) for ln in out1.read_text().splitlines() if ln.strip()]
    assert rows, "corpus must be non-empty"
    for r in rows:
        assert isinstance(r.get("text"), str) and r["text"].strip()
        assert r.get("source")
        # Development uses a real byte tokenizer, not a chars/token estimate.
        assert len(r["text"].encode("utf-8")) <= cfg.max_seq_length
        meta = r["source_metadata"]
        for key in ("repository_url", "commit", "path", "license", "row_id",
                    "content_hash", "root_content_hash", "lineage_id"):
            assert meta.get(key), (key, meta)

    assert rep1["total"] == len(rows)
    # Every kernel-domain source produced at least one chunk.
    counts = rep1["counts"]
    for src in ("kore_tasks", "pytorch_triton_pairs", "triton", "rocm_hip", "docs"):
        assert counts.get(src, 0) > 0, f"expected chunks for source {src!r}: {counts}"


def test_corpus_dedup_collapses_identical_files(tmp_path, monkeypatch):
    # Isolate dedup: disable source-weighting (which intentionally re-duplicates
    # high-signal chunks AFTER dedup) so the uniqueness assertion tests dedup only.
    monkeypatch.setenv("KORE_MIDTRAIN_WEIGHTING", "0")
    repo_root, task_root = _make_tmp_sources(tmp_path)
    cfg = _make_config(max_seq_length=256)  # big budget: 1 chunk per triton file
    out = tmp_path / "c.jsonl"
    rep = _dev_build(out, cfg, seed=0, source_roots=[repo_root],
                     task_root=task_root)
    # The duplicate triton file body must be dropped by content dedup.
    assert rep["n_dropped_dup"] >= 1
    rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    by_channel = {}
    for row in rows:
        key = (row["source"], row["text"])
        by_channel[key] = by_channel.get(key, 0) + 1
    assert max(by_channel.values(), default=0) == 1
    # Cross-channel copies are intentional: dedup must not erase weighted pairs.
    assert rep["counts"]["pytorch_triton_pairs"] > 0
    assert rep["counts"]["triton"] > 0


def test_corpus_source_weighting_oversamples_high_signal(tmp_path, monkeypatch):
    repo_root, task_root = _make_tmp_sources(tmp_path)
    # Isolate kernel-source counts (no general replay) and use a big budget.
    cfg = _make_config(max_seq_length=256, general_replay_frac=0.0)

    monkeypatch.setenv("KORE_MIDTRAIN_WEIGHTING", "0")
    off = tmp_path / "off.jsonl"
    rep_off = _dev_build(off, cfg, seed=0, source_roots=[repo_root],
                         task_root=task_root)
    assert rep_off["n_weighted_added"] == 0

    monkeypatch.setenv("KORE_MIDTRAIN_WEIGHTING", "1")
    on = tmp_path / "on.jsonl"
    rep_on = _dev_build(on, cfg, seed=0, source_roots=[repo_root],
                        task_root=task_root)
    # High-signal channels (torch->Triton pairs @2x, kore_tasks @1.5x) are
    # oversampled; a neutral channel (triton @1.0x) is unchanged.
    assert rep_on["n_weighted_added"] > 0
    assert rep_on["counts"]["pytorch_triton_pairs"] > rep_off["counts"]["pytorch_triton_pairs"]
    assert rep_on["counts"]["triton"] == rep_off["counts"]["triton"]
    # Determinism holds with weighting on.
    on2 = tmp_path / "on2.jsonl"
    _dev_build(on2, cfg, seed=0, source_roots=[repo_root], task_root=task_root)
    assert on.read_bytes() == on2.read_bytes()


def test_source_weights_env_override(monkeypatch):
    from kore.data.midtrain_corpus import _source_weights
    monkeypatch.setenv("KORE_MIDTRAIN_WEIGHTS", "amd_kernels=3.5,triton=2,bogus")
    w = _source_weights()
    assert w["amd_kernels"] == 3.5   # overridden
    assert w["triton"] == 2.0        # newly added
    assert w["pytorch_triton_pairs"] == 2.0  # default preserved
    # factors are floored at 1.0 (never drop via weighting)
    monkeypatch.setenv("KORE_MIDTRAIN_WEIGHTS", "amd_kernels=0.2")
    assert _source_weights()["amd_kernels"] == 1.0


def test_corpus_docs_pathfilter_excludes_unrelated_md(tmp_path):
    repo_root, task_root = _make_tmp_sources(tmp_path)
    cfg = _make_config(max_seq_length=512)
    out = tmp_path / "c.jsonl"
    _dev_build(out, cfg, seed=0, source_roots=[repo_root], task_root=task_root)
    doc_texts = [json.loads(ln)["text"] for ln in out.read_text().splitlines()
                 if ln.strip() and json.loads(ln)["source"] == "docs"]
    joined = "\n".join(doc_texts)
    assert "rocprof" in joined
    assert "Random notes" not in joined  # unrelated changelog.md was filtered out


def test_amd_asm_and_rst_docs_are_collected(tmp_path):
    """The strengthened device-code sources ingest AMD ISA assembly (.s -> amd_asm)
    and reStructuredText docs (.rst -> docs), not just .py/.cu/.md."""
    repo_root, task_root = _make_tmp_sources(tmp_path)
    cfg = _make_config(max_seq_length=512)
    out = tmp_path / "c.jsonl"
    rep = _dev_build(out, cfg, seed=0, source_roots=[repo_root],
                     task_root=task_root)
    rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]

    def _txt(src):
        return "\n".join(r["text"] for r in rows if r["source"] == src)

    # AMD ISA assembly is a first-class device-code channel now.
    assert rep["counts"].get("amd_asm", 0) > 0, rep["counts"]
    assert "v_mfma_f32_16x16x32_fp8_fp8" in _txt("amd_asm")  # real ISA content
    # The .rst optimization guide is picked up by the docs channel.
    assert "CDNA ISA Optimization Guide" in _txt("docs")


def test_external_repo_text_not_arch_normalized_but_kore_authored_is(tmp_path):
    """arch_normalize scrubs stale gfx942/MI300X/CDNA3 labels ONLY in KORE-authored
    slices (their code is genuinely gfx950). EXTERNAL repo text (repo asm/docs) is
    genuinely other-hardware and must be preserved verbatim -- rewriting it would
    corrupt true facts (e.g. a real gfx942 CustomKernel's target triple)."""
    repo_root, task_root = _make_tmp_sources(tmp_path)
    cfg = _make_config(max_seq_length=512)
    out = tmp_path / "c.jsonl"
    _dev_build(out, cfg, seed=0, source_roots=[repo_root], task_root=task_root)
    rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]

    def _txt(src):
        return "\n".join(r["text"] for r in rows if r["source"] == src)

    # KORE-authored slices: stale arch labels are scrubbed to the gfx950 target.
    for src in ("kore_tasks", "pytorch_triton_pairs"):
        t = _txt(src)
        assert t, f"expected rows for {src}"
        assert "gfx942" not in t and "gfx950" in t, src
        assert "MI300X" not in t, src
        assert "CDNA3" not in t, src

    # EXTERNAL repo text is left verbatim (genuinely other-hardware).
    assert "gfx942" in _txt("amd_asm")   # real gfx942 CustomKernel target triple
    asm = _txt("amd_asm")
    assert "gfx950" not in asm            # NOT rewritten
    docs = _txt("docs")
    assert "gfx942" in docs and "CDNA3" in docs  # ROCm doc discussing gfx942 kept as-is


def test_corpus_scale_env_scales_caps_and_env_overrides_win(tmp_path, monkeypatch):
    """KORE_MIDTRAIN_SCALE multiplies the (raised) default caps so a big run pulls
    proportionally more; an explicit per-source env cap still wins absolutely. The
    report echoes the resolved scale + cap so a build reports what it was sized for."""
    repo_root, task_root = _make_tmp_sources(tmp_path)
    cfg = _make_config(max_seq_length=256)

    monkeypatch.setenv("KORE_MIDTRAIN_SCALE", "2.0")
    rep = _dev_build(tmp_path / "s.jsonl", cfg, seed=0,
                     source_roots=[repo_root], task_root=task_root,
                     max_files_per_source=10, scan_budget=100)
    assert rep["corpus_scale"] == 2.0
    assert rep["max_files_per_source"] == 20  # 10 base * 2.0 scale

    # An explicit absolute cap overrides the scale dial.
    monkeypatch.setenv("KORE_MIDTRAIN_MAX_FILES", "7")
    rep2 = _dev_build(tmp_path / "s2.jsonl", cfg, seed=0,
                      source_roots=[repo_root], task_root=task_root,
                      max_files_per_source=10, scan_budget=100)
    assert rep2["max_files_per_source"] == 7


def test_default_caps_are_sized_for_a_big_run():
    """The CODE default caps are large (a frontier run pulls much more); small values
    are per-run env overrides, not the default."""
    import inspect
    sig = inspect.signature(build_midtrain_corpus)
    assert sig.parameters["max_files_per_source"].default >= 20000
    assert sig.parameters["scan_budget"].default >= 100000


def test_eval_benchmark_decontam_drops_train_on_test(tmp_path):
    """audit R2 midtrain: the CPT corpus is decontaminated against the RETENTION eval
    benchmarks, so a general-replay shard carrying a full eval prompt (train-on-test,
    which would inflate the gate) is dropped while real kernels survive."""
    from kore.data.decontam import decontaminate_corpus, eval_benchmark_texts

    texts = eval_benchmark_texts(development_mode=True)
    assert len(texts) >= 10  # smoke benches loaded (MMLU/HumanEval/LCB/IFEval/BFCL/MT)
    longest = max(texts, key=lambda t: len(t.split()))
    kernel = (
        "import triton\nimport triton.language as tl\n\n@triton.jit\n"
        "def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, N, eps, BLOCK: tl.constexpr):\n"
        "    row = tl.program_id(0)\n    offs = tl.arange(0, BLOCK)\n"
        "    mask = offs < N\n    x = tl.load(x_ptr + row * N + offs, mask=mask).to(tl.float32)\n"
        "    var = tl.sum(x * x, axis=0) / N\n    inv = tl.rsqrt(var + eps)\n"
        "    w = tl.load(w_ptr + offs, mask=mask).to(tl.float32)\n"
        "    tl.store(y_ptr + row * N + offs, (x * inv * w).to(tl.bfloat16), mask=mask)\n"
    )
    rows = [
        {"text": kernel, "source": "triton"},
        {"text": longest, "source": "general_replay"},   # carries a full eval prompt
    ]
    kept, dc = decontaminate_corpus(list(rows), text_key="text", n=8, threshold=0.10,
                                    extra_sources=texts)
    assert dc["n_dropped_contaminated"] >= 1
    assert any(r["source"] == "triton" for r in kept)     # legit kernel survives
    # without the eval sources the same row is NOT dropped (proves the extra source did it)
    kept2, dc2 = decontaminate_corpus(list(rows), text_key="text", n=8, threshold=0.10)
    assert dc2["n_dropped_contaminated"] == 0


def test_general_replay_fraction_is_about_15pct(tmp_path):
    repo_root, task_root = _make_tmp_sources(tmp_path)
    cfg = _make_config(max_seq_length=64, general_replay_frac=0.15)
    out = tmp_path / "c.jsonl"
    rep = _dev_build(out, cfg, seed=0, source_roots=[repo_root],
                     task_root=task_root, use_hf=False)
    assert rep["counts"].get("general_replay", 0) > 0
    # ~15% of the FINAL total, with rounding/dedup tolerance.
    assert 0.08 <= rep["general_frac"] <= 0.22, rep["general_frac"]


def test_general_replay_zero_when_frac_zero(tmp_path):
    repo_root, task_root = _make_tmp_sources(tmp_path)
    cfg = _make_config(max_seq_length=64, general_replay_frac=0.0)
    out = tmp_path / "c.jsonl"
    rep = _dev_build(out, cfg, seed=0, source_roots=[repo_root],
                     task_root=task_root)
    assert rep["counts"].get("general_replay", 0) == 0
    assert rep["general_frac"] == 0.0


# --------------------------------------------------------------------------- #
# corpus builder - REAL repo paths (guarded)
# --------------------------------------------------------------------------- #
def test_corpus_from_real_repos_if_present(tmp_path):
    roots = discover_repo_roots()
    if not roots:
        import pytest
        pytest.skip("local source repos not present on this box")
    cfg = _make_config(max_seq_length=1024)
    out = tmp_path / "real.jsonl"
    # Small caps keep this fast even against the full repo tree.
    rep = _dev_build(out, cfg, seed=0, max_files_per_source=8,
                     scan_budget=400)
    assert rep["total"] > 0
    # The KORE task kernels always exist, so that source is non-empty.
    assert rep["counts"].get("kore_tasks", 0) > 0
    rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert all(isinstance(r["text"], str) and r["text"] for r in rows)


# --------------------------------------------------------------------------- #
# campaign wiring - dry-run import-correctness
# --------------------------------------------------------------------------- #
def _load_run_campaign():
    path = _REPO_ROOT / "scripts" / "run_campaign.py"
    spec = importlib.util.spec_from_file_location("kore_run_campaign", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_midtrain_is_first_default_stage():
    rc = _load_run_campaign()
    assert rc.DEFAULT_STAGES[0] == "midtrain"
    assert "midtrain" in rc.ALL_STAGES


def test_dry_import_check_passes_with_midtrain_symbols():
    rc = _load_run_campaign()
    # Raises SystemExit on any missing symbol / signature drift.
    rc._dry_import_check()
    names = {(mod, attr) for (mod, attr, *_rest) in rc._IMPORT_CHECKS}
    assert ("kore.data.midtrain_corpus", "build_midtrain_corpus") in names
    assert ("kore.policy.midtrain", "train_midtrain") in names


def test_stage_midtrain_dry_run_is_side_effect_free(tmp_path):
    rc = _load_run_campaign()
    ctx = {"dry": True, "data_root": tmp_path, "base": "Qwen/Qwen3-14B",
           "args": SimpleNamespace(lora=True, use_hf=False, midtrain_out="runs/midtrain")}
    # Dry-run must not build a corpus or import the training stack.
    rc._stage_midtrain(ctx)
    assert not (tmp_path / "midtrain").exists()
    assert ctx.get("midtrain_ckpt") is None


def test_trainer_missing_corpus_raises(tmp_path):
    from kore.policy.midtrain import train_midtrain
    cfg = MidTrainConfig(corpus_path=str(tmp_path / "nope.jsonl"))
    try:
        train_midtrain(cfg)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for a missing corpus")
