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
    CHARS_PER_TOKEN,
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


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + row * N + offs, mask=offs < N).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    y = x * (1.0 / tl.sqrt(var + eps))
    tl.store(y_ptr + row * N + offs, y, mask=offs < N)
'''


def _make_tmp_sources(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny fake (repo_root, task_root) source tree. Returns both roots."""
    repo = tmp_path / "repos" / "FakeKernelRepo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "add_triton.py").write_text(_TRITON_SRC)
    # A duplicate triton file (identical body, different name) -> must dedup.
    (repo / "src" / "add_triton_copy.py").write_text(_TRITON_SRC)
    (repo / "src" / "saxpy.cu").write_text(_HIP_SRC)
    (repo / "docs" / "rocprof").mkdir(parents=True)
    (repo / "docs" / "rocprof" / "tuning_guide.md").write_text(_DOC_SRC)
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
    budget = cfg.max_seq_length * CHARS_PER_TOKEN

    out1 = tmp_path / "c1.jsonl"
    out2 = tmp_path / "c2.jsonl"
    rep1 = build_midtrain_corpus(out1, cfg, seed=0, source_roots=[repo_root],
                                 task_root=task_root)
    rep2 = build_midtrain_corpus(out2, cfg, seed=0, source_roots=[repo_root],
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
        assert len(r["text"]) <= budget  # chunked to the token/char budget

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
    rep = build_midtrain_corpus(out, cfg, seed=0, source_roots=[repo_root],
                                task_root=task_root)
    # The duplicate triton file body must be dropped by content dedup.
    assert rep["n_dropped_dup"] >= 1
    texts = [json.loads(ln)["text"] for ln in out.read_text().splitlines() if ln.strip()]
    assert len(texts) == len(set(texts)), "no duplicate chunk texts should remain"


def test_corpus_source_weighting_oversamples_high_signal(tmp_path, monkeypatch):
    repo_root, task_root = _make_tmp_sources(tmp_path)
    # Isolate kernel-source counts (no general replay) and use a big budget.
    cfg = _make_config(max_seq_length=256, general_replay_frac=0.0)

    monkeypatch.setenv("KORE_MIDTRAIN_WEIGHTING", "0")
    off = tmp_path / "off.jsonl"
    rep_off = build_midtrain_corpus(off, cfg, seed=0, source_roots=[repo_root],
                                    task_root=task_root)
    assert rep_off["n_weighted_added"] == 0

    monkeypatch.setenv("KORE_MIDTRAIN_WEIGHTING", "1")
    on = tmp_path / "on.jsonl"
    rep_on = build_midtrain_corpus(on, cfg, seed=0, source_roots=[repo_root],
                                   task_root=task_root)
    # High-signal channels (torch->Triton pairs @2x, kore_tasks @1.5x) are
    # oversampled; a neutral channel (triton @1.0x) is unchanged.
    assert rep_on["n_weighted_added"] > 0
    assert rep_on["counts"]["pytorch_triton_pairs"] > rep_off["counts"]["pytorch_triton_pairs"]
    assert rep_on["counts"]["triton"] == rep_off["counts"]["triton"]
    # Determinism holds with weighting on.
    on2 = tmp_path / "on2.jsonl"
    build_midtrain_corpus(on2, cfg, seed=0, source_roots=[repo_root], task_root=task_root)
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
    build_midtrain_corpus(out, cfg, seed=0, source_roots=[repo_root], task_root=task_root)
    doc_texts = [json.loads(ln)["text"] for ln in out.read_text().splitlines()
                 if ln.strip() and json.loads(ln)["source"] == "docs"]
    joined = "\n".join(doc_texts)
    assert "rocprof" in joined
    assert "Random notes" not in joined  # unrelated changelog.md was filtered out


def test_eval_benchmark_decontam_drops_train_on_test(tmp_path):
    """audit R2 midtrain: the CPT corpus is decontaminated against the RETENTION eval
    benchmarks, so a general-replay shard carrying a full eval prompt (train-on-test,
    which would inflate the gate) is dropped while real kernels survive."""
    from kore.data.decontam import decontaminate_corpus, eval_benchmark_texts

    texts = eval_benchmark_texts()
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
    rep = build_midtrain_corpus(out, cfg, seed=0, source_roots=[repo_root],
                                task_root=task_root, use_hf=False)
    assert rep["counts"].get("general_replay", 0) > 0
    # ~15% of the FINAL total, with rounding/dedup tolerance.
    assert 0.08 <= rep["general_frac"] <= 0.22, rep["general_frac"]


def test_general_replay_zero_when_frac_zero(tmp_path):
    repo_root, task_root = _make_tmp_sources(tmp_path)
    cfg = _make_config(max_seq_length=64, general_replay_frac=0.0)
    out = tmp_path / "c.jsonl"
    rep = build_midtrain_corpus(out, cfg, seed=0, source_roots=[repo_root],
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
    rep = build_midtrain_corpus(out, cfg, seed=0, max_files_per_source=8,
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
