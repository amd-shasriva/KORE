"""Tests for external public-dataset ingestion (kore.data.external).

Verifies the frontier-quality gate: license admission, load, decontamination,
on-hardware verification (injected), and canonical-contract normalization — all
offline/CPU (no network, no torch/datasets needed).
"""
from kore.data import external as ex


_ROWS = [
    {"pytorch": "class M(torch.nn.Module):\n  def forward(self,x): return torch.softmax(x,-1)",
     "triton": "@triton.jit\ndef _k(): pass\ndef softmax(x): return x"},
    {"pytorch": "def f(a,b): return a@b  # matmul", "triton": "def gemm(a,b): return a"},
    {"pytorch": "def f(x): return torch.nn.functional.scaled_dot_product_attention(x,x,x)",
     "triton": "def attn(x): return x"},
]


def test_license_gate():
    # RAIL-D reciprocity source is NOT admitted for training; eval-only kinds aren't either.
    assert ex.license_ok("kb_multiturn_traces") is True
    assert ex.license_ok("kernelbook") is False
    assert ex.license_ok("kernelbench") is False
    rep = {r["name"]: r for r in ex.license_report()}
    assert rep["kernelbook"]["admitted_for_training"] is False


def test_load_and_op_hints():
    pairs = ex.load_pairs("kb_multiturn_traces", local_rows=_ROWS)
    assert [p.op_hint for p in pairs] == ["softmax", "matmul", "attention"]
    assert all(p.torch_src and p.triton_src for p in pairs)


def test_decontaminate_drops_heldout_family():
    pairs = ex.load_pairs("kb_multiturn_traces", local_rows=_ROWS)
    kept, dropped = ex.decontaminate(pairs, heldout_op_families={"attention"})
    assert dropped == 1
    assert all(p.op_hint != "attention" for p in kept)


def test_verify_filter_keeps_only_verified():
    pairs = ex.load_pairs("kb_multiturn_traces", local_rows=_ROWS)
    survivors, stats = ex.verify_filter(pairs, verify=lambda r: r.op_hint == "softmax")
    assert stats["verified"] == 1 and stats["dropped"] == 2
    assert survivors[0].op_hint == "softmax"


def test_to_sft_rows_is_canonical_contract():
    pairs = ex.load_pairs("kb_multiturn_traces", local_rows=_ROWS[:1])
    rows = ex.to_sft_rows(pairs)
    msgs = rows[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "FULL_KERNEL" in msgs[-1]["content"]
    assert rows[0]["_provenance"]["verified_gfx942"] is True


def test_ingest_requires_verifier_and_gates():
    # No verifier -> not admitted (never mix unverified external data).
    r0 = ex.ingest("kb_multiturn_traces", heldout_op_families={"attention"}, local_rows=_ROWS)
    assert r0["admitted"] is False
    # License-blocked source -> not admitted regardless.
    rb = ex.ingest("kernelbook", heldout_op_families=set(), local_rows=_ROWS,
                   verify=lambda r: True)
    assert rb["admitted"] is False
    # Verified + admitted -> exactly the verified, decontaminated survivors.
    r1 = ex.ingest("kb_multiturn_traces", heldout_op_families={"attention"},
                   local_rows=_ROWS, verify=lambda r: r.op_hint == "softmax")
    assert r1["admitted"] is True and len(r1["sft_rows"]) == 1
    assert r1["stats"]["decontam_dropped"] == 1
