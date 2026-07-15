"""Deep-CoT contract: optional <think> scratchpad is additive + backward-compatible.

The audited gap: the OUTPUT_CONTRACT structurally CAPPED reasoning depth (a "2-3
sentence" ANALYSIS) and ``summarize_cot`` truncated with a blind head/tail char
slice. Frontier kernel-optimization reasoning is long, branchy and self-correcting
(roofline math, hypotheses, counter-citations, hypothesize->measure->revise).

These CPU-only tests pin the four guarantees of the fix:
  1. BACK-COMPAT - a response with NO ``<think>`` block parses exactly as before
     (kernel/analysis/proposed_change unchanged, and no deep block leaks into the
     extracted kernel); the ANALYSIS is no longer length-capped.
  2. DEEP BLOCK ACCEPTED + PRESERVED - an optional ``<think>`` scratchpad is
     parsed into an additive ``think`` field, round-trips through
     ``format_assistant_turn(..., think=...)``, and NEVER contaminates the kernel
     (even when the scratchpad quotes ``FULL_KERNEL:`` / a fenced block / a
     forbidden op as a counter-citation).
  3. SUMMARIZED FOR CONTEXT - ``summarize_cot`` drops the scratchpad and keeps the
     ANALYSIS/PROPOSED_CHANGE conclusion; ``build_transcript`` re-renders a PRIOR
     turn WITHOUT its scratchpad, while the trained turn keeps its full CoT.
  4. CONSISTENCY GATE - ``check_change_consistency`` catches a described-but-not-
     implemented change and accepts a real one.

Nothing here imports torch/vllm/transformers.
"""

from __future__ import annotations

import kore.data.prompts as P
from kore.data.prompts import extract_kernel
from kore.policy.format import (
    OUTPUT_CONTRACT,
    SYSTEM_PROMPT,
    ChangeConsistency,
    _extract_kernel,
    build_transcript,
    check_change_consistency,
    format_assistant_turn,
    parse_response,
    summarize_cot,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
# A legacy (pre-deep-CoT) response: three sections, NO scratchpad.
OLD_RESPONSE = (
    "ANALYSIS:\n"
    "The tile is too small, increasing occupancy pressure.\n\n"
    "PROPOSED_CHANGE:\n"
    "Bump BLOCK_M to 128.\n\n"
    "FULL_KERNEL:\n"
    "```python\n"
    "import triton\n"
    "@triton.jit\n"
    "def k():\n"
    "    pass\n"
    "```\n"
)

# A DEEP response whose <think> scratchpad is deliberately adversarial: it quotes
# the "FULL_KERNEL:" header, drafts a WRONG fenced kernel, and cites a forbidden
# op ("torch.matmul") as a counter-argument. None of that may leak into the kernel.
DEEP_RESPONSE = (
    "<think>\n"
    "Roofline: bf16 GEMM at 4096^3 has arithmetic intensity ~683 FLOP/byte >> the "
    "~26 ridge point, so it is COMPUTE bound; the seed is memory bound because "
    "BLOCK_K=16 under-feeds the MFMA. Hypothesis: raise BLOCK_K and num_warps.\n"
    "Counter-citation: do NOT fall back to torch.matmul or call rocBLAS/hipBLASLt "
    "-- vendor fallbacks score zero.\n"
    "A draft I will NOT ship (scratch only):\n"
    "```python\n"
    "def wrong():\n"
    "    return torch.matmul(a, b)  # would be a hack\n"
    "```\n"
    "Note the literal header FULL_KERNEL: even appears in my scratchpad here.\n"
    "Revise: keep tl.dot, widen BLOCK_K, bump num_warps 4->8.\n"
    "</think>\n\n"
    "ANALYSIS:\n"
    "Seed is memory-bound: BLOCK_K=16 under-feeds the MFMA and arithmetic intensity "
    "is far below the compute roof. Widen BLOCK_K to 64 and raise num_warps to 8 so "
    "the K loop keeps the matrix cores fed.\n\n"
    "PROPOSED_CHANGE:\n"
    "Increase BLOCK_K from 16 to 64 and num_warps from 4 to 8.\n\n"
    "FULL_KERNEL:\n"
    "```python\n"
    "import triton\n"
    "import triton.language as tl\n"
    "@triton.jit\n"
    "def matmul_kernel(a_ptr, b_ptr, c_ptr, BLOCK_K: tl.constexpr):\n"
    "    acc = tl.dot(a_tile, b_tile, acc)\n"
    "```\n"
)

REAL_KERNEL_MARKERS = ("matmul_kernel", "tl.dot", "BLOCK_K")
LEAK_MARKERS = ("torch.matmul", "rocBLAS", "def wrong", "<think>", "Roofline",
                "Counter-citation")


# --------------------------------------------------------------------------- #
# 1. Contract text: invites deep reasoning, still requires the parseable sections
# --------------------------------------------------------------------------- #
def test_contract_invites_deep_reasoning_but_keeps_required_sections():
    for contract in (OUTPUT_CONTRACT, SYSTEM_PROMPT):
        low = contract.lower()
        # invites an optional <think> scratchpad + evidence-grounded reasoning
        assert "<think>" in contract and "</think>" in contract
        assert "roofline" in low
        # the "2-3 sentence" structural cap is gone
        assert "2-3 sentence" not in low and "2-3 sentences" not in low
        # but the three parseable sections are still required
        assert "ANALYSIS" in contract
        assert "PROPOSED_CHANGE" in contract
        assert "FULL_KERNEL" in contract


def test_single_source_of_truth_reexported_by_prompts():
    # format.py owns the contract; prompts.py re-exports the SAME objects.
    assert P.SYSTEM_PROMPT is SYSTEM_PROMPT
    assert P._OUTPUT_CONTRACT is OUTPUT_CONTRACT
    # data-gen turn prompt embeds the (deep-reasoning) contract too.
    btp = P.build_turn_prompt(parent_source="def k(): pass")
    assert "<think>" in btp and "PROPOSED_CHANGE" in btp and "FULL_KERNEL" in btp
    # the offline normalizer detects the KORE system prompt by this prefix - it
    # MUST survive the contract edit or existing shards stop being canonicalized.
    assert SYSTEM_PROMPT.lstrip().startswith(
        "You are KORE, an expert AMD GPU kernel engineer")


# --------------------------------------------------------------------------- #
# 2. Back-compat: old responses parse identically; no deep block leaks
# --------------------------------------------------------------------------- #
def test_backcompat_old_response_parses_unchanged():
    p = parse_response(OLD_RESPONSE)
    # legacy keys always present
    assert {"analysis", "proposed_change", "kernel"} <= set(p)
    assert p["analysis"] == "The tile is too small, increasing occupancy pressure."
    assert p["proposed_change"] == "Bump BLOCK_M to 128."
    assert "@triton.jit" in p["kernel"] and "def k():" in p["kernel"]
    assert "ANALYSIS" not in p["kernel"]
    # additive key defaults to empty for a scratchpad-free response
    assert p["think"] == ""


def test_backcompat_both_extractors_agree_on_old_response():
    # format._extract_kernel (parse_response path) and prompts.extract_kernel
    # (data-gen path) must both still recover the old kernel.
    k_fmt = _extract_kernel(OLD_RESPONSE).strip()
    k_prm = extract_kernel(OLD_RESPONSE).strip()
    assert "def k():" in k_fmt and "def k():" in k_prm
    assert "ANALYSIS" not in k_fmt and "ANALYSIS" not in k_prm


def test_backcompat_format_assistant_turn_no_think_is_identical():
    t = format_assistant_turn("mem bound", "Vectorize loads", "def k():\n    return 1")
    # no scratchpad -> the render still starts with ANALYSIS: (unchanged shape)
    assert t.startswith("ANALYSIS:") and "<think>" not in t
    p = parse_response(t)
    assert p["analysis"] == "mem bound"
    assert p["proposed_change"] == "Vectorize loads"
    assert p["kernel"].strip() == "def k():\n    return 1"
    assert p["think"] == ""
    # PROPOSED_CHANGE still omitted when empty
    t2 = format_assistant_turn("just analysis", "", "def k(): pass")
    assert "PROPOSED_CHANGE" not in t2 and t2.startswith("ANALYSIS:")


def test_analysis_is_no_longer_length_capped():
    # The "expanded ANALYSIS" alternative to <think>: a long ANALYSIS must survive
    # parsing in full (no structural 2-3 sentence cap anywhere in the pipeline).
    long_analysis = ("The roofline puts this kernel below the HBM bandwidth line; "
                     "arithmetic intensity is low and occupancy is bounded by VGPRs. ") * 30
    resp = ("ANALYSIS:\n" + long_analysis + "\n\nPROPOSED_CHANGE:\ndo X\n\n"
            "FULL_KERNEL:\n```python\ndef k(): pass\n```")
    p = parse_response(resp)
    assert p["think"] == ""
    assert len(p["analysis"]) > 1000
    assert long_analysis.strip() in p["analysis"]


# --------------------------------------------------------------------------- #
# 3. Deep block accepted + preserved for training; never leaks into the kernel
# --------------------------------------------------------------------------- #
def test_deep_block_parsed_into_additive_think_field():
    p = parse_response(DEEP_RESPONSE)
    # scratchpad captured (with its counter-citation) under the additive key
    assert "Roofline" in p["think"]
    assert "torch.matmul" in p["think"]  # counter-citation preserved IN the think
    # terse sections parsed from the think-stripped body
    assert "memory-bound" in p["analysis"] and "<think>" not in p["analysis"]
    assert p["proposed_change"] == (
        "Increase BLOCK_K from 16 to 64 and num_warps from 4 to 8.")


def test_deep_block_never_leaks_into_kernel():
    # The adversarial scratchpad (fake FULL_KERNEL:, wrong fence, forbidden op)
    # must be invisible to EVERY kernel extractor.
    k_parse = parse_response(DEEP_RESPONSE)["kernel"]
    k_fmt = _extract_kernel(DEEP_RESPONSE)
    k_prm = extract_kernel(DEEP_RESPONSE)
    for k in (k_parse, k_fmt, k_prm):
        for good in REAL_KERNEL_MARKERS:
            assert good in k, f"missing {good!r} in extracted kernel"
        for bad in LEAK_MARKERS:
            assert bad not in k, f"scratchpad leaked {bad!r} into the kernel"


def test_deep_block_round_trips_and_is_preserved_for_training_target():
    think = ("Roofline says compute-bound; hypothesize num_warps 4->8; "
             "counter-argument: watch VGPR spills; revise and measure.")
    target = format_assistant_turn(
        "Compute-bound; raise num_warps to feed the MFMA.",
        "Raise num_warps from 4 to 8.",
        "def k():\n    return tl.dot(a, b, acc)",
        think=think,
    )
    # the TRAINED turn keeps the full CoT verbatim, ahead of the sections
    assert target.startswith("<think>\n" + think + "\n</think>")
    rp = parse_response(target)
    assert rp["think"] == think
    assert rp["analysis"] == "Compute-bound; raise num_warps to feed the MFMA."
    assert rp["proposed_change"] == "Raise num_warps from 4 to 8."
    assert rp["kernel"].strip() == "def k():\n    return tl.dot(a, b, acc)"


def test_unclosed_think_still_recovers_kernel_and_sections():
    # A long scratchpad whose </think> the model forgot: everything up to the
    # first real header is the scratchpad; the kernel is still recovered.
    unclosed = (
        "<think>\n"
        "long reasoning that never closes and even says FULL_KERNEL: mid-sentence\n"
        "ANALYSIS:\n"
        "real analysis after the scratchpad\n\n"
        "PROPOSED_CHANGE:\n"
        "do the real change\n\n"
        "FULL_KERNEL:\n"
        "```python\n"
        "def real():\n"
        "    return 2\n"
        "```\n"
    )
    p = parse_response(unclosed)
    assert "never closes" in p["think"]
    assert p["analysis"] == "real analysis after the scratchpad"
    assert p["proposed_change"] == "do the real change"
    assert p["kernel"].strip() == "def real():\n    return 2"
    assert "FULL_KERNEL" not in p["kernel"] and "<think>" not in p["kernel"]


# --------------------------------------------------------------------------- #
# 3b. Summarized for cross-turn context (scratchpad dropped, conclusion kept)
# --------------------------------------------------------------------------- #
def test_summarize_cot_backcompat_bounds_and_shortcircuits():
    # unchanged legacy behavior for plain text
    assert summarize_cot("short", max_chars=200) == "short"
    assert len(summarize_cot("x" * 5000, max_chars=200)) <= 200


def test_summarize_cot_drops_scratchpad_keeps_conclusion():
    raw = "<think>\n" + ("z" * 5000) + "\n</think>\nkeep this terse conclusion"
    out = summarize_cot(raw, max_chars=2000)
    assert "keep this terse conclusion" in out
    assert "z" not in out and "<think>" not in out  # scratchpad dropped


def test_summarize_cot_prefers_structured_conclusion_over_blind_slice():
    text = (
        "<think>\n" + ("z" * 3000) + "\n</think>\n"
        "ANALYSIS:\nThe bottleneck is HBM bandwidth, not compute.\n\n"
        "PROPOSED_CHANGE:\nRaise BLOCK_K to 64.\n\n"
        "FULL_KERNEL:\n```python\n" + ("x = 1\n" * 2000) + "```\n"
    )
    out = summarize_cot(text, max_chars=300)
    assert len(out) <= 300
    assert "HBM bandwidth" in out          # ANALYSIS kept
    assert "Raise BLOCK_K to 64" in out    # PROPOSED_CHANGE kept
    assert "x = 1" not in out              # verbose kernel body dropped
    assert "z" not in out                  # scratchpad dropped


def test_build_transcript_summarizes_prior_turn_but_keeps_kernel():
    think = "LONG scratchpad reasoning " * 50
    resp = format_assistant_turn(
        "mem bound; widen BLOCK_K",
        "Raise BLOCK_K to 64.",
        "def k():\n    return tl.dot(a, b)",
        think=think,
    )
    msgs = build_transcript("optimize this", [{"response": resp, "feedback": "RESULT: CORRECT"}])
    asst = next(m["content"] for m in msgs if m["role"] == "assistant")
    # PRIOR-turn context: scratchpad fully dropped ...
    assert "<think>" not in asst and "scratchpad" not in asst
    # ... but the durable artifacts survive
    assert "def k()" in asst and "tl.dot" in asst
    assert "Raise BLOCK_K to 64" in asst
    # contrast: the raw (trained) turn still carries the full CoT
    assert "<think>" in resp and "scratchpad" in resp


# --------------------------------------------------------------------------- #
# 4. Claim <-> code consistency gate
# --------------------------------------------------------------------------- #
_PREV = (
    "import triton\nimport triton.language as tl\n"
    "@triton.jit\n"
    "def mm(a, b, c, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):\n"
    "    acc = tl.zeros((BLOCK_M, BLOCK_M), tl.float32)\n"
    "\n"
    "def launch(a, b, c):\n"
    "    mm[grid](a, b, c, BLOCK_M=64, BLOCK_K=16, num_warps=4, num_stages=2)\n"
)
_NEW_WARPS = _PREV.replace("num_warps=4", "num_warps=8")
_NEW_TILE = _PREV.replace("BLOCK_K=16", "BLOCK_K=64")

_PREV_LOOP = (
    "@triton.jit\n"
    "def mm(a, b, c):\n"
    "    acc = 0.0\n"
    "    for k in range(K):\n"
    "        acc += a[k] * b[k]\n"
)
_NEW_DOT = (
    "@triton.jit\n"
    "def mm(a, b, c):\n"
    "    acc = tl.dot(a, b, acc)\n"
)


def test_consistency_catches_described_but_not_implemented():
    # Claims a num_warps bump but ships an UNCHANGED kernel -> inconsistent.
    r = check_change_consistency("Bump num_warps from 4 to 8 for occupancy.",
                                 _PREV, _PREV)
    assert isinstance(r, ChangeConsistency)
    assert bool(r) is False and r.consistent is False
    assert "num_warps" in r.claimed and "num_warps" in r.missing
    assert r.applied == ()


def test_consistency_accepts_a_real_numeric_change():
    r = check_change_consistency("Bump num_warps from 4 to 8 for occupancy.",
                                 _PREV, _NEW_WARPS)
    assert bool(r) is True
    assert "num_warps" in r.applied and r.missing == ()


def test_consistency_accepts_tile_change_and_structural_tl_dot():
    tile = check_change_consistency("Increase the BLOCK_K tile size.", _PREV, _NEW_TILE)
    assert bool(tile) is True and "block" in tile.applied

    dot = check_change_consistency(
        "Replace the manual FMA loop with tl.dot to use the MFMA cores.",
        _PREV_LOOP, _NEW_DOT)
    assert bool(dot) is True and "tl.dot" in dot.applied
    # ... and the same tl.dot claim against an unchanged kernel is caught.
    none = check_change_consistency("Switch to tl.dot for MFMA.", _PREV_LOOP, _PREV_LOOP)
    assert bool(none) is False and "tl.dot" in none.missing


def test_consistency_vague_claim_is_not_falsifiable():
    # No concrete knob named -> cannot be disproved -> consistent (do not block).
    r = check_change_consistency("Make it faster somehow.", _PREV, _NEW_WARPS)
    assert bool(r) is True and r.claimed == ()


def test_consistency_uses_proposed_change_text_too():
    # The knob may be named only in PROPOSED_CHANGE.
    r = check_change_consistency("", _PREV, _NEW_WARPS,
                                 proposed_change="Raise num_warps to 8.")
    assert bool(r) is True and "num_warps" in r.applied
