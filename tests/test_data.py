"""CPU-only tests for the KORE data-generation module.

No GPU, no teacher model, no torch/transformers/vllm. A StubTeacher is used
wherever a teacher is required.
"""

from __future__ import annotations

import random

from kore.data.schemas import (
    RepairRecord,
    RankedGroupRecord,
    WinRecord,
    write_jsonl,
    read_jsonl,
    record_from_dict,
)
from kore.data.prompts import (
    SYSTEM_PROMPT,
    build_turn_prompt,
    extract_kernel,
)
from kore.data.teacher import StubTeacher, TeacherClient
from kore.data import mutate
from kore.data.gen_groups import rank_candidates, build_preferences
from kore.data.gen_repair import make_repair_record
from kore.reward.reward import Observation
from kore.data.build_datasets import (
    build_sft,
    build_dpo,
    build_rft,
    dedup_by_source_hash,
    leakage_split,
)


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #
def _sample_repair():
    return RepairRecord(
        task_id="gemm_bf16",
        failure_class="snr_fail",
        parent_hash="deadbeef",
        error_text="worst SNR 5.0 < 25.0 dB",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "FULL_KERNEL:\n```python\ndef matmul(a,b):\n    return a@b\n```"},
        ],
        child_snr_db=42.0,
    )


def _sample_group():
    return RankedGroupRecord(
        task_id="gemm_bf16",
        parent_id="parent123",
        candidates=[
            {"source": "def matmul(a,b): return a@b  # A", "wall_us": 100.0, "snr_db": 40.0, "rank": 0},
            {"source": "def matmul(a,b): return a@b  # B", "wall_us": 200.0, "snr_db": 39.0, "rank": 1},
            {"source": "def matmul(a,b): return a@b  # C", "wall_us": None, "snr_db": None, "rank": 2},
        ],
        preferences=[[0, 1], [0, 2], [1, 2]],
    )


def _sample_win():
    return WinRecord(
        task_id="gemm_bf16",
        trajectory=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "opt"},
            {"role": "assistant", "content": "FULL_KERNEL:\n```python\ndef matmul(a,b): return a@b\n```"},
        ],
        initial_wall_us=200.0,
        final_wall_us=100.0,
        speedup=2.0,
        final_source="def matmul(a,b): return a@b  # fast",
        snr_db=41.0,
    )


def test_schema_roundtrip_dict():
    for rec in (_sample_repair(), _sample_group(), _sample_win()):
        d = rec.to_dict()
        again = record_from_dict(d)
        assert again == rec
        assert again.to_dict() == d


def test_schema_roundtrip_jsonl(tmp_path):
    recs = [_sample_repair(), _sample_group(), _sample_win()]
    path = tmp_path / "records.jsonl"
    write_jsonl(path, recs)
    loaded = read_jsonl(path)
    assert loaded == recs
    # type dispatch produced the right classes
    assert isinstance(loaded[0], RepairRecord)
    assert isinstance(loaded[1], RankedGroupRecord)
    assert isinstance(loaded[2], WinRecord)


def test_read_jsonl_raw(tmp_path):
    path = tmp_path / "r.jsonl"
    write_jsonl(path, [_sample_repair()])
    raw = read_jsonl(path, typed=False)
    assert isinstance(raw[0], dict) and raw[0]["type"] == "repair"


# --------------------------------------------------------------------------- #
# prompts / extraction
# --------------------------------------------------------------------------- #
def test_build_turn_prompt_modes():
    for mode in ("exploit", "explore", "repair"):
        p = build_turn_prompt("def matmul(a,b): return a@b", feedback="err", mode=mode)
        assert "FULL_KERNEL:" in p
        assert "def matmul" in p
    # repair mode surfaces the error feedback
    p = build_turn_prompt("src", feedback="boom-error", mode="repair")
    assert "boom-error" in p


def test_extract_kernel_full_kernel_block():
    resp = (
        "ANALYSIS: did a thing.\n"
        "CHANGE: tile\n"
        "FULL_KERNEL:\n"
        "```python\n"
        "import triton\n"
        "def matmul(a, b):\n"
        "    return a @ b\n"
        "```\n"
    )
    src = extract_kernel(resp)
    assert "def matmul(a, b):" in src
    assert "import triton" in src
    assert "FULL_KERNEL" not in src


def test_extract_kernel_fenced_only():
    resp = "Here you go:\n```python\ndef f(x):\n    return x\n```\nthanks"
    src = extract_kernel(resp)
    assert src == "def f(x):\n    return x"


def test_extract_kernel_full_kernel_no_fence():
    resp = "FULL_KERNEL:\ndef matmul(a, b):\n    return a @ b\n"
    src = extract_kernel(resp)
    assert "def matmul(a, b):" in src


def test_extract_kernel_empty():
    assert extract_kernel("no code here at all") == ""
    assert extract_kernel("") == ""


# --------------------------------------------------------------------------- #
# teacher (stub only)
# --------------------------------------------------------------------------- #
def test_stub_teacher_is_teacherclient():
    t = StubTeacher()
    assert isinstance(t, TeacherClient)
    out = t.generate([{"role": "user", "content": "hi"}])
    assert "FULL_KERNEL:" in out
    assert extract_kernel(out)
    assert len(t.calls) == 1


def test_stub_teacher_custom_fn():
    t = StubTeacher(fn=lambda msgs: "FULL_KERNEL:\n```python\nx=1\n```")
    assert extract_kernel(t.generate([])) == "x=1"


# --------------------------------------------------------------------------- #
# mutate
# --------------------------------------------------------------------------- #
SEED_SRC = """
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

def matmul(a, b):
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8
    return a @ b
"""


def test_break_block_size_changes_source():
    out, hint = mutate.break_block_size(SEED_SRC)
    assert out != SEED_SRC
    assert hint == "compile_fail"


def test_break_accumulator_dtype_changes_source():
    out, hint = mutate.break_accumulator_dtype(SEED_SRC)
    assert out != SEED_SRC
    assert "tl.float32" not in out or "tl.float16" in out
    assert hint == "snr_fail"


def test_break_index_offset_changes_source():
    out, hint = mutate.break_index_offset(SEED_SRC)
    assert out != SEED_SRC
    assert hint == "snr_fail"


def test_apply_random_breakage_changes_source():
    rng = random.Random(1234)
    for _ in range(10):
        out, hint, name = mutate.apply_random_breakage(SEED_SRC, rng)
        assert out != SEED_SRC
        assert hint in ("compile_fail", "snr_fail")
        assert isinstance(name, str) and name


# --------------------------------------------------------------------------- #
# mutate: op-family-aware mutators
# --------------------------------------------------------------------------- #
# A norm/softmax-style seed exercising reduction, mask, eps, fp32 cast, scale.
NORM_SRC = """
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, N, eps, scale, BLOCK_N: tl.constexpr):
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = x * rstd * scale
    tl.store(y_ptr + offs, y.to(tl.bfloat16), mask=mask)
"""


def test_break_reduction_axis_changes_source():
    out, hint = mutate.break_reduction_axis(NORM_SRC)
    assert out != NORM_SRC
    assert "axis=1" in out
    assert hint == "snr_fail"


def test_break_mask_changes_source():
    out, hint = mutate.break_mask(NORM_SRC)
    assert out != NORM_SRC
    assert hint == "snr_fail"


def test_break_eps_changes_source():
    out, hint = mutate.break_eps(NORM_SRC)
    assert out != NORM_SRC
    assert "+ eps" not in out
    assert hint == "snr_fail"


def test_break_dtype_cast_changes_source():
    out, hint = mutate.break_dtype_cast(NORM_SRC)
    assert out != NORM_SRC
    assert ".to(tl.float32)" not in out
    assert hint == "snr_fail"


def test_break_scale_changes_source():
    out, hint = mutate.break_scale(NORM_SRC)
    assert out != NORM_SRC
    assert hint == "snr_fail"


def test_apply_random_breakage_norm_family():
    rng = random.Random(7)
    for _ in range(10):
        out, hint, name = mutate.apply_random_breakage(NORM_SRC, rng, family="norm")
        assert out != NORM_SRC
        assert hint in ("compile_fail", "snr_fail")
        assert name


def test_apply_random_breakage_generic_family():
    rng = random.Random(9)
    for _ in range(10):
        out, hint, name = mutate.apply_random_breakage(NORM_SRC, rng, family="generic")
        assert out != NORM_SRC
        assert hint in ("compile_fail", "snr_fail")
        assert name


def test_apply_random_breakage_unknown_family_falls_back():
    rng = random.Random(3)
    out, hint, name = mutate.apply_random_breakage(NORM_SRC, rng, family="nope")
    assert out != NORM_SRC
    assert name


def test_infer_family_mapping():
    cases = {
        "rmsnorm": "norm",
        "rmsnorm_bf16": "norm",
        "layernorm": "norm",
        "silu": "activation",
        "gelu_bf16": "activation",
        "act_fn": "activation",
        "gemm": "gemm",
        "matmul_fp16": "gemm",
        "attn": "attention",
        "mha": "attention",
        "mla_decode": "attention",
        "moe_router": "moe",
        "something_else": "generic",
    }
    for key, expected in cases.items():
        assert mutate.infer_family(key) == expected, key


def test_op_family_mutators_covers_all_families():
    expected = {"gemm", "norm", "activation", "attention", "moe", "generic"}
    assert expected <= set(mutate.OP_FAMILY_MUTATORS)
    for fam, muts in mutate.OP_FAMILY_MUTATORS.items():
        assert muts, fam
        assert all(callable(fn) for fn in muts)


def test_generic_mutators_apply_to_gemm_seed():
    # generic mutators must plausibly break a non-norm kernel too
    out, hint, name = mutate.apply_random_breakage(
        SEED_SRC, random.Random(1), family="generic"
    )
    assert out != SEED_SRC
    assert name


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #
def test_rank_candidates_ordering():
    results = [
        {"compiled": True, "correct": True, "speedup": 1.5, "snr_db": 40.0},   # 0 slower-correct
        {"compiled": True, "correct": True, "speedup": 3.0, "snr_db": 41.0},   # 1 faster-correct
        {"compiled": True, "correct": False, "speedup": None, "snr_db": 5.0},  # 2 incorrect
        {"compiled": False, "correct": False, "speedup": None, "snr_db": None},# 3 noncompile
    ]
    order = rank_candidates(results)
    assert order == [1, 0, 2, 3]


# --------------------------------------------------------------------------- #
# margin gate (measurement-noise ties -> non-preferences)
# --------------------------------------------------------------------------- #
def test_build_preferences_default_no_gate():
    # default 0.0 bands preserve the pure ranking behaviour exactly
    results = [
        {"compiled": True, "correct": True, "speedup": 1.5, "snr_db": 40.0},
        {"compiled": True, "correct": True, "speedup": 3.0, "snr_db": 41.0},
    ]
    assert build_preferences(results) == [[1, 0]]


def test_build_preferences_margin_gate_drops_near_ties():
    results = [
        {"compiled": True, "correct": True, "speedup": 2.00, "snr_db": 40.0},
        {"compiled": True, "correct": True, "speedup": 2.02, "snr_db": 40.1},
    ]
    # ungated: 2.02 > 2.00 is a strict ordering
    assert build_preferences(results) == [[1, 0]]
    # margin-gated: within 3% speed AND 0.5 dB SNR -> measurement-noise tie -> drop
    gated = build_preferences(results, speedup_noise_band=0.03, snr_noise_band_db=0.5)
    assert gated == []


def test_build_preferences_margin_gate_keeps_real_speed_win():
    results = [
        {"compiled": True, "correct": True, "speedup": 1.0, "snr_db": 40.0},
        {"compiled": True, "correct": True, "speedup": 3.0, "snr_db": 40.0},
    ]
    prefs = build_preferences(results, speedup_noise_band=0.03, snr_noise_band_db=0.5)
    assert [1, 0] in prefs  # a real 3x speedup is not a noise tie


def test_build_preferences_margin_gate_keeps_cross_tier():
    # correct-vs-incorrect is always a real preference, regardless of the bands
    results = [
        {"compiled": True, "correct": True, "speedup": 2.0, "snr_db": 40.0},
        {"compiled": True, "correct": False, "speedup": None, "snr_db": 5.0},
    ]
    prefs = build_preferences(results, speedup_noise_band=0.99, snr_noise_band_db=100.0)
    assert [0, 1] in prefs


# --------------------------------------------------------------------------- #
# diagnostic-augmented repair corpus (LLM-VeriOpt <think>/<answer>)
# --------------------------------------------------------------------------- #
class _RepairTask:
    task_id = "gemm_bf16"
    dtype = "bf16"
    gpu_target = "gfx942"
    operation = "gemm"


class _FixEnv:
    """Env whose fix always validates (the repair path only steps the fix)."""

    def step(self, source, full_validation=True, multi_shape=True):
        return Observation(compiled=True, validation_passed=True, snr_db=40.0,
                           snr_by_shape={"primary": 40.0}, wall_ms=1.0,
                           baseline_ms=2.0, dtype="bf16")


def _broken_snr_obs():
    return Observation(compiled=True, validation_passed=False, snr_db=5.0,
                       snr_by_shape={"primary": 5.0},
                       error_text="worst SNR 5.0 < 25.0 dB", dtype="bf16")


def test_make_repair_record_diagnostic_format():
    teacher = StubTeacher(fn=lambda m: "FULL_KERNEL:\n```python\ndef k():\n    return 1\n```")
    rec = make_repair_record(_RepairTask(), teacher, _FixEnv(),
                             broken_src="def broken():\n    return 0",
                             broken_obs=_broken_snr_obs())
    assert rec is not None
    assert rec.failure_class == "snr_fail"
    content = rec.messages[-1]["content"]
    assert rec.messages[-1]["role"] == "assistant"
    # CANONICAL diagnose-then-fix contract (ANALYSIS / PROPOSED_CHANGE / FULL_KERNEL) —
    # NOT the old <think>/<answer> shape (Pillar 0: single contract shared with inference)
    assert "<think>" not in content and "<answer>" not in content
    assert content.startswith("ANALYSIS:")
    assert "PROPOSED_CHANGE:" in content and "FULL_KERNEL:" in content
    # the verifier error is folded into the ANALYSIS reasoning
    assert "worst SNR 5.0" in content
    # the verified fix is still parseable as a FULL_KERNEL block
    assert extract_kernel(content).strip() == "def k():\n    return 1"


def test_make_repair_record_plain_format_when_disabled():
    resp = "ANALYSIS: ok\nFULL_KERNEL:\n```python\ndef k():\n    return 1\n```"
    teacher = StubTeacher(fn=lambda m: resp)
    rec = make_repair_record(_RepairTask(), teacher, _FixEnv(),
                             broken_src="def broken():\n    return 0",
                             broken_obs=_broken_snr_obs(), diagnostic=False)
    assert rec is not None
    content = rec.messages[-1]["content"]
    assert content == resp
    assert "<think>" not in content


def test_make_repair_record_none_when_broken_side_passes():
    # if the "broken" observation actually passed, there's nothing to repair
    passing = Observation(compiled=True, validation_passed=True, snr_db=40.0,
                          snr_by_shape={"primary": 40.0}, dtype="bf16")
    teacher = StubTeacher()
    rec = make_repair_record(_RepairTask(), teacher, _FixEnv(),
                             broken_src="x", broken_obs=passing)
    assert rec is None


def test_build_preferences():
    results = [
        {"compiled": True, "correct": True, "speedup": 1.5, "snr_db": 40.0},
        {"compiled": True, "correct": True, "speedup": 3.0, "snr_db": 41.0},
        {"compiled": True, "correct": False, "speedup": None, "snr_db": 5.0},
        {"compiled": False, "correct": False, "speedup": None, "snr_db": None},
    ]
    prefs = build_preferences(results)
    # faster-correct(1) beats everyone
    assert [1, 0] in prefs and [1, 2] in prefs and [1, 3] in prefs
    # slower-correct(0) beats incorrect + noncompile but not faster-correct
    assert [0, 2] in prefs and [0, 3] in prefs
    assert [0, 1] not in prefs
    # incorrect(2) beats noncompile(3)
    assert [2, 3] in prefs
    # no self / reverse duplicates
    assert all(p[0] != p[1] for p in prefs)


# --------------------------------------------------------------------------- #
# build_datasets
# --------------------------------------------------------------------------- #
def test_build_sft_wellformed():
    rows = build_sft([_sample_repair(), _sample_win(), _sample_group()])
    # repair + win produce rows; ranked_group does not
    assert len(rows) == 2
    for row in rows:
        assert "messages" in row
        assert all("role" in m and "content" in m for m in row["messages"])


def test_build_dpo_wellformed():
    rows = build_dpo([_sample_group()])
    assert len(rows) == 3  # three preference pairs
    for row in rows:
        # trl reads prompt/chosen/rejected; _provenance is audit/curation metadata
        assert {"prompt", "chosen", "rejected"} <= set(row)
        assert "_provenance" in row
        assert isinstance(row["prompt"], list)
        # conversational trl.DPOTrainer shape: chosen/rejected are message lists
        assert isinstance(row["chosen"], list)
        assert isinstance(row["rejected"], list)
        assert row["chosen"][0]["role"] == "assistant"
        assert row["rejected"][0]["role"] == "assistant"
        assert "FULL_KERNEL" in row["chosen"][0]["content"]
        assert row["chosen"] != row["rejected"]


def test_build_rft_wellformed():
    rows = build_rft([_sample_group(), _sample_win()])
    assert len(rows) == 2
    for row in rows:
        assert "messages" in row
        assert row["messages"][-1]["role"] == "assistant"


def test_dedup_by_source_hash():
    a = _sample_win()
    b = _sample_win()  # identical final_source -> duplicate
    c = _sample_win()
    c.final_source = "def matmul(a,b): return b@a  # different"
    out = dedup_by_source_hash([a, b, c])
    assert len(out) == 2


def test_dedup_keeps_distinct_repair_transitions(monkeypatch=None):
    """audit R2 sft I1: two repairs that converge to the SAME fixed kernel from
    DIFFERENT broken parents are DISTINCT lessons and must both survive dedup; only a
    true (same parent + same fix) duplicate is dropped, and a repair never collides
    with a win whose final_source equals the fix."""
    r1 = _sample_repair()                       # parent deadbeef -> fix a@b
    r2 = _sample_repair(); r2.parent_hash = "cafef00d"   # DIFFERENT broken parent, same fix
    r_dup = _sample_repair()                    # identical to r1 (same parent + fix)
    w = _sample_win(); w.final_source = "def matmul(a,b):\n    return a@b"  # == the repair fix
    out = dedup_by_source_hash([r1, r2, r_dup, w])
    kinds = [type(x).__name__ for x in out]
    assert kinds.count("RepairRecord") == 2      # r1 + r2 kept, r_dup collapsed
    assert "WinRecord" in kinds                  # the win is NOT dropped by the repair fix


def test_leakage_split_no_group_crosses():
    recs = []
    for op in ("gemm", "softmax", "layernorm", "relu", "add", "conv"):
        for i in range(3):
            recs.append(
                RepairRecord(
                    task_id=f"{op}_bf16",
                    failure_class="snr_fail",
                    parent_hash=f"{op}{i}",
                    error_text="e",
                    messages=[{"role": "user", "content": "x"}],
                )
            )
    train, val, test = leakage_split(recs, by=("operation",), ratios=(0.6, 0.2, 0.2))

    def ops(split):
        return {r.task_id.split("_")[0] for r in split}

    tr, va, te = ops(train), ops(val), ops(test)
    # disjoint operations across splits
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    # nothing lost
    assert len(train) + len(val) + len(test) == len(recs)


def test_leakage_split_deterministic():
    recs = [
        RepairRecord(task_id=f"op{i}_bf16", failure_class="snr_fail",
                     parent_hash=str(i), error_text="e",
                     messages=[{"role": "user", "content": "x"}])
        for i in range(10)
    ]
    s1 = leakage_split(recs, by=("operation",), seed=7)
    s2 = leakage_split(recs, by=("operation",), seed=7)
    assert [len(x) for x in s1] == [len(x) for x in s2]
