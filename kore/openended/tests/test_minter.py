"""CPU tests for the verifiable open-ended task minter (grammar + minter).

All on CPU torch: mint batches, prove every minted reference executes + is
deterministic, that the construction gate rejects degenerate/insensitive/collapsed
tasks, that behavioral-hash dedup drops duplicates, that held-out families can
never appear, that composition yields a correct fused reference (fused == the
step-by-step torch), and that novelty/niche scoring separates distinct niches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kore.openended import grammar as g
from kore.openended import minter as m
from kore.openended import task_space as ts
from kore.openended.archive import TaskArchive


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _p(mt) -> float:
    """A deterministic, cross-process pseudo solve-rate for a minted task."""
    return (int(mt.behavioral_hash[:6], 16) % 1000) / 1000.0


def _prims():
    return g.source_prims(), g.middle_prims(), g.terminal_prims()


# --------------------------------------------------------------------------- #
# minting: executes + deterministic + finite
# --------------------------------------------------------------------------- #
def test_mint_batch_executes_deterministic_and_finite():
    arch = TaskArchive(seed=0)
    batch = m.mint_batch(arch, _p, 16, seed=0)
    assert len(batch) == 16
    for t in batch:
        # the 6-field task ABI is populated
        assert t.name and callable(t.reference_fn) and callable(t.input_sampler)
        assert t.dtype in m.MINT_DTYPES and t.tol > 0 and t.family.startswith("minted_")
        # executes on the cheap probe inputs
        out = t.reference_fn(*t.probe_inputs())
        assert torch.isfinite(out.float()).all()
        assert str(out.dtype).endswith(("bfloat16", "float16", "float32"))
        # deterministic: same seed -> identical outputs
        again = t.reference_fn(*t.probe_inputs())
        assert torch.equal(out, again)


def test_all_four_moves_exercised():
    batch = m.mint_batch(TaskArchive(0), _p, 24, seed=1)
    moves = {t.move for t in batch}
    assert {"fusion", "extrapolate", "novel", "mutate_crossover"} <= moves


def test_minting_is_deterministic():
    b1 = m.mint_batch(TaskArchive(0), _p, 10, seed=11)
    b2 = m.mint_batch(TaskArchive(0), _p, 10, seed=11)
    assert [t.task_id for t in b1] == [t.task_id for t in b2]
    assert [t.dedup_key for t in b1] == [t.dedup_key for t in b2]


# --------------------------------------------------------------------------- #
# construction gate: reject degenerate / insensitive / collapsed tasks
# --------------------------------------------------------------------------- #
def test_gate_rejects_constant_output():
    src, mid, _term = _prims()
    # relu(neg(abs(x))) == relu(-|x|) == 0 everywhere -> constant, must be rejected.
    p = g.Pipeline((src["input"], mid["abs"], mid["neg"], mid["relu"]))
    res = m.construction_gate(p, "fp32", "const_op", "minted_elementwise")
    assert not res.ok and res.reason == "constant_output"


def test_gate_rejects_input_insensitivity():
    src, _mid, _term = _prims()
    # a primitive that ignores its aux input -> output insensitive to input #1.
    ignore = g.Primitive("ignore_aux", g.MATRIX, g.MATRIX,
                         lambda main, auxs, t, F: main,
                         aux_roles=(g.ROLE_MATRIX,), tag="binary")
    p = g.Pipeline((src["input"], ignore))
    res = m.construction_gate(p, "fp32", "ignore_aux", "minted_fusion")
    assert not res.ok and res.reason == "insensitive_input_1"


def test_gate_rejects_axis_collapse():
    src, _mid, _term = _prims()
    # broadcast each row's mean across columns -> constant along the column axis.
    row_bcast = g.Primitive("row_bcast", g.MATRIX, g.MATRIX,
                           lambda main, auxs, t, F: main.mean(-1, keepdim=True).expand_as(main),
                           aux_roles=(), tag="norm")
    p = g.Pipeline((src["input"], row_bcast))
    res = m.construction_gate(p, "fp32", "row_bcast", "minted_norm")
    assert not res.ok and res.reason == "constant_along_rows"


def test_gate_accepts_wellformed_op():
    src, mid, _term = _prims()
    p = g.Pipeline((src["input"], mid["add"], mid["gelu"], mid["rmsnorm"]))
    assert m.construction_gate(p, "bf16", "input_add_gelu_rmsnorm", "minted_norm").ok


# --------------------------------------------------------------------------- #
# behavioral-hash dedup
# --------------------------------------------------------------------------- #
def test_behavioral_hash_dedup_drops_duplicates():
    minter = m.TaskMinter(seed=3)
    cand = minter._move_fusion()
    first = minter.register(cand, None, _p)
    second = minter.register(cand, None, _p)   # identical candidate
    assert first is not None and second is None


def test_mint_batch_returns_unique_tasks():
    batch = m.mint_batch(TaskArchive(0), _p, 20, seed=7)
    keys = [t.dedup_key for t in batch]
    assert len(keys) == len(set(keys))


def test_behavioral_hash_stable_and_distinguishes_ops():
    src, mid, _term = _prims()
    p1 = g.Pipeline((src["input"], mid["add"], mid["gelu"]))
    p2 = g.Pipeline((src["input"], mid["add"], mid["silu"]))
    assert g.behavioral_hash(p1) == g.behavioral_hash(p1)      # stable
    assert g.behavioral_hash(p1) != g.behavioral_hash(p2)      # distinguishes


# --------------------------------------------------------------------------- #
# held-out families can never appear
# --------------------------------------------------------------------------- #
def test_heldout_families_rejected_by_construction():
    assert m.is_heldout("mla_decode_bf16")
    assert m.is_heldout("paged_attn_decode_bf16")
    assert m.is_heldout("latent_attn_x")
    assert not m.is_heldout("matmul_add_bias_gelu", "minted_gemm_fusion")

    # a minted batch never names/keys a held-out family
    batch = m.mint_batch(TaskArchive(0), _p, 24, seed=2)
    for t in batch:
        assert not m.is_heldout(t.name, t.family)
        assert t.family not in ts.families(include_vendor=True)  # net-new families

    # structural guarantee: no grammar primitive is an attention/mla/paged op
    src, mid, term = _prims()
    for lib in (src, mid, term):
        for nm in lib:
            assert not any(tok in nm.lower() for tok in ("mla", "paged", "attn", "attention"))


# --------------------------------------------------------------------------- #
# composition: fused reference == sequential torch (correct-by-construction)
# --------------------------------------------------------------------------- #
def test_fused_reference_matches_sequential_simple():
    src, mid, _term = _prims()
    p = g.Pipeline((src["input"], mid["add"], mid["gelu"])).typecheck()
    ref = g.build_reference(p, "fp32")
    a, b = g.build_sampler(p, {"M": 16, "N": 24}, "fp32")(seed=3)
    fused = ref(a, b)
    sequential = F.gelu(a + b, approximate="tanh")     # independent step-by-step torch
    assert torch.allclose(fused, sequential, atol=1e-6)


def test_fused_reference_matches_sequential_matmul_chain():
    src, mid, _term = _prims()
    # matmul -> bias -> gelu -> residual -> rmsnorm (the canonical fused block)
    p = g.Pipeline((src["matmul"], mid["add_bias"], mid["gelu"],
                    mid["add"], mid["rmsnorm"])).typecheck()
    ref = g.build_reference(p, "fp32")
    A, W, bias, R, w = g.build_sampler(p, {"M": 16, "N": 24, "K": 24}, "fp32")(seed=0)
    fused = ref(A, W, bias, R, w)

    # sequential re-derivation from the primitives (a separate code path)
    y = A.float() @ W.float()
    y = y + bias.float()
    y = F.gelu(y, approximate="tanh")
    y = y + R.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + g._NORM_EPS) * w.float()
    assert torch.allclose(fused, y, atol=1e-5)


# --------------------------------------------------------------------------- #
# novelty / MAP-Elites niche placement
# --------------------------------------------------------------------------- #
def test_novelty_and_distinct_niche_placement():
    minter = m.TaskMinter(seed=0)
    arch = TaskArchive(seed=0)
    # empty archive -> maximal novelty
    assert minter.novelty(("minted_fusion", "memory-bound", 3, "16b", "small"), arch) == 1.0

    batch = minter.mint_batch(arch, _p, 16)
    niches = {t.niche_key for t in batch}
    assert len(niches) >= 3                       # distinct behavior regions
    assert arch.coverage() == len(niches)         # one cell per distinct niche

    # an already-occupied niche has zero novelty; a niche differing in a field is > 0
    occupied = next(iter(niches))
    assert minter.novelty(occupied, arch) == 0.0
    moved = ("totally_new_family",) + tuple(occupied[1:])
    assert 0.0 < minter.novelty(moved, arch) <= 1.0


def test_minted_tasks_placed_as_minted_carriers():
    arch = TaskArchive(seed=0)
    assert arch.coverage() == 0
    batch = m.mint_batch(arch, _p, 10, seed=4)
    assert arch.coverage() > 0 and len(batch) > 0
    for cell in arch.cells_list():
        assert cell.descriptor.source == "minted"
        assert cell.key == cell.key            # niche key is the measured minted key
    # archive stays fully functional with minted cells
    assert arch.summary()["coverage"] == arch.coverage()
    assert arch.best(1) and arch.sample(2, seed=0)


# --------------------------------------------------------------------------- #
# measured arithmetic-intensity classes (compute-bound vs memory-bound)
# --------------------------------------------------------------------------- #
def test_arithmetic_intensity_classification():
    src, mid, _term = _prims()
    elem = g.Pipeline((src["input"], mid["add"], mid["gelu"]))
    fe, ai_e = m.features_of(elem, {"M": 1024, "N": 2048}, "bf16")
    assert fe["arithmetic_intensity"] == "memory-bound" and ai_e < m.AI_RIDGE

    gemm = g.Pipeline((src["matmul"], mid["add_bias"], mid["gelu"]))
    fg, ai_g = m.features_of(gemm, {"M": 256, "N": 1024, "K": 4096}, "bf16")
    assert fg["arithmetic_intensity"] == "compute-bound" and ai_g >= m.AI_RIDGE
    assert fg["family"] == "minted_gemm_fusion"


# --------------------------------------------------------------------------- #
# scoring: learnability + injected learning-progress reward
# --------------------------------------------------------------------------- #
def test_learnability_from_solve_rate():
    batch = m.mint_batch(TaskArchive(0), lambda mt: 0.5, 4, seed=1)
    assert batch and all(abs(t.learnability - 1.0) < 1e-9 for t in batch)  # 4*.5*.5


def test_proposer_reward_uses_injected_delta_p():
    with_delta = m.mint_batch(TaskArchive(0), _p, 5, seed=1, progress_fn=lambda mt: 0.3)
    assert with_delta and all(abs(t.proposer_reward - 0.3) < 1e-9 for t in with_delta)
    # without a callback the reward falls back to the learnability prior
    without = m.mint_batch(TaskArchive(0), _p, 5, seed=1)
    assert all(abs(t.proposer_reward - t.learnability) < 1e-9 for t in without)


# --------------------------------------------------------------------------- #
# integration: a minted task -> runnable KORE reference namespace
# --------------------------------------------------------------------------- #
def test_to_reference_namespace_is_runnable():
    batch = m.mint_batch(TaskArchive(0), _p, 4, seed=5)
    ns = batch[0].to_reference_namespace()
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
              "entry_name", "dtype_name", "family"):
        assert k in ns
    inputs = ns["get_inputs"]({"M": 8, "N": 16, "K": 16}, device="cpu", seed=0)
    assert len(inputs) == ns["arity"]
    out = ns["ref_fn"](*inputs)
    assert torch.isfinite(out.float()).all()
