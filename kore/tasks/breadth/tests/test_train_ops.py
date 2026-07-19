"""CPU-only tests for the breadth training-critical op authoring engine.

Every ``ref_fn`` oracle is checked against an INDEPENDENT torch computation:
  * losses vs hand-rolled formulas AND the ``torch.nn.functional`` op,
  * ``fused_adamw`` vs ``torch.optim.AdamW`` (state-injected single step),
  * ``fused_lion`` vs the Lion update formula,
  * ``fused_muon`` vs an independently-written Newton-Schulz iteration AND the
    orthogonalization property (shares the gradient's singular vectors),
  * ``grad_clip_global_norm`` vs ``torch.nn.utils.clip_grad_norm_``.

Plus the ABI / arity / seed-compiles / shapes-parse / mutates_input contract.
All CPU-only (torch CPU tensors); no GPU is touched.
"""

from __future__ import annotations

import ast

import pytest
import torch
import torch.nn.functional as F

from kore.tasks.breadth import train_ops as T

LOSSES = ("cross_entropy", "fused_linear_cross_entropy", "kl_div",
          "label_smoothing_ce", "mse_loss")
OPTIMIZERS = ("fused_adamw", "fused_lion", "fused_muon", "grad_clip_global_norm")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clone(inp):
    """Clone tensor inputs (scalars pass through) - mirrors the driver's per-call
    clone so an in-place optimizer ref cannot corrupt the shared inputs."""
    return tuple(t.clone() if torch.is_tensor(t) else t for t in inp)


def _close(a, b, atol=1e-5, rtol=1e-5):
    return torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)


def _shape_str(s: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in s.items())


def _all_shapes(op: str) -> list[dict]:
    sh = T.SHAPES[op]
    return [sh["minimal"], sh["primary"], *sh["validation"]]


# --------------------------------------------------------------------------- #
# ABI surface
# --------------------------------------------------------------------------- #
def test_abi_surface():
    assert isinstance(T.OPS, tuple) and len(T.OPS) == 9
    assert set(T.OPS) == set(LOSSES) | set(OPTIMIZERS)
    assert set(T.OP_DTYPES) == set(T.OPS)          # complete per-op dtype map
    assert set(T.SHAPES) == set(T.OPS)
    assert T.DEFAULT_DTYPES == ("bf16", "fp16")
    for op in T.OPS:
        assert T.op_dtypes(op) == T.OP_DTYPES[op]
        assert all(dt in ("bf16", "fp16", "fp32") for dt in T.op_dtypes(op))
    # engine mirrors the vendor_ops ABI (same callables/collections exist)
    for attr in ("OPS", "OP_DTYPES", "SHAPES", "make_reference", "seed_source", "op_dtypes"):
        assert hasattr(T, attr)


def test_torch_imported_lazily():
    """Registry discovery must be GPU-free: torch is imported INSIDE the GPU paths,
    never at module scope (mirrors vendor_ops). Checked via the AST top-level import
    statements only (so the `import torch` text inside the seed STRINGS is ignored)."""
    import inspect
    tree = ast.parse(inspect.getsource(T))
    for node in tree.body:                            # module top level only
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("torch") for a in node.names), \
                "train_ops imports torch at module scope; must be lazy"
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("torch"), \
                "train_ops imports torch at module scope; must be lazy"


# --------------------------------------------------------------------------- #
# seed compiles + defines its entry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", T.OPS)
def test_seed_parses_compiles_defines_entry(op):
    for dt in T.op_dtypes(op):
        src = T.seed_source(op, dt)
        tree = ast.parse(src)                        # valid python
        compile(src, f"<{op}:{dt}>", "exec")         # syntactically compiles
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert op in funcs, f"seed for {op}/{dt} does not define entry {op!r}"


@pytest.mark.parametrize("op", T.OPS)
def test_seed_loads_as_module(op, tmp_path):
    """Stronger 'compiles' check: import the seed from a real file so @triton.jit
    binds each kernel + the entry (no kernel is LAUNCHED, so this stays CPU-safe;
    triton.jit needs a file on disk to read the kernel source at decoration time)."""
    import importlib.util

    pytest.importorskip("triton")
    dt = T.op_dtypes(op)[0]
    path = tmp_path / f"seed_{op}.py"
    path.write_text(T.seed_source(op, dt))
    spec = importlib.util.spec_from_file_location(f"breadth_seed_{op}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(getattr(mod, op))


# --------------------------------------------------------------------------- #
# reference namespace / arity / mutates_input / shapes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", T.OPS)
def test_reference_namespace(op):
    for dt in T.op_dtypes(op):
        ns = T.make_reference(op, dt)
        for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
                  "entry_name", "dtype_name", "family", "mutates_input"):
            assert k in ns, f"{op}/{dt} missing ns key {k!r}"
        assert ns["entry_name"] == op
        assert ns["dtype_name"] == dt
        assert ns["family"] == f"breadth_{op}"
        assert isinstance(ns["arity"], int) and ns["arity"] > 0
        assert ns[f"{op}_ref"] is ns["ref_fn"]       # conventional alias


@pytest.mark.parametrize("op", T.OPS)
def test_arity_matches_get_inputs(op):
    ns = T.make_reference(op, "fp32")
    inp = ns["get_inputs"](T.SHAPES[op]["minimal"], device="cpu", seed=0)
    assert ns["arity"] == len(inp)


@pytest.mark.parametrize("op", T.OPS)
def test_mutates_input_flag(op):
    ns = T.make_reference(op, "bf16")
    expected = op in OPTIMIZERS
    assert ns["mutates_input"] is expected
    assert (op in T.TRAIN_MUTATES_INPUT) is expected


def test_losses_pure_optimizers_inplace():
    for op in LOSSES:
        assert T.make_reference(op, "bf16")["mutates_input"] is False
    for op in OPTIMIZERS:
        assert T.make_reference(op, "bf16")["mutates_input"] is True


@pytest.mark.parametrize("op", T.OPS)
def test_shapes_parse(op):
    ns = T.make_reference(op, "fp32")
    ps = ns["parse_shape"]
    sh = T.SHAPES[op]
    assert {"minimal", "primary", "validation"} <= set(sh)
    assert isinstance(sh["validation"], list) and sh["validation"]
    for s in _all_shapes(op):
        assert ps(_shape_str(s)) == s                # round-trips through the driver parser


@pytest.mark.parametrize("op", T.OPS)
def test_get_inputs_dtype(op):
    ns = T.make_reference(op, "bf16")
    inp = ns["get_inputs"](T.SHAPES[op]["minimal"], device="cpu", seed=0)
    for t in inp:
        if torch.is_tensor(t):
            if t.is_floating_point():
                assert t.dtype == torch.bfloat16
            else:
                assert t.dtype == torch.int64      # target indices


# --------------------------------------------------------------------------- #
# LOSSES: ref_fn vs independent torch
# --------------------------------------------------------------------------- #
def test_cross_entropy_matches_torch():
    ns = T.make_reference("cross_entropy", "fp32")
    inp = ns["get_inputs"]({"M": 32, "V": 64}, device="cpu", seed=0)
    r = ns["ref_fn"](*_clone(inp))
    assert r.ndim == 0 and r.dtype == inp[0].dtype
    assert _close(r, F.cross_entropy(inp[0].float(), inp[1]))


def test_fused_linear_cross_entropy_matches_torch():
    ns = T.make_reference("fused_linear_cross_entropy", "fp32")
    inp = ns["get_inputs"]({"M": 16, "H": 32, "V": 48}, device="cpu", seed=1)
    x, w, tgt = inp
    r = ns["ref_fn"](*_clone(inp))
    logits = torch.einsum("mh,vh->mv", x.float(), w.float())   # independent (x @ W^T)
    assert _close(r, F.cross_entropy(logits, tgt))


def test_kl_div_matches_batchmean():
    ns = T.make_reference("kl_div", "fp32")
    inp = ns["get_inputs"]({"M": 16, "V": 40}, device="cpu", seed=2)
    log_p, q = inp
    r = ns["ref_fn"](*_clone(inp))
    manual = (q.float() * (q.float().log() - log_p.float())).sum() / log_p.shape[0]
    assert _close(r, manual)
    assert _close(r, F.kl_div(log_p.float(), q.float(), reduction="batchmean"))


def test_label_smoothing_ce_matches_formula():
    ns = T.make_reference("label_smoothing_ce", "fp32")
    inp = ns["get_inputs"]({"M": 24, "V": 50}, device="cpu", seed=3)
    logits, tgt = inp
    r = ns["ref_fn"](*_clone(inp))
    lp = F.log_softmax(logits.float(), dim=-1)
    manual = (1 - T.LS_EPS) * F.nll_loss(lp, tgt) + T.LS_EPS * (-lp.mean(dim=-1).mean())
    assert _close(r, manual)
    # smoothing is actually applied (differs from vanilla CE)
    assert not _close(r, F.cross_entropy(logits.float(), tgt), atol=1e-3, rtol=1e-3)


def test_mse_loss_matches_torch():
    ns = T.make_reference("mse_loss", "fp32")
    inp = ns["get_inputs"]({"M": 16, "N": 32}, device="cpu", seed=4)
    r = ns["ref_fn"](*_clone(inp))
    assert _close(r, ((inp[0].float() - inp[1].float()) ** 2).mean())
    assert _close(r, F.mse_loss(inp[0].float(), inp[1].float()))


# --------------------------------------------------------------------------- #
# OPTIMIZERS: ref_fn vs independent torch (returns updated param(s))
# --------------------------------------------------------------------------- #
def test_fused_adamw_matches_torch_optim():
    ns = T.make_reference("fused_adamw", "fp32")
    inp = ns["get_inputs"]({"M": 8, "N": 16}, device="cpu", seed=5)
    param, grad, exp_avg, exp_avg_sq, lr, b1, b2, eps, wd, step = inp
    p_new, m_new, v_new = ns["ref_fn"](*_clone(inp))

    # INDEPENDENT: a real torch.optim.AdamW single step from the injected state.
    p = param.clone().detach().requires_grad_(True)
    p.grad = grad.clone()
    opt = torch.optim.AdamW([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    st = opt.state[p]
    st["step"] = torch.tensor(float(step - 1))      # -> bias-correction exponent = step
    st["exp_avg"] = exp_avg.clone()
    st["exp_avg_sq"] = exp_avg_sq.clone()
    opt.step()

    assert _close(p_new, p.detach())
    assert _close(m_new, st["exp_avg"])
    assert _close(v_new, st["exp_avg_sq"])
    assert not _close(p_new, param)                 # the step actually moved the param


def test_fused_lion_matches_formula():
    ns = T.make_reference("fused_lion", "fp32")
    inp = ns["get_inputs"]({"M": 8, "N": 16}, device="cpu", seed=6)
    param, grad, exp_avg, lr, b1, b2, wd = inp
    p_new, m_new = ns["ref_fn"](*_clone(inp))

    update = torch.sign(b1 * exp_avg.float() + (1 - b1) * grad.float())
    p_ind = param.float() - lr * (update + wd * param.float())
    m_ind = b2 * exp_avg.float() + (1 - b2) * grad.float()
    assert _close(p_new, p_ind)
    assert _close(m_new, m_ind)
    assert not _close(p_new, param)


def test_fused_muon_newton_schulz_and_update():
    ns = T.make_reference("fused_muon", "fp32")
    inp = ns["get_inputs"]({"M": 6, "N": 10}, device="cpu", seed=7)
    param, grad, buf0, lr, mu = inp
    p_new, buf_new = ns["ref_fn"](*_clone(inp))

    # momentum buffer (nesterov lerp form)
    buf_ind = mu * buf0.float() + (1 - mu) * grad.float()
    assert _close(buf_new, buf_ind)
    g_eff = (1 - mu) * grad.float() + mu * buf_ind

    # INDEPENDENT Newton-Schulz (separately written, matmul form) on g_eff.
    a, b, c = T.NS_COEFFS
    X = g_eff.clone()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.t().contiguous()
    X = X / (torch.sqrt((X * X).sum()) + 1e-7)
    for _ in range(T.NS_STEPS):
        A = torch.matmul(X, X.t())
        X = a * X + torch.matmul(b * A + c * torch.matmul(A, A), X)
    o_ind = X.t() if transposed else X

    scale = max(1.0, param.shape[0] / param.shape[1]) ** 0.5
    o_rec = (param.float() - p_new.float()) / (lr * scale)   # recover orthogonalized grad
    assert _close(o_rec, o_ind, atol=1e-4, rtol=1e-4)

    # ORTHOGONALIZATION property (independent of the iteration itself): the NS output
    # shares g_eff's singular vectors, i.e. U^T O V is diagonal.
    U, S, Vh = torch.linalg.svd(g_eff, full_matrices=False)
    D = U.mT @ o_rec @ Vh.mT
    off_diag = (D - torch.diag(torch.diag(D))).abs().max()
    assert off_diag < 1e-3, f"NS is not a valid orthogonalization: off-diag={off_diag}"


def test_grad_clip_matches_clip_grad_norm():
    ns = T.make_reference("grad_clip_global_norm", "fp32")
    inp = ns["get_inputs"]({"G": 4, "N": 10}, device="cpu", seed=8)
    grads, max_norm = inp
    clipped = ns["ref_fn"](*_clone(inp))

    # INDEPENDENT: torch.nn.utils.clip_grad_norm_ over the equivalent list of grads.
    ps = [torch.zeros(grads.shape[1], requires_grad=True) for _ in range(grads.shape[0])]
    for i, pp in enumerate(ps):
        pp.grad = grads[i].clone()
    total = torch.nn.utils.clip_grad_norm_(ps, max_norm)
    torch_clipped = torch.stack([pp.grad for pp in ps])

    assert _close(clipped, torch_clipped)
    assert _close(total, grads.float().norm())                       # reported global norm
    assert grads.float().norm() > max_norm                           # clipping is active here
    assert _close(clipped.float().norm(), torch.tensor(float(max_norm)), atol=1e-3, rtol=1e-3)


def test_grad_clip_noop_below_threshold():
    ns = T.make_reference("grad_clip_global_norm", "fp32")
    grads = torch.randn(4, 10) * 1e-3                                # global norm << max_norm
    clipped = ns["ref_fn"](grads.clone(), 10.0)
    assert _close(clipped, grads)                                    # unchanged (coef == 1)


# --------------------------------------------------------------------------- #
# baseline_fn is runnable and structurally matches the oracle (perf bar sanity)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", T.OPS)
def test_baseline_fn_runs_and_matches_structure(op):
    ns = T.make_reference(op, "fp32")
    inp = ns["get_inputs"](T.SHAPES[op]["minimal"], device="cpu", seed=0)
    ref_out = ns["ref_fn"](*_clone(inp))
    base_out = ns["baseline_fn"](*_clone(inp))
    ref_t = ref_out if isinstance(ref_out, tuple) else (ref_out,)
    base_t = base_out if isinstance(base_out, tuple) else (base_out,)
    assert len(ref_t) == len(base_t)
    for r, b in zip(ref_t, base_t):
        assert r.shape == b.shape
        assert _close(r, b, atol=1e-2, rtol=1e-2)    # fp32 baseline == fp32 oracle
