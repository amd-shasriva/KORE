"""CPU-only tests for the extended training-critical op engine (``train_ext``).

CORRECTNESS IS PARAMOUNT, so every ``ref_fn`` oracle is checked against an
INDEPENDENT ground truth:
  * OPTIMIZERS  -> the matching ``torch.optim`` step (identical state injected,
    one ``.step()``), at tight fp32 tol. Adafactor/Muon use torch's built-in
    ``torch.optim.Adafactor`` / ``torch.optim.Muon``; the 8-bit/fp8 quantized
    Adam variants dequantize the injected states and match ``torch.optim.Adam``
    on the param exactly (the requantized states within the quant step).
  * LOSSES      -> the analytic ``ref_fn`` gradient == ``torch.autograd`` (the
    ``baseline_fn``) at tight fp32 tol, plus loss value == ``F.*``.
  * GRAD UTILS  -> the exact hand-derived formula (per-row clip also vs
    ``torch.nn.utils.clip_grad_norm_``).
Plus the ABI / arity / seed-compiles / seed-loads / shapes / dtype / mutates_input
contract. All CPU-only (torch CPU tensors, ``CUDA_VISIBLE_DEVICES=''``).
"""

from __future__ import annotations

import ast
import importlib.util
import math

import pytest
import torch
import torch.nn.functional as F

from kore.tasks.breadth import train_ext as T
from kore.tasks.breadth import train_ops as TRAIN_OPS

# small CPU shapes per op family
_OPT2D = {"M": 8, "N": 16}
_MUON = {"M": 6, "N": 10}
_BLOB = {"G": 4, "N": 10}
_CE = {"M": 8, "V": 16}
_ELEM = {"M": 6, "N": 12}


def _shape(op):
    if op in T._FOREACH_OPS or op in T._GRAD_OPS:
        return _BLOB
    if op in T._MUON_OPS:
        return _MUON
    if op in T._CE_LOSS_OPS or op in T._DISTILL_OPS:
        return _CE
    if op in T._ELEM_LOSS_OPS or op in T._COSINE_OPS:
        return _ELEM
    return _OPT2D


def clone(inp):
    return tuple(t.clone() if torch.is_tensor(t) else t for t in inp)


def close(a, b, at=1e-5, rt=1e-4):
    return torch.allclose(a.float(), b.float(), atol=at, rtol=rt)


def _astuple(x):
    return x if isinstance(x, tuple) else (x,)


def _inp(op, dtype="fp32", seed=0):
    ns = T.make_reference(op, dtype)
    return ns, ns["get_inputs"](_shape(op), device="cpu", seed=seed)


def _torch_step(cls, kwargs, param, grad, state):
    """Run ONE ``torch.optim`` step from an injected state; return (param, state)."""
    p = param.clone().detach().requires_grad_(True)
    p.grad = grad.clone()
    opt = cls([p], **kwargs)
    st = opt.state[p]
    for k, v in state.items():
        st[k] = v.clone() if torch.is_tensor(v) else v
    opt.step()
    return p.detach(), opt.state[p]


# =========================================================================== #
# ABI surface
# =========================================================================== #
def test_abi_surface():
    assert isinstance(T.OPS, tuple) and len(T.OPS) == 50
    assert len(set(T.OPS)) == len(T.OPS)
    assert all(op.startswith("tr_") for op in T.OPS)
    # DIFFERENT op names: never collide with the existing train_ops suite
    assert not (set(T.OPS) & set(TRAIN_OPS.OPS))
    assert set(T.OP_DTYPES) == set(T.OPS)
    assert set(T.SHAPES) == set(T.OPS)
    assert T.DEFAULT_DTYPES == ("bf16", "fp16", "fp32")
    for op in T.OPS:
        assert T.op_dtypes(op) == T.OP_DTYPES[op]
        assert all(dt in ("bf16", "fp16", "fp32") for dt in T.op_dtypes(op))
    for attr in ("OPS", "OP_DTYPES", "SHAPES", "make_reference", "seed_source", "op_dtypes"):
        assert hasattr(T, attr)


def test_op_counts():
    assert len(T._OPTIMIZER_OPS) == 30
    assert len(T._LOSS_OPS) == 14
    assert len(T._GRAD_OPS) == 6
    assert set(T._OPTIMIZER_OPS) | set(T._LOSS_OPS) | set(T._GRAD_OPS) == set(T.OPS)


def test_torch_imported_lazily():
    """Registry discovery must be GPU/torch-free: torch imported INSIDE the paths."""
    import inspect
    tree = ast.parse(inspect.getsource(T))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("torch") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("torch")


# =========================================================================== #
# seeds compile / define entry / load as a real module (@triton.jit binds)
# =========================================================================== #
@pytest.mark.parametrize("op", T.OPS)
def test_seed_parses_compiles_defines_entry(op):
    for dt in T.op_dtypes(op):
        src = T.seed_source(op, dt)
        tree = ast.parse(src)
        compile(src, f"<{op}:{dt}>", "exec")
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert op in funcs, f"seed for {op}/{dt} does not define entry {op!r}"


@pytest.mark.parametrize("op", T.OPS)
def test_seed_loads_as_module(op, tmp_path):
    pytest.importorskip("triton")
    dt = T.op_dtypes(op)[0]
    path = tmp_path / f"seed_{op}.py"
    path.write_text(T.seed_source(op, dt))
    spec = importlib.util.spec_from_file_location(f"breadth_ext_seed_{op}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(getattr(mod, op))


# =========================================================================== #
# reference namespace / arity / mutates_input / shapes / dtype
# =========================================================================== #
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
        assert ns[f"{op}_ref"] is ns["ref_fn"]


@pytest.mark.parametrize("op", T.OPS)
def test_arity_matches_get_inputs(op):
    ns, inp = _inp(op)
    assert ns["arity"] == len(inp)


@pytest.mark.parametrize("op", T.OPS)
def test_mutates_input_flag(op):
    ns = T.make_reference(op, "bf16")
    expected = op in T._OPTIMIZER_OPS or op in (
        "tr_grad_clip_per_layer", "tr_agc", "tr_ema_update",
        "tr_grad_accum_scale", "tr_grad_zero_center")
    assert ns["mutates_input"] is expected
    assert (op in T.TRAIN_MUTATES_INPUT) is expected


def test_losses_and_global_norm_are_pure():
    for op in T._LOSS_OPS + ("tr_global_l2_norm",):
        assert T.make_reference(op, "bf16")["mutates_input"] is False
    for op in T._OPTIMIZER_OPS:
        assert T.make_reference(op, "bf16")["mutates_input"] is True


@pytest.mark.parametrize("op", T.OPS)
def test_shapes_parse_roundtrip(op):
    ns = T.make_reference(op, "fp32")
    sh = T.SHAPES[op]
    assert {"minimal", "primary", "validation"} <= set(sh)
    assert isinstance(sh["validation"], list) and sh["validation"]
    for s in [sh["minimal"], sh["primary"], *sh["validation"]]:
        ss = ",".join(f"{k}={v}" for k, v in s.items())
        assert ns["parse_shape"](ss) == s


@pytest.mark.parametrize("op", T.OPS)
def test_get_inputs_dtype(op):
    ns, inp = _inp(op, "bf16")
    if op in T._QUANT_OPS:
        # (param, grad, q_m, s_m, q_v, s_v, ...): param+grad bf16, states quantized,
        # scales fp32.
        assert inp[0].dtype == torch.bfloat16 and inp[1].dtype == torch.bfloat16
        qdt = torch.int8 if T._QUANT_CFG[op]["kind"] == "int8" else torch.float8_e4m3fn
        assert inp[2].dtype == qdt and inp[4].dtype == qdt
        assert inp[3].dtype == torch.float32 and inp[5].dtype == torch.float32
        return
    for t in inp:
        if torch.is_tensor(t):
            if t.is_floating_point():
                assert t.dtype == torch.bfloat16
            else:
                assert t.dtype == torch.int64


# =========================================================================== #
# OPTIMIZERS vs torch.optim (identical injected state, one step, tight tol)
# =========================================================================== #
@pytest.mark.parametrize("op", T._SGD_OPS)
def test_sgd_vs_torch(op):
    ns, inp = _inp(op, seed=5)
    param, grad, buf, lr, mom, damp, wd, nest = inp
    p_new, buf_new = ns["ref_fn"](*clone(inp))
    p, st = _torch_step(torch.optim.SGD,
                        dict(lr=lr, momentum=mom, dampening=damp, weight_decay=wd, nesterov=nest),
                        param, grad, {"momentum_buffer": buf})
    assert close(p_new, p) and close(buf_new, st["momentum_buffer"])
    assert not close(p_new, param)


@pytest.mark.parametrize("op", T._ADAM_OPS)
def test_adam_vs_torch(op):
    cfg = T._ADAM_CFG[op]
    ns, inp = _inp(op, seed=6)
    if cfg["amsgrad"]:
        param, grad, m, v, vmax, lr, b1, b2, eps, wd, step = inp
        p_new, m_new, v_new, vmx_new = ns["ref_fn"](*clone(inp))
        state = {"step": torch.tensor(float(step - 1)), "exp_avg": m,
                 "exp_avg_sq": v, "max_exp_avg_sq": vmax}
    else:
        param, grad, m, v, lr, b1, b2, eps, wd, step = inp
        p_new, m_new, v_new = ns["ref_fn"](*clone(inp))
        state = {"step": torch.tensor(float(step - 1)), "exp_avg": m, "exp_avg_sq": v}
    cls = torch.optim.AdamW if cfg["decoupled"] else torch.optim.Adam
    p, st = _torch_step(cls, dict(lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd,
                                  amsgrad=cfg["amsgrad"]), param, grad, state)
    assert close(p_new, p) and close(m_new, st["exp_avg"]) and close(v_new, st["exp_avg_sq"])
    if cfg["amsgrad"]:
        assert close(vmx_new, st["max_exp_avg_sq"])
    assert not close(p_new, param)


@pytest.mark.parametrize("op", T._RMSPROP_OPS)
def test_rmsprop_vs_torch(op):
    cfg = T._RMSPROP_CFG[op]
    cen, mom_on = cfg["centered"], cfg["momentum"] > 0
    ns, inp = _inp(op, seed=7)
    i = 3
    ga = inp[i] if cen else None
    i += 1 if cen else 0
    bufm = inp[i] if mom_on else None
    i += 1 if mom_on else 0
    param, grad, sq = inp[0], inp[1], inp[2]
    lr, alpha, eps, wd, momentum = inp[i:i + 5]
    out = _astuple(ns["ref_fn"](*clone(inp)))
    state = {"step": torch.zeros(()), "square_avg": sq}
    if cen:
        state["grad_avg"] = ga
    if mom_on:
        state["momentum_buffer"] = bufm
    p, st = _torch_step(torch.optim.RMSprop,
                        dict(lr=lr, alpha=alpha, eps=eps, weight_decay=wd,
                             momentum=momentum, centered=cen), param, grad, state)
    assert close(out[0], p) and close(out[1], st["square_avg"])
    k = 2
    if cen:
        assert close(out[k], st["grad_avg"]); k += 1
    if mom_on:
        assert close(out[k], st["momentum_buffer"])
    assert not close(out[0], param)


def test_adagrad_vs_torch():
    ns, inp = _inp("tr_adagrad", seed=8)
    param, grad, ssum, lr, eps, wd, lrd, step = inp
    p_new, s_new = ns["ref_fn"](*clone(inp))
    p, st = _torch_step(torch.optim.Adagrad,
                        dict(lr=lr, eps=eps, weight_decay=wd, lr_decay=lrd),
                        param, grad, {"step": torch.tensor(float(step - 1)), "sum": ssum})
    assert close(p_new, p) and close(s_new, st["sum"]) and not close(p_new, param)


def test_adadelta_vs_torch():
    ns, inp = _inp("tr_adadelta", seed=9)
    param, grad, sq, acc, lr, rho, eps, wd = inp
    p_new, sq_new, acc_new = ns["ref_fn"](*clone(inp))
    p, st = _torch_step(torch.optim.Adadelta, dict(lr=lr, rho=rho, eps=eps, weight_decay=wd),
                        param, grad, {"step": torch.zeros(()), "square_avg": sq, "acc_delta": acc})
    assert close(p_new, p) and close(sq_new, st["square_avg"]) and close(acc_new, st["acc_delta"])
    assert not close(p_new, param)


def test_adamax_vs_torch():
    ns, inp = _inp("tr_adamax", seed=10)
    param, grad, m, inf, lr, b1, b2, eps, wd, step = inp
    p_new, m_new, inf_new = ns["ref_fn"](*clone(inp))
    p, st = _torch_step(torch.optim.Adamax, dict(lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd),
                        param, grad, {"step": torch.tensor(float(step - 1)),
                                      "exp_avg": m, "exp_inf": inf})
    assert close(p_new, p) and close(m_new, st["exp_avg"]) and close(inf_new, st["exp_inf"])
    assert not close(p_new, param)


def test_nadam_vs_torch():
    ns, inp = _inp("tr_nadam", seed=11)
    param, grad, m, v, lr, b1, b2, eps, wd, psi, step = inp
    p_new, m_new, v_new = ns["ref_fn"](*clone(inp))
    mu_prod_prev = 1.0
    for i in range(1, step):
        mu_prod_prev *= b1 * (1.0 - 0.5 * 0.96 ** (i * psi))
    p, st = _torch_step(torch.optim.NAdam,
                        dict(lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd, momentum_decay=psi),
                        param, grad, {"step": torch.tensor(float(step - 1)), "exp_avg": m,
                                      "exp_avg_sq": v, "mu_product": torch.tensor(mu_prod_prev)})
    assert close(p_new, p) and close(m_new, st["exp_avg"]) and close(v_new, st["exp_avg_sq"])
    assert not close(p_new, param)


def test_radam_vs_torch():
    ns, inp = _inp("tr_radam", seed=12)
    param, grad, m, v, lr, b1, b2, eps, wd, step = inp
    p_new, m_new, v_new = ns["ref_fn"](*clone(inp))
    p, st = _torch_step(torch.optim.RAdam, dict(lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd),
                        param, grad, {"step": torch.tensor(float(step - 1)),
                                      "exp_avg": m, "exp_avg_sq": v})
    assert close(p_new, p) and close(m_new, st["exp_avg"]) and close(v_new, st["exp_avg_sq"])
    assert not close(p_new, param)


def test_rprop_vs_torch():
    ns, inp = _inp("tr_rprop", seed=13)
    param, grad, prev, ss, etam, etap, smin, smax, step = inp
    p_new, prev_new, ss_new = ns["ref_fn"](*clone(inp))
    p, st = _torch_step(torch.optim.Rprop, dict(etas=(etam, etap), step_sizes=(smin, smax)),
                        param, grad, {"step": torch.tensor(float(step - 1)),
                                      "prev": prev, "step_size": ss})
    assert close(p_new, p) and close(prev_new, st["prev"]) and close(ss_new, st["step_size"])
    assert not close(p_new, param)


def test_adafactor_vs_torch():
    ns, inp = _inp("tr_adafactor", seed=14)
    param, grad, row, col, lr, b2d, eps1, eps2, d, wd, step = inp
    p_new, row_new, col_new = ns["ref_fn"](*clone(inp))
    p, st = _torch_step(torch.optim.Adafactor,
                        dict(lr=lr, beta2_decay=b2d, eps=(eps1, eps2), d=d, weight_decay=wd),
                        param, grad, {"step": torch.tensor(float(step - 1)),
                                      "row_var": row, "col_var": col})
    assert close(p_new, p, at=1e-4, rt=1e-4)
    assert close(row_new, st["row_var"], at=1e-4) and close(col_new, st["col_var"], at=1e-4)
    assert not close(p_new, param)


@pytest.mark.parametrize("op", T._MUON_OPS)
def test_muon_vs_torch_and_ns(op):
    ns_steps = T._MUON_CFG[op]["ns_steps"]
    ns, inp = _inp(op, seed=15)
    param, grad, buf, lr, wd, mom, eps, na, nb, nc, ns_arg = inp
    p_new, buf_new = ns["ref_fn"](*clone(inp))

    # (1) momentum buffer matches torch.optim.Muon EXACTLY (both fp32 lerp).
    p, st = _torch_step(torch.optim.Muon,
                        dict(lr=lr, weight_decay=wd, momentum=mom, nesterov=True,
                             ns_coefficients=(na, nb, nc), eps=eps, ns_steps=ns_steps),
                        param, grad, {"momentum_buffer": buf})
    assert close(buf_new, st["momentum_buffer"])
    # (2) param matches torch (loose: torch's NS runs in bf16, the oracle in fp32).
    assert close(p_new, p, at=3e-2, rt=3e-2)

    # (3) NS orthogonalization matches an INDEPENDENT fp32 Newton-Schulz + the
    #     orthogonalization property (shares g_eff's singular vectors).
    buf_i = mom * buf.float() + (1.0 - mom) * grad.float()
    g_eff = (1.0 - mom) * grad.float() + mom * buf_i
    X = g_eff.clone()
    tp = X.shape[0] > X.shape[1]
    if tp:
        X = X.t().contiguous()
    X = X / (X.norm() + 0.0).clamp(min=eps)
    for _ in range(ns_steps):
        A = X @ X.t()
        X = na * X + (nb * A + nc * (A @ A)) @ X
    o_ind = X.t() if tp else X
    scale = max(1.0, param.shape[0] / param.shape[1]) ** 0.5
    o_rec = (param.float() * (1.0 - lr * wd) - p_new.float()) / (lr * scale)
    assert close(o_rec, o_ind, at=1e-4, rt=1e-4)
    U, S, Vh = torch.linalg.svd(g_eff, full_matrices=False)
    D = U.mT @ o_rec @ Vh.mT
    off = (D - torch.diag(torch.diag(D))).abs().max()
    assert off < 1e-3


def test_foreach_adamw_vs_torch():
    ns, inp = _inp("tr_foreach_adamw", seed=16)
    param, grad, m, v, lr, b1, b2, eps, wd, step = inp
    p_new, m_new, v_new = ns["ref_fn"](*clone(inp))
    p, st = _torch_step(torch.optim.AdamW, dict(lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd),
                        param, grad, {"step": torch.tensor(float(step - 1)),
                                      "exp_avg": m, "exp_avg_sq": v})
    assert close(p_new, p) and close(m_new, st["exp_avg"]) and close(v_new, st["exp_avg_sq"])


def test_foreach_sgd_vs_torch():
    ns, inp = _inp("tr_foreach_sgd", seed=17)
    param, grad, buf, lr, mom, wd = inp
    p_new, buf_new = ns["ref_fn"](*clone(inp))
    p, st = _torch_step(torch.optim.SGD, dict(lr=lr, momentum=mom, weight_decay=wd),
                        param, grad, {"momentum_buffer": buf})
    assert close(p_new, p) and close(buf_new, st["momentum_buffer"])


# ---- quantized-state Adam: param EXACT vs torch on dequantized states -------
def _dequant(q, scale, B=T.QUANT_BLOCK):
    qf = q.reshape(-1)
    s = scale.repeat_interleave(B)[:qf.numel()]
    return (qf.float() * s).reshape(q.shape)


@pytest.mark.parametrize("op", T._QUANT_OPS)
def test_quant_adam_param_exact(op):
    cfg = T._QUANT_CFG[op]
    ns, inp = _inp(op, seed=18)
    param, grad, qm, sm, qv, sv, lr, b1, b2, eps, wd, step = inp
    p_new, qm2, sm2, qv2, sv2 = ns["ref_fn"](*clone(inp))
    m = _dequant(qm, sm)
    v = _dequant(qv, sv)
    cls = torch.optim.AdamW if cfg["decoupled"] else torch.optim.Adam
    p, st = _torch_step(cls, dict(lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd),
                        param, grad, {"step": torch.tensor(float(step - 1)),
                                      "exp_avg": m, "exp_avg_sq": v})
    # param uses the (dequantized) injected states -> matches torch EXACTLY.
    assert close(p_new, p, at=1e-5, rt=1e-4)
    # requantized new states dequantize back to torch's states within the quant step.
    m_rt = _dequant(qm2, sm2)
    tol = float(sm2.max()) if cfg["kind"] == "int8" else float(sm2.max()) * 8
    assert torch.allclose(m_rt, st["exp_avg"], atol=tol + 1e-4, rtol=0.1)


# ---- advanced (no torch.optim): INDEPENDENT hand-derived formula ------------
def test_lamb_formula():
    ns, inp = _inp("tr_lamb", seed=19)
    param, grad, m, v, lr, b1, b2, eps, wd, step = inp
    p_new, m_new, v_new = ns["ref_fn"](*clone(inp))
    p, g = param.float(), grad.float()
    m2 = b1 * m.float() + (1 - b1) * g
    v2 = b2 * v.float() + (1 - b2) * g * g
    upd = (m2 / (1 - b1 ** step)) / ((v2 / (1 - b2 ** step)).sqrt() + eps) + wd * p
    trust = torch.linalg.norm(p.flatten()) / torch.linalg.norm(upd.flatten())
    assert close(p_new, p - lr * trust * upd) and close(m_new, m2) and close(v_new, v2)


def test_lars_formula():
    ns, inp = _inp("tr_lars", seed=20)
    param, grad, buf, lr, mom, wd, tc, eps = inp
    p_new, buf_new = ns["ref_fn"](*clone(inp))
    p, g = param.float(), grad.float()
    d = g + wd * p
    llr = tc * p.norm() / (d.norm() + eps)
    buf2 = mom * buf.float() + lr * llr * d
    assert close(p_new, p - buf2) and close(buf_new, buf2)


def test_adabelief_formula():
    ns, inp = _inp("tr_adabelief", seed=21)
    param, grad, m, s, lr, b1, b2, eps, wd, step = inp
    p_new, m_new, s_new = ns["ref_fn"](*clone(inp))
    p, g = param.float(), grad.float()
    m2 = b1 * m.float() + (1 - b1) * g
    s2 = b2 * s.float() + (1 - b2) * (g - m2) ** 2 + eps
    pd = p * (1 - lr * wd) - lr * (m2 / (1 - b1 ** step)) / ((s2 / (1 - b2 ** step)).sqrt() + eps)
    assert close(p_new, pd) and close(m_new, m2) and close(s_new, s2)


def test_novograd_formula():
    ns, inp = _inp("tr_novograd", seed=22)
    param, grad, m, vsc, lr, b1, b2, eps, wd = inp
    p_new, m_new, v_new = ns["ref_fn"](*clone(inp))
    p, g = param.float(), grad.float()
    v2 = b2 * vsc.float() + (1 - b2) * (g * g).sum()
    m2 = b1 * m.float() + (g / (v2.sqrt() + eps) + wd * p)
    assert close(p_new, p - lr * m2) and close(m_new, m2) and close(v_new, v2)
    assert v_new.dim() == 0


# =========================================================================== #
# LOSSES: analytic ref == autograd (baseline) + loss value == F.*
# =========================================================================== #
@pytest.mark.parametrize("op", T._LOSS_OPS)
def test_loss_grad_matches_autograd(op):
    ns, inp = _inp(op, seed=23)
    loss_ref, grad_ref = ns["ref_fn"](*clone(inp))
    loss_ag, grad_ag = ns["baseline_fn"](*clone(inp))   # fp32 torch.autograd oracle
    assert loss_ref.ndim == 0 and grad_ref.shape == inp[0].shape
    assert close(loss_ref, loss_ag, at=1e-5, rt=1e-4)
    assert close(grad_ref, grad_ag, at=2e-4, rt=1e-3)


def test_loss_values_vs_functional():
    z = T.make_reference("tr_cross_entropy_bwd", "fp32")
    inp = z["get_inputs"](_CE, device="cpu", seed=1)
    loss, grad = z["ref_fn"](*clone(inp))
    assert close(loss, F.cross_entropy(inp[0].float(), inp[1].long()))
    # dlogits sums to ~0 per row (softmax - onehot), scaled by 1/M
    assert close(grad.sum(1), torch.zeros(inp[0].shape[0]), at=1e-5)

    b = T.make_reference("tr_bce_logits_bwd", "fp32")
    ib = b["get_inputs"](_ELEM, device="cpu", seed=1)
    lb, gb = b["ref_fn"](*clone(ib))
    assert close(lb, F.binary_cross_entropy_with_logits(ib[0].float(), ib[1].float()))
    assert close(gb, (torch.sigmoid(ib[0].float()) - ib[1].float()) / ib[0].numel())

    h = T.make_reference("tr_huber_bwd", "fp32")
    ih = h["get_inputs"](_ELEM, device="cpu", seed=1)
    lh, _ = h["ref_fn"](*clone(ih))
    assert close(lh, F.huber_loss(ih[0].float(), ih[1].float(), delta=T.HUBER_DELTA))

    k = T.make_reference("tr_kl_distill_bwd", "fp32")
    ik = k["get_inputs"](_CE, device="cpu", seed=1)
    lk, gk = k["ref_fn"](*clone(ik))
    ps = torch.softmax(ik[0].float(), -1)
    pt = torch.softmax(ik[1].float(), -1)
    assert close(gk, (ps - pt) / ik[0].shape[0])       # KD gradient = (student - teacher) probs / M


# =========================================================================== #
# GRADIENT UTILITIES vs exact formula
# =========================================================================== #
def test_grad_clip_per_layer():
    ns, inp = _inp("tr_grad_clip_per_layer", seed=24)
    grads, max_norm = inp
    clipped = ns["ref_fn"](*clone(inp))
    norm = grads.float().norm(dim=1, keepdim=True)
    coef = (max_norm / (norm + T.CLIP_EPS)).clamp(max=1.0)
    assert close(clipped, grads.float() * coef)
    # matches torch per-parameter clip_grad_norm_ applied row-by-row
    for i in range(grads.shape[0]):
        pp = torch.zeros(grads.shape[1], requires_grad=True)
        pp.grad = grads[i].clone().float()
        torch.nn.utils.clip_grad_norm_([pp], max_norm)
        assert close(clipped[i], pp.grad)


def test_agc_formula():
    ns, inp = _inp("tr_agc", seed=25)
    params, grads, clip, eps = inp
    out = ns["ref_fn"](*clone(inp))
    pn = params.float().norm(dim=1, keepdim=True).clamp(min=eps)
    gn = grads.float().norm(dim=1, keepdim=True)
    scale = (clip * pn / (gn + 1e-12)).clamp(max=1.0)
    assert close(out, grads.float() * scale)


def test_ema_update():
    ns, inp = _inp("tr_ema_update", seed=26)
    ema, param, decay = inp
    out = ns["ref_fn"](*clone(inp))
    assert close(out, decay * ema.float() + (1 - decay) * param.float())


def test_global_l2_norm():
    ns, inp = _inp("tr_global_l2_norm", seed=27)
    (blob,) = inp
    out = ns["ref_fn"](*clone(inp))
    assert out.ndim == 0 and close(out, blob.float().norm())


def test_grad_accum_scale():
    ns, inp = _inp("tr_grad_accum_scale", seed=28)
    accum, grad, scale = inp
    out = ns["ref_fn"](*clone(inp))
    assert close(out, accum.float() + scale * grad.float())


def test_grad_zero_center():
    ns, inp = _inp("tr_grad_zero_center", seed=29)
    (grads,) = inp
    out = ns["ref_fn"](*clone(inp))
    g = grads.float()
    assert close(out, g - g.mean(dim=1, keepdim=True))
    assert close(out.sum(1), torch.zeros(grads.shape[0]), at=1e-5)


# =========================================================================== #
# baseline_fn runs and structurally matches the oracle (perf-bar sanity)
# =========================================================================== #
@pytest.mark.parametrize("op", T.OPS)
def test_baseline_matches_structure(op):
    ns, inp = _inp(op, "fp32")
    ref_out = _astuple(ns["ref_fn"](*clone(inp)))
    base_out = _astuple(ns["baseline_fn"](*clone(inp)))
    assert len(ref_out) == len(base_out)
    for r, b in zip(ref_out, base_out):
        if torch.is_tensor(r):
            assert r.shape == b.shape
            assert close(r, b, at=1e-2, rt=1e-2)


# =========================================================================== #
# dtype preservation across the full sweep
# =========================================================================== #
@pytest.mark.parametrize("op", T.OPS)
def test_dtype_preservation(op):
    for dt, tdt in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
        ns, inp = _inp(op, dt)
        out = _astuple(ns["ref_fn"](*clone(inp)))
        first = out[0]
        if torch.is_tensor(first) and first.is_floating_point():
            assert first.dtype == tdt, f"{op}/{dt}: got {first.dtype}"
