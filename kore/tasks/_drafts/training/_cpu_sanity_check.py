"""CPU sanity check for the DRAFT training-side (BACKWARD) oracles.

Gradients are where silent bugs live, so each reference is corroborated THREE
independent ways on a tiny CPU case (all in float64):

  1. FORWARD: ``reference.reference_forward`` vs an INDEPENDENT forward implemented
     here (a different code path: torch.einsum / manual softmax / manual LayerNorm),
     so a wrong forward (and therefore a wrong autograd gradient) is caught.
  2. FINITE DIFFERENCE (the mandated check): ``reference.reference_grads`` (the torch
     AUTOGRAD oracle) vs a CENTRAL numerical gradient of the scalar loss
     L = sum(upstream * F_independent(inputs)). Because the finite difference
     differentiates the INDEPENDENT forward, agreement proves the oracle returns the
     true gradient of the intended operation (not merely that autograd is self
     consistent).
  3. ANALYTIC: ``reference.reference_grads`` vs the closed-form backward that the
     Triton SEED implements (softmax JVP, LayerNorm dx/dgamma/dbeta, dgrad/wgrad,
     the FlashAttention-2 dS/dQ/dK/dV). This validates the SEED's math against the
     oracle on CPU (the Triton kernel itself still needs gfx950 -- see
     VERIFICATION_CHECKLIST.md).

This proves REFERENCE + SEED-MATH CORRECTNESS ON CPU ONLY. It does NOT compile the
Triton seeds, run the framework/vendor baselines, or measure anything on gfx950.
Run from the repo root:

    python kore/tasks/_drafts/training/_cpu_sanity_check.py
"""

from __future__ import annotations

import importlib.util
import math
import os

import torch

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
F64 = torch.float64


def _load_ref(task_id):
    path = os.path.join(HERE, task_id, "reference.py")
    spec = importlib.util.spec_from_file_location(f"train_ref_{task_id}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _snr_db(o, r):
    o, r = o.double(), r.double()
    noise = (o - r).norm().item()
    signal = r.norm().item()
    if noise == 0:
        return 999.0
    return 20.0 * math.log10(signal / noise) if signal > 0 else -999.0


def _causal_add(S):
    i = torch.arange(S)[:, None]
    j = torch.arange(S)[None, :]
    m = torch.zeros((S, S), dtype=F64)
    m.masked_fill_(j > i, float("-inf"))
    return m


# --------------------------------------------------------------------------- #
# Independent forwards (F_indep), analytic backwards (seed math), and the
# differentiable-input / upstream selectors, per task.
# --------------------------------------------------------------------------- #
def _sm_fwd(inp):
    x, _dy = inp
    e = (x - x.amax(-1, keepdim=True)).exp()
    return e / e.sum(-1, keepdim=True)


def _sm_analytic(inp, EPS):
    x, dy = inp
    y = _sm_fwd(inp)
    dx = y * (dy - (dy * y).sum(-1, keepdim=True))
    return (dx,)


def _ln_fwd_stats(x, EPS):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + EPS)
    return mu, rstd


def _ln_fwd(inp, EPS):
    x, gamma, beta, _dy = inp
    mu, rstd = _ln_fwd_stats(x, EPS)
    return (x - mu) * rstd * gamma + beta


def _ln_analytic(inp, EPS):
    x, gamma, beta, dy = inp
    mu, rstd = _ln_fwd_stats(x, EPS)
    xhat = (x - mu) * rstd
    g = dy * gamma
    dx = rstd * (g - g.mean(-1, keepdim=True) - xhat * (g * xhat).mean(-1, keepdim=True))
    dgamma = (dy * xhat).sum(0)
    dbeta = dy.sum(0)
    return (dx, dgamma, dbeta)


def _gm_fwd(inp):
    x, w, _dy = inp
    return torch.einsum("mk,nk->mn", x, w)


def _gm_analytic(inp, EPS):
    x, w, dy = inp
    dx = dy @ w
    dw = dy.transpose(0, 1) @ x
    return (dx, dw)


def _fa_pk(q, k, v, scale):
    """Independent attention forward internals -> (p [B,H,S,S], o [B,H,S,D])."""
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    S = q.shape[1]
    scores = torch.einsum("bhid,bhjd->bhij", qh, kh) * scale + _causal_add(S)
    p = torch.softmax(scores, dim=-1)
    o = torch.einsum("bhij,bhjd->bhid", p, vh)
    return p, o


def _fa_fwd(inp):
    q, k, v = inp[0], inp[1], inp[2]
    scale = 1.0 / (q.shape[-1] ** 0.5)
    _p, o = _fa_pk(q, k, v, scale)
    return o.transpose(1, 2)                      # [B,S,H,D]


def _fa_analytic(inp, EPS):
    q, k, v, o, do, lse = inp
    scale = 1.0 / (q.shape[-1] ** 0.5)
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    doh = do.transpose(1, 2)
    p, oo = _fa_pk(q, k, v, scale)
    delta = (oo * doh).sum(-1)                     # [B,H,S]
    dp = torch.einsum("bhid,bhjd->bhij", doh, vh)  # [B,H,S,S]
    ds = p * (dp - delta[..., None])
    dq = scale * torch.einsum("bhij,bhjd->bhid", ds, kh)
    dk = scale * torch.einsum("bhij,bhid->bhjd", ds, qh)
    dv = torch.einsum("bhij,bhid->bhjd", p, doh)
    return (dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2))


# task_id -> (forward, analytic, diff-input indices, upstream index)
SPEC = {
    "softmax_backward_bf16": (_sm_fwd, _sm_analytic, (0,), 1),
    "layernorm_backward_bf16": (_ln_fwd, _ln_analytic, (0, 1, 2), 3),
    "gemm_backward_bf16": (_gm_fwd, _gm_analytic, (0, 1), 2),
    "flash_attn_backward_bf16": (_fa_fwd, _fa_analytic, (0, 1, 2), 4),
}

TINY = {
    "softmax_backward_bf16": [{"M": 3, "N": 5}, {"M": 2, "N": 7}],
    "layernorm_backward_bf16": [{"M": 3, "N": 5}, {"M": 4, "N": 6}],
    "gemm_backward_bf16": [{"M": 3, "N": 4, "K": 5}, {"M": 2, "N": 5, "K": 3}],
    "flash_attn_backward_bf16": [{"B": 1, "H": 1, "S": 4, "D": 4},
                                 {"B": 1, "H": 2, "S": 5, "D": 4}],
}


def _needs_eps(fwd):
    return fwd in (_ln_fwd,)


def _fd_grad(Lfn, t, h=1e-5):
    """Central finite-difference gradient of scalar Lfn() w.r.t. contiguous t."""
    g = torch.zeros_like(t)
    tf = t.view(-1)
    gf = g.view(-1)
    for i in range(tf.numel()):
        orig = tf[i].item()
        tf[i] = orig + h
        lp = Lfn()
        tf[i] = orig - h
        lm = Lfn()
        tf[i] = orig
        gf[i] = (lp - lm) / (2.0 * h)
    return g


def _run_one(task_id, ref, shape):
    fwd, analytic, diff_idx, up_idx = SPEC[task_id]
    EPS = getattr(ref, "EPS", 1e-5)
    inp = ref.get_inputs(shape, device="cpu", seed=0, dtype=F64)

    def call_fwd():
        return fwd(inp, EPS) if _needs_eps(fwd) else fwd(inp)

    # 1. forward: reference_forward vs independent forward
    ref_fwd = ref.reference_forward(shape, inp).double()
    ind_fwd = call_fwd().double()
    snr_fwd = _snr_db(ref_fwd, ind_fwd)

    # 2/3. gradients: reference autograd oracle (computed once, unperturbed inputs)
    ref_grads = [g.double() for g in _as_tuple(ref.reference_grads(shape, inp))]

    up = inp[up_idx].detach().clone().double()

    def Lfn():
        return (up * call_fwd().double()).sum().item()

    fd_grads = [_fd_grad(Lfn, inp[i]) for i in diff_idx]
    an_grads = [g.double() for g in _as_tuple(analytic(inp, EPS))]

    names = getattr(ref, "GRAD_NAMES", tuple(f"g{i}" for i in range(len(ref_grads))))
    rows = []
    ok_all = snr_fwd > 50.0
    for gi in range(len(ref_grads)):
        snr_fd = _snr_db(ref_grads[gi], fd_grads[gi])
        snr_an = _snr_db(ref_grads[gi], an_grads[gi])
        ok = (snr_fd > 40.0) and (snr_an > 45.0)
        ok_all = ok_all and ok
        rows.append((names[gi], snr_fd, snr_an, ok))
    return snr_fwd, rows, ok_all


def _as_tuple(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def main():
    all_ok = True
    print(f"{'task_id':<26} {'shape':<26} {'grad':<8} {'fwdSNR':>8} {'fdSNR':>8} {'anSNR':>8} {'ok':>5}")
    print("-" * 96)
    for task_id in sorted(SPEC):
        ref = _load_ref(task_id)
        for shape in TINY[task_id]:
            snr_fwd, rows, ok = _run_one(task_id, ref, shape)
            all_ok = all_ok and ok
            sh = ",".join(f"{k}={v}" for k, v in shape.items())
            for j, (nm, sfd, san, gok) in enumerate(rows):
                head_task = task_id if j == 0 else ""
                head_sh = sh if j == 0 else ""
                fw = f"{snr_fwd:8.1f}" if j == 0 else " " * 8
                print(f"{head_task:<26} {head_sh:<26} {nm:<8} {fw} {sfd:8.1f} {san:8.1f} "
                      f"{('OK' if gok else 'FAIL'):>5}")
    print("-" * 96)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
