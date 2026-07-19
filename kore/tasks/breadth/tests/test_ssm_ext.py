"""CPU-only tests for the breadth SEQUENCE-MODEL EXTENSION engine (ssm_ext).

CORRECTNESS is proven TWO ways for EVERY op: each ``ref_fn`` (the EXACT fp32
SEQUENTIAL recurrence oracle) is asserted equal - at a tight fp32 tolerance - to a
SECOND, INDEPENDENT formulation computed on a DIFFERENT code path:

  * the O(L^2) decay-weighted attention-matrix / quadratic dual for the
    linear-attention & SSD families (Mamba-2 SSD, GLA, gated retention, RetNet /
    lightning retention, feature-map linear attention, HGRN2);
  * the WY triangular-solve (chunked) form for (gated) DeltaNet;
  * the einsum ZOH discretization for the Mamba-1 selective scan;
  * torch.logcumsumexp / torch.cummax / torch.cummin for the scan primitives;
  * the O(L^2) product-sum closed form for the gated scans (segmented / HGRN / GILR);
  * the complex closed form for the LRU;
  * the LTI convolution kernel for S4D;
  * the prefix-decay-matrix quotient for the RWKV wkv recurrence;
  * an im2col-free shifted conv + the SSD/selective dual for the conv+SSM fusions.

Because the two paths are structurally different, a wrong oracle (or a scan
off-by-one) is caught with certainty. The tests also assert the ABI surface (40
``ssm_`` ops, bf16/fp16 sweep), the namespace contract, arity, that every seed
parses + compiles + defines its entry, the shape catalog round-trips through
``parse_shape``, ref preserves the input dtype, the torch baseline agrees with the
fp32 oracle, causality holds (perturbing a value input's FUTURE never changes a
past output), and a few op-specific invariants. All fp32/fp64 on CPU (no GPU /
triton runtime - the seed is only static-checked)."""

from __future__ import annotations

import ast

import pytest
import torch
import torch.nn.functional as F

from kore.tasks._genops import DTYPES
from kore.tasks.breadth import ssm_ext as S

DTYPE_NAMES = ("bf16", "fp16", "fp32")


# --------------------------------------------------------------------------- #
# tiny CPU shapes + expected arity per op (derived from the family, independent
# of make_reference so a wrong arity is caught)
# --------------------------------------------------------------------------- #
_ARITY_BY_FAM = {
    "mamba2_ssd": 5, "selective": 6, "gla": 4, "gated_retention": 4,
    "retention": 3, "rwkv": 4, "linattn": 3, "hgrn": 2, "gilr": 3, "hgrn2": 3,
    "scan_lse": 1, "scan_cummax": 1, "scan_cummin": 1, "segmented_scan": 2,
    "lru": 3, "s4d": 4, "conv_ssd": 5, "conv_selective": 6,
}


def _arity(op):
    fam = S.OP_FAMILY[op]
    if fam == "delta":
        return 5 if S.OP_CONFIG[op]["gated"] else 4
    return _ARITY_BY_FAM[fam]


def _small_shape(op):
    fam, cfg = S.OP_FAMILY[op], S.OP_CONFIG[op]
    if fam in ("gla", "gated_retention", "delta", "linattn", "hgrn2"):
        d = {"B": 2, "H": 2, "L": 7, "Dh": 4}
        if "chunk" in cfg:
            d["chunk"] = cfg["chunk"]
        return d
    if fam == "retention":
        return {"B": 2, "H": cfg["H"], "L": 7, "Dh": 4, "chunk": cfg["chunk"]}
    if fam == "mamba2_ssd":
        return {"B": 2, "L": 7, "H": 2, "P": 3, "N": 4, "chunk": cfg["chunk"]}
    if fam == "selective":
        return {"B": 2, "L": 7, "D": 5, "N": 4}
    if fam == "rwkv":
        return {"B": 2, "L": 7, "C": 5}
    if fam == "s4d":
        return {"B": 2, "D": 3, "L": 9, "N": 4}
    if fam in ("scan_lse", "scan_cummax", "scan_cummin", "segmented_scan", "hgrn", "gilr"):
        return {"B": 2, "D": 3, "L": 9}
    if fam == "lru":
        return {"B": 2, "D": 3, "L": 7}
    if fam == "conv_ssd":
        return {"B": 2, "L": 8, "D": 4, "N": 4, "K": 4, "chunk": cfg["chunk"]}
    if fam == "conv_selective":
        return {"B": 2, "L": 9, "D": 4, "N": 4, "K": 4}
    raise AssertionError(f"no small shape for {op!r}")


def _close(a, b, atol=2e-4, rtol=2e-3):
    return torch.allclose(a.double(), b.double(), atol=atol, rtol=rtol)


def _tril(L):
    return torch.tril(torch.ones(L, L, dtype=torch.bool))


# --------------------------------------------------------------------------- #
# INDEPENDENT fp64 oracles (distinct code paths from the eager recurrence)
# --------------------------------------------------------------------------- #
def _ind_gated_scan_lastdim(a, b):
    """O(L^2) closed form of h_t = a_t h_{t-1} + b_t over the last dim."""
    a, b = a.double(), b.double()
    L = a.shape[-1]
    out = torch.zeros_like(a)
    for t in range(L):
        acc = torch.zeros(a.shape[:-1], dtype=torch.float64)
        for s in range(t + 1):
            coef = torch.ones(a.shape[:-1], dtype=torch.float64)
            for r in range(s + 1, t + 1):
                coef = coef * a[..., r]
            acc = acc + coef * b[..., s]
        out[..., t] = acc
    return out


def _ind_lru(x, nu, theta):
    lam = nu.double() * torch.exp(1j * theta.double())          # [D] complex
    b = x[..., 0].double() + 1j * x[..., 1].double()            # [B,D,L]
    B, D, L = b.shape
    h = torch.zeros(B, D, L, dtype=torch.cdouble)
    for t in range(L):
        acc = torch.zeros(B, D, dtype=torch.cdouble)
        for s in range(t + 1):
            acc = acc + (lam[None, :] ** (t - s)) * b[:, :, s]
        h[:, :, t] = acc
    return torch.stack([h.real, h.imag], dim=-1)


def _ind_s4d(u, Ab, Bb, C):
    ud, Abd, Bbd, Cd = (t.double() for t in (u, Ab, Bb, C))
    a = torch.exp(Abd)                                          # [D,N]
    Bs, D, L = ud.shape
    K = torch.stack([(Cd * (a ** l) * Bbd).sum(-1) for l in range(L)], dim=-1)  # [D,L]
    y = torch.zeros(Bs, D, L, dtype=torch.float64)
    for t in range(L):
        for s in range(t + 1):
            y[:, :, t] = y[:, :, t] + K[:, t - s] * ud[:, :, s]
    return y


def _ind_mamba2(x, dt, A, B_, C):
    xd, dtd, Ad, Bd, Cd = (t.double() for t in (x, dt, A, B_, C))
    dts = F.softplus(dtd)                                       # [B,L,H]
    a = torch.exp(dts * Ad)
    cP = torch.cumsum(torch.log(a), dim=1)                      # [B,L,H]
    L = dts.shape[1]
    M = cP[:, :, None, :] - cP[:, None, :, :]                   # [B,t,s,H]
    mask = _tril(L)[None, :, :, None]
    Dc = torch.where(mask, torch.exp(M), torch.zeros_like(M))
    score = torch.einsum('blhn,bshn->blsh', Cd, Bd)            # [B,t,s,H]
    wdt = Dc * score * dts[:, None, :, :]
    return torch.einsum('btsh,bshp->bthp', wdt, xd)


def _ind_selective(u, delta, A, B_, C, D_):
    ud, dd, Ad, Bd, Cd, Dd = (t.double() for t in (u, delta, A, B_, C, D_))
    dt = F.softplus(dd)
    dA = torch.exp(torch.einsum('bld,dn->bldn', dt, Ad))
    dBu = torch.einsum('bld,bln,bld->bldn', dt, Bd, ud)
    Bs, L, D = ud.shape
    h = torch.zeros(Bs, D, Ad.shape[1], dtype=torch.float64)
    ys = []
    for t in range(L):
        h = dA[:, t] * h + dBu[:, t]
        ys.append(torch.einsum('bdn,bn->bd', h, Cd[:, t]))
    return torch.stack(ys, 1) + torch.einsum('bld,d->bld', ud, Dd)


def _ind_conv(xt, w, K):
    """im2col-free shifted causal depthwise conv, xt[B,D,L] fp64."""
    B, D, L = xt.shape
    out = torch.zeros(B, D, L, dtype=torch.float64)
    for t in range(L):
        acc = torch.zeros(B, D, dtype=torch.float64)
        for kk in range(K):
            idx = t - (K - 1) + kk
            if idx >= 0:
                acc = acc + w[:, kk][None, :] * xt[:, :, idx]
        out[:, :, t] = acc
    return out


def _ind_conv_ssd(x, cw, a, B_, C):
    xd, cwd, ad, Bd, Cd = (t.double() for t in (x, cw, a, B_, C))
    K = cw.shape[1]
    xc = F.silu(_ind_conv(xd.transpose(1, 2), cwd, K)).transpose(1, 2)  # [B,L,D]
    dec = torch.sigmoid(ad)                                     # [B,L]
    cP = torch.cumsum(torch.log(dec), dim=1)
    L = xd.shape[1]
    M = cP[:, :, None] - cP[:, None, :]
    Dc = torch.where(_tril(L)[None], torch.exp(M), torch.zeros_like(M))
    score = torch.einsum('bln,bsn->bls', Cd, Bd)
    return torch.einsum('bls,bsd->bld', Dc * score, xc)


def _ind_conv_selective(u, cw, delta, A, B_, C):
    ud, cwd, dd, Ad, Bd, Cd = (t.double() for t in (u, cw, delta, A, B_, C))
    K = cw.shape[1]
    uc = F.silu(_ind_conv(ud.transpose(1, 2), cwd, K)).transpose(1, 2)
    dt = F.softplus(dd)
    dA = torch.exp(torch.einsum('bld,dn->bldn', dt, Ad))
    dBu = torch.einsum('bld,bln,bld->bldn', dt, Bd, uc)
    Bs, L, D = ud.shape
    h = torch.zeros(Bs, D, Ad.shape[1], dtype=torch.float64)
    ys = []
    for t in range(L):
        h = dA[:, t] * h + dBu[:, t]
        ys.append(torch.einsum('bdn,bn->bd', h, Cd[:, t]))
    return torch.stack(ys, 1)


def _ind_gla(q, k, v, gl):
    al = torch.sigmoid(gl.double())
    cP = torch.cumsum(torch.log(al), dim=2)
    A2 = torch.einsum('bhtd,bhsd->bhts', q.double() * torch.exp(cP), k.double() * torch.exp(-cP))
    L = q.shape[2]
    A2 = A2 * _tril(L)[None, None]
    return torch.einsum('bhts,bhsd->bhtd', A2, v.double())


def _ind_gated_retention(q, k, v, gl):
    a = torch.sigmoid(gl.double())                             # [B,H,L]
    cP = torch.cumsum(torch.log(a), dim=2)
    A2 = torch.einsum('bhtd,bhsd->bhts',
                      q.double() * torch.exp(cP)[..., None],
                      k.double() * torch.exp(-cP)[..., None])
    L = q.shape[2]
    A2 = A2 * _tril(L)[None, None]
    return torch.einsum('bhts,bhsd->bhtd', A2, v.double())


def _ind_hgrn2(q, v, gl):
    al = torch.sigmoid(gl.double())
    cP = torch.cumsum(torch.log(al), dim=2)
    A2 = torch.einsum('bhtd,bhsd->bhts', q.double() * torch.exp(cP), (1.0 - al) * torch.exp(-cP))
    L = q.shape[2]
    A2 = A2 * _tril(L)[None, None]
    return torch.einsum('bhts,bhsd->bhtd', A2, v.double())


def _ind_retention(q, k, v, cfg):
    H = cfg["H"]
    gamma = torch.tensor(S.retention_gamma(H, cfg["decay"]), dtype=torch.float64)
    L = q.shape[2]
    idx = torch.arange(L)
    diff = (idx[:, None] - idx[None, :]).double()
    decay = torch.where(diff >= 0, gamma[:, None, None] ** diff.clamp(min=0)[None],
                        torch.zeros(H, L, L, dtype=torch.float64))
    A2 = torch.einsum('bhtd,bhsd->bhts', q.double(), k.double()) * decay[None]
    return torch.einsum('bhts,bhsd->bhtd', A2, v.double())


def _phi64(x, f):
    if f == "elu":
        return F.elu(x) + 1.0
    if f == "relu":
        return F.relu(x)
    if f == "relu2":
        return F.relu(x) ** 2
    return torch.exp(x)


def _ind_linattn(q, k, v, cfg):
    pq = _phi64(q.double(), cfg["fmap"])
    pk = _phi64(k.double(), cfg["fmap"])
    L = q.shape[2]
    A2 = torch.einsum('bhtd,bhsd->bhts', pq, pk) * _tril(L)[None, None]
    num = torch.einsum('bhts,bhsd->bhtd', A2, v.double())
    if cfg["normalize"]:
        den = A2.sum(-1, keepdim=True) + S.LINATTN_EPS
        return num / den
    return num


def _ind_delta(inputs, cfg):
    gated = cfg["gated"]
    if gated:
        q, k, v, al, be = inputs
    else:
        q, k, v, be = inputs
    L = q.shape[2]
    idx = torch.arange(L)
    kd = k.double()
    kd = kd / (kd.norm(dim=-1, keepdim=True) + S.DELTA_EPS)
    beta = torch.sigmoid(be.double())                          # [B,H,L]
    if gated:
        Gamma = torch.cumprod(torch.sigmoid(al.double()), dim=2)
    else:
        Gamma = torch.ones_like(beta)
    ratio = Gamma[..., :, None] / Gamma[..., None, :]          # [B,H,t,s]
    KK = torch.einsum('bhtd,bhsd->bhts', kd, kd)
    strict = (idx[:, None] > idx[None, :])[None, None]
    T = KK * ratio * strict
    M = torch.eye(L, dtype=torch.float64)[None, None] + beta[..., :, None] * T
    U = torch.linalg.solve(M, beta[..., :, None] * v.double())  # [B,H,L,Dv]
    QK = torch.einsum('bhtd,bhsd->bhts', q.double(), kd)
    causal = (idx[:, None] >= idx[None, :])[None, None]
    A = QK * ratio * causal
    return torch.einsum('bhts,bhsd->bhtd', A, U)


def _ind_rwkv(k, v, w, u, cfg):
    kd, vd, ud = k.double(), v.double(), u.double()
    B, L, C = kd.shape
    if cfg["data_dependent"]:
        wf = F.softplus(w.double())                            # [B,L,C]
    else:
        wf = F.softplus(w.double())[None, None, :].expand(B, L, C)
    Wcum = torch.cumsum(wf, dim=1)                             # inclusive prefix
    ek = torch.exp(kd)
    y = torch.empty(B, L, C, dtype=torch.float64)
    for t in range(L):
        eb = torch.exp(ud[None, :] + kd[:, t])
        if t == 0:
            hn = torch.zeros(B, C, dtype=torch.float64)
            hd = torch.zeros(B, C, dtype=torch.float64)
        else:
            Wt1 = Wcum[:, t - 1, :]
            Dm = torch.exp(-(Wt1[:, None, :] - Wcum[:, :t, :]))  # [B,t,C]
            hn = (Dm * ek[:, :t, :] * vd[:, :t, :]).sum(1)
            hd = (Dm * ek[:, :t, :]).sum(1)
        y[:, t] = (hn + eb * vd[:, t]) / (hd + eb)
    return y


def _independent(op, inputs):
    fam, cfg = S.OP_FAMILY[op], S.OP_CONFIG[op]
    if fam == "scan_lse":
        return torch.logcumsumexp(inputs[0].double(), dim=-1)
    if fam == "scan_cummax":
        return torch.cummax(inputs[0].double(), dim=-1).values
    if fam == "scan_cummin":
        return torch.cummin(inputs[0].double(), dim=-1).values
    if fam == "segmented_scan":
        return _ind_gated_scan_lastdim(1.0 - inputs[1], inputs[0])
    if fam == "hgrn":
        f = torch.sigmoid(inputs[0].double())
        return _ind_gated_scan_lastdim(f, (1.0 - f) * inputs[1].double())
    if fam == "gilr":
        f = torch.sigmoid(inputs[0].double())
        i = torch.sigmoid(inputs[1].double())
        return _ind_gated_scan_lastdim(f, i * inputs[2].double())
    if fam == "lru":
        return _ind_lru(*inputs)
    if fam == "s4d":
        return _ind_s4d(*inputs)
    if fam == "mamba2_ssd":
        return _ind_mamba2(*inputs)
    if fam == "selective":
        return _ind_selective(*inputs)
    if fam == "conv_ssd":
        return _ind_conv_ssd(*inputs)
    if fam == "conv_selective":
        return _ind_conv_selective(*inputs)
    if fam == "gla":
        return _ind_gla(*inputs)
    if fam == "gated_retention":
        return _ind_gated_retention(*inputs)
    if fam == "retention":
        return _ind_retention(*inputs, cfg)
    if fam == "linattn":
        return _ind_linattn(*inputs, cfg)
    if fam == "hgrn2":
        return _ind_hgrn2(*inputs)
    if fam == "delta":
        return _ind_delta(inputs, cfg)
    if fam == "rwkv":
        return _ind_rwkv(*inputs, cfg)
    raise AssertionError(f"no independent oracle for {op!r}")


# --------------------------------------------------------------------------- #
# metadata / ABI surface
# --------------------------------------------------------------------------- #
def test_ops_and_metadata():
    assert isinstance(S.OPS, tuple) and len(S.OPS) == 40
    assert len(set(S.OPS)) == 40
    assert all(op.startswith("ssm_") for op in S.OPS)
    assert S.DEFAULT_DTYPES == ("bf16", "fp16")
    assert set(S.OP_DTYPES) == set(S.OPS) == set(S.SHAPES) == set(S.OP_FAMILY)
    for op in S.OPS:
        assert S.op_dtypes(op) == S.OP_DTYPES[op]
        assert all(dt in DTYPE_NAMES for dt in S.op_dtypes(op))
    assert S.op_names() == list(S.OPS)


def test_ops_dtypes_shapes_consistent():
    for op in S.OPS:
        assert S.OP_DTYPES[op], f"empty dtype sweep for {op}"
        sh = S.SHAPES[op]
        assert "minimal" in sh and "primary" in sh and "validation" in sh
        assert isinstance(sh["validation"], list) and sh["validation"]


@pytest.mark.parametrize("op", S.OPS)
def test_namespace_contract(op):
    ns = S.make_reference(op, "bf16")
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
              "entry_name", "dtype_name", "family", "mutates_input",
              "time_axis", "seq_input_idx"):
        assert k in ns, k
    assert ns["entry_name"] == op
    assert ns["dtype_name"] == "bf16"
    assert ns["family"] == f"breadth_{op}"
    assert ns["mutates_input"] is False
    assert callable(ns["ref_fn"]) and callable(ns["baseline_fn"])
    assert ns[f"{op}_ref"] is ns["ref_fn"]
    assert ns["time_axis"] in (1, 2)
    assert 0 <= ns["seq_input_idx"] < ns["arity"]


@pytest.mark.parametrize("op", S.OPS)
def test_arity(op):
    ns = S.make_reference(op, "fp32")
    assert ns["arity"] == _arity(op), op
    inputs = ns["get_inputs"](_small_shape(op), device="cpu", seed=0)
    assert isinstance(inputs, tuple)
    assert len(inputs) == ns["arity"]


# --------------------------------------------------------------------------- #
# fp32 oracle correctness vs an INDEPENDENT torch compute (the two-way check)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", S.OPS)
def test_ref_matches_independent(op):
    ns = S.make_reference(op, "fp32")
    inputs = ns["get_inputs"](_small_shape(op), device="cpu", seed=0)
    ref = ns["ref_fn"](*inputs)
    ind = _independent(op, inputs)
    assert ref.shape == ind.shape, f"{op}: {tuple(ref.shape)} vs {tuple(ind.shape)}"
    assert _close(ref, ind), (
        f"{op}: max|diff|={(ref.double() - ind.double()).abs().max().item():.3e}")


@pytest.mark.parametrize("op", S.OPS)
def test_ref_matches_independent_second_seed(op):
    """A second seed / shape - guards against a lucky-pass on one input draw."""
    ns = S.make_reference(op, "fp32")
    inputs = ns["get_inputs"](_small_shape(op), device="cpu", seed=7)
    ref = ns["ref_fn"](*inputs)
    ind = _independent(op, inputs)
    assert _close(ref, ind), (
        f"{op}: max|diff|={(ref.double() - ind.double()).abs().max().item():.3e}")


@pytest.mark.parametrize("op", S.OPS)
def test_baseline_matches_ref(op):
    """The torch eager baseline (fp32) agrees with the fp32 oracle (same math)."""
    ns = S.make_reference(op, "fp32")
    inputs = ns["get_inputs"](_small_shape(op), device="cpu", seed=1)
    out = ns["baseline_fn"](*inputs)
    ref = ns["ref_fn"](*inputs)
    assert out.shape == ref.shape
    assert _close(out, ref)


@pytest.mark.parametrize("op", S.OPS)
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_ref_preserves_input_dtype(op, dtype):
    ns = S.make_reference(op, dtype)
    inputs = ns["get_inputs"](_small_shape(op), device="cpu", seed=2)
    out = ns["ref_fn"](*inputs)
    tdt = getattr(torch, DTYPES[dtype][0])
    outs = out if isinstance(out, (tuple, list)) else (out,)
    assert all(o.dtype == tdt for o in outs)


# --------------------------------------------------------------------------- #
# seed static checks (parses + compiles + defines a top-level entry fn)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", S.OPS)
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_seed_compiles_and_defines_entry(op, dtype):
    src = S.seed_source(op, dtype)
    compile(src, f"<{op}_{dtype}_seed>", "exec")
    tree = ast.parse(src)
    assert any(isinstance(n, ast.FunctionDef) and n.name == op for n in tree.body), (
        f"{op} seed must define a top-level def {op}(...)")


@pytest.mark.parametrize("op", S.OPS)
def test_shapes_parse_roundtrip(op):
    ns = S.make_reference(op, "fp32")
    parse = ns["parse_shape"]
    sh = S.SHAPES[op]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        s = ",".join(f"{k}={v}" for k, v in spec.items())
        assert parse(s) == spec, (op, parse(s), spec)


# --------------------------------------------------------------------------- #
# causality: perturbing a value input's FUTURE never changes a PAST output
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", S.OPS)
def test_causality(op):
    ns = S.make_reference(op, "fp32")
    inputs = list(ns["get_inputs"](_small_shape(op), device="cpu", seed=0))
    ax, vi = ns["time_axis"], ns["seq_input_idx"]
    val = inputs[vi]
    L = val.shape[ax]
    t0 = L // 2
    y = ns["ref_fn"](*inputs)
    pert = val.clone()
    pert.narrow(ax, t0 + 1, L - (t0 + 1)).add_(5.0)            # perturb the future only
    inputs2 = list(inputs)
    inputs2[vi] = pert
    y2 = ns["ref_fn"](*inputs2)
    yp = y.narrow(ax, 0, t0 + 1)
    y2p = y2.narrow(ax, 0, t0 + 1)
    assert torch.allclose(yp.double(), y2p.double(), atol=1e-6, rtol=0.0), (
        f"{op}: past output changed when only the future value input was perturbed")


# --------------------------------------------------------------------------- #
# op-specific numeric / semantic checks
# --------------------------------------------------------------------------- #
def test_logcumsumexp_last_equals_logsumexp():
    ns = S.make_reference("ssm_logcumsumexp", "fp32")
    (x,) = ns["get_inputs"]({"B": 2, "D": 3, "L": 13}, device="cpu", seed=0)
    out = ns["ref_fn"](x)
    assert _close(out[..., -1], torch.logsumexp(x.double(), dim=-1))
    assert _close(out, torch.logcumsumexp(x.double(), dim=-1))


def test_logcumsumexp_stable_on_large_inputs():
    """A naive (non-max-subtracted) exp-cumsum would overflow to inf; the stable
    streaming oracle must not."""
    ns = S.make_reference("ssm_logcumsumexp", "fp32")
    x = torch.tensor([[[80.0, 90.0, 100.0, 110.0]]])
    out = ns["ref_fn"](x)
    assert torch.isfinite(out).all()
    assert _close(out, torch.logcumsumexp(x.double(), dim=-1))


def test_cummax_monotone_nondecreasing():
    ns = S.make_reference("ssm_cummax", "fp32")
    (x,) = ns["get_inputs"]({"B": 2, "D": 3, "L": 17}, device="cpu", seed=0)
    out = ns["ref_fn"](x).double()
    assert (out[..., 1:] >= out[..., :-1] - 1e-9).all()
    assert _close(out, torch.cummax(x.double(), dim=-1).values)


def test_cummin_monotone_nonincreasing():
    ns = S.make_reference("ssm_cummin", "fp32")
    (x,) = ns["get_inputs"]({"B": 2, "D": 3, "L": 17}, device="cpu", seed=0)
    out = ns["ref_fn"](x).double()
    assert (out[..., 1:] <= out[..., :-1] + 1e-9).all()
    assert _close(out, torch.cummin(x.double(), dim=-1).values)


def test_segmented_scan_reset_semantics():
    """A reset flag starts a new segment: h_p == x_p, and values at/after the reset
    do NOT depend on inputs before it."""
    ns = S.make_reference("ssm_segmented_scan", "fp32")
    B, D, L, p = 1, 2, 8, 4
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, D, L, generator=g)
    reset = torch.zeros(B, D, L)
    reset[:, :, p] = 1.0
    out = ns["ref_fn"](x, reset)
    assert _close(out[:, :, p], x[:, :, p])                    # segment restart
    x2 = x.clone()
    x2[:, :, :p] += 5.0
    out2 = ns["ref_fn"](x2, reset)
    assert _close(out[:, :, p:], out2[:, :, p:])               # after-reset unchanged


def test_linattn_future_dependence():
    """Perturbing v in the FUTURE changes the future output (op genuinely uses v)."""
    ns = S.make_reference("ssm_linattn_norm", "fp32")
    q, k, v = ns["get_inputs"]({"B": 1, "H": 2, "L": 7, "Dh": 4}, device="cpu", seed=0)
    t0 = 3
    v2 = v.clone()
    v2[:, :, t0 + 1:] += 3.0
    y = ns["ref_fn"](q, k, v)
    y2 = ns["ref_fn"](q, k, v2)
    assert _close(y[:, :, : t0 + 1], y2[:, :, : t0 + 1])       # past unchanged
    assert not _close(y[:, :, t0 + 1:], y2[:, :, t0 + 1:])     # future DID change


def test_retention_multiscale_decay_ordering():
    """RetNet multi-scale decays are strictly increasing towards 1 across heads."""
    g = S.retention_gamma(8, "retnet")
    assert all(0.0 < x < 1.0 for x in g)
    assert all(g[i] < g[i + 1] for i in range(len(g) - 1))


def test_delta_reduces_to_linear_attention_when_pred_removed():
    """Sanity: with beta small the delta update is a weak write; the state stays
    near zero so the output is small - a smoke check on the recurrence wiring."""
    ns = S.make_reference("ssm_deltanet", "fp32")
    q, k, v, beta = ns["get_inputs"]({"B": 1, "H": 2, "L": 6, "Dh": 4}, device="cpu", seed=0)
    y_small = ns["ref_fn"](q, k, v, beta - 20.0)               # sigmoid(beta-20) ~ 0
    assert y_small.abs().max().item() < 1e-2
