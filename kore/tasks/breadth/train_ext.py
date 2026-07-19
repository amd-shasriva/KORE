"""Extended training-critical task engine: the HARD frontier of loss+backward and
fused/quantized optimizer-step Triton kernels (companion to ``train_ops``).

``train_ops`` covers the entry loss heads + a handful of fused optimizer steps.
This engine authors the *frontier* training kernels that still have real headroom
on MI350X/gfx950 and that a policy must actually fuse + get numerically right:

  * FUSED OPTIMIZER STEPS (mutate param + moment buffers in place, one fused pass):
    the full ``torch.optim`` zoo (SGD/Adam/AdamW/RMSprop/Adagrad/Adadelta/Adamax/
    NAdam/RAdam/Rprop) plus the memory-frontier ones - Adafactor (factored 2nd
    moment), Muon (Newton-Schulz orthogonalization), LAMB/LARS (layer-wise trust
    ratio), AdaBelief/NovoGrad, multi-tensor/foreach blobs, and 8-bit / fp8
    BLOCKWISE-QUANTIZED-state Adam (dequant -> fp32 update -> requant).
  * FUSED LOSS + BACKWARD (return the loss AND the input gradient in one kernel):
    cross-entropy over large vocab and its cousins (logit-softcap CE, z-loss CE,
    focal, label-smoothing, PolyLoss), distillation losses (forward/reverse KL,
    Jensen-Shannon, temperature-scaled KD), and BCE-with-logits / Huber / smooth-L1
    / cosine-embedding.
  * GRADIENT UTILITIES: per-layer (per-row) grad-norm clip, adaptive gradient
    clipping (AGC), EMA weight update, gradient centralization, fused grad
    accumulation, and the multi-tensor global L2 norm.

These ops have NO vendor kernel, so the honest bar is torch: ``ref_fn`` is the
EXACT fp32 oracle (optimizer math == the matching ``torch.optim`` step; loss +
gradient == ``F.*`` / hand-derived analytic == ``torch.autograd``), cast back to
the task dtype; ``baseline_fn`` is the eager torch path a fused Triton kernel must
beat. Optimizer/grad-util steps set ``mutates_input=True`` (they update param +
moment buffers in place), so the bench loop feeds a fresh clone each timed call
(see ``_genops._build_bench_fn`` mutates_input path).

Contract mirrors ``train_ops`` / ``vendor_ops`` (OPS / OP_DTYPES / SHAPES /
make_reference / seed_source / op_dtypes) so the generic ``_genops`` driver +
generator machinery consume it unchanged. torch imported lazily (registry
discovery never needs a GPU/torch); the DIFFERENT op names (all ``tr_`` prefixed)
never collide with ``train_ops``.
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# --------------------------------------------------------------------------- #
# Task math constants (shared by the fp32 oracle AND the generated seeds)
# --------------------------------------------------------------------------- #
STEP = 10                 # optimizer step index used for bias-correction sweeps
SOFTCAP = 30.0            # Gemma-2 style logit softcap: cap * tanh(logits / cap)
ZLOSS_LAMBDA = 1e-4       # PaLM auxiliary z-loss weight: lam * mean(logsumexp**2)
FOCAL_GAMMA = 2.0         # focal-loss focusing parameter
LS_EPS = 0.1              # label-smoothing epsilon
POLY1_EPS = 1.0           # PolyLoss Poly-1 coefficient: CE + eps * (1 - p_target)
DISTILL_T = 2.0           # knowledge-distillation temperature
HUBER_DELTA = 1.0         # Huber transition point
SMOOTHL1_BETA = 0.5       # smooth-L1 transition point (!= HUBER_DELTA so distinct)
AGC_CLIP = 0.01           # adaptive-gradient-clipping ratio (NFNets)
AGC_EPS = 1e-3            # AGC param-norm floor
EMA_DECAY = 0.999         # EMA (Polyak) weight-averaging decay
ACCUM_SCALE = 0.125       # gradient-accumulation scale (1 / micro_batches)
CLIP_EPS = 1e-6           # grad-clip denominator epsilon (torch default)
NS_COEFFS = (3.4445, -4.7750, 2.0315)  # Muon Newton-Schulz quintic coefficients
MUON_EPS = 1e-7           # Muon NS spectral-norm floor
QUANT_BLOCK = 128         # blockwise quantization group size (bitsandbytes-style)


# --------------------------------------------------------------------------- #
# Op families (config variants -> distinct op names; ALL prefixed ``tr_``)
# --------------------------------------------------------------------------- #
# SGD config: (momentum, dampening, weight_decay, nesterov)
_SGD_CFG: dict[str, dict] = {
    "tr_sgd_momentum":  {"momentum": 0.9, "dampening": 0.0, "wd": 0.01, "nesterov": False},
    "tr_sgd_nesterov":  {"momentum": 0.9, "dampening": 0.0, "wd": 0.01, "nesterov": True},
    "tr_sgd_dampening": {"momentum": 0.9, "dampening": 0.1, "wd": 0.01, "nesterov": False},
}
# Adam config: (decoupled weight decay?, amsgrad?, weight_decay)
_ADAM_CFG: dict[str, dict] = {
    "tr_adam":          {"decoupled": False, "amsgrad": False, "wd": 0.0},
    "tr_adam_wd":       {"decoupled": False, "amsgrad": False, "wd": 0.01},
    "tr_adam_amsgrad":  {"decoupled": False, "amsgrad": True,  "wd": 0.0},
    "tr_adamw":         {"decoupled": True,  "amsgrad": False, "wd": 0.01},
    "tr_adamw_amsgrad": {"decoupled": True,  "amsgrad": True,  "wd": 0.01},
}
# RMSprop config: (momentum, centered)
_RMSPROP_CFG: dict[str, dict] = {
    "tr_rmsprop":                   {"momentum": 0.0, "centered": False},
    "tr_rmsprop_momentum":          {"momentum": 0.9, "centered": False},
    "tr_rmsprop_centered":          {"momentum": 0.0, "centered": True},
    "tr_rmsprop_centered_momentum": {"momentum": 0.9, "centered": True},
}
# Muon config: Newton-Schulz iteration count
_MUON_CFG: dict[str, dict] = {
    "tr_muon_ns5": {"ns_steps": 5},
    "tr_muon_ns3": {"ns_steps": 3},
}
# Quantized-state Adam config: (kind, decoupled weight decay?)
_QUANT_CFG: dict[str, dict] = {
    "tr_adam_8bit":  {"kind": "int8", "decoupled": False, "wd": 0.0},
    "tr_adamw_8bit": {"kind": "int8", "decoupled": True,  "wd": 0.01},
    "tr_adam_fp8":   {"kind": "fp8",  "decoupled": False, "wd": 0.0},
}

_SGD_OPS = tuple(_SGD_CFG)
_ADAM_OPS = tuple(_ADAM_CFG)
_RMSPROP_OPS = tuple(_RMSPROP_CFG)
_MUON_OPS = tuple(_MUON_CFG)
_QUANT_OPS = tuple(_QUANT_CFG)
_SINGLE_OPT_OPS = ("tr_adagrad", "tr_adadelta", "tr_adamax", "tr_nadam",
                   "tr_radam", "tr_rprop", "tr_adafactor")
_FOREACH_OPS = ("tr_foreach_adamw", "tr_foreach_sgd")
_ADV_OPS = ("tr_lamb", "tr_lars", "tr_adabelief", "tr_novograd")

_CE_LOSS_OPS = ("tr_cross_entropy_bwd", "tr_softcap_ce_bwd", "tr_zloss_ce_bwd",
                "tr_focal_ce_bwd", "tr_ls_ce_bwd", "tr_poly1_ce_bwd")
_DISTILL_OPS = ("tr_kl_distill_bwd", "tr_reverse_kl_distill_bwd",
                "tr_js_distill_bwd", "tr_temp_distill_bwd")
_ELEM_LOSS_OPS = ("tr_bce_logits_bwd", "tr_huber_bwd", "tr_smooth_l1_bwd")
_COSINE_OPS = ("tr_cosine_embed_bwd",)
_LOSS_OPS = _CE_LOSS_OPS + _DISTILL_OPS + _ELEM_LOSS_OPS + _COSINE_OPS

_GRAD_OPS = ("tr_grad_clip_per_layer", "tr_agc", "tr_ema_update",
             "tr_global_l2_norm", "tr_grad_accum_scale", "tr_grad_zero_center")

_OPTIMIZER_OPS = (_SGD_OPS + _ADAM_OPS + _RMSPROP_OPS + _MUON_OPS + _QUANT_OPS
                  + _SINGLE_OPT_OPS + _FOREACH_OPS + _ADV_OPS)

OPS: tuple[str, ...] = _OPTIMIZER_OPS + _LOSS_OPS + _GRAD_OPS

# In-place ops (kernel updates its param/moment/grad tensors) -> the bench loop
# feeds a fresh clone each timed call (see _genops mutates_input path). Every
# optimizer step mutates; grad utilities that rewrite grads/buffers do too; the
# losses and the pure global-norm reduction do NOT.
_GRAD_MUTATING = ("tr_grad_clip_per_layer", "tr_agc", "tr_ema_update",
                  "tr_grad_accum_scale", "tr_grad_zero_center")
TRAIN_MUTATES_INPUT: frozenset[str] = frozenset(_OPTIMIZER_OPS + _GRAD_MUTATING)

# bf16/fp16 are the compute dtypes; fp32 is swept too because optimizers keep
# fp32 master/moments in real mixed-precision training.
DEFAULT_DTYPES: tuple[str, ...] = ("bf16", "fp16", "fp32")
OP_DTYPES: dict[str, tuple[str, ...]] = {op: DEFAULT_DTYPES for op in OPS}


def op_dtypes(op: str) -> tuple[str, ...]:
    """The dtype sweep for an op (per-op override or the global default)."""
    return OP_DTYPES.get(op, DEFAULT_DTYPES)


# --------------------------------------------------------------------------- #
# Realistic training shapes (M = tokens/rows, N = width, V = vocab, G = tensors)
# --------------------------------------------------------------------------- #
_OPT2D = {  # param[M, N] (+ grad + moments); a 2D hidden weight (Muon needs 2D)
    "minimal": {"M": 64, "N": 128},
    "primary": {"M": 4096, "N": 4096},                       # H x H
    "validation": [{"M": 16384, "N": 4096}, {"M": 4096, "N": 14336},
                   {"M": 8192, "N": 4096}],                  # huge, MLP up-proj, tall
}
_CE = {  # logits[M, V] + targets[M] ; loss + dlogits over the vocab V
    "minimal": {"M": 64, "V": 2048},
    "primary": {"M": 4096, "V": 32000},                      # Llama-2 vocab
    "validation": [{"M": 16384, "V": 32000}, {"M": 4096, "V": 128256},
                   {"M": 8192, "V": 32000}],                 # huge batch, Llama-3 vocab
}
_DISTILL = _CE            # student[M,V] + teacher[M,V]
_ELEM2D = {  # input[M, N] + target[M, N] ; elementwise regression/BCE loss
    "minimal": {"M": 64, "N": 512},
    "primary": {"M": 4096, "N": 4096},
    "validation": [{"M": 16384, "N": 4096}, {"M": 8192, "N": 8192},
                   {"M": 4096, "N": 14336}],
}
_BLOB = {  # G stacked tensors [G, N] ; multi-tensor / foreach / clip / grad utils
    "minimal": {"G": 4, "N": 256},
    "primary": {"G": 16, "N": 4096},
    "validation": [{"G": 32, "N": 4096}, {"G": 8, "N": 14336}, {"G": 16, "N": 8192}],
}

SHAPES: dict[str, dict] = {}
for _op in _OPTIMIZER_OPS:
    SHAPES[_op] = _BLOB if _op in _FOREACH_OPS else _OPT2D
for _op in _CE_LOSS_OPS:
    SHAPES[_op] = _CE
for _op in _DISTILL_OPS:
    SHAPES[_op] = _DISTILL
for _op in _ELEM_LOSS_OPS + _COSINE_OPS:
    SHAPES[_op] = _ELEM2D
for _op in _GRAD_OPS:
    SHAPES[_op] = _BLOB
del _op


# --------------------------------------------------------------------------- #
# reference.py namespace: EXACT fp32 oracle (== torch.optim / F.* / autograd),
# cast back to the task dtype, + the eager torch perf baseline.
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import torch
    import torch.nn.functional as F

    tdt = getattr(torch, DTYPES[dtype][0])

    def gen(seed, device):
        return torch.Generator(device=device).manual_seed(int(seed))

    def randn(shape, seed, device, scale=1.0):
        x = torch.randn(shape, generator=gen(seed, device), device=device, dtype=torch.float32)
        return (x * scale).to(tdt)

    def randn_pos(shape, seed, device, scale=1.0, floor=0.0):
        x = torch.randn(shape, generator=gen(seed, device), device=device, dtype=torch.float32)
        return (x * x * scale + floor).to(tdt)

    def targets(M, V, seed, device):
        return torch.randint(0, V, (M,), generator=gen(seed, device), device=device, dtype=torch.int64)

    def _mk(run):
        """Wrap a precision-parameterized ``run(args, hi)`` into (ref_fn, baseline_fn).

        ``ref_fn`` runs the EXACT fp32 math then casts every tensor output back to
        the task dtype; ``baseline_fn`` is the eager native-dtype path (perf bar)."""
        def ref_fn(*a):
            r = run(a, True)
            r = tuple(x.to(tdt) if torch.is_tensor(x) else x for x in r)
            return r[0] if len(r) == 1 else r

        def baseline_fn(*a):
            r = run(a, False)
            return r[0] if len(r) == 1 else r
        return ref_fn, baseline_fn

    family = f"breadth_{op}"

    # ===================================================================== #
    # OPTIMIZERS (mutate param + moment buffers in place; ref returns them)
    # ===================================================================== #
    if op in _SGD_OPS:
        c = _SGD_CFG[op]

        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (randn((M, N), seed, device, 0.02), randn((M, N), seed + 1, device, 0.01),
                    randn((M, N), seed + 2, device, 0.01),
                    1e-2, c["momentum"], c["dampening"], c["wd"], c["nesterov"])

        def run(a, hi):
            param, grad, buf, lr, momentum, dampening, wd, nesterov = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, b = cv(param), cv(grad), cv(buf)
            if wd != 0:
                g = g + wd * p
            b = momentum * b + (1.0 - dampening) * g
            d = g + momentum * b if nesterov else b
            return (p - lr * d, b)
        ref_fn, baseline_fn = _mk(run)
        arity = 8

    elif op in _ADAM_OPS:
        c = _ADAM_CFG[op]
        dec, ams, WD = c["decoupled"], c["amsgrad"], c["wd"]

        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            base = (randn((M, N), seed, device, 0.02), randn((M, N), seed + 1, device, 0.01),
                    randn((M, N), seed + 2, device, 0.01), randn_pos((M, N), seed + 3, device, 0.001))
            if ams:
                return base + (randn_pos((M, N), seed + 4, device, 0.001),
                               1e-3, 0.9, 0.999, 1e-8, WD, STEP)
            return base + (1e-3, 0.9, 0.999, 1e-8, WD, STEP)

        def run(a, hi):
            if ams:
                param, grad, m, v, vmax, lr, b1, b2, eps, wd, step = a
            else:
                param, grad, m, v, lr, b1, b2, eps, wd, step = a
                vmax = None
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, m, v = cv(param), cv(grad), cv(m), cv(v)
            if wd != 0:
                if dec:
                    p = p * (1.0 - lr * wd)
                else:
                    g = g + wd * p
            m = m + (1.0 - b1) * (g - m)
            v = b2 * v + (1.0 - b2) * g * g
            bc1, bc2 = 1.0 - b1 ** step, 1.0 - b2 ** step
            if ams:
                vmx = torch.maximum(cv(vmax), v)
                denom = vmx.sqrt() / (bc2 ** 0.5) + eps
                return (p - (lr / bc1) * m / denom, m, v, vmx)
            denom = v.sqrt() / (bc2 ** 0.5) + eps
            return (p - (lr / bc1) * m / denom, m, v)
        ref_fn, baseline_fn = _mk(run)
        arity = 11 if ams else 10

    elif op in _RMSPROP_OPS:
        c = _RMSPROP_CFG[op]
        CEN, MOM = c["centered"], c["momentum"]

        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            ts = [randn((M, N), seed, device, 0.02), randn((M, N), seed + 1, device, 0.01),
                  randn_pos((M, N), seed + 2, device, 0.1, floor=0.25)]
            s = seed + 3
            if CEN:
                ts.append(randn((M, N), s, device, 0.05)); s += 1
            if MOM > 0:
                ts.append(randn((M, N), s, device, 0.05)); s += 1
            return tuple(ts) + (1e-2, 0.99, 1e-8, 0.01, MOM)

        def run(a, hi):
            param, grad, sq = a[0], a[1], a[2]
            i = 3
            ga = a[i] if CEN else None
            i += 1 if CEN else 0
            bufm = a[i] if MOM > 0 else None
            i += 1 if MOM > 0 else 0
            lr, alpha, eps, wd, momentum = a[i:i + 5]
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, sq = cv(param), cv(grad), cv(sq)
            if wd != 0:
                g = g + wd * p
            sq = alpha * sq + (1.0 - alpha) * g * g
            if CEN:
                ga = cv(ga); ga = ga + (1.0 - alpha) * (g - ga)
                avg = (sq - ga * ga).sqrt() + eps
            else:
                avg = sq.sqrt() + eps
            if momentum > 0:
                bufm = cv(bufm); bufm = momentum * bufm + g / avg
                p = p - lr * bufm
            else:
                p = p - lr * g / avg
            res = [p, sq]
            if CEN:
                res.append(ga)
            if momentum > 0:
                res.append(bufm)
            return tuple(res)
        ref_fn, baseline_fn = _mk(run)
        arity = 3 + (1 if CEN else 0) + (1 if MOM > 0 else 0) + 5

    elif op == "tr_adagrad":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (randn((M, N), seed, device, 0.02), randn((M, N), seed + 1, device, 0.01),
                    randn_pos((M, N), seed + 2, device, 0.1, floor=0.01),
                    1e-2, 1e-10, 0.01, 0.01, STEP)

        def run(a, hi):
            param, grad, ssum, lr, eps, wd, lrd, step = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, s = cv(param), cv(grad), cv(ssum)
            if wd != 0:
                g = g + wd * p
            clr = lr / (1.0 + (step - 1) * lrd)
            s = s + g * g
            return (p - clr * g / (s.sqrt() + eps), s)
        ref_fn, baseline_fn = _mk(run)
        arity = 8

    elif op == "tr_adadelta":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (randn((M, N), seed, device, 0.02), randn((M, N), seed + 1, device, 0.01),
                    randn_pos((M, N), seed + 2, device, 0.1, floor=0.01),
                    randn_pos((M, N), seed + 3, device, 0.1, floor=0.01),
                    1.0, 0.9, 1e-6, 0.01)

        def run(a, hi):
            param, grad, sq, acc, lr, rho, eps, wd = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, sq, acc = cv(param), cv(grad), cv(sq), cv(acc)
            if wd != 0:
                g = g + wd * p
            sq = rho * sq + (1.0 - rho) * g * g
            delta = (acc + eps).sqrt() / (sq + eps).sqrt() * g
            acc = rho * acc + (1.0 - rho) * delta * delta
            return (p - lr * delta, sq, acc)
        ref_fn, baseline_fn = _mk(run)
        arity = 8

    elif op == "tr_adamax":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (randn((M, N), seed, device, 0.02), randn((M, N), seed + 1, device, 0.01),
                    randn((M, N), seed + 2, device, 0.01),
                    randn_pos((M, N), seed + 3, device, 0.01, floor=0.01),
                    2e-3, 0.9, 0.999, 1e-8, 0.01, STEP)

        def run(a, hi):
            param, grad, m, inf, lr, b1, b2, eps, wd, step = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, m, inf = cv(param), cv(grad), cv(m), cv(inf)
            if wd != 0:
                g = g + wd * p
            m = m + (1.0 - b1) * (g - m)
            inf = torch.maximum(b2 * inf, g.abs() + eps)
            clr = lr / (1.0 - b1 ** step)
            return (p - clr * m / inf, m, inf)
        ref_fn, baseline_fn = _mk(run)
        arity = 10

    elif op == "tr_nadam":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (randn((M, N), seed, device, 0.02), randn((M, N), seed + 1, device, 0.01),
                    randn((M, N), seed + 2, device, 0.01), randn_pos((M, N), seed + 3, device, 0.001),
                    2e-3, 0.9, 0.999, 1e-8, 0.0, 4e-3, STEP)

        def run(a, hi):
            param, grad, m, v, lr, b1, b2, eps, wd, psi, step = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, m, v = cv(param), cv(grad), cv(m), cv(v)
            if wd != 0:
                g = g + wd * p
            bc2 = 1.0 - b2 ** step
            m = m + (1.0 - b1) * (g - m)
            v = b2 * v + (1.0 - b2) * g * g
            den = (v / bc2).sqrt() + eps
            def mu(t):
                return b1 * (1.0 - 0.5 * 0.96 ** (t * psi))
            mu_t, mu_next = mu(step), mu(step + 1)
            mu_prod = 1.0
            for i in range(1, step + 1):
                mu_prod *= mu(i)
            mu_prod_next = mu_prod * mu_next
            p = (p + g * (-lr * (1.0 - mu_t) / (1.0 - mu_prod)) / den
                 + m * (-lr * mu_next / (1.0 - mu_prod_next)) / den)
            return (p, m, v)
        ref_fn, baseline_fn = _mk(run)
        arity = 11

    elif op == "tr_radam":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (randn((M, N), seed, device, 0.02), randn((M, N), seed + 1, device, 0.01),
                    randn((M, N), seed + 2, device, 0.01), randn_pos((M, N), seed + 3, device, 0.001),
                    1e-3, 0.9, 0.999, 1e-8, 0.0, STEP)

        def run(a, hi):
            param, grad, m, v, lr, b1, b2, eps, wd, step = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, m, v = cv(param), cv(grad), cv(m), cv(v)
            if wd != 0:
                g = g + wd * p
            m = m + (1.0 - b1) * (g - m)
            v = b2 * v + (1.0 - b2) * g * g
            bc1, bc2 = 1.0 - b1 ** step, 1.0 - b2 ** step
            mhat = m / bc1
            rho_inf = 2.0 / (1.0 - b2) - 1.0
            rho_t = rho_inf - 2.0 * step * (b2 ** step) / bc2
            if rho_t > 5.0:
                rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                        / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)) ** 0.5
                adalr = (bc2 ** 0.5) / (v.sqrt() + eps)
                p = p - lr * mhat * adalr * rect
            else:
                p = p - lr * mhat
            return (p, m, v)
        ref_fn, baseline_fn = _mk(run)
        arity = 10

    elif op == "tr_rprop":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (randn((M, N), seed, device, 1.0), randn((M, N), seed + 1, device, 1.0),
                    randn((M, N), seed + 2, device, 1.0),
                    randn_pos((M, N), seed + 3, device, 0.01, floor=0.01),
                    0.5, 1.2, 1e-6, 50.0, STEP)

        def run(a, hi):
            param, grad, prev, ss, etam, etap, smin, smax, step = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, prev, ss = cv(param), cv(grad), cv(prev), cv(ss)
            sign = torch.sign(g * prev)
            mult = torch.where(sign > 0, torch.full_like(sign, etap),
                               torch.where(sign < 0, torch.full_like(sign, etam),
                                           torch.ones_like(sign)))
            ss = (ss * mult).clamp(smin, smax)
            g2 = torch.where(sign < 0, torch.zeros_like(g), g)
            return (p - torch.sign(g2) * ss, g2, ss)
        ref_fn, baseline_fn = _mk(run)
        arity = 9

    elif op == "tr_adafactor":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (randn((M, N), seed, device, 0.1), randn((M, N), seed + 1, device, 0.1),
                    randn_pos((M, 1), seed + 2, device, 0.1, floor=0.01),
                    randn_pos((1, N), seed + 3, device, 0.1, floor=0.01),
                    1e-2, -0.8, 1e-30, 1e-3, 1.0, 0.0, STEP)

        def run(a, hi):
            param, grad, row, col, lr, b2d, eps1, eps2, d, wd, step = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, row, col = cv(param), cv(grad), cv(row), cv(col)
            sf = float(step)
            omb2 = sf ** b2d
            rho_t = min(lr, 1.0 / (sf ** 0.5))
            alpha = max(eps2, p.norm().item() / (p.numel() ** 0.5)) * rho_t
            if wd != 0:
                p = p * (1.0 - lr * wd)
            row = row + omb2 * ((g * g).mean(dim=-1, keepdim=True) - row)
            col = col + omb2 * ((g * g).mean(dim=-2, keepdim=True) - col)
            var = (row @ col) / row.mean(dim=-2, keepdim=True).clamp(min=eps1)
            upd = var.clamp(min=eps1 * eps1).rsqrt() * g
            denom = max(1.0, upd.norm().item() / ((upd.numel() ** 0.5) * d))
            return (p - (alpha / denom) * upd, row, col)
        ref_fn, baseline_fn = _mk(run)
        arity = 11

    elif op in _MUON_OPS:
        ns_steps = _MUON_CFG[op]["ns_steps"]

        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (randn((M, N), seed, device, 0.02), randn((M, N), seed + 1, device, 1.0),
                    randn((M, N), seed + 2, device, 1.0),
                    2e-2, 0.1, 0.95, MUON_EPS,
                    NS_COEFFS[0], NS_COEFFS[1], NS_COEFFS[2], ns_steps)

        def run(a, hi):
            param, grad, buf, lr, wd, momentum, eps, na, nb, nc, steps = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, buf = cv(param), cv(grad), cv(buf)
            buf = buf + (1.0 - momentum) * (g - buf)            # buf.lerp_(g, 1-mu)
            upd = g + momentum * (buf - g)                      # nesterov g.lerp(buf, mu)
            x = upd
            transposed = x.shape[-2] > x.shape[-1]
            if transposed:
                x = x.mT
            x = x / x.norm().clamp(min=eps)
            for _ in range(steps):
                A = x @ x.mT
                B = nb * A + nc * (A @ A)
                x = na * x + B @ x
            if transposed:
                x = x.mT
            adj = lr * max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5
            return (p * (1.0 - lr * wd) - adj * x, buf)
        ref_fn, baseline_fn = _mk(run)
        arity = 11

    elif op in _QUANT_OPS:
        c = _QUANT_CFG[op]
        kind, dec, WD = c["kind"], c["decoupled"], c["wd"]
        QB = QUANT_BLOCK
        qmax = 127.0 if kind == "int8" else 448.0
        qdt = torch.int8 if kind == "int8" else torch.float8_e4m3fn

        def _blockq(x):
            xf = x.reshape(-1).float()
            n = xf.numel()
            nb = (n + QB - 1) // QB
            pad = nb * QB - n
            xp = torch.cat([xf, xf.new_zeros(pad)]) if pad else xf
            xb = xp.reshape(nb, QB)
            amax = xb.abs().amax(1)
            scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
            q = xb / scale[:, None]
            if kind == "int8":
                q = q.round().clamp(-127, 127)
            q = q.to(qdt).reshape(-1)[:n].reshape(x.shape)
            return q, scale

        def _dequant(q, scale):
            qf = q.reshape(-1)
            n = qf.numel()
            s = scale.repeat_interleave(QB)[:n]
            return (qf.float() * s).reshape(q.shape)

        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            param = randn((M, N), seed, device, 0.02)
            grad = randn((M, N), seed + 1, device, 0.01)
            m_raw = (torch.randn((M, N), generator=gen(seed + 2, device), device=device,
                                 dtype=torch.float32) * 0.01)
            v_raw = (torch.randn((M, N), generator=gen(seed + 3, device), device=device,
                                 dtype=torch.float32) ** 2 * 0.001)
            qm, sm = _blockq(m_raw)
            qv, sv = _blockq(v_raw)
            return (param, grad, qm, sm, qv, sv, 1e-3, 0.9, 0.999, 1e-8, WD, STEP)

        def run(a, hi):
            param, grad, qm, sm, qv, sv, lr, b1, b2, eps, wd, step = a
            p = param.float()
            g = grad.float()
            m = _dequant(qm, sm)
            v = _dequant(qv, sv)
            if wd != 0:
                if dec:
                    p = p * (1.0 - lr * wd)
                else:
                    g = g + wd * p
            m = m + (1.0 - b1) * (g - m)
            v = b2 * v + (1.0 - b2) * g * g
            bc1, bc2 = 1.0 - b1 ** step, 1.0 - b2 ** step
            p = p - (lr / bc1) * m / (v.sqrt() / (bc2 ** 0.5) + eps)
            qm2, sm2 = _blockq(m)
            qv2, sv2 = _blockq(v)
            return (p.to(tdt), qm2, sm2, qv2, sv2)

        def ref_fn(*a):
            return run(a, True)
        baseline_fn = ref_fn        # honest eager bar is the same fp32 dequant->update->requant
        arity = 12

    elif op == "tr_lamb":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (randn((M, N), seed, device, 0.02), randn((M, N), seed + 1, device, 0.01),
                    randn((M, N), seed + 2, device, 0.01), randn_pos((M, N), seed + 3, device, 0.001),
                    1e-3, 0.9, 0.999, 1e-6, 0.01, STEP)

        def run(a, hi):
            param, grad, m, v, lr, b1, b2, eps, wd, step = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, m, v = cv(param), cv(grad), cv(m), cv(v)
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * g * g
            mhat = m / (1.0 - b1 ** step)
            vhat = v / (1.0 - b2 ** step)
            upd = mhat / (vhat.sqrt() + eps) + wd * p
            wn = p.norm()
            un = upd.norm()
            trust = wn / un.clamp(min=1e-30)
            return (p - lr * trust * upd, m, v)
        ref_fn, baseline_fn = _mk(run)
        arity = 10

    elif op == "tr_lars":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (randn((M, N), seed, device, 0.1), randn((M, N), seed + 1, device, 0.01),
                    randn((M, N), seed + 2, device, 0.01),
                    1e-2, 0.9, 1e-4, 1e-3, 1e-8)

        def run(a, hi):
            param, grad, buf, lr, momentum, wd, trust_coef, eps = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, buf = cv(param), cv(grad), cv(buf)
            d = g + wd * p
            local_lr = trust_coef * p.norm() / (d.norm() + eps)
            buf = momentum * buf + lr * local_lr * d
            return (p - buf, buf)
        ref_fn, baseline_fn = _mk(run)
        arity = 8

    elif op == "tr_adabelief":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (randn((M, N), seed, device, 0.02), randn((M, N), seed + 1, device, 0.01),
                    randn((M, N), seed + 2, device, 0.01), randn_pos((M, N), seed + 3, device, 0.001),
                    1e-3, 0.9, 0.999, 1e-8, 0.01, STEP)

        def run(a, hi):
            param, grad, m, s, lr, b1, b2, eps, wd, step = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, m, s = cv(param), cv(grad), cv(m), cv(s)
            m = b1 * m + (1.0 - b1) * g
            s = b2 * s + (1.0 - b2) * (g - m) * (g - m) + eps
            mhat = m / (1.0 - b1 ** step)
            shat = s / (1.0 - b2 ** step)
            p = p * (1.0 - lr * wd) - lr * mhat / (shat.sqrt() + eps)
            return (p, m, s)
        ref_fn, baseline_fn = _mk(run)
        arity = 10

    elif op == "tr_novograd":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            v0 = randn_pos((), seed + 3, device, 1.0, floor=0.1)
            return (randn((M, N), seed, device, 0.02), randn((M, N), seed + 1, device, 0.01),
                    randn((M, N), seed + 2, device, 0.01), v0,
                    1e-3, 0.9, 0.25, 1e-8, 0.01)

        def run(a, hi):
            param, grad, m, vsc, lr, b1, b2, eps, wd = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, m, vsc = cv(param), cv(grad), cv(m), cv(vsc)
            v = b2 * vsc + (1.0 - b2) * (g * g).sum()
            ghat = g / (v.sqrt() + eps) + wd * p
            m = b1 * m + ghat
            return (p - lr * m, m, v)
        ref_fn, baseline_fn = _mk(run)
        arity = 9

    elif op == "tr_foreach_adamw":
        def get_inputs(shape, device="cuda", seed=0):
            G, N = shape["G"], shape["N"]
            return (randn((G, N), seed, device, 0.02), randn((G, N), seed + 1, device, 0.01),
                    randn((G, N), seed + 2, device, 0.01), randn_pos((G, N), seed + 3, device, 0.001),
                    1e-3, 0.9, 0.999, 1e-8, 0.01, STEP)

        def run(a, hi):
            param, grad, m, v, lr, b1, b2, eps, wd, step = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, m, v = cv(param), cv(grad), cv(m), cv(v)
            p = p * (1.0 - lr * wd)
            m = m + (1.0 - b1) * (g - m)
            v = b2 * v + (1.0 - b2) * g * g
            bc1, bc2 = 1.0 - b1 ** step, 1.0 - b2 ** step
            return (p - (lr / bc1) * m / (v.sqrt() / (bc2 ** 0.5) + eps), m, v)
        ref_fn, baseline_fn = _mk(run)
        arity = 10

    elif op == "tr_foreach_sgd":
        def get_inputs(shape, device="cuda", seed=0):
            G, N = shape["G"], shape["N"]
            return (randn((G, N), seed, device, 0.02), randn((G, N), seed + 1, device, 0.01),
                    randn((G, N), seed + 2, device, 0.01), 1e-2, 0.9, 0.01)

        def run(a, hi):
            param, grad, buf, lr, momentum, wd = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g, buf = cv(param), cv(grad), cv(buf)
            if wd != 0:
                g = g + wd * p
            buf = momentum * buf + g
            return (p - lr * buf, buf)
        ref_fn, baseline_fn = _mk(run)
        arity = 6

    # ===================================================================== #
    # FUSED LOSS + BACKWARD (return (loss, dinput); NOT in-place)
    # ===================================================================== #
    elif op in _LOSS_OPS:
        def _ce(z, y):
            lse = torch.logsumexp(z, dim=-1)
            sm = torch.softmax(z, dim=-1)
            oh = F.one_hot(y, z.shape[-1]).to(z.dtype)
            return lse, sm, oh

        def a_cross_entropy(logits, tgt):
            z = logits.float(); y = tgt.long(); M = z.shape[0]
            lse, sm, oh = _ce(z, y)
            loss = (lse - z.gather(1, y[:, None]).squeeze(1)).mean()
            return loss, (sm - oh) / M

        def f_cross_entropy(x, r):
            return F.cross_entropy(x, r[0].long())

        def a_softcap(logits, tgt):
            z = logits.float(); y = tgt.long(); M = z.shape[0]
            capd = SOFTCAP * torch.tanh(z / SOFTCAP)
            _, sm, oh = _ce(capd, y)
            loss = F.cross_entropy(capd, y)
            grad = ((sm - oh) / M) * (1.0 - torch.tanh(z / SOFTCAP) ** 2)
            return loss, grad

        def f_softcap(x, r):
            return F.cross_entropy(SOFTCAP * torch.tanh(x / SOFTCAP), r[0].long())

        def a_zloss(logits, tgt):
            z = logits.float(); y = tgt.long(); M = z.shape[0]
            lse, sm, oh = _ce(z, y)
            loss = F.cross_entropy(z, y) + ZLOSS_LAMBDA * (lse ** 2).mean()
            grad = (sm - oh) / M + 2.0 * ZLOSS_LAMBDA * lse[:, None] * sm / M
            return loss, grad

        def f_zloss(x, r):
            return F.cross_entropy(x, r[0].long()) + ZLOSS_LAMBDA * (torch.logsumexp(x, -1) ** 2).mean()

        def a_focal(logits, tgt):
            z = logits.float(); y = tgt.long(); M = z.shape[0]
            _, sm, oh = _ce(z, y)
            pt = sm.gather(1, y[:, None]).squeeze(1)
            g = FOCAL_GAMMA
            loss = (((1.0 - pt) ** g) * (-torch.log(pt))).mean()
            dLdpt = g * (1.0 - pt) ** (g - 1.0) * torch.log(pt) - (1.0 - pt) ** g / pt
            grad = (dLdpt[:, None] * (pt[:, None] * (oh - sm))) / M
            return loss, grad

        def f_focal(x, r):
            p = torch.softmax(x, -1)
            pt = p.gather(1, r[0].long()[:, None]).squeeze(1)
            return (((1.0 - pt) ** FOCAL_GAMMA) * (-torch.log(pt))).mean()

        def a_ls(logits, tgt):
            z = logits.float(); y = tgt.long(); M, V = z.shape
            _, sm, oh = _ce(z, y)
            loss = F.cross_entropy(z, y, label_smoothing=LS_EPS)
            grad = (sm - (1.0 - LS_EPS) * oh - LS_EPS / V) / M
            return loss, grad

        def f_ls(x, r):
            return F.cross_entropy(x, r[0].long(), label_smoothing=LS_EPS)

        def a_poly1(logits, tgt):
            z = logits.float(); y = tgt.long(); M = z.shape[0]
            _, sm, oh = _ce(z, y)
            pt = sm.gather(1, y[:, None]).squeeze(1)
            loss = (F.cross_entropy(z, y, reduction="none") + POLY1_EPS * (1.0 - pt)).mean()
            grad = ((1.0 + POLY1_EPS * pt)[:, None] * (sm - oh)) / M
            return loss, grad

        def f_poly1(x, r):
            pt = torch.softmax(x, -1).gather(1, r[0].long()[:, None]).squeeze(1)
            return (F.cross_entropy(x, r[0].long(), reduction="none") + POLY1_EPS * (1.0 - pt)).mean()

        def a_kl(student, teacher):
            s = student.float(); te = teacher.float(); M = s.shape[0]
            ps = torch.softmax(s, -1); pt = torch.softmax(te, -1)
            loss = (pt * (torch.log(pt) - F.log_softmax(s, -1))).sum(1).mean()
            return loss, (ps - pt) / M

        def f_kl(x, r):
            pt = torch.softmax(r[0].float(), -1)
            return (pt * (torch.log(pt) - F.log_softmax(x, -1))).sum(1).mean()

        def a_rkl(student, teacher):
            s = student.float(); te = teacher.float(); M = s.shape[0]
            ps = torch.softmax(s, -1); pt = torch.softmax(te, -1)
            av = torch.log(ps) - torch.log(pt)
            row = (ps * av).sum(1, keepdim=True)
            return (ps * av).sum(1).mean(), (ps * (av - row)) / M

        def f_rkl(x, r):
            ps = torch.softmax(x, -1); pt = torch.softmax(r[0].float(), -1)
            return (ps * (torch.log(ps) - torch.log(pt))).sum(1).mean()

        def a_js(student, teacher):
            s = student.float(); te = teacher.float(); M = s.shape[0]
            ps = torch.softmax(s, -1); pt = torch.softmax(te, -1)
            mid = 0.5 * (ps + pt)
            loss = (0.5 * (ps * (torch.log(ps) - torch.log(mid))).sum(1)
                    + 0.5 * (pt * (torch.log(pt) - torch.log(mid))).sum(1)).mean()
            cv = torch.log(ps) - torch.log(mid)
            klm = (ps * cv).sum(1, keepdim=True)
            return loss, 0.5 * ps * (cv - klm) / M

        def f_js(x, r):
            ps = torch.softmax(x, -1); pt = torch.softmax(r[0].float(), -1); mid = 0.5 * (ps + pt)
            return (0.5 * (ps * (torch.log(ps) - torch.log(mid))).sum(1)
                    + 0.5 * (pt * (torch.log(pt) - torch.log(mid))).sum(1)).mean()

        def a_temp(student, teacher):
            s = student.float(); te = teacher.float(); M = s.shape[0]; T = DISTILL_T
            psT = torch.softmax(s / T, -1); ptT = torch.softmax(te / T, -1)
            loss = (T * T) * (ptT * (torch.log(ptT) - F.log_softmax(s / T, -1))).sum(1).mean()
            return loss, T * (psT - ptT) / M

        def f_temp(x, r):
            T = DISTILL_T
            ptT = torch.softmax(r[0].float() / T, -1)
            return (T * T) * (ptT * (torch.log(ptT) - F.log_softmax(x / T, -1))).sum(1).mean()

        def a_bce(inp, tgt):
            z = inp.float(); y = tgt.float()
            loss = F.binary_cross_entropy_with_logits(z, y)
            return loss, (torch.sigmoid(z) - y) / z.numel()

        def f_bce(x, r):
            return F.binary_cross_entropy_with_logits(x, r[0].float())

        def a_huber(inp, tgt):
            aa = inp.float(); bb = tgt.float()
            loss = F.huber_loss(aa, bb, delta=HUBER_DELTA)
            d = aa - bb
            grad = torch.where(d.abs() <= HUBER_DELTA, d, HUBER_DELTA * torch.sign(d)) / aa.numel()
            return loss, grad

        def f_huber(x, r):
            return F.huber_loss(x, r[0].float(), delta=HUBER_DELTA)

        def a_smooth_l1(inp, tgt):
            aa = inp.float(); bb = tgt.float()
            loss = F.smooth_l1_loss(aa, bb, beta=SMOOTHL1_BETA)
            d = aa - bb
            grad = torch.where(d.abs() < SMOOTHL1_BETA, d / SMOOTHL1_BETA, torch.sign(d)) / aa.numel()
            return loss, grad

        def f_smooth_l1(x, r):
            return F.smooth_l1_loss(x, r[0].float(), beta=SMOOTHL1_BETA)

        def a_cosine(x1, x2):
            a1 = x1.float(); a2 = x2.float(); M = a1.shape[0]
            n1 = a1.norm(dim=1, keepdim=True); n2 = a2.norm(dim=1, keepdim=True)
            cosv = (a1 * a2).sum(1, keepdim=True) / (n1 * n2)
            loss = (1.0 - cosv).mean()
            grad = -(a2 / (n1 * n2) - cosv * a1 / (n1 * n1)) / M
            return loss, grad

        def f_cosine(x, r):
            return F.cosine_embedding_loss(x, r[0].float(), torch.ones(x.shape[0], device=x.device))

        _ANALYTIC = {
            "tr_cross_entropy_bwd": a_cross_entropy, "tr_softcap_ce_bwd": a_softcap,
            "tr_zloss_ce_bwd": a_zloss, "tr_focal_ce_bwd": a_focal, "tr_ls_ce_bwd": a_ls,
            "tr_poly1_ce_bwd": a_poly1, "tr_kl_distill_bwd": a_kl,
            "tr_reverse_kl_distill_bwd": a_rkl, "tr_js_distill_bwd": a_js,
            "tr_temp_distill_bwd": a_temp, "tr_bce_logits_bwd": a_bce, "tr_huber_bwd": a_huber,
            "tr_smooth_l1_bwd": a_smooth_l1, "tr_cosine_embed_bwd": a_cosine,
        }
        _FWD = {
            "tr_cross_entropy_bwd": f_cross_entropy, "tr_softcap_ce_bwd": f_softcap,
            "tr_zloss_ce_bwd": f_zloss, "tr_focal_ce_bwd": f_focal, "tr_ls_ce_bwd": f_ls,
            "tr_poly1_ce_bwd": f_poly1, "tr_kl_distill_bwd": f_kl,
            "tr_reverse_kl_distill_bwd": f_rkl, "tr_js_distill_bwd": f_js,
            "tr_temp_distill_bwd": f_temp, "tr_bce_logits_bwd": f_bce, "tr_huber_bwd": f_huber,
            "tr_smooth_l1_bwd": f_smooth_l1, "tr_cosine_embed_bwd": f_cosine,
        }
        analytic = _ANALYTIC[op]
        forward = _FWD[op]

        if op in _CE_LOSS_OPS:
            def get_inputs(shape, device="cuda", seed=0):
                M, V = shape["M"], shape["V"]
                return (randn((M, V), seed, device, 2.0), targets(M, V, seed + 1, device))
        elif op in _DISTILL_OPS:
            def get_inputs(shape, device="cuda", seed=0):
                M, V = shape["M"], shape["V"]
                return (randn((M, V), seed, device, 2.0), randn((M, V), seed + 1, device, 2.0))
        elif op in _ELEM_LOSS_OPS:
            def get_inputs(shape, device="cuda", seed=0):
                M, N = shape["M"], shape["N"]
                x = randn((M, N), seed, device, 1.0)
                if op == "tr_bce_logits_bwd":
                    tgt = torch.rand((M, N), generator=gen(seed + 1, device),
                                     device=device, dtype=torch.float32).to(tdt)
                else:
                    tgt = randn((M, N), seed + 1, device, 1.0)
                return (x, tgt)
        else:  # cosine
            def get_inputs(shape, device="cuda", seed=0):
                M, N = shape["M"], shape["N"]
                return (randn((M, N), seed, device, 1.0), randn((M, N), seed + 1, device, 1.0))

        def ref_fn(*inp):
            loss, grad = analytic(*inp)
            return (loss.to(tdt), grad.to(tdt))

        def baseline_fn(*inp):
            x = inp[0].detach().float().requires_grad_(True)
            loss = forward(x, inp[1:])
            loss.backward()
            return (loss.detach().to(tdt), x.grad.to(tdt))
        arity = 2

    # ===================================================================== #
    # GRADIENT UTILITIES
    # ===================================================================== #
    elif op == "tr_grad_clip_per_layer":
        def get_inputs(shape, device="cuda", seed=0):
            G, N = shape["G"], shape["N"]
            return (randn((G, N), seed, device, 1.0), 1.0)

        def run(a, hi):
            grads, max_norm = a
            g = grads.float() if hi else grads
            norm = g.norm(dim=1, keepdim=True)
            coef = (max_norm / (norm + CLIP_EPS)).clamp(max=1.0)
            return (g * coef,)
        ref_fn, baseline_fn = _mk(run)
        arity = 2

    elif op == "tr_agc":
        def get_inputs(shape, device="cuda", seed=0):
            G, N = shape["G"], shape["N"]
            return (randn((G, N), seed, device, 0.1), randn((G, N), seed + 1, device, 1.0),
                    AGC_CLIP, AGC_EPS)

        def run(a, hi):
            params, grads, clip, eps = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            p, g = cv(params), cv(grads)
            pn = p.norm(dim=1, keepdim=True).clamp(min=eps)
            gn = g.norm(dim=1, keepdim=True)
            scale = (clip * pn / (gn + 1e-12)).clamp(max=1.0)
            return (g * scale,)
        ref_fn, baseline_fn = _mk(run)
        arity = 4

    elif op == "tr_ema_update":
        def get_inputs(shape, device="cuda", seed=0):
            G, N = shape["G"], shape["N"]
            return (randn((G, N), seed, device, 0.02), randn((G, N), seed + 1, device, 0.02), EMA_DECAY)

        def run(a, hi):
            ema, param, decay = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            e, p = cv(ema), cv(param)
            return (decay * e + (1.0 - decay) * p,)
        ref_fn, baseline_fn = _mk(run)
        arity = 3

    elif op == "tr_global_l2_norm":
        def get_inputs(shape, device="cuda", seed=0):
            G, N = shape["G"], shape["N"]
            return (randn((G, N), seed, device, 1.0),)

        def run(a, hi):
            (blob,) = a
            b = blob.float() if hi else blob
            return (b.norm(),)
        ref_fn, baseline_fn = _mk(run)
        arity = 1

    elif op == "tr_grad_accum_scale":
        def get_inputs(shape, device="cuda", seed=0):
            G, N = shape["G"], shape["N"]
            return (randn((G, N), seed, device, 0.05), randn((G, N), seed + 1, device, 0.01), ACCUM_SCALE)

        def run(a, hi):
            accum, grad, scale = a
            cv = (lambda x: x.float()) if hi else (lambda x: x)
            return (cv(accum) + scale * cv(grad),)
        ref_fn, baseline_fn = _mk(run)
        arity = 3

    elif op == "tr_grad_zero_center":
        def get_inputs(shape, device="cuda", seed=0):
            G, N = shape["G"], shape["N"]
            return (randn((G, N), seed, device, 1.0),)

        def run(a, hi):
            (grads,) = a
            g = grads.float() if hi else grads
            return (g - g.mean(dim=1, keepdim=True),)
        ref_fn, baseline_fn = _mk(run)
        arity = 1

    else:
        raise ValueError(f"unknown breadth op {op!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op, "dtype_name": dtype,
          "family": family, "mutates_input": op in TRAIN_MUTATES_INPUT}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Naive compiling+correct Triton starter seeds (the policy optimizes these).
# Elementwise optimizers / grad-utils are REAL per-element/per-row Triton; the
# reduction/matmul frontier ops (Muon NS, Adafactor factoring, quant de/requant,
# LAMB/LARS/NovoGrad trust ratios, the loss backwards) do the reduction in torch
# and the bulk elementwise pass in Triton - FUSING the torch part into Triton is
# precisely the optimization target.
# --------------------------------------------------------------------------- #
def _elem_seed(op, tldt, doc, entry_sig, tensors, kscalars, host, body, ret, numel_src="param"):
    kptr = ", ".join(b + "_ptr" for b, _, _ in tensors)
    knames = "".join(", " + n for n, _ in kscalars)
    loads = "\n".join("    %s = tl.load(%s_ptr + offs, mask=mask).to(tl.float32)" % (b, b)
                      for b, ld, _ in tensors if ld)
    bod = "\n".join("    " + ln for ln in body)
    stores = "\n".join("    tl.store(%s_ptr + offs, %s.to(%s), mask=mask)" % (b, b, tldt)
                       for b, _, st in tensors if st)
    hostl = "".join("\n    " + ln for ln in host)
    lptr = ", ".join(b for b, _, _ in tensors)
    lval = "".join(", " + e for _, e in kscalars)
    return ('"""%s"""\n' % doc
            + "from __future__ import annotations\n"
            + "import torch, triton, triton.language as tl\n\n\n"
            + "@triton.jit\n"
            + "def _%s_kernel(%s%s, numel, BLOCK: tl.constexpr):\n" % (op, kptr, knames)
            + "    pid = tl.program_id(0)\n"
            + "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            + "    mask = offs < numel\n"
            + loads + "\n" + bod + "\n" + stores + "\n\n\n"
            + "def %s(%s):\n" % (op, entry_sig)
            + "    numel = %s.numel()" % numel_src + hostl + "\n"
            + "    BLOCK = 1024\n"
            + "    grid = (triton.cdiv(numel, BLOCK),)\n"
            + "    _%s_kernel[grid](%s%s, numel, BLOCK=BLOCK, num_warps=4)\n" % (op, lptr, lval)
            + "    return %s\n" % ret)


def _loss_seed(op, tldt, first, other, fwd):
    """Loss+backward seed: fp32 forward + autograd dinput, then a Triton pass
    materializes the [., .] gradient (fuse the whole loss+bwd into Triton)."""
    doc = ("GENERATED breadth %s seed. Naive: fp32 forward + autograd input-gradient, "
           "then a Triton elementwise pass materializes the gradient (the FUSED "
           "loss+backward Triton kernel is the optimization target)." % op)
    return ('"""%s"""\n' % doc
            + "from __future__ import annotations\n"
            + "import torch, triton, triton.language as tl\n"
            + "import torch.nn.functional as F\n\n\n"
            + "@triton.jit\n"
            + "def _%s_copy_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):\n" % op
            + "    pid = tl.program_id(0)\n"
            + "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            + "    mask = offs < numel\n"
            + "    v = tl.load(src_ptr + offs, mask=mask).to(tl.float32)\n"
            + "    tl.store(dst_ptr + offs, v.to(%s), mask=mask)\n\n\n" % tldt
            + "def %s(%s, %s):\n" % (op, first, other)
            + "    x = %s.float().detach().requires_grad_(True)\n" % first
            + "    %s\n" % fwd
            + "    (g,) = torch.autograd.grad(loss, x)\n"
            + "    grad = torch.empty_like(%s)\n" % first
            + "    numel = grad.numel()\n"
            + "    BLOCK = 1024\n"
            + "    grid = (triton.cdiv(numel, BLOCK),)\n"
            + "    _%s_copy_kernel[grid](g.contiguous(), grad, numel, BLOCK=BLOCK, num_warps=4)\n" % op
            + "    return loss.detach().to(%s.dtype), grad\n" % first)


def _muon_seed(op, tldt, ns_steps):
    doc = ("GENERATED breadth %s seed. (Nesterov) momentum + aspect-scaled decoupled "
           "update in Triton elementwise kernels; the %d-iter Newton-Schulz "
           "orthogonalization runs as fp32 torch matmuls (FUSE them into Triton). "
           "Returns (param, momentum_buffer)." % (op, ns_steps))
    a, b, c = NS_COEFFS
    return ('"""%s"""\n' % doc
            + "from __future__ import annotations\n"
            + "import torch, triton, triton.language as tl\n\n"
            + "_A, _B, _C, _STEPS, _EPS = %r, %r, %r, %d, %r\n\n\n" % (a, b, c, ns_steps, MUON_EPS)
            + "@triton.jit\n"
            + "def _%s_mom_kernel(g_ptr, buf_ptr, out_ptr, numel, momentum, BLOCK: tl.constexpr):\n" % op
            + "    pid = tl.program_id(0)\n"
            + "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            + "    mask = offs < numel\n"
            + "    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)\n"
            + "    buf = tl.load(buf_ptr + offs, mask=mask).to(tl.float32)\n"
            + "    buf = buf + (1.0 - momentum) * (g - buf)\n"
            + "    upd = g + momentum * (buf - g)\n"
            + "    tl.store(buf_ptr + offs, buf.to(%s), mask=mask)\n" % tldt
            + "    tl.store(out_ptr + offs, upd, mask=mask)\n\n\n"
            + "@triton.jit\n"
            + "def _%s_upd_kernel(p_ptr, o_ptr, numel, decay, alpha, BLOCK: tl.constexpr):\n" % op
            + "    pid = tl.program_id(0)\n"
            + "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            + "    mask = offs < numel\n"
            + "    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)\n"
            + "    o = tl.load(o_ptr + offs, mask=mask).to(tl.float32)\n"
            + "    p = p * decay - alpha * o\n"
            + "    tl.store(p_ptr + offs, p.to(%s), mask=mask)\n\n\n" % tldt
            + "def _ns(x):\n"
            + "    t = x.shape[-2] > x.shape[-1]\n"
            + "    if t:\n        x = x.mT\n"
            + "    x = x / x.norm().clamp(min=_EPS)\n"
            + "    for _ in range(_STEPS):\n"
            + "        A = x @ x.mT\n"
            + "        B = _B * A + _C * (A @ A)\n"
            + "        x = _A * x + B @ x\n"
            + "    if t:\n        x = x.mT\n"
            + "    return x\n\n\n"
            + "def %s(param, grad, momentum_buffer, lr, weight_decay, momentum, eps, "
              "ns_a, ns_b, ns_c, ns_steps):\n" % op
            + "    numel = param.numel()\n"
            + "    geff = torch.empty_like(param, dtype=torch.float32)\n"
            + "    BLOCK = 1024\n"
            + "    grid = (triton.cdiv(numel, BLOCK),)\n"
            + "    _%s_mom_kernel[grid](grad, momentum_buffer, geff, numel, momentum, "
              "BLOCK=BLOCK, num_warps=4)\n" % op
            + "    o = _ns(geff.view_as(param)).contiguous()\n"
            + "    scale = max(1.0, param.shape[-2] / param.shape[-1]) ** 0.5\n"
            + "    _%s_upd_kernel[grid](param, o, numel, 1.0 - lr * weight_decay, lr * scale, "
              "BLOCK=BLOCK, num_warps=4)\n" % op
            + "    return param, momentum_buffer\n")


def _adafactor_seed(op, tldt):
    doc = ("GENERATED breadth %s seed. Factored (row/col) second-moment estimate + "
           "update-clipping in torch; the final decoupled-decay scaled update runs in "
           "a Triton elementwise kernel. Returns (param, row_var, col_var)." % op)
    return ('"""%s"""\n' % doc
            + "from __future__ import annotations\n"
            + "import torch, triton, triton.language as tl\n\n\n"
            + "@triton.jit\n"
            + "def _%s_kernel(p_ptr, u_ptr, numel, decay, coef, BLOCK: tl.constexpr):\n" % op
            + "    pid = tl.program_id(0)\n"
            + "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            + "    mask = offs < numel\n"
            + "    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)\n"
            + "    u = tl.load(u_ptr + offs, mask=mask).to(tl.float32)\n"
            + "    p = p * decay - coef * u\n"
            + "    tl.store(p_ptr + offs, p.to(%s), mask=mask)\n\n\n" % tldt
            + "def %s(param, grad, row_var, col_var, lr, beta2_decay, eps1, eps2, d, "
              "weight_decay, step):\n" % op
            + "    sf = float(step)\n"
            + "    omb2 = sf ** beta2_decay\n"
            + "    rho_t = min(lr, 1.0 / (sf ** 0.5))\n"
            + "    g = grad.float()\n"
            + "    alpha = max(eps2, param.float().norm().item() / (param.numel() ** 0.5)) * rho_t\n"
            + "    rv = row_var.float() + omb2 * ((g * g).mean(dim=-1, keepdim=True) - row_var.float())\n"
            + "    cv = col_var.float() + omb2 * ((g * g).mean(dim=-2, keepdim=True) - col_var.float())\n"
            + "    var = (rv @ cv) / rv.mean(dim=-2, keepdim=True).clamp(min=eps1)\n"
            + "    upd = var.clamp(min=eps1 * eps1).rsqrt() * g\n"
            + "    denom = max(1.0, upd.norm().item() / ((upd.numel() ** 0.5) * d))\n"
            + "    row_var.copy_(rv.to(row_var.dtype)); col_var.copy_(cv.to(col_var.dtype))\n"
            + "    numel = param.numel()\n"
            + "    BLOCK = 1024\n"
            + "    grid = (triton.cdiv(numel, BLOCK),)\n"
            + "    _%s_kernel[grid](param, upd.contiguous(), numel, 1.0 - lr * weight_decay, "
              "alpha / denom, BLOCK=BLOCK, num_warps=4)\n" % op
            + "    return param, row_var, col_var\n")


def _quant_seed(op, tldt, kind, dec):
    doc = ("GENERATED breadth %s seed. Blockwise %s-quantized Adam states: dequant "
           "(torch) -> fused fp32 Adam update in a Triton kernel -> requant (torch). "
           "Fusing the de/requant into the Triton kernel is the target." % (op, kind))
    qmax = 127.0 if kind == "int8" else 448.0
    qdt = "torch.int8" if kind == "int8" else "torch.float8_e4m3fn"
    rnd = ".round().clamp(-127, 127)" if kind == "int8" else ""
    decay_line = ("    p = p * (1.0 - lr * weight_decay)\n" if dec
                  else "    g = g + weight_decay * p\n")
    return ('"""%s"""\n' % doc
            + "from __future__ import annotations\n"
            + "import torch, triton, triton.language as tl\n\n"
            + "_QB, _QMAX, _QDT = %d, %r, %s\n\n\n" % (QUANT_BLOCK, qmax, qdt)
            + "def _dequant(q, scale):\n"
            + "    qf = q.reshape(-1)\n"
            + "    s = scale.repeat_interleave(_QB)[:qf.numel()]\n"
            + "    return (qf.float() * s).reshape(q.shape)\n\n\n"
            + "def _quant(x):\n"
            + "    xf = x.reshape(-1).float()\n"
            + "    n = xf.numel()\n"
            + "    nb = (n + _QB - 1) // _QB\n"
            + "    pad = nb * _QB - n\n"
            + "    xp = torch.cat([xf, xf.new_zeros(pad)]) if pad else xf\n"
            + "    xb = xp.reshape(nb, _QB)\n"
            + "    amax = xb.abs().amax(1)\n"
            + "    scale = torch.where(amax > 0, amax / _QMAX, torch.ones_like(amax))\n"
            + "    q = (xb / scale[:, None])%s.to(_QDT).reshape(-1)[:n].reshape(x.shape)\n" % rnd
            + "    return q, scale\n\n\n"
            + "@triton.jit\n"
            + "def _%s_kernel(p_ptr, g_ptr, m_ptr, v_ptr, mo_ptr, vo_ptr, numel, lr, "
              "weight_decay, b1, b2, eps, step_size, bc2sqrt, BLOCK: tl.constexpr):\n" % op
            + "    pid = tl.program_id(0)\n"
            + "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            + "    mask = offs < numel\n"
            + "    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)\n"
            + "    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)\n"
            + "    m = tl.load(m_ptr + offs, mask=mask).to(tl.float32)\n"
            + "    v = tl.load(v_ptr + offs, mask=mask).to(tl.float32)\n"
            + decay_line
            + "    m = m + (1.0 - b1) * (g - m)\n"
            + "    v = b2 * v + (1.0 - b2) * g * g\n"
            + "    p = p - step_size * m / (tl.sqrt(v) / bc2sqrt + eps)\n"
            + "    tl.store(p_ptr + offs, p.to(%s), mask=mask)\n" % tldt
            + "    tl.store(mo_ptr + offs, m, mask=mask)\n"
            + "    tl.store(vo_ptr + offs, v, mask=mask)\n\n\n"
            + "def %s(param, grad, q_exp_avg, s_exp_avg, q_exp_avg_sq, s_exp_avg_sq, "
              "lr, b1, b2, eps, weight_decay, step):\n" % op
            + "    m = _dequant(q_exp_avg, s_exp_avg)\n"
            + "    v = _dequant(q_exp_avg_sq, s_exp_avg_sq)\n"
            + "    mo = torch.empty_like(m); vo = torch.empty_like(v)\n"
            + "    bc1 = 1.0 - b1 ** step\n"
            + "    step_size = lr / bc1\n"
            + "    bc2sqrt = (1.0 - b2 ** step) ** 0.5\n"
            + "    numel = param.numel()\n"
            + "    BLOCK = 1024\n"
            + "    grid = (triton.cdiv(numel, BLOCK),)\n"
            + "    _%s_kernel[grid](param, grad, m, v, mo, vo, numel, lr, weight_decay, b1, b2, "
              "eps, step_size, bc2sqrt, BLOCK=BLOCK, num_warps=4)\n" % op
            + "    qm, sm = _quant(mo); qv, sv = _quant(vo)\n"
            + "    return param, qm, sm, qv, sv\n")


def _trust_seed(op, tldt, doc, entry_sig, host, update_expr, extra_stores, ret):
    """Layer-wise/trust-ratio optimizer seed: torch computes the (global-norm)
    trust scalar + the fp32 update tensor ``upd``, a Triton kernel applies the
    elementwise param step, buffers copied back in torch."""
    return ('"""%s"""\n' % doc
            + "from __future__ import annotations\n"
            + "import torch, triton, triton.language as tl\n\n\n"
            + "@triton.jit\n"
            + "def _%s_kernel(p_ptr, u_ptr, numel, coef, BLOCK: tl.constexpr):\n" % op
            + "    pid = tl.program_id(0)\n"
            + "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            + "    mask = offs < numel\n"
            + "    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)\n"
            + "    u = tl.load(u_ptr + offs, mask=mask).to(tl.float32)\n"
            + "    p = p - coef * u\n"
            + "    tl.store(p_ptr + offs, p.to(%s), mask=mask)\n\n\n" % tldt
            + "def %s(%s):\n" % (op, entry_sig)
            + "".join("    %s\n" % ln for ln in host)
            + "    %s\n" % update_expr
            + "".join("    %s\n" % ln for ln in extra_stores)
            + "    numel = param.numel()\n"
            + "    BLOCK = 1024\n"
            + "    grid = (triton.cdiv(numel, BLOCK),)\n"
            + "    _%s_kernel[grid](param, upd.contiguous(), numel, coef, BLOCK=BLOCK, num_warps=4)\n" % op
            + "    return %s\n" % ret)


def _perrow_seed(op, tldt, doc, entry_sig, load_ptrs, prep, coef_expr, apply_ptr, ret,
                 second_pass=True, out_scalar=False):
    """Per-row (per-layer) reduction seed: one program per row streams the row in
    BLOCK chunks to compute a norm/mean, then rescales the row in place."""
    lines = ['"""%s"""' % doc, "from __future__ import annotations",
             "import torch, triton, triton.language as tl", "", "", "@triton.jit"]
    kargs = ", ".join(load_ptrs)
    lines.append("def _%s_kernel(%s, out_ptr, sm, N, arg0, BLOCK: tl.constexpr):" % (op, kargs))
    lines += ["    row = tl.program_id(0)", "    base = row * sm", "    acc = 0.0", "    acc2 = 0.0",
              "    for start in range(0, N, BLOCK):", "        offs = start + tl.arange(0, BLOCK)",
              "        m = offs < N"]
    lines += ["        " + ln for ln in prep]
    lines.append("    " + coef_expr)
    if out_scalar:
        lines.append("    tl.store(out_ptr + row, coef)")
    else:
        lines += ["    for start in range(0, N, BLOCK):", "        offs = start + tl.arange(0, BLOCK)",
                  "        m = offs < N",
                  "        x = tl.load(%s + base + offs, mask=m, other=0.0).to(tl.float32)" % apply_ptr,
                  "        tl.store(%s + base + offs, (x * coef).to(%s), mask=m)" % (apply_ptr, tldt)]
    lines += ["", "", "def %s(%s):" % (op, entry_sig)]
    lines += ["    " + ln for ln in ret]
    return "\n".join(lines) + "\n"


def seed_source(op: str, dtype: str) -> str:
    tldt = DTYPES[dtype][1]

    # ---------------- flat elementwise optimizers (real Triton) --------------
    if op in _SGD_OPS:
        doc = "GENERATED breadth %s seed (%s). Fused SGD(+momentum/nesterov/wd) step." % (op, dtype)
        return _elem_seed(op, tldt, doc, "param, grad, buf, lr, momentum, dampening, wd, nesterov",
                          [("param", True, True), ("grad", True, False), ("buf", True, True)],
                          [("lr", "lr"), ("momentum", "momentum"), ("dampening", "dampening"),
                           ("wd", "wd"), ("nesterov", "nesterov")], [],
                          ["grad = grad + wd * param",
                           "buf = momentum * buf + (1.0 - dampening) * grad",
                           "d = tl.where(nesterov, grad + momentum * buf, buf)",
                           "param = param - lr * d"], "param, buf")

    if op == "tr_foreach_sgd":
        doc = "GENERATED breadth %s seed (%s). Fused multi-tensor SGD+momentum over a blob." % (op, dtype)
        return _elem_seed(op, tldt, doc, "param, grad, buf, lr, momentum, wd",
                          [("param", True, True), ("grad", True, False), ("buf", True, True)],
                          [("lr", "lr"), ("momentum", "momentum"), ("wd", "wd")], [],
                          ["grad = grad + wd * param", "buf = momentum * buf + grad",
                           "param = param - lr * buf"], "param, buf")

    if op in _ADAM_OPS or op == "tr_foreach_adamw":
        if op == "tr_foreach_adamw":
            dec, ams = True, False
        else:
            dec, ams = _ADAM_CFG[op]["decoupled"], _ADAM_CFG[op]["amsgrad"]
        tensors = [("param", True, True), ("grad", True, False),
                   ("exp_avg", True, True), ("exp_avg_sq", True, True)]
        sig = "param, grad, exp_avg, exp_avg_sq"
        ret = "param, exp_avg, exp_avg_sq"
        body = ["param = param * (1.0 - lr * wd)"] if dec else ["grad = grad + wd * param"]
        body += ["exp_avg = exp_avg + (1.0 - b1) * (grad - exp_avg)",
                 "exp_avg_sq = b2 * exp_avg_sq + (1.0 - b2) * grad * grad"]
        if ams:
            tensors.insert(4, ("max_exp_avg_sq", True, True))
            sig += ", max_exp_avg_sq"
            ret += ", max_exp_avg_sq"
            body += ["max_exp_avg_sq = tl.maximum(max_exp_avg_sq, exp_avg_sq)",
                     "denom = tl.sqrt(max_exp_avg_sq) / bc2sqrt + eps"]
        else:
            body += ["denom = tl.sqrt(exp_avg_sq) / bc2sqrt + eps"]
        body += ["param = param - step_size * exp_avg / denom"]
        sig += ", lr, b1, b2, eps, wd, step"
        doc = "GENERATED breadth %s seed (%s). Fused Adam-family step." % (op, dtype)
        return _elem_seed(op, tldt, doc, sig, tensors,
                          [("lr", "lr"), ("wd", "wd"), ("b1", "b1"), ("b2", "b2"),
                           ("eps", "eps"), ("step_size", "step_size"), ("bc2sqrt", "bc2sqrt")],
                          ["bc1 = 1.0 - b1 ** step", "step_size = lr / bc1",
                           "bc2sqrt = (1.0 - b2 ** step) ** 0.5"], body, ret)

    if op in _RMSPROP_OPS:
        cfg = _RMSPROP_CFG[op]
        cen, mom = cfg["centered"], cfg["momentum"] > 0
        tensors = [("param", True, True), ("grad", True, False), ("square_avg", True, True)]
        sig = "param, grad, square_avg"
        ret = "param, square_avg"
        body = ["grad = grad + wd * param",
                "square_avg = alpha * square_avg + (1.0 - alpha) * grad * grad"]
        if cen:
            tensors.append(("grad_avg", True, True)); sig += ", grad_avg"; ret += ", grad_avg"
            body += ["grad_avg = grad_avg + (1.0 - alpha) * (grad - grad_avg)",
                     "avg = tl.sqrt(square_avg - grad_avg * grad_avg) + eps"]
        else:
            body += ["avg = tl.sqrt(square_avg) + eps"]
        if mom:
            tensors.append(("momentum_buffer", True, True))
            sig += ", momentum_buffer"; ret += ", momentum_buffer"
            body += ["momentum_buffer = momentum * momentum_buffer + grad / avg",
                     "param = param - lr * momentum_buffer"]
        else:
            body += ["param = param - lr * grad / avg"]
        sig += ", lr, alpha, eps, wd, momentum"
        doc = "GENERATED breadth %s seed (%s). Fused RMSprop step." % (op, dtype)
        return _elem_seed(op, tldt, doc, sig, tensors,
                          [("lr", "lr"), ("alpha", "alpha"), ("eps", "eps"),
                           ("wd", "wd"), ("momentum", "momentum")], [], body, ret)

    if op == "tr_adagrad":
        doc = "GENERATED breadth %s seed (%s). Fused Adagrad step." % (op, dtype)
        return _elem_seed(op, tldt, doc, "param, grad, state_sum, lr, eps, wd, lr_decay, step",
                          [("param", True, True), ("grad", True, False), ("state_sum", True, True)],
                          [("clr", "clr"), ("eps", "eps"), ("wd", "wd")],
                          ["clr = lr / (1.0 + (step - 1) * lr_decay)"],
                          ["grad = grad + wd * param", "state_sum = state_sum + grad * grad",
                           "param = param - clr * grad / (tl.sqrt(state_sum) + eps)"],
                          "param, state_sum")

    if op == "tr_adadelta":
        doc = "GENERATED breadth %s seed (%s). Fused Adadelta step." % (op, dtype)
        return _elem_seed(op, tldt, doc, "param, grad, square_avg, acc_delta, lr, rho, eps, wd",
                          [("param", True, True), ("grad", True, False),
                           ("square_avg", True, True), ("acc_delta", True, True)],
                          [("lr", "lr"), ("rho", "rho"), ("eps", "eps"), ("wd", "wd")], [],
                          ["grad = grad + wd * param",
                           "square_avg = rho * square_avg + (1.0 - rho) * grad * grad",
                           "delta = tl.sqrt(acc_delta + eps) / tl.sqrt(square_avg + eps) * grad",
                           "acc_delta = rho * acc_delta + (1.0 - rho) * delta * delta",
                           "param = param - lr * delta"], "param, square_avg, acc_delta")

    if op == "tr_adamax":
        doc = "GENERATED breadth %s seed (%s). Fused Adamax step." % (op, dtype)
        return _elem_seed(op, tldt, doc, "param, grad, exp_avg, exp_inf, lr, b1, b2, eps, wd, step",
                          [("param", True, True), ("grad", True, False),
                           ("exp_avg", True, True), ("exp_inf", True, True)],
                          [("b1", "b1"), ("b2", "b2"), ("eps", "eps"), ("wd", "wd"), ("clr", "clr")],
                          ["clr = lr / (1.0 - b1 ** step)"],
                          ["grad = grad + wd * param",
                           "exp_avg = exp_avg + (1.0 - b1) * (grad - exp_avg)",
                           "exp_inf = tl.maximum(b2 * exp_inf, tl.abs(grad) + eps)",
                           "param = param - clr * exp_avg / exp_inf"], "param, exp_avg, exp_inf")

    if op == "tr_nadam":
        host = ["bc2 = 1.0 - b2 ** step",
                "mu_t = b1 * (1.0 - 0.5 * 0.96 ** (step * momentum_decay))",
                "mu_next = b1 * (1.0 - 0.5 * 0.96 ** ((step + 1) * momentum_decay))",
                "mu_prod = 1.0",
                "for _i in range(1, step + 1):",
                "    mu_prod = mu_prod * b1 * (1.0 - 0.5 * 0.96 ** (_i * momentum_decay))",
                "c1 = -lr * (1.0 - mu_t) / (1.0 - mu_prod)",
                "c2 = -lr * mu_next / (1.0 - mu_prod * mu_next)"]
        doc = "GENERATED breadth %s seed (%s). Fused NAdam step." % (op, dtype)
        return _elem_seed(op, tldt, doc,
                          "param, grad, exp_avg, exp_avg_sq, lr, b1, b2, eps, wd, momentum_decay, step",
                          [("param", True, True), ("grad", True, False),
                           ("exp_avg", True, True), ("exp_avg_sq", True, True)],
                          [("b1", "b1"), ("b2", "b2"), ("eps", "eps"), ("bc2", "bc2"),
                           ("c1", "c1"), ("c2", "c2")], host,
                          ["exp_avg = exp_avg + (1.0 - b1) * (grad - exp_avg)",
                           "exp_avg_sq = b2 * exp_avg_sq + (1.0 - b2) * grad * grad",
                           "den = tl.sqrt(exp_avg_sq / bc2) + eps",
                           "param = param + grad * c1 / den + exp_avg * c2 / den"],
                          "param, exp_avg, exp_avg_sq")

    if op == "tr_radam":
        host = ["bc1 = 1.0 - b1 ** step", "bc2 = 1.0 - b2 ** step",
                "rho_inf = 2.0 / (1.0 - b2) - 1.0",
                "rho_t = rho_inf - 2.0 * step * (b2 ** step) / bc2",
                "if rho_t > 5.0:",
                "    rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / "
                "((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)) ** 0.5",
                "    step_coef = lr * rect * (bc2 ** 0.5) / bc1",
                "    plain_coef = 0.0",
                "    use_rect = 1",
                "else:",
                "    step_coef = 0.0",
                "    plain_coef = lr / bc1",
                "    use_rect = 0"]
        doc = "GENERATED breadth %s seed (%s). Fused RAdam step (variance rectification)." % (op, dtype)
        return _elem_seed(op, tldt, doc,
                          "param, grad, exp_avg, exp_avg_sq, lr, b1, b2, eps, wd, step",
                          [("param", True, True), ("grad", True, False),
                           ("exp_avg", True, True), ("exp_avg_sq", True, True)],
                          [("b1", "b1"), ("b2", "b2"), ("eps", "eps"), ("step_coef", "step_coef"),
                           ("plain_coef", "plain_coef"), ("use_rect", "use_rect")], host,
                          ["exp_avg = exp_avg + (1.0 - b1) * (grad - exp_avg)",
                           "exp_avg_sq = b2 * exp_avg_sq + (1.0 - b2) * grad * grad",
                           "den = tl.sqrt(exp_avg_sq) + eps",
                           "upd = tl.where(use_rect != 0, step_coef * exp_avg / den, "
                           "plain_coef * exp_avg)",
                           "param = param - upd"], "param, exp_avg, exp_avg_sq")

    if op == "tr_rprop":
        doc = "GENERATED breadth %s seed (%s). Fused Rprop step (per-elem step sizes)." % (op, dtype)
        return _elem_seed(op, tldt, doc, "param, grad, prev, step_size, etam, etap, smin, smax, step",
                          [("param", True, True), ("grad", True, False),
                           ("prev", True, True), ("step_size", True, True)],
                          [("etam", "etam"), ("etap", "etap"), ("smin", "smin"), ("smax", "smax")], [],
                          ["sign = grad * prev",
                           "mult = tl.where(sign > 0.0, etap, tl.where(sign < 0.0, etam, 1.0))",
                           "step_size = tl.minimum(tl.maximum(step_size * mult, smin), smax)",
                           "g2 = tl.where(sign < 0.0, 0.0, grad)",
                           "gs = tl.where(g2 > 0.0, 1.0, tl.where(g2 < 0.0, -1.0, 0.0))",
                           "param = param - gs * step_size", "prev = g2"],
                          "param, prev, step_size")

    if op == "tr_adabelief":
        doc = "GENERATED breadth %s seed (%s). Fused AdaBelief step (variance of grad-EMA)." % (op, dtype)
        return _elem_seed(op, tldt, doc, "param, grad, exp_avg, exp_avg_var, lr, b1, b2, eps, wd, step",
                          [("param", True, True), ("grad", True, False),
                           ("exp_avg", True, True), ("exp_avg_var", True, True)],
                          [("lr", "lr"), ("b1", "b1"), ("b2", "b2"), ("eps", "eps"),
                           ("wd", "wd"), ("bc1", "bc1"), ("bc2", "bc2")],
                          ["bc1 = 1.0 - b1 ** step", "bc2 = 1.0 - b2 ** step"],
                          ["param = param * (1.0 - lr * wd)",
                           "exp_avg = b1 * exp_avg + (1.0 - b1) * grad",
                           "diff = grad - exp_avg",
                           "exp_avg_var = b2 * exp_avg_var + (1.0 - b2) * diff * diff + eps",
                           "mhat = exp_avg / bc1", "shat = exp_avg_var / bc2",
                           "param = param - lr * mhat / (tl.sqrt(shat) + eps)"],
                          "param, exp_avg, exp_avg_var")

    # ---------------- frontier reduction/matmul optimizers -------------------
    if op in _MUON_OPS:
        return _muon_seed(op, tldt, _MUON_CFG[op]["ns_steps"])
    if op == "tr_adafactor":
        return _adafactor_seed(op, tldt)
    if op in _QUANT_OPS:
        c = _QUANT_CFG[op]
        return _quant_seed(op, tldt, c["kind"], c["decoupled"])

    if op == "tr_lamb":
        doc = "GENERATED breadth %s seed (%s). LAMB: Adam ratio + layer-wise trust ratio." % (op, dtype)
        host = ["m = b1 * exp_avg.float() + (1.0 - b1) * grad.float()",
                "v = b2 * exp_avg_sq.float() + (1.0 - b2) * grad.float() ** 2",
                "mhat = m / (1.0 - b1 ** step)", "vhat = v / (1.0 - b2 ** step)",
                "upd = mhat / (vhat.sqrt() + eps) + wd * param.float()",
                "trust = param.float().norm() / upd.norm().clamp(min=1e-30)",
                "coef = float(lr * trust)"]
        stores = ["exp_avg.copy_(m.to(exp_avg.dtype)); exp_avg_sq.copy_(v.to(exp_avg_sq.dtype))"]
        return _trust_seed(op, tldt, doc,
                           "param, grad, exp_avg, exp_avg_sq, lr, b1, b2, eps, wd, step",
                           host, "upd = upd", stores, "param, exp_avg, exp_avg_sq")

    if op == "tr_lars":
        doc = "GENERATED breadth %s seed (%s). LARS: layer-wise adaptive rate scaling." % (op, dtype)
        host = ["d = grad.float() + wd * param.float()",
                "local_lr = trust_coef * param.float().norm() / (d.norm() + eps)",
                "buf = momentum * momentum_buffer.float() + lr * local_lr * d",
                "upd = buf", "coef = 1.0"]
        stores = ["momentum_buffer.copy_(buf.to(momentum_buffer.dtype))"]
        return _trust_seed(op, tldt, doc,
                           "param, grad, momentum_buffer, lr, momentum, wd, trust_coef, eps",
                           host, "upd = upd", stores, "param, momentum_buffer")

    if op == "tr_novograd":
        doc = "GENERATED breadth %s seed (%s). NovoGrad: layer-wise 2nd moment." % (op, dtype)
        host = ["v = b2 * exp_avg_sq.float() + (1.0 - b2) * (grad.float() ** 2).sum()",
                "ghat = grad.float() / (v.sqrt() + eps) + wd * param.float()",
                "m = b1 * exp_avg.float() + ghat", "upd = m", "coef = float(lr)"]
        stores = ["exp_avg.copy_(m.to(exp_avg.dtype)); exp_avg_sq = v.to(exp_avg_sq.dtype)"]
        return _trust_seed(op, tldt, doc,
                           "param, grad, exp_avg, exp_avg_sq, lr, b1, b2, eps, wd",
                           host, "upd = upd", stores, "param, exp_avg, exp_avg_sq")

    # ---------------- fused loss + backward (14) -----------------------------
    _FWD = {
        "tr_cross_entropy_bwd": ("logits", "targets", "loss = F.cross_entropy(x, targets.long())"),
        "tr_softcap_ce_bwd": ("logits", "targets",
                              "loss = F.cross_entropy(%r * torch.tanh(x / %r), targets.long())"
                              % (SOFTCAP, SOFTCAP)),
        "tr_zloss_ce_bwd": ("logits", "targets",
                            "loss = F.cross_entropy(x, targets.long()) + %r * "
                            "(torch.logsumexp(x, -1) ** 2).mean()" % ZLOSS_LAMBDA),
        "tr_focal_ce_bwd": ("logits", "targets",
                            "p = torch.softmax(x, -1); pt = p.gather(1, targets.long()[:, None])"
                            ".squeeze(1); loss = (((1.0 - pt) ** %r) * (-torch.log(pt))).mean()"
                            % FOCAL_GAMMA),
        "tr_ls_ce_bwd": ("logits", "targets",
                         "loss = F.cross_entropy(x, targets.long(), label_smoothing=%r)" % LS_EPS),
        "tr_poly1_ce_bwd": ("logits", "targets",
                            "pt = torch.softmax(x, -1).gather(1, targets.long()[:, None]).squeeze(1); "
                            "loss = (F.cross_entropy(x, targets.long(), reduction='none') + %r * "
                            "(1.0 - pt)).mean()" % POLY1_EPS),
        "tr_kl_distill_bwd": ("student", "teacher",
                              "pt = torch.softmax(teacher.float(), -1); "
                              "loss = (pt * (torch.log(pt) - F.log_softmax(x, -1))).sum(1).mean()"),
        "tr_reverse_kl_distill_bwd": ("student", "teacher",
                                      "ps = torch.softmax(x, -1); pt = torch.softmax(teacher.float(), -1); "
                                      "loss = (ps * (torch.log(ps) - torch.log(pt))).sum(1).mean()"),
        "tr_js_distill_bwd": ("student", "teacher",
                              "ps = torch.softmax(x, -1); pt = torch.softmax(teacher.float(), -1); "
                              "mid = 0.5 * (ps + pt); loss = (0.5 * (ps * (torch.log(ps) - "
                              "torch.log(mid))).sum(1) + 0.5 * (pt * (torch.log(pt) - "
                              "torch.log(mid))).sum(1)).mean()"),
        "tr_temp_distill_bwd": ("student", "teacher",
                                "T = %r; ptT = torch.softmax(teacher.float() / T, -1); "
                                "loss = (T * T) * (ptT * (torch.log(ptT) - "
                                "F.log_softmax(x / T, -1))).sum(1).mean()" % DISTILL_T),
        "tr_bce_logits_bwd": ("inp", "target",
                              "loss = F.binary_cross_entropy_with_logits(x, target.float())"),
        "tr_huber_bwd": ("inp", "target", "loss = F.huber_loss(x, target.float(), delta=%r)" % HUBER_DELTA),
        "tr_smooth_l1_bwd": ("inp", "target",
                             "loss = F.smooth_l1_loss(x, target.float(), beta=%r)" % SMOOTHL1_BETA),
        "tr_cosine_embed_bwd": ("x1", "x2",
                                "loss = F.cosine_embedding_loss(x, x2.float(), "
                                "torch.ones(x.shape[0], device=x.device))"),
    }
    if op in _FWD:
        first, other, fwd = _FWD[op]
        return _loss_seed(op, tldt, first, other, fwd)

    # ---------------- gradient utilities -------------------------------------
    if op == "tr_ema_update":
        doc = "GENERATED breadth %s seed (%s). Fused EMA (Polyak) weight update." % (op, dtype)
        return _elem_seed(op, tldt, doc, "ema, param, decay",
                          [("ema", True, True), ("param", True, False)],
                          [("decay", "decay")], [],
                          ["ema = decay * ema + (1.0 - decay) * param"], "ema", numel_src="ema")

    if op == "tr_grad_accum_scale":
        doc = "GENERATED breadth %s seed (%s). Fused scaled gradient accumulation." % (op, dtype)
        return _elem_seed(op, tldt, doc, "accum, grad, scale",
                          [("accum", True, True), ("grad", True, False)],
                          [("scale", "scale")], [], ["accum = accum + scale * grad"],
                          "accum", numel_src="accum")

    if op == "tr_grad_clip_per_layer":
        return _perrow_seed(
            op, tldt, "GENERATED breadth %s seed (%s). Per-layer (per-row) grad-norm clip." % (op, dtype),
            "grads, max_norm", ["grads_ptr"],
            ["x = tl.load(grads_ptr + base + offs, mask=m, other=0.0).to(tl.float32)",
             "acc += tl.sum(x * x, axis=0)"],
            "coef = tl.minimum(arg0 / (tl.sqrt(acc) + %r), 1.0)" % CLIP_EPS, "grads_ptr",
            ["G, N = grads.shape",
             "_%s_kernel[(G,)](grads, grads, grads.stride(0), N, max_norm, BLOCK=1024, num_warps=8)" % op,
             "return grads"])

    if op == "tr_agc":
        lines = ['"""GENERATED breadth %s seed (%s). Adaptive gradient clipping (NFNets)."""' % (op, dtype),
                 "from __future__ import annotations", "import torch, triton, triton.language as tl",
                 "", "", "@triton.jit",
                 "def _%s_kernel(p_ptr, g_ptr, sm, N, clip, eps, BLOCK: tl.constexpr):" % op,
                 "    row = tl.program_id(0)", "    base = row * sm", "    pn = 0.0", "    gn = 0.0",
                 "    for start in range(0, N, BLOCK):", "        offs = start + tl.arange(0, BLOCK)",
                 "        m = offs < N",
                 "        p = tl.load(p_ptr + base + offs, mask=m, other=0.0).to(tl.float32)",
                 "        g = tl.load(g_ptr + base + offs, mask=m, other=0.0).to(tl.float32)",
                 "        pn += tl.sum(p * p, axis=0)", "        gn += tl.sum(g * g, axis=0)",
                 "    pnorm = tl.maximum(tl.sqrt(pn), eps)", "    gnorm = tl.sqrt(gn)",
                 "    coef = tl.minimum(clip * pnorm / (gnorm + 1e-12), 1.0)",
                 "    for start in range(0, N, BLOCK):", "        offs = start + tl.arange(0, BLOCK)",
                 "        m = offs < N",
                 "        g = tl.load(g_ptr + base + offs, mask=m, other=0.0).to(tl.float32)",
                 "        tl.store(g_ptr + base + offs, (g * coef).to(%s), mask=m)" % tldt,
                 "", "", "def %s(params, grads, clip, eps):" % op, "    G, N = grads.shape",
                 "    _%s_kernel[(G,)](params, grads, grads.stride(0), N, clip, eps, "
                 "BLOCK=1024, num_warps=8)" % op, "    return grads"]
        return "\n".join(lines) + "\n"

    if op == "tr_grad_zero_center":
        lines = ['"""GENERATED breadth %s seed (%s). Gradient centralization (per-row mean-0)."""'
                 % (op, dtype),
                 "from __future__ import annotations", "import torch, triton, triton.language as tl",
                 "", "", "@triton.jit",
                 "def _%s_kernel(g_ptr, sm, N, BLOCK: tl.constexpr):" % op,
                 "    row = tl.program_id(0)", "    base = row * sm", "    acc = 0.0",
                 "    for start in range(0, N, BLOCK):", "        offs = start + tl.arange(0, BLOCK)",
                 "        m = offs < N",
                 "        x = tl.load(g_ptr + base + offs, mask=m, other=0.0).to(tl.float32)",
                 "        acc += tl.sum(x, axis=0)", "    mean = acc / N",
                 "    for start in range(0, N, BLOCK):", "        offs = start + tl.arange(0, BLOCK)",
                 "        m = offs < N",
                 "        x = tl.load(g_ptr + base + offs, mask=m, other=0.0).to(tl.float32)",
                 "        tl.store(g_ptr + base + offs, (x - mean).to(%s), mask=m)" % tldt,
                 "", "", "def %s(grads):" % op, "    G, N = grads.shape",
                 "    _%s_kernel[(G,)](grads, grads.stride(0), N, BLOCK=1024, num_warps=8)" % op,
                 "    return grads"]
        return "\n".join(lines) + "\n"

    if op == "tr_global_l2_norm":
        lines = ['"""GENERATED breadth %s seed (%s). Multi-tensor global L2 norm."""' % (op, dtype),
                 "from __future__ import annotations", "import torch, triton, triton.language as tl",
                 "", "", "@triton.jit",
                 "def _%s_kernel(g_ptr, part_ptr, sm, N, BLOCK: tl.constexpr):" % op,
                 "    row = tl.program_id(0)", "    base = row * sm", "    acc = 0.0",
                 "    for start in range(0, N, BLOCK):", "        offs = start + tl.arange(0, BLOCK)",
                 "        m = offs < N",
                 "        x = tl.load(g_ptr + base + offs, mask=m, other=0.0).to(tl.float32)",
                 "        acc += tl.sum(x * x, axis=0)", "    tl.store(part_ptr + row, acc)",
                 "", "", "def %s(blob):" % op, "    G, N = blob.shape",
                 "    part = torch.empty((G,), device=blob.device, dtype=torch.float32)",
                 "    _%s_kernel[(G,)](blob, part, blob.stride(0), N, BLOCK=1024, num_warps=8)" % op,
                 "    return torch.sqrt(part.sum()).to(blob.dtype)"]
        return "\n".join(lines) + "\n"

    raise ValueError(f"unknown breadth op {op!r}")
