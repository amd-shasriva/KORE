"""GENERATED breadth tr_zloss_ce_bwd seed. Triton computes the fp32 row
loss and analytic input gradient directly; a second Triton kernel reduces row
losses to the scalar mean. No framework loss, autograd, or oracle delegation."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _loss_rows_kernel(x_ptr, other_ptr, rows_ptr, grad_ptr, M, V,
                      KIND: tl.constexpr, DISTILL: tl.constexpr,
                      BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * V
    max_s = -float("inf")
    max_t = -float("inf")
    raw_sum = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + base + offs, mask=mask,
                    other=-float("inf")).to(tl.float32)
        raw_sum += tl.sum(tl.where(mask, x, 0.0), axis=0)
        if KIND == 1:
            th = 2.0 * tl.sigmoid(2.0 * x / 30.0) - 1.0
            sx = 30.0 * th
        elif KIND == 9:
            sx = x / 2.0
        else:
            sx = x
        max_s = tl.maximum(max_s, tl.max(sx, axis=0))
        if DISTILL:
            te = tl.load(other_ptr + base + offs, mask=mask,
                         other=-float("inf")).to(tl.float32)
            if KIND == 9:
                te = te / 2.0
            max_t = tl.maximum(max_t, tl.max(te, axis=0))

    sum_s = 0.0
    sum_t = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + base + offs, mask=mask,
                    other=-float("inf")).to(tl.float32)
        if KIND == 1:
            th = 2.0 * tl.sigmoid(2.0 * x / 30.0) - 1.0
            sx = 30.0 * th
        elif KIND == 9:
            sx = x / 2.0
        else:
            sx = x
        sum_s += tl.sum(tl.where(mask, tl.exp(sx - max_s), 0.0), axis=0)
        if DISTILL:
            te = tl.load(other_ptr + base + offs, mask=mask,
                         other=-float("inf")).to(tl.float32)
            if KIND == 9:
                te = te / 2.0
            sum_t += tl.sum(tl.where(mask, tl.exp(te - max_t), 0.0), axis=0)
    lse_s = max_s + tl.log(sum_s)
    lse_t = max_t + tl.log(sum_t)

    row_value = 0.0
    aux = 0.0
    target = 0
    if DISTILL:
        for start in range(0, V, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < V
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            te = tl.load(other_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            if KIND == 9:
                x = x / 2.0
                te = te / 2.0
            log_ps = x - lse_s
            log_pt = te - lse_t
            ps = tl.exp(log_ps)
            pt = tl.exp(log_pt)
            if KIND == 6 or KIND == 9:
                term = pt * (log_pt - log_ps)
            elif KIND == 7:
                term = ps * (log_ps - log_pt)
            else:
                mid = 0.5 * (ps + pt)
                log_mid = tl.log(mid)
                cv = log_ps - log_mid
                term = 0.5 * ps * cv + 0.5 * pt * (log_pt - log_mid)
                aux += tl.sum(tl.where(mask, ps * cv, 0.0), axis=0)
            row_value += tl.sum(tl.where(mask, term, 0.0), axis=0)
        if KIND == 9:
            row_value = (2.0 * 2.0) * row_value
    else:
        target = tl.load(other_ptr + row).to(tl.int64)
        xt = tl.load(x_ptr + base + target).to(tl.float32)
        if KIND == 1:
            target_th = 2.0 * tl.sigmoid(2.0 * xt / 30.0) - 1.0
            sxt = 30.0 * target_th
        else:
            sxt = xt
        pt = tl.exp(sxt - lse_s)
        ce = lse_s - sxt
        if KIND == 2:
            row_value = ce + 0.0001 * lse_s * lse_s
        elif KIND == 3:
            row_value = (1.0 - pt) * (1.0 - pt) * (-tl.log(pt))
        elif KIND == 4:
            row_value = (1.0 - 0.1) * ce + 0.1 * (lse_s - raw_sum / V)
        elif KIND == 5:
            row_value = ce + 1.0 * (1.0 - pt)
        else:
            row_value = ce
    tl.store(rows_ptr + row, row_value)

    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        if DISTILL:
            te = tl.load(other_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            if KIND == 9:
                xs = x / 2.0
                ts = te / 2.0
            else:
                xs = x
                ts = te
            log_ps = xs - lse_s
            log_pt = ts - lse_t
            ps = tl.exp(log_ps)
            pt = tl.exp(log_pt)
            if KIND == 6:
                grad = (ps - pt) / M
            elif KIND == 7:
                grad = ps * ((log_ps - log_pt) - row_value) / M
            elif KIND == 8:
                cv = log_ps - tl.log(0.5 * (ps + pt))
                grad = 0.5 * ps * (cv - aux) / M
            else:
                grad = 2.0 * (ps - pt) / M
        else:
            if KIND == 1:
                th = 2.0 * tl.sigmoid(2.0 * x / 30.0) - 1.0
                sx = 30.0 * th
            else:
                sx = x
            ps = tl.exp(sx - lse_s)
            onehot = offs == target
            if KIND == 1:
                grad = (ps - onehot) * (1.0 - th * th) / M
            elif KIND == 2:
                grad = ((ps - onehot)
                        + 2.0 * 0.0001 * lse_s * ps) / M
            elif KIND == 3:
                pt = tl.exp((tl.load(x_ptr + base + target).to(tl.float32)) - lse_s)
                dldpt = (2.0 * (1.0 - pt) * tl.log(pt)
                         - (1.0 - pt) * (1.0 - pt) / pt)
                grad = dldpt * (pt * (onehot - ps)) / M
            elif KIND == 4:
                grad = (ps - (1.0 - 0.1) * onehot - 0.1 / V) / M
            elif KIND == 5:
                pt = tl.exp((tl.load(x_ptr + base + target).to(tl.float32)) - lse_s)
                grad = (1.0 + 1.0 * pt) * (ps - onehot) / M
            else:
                grad = (ps - onehot) / M
        tl.store(grad_ptr + base + offs, grad.to(tl.float16), mask=mask)


@triton.jit
def _mean_rows_kernel(rows_ptr, loss_ptr, M, BLOCK: tl.constexpr):
    acc = 0.0
    for start in range(0, M, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < M
        vals = tl.load(rows_ptr + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(loss_ptr, (acc / M).to(loss_ptr.dtype.element_ty))


def tr_zloss_ce_bwd(logits, targets):
    x = logits.contiguous()
    aux = targets.contiguous()
    M, V = x.shape
    rows = torch.empty((M,), device=x.device, dtype=torch.float32)
    grad = torch.empty_like(x)
    loss = torch.empty((), device=x.device, dtype=x.dtype)
    _loss_rows_kernel[(M,)](
        x, aux, rows, grad, M, V, KIND=2, DISTILL=False,
        BLOCK=1024, num_warps=8)
    _mean_rows_kernel[(1,)](rows, loss, M, BLOCK=1024, num_warps=8)
    return loss, grad
