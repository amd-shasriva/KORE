"""Instruction-residual (chat-vector) transfer.

    theta_out = theta_target + scale * (theta_instruct - theta_base)

Continued pretraining has to run on the BASE model: doing it on the instruct
model destroyed instruction-following outright (docs/EVAL_RESULTS.md). That
leaves midtrain holding the Triton domain knowledge but unable to follow an
instruction, and an SFT budget of ~56k rows is far too small to teach chat from
scratch -- Tulu 3 needed 939k samples to do that from a Llama base.

So instead of paying for it, we transplant it. The difference between a vendor's
instruct and base checkpoints is the entire post-training run expressed as a
delta, and adding that delta to a continually-pretrained sibling transfers the
behaviour without training (Huang et al., ACL 2024, "Chat Vector"). Rearranged,

    theta_out = theta_instruct + (theta_target - theta_base)

reads more honestly: the vendor's instruct model carrying OUR domain delta.

This is only sound within one model family, so callers must validate that the
three checkpoints agree on keys, shapes and tensor kinds; ``apply_residual``
refuses otherwise. Arithmetic is done in FP32 and cast back once, so a bf16
round-trip does not accumulate. Pure tensor math stays unit-testable on CPU.
"""

from __future__ import annotations

import math
from typing import Optional

from kore.obs import get_logger

log = get_logger("policy.residual")


class ResidualError(RuntimeError):
    """The residual or its inputs violate the transfer contract."""


def _scale(value: float) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResidualError(f"scale must be numeric, got {value!r}") from exc
    if not math.isfinite(scale):
        raise ResidualError(f"scale must be finite, got {value!r}")
    return scale


def validate_triplet(base_sd: dict, instruct_sd: dict, target_sd: dict) -> dict:
    """Require identical keys, shapes and tensor kinds across all three models.

    Task arithmetic is only meaningful when the three checkpoints are the same
    architecture with the same vocabulary, so a mismatch is an error rather than
    something to paper over by skipping keys.
    """
    for name, sd in (("base", base_sd), ("instruct", instruct_sd), ("target", target_sd)):
        if not isinstance(sd, dict) or not sd:
            raise ResidualError(f"{name} state dict must be a non-empty dictionary")
    keys = set(base_sd)
    for name, sd in (("instruct", instruct_sd), ("target", target_sd)):
        if set(sd) != keys:
            missing = sorted(keys - set(sd))
            extra = sorted(set(sd) - keys)
            raise ResidualError(
                f"{name} key mismatch (missing={missing[:8]}, extra={extra[:8]})"
            )
    n_float = 0
    for key in sorted(keys):
        tensors = (base_sd[key], instruct_sd[key], target_sd[key])
        for t in tensors:
            if not hasattr(t, "shape"):
                raise ResidualError(f"entry {key!r} is not tensor-like")
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) != 1:
            raise ResidualError(f"shape mismatch for {key!r}: {sorted(shapes)}")
        kinds = {bool(getattr(t, "is_floating_point", lambda: False)()) for t in tensors}
        if len(kinds) != 1:
            raise ResidualError(f"tensor-kind mismatch for {key!r}")
        n_float += int(kinds.pop())
    return {"n_keys": len(keys), "n_float": n_float}


def apply_residual(
    base_sd: dict,
    instruct_sd: dict,
    target_sd: dict,
    scale: float = 1.0,
    skip_keys: Optional[set] = None,
) -> dict:
    """Return ``target + scale * (instruct - base)``, computed in FP32.

    Integer tensors and buffers are passed through from the target untouched:
    a delta is only meaningful for learned floating-point weights.
    """
    import torch

    scale = _scale(scale)
    validate_triplet(base_sd, instruct_sd, target_sd)
    skip = set(skip_keys or ())
    out = {}
    for key in sorted(target_sd):
        tv = target_sd[key]
        if key in skip or not tv.is_floating_point():
            out[key] = tv.detach().clone()
            continue
        acc = tv.detach().to(dtype=torch.float32, copy=True)
        delta = instruct_sd[key].detach().to(dtype=torch.float32)
        delta = delta - base_sd[key].detach().to(dtype=torch.float32)
        acc.add_(delta, alpha=scale)
        out[key] = acc.to(dtype=tv.dtype)
    return out


def residual_report(base_sd: dict, instruct_sd: dict) -> dict:
    """Summarise the delta's magnitude, to catch a mispaired checkpoint.

    A correct base/instruct pair yields a delta that is small but clearly
    nonzero. An all-zero delta means the same checkpoint was passed twice; a
    delta comparable in norm to the weights means these are not siblings.
    """
    import torch

    n_zero = 0
    ratios = []
    for key in sorted(instruct_sd):
        if not instruct_sd[key].is_floating_point():
            continue
        b = base_sd[key].detach().to(dtype=torch.float32)
        i = instruct_sd[key].detach().to(dtype=torch.float32)
        d = i - b
        dn, bn = float(d.norm()), float(b.norm())
        if dn == 0.0:
            n_zero += 1
        if bn > 0:
            ratios.append(dn / bn)
    ratios.sort()
    if not ratios:
        raise ResidualError("no floating-point tensors to compare")
    return {
        "n_float": len(ratios),
        "n_zero_delta": n_zero,
        "rel_delta_median": ratios[len(ratios) // 2],
        "rel_delta_max": ratios[-1],
        "rel_delta_min": ratios[0],
    }


def verify_identity(base_sd: dict, instruct_sd: dict) -> dict:
    """Applying the residual to the base must reproduce the instruct model.

    This is the strongest cheap check available: it exercises the exact code
    path used for the real transfer, on the one input whose answer is known in
    advance, so a sign error or dtype bug cannot slip through silently.
    """
    import torch

    rebuilt = apply_residual(base_sd, instruct_sd, base_sd, scale=1.0)
    worst, n_exact, n_float = 0.0, 0, 0
    for key in sorted(instruct_sd):
        want = instruct_sd[key]
        if not want.is_floating_point():
            continue
        n_float += 1
        got = rebuilt[key]
        if torch.equal(got, want):
            n_exact += 1
        else:
            diff = (got.to(torch.float32) - want.to(torch.float32)).abs().max()
            worst = max(worst, float(diff))
    return {
        "n_float": n_float,
        "n_exact": n_exact,
        "exact_fraction": n_exact / max(n_float, 1),
        "max_abs_diff": worst,
    }
