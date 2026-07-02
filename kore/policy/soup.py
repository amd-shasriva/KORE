"""Stage-4 base-ward model soup (WiSE-FT interpolation).

    theta_final = (1 - alpha) * theta_base_instruct + alpha * theta_kore

Interpolating the RL specialist back toward the instruct base recovers general
chat/code/reasoning at zero inference cost while keeping most kernel gains. We
sweep alpha and pick the largest kernel improvement subject to no general-metric
regression (the retention gate). Pure tensor math so it is unit-testable on CPU.
"""

from __future__ import annotations

from typing import Callable, Optional


def interpolate_state_dicts(base_sd: dict, kore_sd: dict, alpha: float) -> dict:
    """Elementwise (1-alpha)*base + alpha*kore over shared float tensors.

    Non-float or non-shared keys are taken from ``kore_sd`` unchanged (e.g. int
    buffers, or keys only present after fine-tuning).
    """
    out = {}
    for k, kv in kore_sd.items():
        bv = base_sd.get(k)
        if bv is not None and hasattr(kv, "dtype") and getattr(kv, "is_floating_point", lambda: False)() \
                and bv.shape == kv.shape:
            out[k] = (1.0 - alpha) * bv.to(kv.dtype) + alpha * kv
        else:
            out[k] = kv
    return out


def soup_sweep(base_sd: dict, kore_sd: dict, alphas, eval_fn: Callable[[dict], dict],
               *, kernel_key: str, general_keys: list[str], base_scores: dict,
               epsilon: float = 0.005) -> dict:
    """Sweep alpha; return the best interpolation subject to the retention gate.

    ``eval_fn(state_dict)->{metric: value}``. Accept an alpha only if no
    ``general_keys`` metric drops > epsilon below ``base_scores``; among accepted,
    maximize ``kernel_key``. Falls back to the highest-kernel alpha if none pass.
    """
    results = []
    for a in alphas:
        sd = interpolate_state_dicts(base_sd, kore_sd, a)
        scores = eval_fn(sd)
        regressed = any(scores.get(g, 0.0) < base_scores.get(g, 0.0) - epsilon for g in general_keys)
        results.append({"alpha": a, "scores": scores, "passed": not regressed,
                        "kernel": scores.get(kernel_key, 0.0)})
    passed = [r for r in results if r["passed"]]
    pool = passed or results
    best = max(pool, key=lambda r: r["kernel"])
    return {"best_alpha": best["alpha"], "best": best, "sweep": results,
            "gate_satisfied": bool(passed)}


def build_soup(base_model_id: str, kore_checkpoint: str, alpha: float, output_dir: str,
               ref_base_sd: Optional[dict] = None) -> str:
    """Materialize a souped HF model at ``output_dir`` for a chosen alpha."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
    kore = AutoModelForCausalLM.from_pretrained(kore_checkpoint, torch_dtype=torch.bfloat16)
    souped = interpolate_state_dicts(base.state_dict(), kore.state_dict(), alpha)
    kore.load_state_dict(souped)
    kore.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(kore_checkpoint).save_pretrained(output_dir)
    return output_dir
