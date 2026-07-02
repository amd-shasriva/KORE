"""Stage-4 base-ward model soup (WiSE-FT interpolation).

    theta_final = (1 - alpha) * theta_base_instruct + alpha * theta_kore

Interpolating the RL specialist back toward the instruct base recovers general
chat/code/reasoning at zero inference cost while keeping most kernel gains. We
sweep alpha and pick the largest kernel improvement subject to no general-metric
regression (the retention gate). Pure tensor math so it is unit-testable on CPU.
"""

from __future__ import annotations

from typing import Callable, Optional

from kore.obs import get_logger

log = get_logger("policy.soup")


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
    log.info("soup: alpha sweep starting", alphas=list(alphas), kernel_key=kernel_key,
             general_keys=list(general_keys), epsilon=epsilon)
    results = []
    for a in alphas:
        sd = interpolate_state_dicts(base_sd, kore_sd, a)
        scores = eval_fn(sd)
        regressed = any(scores.get(g, 0.0) < base_scores.get(g, 0.0) - epsilon for g in general_keys)
        results.append({"alpha": a, "scores": scores, "passed": not regressed,
                        "kernel": scores.get(kernel_key, 0.0)})
        log.event("soup_alpha", alpha=a, kernel=scores.get(kernel_key, 0.0),
                  general={g: scores.get(g) for g in general_keys}, passed=not regressed)
    passed = [r for r in results if r["passed"]]
    pool = passed or results
    best = max(pool, key=lambda r: r["kernel"])
    log.metric("soup_best", best_alpha=best["alpha"], gate_satisfied=bool(passed),
               kernel=best["kernel"])
    return {"best_alpha": best["alpha"], "best": best, "sweep": results,
            "gate_satisfied": bool(passed)}


def _is_lora_adapter_dir(path: str) -> bool:
    """A PEFT adapter dir is identified by its ``adapter_config.json``."""
    import os

    return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


def _load_kore_model(base_model_id: str, kore_checkpoint: str, dtype):
    """Load the KORE specialist as a full model.

    If ``kore_checkpoint`` is a LoRA adapter dir (rather than a full checkpoint),
    load ``base_model_id`` and merge the adapter into it first so the state dict
    has the same parameter keys/shapes as the base for interpolation.
    """
    from transformers import AutoModelForCausalLM

    if _is_lora_adapter_dir(kore_checkpoint):
        from peft import PeftModel

        base_for_adapter = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype)
        peft_model = PeftModel.from_pretrained(base_for_adapter, kore_checkpoint)
        return peft_model.merge_and_unload()
    return AutoModelForCausalLM.from_pretrained(kore_checkpoint, torch_dtype=dtype)


def build_soup(base_model_id: str, kore_checkpoint: str, alpha: float, output_dir: str,
               ref_base_sd: Optional[dict] = None) -> str:
    """Materialize a souped HF model at ``output_dir`` for a chosen alpha.

    ``kore_checkpoint`` may be a full HF checkpoint OR a LoRA adapter dir; in the
    latter case the adapter is merged onto ``base_model_id`` before interpolation
    so both operands share identical parameter keys/shapes.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("soup: materializing", alpha=alpha, base=base_model_id,
             kore=kore_checkpoint, out=output_dir)
    dtype = torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype)
    kore = _load_kore_model(base_model_id, kore_checkpoint, dtype)
    souped = interpolate_state_dicts(base.state_dict(), kore.state_dict(), alpha)
    kore.load_state_dict(souped)
    kore.save_pretrained(output_dir)

    tok_src = base_model_id if _is_lora_adapter_dir(kore_checkpoint) else kore_checkpoint
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(output_dir)
    log.info("soup: materialized best-alpha model", alpha=alpha, out=output_dir)
    return output_dir
