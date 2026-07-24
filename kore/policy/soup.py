"""Stage-4 base-ward model soup (WiSE-FT interpolation).

    theta_final = (1 - alpha) * theta_base_instruct + alpha * theta_kore

Interpolating the RL specialist back toward the instruct base recovers general
chat/code/reasoning at zero inference cost while keeping most kernel gains. We
sweep alpha with an explicit alpha-zero safety point and pick the largest nonzero
kernel improvement subject to no general-metric regression. If no nonzero point
passes, promotion aborts. Pure tensor math remains unit-testable on CPU.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable, Optional

from kore.obs import get_logger

log = get_logger("policy.soup")


class SoupError(RuntimeError):
    """The interpolation or its promotion contract is unsafe."""


class SoupPromotionError(SoupError):
    """No non-base soup candidate satisfied the promotion gate."""

    def __init__(self, message: str, sweep: Optional[list[dict]] = None):
        self.sweep = list(sweep or [])
        super().__init__(message)


def _alpha(value: float) -> float:
    try:
        alpha = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SoupError(f"alpha must be numeric, got {value!r}") from exc
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise SoupError(f"alpha must be finite and in [0, 1], got {value!r}")
    return alpha


def validate_state_dict_compatibility(base_sd: dict, kore_sd: dict) -> dict:
    """Require exact keys, tensor kinds, and shapes before interpolation."""
    if not isinstance(base_sd, dict) or not isinstance(kore_sd, dict) or not base_sd or not kore_sd:
        raise SoupError("state dicts must be non-empty dictionaries")
    base_keys, kore_keys = set(base_sd), set(kore_sd)
    if base_keys != kore_keys:
        missing = sorted(base_keys - kore_keys)
        extra = sorted(kore_keys - base_keys)
        raise SoupError(
            f"state-dict key mismatch (missing_from_kore={missing[:8]}, "
            f"extra_in_kore={extra[:8]})"
        )
    n_float = 0
    for key in sorted(base_keys):
        bv, kv = base_sd[key], kore_sd[key]
        if not hasattr(bv, "shape") or not hasattr(kv, "shape"):
            raise SoupError(f"state-dict entry {key!r} is not tensor-like")
        if tuple(bv.shape) != tuple(kv.shape):
            raise SoupError(
                f"state-dict shape mismatch for {key!r}: "
                f"{tuple(bv.shape)} != {tuple(kv.shape)}"
            )
        b_float = bool(getattr(bv, "is_floating_point", lambda: False)())
        k_float = bool(getattr(kv, "is_floating_point", lambda: False)())
        if b_float != k_float:
            raise SoupError(f"state-dict tensor-kind mismatch for {key!r}")
        n_float += int(b_float)
    return {"n_keys": len(base_keys), "n_float": n_float}


def interpolate_state_dicts(base_sd: dict, kore_sd: dict, alpha: float) -> dict:
    """FP32 elementwise interpolation after exact compatibility validation.

    This pure helper returns a full dictionary for tests/small models.  Production
    materialization uses :func:`_stream_interpolate_into` to keep only one
    parameter-sized FP32 temporary alive at a time.
    """
    import torch

    alpha = _alpha(alpha)
    validate_state_dict_compatibility(base_sd, kore_sd)
    out = {}
    for k in sorted(kore_sd):
        bv, kv = base_sd[k], kore_sd[k]
        if kv.is_floating_point():
            if alpha == 0.0:
                out[k] = bv.detach().to(device=kv.device, dtype=kv.dtype).clone()
            elif alpha == 1.0:
                out[k] = kv.detach().clone()
            else:
                mixed = bv.detach().to(
                    device=kv.device, dtype=torch.float32, copy=True,
                )
                mixed.mul_(1.0 - alpha)
                mixed.add_(kv.detach(), alpha=alpha)
                out[k] = mixed.to(dtype=kv.dtype)
        else:
            # Alpha zero is a literal base-model safety point, including buffers.
            src = bv if alpha == 0.0 else kv
            out[k] = src.detach().to(device=kv.device).clone()
    return out


def _finite_metric(scores: dict, key: str) -> float:
    if not isinstance(scores, dict) or not scores:
        raise SoupPromotionError("soup evaluation returned empty metrics")
    try:
        value = float(scores[key])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise SoupPromotionError(f"soup metric {key!r} is missing or non-numeric") from exc
    if not math.isfinite(value):
        raise SoupPromotionError(f"soup metric {key!r} is non-finite: {value!r}")
    return value


def _sweep_alphas(
    alphas,
    evaluate_alpha: Callable[[float], dict],
    *,
    kernel_key: str,
    general_keys: list[str],
    base_scores: dict,
    epsilon: float,
) -> dict:
    """Shared fail-closed alpha selection for in-memory and materialized sweeps."""
    from kore.eval.gates import StageGate, retention_gate

    requested = [_alpha(a) for a in alphas]
    ordered = [0.0] + [a for a in requested if a != 0.0]
    ordered = list(dict.fromkeys(ordered))
    if not general_keys:
        raise SoupPromotionError("soup sweep requires non-empty general retention keys")
    log.info(
        "soup: alpha sweep starting", alphas=ordered, kernel_key=kernel_key,
        general_keys=list(general_keys), epsilon=epsilon,
    )
    results: list[dict] = []

    safety_scores = evaluate_alpha(0.0)
    safety_kernel = _finite_metric(safety_scores, kernel_key)
    safety_base = {g: base_scores.get(g) for g in general_keys}
    safety_general = {g: safety_scores.get(g) for g in general_keys}
    safety_gate = retention_gate(safety_base, safety_general, epsilon=epsilon)
    safety = {
        "alpha": 0.0,
        "scores": safety_scores,
        "kernel": safety_kernel,
        "passed": bool(safety_gate.passed),
        "safety_only": True,
        "gate": {
            "passed": safety_gate.passed,
            "regressions": safety_gate.regressions,
            "detail": safety_gate.detail,
        },
    }
    results.append(safety)
    log.event(
        "soup_alpha", alpha=0.0, kernel=safety_kernel,
        general=safety_general, passed=safety_gate.passed, safety_only=True,
    )
    if not safety_gate.passed:
        raise SoupPromotionError("alpha=0 base safety evaluation failed retention", results)

    before = {kernel_key: safety_kernel, **safety_base}
    gate = StageGate(epsilon=epsilon, require_all_kernel=True)
    for alpha in ordered[1:]:
        scores = evaluate_alpha(alpha)
        kernel = _finite_metric(scores, kernel_key)
        after = {kernel_key: kernel, **{g: scores.get(g) for g in general_keys}}
        verdict = gate.evaluate(
            before, after, kernel_keys=[kernel_key], general_keys=general_keys,
        )
        row = {
            "alpha": alpha,
            "scores": scores,
            "kernel": kernel,
            "passed": bool(verdict.passed),
            "safety_only": False,
            "gate": {
                "passed": verdict.passed,
                "regressions": verdict.regressions,
                "improvements": verdict.improvements,
                "detail": verdict.detail,
            },
        }
        results.append(row)
        log.event(
            "soup_alpha", alpha=alpha, kernel=kernel,
            general={g: scores.get(g) for g in general_keys},
            passed=verdict.passed, safety_only=False,
        )

    promotable = [r for r in results if r["alpha"] > 0.0 and r["passed"]]
    if not promotable:
        raise SoupPromotionError(
            "no nonzero alpha improved the kernel while passing full retention; "
            "soup promotion aborted",
            results,
        )
    best = max(promotable, key=lambda r: r["kernel"])
    log.metric(
        "soup_best", best_alpha=best["alpha"], gate_satisfied=True,
        kernel=best["kernel"],
    )
    return {
        "best_alpha": best["alpha"],
        "best": best,
        "sweep": results,
        "gate_satisfied": True,
        "alpha_zero_included": True,
        "nonzero_promoted": True,
    }


def soup_sweep(base_sd: dict, kore_sd: dict, alphas, eval_fn: Callable[[dict], dict],
               *, kernel_key: str, general_keys: list[str], base_scores: dict,
               epsilon: float = 0.005) -> dict:
    """Sweep in-memory state dicts; never promote the alpha-zero safety point."""
    validate_state_dict_compatibility(base_sd, kore_sd)
    return _sweep_alphas(
        alphas,
        lambda alpha: eval_fn(interpolate_state_dicts(base_sd, kore_sd, alpha)),
        kernel_key=kernel_key,
        general_keys=general_keys,
        base_scores=base_scores,
        epsilon=epsilon,
    )


def soup_sweep_materialized(
    alphas,
    eval_alpha_fn: Callable[[float], dict],
    *,
    kernel_key: str,
    general_keys: list[str],
    base_scores: dict,
    epsilon: float = 0.005,
) -> dict:
    """Sweep an alpha callback, allowing checkpoint-at-a-time evaluation."""
    return _sweep_alphas(
        alphas, eval_alpha_fn, kernel_key=kernel_key, general_keys=general_keys,
        base_scores=base_scores, epsilon=epsilon,
    )


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

    load_kw = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if _is_lora_adapter_dir(kore_checkpoint):
        from peft import PeftModel

        base_for_adapter = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kw)
        peft_model = PeftModel.from_pretrained(base_for_adapter, kore_checkpoint)
        return peft_model.merge_and_unload()
    return AutoModelForCausalLM.from_pretrained(kore_checkpoint, **load_kw)


def _model_signature(model) -> dict:
    from kore.campaign_lineage import architecture_signature

    config = model.config.to_dict() if hasattr(model.config, "to_dict") else vars(model.config)
    return {
        "config_class": type(model.config).__name__,
        "model_class": type(model).__name__,
        "architecture": architecture_signature(config),
    }


def _validate_model_architecture(base, kore) -> None:
    base_sig, kore_sig = _model_signature(base), _model_signature(kore)
    # A PEFT merge returns the same underlying causal-LM architecture but may use a
    # transient wrapper class, so config class + architectural fields are the hard
    # contract; exact tensor keys/shapes are checked immediately afterward.
    comparable_base = {
        "config_class": base_sig["config_class"],
        "architecture": base_sig["architecture"],
    }
    comparable_kore = {
        "config_class": kore_sig["config_class"],
        "architecture": kore_sig["architecture"],
    }
    if comparable_base != comparable_kore:
        raise SoupError(
            "model architecture mismatch; refusing to interpolate "
            f"(base={comparable_base}, kore={comparable_kore})"
        )


def _tokenizer_signature(tokenizer) -> dict:
    vocab_hash = hashlib.sha256()
    vocab = tokenizer.get_vocab()
    for token, idx in sorted(vocab.items(), key=lambda kv: kv[0]):
        vocab_hash.update(token.encode("utf-8", "surrogatepass"))
        vocab_hash.update(b"\0")
        vocab_hash.update(str(idx).encode())
        vocab_hash.update(b"\0")
    chat = getattr(tokenizer, "chat_template", None) or ""
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": len(vocab),
        "vocab_digest": vocab_hash.hexdigest(),
        "chat_template_digest": hashlib.sha256(chat.encode("utf-8")).hexdigest(),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def _stream_interpolate_into(base_sd: dict, kore_sd: dict, alpha: float) -> None:
    """Write the soup into ``kore_sd`` with one parameter-sized FP32 work buffer."""
    import torch

    alpha = _alpha(alpha)
    validate_state_dict_compatibility(base_sd, kore_sd)
    with torch.no_grad():
        for key in sorted(kore_sd):
            base_tensor, target = base_sd[key], kore_sd[key]
            if not target.is_floating_point():
                if alpha == 0.0:
                    target.copy_(base_tensor.to(device=target.device, dtype=target.dtype))
                continue
            if alpha == 1.0:
                continue
            if alpha == 0.0:
                target.copy_(base_tensor.to(device=target.device, dtype=target.dtype))
                continue
            mixed = base_tensor.detach().to(
                device=target.device, dtype=torch.float32, copy=True,
            )
            mixed.mul_(1.0 - alpha)
            mixed.add_(target.detach().to(dtype=torch.float32), alpha=alpha)
            target.copy_(mixed.to(dtype=target.dtype))
            del mixed


def build_soup(base_model_id: str, kore_checkpoint: str, alpha: float, output_dir: str,
               ref_base_sd: Optional[dict] = None) -> str:
    """Materialize a souped HF model at ``output_dir`` for a chosen alpha.

    ``kore_checkpoint`` may be a full HF checkpoint OR a LoRA adapter dir; in the
    latter case the adapter is merged onto ``base_model_id`` before interpolation
    so both operands share identical parameter keys/shapes.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    alpha = _alpha(alpha)
    log.info("soup: materializing", alpha=alpha, base=base_model_id,
             kore=kore_checkpoint, out=output_dir)
    dtype = torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    kore = _load_kore_model(base_model_id, kore_checkpoint, dtype)
    _validate_model_architecture(base, kore)
    base_sd = ref_base_sd if ref_base_sd is not None else base.state_dict()
    kore_sd = kore.state_dict()
    _stream_interpolate_into(base_sd, kore_sd, alpha)
    kore.save_pretrained(output_dir)

    tok_src = base_model_id if _is_lora_adapter_dir(kore_checkpoint) else kore_checkpoint
    base_tok = AutoTokenizer.from_pretrained(base_model_id)
    kore_tok = AutoTokenizer.from_pretrained(tok_src)
    base_tok_sig, kore_tok_sig = _tokenizer_signature(base_tok), _tokenizer_signature(kore_tok)
    if base_tok_sig != kore_tok_sig:
        raise SoupError(
            "tokenizer mismatch between base and specialist; refusing to promote "
            f"(base={base_tok_sig}, kore={kore_tok_sig})"
        )
    base_tok.save_pretrained(output_dir)
    log.info("soup: materialized best-alpha model", alpha=alpha, out=output_dir)
    return output_dir


__all__ = [
    "SoupError",
    "SoupPromotionError",
    "build_soup",
    "interpolate_state_dicts",
    "soup_sweep",
    "soup_sweep_materialized",
    "validate_state_dict_compatibility",
]
