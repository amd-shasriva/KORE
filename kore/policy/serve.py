"""vLLM-ROCm serving wrapper for the KORE policy (import-guarded).

``VLLMPolicy`` is a thin façade over vLLM's offline ``LLM`` engine used by the
GRPO rollout to sample trajectories fast. vLLM is imported lazily so this module
loads on a CPU box without vLLM installed.

ROCm / gfx942 environment notes (set these before constructing the engine):
  - ``RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`` - when vLLM runs under
    Ray with tensor parallelism, Ray otherwise rewrites ``HIP_VISIBLE_DEVICES``
    per worker and the ROCm runtime loses the intended device mask. Setting this
    keeps the process-level device visibility that vLLM expects.
  - ``VLLM_ROCM_USE_AITER=1`` - enable AMD AITER fused kernels (attention/MoE)
    for MI3xx, giving faster decode on gfx942.
  - ``HIP_VISIBLE_DEVICES`` - the sole authoritative visibility mask. Setting
    both HIP and ROCR masks can remap an already-remapped ordinal and is rejected.
"""

from __future__ import annotations

import gc
import os
import sys
from typing import Any, Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class GenerationProtocol(Protocol):
    """Validated single-example generation interface used by KORE data loops."""

    @property
    def closed(self) -> bool: ...

    def __call__(self, prompt_or_messages: Any, **kwargs: Any) -> str: ...

    def generate(self, messages: Any, **kwargs: Any) -> str: ...

    def close(self) -> None: ...

    def release(self) -> None: ...


class GenerationClient:
    """Callable/chat-compatible generation client with explicit ownership.

    ``close`` and ``release`` are idempotent aliases.  Calls after release fail
    loudly instead of accidentally using a partially torn-down GPU backend.
    """

    def __init__(
        self,
        generate_fn: Callable[..., str],
        close_fn: Optional[Callable[[], None]] = None,
        *,
        model_id: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")
        if close_fn is not None and not callable(close_fn):
            raise TypeError("close_fn must be callable when provided")
        self._generate_fn: Optional[Callable[..., str]] = generate_fn
        self._close_fn = close_fn
        self._closed = False
        self.model_id = model_id
        self.backend = backend

    @property
    def closed(self) -> bool:
        return self._closed

    def generate(self, messages: Any, **kwargs: Any) -> str:
        if self._closed or self._generate_fn is None:
            raise RuntimeError("generation client is closed")
        result = self._generate_fn(messages, **kwargs)
        if not isinstance(result, str):
            raise TypeError(
                f"generation backend returned {type(result).__name__}, expected str"
            )
        return result

    def __call__(self, prompt_or_messages: Any, **kwargs: Any) -> str:
        return self.generate(prompt_or_messages, **kwargs)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_fn = self._close_fn
        try:
            if close_fn is not None:
                close_fn()
        finally:
            # Drop bound methods/closures even if backend shutdown raises.
            self._generate_fn = None
            self._close_fn = None

    def release(self) -> None:
        self.close()

    def __enter__(self) -> "GenerationClient":
        if self._closed:
            raise RuntimeError("cannot enter a closed generation client")
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


def as_generation_client(policy: Any) -> GenerationClient:
    """Validate/adapt a callable or ``.generate`` object to one client type."""

    if isinstance(policy, GenerationClient):
        return policy
    generate_fn = getattr(policy, "generate", None)
    if not callable(generate_fn):
        generate_fn = policy if callable(policy) else None
    if not callable(generate_fn):
        raise TypeError(
            "policy must be callable or expose generate(messages, **kwargs)"
        )
    close_fn = getattr(policy, "close", None)
    if not callable(close_fn):
        close_fn = getattr(policy, "release", None)
    if not callable(close_fn):
        close_fn = None
    return GenerationClient(
        generate_fn,
        close_fn,
        model_id=getattr(policy, "model_id", None)
        or getattr(policy, "model", None),
        backend=getattr(policy, "backend", None),
    )

# Documented, applied via ``configure_rocm_env``.
ROCM_ENV_DEFAULTS = {
    "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
    "VLLM_ROCM_USE_AITER": "1",
}
ROCM_VISIBILITY_ENV = "HIP_VISIBLE_DEVICES"
_ROCM_SECONDARY_VISIBILITY_ENV = "ROCR_VISIBLE_DEVICES"


class DeviceVisibilityError(ValueError):
    """Raised when ROCm visibility cannot be represented by one exact mask."""


def _validated_gpu_ordinals(values, *, field: str) -> tuple[int, ...]:
    if isinstance(values, str):
        pieces = [piece.strip() for piece in values.split(",")]
        if not pieces or any(not piece or not piece.isdigit() for piece in pieces):
            raise DeviceVisibilityError(
                f"{field} must be a comma-separated list of non-negative ordinals"
            )
        ordinals = tuple(int(piece) for piece in pieces)
    else:
        if values is None:
            raise DeviceVisibilityError(f"{field} is missing")
        ordinals = tuple(values)
        if not ordinals:
            raise DeviceVisibilityError(f"{field} cannot be an empty mask")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ordinals
        ):
            raise DeviceVisibilityError(
                f"{field} must contain only non-negative integer ordinals"
            )
    if len(set(ordinals)) != len(ordinals):
        raise DeviceVisibilityError(f"{field} contains duplicate ordinals")
    return ordinals


def configure_rocm_env(gpu_ids: Optional[list[int]] = None) -> dict:
    """Apply one validated, non-composed HIP visibility policy.

    An inherited ROCR-only mask is normalized to HIP before GPU initialization.
    Two masks, malformed masks, or a requested mask that differs from an
    inherited mask are rejected instead of intersecting/remapping them.
    """
    for k, v in ROCM_ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)

    hip_raw = os.environ.get(ROCM_VISIBILITY_ENV)
    rocr_raw = os.environ.get(_ROCM_SECONDARY_VISIBILITY_ENV)
    if hip_raw is not None and rocr_raw is not None:
        raise DeviceVisibilityError(
            "HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES are both set; "
            "double masks are forbidden"
        )

    requested = (
        _validated_gpu_ordinals(gpu_ids, field="gpu_ids")
        if gpu_ids is not None
        else None
    )
    inherited_raw = hip_raw if hip_raw is not None else rocr_raw
    inherited = (
        _validated_gpu_ordinals(
            inherited_raw,
            field=(
                ROCM_VISIBILITY_ENV
                if hip_raw is not None
                else _ROCM_SECONDARY_VISIBILITY_ENV
            ),
        )
        if inherited_raw is not None
        else None
    )
    if requested is not None and inherited is not None and requested != inherited:
        raise DeviceVisibilityError(
            "requested GPU ordinals differ from the inherited visibility mask; "
            "refusing to compose masks"
        )
    effective = requested or inherited
    os.environ.pop(_ROCM_SECONDARY_VISIBILITY_ENV, None)
    if effective is not None:
        os.environ[ROCM_VISIBILITY_ENV] = ",".join(str(i) for i in effective)

    keys = list(ROCM_ENV_DEFAULTS) + [ROCM_VISIBILITY_ENV]
    return {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}


class VLLMPolicy:
    """vLLM-backed policy for fast rollout generation on ROCm.

    Heavy imports (vLLM/torch) happen in ``__init__`` so the class *definition*
    is import-safe. Construct only on a GPU box with vLLM-ROCm installed.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_ids: Optional[list[int]] = None,
        dtype: str = "bfloat16",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        seed: int = 0,
        revision: Optional[str] = None,
        **engine_kwargs,
    ):
        from kore.policy.model_spec import validate_pinned_revision

        revision = validate_pinned_revision(revision)
        configure_rocm_env(gpu_ids)

        from vllm import LLM  # guarded heavy import

        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self._closed = False
        load_kwargs = dict(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            **engine_kwargs,
        )
        load_kwargs["revision"] = revision
        self._llm = LLM(**load_kwargs)

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self):
        if self._closed or self._llm is None:
            raise RuntimeError("vLLM policy is closed")
        return self._llm

    def generate(
        self,
        prompts: list[str],
        temperature: float = 1.0,
        max_tokens: int = 8192,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        n: int = 1,
    ) -> list[str]:
        """Generate one completion per prompt (or ``n`` if requested).

        Returns a flat list of generated strings (length ``len(prompts) * n``).
        """
        from vllm import SamplingParams  # guarded heavy import

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            n=n,
        )
        outputs = self._require_open().generate(prompts, params)
        texts: list[str] = []
        for out in outputs:
            for comp in out.outputs:
                texts.append(comp.text)
        return texts

    def chat(
        self,
        messages_batch: list[list[dict]],
        temperature: float = 1.0,
        max_tokens: int = 8192,
        top_p: float = 1.0,
        enable_thinking: Optional[bool] = None,
    ) -> list[str]:
        """Generate from a batch of chat-message lists via vLLM's chat API.

        ``enable_thinking=False`` disables the Qwen3-style ``<think>`` trace at the
        chat-template level (used by eval/retention so answers are direct); left as
        ``None`` it uses the template default. Silently ignored by templates that
        don't support the kwarg (older vLLM / non-Qwen)."""
        from vllm import SamplingParams

        params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        kw = {}
        if enable_thinking is not None:
            kw["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        try:
            outputs = self._require_open().chat(messages_batch, params, **kw)
        except TypeError:  # older vLLM without chat_template_kwargs
            outputs = self._require_open().chat(messages_batch, params)
        return [out.outputs[0].text for out in outputs]

    def close(self) -> None:
        """Shut down vLLM workers and release backend/GPU references."""

        if self._closed:
            return
        self._closed = True
        llm, self._llm = self._llm, None
        if llm is not None:
            targets = (llm, getattr(llm, "llm_engine", None))
            for target in targets:
                if target is None:
                    continue
                shutdown = getattr(target, "shutdown", None)
                if not callable(shutdown):
                    shutdown = getattr(target, "close", None)
                if callable(shutdown):
                    shutdown()
                    break
        gc.collect()
        torch = sys.modules.get("torch")
        cuda = getattr(torch, "cuda", None) if torch is not None else None
        if cuda is not None:
            try:
                cuda.empty_cache()
            except Exception:
                pass

    release = close

    def __enter__(self) -> "VLLMPolicy":
        self._require_open()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


import re as _re

_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)
_THINK_OPEN_RE = _re.compile(r"<think>.*$", _re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning spans (and a dangling unclosed
    ``<think>`` when the token budget cut it off) so downstream answer parsing --
    MMLU letter, code block, JSON tool call -- never sees the trace. Retention/eval
    scoring must read the ANSWER, not the reasoning (audit R2 soup-eval C1)."""
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


def _is_lora_adapter_dir(path: str) -> bool:
    """A PEFT LoRA checkpoint is a dir containing ``adapter_config.json`` (mirrors
    ``soup._is_lora_adapter_dir``)."""
    import os
    return isinstance(path, str) and os.path.isdir(path) and \
        os.path.exists(os.path.join(path, "adapter_config.json"))


def _apply_chat_no_think(tok, messages):
    """``apply_chat_template`` with thinking DISABLED for hybrid-reasoning models
    (Qwen3). Retention/eval answers must be DIRECT: with thinking on, Qwen3 spends
    the (tiny, e.g. MMLU=32) token budget on a ``<think>`` block and never emits the
    answer, so base and candidate both score ~random and the gate rubber-stamps
    catastrophic forgetting (audit R2 soup-eval C1). Falls back cleanly for
    tokenizers whose template doesn't accept ``enable_thinking``."""
    try:
        return tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False)
    except TypeError:  # template doesn't support the kwarg (non-Qwen) -> plain
        return tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt")


def load_generate(
    model_id: str,
    *,
    backend: str = "hf",
    tensor_parallel_size: int = 1,
    max_new_tokens: int = 1024,
    gpu_ids: Optional[list[int]] = None,
    revision: Optional[str] = None,
    base_revision: Optional[str] = None,
    model_spec=None,
    **kw,
) -> GenerationClient:
    """Load ``model_id`` and return an owned :class:`GenerationClient`.

    The client is both callable and exposes ``generate(messages, **kwargs)``.
    Call ``close()``/``release()`` (or use it as a context manager) before a
    training process claims the same GPUs.

    When ``model_spec`` is supplied, its local checkpoint, immutable revision,
    architecture, and safetensors compatibility have already been checked
    offline and are re-bound here *before* importing a GPU framework. Passing a
    bare ``revision`` still rejects floating refs but cannot replace a complete
    :class:`kore.policy.model_spec.ModelSpec` validation.

    backend:
      - ``"hf"``   : transformers ``AutoModelForCausalLM`` + ``apply_chat_template``.
      - ``"vllm"`` : :class:`VLLMPolicy` (fast rollout serving on ROCm).
    """
    from kore.policy.model_spec import ModelSpec, validate_pinned_revision

    if model_spec is not None:

        if not isinstance(model_spec, ModelSpec):
            raise TypeError("model_spec must be a resolved ModelSpec")
        model_spec.validate_for_load(model_id, revision=revision)
        revision = model_spec.revision
    else:
        revision = validate_pinned_revision(revision)

    if backend == "vllm":
        policy = VLLMPolicy(model_id, tensor_parallel_size=tensor_parallel_size,
                            gpu_ids=gpu_ids, revision=revision, **kw)

        def generate(prompt_or_messages, max_tokens: int = max_new_tokens,
                     temperature: float = 0.0) -> str:
            if isinstance(prompt_or_messages, str):
                out = policy.generate([prompt_or_messages], temperature=temperature,
                                      max_tokens=max_tokens)[0]
            else:
                # thinking OFF for eval/retention so the answer isn't a <think> trace
                out = policy.chat([prompt_or_messages], temperature=temperature,
                                  max_tokens=max_tokens, enable_thinking=False)[0]
            return _strip_think(out)

        return GenerationClient(
            generate, policy.close, model_id=model_id, backend="vllm"
        )

    if backend == "hf":
        # Pin device_map="auto" to the requested physical GPUs BEFORE the first HIP
        # init (from_pretrained below), so the retention gate stays on the free GPUs
        # of a shared node instead of grabbing every visible (incl. busy) GPU.
        configure_rocm_env(gpu_ids)
        import torch  # guarded heavy import
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # A LoRA checkpoint is an ADAPTER dir (adapter_config.json), not a full model:
        # AutoModelForCausalLM.from_pretrained on it loads only the tiny adapter and the
        # eval/retention gate would score garbage. Detect it, load the base named in the
        # adapter config, attach + merge the adapter (audit R2 soup-eval I1: eval path
        # must load adapters like soup._load_kore_model already does for the soup).
        revision_kw = {"revision": revision} if revision is not None else {}
        if _is_lora_adapter_dir(model_id):
            import json as _json
            import os as _os
            with open(_os.path.join(model_id, "adapter_config.json")) as handle:
                cfg = _json.load(handle)
            base_id = cfg.get("base_model_name_or_path")
            if not isinstance(base_id, str) or not base_id.strip():
                raise ValueError(
                    "adapter_config.json must identify its exact base model"
                )
            configured_base_revision = cfg.get("revision")
            if (
                base_revision is not None
                and configured_base_revision is not None
                and validate_pinned_revision(base_revision)
                != validate_pinned_revision(configured_base_revision)
            ):
                raise ValueError(
                    "explicit base_revision conflicts with adapter_config.json"
                )
            resolved_base_revision = validate_pinned_revision(
                base_revision or configured_base_revision
            )
            from peft import PeftModel
            tok = AutoTokenizer.from_pretrained(
                base_id, revision=resolved_base_revision
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_id, torch_dtype=torch.bfloat16, device_map="auto",
                revision=resolved_base_revision, **kw)
            model = PeftModel.from_pretrained(
                base, model_id, revision=revision
            ).merge_and_unload()
        else:
            if base_revision is not None:
                raise ValueError(
                    "base_revision is only valid when model_id is a LoRA adapter"
                )
            tok = AutoTokenizer.from_pretrained(model_id, **revision_kw)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto",
                **revision_kw, **kw)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model.eval()
        state = {"tokenizer": tok, "model": model}

        def generate(prompt_or_messages, max_tokens: int = max_new_tokens,
                     temperature: float = 0.0) -> str:
            tok = state.get("tokenizer")
            model = state.get("model")
            if tok is None or model is None:
                raise RuntimeError("HF generation backend is closed")
            messages = ([{"role": "user", "content": prompt_or_messages}]
                        if isinstance(prompt_or_messages, str) else prompt_or_messages)
            ids = _apply_chat_no_think(tok, messages).to(model.device)
            do_sample = bool(temperature and temperature > 0.0)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=max_tokens, do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=tok.pad_token_id)
            gen = out[0][ids.shape[1]:]
            return _strip_think(tok.decode(gen, skip_special_tokens=True))

        def close() -> None:
            state["model"] = None
            state["tokenizer"] = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return GenerationClient(
            generate, close, model_id=model_id, backend="hf"
        )

    raise ValueError(f"unknown backend {backend!r}; expected 'hf' or 'vllm'")


__all__ = [
    "DeviceVisibilityError",
    "GenerationClient",
    "GenerationProtocol",
    "VLLMPolicy",
    "ROCM_VISIBILITY_ENV",
    "as_generation_client",
    "configure_rocm_env",
    "load_generate",
]
