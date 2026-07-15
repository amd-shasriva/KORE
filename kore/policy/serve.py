"""vLLM-ROCm serving wrapper for the KORE policy (import-guarded).

``VLLMPolicy`` is a thin façade over vLLM's offline ``LLM`` engine used by the
GRPO rollout to sample trajectories fast. vLLM is imported lazily so this module
loads on a CPU box without vLLM installed.

ROCm / gfx942 environment notes (set these before constructing the engine):
  - ``RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`` — when vLLM runs under
    Ray with tensor parallelism, Ray otherwise rewrites ``HIP_VISIBLE_DEVICES``
    per worker and the ROCm runtime loses the intended device mask. Setting this
    keeps the process-level device visibility that vLLM expects.
  - ``VLLM_ROCM_USE_AITER=1`` — enable AMD AITER fused kernels (attention/MoE)
    for MI3xx, giving faster decode on gfx942.
  - ``HIP_VISIBLE_DEVICES`` / ``ROCR_VISIBLE_DEVICES`` — pin the specific GPUs.
"""

from __future__ import annotations

import os
from typing import Optional

# Documented, applied via ``configure_rocm_env``.
ROCM_ENV_DEFAULTS = {
    "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
    "VLLM_ROCM_USE_AITER": "1",
}


def configure_rocm_env(gpu_ids: Optional[list[int]] = None) -> dict:
    """Apply the ROCm/gfx942 env defaults (idempotent) and pin GPUs if given.

    Returns the resulting relevant env subset for logging.
    """
    for k, v in ROCM_ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)
    if gpu_ids:
        ids = ",".join(str(i) for i in gpu_ids)
        os.environ["HIP_VISIBLE_DEVICES"] = ids
        os.environ["ROCR_VISIBLE_DEVICES"] = ids
    keys = list(ROCM_ENV_DEFAULTS) + ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]
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
        **engine_kwargs,
    ):
        configure_rocm_env(gpu_ids)

        from vllm import LLM  # guarded heavy import

        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            **engine_kwargs,
        )

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
        outputs = self._llm.generate(prompts, params)
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
            outputs = self._llm.chat(messages_batch, params, **kw)
        except TypeError:  # older vLLM without chat_template_kwargs
            outputs = self._llm.chat(messages_batch, params)
        return [out.outputs[0].text for out in outputs]


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


def load_generate(model_id: str, *, backend: str = "hf", tensor_parallel_size: int = 1,
                  max_new_tokens: int = 1024, **kw):
    """Load ``model_id`` and return a single-example ``generate`` callable.

    The returned ``generate(prompt_or_messages, max_tokens=, temperature=0.0) ->
    str`` accepts EITHER a plain string prompt OR a chat-message list, so it can
    drive both simple completion and multi-turn/agentic rollouts. Heavy imports
    are guarded inside so this module still loads on a CPU box.

    backend:
      - ``"hf"``   : transformers ``AutoModelForCausalLM`` + ``apply_chat_template``.
      - ``"vllm"`` : :class:`VLLMPolicy` (fast rollout serving on ROCm).
    """
    if backend == "vllm":
        policy = VLLMPolicy(model_id, tensor_parallel_size=tensor_parallel_size, **kw)

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

        return generate

    if backend == "hf":
        import torch  # guarded heavy import
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # A LoRA checkpoint is an ADAPTER dir (adapter_config.json), not a full model:
        # AutoModelForCausalLM.from_pretrained on it loads only the tiny adapter and the
        # eval/retention gate would score garbage. Detect it, load the base named in the
        # adapter config, attach + merge the adapter (audit R2 soup-eval I1: eval path
        # must load adapters like soup._load_kore_model already does for the soup).
        if _is_lora_adapter_dir(model_id):
            import json as _json
            import os as _os
            cfg = _json.loads(open(_os.path.join(model_id, "adapter_config.json")).read())
            base_id = cfg.get("base_model_name_or_path") or model_id
            from peft import PeftModel
            tok = AutoTokenizer.from_pretrained(base_id)
            base = AutoModelForCausalLM.from_pretrained(
                base_id, torch_dtype=torch.bfloat16, device_map="auto", **kw)
            model = PeftModel.from_pretrained(base, model_id).merge_and_unload()
        else:
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto", **kw)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model.eval()

        def generate(prompt_or_messages, max_tokens: int = max_new_tokens,
                     temperature: float = 0.0) -> str:
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

        return generate

    raise ValueError(f"unknown backend {backend!r}; expected 'hf' or 'vllm'")
