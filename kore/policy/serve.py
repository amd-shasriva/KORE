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
    ) -> list[str]:
        """Generate from a batch of chat-message lists via vLLM's chat API."""
        from vllm import SamplingParams

        params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        outputs = self._llm.chat(messages_batch, params)
        return [out.outputs[0].text for out in outputs]


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
                return policy.generate([prompt_or_messages], temperature=temperature,
                                       max_tokens=max_tokens)[0]
            return policy.chat([prompt_or_messages], temperature=temperature,
                               max_tokens=max_tokens)[0]

        return generate

    if backend == "hf":
        import torch  # guarded heavy import
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", **kw)
        model.eval()

        def generate(prompt_or_messages, max_tokens: int = max_new_tokens,
                     temperature: float = 0.0) -> str:
            messages = ([{"role": "user", "content": prompt_or_messages}]
                        if isinstance(prompt_or_messages, str) else prompt_or_messages)
            ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            do_sample = bool(temperature and temperature > 0.0)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=max_tokens, do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=tok.pad_token_id)
            gen = out[0][ids.shape[1]:]
            return tok.decode(gen, skip_special_tokens=True)

        return generate

    raise ValueError(f"unknown backend {backend!r}; expected 'hf' or 'vllm'")
