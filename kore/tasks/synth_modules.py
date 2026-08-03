"""Deterministic synthesis of diverse, valid PyTorch modules for the task pool.

``meta-pytorch/popcorn-kernels`` grows a kernel corpus in two ways: mining GitHub
(which is what ``GPUMODE/KernelBook`` is) and *synthesizing* modules by sampling
combinations from categorized operator sets and asking an LLM to write the
module.  Its released synthetic output,
``simonguozirui/popcorn-synth-pytorch-triton``, is a gated HuggingFace dataset,
and the generator itself needs an LLM API credential; neither is available here,
and the GPU budget is committed elsewhere.

What *is* reproducible is the part that carries the diversity.  The LLM in that
pipeline turns an operator combination into source; the operator combination is
what makes one synthesized module different from the next.  So this module keeps
popcorn's architecture -- categorized operator sets, sampled combinations,
generated module, validation pipeline, KernelBench contamination filter -- and
replaces the LLM with deterministic template composition.

That substitution has a cost and a benefit, and both are worth stating.  The cost
is less surface variety: an LLM writes idiosyncratic code, these modules share a
skeleton.  The benefit is that every emitted module is valid by construction and
the corpus is exactly reproducible from a seed, so no yield-rate tuning or
retry loop is needed.  Validity is still *measured*, not assumed: every
synthesized module goes through the same execution probe as a mined one.

Every operator here is shape-preserving within its regime, so any chain composes
and the emitted module always runs.  Two regimes are covered: ``seq`` for
``[B, L, C]`` activations and ``img`` for ``[B, C, H, W]``.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

#: Shape regimes and the input each one draws.
REGIME_SHAPES = {
    "seq": (4, 32, 64),
    "img": (4, 16, 16, 16),
}


@dataclass(frozen=True)
class OpSpec:
    """One shape-preserving operator, as a constructor and a forward line."""

    name: str
    regime: str
    kind: str            # core | compound | supporting
    ctor: Optional[str]  # ``nn.Module`` constructor, or None for a functional op
    body: str            # forward body; ``{m}`` is the submodule attribute
    needs: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Operator sets, categorized as popcorn does
# --------------------------------------------------------------------------- #
SEQ_CORE: tuple[OpSpec, ...] = (
    OpSpec("linear", "seq", "core", "nn.Linear(c, c)", "x = self.{m}(x)"),
    OpSpec("linear_nobias", "seq", "core", "nn.Linear(c, c, bias=False)",
           "x = self.{m}(x)"),
    OpSpec("conv1d", "seq", "core", "nn.Conv1d(c, c, 3, padding=1)",
           "x = self.{m}(x.transpose(1, 2)).transpose(1, 2)"),
    OpSpec("layernorm", "seq", "core", "nn.LayerNorm(c)", "x = self.{m}(x)"),
    OpSpec("rmsnorm", "seq", "core", "nn.Parameter(torch.ones(c))",
           "x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.{m}"),
    OpSpec("groupnorm", "seq", "core", "nn.GroupNorm(4, c)",
           "x = self.{m}(x.transpose(1, 2)).transpose(1, 2)"),
    OpSpec("bmm_self", "seq", "core", None,
           "x = torch.softmax(x @ x.transpose(-1, -2) / 8.0, dim=-1) @ x"),
    OpSpec("gated_mlp", "seq", "compound", "nn.Linear(c, 2 * c)",
           "a, b = self.{m}(x).chunk(2, dim=-1)\n        x = a * torch.sigmoid(b)"),
    OpSpec("residual_linear", "seq", "compound", "nn.Linear(c, c)",
           "x = x + self.{m}(x)"),
    OpSpec("residual_norm", "seq", "compound", "nn.LayerNorm(c)",
           "x = x + self.{m}(x)"),
    OpSpec("scaled_softmax", "seq", "supporting", None,
           "x = torch.softmax(x * 0.5, dim=-1)"),
    OpSpec("log_softmax", "seq", "supporting", None,
           "x = torch.log_softmax(x, dim=-1)"),
    OpSpec("cumsum", "seq", "supporting", None, "x = torch.cumsum(x, dim=1)"),
    OpSpec("mean_center", "seq", "supporting", None,
           "x = x - x.mean(dim=-1, keepdim=True)"),
    OpSpec("l2_normalize", "seq", "supporting", None,
           "x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)"),
)

IMG_CORE: tuple[OpSpec, ...] = (
    OpSpec("conv2d_3x3", "img", "core", "nn.Conv2d(c, c, 3, padding=1)",
           "x = self.{m}(x)"),
    OpSpec("conv2d_1x1", "img", "core", "nn.Conv2d(c, c, 1)", "x = self.{m}(x)"),
    OpSpec("conv2d_5x5", "img", "core", "nn.Conv2d(c, c, 5, padding=2)",
           "x = self.{m}(x)"),
    OpSpec("depthwise_conv", "img", "core", "nn.Conv2d(c, c, 3, padding=1, groups=c)",
           "x = self.{m}(x)"),
    OpSpec("convtranspose2d", "img", "core",
           "nn.ConvTranspose2d(c, c, 3, padding=1)", "x = self.{m}(x)"),
    OpSpec("batchnorm2d", "img", "core", "nn.BatchNorm2d(c)", "x = self.{m}(x)"),
    OpSpec("groupnorm2d", "img", "core", "nn.GroupNorm(4, c)", "x = self.{m}(x)"),
    OpSpec("instancenorm2d", "img", "core", "nn.InstanceNorm2d(c, affine=True)",
           "x = self.{m}(x)"),
    OpSpec("residual_conv", "img", "compound", "nn.Conv2d(c, c, 3, padding=1)",
           "x = x + self.{m}(x)"),
    OpSpec("se_gate", "img", "compound", "nn.Conv2d(c, c, 1)",
           "x = x * torch.sigmoid(self.{m}(x.mean(dim=(2, 3), keepdim=True)))"),
    OpSpec("avgpool_keep", "img", "supporting", None,
           "x = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)"),
    OpSpec("maxpool_keep", "img", "supporting", None,
           "x = torch.nn.functional.max_pool2d(x, 3, stride=1, padding=1)"),
    OpSpec("channel_softmax", "img", "supporting", None,
           "x = torch.softmax(x, dim=1)"),
    OpSpec("spatial_mean_center", "img", "supporting", None,
           "x = x - x.mean(dim=(2, 3), keepdim=True)"),
)

ACTIVATIONS: tuple[OpSpec, ...] = tuple(
    OpSpec(name, "any", "supporting", None, f"x = {expr}")
    for name, expr in (
        ("relu", "torch.relu(x)"),
        ("gelu", "torch.nn.functional.gelu(x)"),
        ("silu", "torch.nn.functional.silu(x)"),
        ("mish", "torch.nn.functional.mish(x)"),
        ("elu", "torch.nn.functional.elu(x)"),
        ("leaky_relu", "torch.nn.functional.leaky_relu(x, 0.1)"),
        ("hardswish", "torch.nn.functional.hardswish(x)"),
        ("softplus", "torch.nn.functional.softplus(x)"),
        ("tanh", "torch.tanh(x)"),
        ("sigmoid", "torch.sigmoid(x)"),
        ("square", "x * x"),
        ("clamp", "torch.clamp(x, -3.0, 3.0)"),
    )
)


def operators_for(regime: str) -> tuple[OpSpec, ...]:
    base = SEQ_CORE if regime == "seq" else IMG_CORE
    return base + ACTIVATIONS


MODULE_TEMPLATE = '''import torch
import torch.nn as nn


class {cls}(nn.Module):
    """Synthesized {regime} module: {chain}."""

    def __init__(self, c={channels}):
        super().__init__()
{ctors}

    def forward(self, x):
{body}
        return x


def get_inputs():
    return [torch.rand({shape})]


def get_init_inputs():
    return [[], {{'c': {channels}}}]
'''


def render_module(chain: Sequence[OpSpec], regime: str) -> tuple[str, str]:
    """Render one chain as ``(class_name, module_source)``."""
    shape = REGIME_SHAPES[regime]
    channels = shape[2] if regime == "seq" else shape[1]
    ctor_lines: list[str] = []
    body_lines: list[str] = []
    for index, op in enumerate(chain):
        attribute = f"op{index}"
        if op.ctor:
            ctor_lines.append(f"        self.{attribute} = {op.ctor}")
        for line in op.body.format(m=attribute).split("\n"):
            body_lines.append(f"        {line}" if not line.startswith(" ") else line)
    if not ctor_lines:
        ctor_lines.append("        self.scale = nn.Parameter(torch.ones(1))")
        body_lines.append("        x = x * self.scale")
    name = "SynthModel_" + "_".join(op.name for op in chain)
    source = MODULE_TEMPLATE.format(
        cls=name,
        regime=regime,
        chain=" -> ".join(op.name for op in chain),
        channels=channels,
        ctors="\n".join(ctor_lines),
        body="\n".join(body_lines),
        shape=list(shape),
    )
    return name, source


def _chain_space(regime: str, lengths: Sequence[int]) -> Iterator[tuple[OpSpec, ...]]:
    """Every operator chain of the requested lengths, in a stable order."""
    ops = operators_for(regime)
    for length in lengths:
        for chain in itertools.permutations(ops, length):
            yield chain


def synthesize(
    count: int,
    seed: int = 0,
    regimes: Sequence[str] = ("seq", "img"),
    lengths: Sequence[int] = (2, 3, 4),
) -> list[tuple[str, str, str]]:
    """Return up to ``count`` ``(regime, class_name, module_source)`` triples.

    Chains are drawn by reservoir-free deterministic sampling: the chain space is
    enumerated in a fixed order and each candidate is accepted with a probability
    that depends only on ``seed``, so the same seed and count always produce the
    same corpus without materializing the (very large) full space.
    """
    rng = random.Random(seed)
    per_regime = max(1, count // max(1, len(regimes)))
    out: list[tuple[str, str, str]] = []
    for regime in regimes:
        seen: set[str] = set()
        emitted = 0
        # A generous ceiling on how much of the chain space to walk; the space is
        # combinatorially large, so a chain is skipped rather than searched for.
        budget = per_regime * 40
        for index, chain in enumerate(_chain_space(regime, lengths)):
            if emitted >= per_regime or index >= budget:
                break
            if rng.random() > 0.5:
                continue
            # A chain of only functional ops with no parameters is a pointwise
            # composition, which the elementwise family already covers densely.
            if not any(op.kind in {"core", "compound"} for op in chain):
                continue
            name, source = render_module(chain, regime)
            if name in seen:
                continue
            seen.add(name)
            out.append((regime, name, source))
            emitted += 1
    return out[:count]


__all__ = [
    "ACTIVATIONS",
    "IMG_CORE",
    "MODULE_TEMPLATE",
    "OpSpec",
    "REGIME_SHAPES",
    "SEQ_CORE",
    "operators_for",
    "render_module",
    "synthesize",
]
