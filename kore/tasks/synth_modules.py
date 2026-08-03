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
    """One shape-preserving operator, as a constructor and a forward line.

    ``weights`` is what makes a parameterized operator ANSWERABLE. A submodule
    holds its weights inside the module, where the candidate kernel can never see
    them; the driver hands the kernel exactly the tensors ``get_inputs()``
    returns, so a task built from ``nn.Linear`` asks the kernel to reproduce a
    number computed from a tensor it was not given. Measured on the overnight
    campaign: synthetic pool tasks reached a correct kernel in 4 of 956 episodes.
    Declaring the weights as extra forward ARGUMENTS instead makes the same
    operator a well-posed kernel task -- the reference is still torch, and only
    the candidate is held to computing rather than delegating.

    Each entry is ``(suffix, shape_expr)``; the rendered forward takes them as
    positional arguments after ``x`` and ``get_inputs()`` draws them.
    """

    name: str
    regime: str
    kind: str            # core | compound | supporting
    ctor: Optional[str]  # ``nn.Module`` constructor, or None for a functional op
    body: str            # forward body; ``{m}`` is the submodule attribute
    needs: tuple[str, ...] = ()
    weights: tuple[tuple[str, str], ...] = ()


# --------------------------------------------------------------------------- #
# Operator sets, categorized as popcorn does
# --------------------------------------------------------------------------- #
SEQ_CORE: tuple[OpSpec, ...] = (
    OpSpec("linear", "seq", "core", None,
           "x = torch.nn.functional.linear(x, {w0}, {w1})",
           weights=(("w", "(c, c)"), ("b", "(c,)"))),
    OpSpec("linear_nobias", "seq", "core", None,
           "x = torch.nn.functional.linear(x, {w0})",
           weights=(("w", "(c, c)"),)),
    OpSpec("conv1d", "seq", "core", None,
           "x = torch.nn.functional.conv1d("
           "x.transpose(1, 2), {w0}, padding=1).transpose(1, 2)",
           weights=(("w", "(c, c, 3)"),)),
    OpSpec("layernorm", "seq", "core", None,
           "x = torch.nn.functional.layer_norm(x, (x.shape[-1],), {w0}, {w1})",
           weights=(("g", "(c,)"), ("b", "(c,)"))),
    OpSpec("rmsnorm", "seq", "core", None,
           "x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * {w0}",
           weights=(("g", "(c,)"),)),
    OpSpec("groupnorm", "seq", "core", None,
           "x = torch.nn.functional.group_norm("
           "x.transpose(1, 2), 4, {w0}, {w1}).transpose(1, 2)",
           weights=(("g", "(c,)"), ("b", "(c,)"))),
    OpSpec("bmm_self", "seq", "core", None,
           "x = torch.softmax(x @ x.transpose(-1, -2) / 8.0, dim=-1) @ x"),
    OpSpec("gated_mlp", "seq", "compound", None,
           "a, b = torch.nn.functional.linear(x, {w0}).chunk(2, dim=-1)\n"
           "        x = a * torch.sigmoid(b)",
           weights=(("w", "(2 * c, c)"),)),
    OpSpec("residual_linear", "seq", "compound", None,
           "x = x + torch.nn.functional.linear(x, {w0})",
           weights=(("w", "(c, c)"),)),
    OpSpec("residual_norm", "seq", "compound", None,
           "x = x + torch.nn.functional.layer_norm(x, (x.shape[-1],), {w0})",
           weights=(("g", "(c,)"),)),
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
    OpSpec("conv2d_3x3", "img", "core", None,
           "x = torch.nn.functional.conv2d(x, {w0}, padding=1)",
           weights=(("w", "(c, c, 3, 3)"),)),
    OpSpec("conv2d_1x1", "img", "core", None,
           "x = torch.nn.functional.conv2d(x, {w0})",
           weights=(("w", "(c, c, 1, 1)"),)),
    OpSpec("conv2d_5x5", "img", "core", None,
           "x = torch.nn.functional.conv2d(x, {w0}, padding=2)",
           weights=(("w", "(c, c, 5, 5)"),)),
    OpSpec("depthwise_conv", "img", "core", None,
           "x = torch.nn.functional.conv2d(x, {w0}, padding=1, groups=x.shape[1])",
           weights=(("w", "(c, 1, 3, 3)"),)),
    OpSpec("convtranspose2d", "img", "core", None,
           "x = torch.nn.functional.conv_transpose2d(x, {w0}, padding=1)",
           weights=(("w", "(c, c, 3, 3)"),)),
    OpSpec("groupnorm2d", "img", "core", None,
           "x = torch.nn.functional.group_norm(x, 4, {w0}, {w1})",
           weights=(("g", "(c,)"), ("b", "(c,)"))),
    OpSpec("instancenorm2d", "img", "core", None,
           "x = torch.nn.functional.instance_norm(x, weight={w0}, bias={w1})",
           weights=(("g", "(c,)"), ("b", "(c,)"))),
    OpSpec("residual_conv", "img", "compound", None,
           "x = x + torch.nn.functional.conv2d(x, {w0}, padding=1)",
           weights=(("w", "(c, c, 3, 3)"),)),
    OpSpec("se_gate", "img", "compound", None,
           "x = x * torch.sigmoid(torch.nn.functional.conv2d("
           "x.mean(dim=(2, 3), keepdim=True), {w0}))",
           weights=(("w", "(c, c, 1, 1)"),)),
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
    """Synthesized {regime} module: {chain}.

    Every learnable tensor arrives as a forward ARGUMENT, so the oracle is a
    function of the tensors a candidate kernel is handed. A module that kept its
    weights internally would be unanswerable: the driver passes the kernel only
    what ``get_inputs()`` returns.
    """

    def __init__(self, c={channels}):
        super().__init__()
        self.c = c

    def forward(self, x{params}):
{body}
        return x


def get_inputs():
    return [torch.rand({shape}){weight_draws}]


def get_init_inputs():
    return [[], {{'c': {channels}}}]
'''


def render_module(chain: Sequence[OpSpec], regime: str) -> tuple[str, str]:
    """Render one chain as ``(class_name, module_source)``."""
    shape = REGIME_SHAPES[regime]
    channels = shape[2] if regime == "seq" else shape[1]
    body_lines: list[str] = []
    param_names: list[str] = []
    weight_draws: list[str] = []
    for index, op in enumerate(chain):
        names: list[str] = []
        for suffix, shape_expr in op.weights:
            name = f"p{index}_{suffix}"
            names.append(name)
            param_names.append(name)
            literal = shape_expr.replace("c", str(channels))
            # Centred draws: a weight from ``rand`` alone is all-positive, which
            # makes a stacked chain's activations grow with fan-in instead of
            # staying in a range the fp32 oracle and the kernel both resolve well.
            weight_draws.append(f"torch.rand({literal}) - 0.5")
        substitutions = {f"w{slot}": name for slot, name in enumerate(names)}
        substitutions["m"] = f"op{index}"
        rendered = op.body.format(**substitutions)
        for line in rendered.split("\n"):
            body_lines.append(f"        {line}" if not line.startswith(" ") else line)
    name = "SynthModel_" + "_".join(op.name for op in chain)
    source = MODULE_TEMPLATE.format(
        cls=name,
        regime=regime,
        chain=" -> ".join(op.name for op in chain),
        channels=channels,
        params="".join(f", {p}" for p in param_names),
        body="\n".join(body_lines),
        shape=list(shape),
        weight_draws="".join(f", {d}" for d in weight_draws),
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
