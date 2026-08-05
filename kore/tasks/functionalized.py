"""Turn a stateful ``nn.Module`` pool task into a pure function of its tensors.

A pool task's oracle closes over an instantiated module, so its learned weights
live inside the closure. A Triton candidate is a Python file and can reach in and
read them; a ``.hip`` file is called with only the declared input tensors and
never can. That asymmetry -- not anything about HIP -- is why only the 3,607
parameter-free modules of 13,570 could become HIP tasks.

The weights do not have to be hidden. ``torch.func.functional_call`` evaluates a
module against parameters supplied from outside, and it is exact:
``functional_call(mod, params, (x,))`` is bit-identical to ``mod(x)``. So a task
can declare its parameters as additional inputs, hand them to the candidate
alongside the activation, and a HIP kernel can compute ``conv2d(x, W)`` for a W
it can actually see. That makes all 13,570 pool modules HIP-eligible.

Two properties are load-bearing:

**The parameters must be identical every time.** They are drawn under a fixed
generator seed, so the reference and the candidate are compared against the same
weights on every run and across processes. A module re-initialised per call would
make correctness unreproducible.

**Only the activation scales.** ``get_inputs(shape)`` resizes the leading
activation and returns the parameter tensors unchanged: a Conv2d's weight is
determined by its channel counts, not by how large a batch you push through it.
Scaling them too would produce shapes the module cannot consume.
"""

from __future__ import annotations

from typing import Any


#: Deterministic init. The candidate is graded against these exact weights, so
#: they must not vary between the reference run and the scored run.
PARAM_SEED = 0


def _instantiate(spec: dict):
    import torch

    ns: dict[str, Any] = {}
    exec(spec["module_source"], ns)  # noqa: S102 - pool sources are admitted upstream
    cls = ns[spec["entry_class"]]
    torch.manual_seed(PARAM_SEED)
    mod = cls(*(spec.get("init_args") or []), **(spec.get("init_kwargs") or {}))
    mod.eval()
    return mod


def parameter_tensors(spec: dict) -> list[tuple[str, Any]]:
    """(name, tensor) for every parameter and buffer, in a stable order."""
    mod = _instantiate(spec)
    named = list(mod.named_parameters()) + list(mod.named_buffers())
    return [(n, t.detach().clone()) for n, t in named]


def functional_namespace_from_spec(spec: dict) -> dict:
    """An oracle namespace whose entry takes ``(activation, *parameters)``."""
    import torch
    from torch.func import functional_call

    mod = _instantiate(spec)
    names = [n for n, _ in list(mod.named_parameters()) + list(mod.named_buffers())]
    base = {n: t.detach().clone()
            for n, t in list(mod.named_parameters()) + list(mod.named_buffers())}
    in_spec = (spec.get("input_specs") or [{}])[0]
    base_shape = list(in_spec.get("shape") or [4, 4, 4, 4])

    def _scaled_shape(shape) -> list[int]:
        """Resize the activation to `shape` elements, keeping its trailing dims."""
        if shape is None:
            return base_shape
        try:
            want = int(shape)
        except (TypeError, ValueError):
            return list(shape)
        tail = 1
        for d in base_shape[1:]:
            tail *= d
        lead = max(1, want // max(1, tail))
        return [lead] + base_shape[1:]

    def get_inputs(shape=None, device="cuda", seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.rand(*_scaled_shape(shape), generator=g).to(device)
        # Parameters keep their own shapes: a Conv2d weight is fixed by its
        # channel counts, not by the batch pushed through it.
        return [x] + [base[n].to(device) for n in names]

    def ref_fn(x, *params):
        supplied = dict(zip(names, params)) if params else base
        return functional_call(mod, supplied, (x,))

    return {
        "entry_name": spec["entry_name"],
        "family": spec.get("family", "?"),
        "dtype_name": spec.get("dtype", "fp32"),
        "arity": 1 + len(names),
        "mutates_input": False,
        "param_names": names,
        "get_inputs": get_inputs,
        "ref_fn": ref_fn,
        "baseline_fn": ref_fn,
        "parse_shape": _scaled_shape,
    }
