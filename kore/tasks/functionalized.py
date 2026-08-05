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


#: Above this many tensor arguments a "kernel" is really a whole network -- the
#: pool's tail reaches 201 parameters. 91% of functionalizable modules sit at or
#: below this, and the excluded tail is not kernel-shaped work.
MAX_ARITY = 9


def functional_namespace_from_spec(spec: dict) -> dict:
    """An oracle namespace whose entry takes ``(*activations, *parameters)``.

    Modules taking several activations (attention's q/k/v, an actor-critic's
    state/action) are ordinary here: every declared input is forwarded, so they
    functionalize on the same path as single-input ones.
    """
    import torch
    from torch.func import functional_call

    mod = _instantiate(spec)
    pairs = list(mod.named_parameters()) + list(mod.named_buffers())
    names = [n for n, _ in pairs]
    base = {n: t.detach().clone() for n, t in pairs}
    in_specs = spec.get("input_specs") or [{}]
    base_shapes = [list(s.get("shape") or [4, 4, 4, 4]) for s in in_specs]

    def _scale_one(base_shape: list[int], shape) -> list[int]:
        if shape is None:
            return list(base_shape)
        try:
            want = int(shape)
        except (TypeError, ValueError):
            return list(shape)
        tail = 1
        for d in base_shape[1:]:
            tail *= d
        return [max(1, want // max(1, tail))] + base_shape[1:]

    def _scaled_shape(shape):
        return _scale_one(base_shapes[0], shape)

    def get_inputs(shape=None, device="cuda", seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        acts = [torch.rand(*_scale_one(bs, shape), generator=g).to(device)
                for bs in base_shapes]
        # Parameters keep their own shapes: a Conv2d weight is fixed by its
        # channel counts, not by the batch pushed through it.
        return acts + [base[n].to(device) for n in names]

    n_act = len(base_shapes)

    def ref_fn(*args):
        acts, params = args[:n_act], args[n_act:]
        supplied = dict(zip(names, params)) if params else base
        return functional_call(mod, supplied, tuple(acts))

    return {
        "entry_name": spec["entry_name"],
        "family": spec.get("family", "?"),
        "dtype_name": spec.get("dtype", "fp32"),
        "arity": n_act + len(names),
        "n_activations": n_act,
        "mutates_input": False,
        "param_names": names,
        "get_inputs": get_inputs,
        "ref_fn": ref_fn,
        "baseline_fn": ref_fn,
        "parse_shape": _scaled_shape,
    }
