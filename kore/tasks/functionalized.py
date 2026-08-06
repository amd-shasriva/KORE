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

    The activations come from the pool's own reference builder rather than being
    regenerated here. That is what keeps a HIP twin comparable to its Triton
    counterpart -- same shapes, dtype, distribution and scaling rules, from one
    implementation -- and it means the shape argument follows the pool's
    convention (``parse_shape`` returns a dict, ``"default"`` is a valid value)
    instead of a second convention that has to be kept in step with it.

    Modules taking several activations (attention's q/k/v, an actor-critic's
    state/action) are ordinary here: every declared input is forwarded, so they
    functionalize on the same path as single-input ones.
    """
    from torch.func import functional_call

    from kore.tasks.external import reference_namespace_from_spec

    base_ns = reference_namespace_from_spec(spec)
    mod = _instantiate(spec)
    pairs = list(mod.named_parameters()) + list(mod.named_buffers())
    names = [n for n, _ in pairs]
    base = {n: t.detach().clone() for n, t in pairs}
    n_act = len(spec.get("input_specs") or [{}])

    def get_inputs(shape=None, device="cuda", seed=0):
        acts = list(base_ns["get_inputs"](shape, device=device, seed=seed))
        # Parameters keep their own shapes: a Conv2d weight is fixed by its
        # channel counts, not by the batch pushed through it, so they are
        # appended untouched however the activations were scaled.
        return acts + [base[n].to(device) for n in names]

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
        "parse_shape": base_ns["parse_shape"],
    }
