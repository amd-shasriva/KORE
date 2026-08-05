"""Invariants for turning a stateful pool module into a pure function.

Functionalization is what makes parameterized modules HIP-eligible, so the
properties a HIP win depends on are pinned here: the oracle must be exact, the
weights must be identical on every rebuild, and only the activation may scale.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kore.tasks.functionalized import (  # noqa: E402
    MAX_ARITY, functional_namespace_from_spec, parameter_tensors)

CONV = {
    "task_id": "t_conv", "entry_name": "fused_conv", "entry_class": "M",
    "family": "convolution", "dtype": "fp32", "primary_scale": 4096,
    "input_specs": [{"shape": [4, 3, 8, 8]}],
    "module_source": (
        "import torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.c = nn.Conv2d(3, 5, 3, padding=1)\n"
        "    def forward(self, x):\n"
        "        return self.c(x).relu()\n"),
}

PARAM_FREE = {
    "task_id": "t_free", "entry_name": "scale", "entry_class": "M",
    "family": "activation", "dtype": "fp32",
    "input_specs": [{"shape": [8, 16]}],
    "module_source": (
        "import torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return (x * 2.0).relu()\n"),
}

TWO_INPUT = {
    "task_id": "t_two", "entry_name": "bilin", "entry_class": "M",
    "family": "gemm", "dtype": "fp32",
    "input_specs": [{"shape": [4, 6]}, {"shape": [4, 6]}],
    "module_source": (
        "import torch\nimport torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.l = nn.Linear(6, 6)\n"
        "    def forward(self, a, b):\n"
        "        return self.l(a) + b\n"),
}


def _plain(spec):
    ns = {}
    exec(spec["module_source"], ns)  # noqa: S102
    torch.manual_seed(0)
    mod = ns[spec["entry_class"]]()
    mod.eval()
    return mod


def test_oracle_is_exact_against_the_stateful_module():
    """The whole approach rests on functional_call being the same computation."""
    ns = functional_namespace_from_spec(CONV)
    ins = ns["get_inputs"](None, device="cpu", seed=0)
    with torch.no_grad():
        assert torch.equal(_plain(CONV)(ins[0]), ns["ref_fn"](*ins))


def test_parameters_are_exposed_as_trailing_arguments():
    ns = functional_namespace_from_spec(CONV)
    assert ns["n_activations"] == 1
    assert ns["param_names"] == ["c.weight", "c.bias"]
    assert ns["arity"] == 3
    ins = ns["get_inputs"](None, device="cpu", seed=0)
    assert [tuple(t.shape) for t in ins[1:]] == [(5, 3, 3, 3), (5,)]


def test_weights_are_identical_across_rebuilds():
    """A candidate is graded in a different process than the reference; drifting
    weights would make correctness unreproducible rather than merely wrong."""
    a = [t for _, t in parameter_tensors(CONV)]
    b = [t for _, t in parameter_tensors(CONV)]
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_scaling_resizes_the_activation_and_leaves_weights_alone():
    """A Conv2d weight is fixed by its channel counts; scaling it with the batch
    would hand the module a shape it cannot consume."""
    ns = functional_namespace_from_spec(CONV)
    small = ns["get_inputs"](None, device="cpu", seed=0)
    big = ns["get_inputs"](8192, device="cpu", seed=0)
    assert big[0].shape[0] > small[0].shape[0]
    assert big[0].shape[1:] == small[0].shape[1:]
    assert [tuple(t.shape) for t in big[1:]] == [tuple(t.shape) for t in small[1:]]
    with torch.no_grad():
        assert ns["ref_fn"](*big).shape[0] == big[0].shape[0]


def test_multi_input_modules_forward_every_activation():
    """Attention-shaped modules take q/k/v; passing only the first raised
    TypeError and silently cost 1,121 pool tasks."""
    ns = functional_namespace_from_spec(TWO_INPUT)
    assert ns["n_activations"] == 2
    ins = ns["get_inputs"](None, device="cpu", seed=0)
    with torch.no_grad():
        assert torch.equal(_plain(TWO_INPUT)(ins[0], ins[1]), ns["ref_fn"](*ins))


def test_parameter_free_modules_report_no_parameters():
    ns = functional_namespace_from_spec(PARAM_FREE)
    assert ns["param_names"] == []
    assert ns["arity"] == ns["n_activations"] == 1


def test_entry_name_is_preserved():
    """The harness looks up this exact symbol; a mismatch reads as a wrong answer."""
    assert functional_namespace_from_spec(CONV)["entry_name"] == "fused_conv"


def test_arity_cap_excludes_whole_networks():
    """The pool's tail reaches 201 parameter tensors, which is a model rather
    than the kernel-shaped work the benchmark measures."""
    from scripts.materialize_pool_hip import _functional_info
    assert MAX_ARITY == 9
    deep = dict(CONV)
    layers = "\n".join(f"        self.c{i} = nn.Conv2d(3, 3, 1)" for i in range(12))
    calls = "\n".join(f"        x = self.c{i}(x)" for i in range(12))
    deep["module_source"] = (
        "import torch.nn as nn\nclass M(nn.Module):\n"
        "    def __init__(self):\n        super().__init__()\n" + layers +
        "\n    def forward(self, x):\n" + calls + "\n        return x\n")
    assert functional_namespace_from_spec(deep)["arity"] > MAX_ARITY
    assert _functional_info(deep) is None
    assert _functional_info(CONV) is not None


def test_multi_output_modules_are_not_admitted():
    """Returning several tensors needs the seed to return a tuple through pybind,
    a different contract from the one the prompt states."""
    from scripts.materialize_pool_hip import _functional_info
    two_out = dict(CONV)
    two_out["module_source"] = (
        "import torch.nn as nn\nclass M(nn.Module):\n"
        "    def __init__(self):\n        super().__init__()\n"
        "        self.c = nn.Conv2d(3, 5, 3, padding=1)\n"
        "    def forward(self, x):\n"
        "        y = self.c(x)\n        return y, y.sum()\n")
    assert _functional_info(two_out) is None


def test_admission_runs_the_module_rather_than_reading_its_source():
    """A module whose weights cannot be supplied from outside must be rejected
    before it costs a teacher call and a gate slot."""
    from scripts.materialize_pool_hip import _functional_info
    assert _functional_info(CONV) is not None
    broken = dict(CONV)
    broken["entry_class"] = "NotAClass"
    assert _functional_info(broken) is None


def test_prompt_names_every_parameter_with_its_shape_and_position():
    """An unnamed parameter list makes the teacher ignore the trailing arguments
    and recompute the module with weights of its own invention."""
    from scripts.materialize_pool_hip import _build_prompt, _functional_info
    prompt, entry = _build_prompt(CONV, _functional_info(CONV))
    assert entry == "fused_conv"
    assert "PARAMETERS ARE ARGUMENTS" in prompt
    assert "arg 1: c.weight  shape (5, 3, 3, 3)" in prompt
    assert "arg 2: c.bias  shape (5,)" in prompt
    assert "do not initialise, hardcode, or assume any weight values" in prompt


def test_prompt_without_functionalization_says_nothing_about_parameters():
    from scripts.materialize_pool_hip import _build_prompt
    prompt, _ = _build_prompt(PARAM_FREE, None)
    assert "PARAMETERS ARE ARGUMENTS" not in prompt
