"""Materialize an in-memory :class:`~kore.openended.minter.MintedTask` into an
on-disk, runnable KORE task that :class:`~kore.env.kore_env.KoreEnv` grades with the
SAME trusted generic driver + reference ABI as the live generated task registry
(:mod:`kore.tasks._genops`).

Why this is safe to run on an unattended flagship
-------------------------------------------------
A minted task's reference oracle is grammar-composed (correct-by-construction) but
must be reconstructed inside a standalone ``reference.py`` that the driver SUBPROCESS
imports -- so in principle a serialization bug could grade a kernel against the wrong
oracle. We eliminate that risk with a **materialize-time self-check**: after writing
``reference.py`` we re-import it in-process and require its oracle to reproduce the
in-memory minted reference on probe inputs (bit-for-bit within the dtype tolerance).
ANY mismatch -- or any primitive the rebuild cannot resolve -- REJECTS the task
(returns ``None``). Combined with the caller's fail-safe skip + fallback to
registered tasks, enabling minting can never crash or corrupt a run: the worst case
is that a minted task is silently skipped.

The minted **seed** is the torch reference itself (a correct, ~1x baseline), so a
minted task is a well-posed "make this correct kernel fast" problem the edit-trained
policy can actually attempt -- not an impossible from-scratch task.

The task dir is written as: ``reference.py`` (rebuilds the pipeline from a NAME spec
+ re-runs the grammar builders), ``driver.py`` (the standard genops shim ->
``driver_main``), ``seed_triton.py`` (the correct torch baseline), and ``task.yaml``.
This module is import-light at module scope (torch/grammar imported lazily) so it is
safe to import on CPU / in the campaign orchestrator.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

# dtype -> correctness SNR gate (matches the generated-task convention in
# kore.tasks.generate_ops / gen_*/task.yaml).
_SNR_BY_DTYPE = {"fp32": 40.0, "bf16": 30.0, "fp16": 30.0}

_DRIVER_SHIM = '''"""GENERATED driver shim for a MINTED task. See kore/openended/materialize.py.
Do not hand-edit."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import reference as ref  # noqa: E402
from kore.tasks._genops import driver_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(driver_main(ref, _here))
'''


# --------------------------------------------------------------------------- #
# Pipeline (de)serialization: a minted pipeline is a tuple of NAMED primitives,
# every one of which is name-addressable via the grammar registries or the genops
# registry -- so a list of names + dtype + shape fully determines it. The
# self-check guarantees any name that cannot be faithfully rebuilt is rejected.
# --------------------------------------------------------------------------- #
def _prim_by_name(name: str):
    """Resolve a primitive name back to a grammar ``Primitive``.

    Checks the static source/middle/terminal registries first, then falls back to
    the genops registry (``wrap_unary`` / ``wrap_reduce`` / ``fused_primitive`` by
    op name), mirroring how the minter composed it. Raises ``KeyError`` for an
    unresolvable name (-> the task is rejected by the self-check).
    """
    from kore.openended import grammar as g
    from kore.openended import task_space as ts
    for lib in (g.source_prims(), g.middle_prims(), g.terminal_prims()):
        if name in lib:
            return lib[name]
    reg = ts._genops_registry()
    if name in reg:
        family, spec = reg[name]
        if family == "unary":
            return g.wrap_unary(name, spec.torch_fn)
        if family == "reduce":
            return g.wrap_reduce(name, spec.torch_fn)
        if family in ("binary", "fusion"):
            return g.fused_primitive(name, spec.torch_fn, getattr(spec, "arity", 2))
    raise KeyError(f"cannot rebuild minted primitive {name!r}")


def rebuild_pipeline(names):
    """Rebuild + typecheck a grammar ``Pipeline`` from a list of primitive names."""
    from kore.openended import grammar as g
    return g.Pipeline(tuple(_prim_by_name(n) for n in names)).typecheck()


def reference_namespace_from_spec(spec: dict) -> dict:
    """Reconstruct the ``_genops``-style reference namespace from a name spec.

    Called by the emitted ``reference.py`` (in the driver subprocess) AND by the
    seed. Rebuilds the exact pipeline and re-runs the SAME grammar builders
    (:func:`grammar.build_reference` / :func:`grammar.build_sampler`) the minter
    used, so the disk oracle is identical to the in-memory one.
    """
    from kore.openended import grammar as g
    pipeline = rebuild_pipeline(spec["names"])
    dtype = spec["dtype"]
    base_shape = dict(spec["shape"])

    def parse_shape(shape_str):
        if not shape_str or shape_str == "default":
            return dict(base_shape)
        out = {}
        for kv in shape_str.split(","):
            k, v = kv.split("=")
            out[k.strip()] = int(v)
        return out

    def get_inputs(shape, device="cuda", seed=0):
        return g.build_sampler(pipeline, shape, dtype)(seed, device)

    ref_fn = g.build_reference(pipeline, dtype)
    return {
        "parse_shape": parse_shape,
        "get_inputs": get_inputs,
        "ref_fn": ref_fn,
        "baseline_fn": ref_fn,
        "arity": int(spec["arity"]),
        "entry_name": spec["name"],
        "dtype_name": dtype,
        "family": spec["family"],
    }


def _spec_of(minted) -> dict:
    return {
        "names": [st.name for st in minted.pipeline.stages],
        "dtype": minted.dtype,
        "shape": dict(minted.shape),
        "name": minted.name,
        "family": minted.family,
        "arity": int(minted.arity),
        "provenance_root": minted.provenance_root,
    }


def _reference_source(spec: dict) -> str:
    return ('"""GENERATED reference for a MINTED task. See kore/openended/materialize.py."""\n'
            "from kore.openended.materialize import reference_namespace_from_spec\n\n"
            f"_SPEC = {json.dumps(spec)}\n"
            "globals().update(reference_namespace_from_spec(_SPEC))\n")


def _seed_source(spec: dict) -> str:
    """A correct torch baseline as the seed kernel: ``<name> = ref_fn``.

    Gives the policy a correct, unoptimized starting point (~1x speed) to make fast
    -- a well-posed task -- rather than an empty/from-scratch prompt.
    """
    return ("import torch  # noqa: F401 (available in the eval env)\n"
            "from kore.openended.materialize import reference_namespace_from_spec\n\n"
            f"_SPEC = {json.dumps(spec)}\n"
            "_NS = reference_namespace_from_spec(_SPEC)\n"
            f"{spec['name']} = _NS['ref_fn']\n")


def _task_yaml(spec: dict) -> str:
    # JSON is valid YAML, so Task.from_dir's yaml.safe_load reads this directly.
    dtype = spec["dtype"]
    shape = dict(spec["shape"])
    meta = {
        "task_id": f"gen_{spec['name']}_{dtype}",
        "operation": spec["name"],
        "dtype": dtype,
        "backend": "triton",
        "gpu_target": "gfx950",
        "seed_kernel_name": "seed_triton.py",
        "snr_threshold": _SNR_BY_DTYPE.get(dtype, 30.0),
        "op_family": spec["family"],
        "taxonomy_family": spec["family"],
        "provenance_root": spec["provenance_root"],
        "generated": True,
        "minted": True,
        "shapes": {"minimal": shape, "primary": shape},
        "targets": {"snr_db": _SNR_BY_DTYPE.get(dtype, 30.0),
                    "comparison_baseline": "torch"},
    }
    return json.dumps(meta, indent=2)


def _tensors_close(a, b, tol: float) -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return a == b
    if a.shape != b.shape:
        return False
    af, bf = a.float(), b.float()
    if not torch.isfinite(af).all() or not torch.isfinite(bf).all():
        return False
    return bool(torch.allclose(af, bf, atol=max(tol, 1e-4), rtol=max(tol, 1e-3)))


def _self_check(task_dir: Path, minted, seed: int = 0) -> bool:
    """The disk-reconstructed oracle MUST reproduce the in-memory minted oracle.

    Imports the freshly-written ``reference.py`` and compares ``ref_fn`` outputs to
    ``minted.reference_fn`` on the SAME probe inputs; also checks ``get_inputs``
    arity + ``entry_name``. Returns False on ANY discrepancy or missing dependency
    (torch), so a mismatch rejects the task rather than corrupting training.
    """
    import importlib.util
    try:
        from kore.openended import grammar as g
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 - cannot validate -> reject (safe)
        return False
    try:
        spec = importlib.util.spec_from_file_location(
            f"_minted_ref_{minted.task_id}", str(task_dir / "reference.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        probe = dict(minted.shape)
        inputs = g.build_sampler(minted.pipeline, probe, minted.dtype)(seed, "cpu")
        if mod.arity != minted.arity or mod.entry_name != minted.name:
            return False
        gi = mod.get_inputs(probe, device="cpu", seed=seed)
        if len(gi) != minted.arity:
            return False
        out_disk = mod.ref_fn(*inputs)
        out_mem = minted.reference_fn(*inputs)
        return _tensors_close(out_disk, out_mem, getattr(minted, "tol", 1e-3))
    except Exception:  # noqa: BLE001 - any failure -> reject (safe)
        return False


def materialize_minted_task(minted, root: Optional[Path] = None):
    """Write a minted task to disk and return a runnable :class:`Task`, or None.

    Fully fail-safe: returns ``None`` on any error OR if the self-check fails, so the
    caller can simply skip the task. ``root`` (a dir) is created if needed; a temp
    dir is used when omitted. The returned Task's ``dir`` persists for the process
    lifetime (KoreEnv copies the ``*.py`` at eval time).
    """
    try:
        from kore.tasks.base import Shape, Task
    except Exception:  # noqa: BLE001
        return None
    try:
        spec = _spec_of(minted)
        root = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="kore_minted_"))
        tdir = root / minted.task_id
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "reference.py").write_text(_reference_source(spec))
        (tdir / "driver.py").write_text(_DRIVER_SHIM)
        (tdir / "seed_triton.py").write_text(_seed_source(spec))
        (tdir / "task.yaml").write_text(_task_yaml(spec))
        if not _self_check(tdir, minted):
            return None
        # Construct via from_dir so the metadata round-trips through the same path
        # the registry uses; fall back to a manual Task if yaml is unavailable.
        try:
            return Task.from_dir(tdir)
        except Exception:  # noqa: BLE001
            shape = dict(minted.shape)
            return Task(
                task_id=minted.task_id, operation=minted.name, dtype=minted.dtype,
                backend="triton", gpu_target="gfx950", dir=tdir,
                seed_kernel_name="seed_triton.py",
                snr_threshold=_SNR_BY_DTYPE.get(minted.dtype, 30.0),
                comparison_baseline="torch",
                shapes=[Shape("minimal", shape), Shape("primary", shape)])
    except Exception:  # noqa: BLE001 - never raise into the training loop
        return None
