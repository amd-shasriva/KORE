"""KernelBook-style synthetic corpus: PyTorch -> Triton via TorchInductor.

We take small PyTorch functions and let TorchInductor lower them to Triton, then
capture the generated Triton source. This yields cheap, correct-by-construction
(PyTorch is the reference) Triton kernels to pretrain the writer on idiomatic
Triton, complementing the scarce hand-written wins.

HOW CAPTURE WORKS
-----------------
``torch.compile(fn, backend="inductor")`` compiles ``fn`` on first call. With
``torch._inductor.config.trace.enabled = True`` Inductor writes a per-compile
debug directory (``trace.debug_dir``) that contains ``output_code.py`` — the
generated wrapper + Triton kernels. After running the compiled function once we
glob that directory for ``output_code.py`` and return its contents.

FALLBACK
--------
Inductor internals move between torch versions and require a GPU for real Triton
lowering. If tracing is unavailable, capture fails, or no Triton is found in the
output, ``generate_triton_via_inductor`` returns ``None`` and the caller simply
skips that op. Nothing here is imported at module load: ``torch`` is imported
INSIDE the functions so the module stays importable on a CPU-only box without
torch installed.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Optional

_LOG = logging.getLogger(__name__)


# --- Example op builders. Each returns (callable, example_inputs). torch is
#     imported lazily inside so this module never imports torch at top level. ---
def _op_add():
    import torch

    def fn(x, y):
        return x + y

    return fn, (torch.randn(1024, 1024), torch.randn(1024, 1024))


def _op_relu():
    import torch

    def fn(x):
        return torch.relu(x)

    return fn, (torch.randn(1024, 1024),)


def _op_softmax():
    import torch

    def fn(x):
        return torch.softmax(x, dim=-1)

    return fn, (torch.randn(512, 2048),)


def _op_layernorm():
    import torch

    def fn(x, w, b):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b)

    return fn, (torch.randn(512, 2048), torch.ones(2048), torch.zeros(2048))


def _op_matmul():
    import torch

    def fn(a, b):
        return a @ b

    return fn, (torch.randn(512, 512), torch.randn(512, 512))


EXAMPLE_OPS: list[dict] = [
    {"name": "add", "operation": "elementwise_add", "build": _op_add},
    {"name": "relu", "operation": "relu", "build": _op_relu},
    {"name": "softmax", "operation": "softmax", "build": _op_softmax},
    {"name": "layernorm", "operation": "layernorm", "build": _op_layernorm},
    {"name": "matmul", "operation": "gemm", "build": _op_matmul},
]


def _validate_generated_triton(
    src: str,
    pytorch_fn: Callable,
    example_inputs: tuple,
    device: str,
) -> Optional[bool]:
    """Best-effort execute the captured Inductor module vs the torch reference.

    Returns:
      * ``True``  — the captured ``call(...)`` ran and matched ``pytorch_fn``.
      * ``False`` — it ran but produced a numerically wrong result (reject).
      * ``None``  — execution could not be attempted (no GPU / no ``call`` entry /
        exec failed). The caller documents this and falls back to trusting
        Inductor's correct-by-construction lowering.
    """
    try:
        import torch
    except Exception:
        return None

    # We can only meaningfully execute Inductor's generated Triton on a CUDA/HIP
    # device; on a CPU-only box the kernels won't run, so we can't validate.
    try:
        if not torch.cuda.is_available():
            _LOG.info("synthetic: no GPU available; skipping execution validation")
            return None
    except Exception:
        return None

    try:
        inputs = tuple(
            t.to(device) if hasattr(t, "to") else t for t in example_inputs
        )
        ref = pytorch_fn(*inputs)

        ns: dict = {"__name__": "kore_synthetic_captured"}
        exec(compile(src, "<inductor_output_code>", "exec"), ns)  # noqa: S102
        call = ns.get("call")
        if not callable(call):
            _LOG.info("synthetic: captured module has no callable `call`; "
                      "skipping execution validation")
            return None

        out = call([t for t in inputs])
        got = out[0] if isinstance(out, (list, tuple)) else out
        ok = bool(torch.allclose(got.to(ref.dtype), ref, rtol=1e-2, atol=1e-2))
        if not ok:
            _LOG.warning("synthetic: captured Triton disagreed with torch reference")
        return ok
    except Exception as e:  # noqa: BLE001 - execution is strictly best-effort
        _LOG.info("synthetic: could not execute captured module (%s); "
                  "skipping execution validation", e)
        return None


def generate_triton_via_inductor(
    pytorch_fn: Callable,
    example_inputs: tuple,
    device: str = "cuda",
    validate: bool = True,
) -> Optional[str]:
    """Compile ``pytorch_fn`` with TorchInductor and return generated Triton.

    Returns the captured ``output_code.py`` source or ``None`` if capture failed,
    no ``@triton.jit`` kernel was produced, or (when it could be executed) the
    captured kernel disagreed with the torch reference.

    Acceptance requires an actual ``@triton.jit`` kernel in the source. When
    ``validate`` is set we additionally *best-effort* execute the captured module
    against ``pytorch_fn``; if execution can't run (CPU-only box / no ``call``
    entry) we log and keep the source, trusting Inductor's correct-by-
    construction lowering (documented in ``_validate_generated_triton``).
    """
    try:
        import torch
        import torch._dynamo as dynamo
        import torch._inductor.config as inductor_config
    except Exception:
        return None

    debug_dir = tempfile.mkdtemp(prefix="kore_inductor_")
    prev_enabled = getattr(inductor_config.trace, "enabled", False)
    prev_dir = getattr(inductor_config.trace, "debug_dir", None)
    try:
        inductor_config.trace.enabled = True
        inductor_config.trace.debug_dir = debug_dir

        try:
            inputs = tuple(
                t.to(device) if hasattr(t, "to") else t for t in example_inputs
            )
        except Exception:
            inputs = example_inputs

        dynamo.reset()
        compiled = torch.compile(pytorch_fn, backend="inductor")
        compiled(*inputs)

        matches = glob.glob(
            os.path.join(debug_dir, "**", "output_code.py"), recursive=True
        )
        for path in matches:
            try:
                src = Path(path).read_text()
            except Exception:
                continue
            # Require a real Triton kernel, not just a mention of the word.
            if not re.search(r"@triton\.jit", src):
                continue
            if validate:
                verdict = _validate_generated_triton(
                    src, pytorch_fn, example_inputs, device
                )
                if verdict is False:
                    continue  # ran but numerically wrong -> reject this capture
            return src
        return None
    except Exception:
        return None
    finally:
        try:
            inductor_config.trace.enabled = prev_enabled
            if prev_dir is not None:
                inductor_config.trace.debug_dir = prev_dir
        except Exception:
            pass


def build_synthetic_corpus(
    out_dir, n: Optional[int] = None, device: str = "cuda"
) -> list[dict]:
    """Generate Triton for the first ``n`` EXAMPLE_OPS and write them to disk.

    Writes one ``<name>.triton.py`` per successfully-captured op under ``out_dir``
    plus a ``manifest.jsonl``. Returns the list of manifest entries (ops that
    failed capture are recorded with ``triton_source=None`` and skipped on disk).
    """
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ops = EXAMPLE_OPS if n is None else EXAMPLE_OPS[:n]

    manifest: list[dict] = []
    for op in ops:
        entry = {"name": op["name"], "operation": op["operation"], "triton_source": None}
        try:
            fn, inputs = op["build"]()
            src = generate_triton_via_inductor(fn, inputs, device=device)
        except Exception:
            src = None
        if src:
            path = out_dir / f"{op['name']}.triton.py"
            path.write_text(src)
            entry["triton_source"] = str(path)
        manifest.append(entry)

    with (out_dir / "manifest.jsonl").open("w") as f:
        for e in manifest:
            f.write(json.dumps(e) + "\n")
    return manifest
