"""The decontamination audit must be able to fail.

``scripts/audit_decontamination.py`` gates a training launch: exit 0 means the
mixture is clean. One of its three checks was structurally incapable of reporting
a hit -- ``build_heldout_ngrams`` returns a ``HoldoutIndex``, which subclasses
``set``, so the script's ``isinstance(ref, set)`` branch iterated shingle strings
and the union it tried to build was always empty. ``[ngrams] ... overlapping: 0``
was printed on every run regardless of the file's contents, and contributed a
free pass to the CLEAN verdict.

A gate that cannot fail is worse than no gate, because it is reported as
evidence. These tests feed the audit a file that verbatim contains a held-out
kernel and require a CONTAMINATED verdict with a non-zero exit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kore.data.decontam import heldout_source_references  # noqa: E402


def _heldout_text() -> str:
    references = heldout_source_references()
    if not references:
        pytest.skip("no held-out task sources available in this checkout")
    # The longest reference gives the containment check the most to match on and
    # is the least likely to be ambiguous with a training kernel.
    return max(references, key=lambda ref: len(ref.text)).text


def _run_audit(path: Path, tmp_path: Path):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "audit_decontamination.py"),
         str(path), "--json-out", str(tmp_path / "audit.json")],
        cwd=str(REPO),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO), "HOME": str(tmp_path)},
        text=True, capture_output=True, timeout=600,
    )


def _write_rows(path: Path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_audit_flags_a_verbatim_heldout_kernel(tmp_path):
    dataset = tmp_path / "contaminated.jsonl"
    # No task id and no family hint, so the id and family checks cannot see it.
    # Only the n-gram check can, which is exactly the one that was dead.
    _write_rows(dataset, [{
        "messages": [
            {"role": "user", "content": "Optimize this kernel."},
            {"role": "assistant", "content": _heldout_text()},
        ],
        "_source": "unit-test",
    }])
    result = _run_audit(dataset, tmp_path)
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["ngram_hits"] >= 1, result.stdout
    assert report["clean"] is False
    assert "CONTAMINATED" in result.stdout
    assert result.returncode == 1


def test_audit_passes_a_row_with_no_heldout_content(tmp_path):
    dataset = tmp_path / "clean.jsonl"
    _write_rows(dataset, [{
        "messages": [
            {"role": "user", "content": "Add two vectors."},
            {"role": "assistant", "content":
                "import triton\nimport triton.language as tl\n\n"
                "@triton.jit\ndef add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):\n"
                "    pid = tl.program_id(0)\n"
                "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
                "    mask = offs < n\n"
                "    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask)"
                " + tl.load(y_ptr + offs, mask=mask), mask=mask)\n"},
        ],
        "_source": "unit-test",
    }])
    result = _run_audit(dataset, tmp_path)
    report = json.loads((tmp_path / "audit.json").read_text())
    # Generic Triton scaffolding must not be mistaken for leakage; that failure
    # mode is why decontam suppresses boilerplate from fuzzy evidence.
    assert report["ngram_hits"] == 0, result.stdout
    assert report["clean"] is True
    assert result.returncode == 0
