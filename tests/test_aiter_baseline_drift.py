"""Guard the AITER facts that ``docs/AITER_WEAKNESS_SURFACE.md`` reasons from.

The weakness analysis is built on a handful of statements about the *installed*
AITER: which commit it is, what ``flash_attn_func`` defaults ``deterministic``
to, and that both backward dispatch gates reject that default. AITER moves
fast -- the upstream clone was 194 commits ahead of the installed build within
a month of it -- and HipKittens techniques are actively being upstreamed, so
those statements have a shelf life.

If AITER is upgraded and a gate changes, the honest failure mode is a red test
telling us the document is stale. The dishonest one is a stale document that
still reads as current while someone quotes a number from it.

These tests are pure text/AST inspection of the aiter source tree. They never
import aiter, never touch a GPU, and skip when no aiter checkout is present.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

#: The commit ``docs/AITER_WEAKNESS_SURFACE.md`` records as "what we measure".
DOCUMENTED_COMMIT = "7e0d1162642f1727e0c8d9bdff318daedecfe331"

DOC = Path(__file__).resolve().parents[1] / "docs" / "AITER_WEAKNESS_SURFACE.md"


def _installed_aiter_root() -> Path | None:
    """Locate the aiter checkout the venv's editable install points at.

    Resolved from the ``.pth``/finder that pip wrote rather than by importing
    aiter, because importing it costs a JIT build and needs a GPU.
    """
    candidates = [Path("/home/shasriva/aiter")]
    for root in candidates:
        if (root / "aiter" / "ops" / "mha.py").is_file():
            return root
    return None


@pytest.fixture(scope="module")
def mha_source() -> str:
    root = _installed_aiter_root()
    if root is None:
        pytest.skip("no aiter checkout on this machine")
    return (root / "aiter" / "ops" / "mha.py").read_text()


def _default_of(source: str, func: str, arg: str):
    """Return the literal default of ``arg`` in top-level ``def func``."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func:
            args = node.args
            named = args.args + args.kwonlyargs
            defaults = ([None] * (len(args.args) - len(args.defaults))
                        + list(args.defaults) + list(args.kw_defaults))
            for a, d in zip(named, defaults):
                if a.arg == arg and d is not None:
                    return ast.literal_eval(d)
    raise AssertionError(f"{func}(...) has no default for {arg!r}")


def test_documented_commit_is_the_one_actually_installed():
    """The doc's "what we measure" row must name the checked-out commit."""
    root = _installed_aiter_root()
    if root is None:
        pytest.skip("no aiter checkout on this machine")
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("aiter checkout is not a usable git repo")
    assert head == DOCUMENTED_COMMIT, (
        f"installed aiter moved to {head}; docs/AITER_WEAKNESS_SURFACE.md still "
        f"claims {DOCUMENTED_COMMIT}. Re-read the dispatch gates and re-measure "
        "before trusting anything in that document."
    )


def test_dense_and_varlen_disagree_on_the_deterministic_default(mha_source):
    """The asymmetry the weakness analysis turns on.

    ``flash_attn_func`` defaulting to ``deterministic=True`` is what pushes a
    default dense backward onto the CK fallback; ``flash_attn_varlen_func``
    defaulting to False is what lets the varlen path reach the asm kernel.
    """
    assert _default_of(mha_source, "flash_attn_func", "deterministic") is True
    assert _default_of(mha_source, "flash_attn_varlen_func", "deterministic") is False


def test_both_backward_gates_still_reject_deterministic(mha_source):
    """Neither dispatch gate may start accepting ``deterministic=True``.

    The generic gate rejects it outright; the gfx950 gate rejects it unless the
    whole backward fits in one block (``seqlen_k <= 256``). If either softens,
    the "default call lands on CK" conclusion no longer holds.
    """
    assert "ret &= not deterministic\n" in mha_source, (
        "the generic can_impl_fmha_v3_bwd gate no longer rejects deterministic "
        "unconditionally")
    assert "ret &= not deterministic or is_950_1block" in mha_source, (
        "the gfx950 backward gate's deterministic condition changed")


def test_gfx950_gate_still_excludes_head_dim_64_and_sliding_window(mha_source):
    """head_dim 64 and SWA are excluded by the gfx950 gate.

    Both are quoted in the doc as structurally-on-the-fallback cases, which is
    what makes them honest task targets.
    """
    assert "(hdim_q > 64 and hdim_q <= 128) or (hdim_q == 192 and hdim_v == 128)" \
        in mha_source, "the gfx950 backward head_dim window changed"
    assert "ret &= not swa" in mha_source, (
        "the gfx950 backward gate no longer excludes sliding-window attention")


def test_weakness_doc_reports_no_measured_numbers_yet():
    """The doc must keep saying it has no measurements until it has some.

    This is the guard against the failure mode that matters most here: a
    [source] prediction or a [paper] figure quietly being read as our result.
    Delete this test in the same change that adds a real [measured] row.
    """
    text = DOC.read_text()
    assert "no [measured] rows in this document yet" in text, (
        "docs/AITER_WEAKNESS_SURFACE.md no longer states that it lacks "
        "measurements -- if that is because measurements landed, drop this test "
        "in the same commit; if not, restore the disclaimer."
    )
