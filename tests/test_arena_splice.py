"""Splicing a reply back into the file it was taken from.

instruction2triton hands the model a whole module and asks for one function. A
model returns just the function, and writing that as the file deletes the imports
it needs -- which reads as a model failure and is not one. These pin the behaviour
that stops it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_agent_kernel_arena import _splice_answer  # noqa: E402

ORIGINAL = '''import torch
import triton
import triton.language as tl


def helper():
    return 42


@triton.jit
def softmax_kernel(out_ptr, in_ptr, n):
    # Your code here.
    pass


def test_softmax():
    assert helper() == 42
'''


def test_a_bare_function_reply_keeps_the_imports_and_tests():
    """The failure this exists to prevent: the written file began with
    ``@triton.jit`` at line 1 and died with NameError on import."""
    reply = '''@triton.jit
def softmax_kernel(out_ptr, in_ptr, n):
    pid = tl.program_id(0)
    tl.store(out_ptr + pid, tl.load(in_ptr + pid))
'''
    out = _splice_answer(ORIGINAL, reply)
    assert "import triton" in out
    assert "import triton.language as tl" in out
    assert "def helper()" in out
    assert "def test_softmax()" in out          # the suite that grades it survives
    assert "tl.program_id(0)" in out            # and the model's body is in
    assert "# Your code here." not in out       # the stub is gone
    assert not out.lstrip().startswith("@triton.jit")


def test_a_complete_module_reply_is_left_alone():
    """A model that returns the whole file must not have it spliced into itself."""
    reply = ORIGINAL.replace("pass", "tl.store(out_ptr, 1)")
    out = _splice_answer(ORIGINAL, reply)
    assert out == reply
    assert out.count("import triton\n") == 1


def test_unrelated_reply_passes_through_untouched():
    """Nothing in common means nothing to splice; the caller's behaviour stands."""
    reply = "def something_else():\n    return 1\n"
    assert _splice_answer(ORIGINAL, reply) == reply


def test_syntactically_invalid_reply_is_not_swallowed():
    """A truncated reply must reach the compiler as-is, so the run records a real
    compile error rather than a silently repaired file."""
    reply = "@triton.jit\ndef softmax_kernel(out_ptr,\n"
    assert _splice_answer(ORIGINAL, reply) == reply


def test_multiple_functions_are_each_replaced_in_place():
    reply = '''def helper():
    return 43


@triton.jit
def softmax_kernel(out_ptr, in_ptr, n):
    tl.store(out_ptr, 7)
'''
    out = _splice_answer(ORIGINAL, reply)
    assert "return 43" in out and "return 42" not in out
    assert "tl.store(out_ptr, 7)" in out
    assert "def test_softmax()" in out
    assert "import triton" in out


def test_decorated_definition_replaces_its_decorator_too():
    """Spanning from the decorator matters: replacing only the def would leave the
    original @triton.jit stranded above the new one."""
    reply = '''@triton.jit
def softmax_kernel(out_ptr, in_ptr, n):
    tl.store(out_ptr, 9)
'''
    out = _splice_answer(ORIGINAL, reply)
    assert out.count("@triton.jit") == 1
