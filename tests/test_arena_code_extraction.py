"""Fenced-code extraction must follow whatever language the prompt asked for.

This is pinned because it already broke once, silently and expensively. The
prompt was changed to request ```cpp for .hip targets while the extractor still
only accepted ```python, so every HIP reply fell through to the "return the whole
message" fallback and the compiler was handed the model's prose along with its
kernel. hip2hip went from 23 compiled to 12 and it read as a model regression.

Nothing in the score can distinguish "the model wrote a bad kernel" from "we fed
the compiler an English sentence", which is why the parser is tested directly.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from run_agent_kernel_arena import _extract_code  # noqa: E402


KERNEL = "int main() { return 0; }"


@pytest.mark.parametrize("tag", ["", "python", "cpp", "c++", "hip", "CUDA",
                                 "python3", "cpp17"])
def test_every_language_tag_is_accepted(tag):
    reply = f"Here is the kernel:\n```{tag}\n{KERNEL}\n```\nHope that helps."
    assert _extract_code(reply) == KERNEL


def test_prose_around_the_block_is_stripped():
    """The compiler gets the block, never the commentary wrapped around it."""
    reply = f"I'll use shared memory.\n\n```cpp\n{KERNEL}\n```\n\nNote the tiling."
    out = _extract_code(reply)
    assert "shared memory" not in out and "tiling" not in out


def test_first_block_wins_when_several_are_present():
    """Models often follow the answer with a usage snippet; the answer is first."""
    reply = f"```cpp\n{KERNEL}\n```\nand to call it:\n```python\nrun()\n```"
    assert _extract_code(reply) == KERNEL


def test_unfenced_reply_is_still_used():
    """A model that forgets the fence usually still emitted valid source, and
    scoring a formatting slip as a compile failure would understate it."""
    assert _extract_code(KERNEL) == KERNEL


def test_crlf_fences_are_handled():
    assert _extract_code(f"```cpp\r\n{KERNEL}\r\n```").strip() == KERNEL


def test_tag_with_trailing_spaces():
    assert _extract_code(f"```cpp   \n{KERNEL}\n```") == KERNEL
