"""The generation path must not corrupt the question or discard the answer.

Three bugs here silently produced zeros that were indistinguishable from a model
that simply could not write kernels, so every one of them is pinned:

* a caller-built prompt was re-templated, doubling it and announcing gfx942
  against gfx950 hardware -- every arena task was asked a garbled question;
* an unclosed ``<think>`` deleted the whole reply, so a complete correct kernel
  was written to disk as an empty file;
* the kernel extractor only matched ```python, so a HIP reply fenced as ```cpp
  yielded prose or an illustrative snippet instead of the kernel.

None of these are visible in a score. They all look like "the model failed".
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from kore.eval.policies import task_prompt  # noqa: E402
from kore.policy.format import parse_response  # noqa: E402
from kore.policy.serve import _strip_think  # noqa: E402

KERNEL = ("#include <torch/extension.h>\n"
          "// a full translation unit\n"
          "int main() { return 0; }")

ARENA_PROMPT = (
    "You are writing a GPU kernel for an AMD MI355X (gfx950).\n\n"
    "Return ONLY the complete contents of `kernel.hip` in a single ```cpp\n"
    "code block, with no commentary before or after.\n")


def test_a_caller_built_prompt_is_passed_through_verbatim():
    """The arena composes its own per-category prompt naming the target file,
    fence language and reference module. Re-templating it wrapped the whole thing
    in 'Optimize the ... kernel', interpolated it twice, and appended a
    contradictory output contract."""
    assert task_prompt(ARENA_PROMPT) == ARENA_PROMPT


def test_passthrough_does_not_announce_the_wrong_architecture():
    """The template's default is gfx942; the hardware is gfx950. Naming the wrong
    architecture invites the wrong intrinsics and matrix instructions."""
    out = task_prompt(ARENA_PROMPT)
    assert "gfx942" not in out
    assert out.count("MI355X") == 1


def test_unclosed_think_does_not_delete_a_correct_kernel():
    """Cutting to end-of-string on a dangling <think> discarded the answer with
    the reasoning. The system prompt invites an unbounded scratchpad, so a
    missing closing tag is an ordinary outcome, not a rare one."""
    reply = f"<think>reasoning that never closed\n\nFULL_KERNEL:\n```cpp\n{KERNEL}\n```"
    assert KERNEL in _strip_think(reply)


def test_closed_think_is_still_stripped():
    assert "secret reasoning" not in _strip_think(
        "<think>secret reasoning</think>\nANSWER: b")


def test_strip_think_never_empties_a_non_empty_reply():
    """Returning "" writes an empty file, which scores exactly like producing
    nothing. Raw text is always more useful to a parser than nothing."""
    assert _strip_think("<think>only ever reasoning") != ""


@pytest.mark.parametrize("tag", ["cpp", "python", "c++", "hip", ""])
def test_any_fence_tag_yields_the_kernel(tag):
    assert "translation unit" in (
        parse_response(f"```{tag}\n{KERNEL}\n```").get("kernel") or "")


def test_the_largest_block_wins_when_there_is_no_full_kernel_header():
    """Models open with a small illustrative snippet before the real file. Taking
    the first block writes the sketch as the answer."""
    reply = f"Sketch:\n```cpp\nint sketch;\n```\nNow the file:\n```cpp\n{KERNEL}\n```"
    assert "translation unit" in (parse_response(reply).get("kernel") or "")


def test_full_kernel_header_beats_size():
    """An explicit marker is authoritative even when a later block is longer --
    a trailing usage example must not displace the declared answer."""
    reply = (f"FULL_KERNEL:\n```cpp\n{KERNEL}\n```\n"
             "usage:\n```python\n" + ("# padding\n" * 80) + "```")
    assert "translation unit" in (parse_response(reply).get("kernel") or "")
