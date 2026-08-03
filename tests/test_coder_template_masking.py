"""Assistant-only loss masking for the Qwen3-Coder chat template.

The production target is Qwen3-Coder-30B-A3B-Instruct, whose template differs
structurally from Qwen3's: assistant turns are emitted from two sites, and the
second is SHARED with user and system:

    {%- elif message.role == "user" or message.role == "system"
             or message.role == "assistant" %}

Wrapping that arm wholesale is the dangerous failure. It would mark user and
system tokens as generated, training the model on its own prompts, and nothing
downstream would complain -- the loss still falls, it just falls on the wrong
tokens. So these tests check that the split actually happened, not merely that
markers appear somewhere.

Pure string transformation, so it runs on CPU with no model weights.
"""

from __future__ import annotations

import pytest

from kore.policy.sft import build_assistant_masked_template

# The two assistant-emitting branches of the real Qwen3-Coder template, verbatim.
CODER_TEMPLATE = (
    '{%- for message in loop_messages %}\n'
    '    {%- if message.role == "assistant" and message.tool_calls is defined '
    'and message.tool_calls is iterable and message.tool_calls | length > 0 %}\n'
    "        {{- '<|im_start|>' + message.role }}\n"
    '        {%- if message.content is defined and message.content is string '
    'and message.content | trim | length > 0 %}\n'
    "            {{- '\\n' + message.content | trim + '\\n' }}\n"
    '        {%- endif %}\n'
    "        {{- '<|im_end|>\\n' }}\n"
    '    {%- elif message.role == "user" or message.role == "system" '
    'or message.role == "assistant" %}\n'
    "        {{- '<|im_start|>' + message.role + '\\n' + message.content "
    "+ '<|im_end|>' + '\\n' }}\n"
    '    {%- elif message.role == "tool" %}\n'
    "        {{- '<tool_response>\\n' }}\n"
    '    {%- endif %}\n'
    '{%- endfor %}\n'
)


def test_markers_are_injected():
    out = build_assistant_masked_template(CODER_TEMPLATE)
    assert "{% generation %}" in out
    assert "{% endgeneration %}" in out


def test_shared_branch_is_split_so_user_and_system_are_not_generated():
    out = build_assistant_masked_template(CODER_TEMPLATE)
    # The three-way arm must be gone; user/system must survive on their own arm
    # with NO generation marker attached to it.
    assert 'message.role == "user" or message.role == "system" or message.role == "assistant"' not in out
    assert 'message.role == "user" or message.role == "system" %}' in out
    user_arm = out.split('message.role == "user" or message.role == "system" %}')[1]
    user_arm = user_arm.split("{%- elif")[0]
    assert "{% generation %}" not in user_arm, (
        "user/system branch is inside a generation span; the model would be "
        "trained on its own prompts"
    )


def test_assistant_header_stays_outside_the_span():
    # The <|im_start|>assistant header is a prompt the model is GIVEN, not
    # something it produced, so it must not be inside the generation span.
    out = build_assistant_masked_template(CODER_TEMPLATE)
    for seg in out.split("{% generation %}")[1:]:
        span = seg.split("{% endgeneration %}")[0]
        assert "'<|im_start|>' + message.role + '\\n' }}" not in span


def test_spans_are_balanced():
    out = build_assistant_masked_template(CODER_TEMPLATE)
    assert out.count("{% generation %}") == out.count("{% endgeneration %}")
    assert out.count("{% generation %}") == 2  # tool-call arm + plain assistant arm


def test_idempotent():
    once = build_assistant_masked_template(CODER_TEMPLATE)
    assert build_assistant_masked_template(once) == once


def test_unrecognised_template_fails_loudly():
    # Training unmasked would look fine and silently learn the prompts, so an
    # unknown template must raise rather than pass the input through.
    with pytest.raises(ValueError, match="could not inject generation markers"):
        build_assistant_masked_template("{{ messages }}")
