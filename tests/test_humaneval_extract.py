"""Regression tests for robust HumanEval/LiveCodeBench solution extraction.

Each case reproduces an output style that broke the old parser (imports before the
def -> NameError; flush-left body -> IndentationError; helper after the def dropped;
chatty prose + fence) and asserts the assembled program runs and passes.
"""
from kore.eval.retention import HumanEvalScorer, _run_python_program

ITEM = {
    "entry_point": "has_close_elements",
    "prompt": (
        "from typing import List\n\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """ True if any two numbers are closer than threshold. """\n'
    ),
    "test": (
        "\ndef check(candidate):\n"
        "    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n"
        "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n\n"
        "check(has_close_elements)\n"
    ),
}

_FENCED_WITH_IMPORT = """Here's the solution:

```python
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

This checks every pair of numbers.
"""

_BARE_WITH_IMPORT = """from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""

_HELPER_AFTER = """def has_close_elements(numbers, threshold):
    return _any_close(numbers, threshold)


def _any_close(nums, t):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if abs(nums[i] - nums[j]) < t:
                return True
    return False
"""

_BODY_ONLY_FLUSH = """for i in range(len(numbers)):
    for j in range(i + 1, len(numbers)):
        if abs(numbers[i] - numbers[j]) < threshold:
            return True
return False
"""

_BODY_ONLY_INDENTED = """    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""

_CASES = {
    "fenced_with_import": _FENCED_WITH_IMPORT,
    "bare_with_import": _BARE_WITH_IMPORT,
    "helper_after": _HELPER_AFTER,
    "body_only_flush": _BODY_ONLY_FLUSH,
    "body_only_indented": _BODY_ONLY_INDENTED,
}


def test_build_program_runs_for_all_output_styles():
    for name, completion in _CASES.items():
        program = HumanEvalScorer.build_program(ITEM, completion)
        res = _run_python_program(program, timeout=10)
        assert res["passed"], f"{name} failed to run/pass: {res.get('detail')}\n---\n{program}"
