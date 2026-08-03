# Specification: `mul_sig` (bf16)

Gate the first tensor by the sigmoid of the second, elementwise.

## Definition

```
y[i] = a[i] * sigmoid(b[i])
```

## Inputs
- `a`: 2-D, shape `[M, N]`, dtype bf16, contiguous.
- `b`: 2-D, shape `[M, N]`, dtype bf16, contiguous.

## Output
- 2-D, shape `[M, N]`, dtype bf16. Allocate it yourself and return it.

## Requirements and pitfalls
- The sigmoid applies to the SECOND operand only; the first passes through ungated.

## Required entry point

Your module MUST define this exact top-level function, with this name and this parameter order:

```python
def mul_sig(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
```

It is called directly by the verifier. A module that does not define `mul_sig` fails to load, which scores zero regardless of what else it contains.

## How you are graded

1. The module must import and expose `mul_sig`.
2. `mul_sig` must reach at least **30 dB SNR** against the fp32 oracle on EVERY validation shape, including the one whose N is not a multiple of a power of two.
3. Only then is it timed, against the production baseline.

Write a real Triton kernel. Calling the framework op (`torch.nn.functional.*`, `torch.matmul`) as the implementation satisfies neither the intent nor the speed target.
