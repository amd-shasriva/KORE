# Specification: `row_max` (fp32)

Take the maximum of each row of a 2-D tensor.

## Definition

```
y[m] = max over n of x[m, n]
```

## Inputs
- `x`: 2-D, shape `[M, N]`, dtype fp32, contiguous, row-major. The reduction is over the LAST axis.

## Output
- 1-D, shape `[M]`, dtype fp32 (one value per row). Allocate it yourself and return it.

## Requirements and pitfalls
- N is not always a multiple of the block size. Mask the tail with -inf (or the dtype's most negative value), NOT with 0.0, or a row that is entirely negative returns 0.

## Required entry point

Your module MUST define this exact top-level function, with this name and this parameter order:

```python
def row_max(x: torch.Tensor) -> torch.Tensor
```

It is called directly by the verifier. A module that does not define `row_max` fails to load, which scores zero regardless of what else it contains.

## How you are graded

1. The module must import and expose `row_max`.
2. `row_max` must reach at least **40 dB SNR** against the fp32 oracle on EVERY validation shape, including the one whose N is not a multiple of a power of two.
3. Only then is it timed, against the production baseline.

Write a real Triton kernel. Calling the framework op (`torch.nn.functional.*`, `torch.matmul`) as the implementation satisfies neither the intent nor the speed target.
