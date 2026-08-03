# Specification: `row_sum` (fp32)

Sum each row of a 2-D tensor.

## Definition

```
y[m] = sum over n of x[m, n]
```

## Inputs
- `x`: 2-D, shape `[M, N]`, dtype fp32, contiguous, row-major. The reduction is over the LAST axis.

## Output
- 1-D, shape `[M]`, dtype fp32 (one value per row). Allocate it yourself and return it.

## Requirements and pitfalls
- Accumulate in fp32 even when the input is bf16/fp16: an N of 8192 summed in bf16 loses the gate on its own.

## Required entry point

Your module MUST define this exact top-level function, with this name and this parameter order:

```python
def row_sum(x: torch.Tensor) -> torch.Tensor
```

It is called directly by the verifier. A module that does not define `row_sum` fails to load, which scores zero regardless of what else it contains.

## How you are graded

1. The module must import and expose `row_sum`.
2. `row_sum` must reach at least **40 dB SNR** against the fp32 oracle on EVERY validation shape, including the one whose N is not a multiple of a power of two.
3. Only then is it timed, against the production baseline.

Write a real Triton kernel. Calling the framework op (`torch.nn.functional.*`, `torch.matmul`) as the implementation satisfies neither the intent nor the speed target.
