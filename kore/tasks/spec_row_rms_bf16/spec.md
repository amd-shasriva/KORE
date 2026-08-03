# Specification: `row_rms` (bf16)

Compute the root-mean-square of each row of a 2-D tensor.

## Definition

```
y[m] = sqrt( (1/N) * sum over n of x[m, n]**2 )
```

## Inputs
- `x`: 2-D, shape `[M, N]`, dtype bf16, contiguous, row-major. The reduction is over the LAST axis.

## Output
- 1-D, shape `[M]`, dtype bf16 (one value per row). Allocate it yourself and return it.

## Requirements and pitfalls
- Divide the sum of squares by N before the square root. This is the normalizer inside RMSNorm.
- Accumulate the squares in fp32.
- Mask the tail of a partial block with 0.0, which is the identity for this accumulator.

## Required entry point

Your module MUST define this exact top-level function, with this name and this parameter order:

```python
def row_rms(x: torch.Tensor) -> torch.Tensor
```

It is called directly by the verifier. A module that does not define `row_rms` fails to load, which scores zero regardless of what else it contains.

## How you are graded

1. The module must import and expose `row_rms`.
2. `row_rms` must reach at least **30 dB SNR** against the fp32 oracle on EVERY validation shape, including the one whose N is not a multiple of a power of two.
3. Only then is it timed, against the production baseline.

Write a real Triton kernel. Calling the framework op (`torch.nn.functional.*`, `torch.matmul`) as the implementation satisfies neither the intent nor the speed target.
