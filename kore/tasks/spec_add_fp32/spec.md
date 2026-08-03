# Specification: `add` (fp32)

Add two tensors elementwise.

## Definition

```
y[i] = a[i] + b[i]
```

## Inputs
- `a`: 2-D, shape `[M, N]`, dtype fp32, contiguous.
- `b`: 2-D, shape `[M, N]`, dtype fp32, contiguous.

## Output
- 2-D, shape `[M, N]`, dtype fp32. Allocate it yourself and return it.

## Requirements and pitfalls
- This is bandwidth-bound: three tensor traversals and one add. The whole problem is issuing wide, coalesced, well-occupied loads.

## Required entry point

Your module MUST define this exact top-level function, with this name and this parameter order:

```python
def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
```

It is called directly by the verifier. A module that does not define `add` fails to load, which scores zero regardless of what else it contains.

## How you are graded

1. The module must import and expose `add`.
2. `add` must reach at least **40 dB SNR** against the fp32 oracle on EVERY validation shape, including the one whose N is not a multiple of a power of two.
3. Only then is it timed, against the production baseline.

Write a real Triton kernel. Calling the framework op (`torch.nn.functional.*`, `torch.matmul`) as the implementation satisfies neither the intent nor the speed target.
