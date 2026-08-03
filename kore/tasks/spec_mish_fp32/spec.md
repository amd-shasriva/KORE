# Specification: `mish` (fp32)

Apply the Mish activation elementwise.

## Definition

```
y[i] = x[i] * tanh(softplus(x[i])),  softplus(x) = log(1 + exp(x))
```

## Inputs
- `x`: 2-D, shape `[M, N]`, dtype fp32, contiguous.

## Output
- 2-D, shape `[M, N]`, dtype fp32. Allocate it yourself and return it.

## Requirements and pitfalls
- softplus overflows in fp32 for large x. Guard it: for x above roughly 20, softplus(x) == x to within fp32 resolution, so return x there instead of evaluating log(1 + exp(x)).

## Required entry point

Your module MUST define this exact top-level function, with this name and this parameter order:

```python
def mish(x: torch.Tensor) -> torch.Tensor
```

It is called directly by the verifier. A module that does not define `mish` fails to load, which scores zero regardless of what else it contains.

## How you are graded

1. The module must import and expose `mish`.
2. `mish` must reach at least **40 dB SNR** against the fp32 oracle on EVERY validation shape, including the one whose N is not a multiple of a power of two.
3. Only then is it timed, against the production baseline.

Write a real Triton kernel. Calling the framework op (`torch.nn.functional.*`, `torch.matmul`) as the implementation satisfies neither the intent nor the speed target.
