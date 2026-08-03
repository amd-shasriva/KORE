# Specification: `hardswish` (bf16)

Apply the Hard-Swish activation elementwise.

## Definition

```
y[i] = x[i] * clamp(x[i] + 3, 0, 6) / 6
```

## Inputs
- `x`: 2-D, shape `[M, N]`, dtype bf16, contiguous.

## Output
- 2-D, shape `[M, N]`, dtype bf16. Allocate it yourself and return it.

## Required entry point

Your module MUST define this exact top-level function, with this name and this parameter order:

```python
def hardswish(x: torch.Tensor) -> torch.Tensor
```

It is called directly by the verifier. A module that does not define `hardswish` fails to load, which scores zero regardless of what else it contains.

## How you are graded

1. The module must import and expose `hardswish`.
2. `hardswish` must reach at least **30 dB SNR** against the fp32 oracle on EVERY validation shape, including the one whose N is not a multiple of a power of two.
3. Only then is it timed, against the production baseline.

Write a real Triton kernel. Calling the framework op (`torch.nn.functional.*`, `torch.matmul`) as the implementation satisfies neither the intent nor the speed target.
