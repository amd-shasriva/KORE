# Specification: `add_silu` (bf16)

Add two tensors, then apply SiLU to the sum, in one pass.

## Definition

```
s[i] = a[i] + b[i];  y[i] = s[i] * sigmoid(s[i])
```

## Inputs
- `a`: 2-D, shape `[M, N]`, dtype bf16, contiguous.
- `b`: 2-D, shape `[M, N]`, dtype bf16, contiguous.

## Output
- 2-D, shape `[M, N]`, dtype bf16. Allocate it yourself and return it.

## Requirements and pitfalls
- Compute the sum ONCE and reuse it for both factors. The baseline is a compiler-fused torch chain, so a two-pass implementation that materializes the sum to memory will not beat it.

## Required entry point

Your module MUST define this exact top-level function, with this name and this parameter order:

```python
def add_silu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
```

It is called directly by the verifier. A module that does not define `add_silu` fails to load, which scores zero regardless of what else it contains.

## How you are graded

1. The module must import and expose `add_silu`.
2. `add_silu` must reach at least **30 dB SNR** against the fp32 oracle on EVERY validation shape, including the one whose N is not a multiple of a power of two.
3. Only then is it timed, against the production baseline.

Write a real Triton kernel. Calling the framework op (`torch.nn.functional.*`, `torch.matmul`) as the implementation satisfies neither the intent nor the speed target.
