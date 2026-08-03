# Specification: `gelu_mul` (bf16)

Apply tanh-GELU to the first tensor and multiply by the second.

## Definition

```
y[i] = gelu_tanh(a[i]) * b[i]
```

## Inputs
- `a`: 2-D, shape `[M, N]`, dtype bf16, contiguous.
- `b`: 2-D, shape `[M, N]`, dtype bf16, contiguous.

## Output
- 2-D, shape `[M, N]`, dtype bf16. Allocate it yourself and return it.

## Requirements and pitfalls
- This is the GeGLU feed-forward gate: `a` is the gate projection and `b` the up projection.
- gelu_tanh is the tanh approximation defined above, with sqrt(2/pi) = 0.7978845608028654.

## Required entry point

Your module MUST define this exact top-level function, with this name and this parameter order:

```python
def gelu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
```

It is called directly by the verifier. A module that does not define `gelu_mul` fails to load, which scores zero regardless of what else it contains.

## How you are graded

1. The module must import and expose `gelu_mul`.
2. `gelu_mul` must reach at least **30 dB SNR** against the fp32 oracle on EVERY validation shape, including the one whose N is not a multiple of a power of two.
3. Only then is it timed, against the production baseline.

Write a real Triton kernel. Calling the framework op (`torch.nn.functional.*`, `torch.matmul`) as the implementation satisfies neither the intent nor the speed target.
