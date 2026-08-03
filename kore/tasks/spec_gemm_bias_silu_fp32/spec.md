# Specification: `gemm_bias_silu` (fp32)

Multiply two matrices, add a per-column bias, then apply SiLU.

## Definition

```
y[m, n] = silu( sum over k of a[m, k] * b[k, n] + bias[n] ),  silu(t) = t * sigmoid(t)
```

## Inputs
- `a`: 2-D, shape `[M, K]`, dtype fp32, contiguous.
- `b`: 2-D, shape `[K, N]`, dtype fp32, contiguous.
- `bias`: 1-D, shape `[N]`, dtype fp32, contiguous.

## Output
- 2-D, shape `[M, N]`, dtype fp32. Allocate it yourself and return it.

## Requirements and pitfalls
- `a` is [M, K], `b` is [K, N], `bias` is [N], output is [M, N].
- Use tl.dot so Triton emits MFMA; do not hand-roll the inner product.
- Accumulate in fp32 and apply the bias and activation to the fp32 accumulator, before the cast back to the output dtype.
- The baseline is a compiler-fused epilogue GEMM, so the bias and the activation must stay in the same kernel as the accumulation.

## Required entry point

Your module MUST define this exact top-level function, with this name and this parameter order:

```python
def gemm_bias_silu(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor
```

It is called directly by the verifier. A module that does not define `gemm_bias_silu` fails to load, which scores zero regardless of what else it contains.

## How you are graded

1. The module must import and expose `gemm_bias_silu`.
2. `gemm_bias_silu` must reach at least **40 dB SNR** against the fp32 oracle on EVERY validation shape, including the one whose N is not a multiple of a power of two.
3. Only then is it timed, against the production baseline.

Write a real Triton kernel. Calling the framework op (`torch.nn.functional.*`, `torch.matmul`) as the implementation satisfies neither the intent nor the speed target.
