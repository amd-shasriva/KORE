# Specification: `gelu_tanh` (bf16)

Apply the tanh approximation of GELU elementwise.

## Definition

```
y[i] = 0.5 * x[i] * (1 + tanh(sqrt(2/pi) * (x[i] + 0.044715 * x[i]**3)))
```

## Inputs
- `x`: 2-D, shape `[M, N]`, dtype bf16, contiguous.

## Output
- 2-D, shape `[M, N]`, dtype bf16. Allocate it yourself and return it.

## Requirements and pitfalls
- Use the tanh approximation, NOT the exact erf form: the oracle is the tanh variant and the two differ by more than the SNR gate allows.
- sqrt(2/pi) is 0.7978845608028654.

## Required entry point

Your module MUST define this exact top-level function, with this name and this parameter order:

```python
def gelu_tanh(x: torch.Tensor) -> torch.Tensor
```

It is called directly by the verifier. A module that does not define `gelu_tanh` fails to load, which scores zero regardless of what else it contains.

## How you are graded

1. The module must import and expose `gelu_tanh`.
2. `gelu_tanh` must reach at least **30 dB SNR** against the fp32 oracle on EVERY validation shape, including the one whose N is not a multiple of a power of two.
3. Only then is it timed, against the production baseline.

Write a real Triton kernel. Calling the framework op (`torch.nn.functional.*`, `torch.matmul`) as the implementation satisfies neither the intent nor the speed target.
