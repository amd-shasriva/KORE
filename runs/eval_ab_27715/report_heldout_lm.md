# Held-out LM loss: `midtrain` vs `base`

## Held-out Triton kernel source (the target domain)

- **candidate**: `midtrain`
- **reference**: `base`
- **documents**: 34 (38,088 tokens, identical tokenization: True)

| arm | bits/token | perplexity |
| --- | --- | --- |
| midtrain | 0.8181 | 1.763 |
| base | 1.5787 | 2.987 |

- **paired per-document effect**: -0.8368 bits/token, 95% CI [-0.9233, -0.7486]
- **p-values**: wilcoxon 3.82e-07, sign 1.16e-10, bootstrap 0.0001
- **documents improved**: 34/34
- **corpus-level delta**: -0.7606 bits/token (perplexity ratio 0.5902x)

**Verdict: candidate_better** - midtrain minus base: -0.8368 bits/token (95% CI [-0.9233, -0.7486], wilcoxon p=3.82e-07); 34/34 documents improved. Negative is better for midtrain.

## Held-out torch reference/oracle source (adjacent domain)

- **candidate**: `midtrain`
- **reference**: `base`
- **documents**: 34 (5,292 tokens, identical tokenization: True)

| arm | bits/token | perplexity |
| --- | --- | --- |
| midtrain | 1.3405 | 2.532 |
| base | 3.1206 | 8.697 |

- **paired per-document effect**: -3.2882 bits/token, 95% CI [-3.5350, -2.9796]
- **p-values**: wilcoxon 3.82e-07, sign 1.16e-10, bootstrap 0.0001
- **documents improved**: 34/34
- **corpus-level delta**: -1.7800 bits/token (perplexity ratio 0.2912x)

**Verdict: candidate_better** - midtrain minus base: -3.2882 bits/token (95% CI [-3.5350, -2.9796], wilcoxon p=3.82e-07); 34/34 documents improved. Negative is better for midtrain.

## General-domain text (forgetting probe, NOT decontaminated)

- **candidate**: `midtrain`
- **reference**: `base`
- **documents**: 18 (578 tokens, identical tokenization: True)

| arm | bits/token | perplexity |
| --- | --- | --- |
| midtrain | 3.0175 | 8.097 |
| base | 2.8152 | 7.038 |

- **paired per-document effect**: +0.3754 bits/token, 95% CI [+0.0589, +0.6743]
- **p-values**: wilcoxon 0.0366, sign 0.0309, bootstrap 0.0154
- **documents improved**: 4/18
- **corpus-level delta**: +0.2023 bits/token (perplexity ratio 1.1505x)

**Verdict: reference_better** - midtrain minus base: +0.3754 bits/token (95% CI [+0.0589, +0.6743], wilcoxon p=0.0366); 4/18 documents improved. Negative is better for midtrain.
