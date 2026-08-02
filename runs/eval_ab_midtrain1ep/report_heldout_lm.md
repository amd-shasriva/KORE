# Held-out LM loss: `midtrain` vs `base`

## Held-out Triton kernel source (the target domain)

- **candidate**: `midtrain`
- **reference**: `base`
- **documents**: 34 (38,088 tokens, identical tokenization: True)

| arm | bits/token | perplexity |
| --- | --- | --- |
| midtrain | 3.0078 | 8.043 |
| base | 1.5787 | 2.987 |

- **paired per-document effect**: +1.4436 bits/token, 95% CI [+1.2735, +1.6055]
- **p-values**: wilcoxon 3.82e-07, sign 1.16e-10, bootstrap 0.0001
- **documents improved**: 0/34
- **corpus-level delta**: +1.4291 bits/token (perplexity ratio 2.6928x)

**Verdict: reference_better** - midtrain minus base: +1.4436 bits/token (95% CI [+1.2735, +1.6055], wilcoxon p=3.82e-07); 0/34 documents improved. Negative is better for midtrain.

## Held-out torch reference/oracle source (adjacent domain)

- **candidate**: `midtrain`
- **reference**: `base`
- **documents**: 34 (5,292 tokens, identical tokenization: True)

| arm | bits/token | perplexity |
| --- | --- | --- |
| midtrain | 4.6612 | 25.302 |
| base | 3.1206 | 8.697 |

- **paired per-document effect**: +2.3862 bits/token, 95% CI [+2.1678, +2.5881]
- **p-values**: wilcoxon 3.82e-07, sign 1.16e-10, bootstrap 0.0001
- **documents improved**: 0/34
- **corpus-level delta**: +1.5406 bits/token (perplexity ratio 2.9092x)

**Verdict: reference_better** - midtrain minus base: +2.3862 bits/token (95% CI [+2.1678, +2.5881], wilcoxon p=3.82e-07); 0/34 documents improved. Negative is better for midtrain.

## General-domain text (forgetting probe, NOT decontaminated)

- **candidate**: `midtrain`
- **reference**: `base`
- **documents**: 18 (578 tokens, identical tokenization: True)

| arm | bits/token | perplexity |
| --- | --- | --- |
| midtrain | 5.4058 | 42.396 |
| base | 2.8152 | 7.038 |

- **paired per-document effect**: +3.1240 bits/token, 95% CI [+2.4403, +3.8542]
- **p-values**: wilcoxon 0.000214, sign 7.63e-06, bootstrap 0.0001
- **documents improved**: 0/18
- **corpus-level delta**: +2.5907 bits/token (perplexity ratio 6.0238x)

**Verdict: reference_better** - midtrain minus base: +3.1240 bits/token (95% CI [+2.4403, +3.8542], wilcoxon p=0.000214); 0/18 documents improved. Negative is better for midtrain.
