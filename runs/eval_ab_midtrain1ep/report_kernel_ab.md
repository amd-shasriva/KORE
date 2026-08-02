# Checkpoint A/B on the held-out generalization scope

- **candidate**: `midtrain`
- **reference**: `base`
- **tasks scored (n)**: 34 of 34 in scope (45 held out, 11 excluded as contaminated)
- **budget**: 1 bench(es)/task, mode=parallel (matched)

## Funnel (per-task rate, Wilson 95% CI)

| stage | midtrain | base | delta | discordant | exact McNemar p |
| --- | --- | --- | --- | --- | --- |
| emitted FULL_KERNEL | 12/34 = 35.3% [21.5%, 52.1%] | 32/34 = 94.1% [80.9%, 98.4%] | -58.8% | 20 (0/20) | 1.91e-06 |
| compiled | 0/34 = 0.0% [0.0%, 10.2%] | 32/34 = 94.1% [80.9%, 98.4%] | -94.1% | 32 (0/32) | 4.66e-10 |
| correct (SNR gate) | 0/34 = 0.0% [0.0%, 10.2%] | 23/34 = 67.6% [50.8%, 80.9%] | -67.6% | 23 (0/23) | 2.38e-07 |
| correct AND publication-timed | 0/34 = 0.0% [0.0%, 10.2%] | 14/34 = 41.2% [26.4%, 57.8%] | -41.2% | 14 (0/14) | 0.000122 |
| faster than its baseline | 0/34 = 0.0% [0.0%, 10.2%] | 6/34 = 17.6% [8.3%, 33.5%] | -17.6% | 6 (0/6) | 0.0312 |

> **Infrastructure faults** (counted as failures above, but a statement about the node rather than the kernel): midtrain 0, base 1.

## fast_p (fraction of the whole scope, uncorrected denominator)

| p | midtrain | base | delta |
| --- | --- | --- | --- |
| 0 | 0.0% | 41.2% | -41.2% |
| 0.5 | 0.0% | 26.5% | -26.5% |
| 1 | 0.0% | 17.6% | -17.6% |
| 1.5 | 0.0% | 8.8% | -8.8% |
| 2 | 0.0% | 2.9% | -2.9% |

**Verdict: reference_better** - correctness 0 vs 23 of 34; 23 discordant pair(s); exact McNemar p=2.38e-07 (significant at alpha=0.05)
