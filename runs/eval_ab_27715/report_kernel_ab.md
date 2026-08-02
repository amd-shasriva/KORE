# Checkpoint A/B on the held-out generalization scope

- **candidate**: `midtrain`
- **reference**: `base`
- **tasks scored (n)**: 34 of 34 in scope (45 held out, 11 excluded as contaminated)
- **budget**: 1 bench(es)/task, mode=parallel (matched)

## Funnel (per-task rate, Wilson 95% CI)

| stage | midtrain | base | delta | discordant | exact McNemar p |
| --- | --- | --- | --- | --- | --- |
| emitted FULL_KERNEL | 33/34 = 97.1% [85.1%, 99.5%] | 32/34 = 94.1% [80.9%, 98.4%] | 2.9% | 3 (2/1) | 1 |
| compiled | 4/34 = 11.8% [4.7%, 26.6%] | 32/34 = 94.1% [80.9%, 98.4%] | -82.4% | 30 (1/29) | 5.77e-08 |
| correct (SNR gate) | 0/34 = 0.0% [0.0%, 10.2%] | 23/34 = 67.6% [50.8%, 80.9%] | -67.6% | 23 (0/23) | 2.38e-07 |
| correct AND publication-timed | 0/34 = 0.0% [0.0%, 10.2%] | 12/34 = 35.3% [21.5%, 52.1%] | -35.3% | 12 (0/12) | 0.000488 |
| faster than its baseline | 0/34 = 0.0% [0.0%, 10.2%] | 4/34 = 11.8% [4.7%, 26.6%] | -11.8% | 4 (0/4) | 0.125 |

> **Infrastructure faults** (counted as failures above, but a statement about the node rather than the kernel): midtrain 0, base 1.

## fast_p (fraction of the whole scope, uncorrected denominator)

| p | midtrain | base | delta |
| --- | --- | --- | --- |
| 0 | 0.0% | 35.3% | -35.3% |
| 0.5 | 0.0% | 17.6% | -17.6% |
| 1 | 0.0% | 11.8% | -11.8% |
| 1.5 | 0.0% | 5.9% | -5.9% |
| 2 | 0.0% | 2.9% | -2.9% |

**Verdict: reference_better** - correctness 0 vs 23 of 34; 23 discordant pair(s); exact McNemar p=2.38e-07 (significant at alpha=0.05)
