# `tests/` - the test suite

CPU-safe `pytest` tests (42 files) covering the science, reward, data, RL math, and campaign wiring. Tests import `kore.*` and avoid GPU work by design (roofline formulas, reward gating, family split, pure RL math, and wiring are all exercised without a device), so the suite runs on any box.

```bash
PYTHONPATH=. python -m pytest -q                                   # whole suite
PYTHONPATH=. python -m pytest tests/test_campaign_wiring.py -q     # one file
```

---

## Coverage map

| Subsystem | Test files |
| --- | --- |
| Roofline / physics / P0 | `test_rooflines.py`, `test_p0_sol.py`, `test_reward_physics.py`, `test_profile_reward.py` |
| Reward ladder / integrity | `test_reward_stats.py`, `test_timing_integrity.py` |
| Correctness oracle | `test_verify_equivalence.py`, `test_verifier_determinism.py` |
| Tasks / ops | `test_genops.py`, `test_vendor_ops.py`, `test_augment.py`, `test_data_scale_audit.py` |
| Data factory | `test_data.py`, `test_parallel_datagen.py`, `test_evolve.py`, `test_assemble.py`, `test_mixing.py`, `test_rejection.py`, `test_hard_negatives.py`, `test_onpolicy.py` |
| Open-ended curriculum | `test_openended_proposer.py`, `test_openended_task_space.py`, `test_openended_archive.py`, `test_openended_coevolve.py`, `test_openended_controller.py`, `test_coevolve_distill.py` |
| Policy / RL | `test_rl_core.py`, `test_policy.py`, `test_grpo_fsdp.py`, `test_grpo_distill_hook.py`, `test_dynamic_steps.py`, `test_midtrain.py`, `test_distributed.py` |
| Value model | `test_value.py` |
| Agent | `test_agent.py` |
| Eval / gates | `test_eval.py`, `test_generalization.py`, `test_retention.py`, `test_champion.py`, `test_korebench.py` |
| Infra | `test_campaign_wiring.py`, `test_obs.py` |

`test_campaign_wiring.py` and `test_distributed.py` are the fastest confidence check that the orchestration and FSDP config are coherent. Some tests that need real datasets/permissions may skip in a bare environment - that is expected.
