# `tests/` — the test suite

The default `pytest` command discovers both this top-level suite and every
package-local `kore/**/tests/test_*.py` module. It exercises science, reward,
data, RL math, breadth generators, and campaign wiring on CPU; use collection
itself as the live inventory rather than maintaining a count in prose.

```bash
pip install -e ".[test]"                              # CPU test environment
python -m pytest                                      # default CPU suite
python -m pytest --collect-only -q                    # live module/item inventory
python -m pytest tests                                # top-level CI split
python -m pytest kore                                 # package-local CI split
python -m pytest tests/test_campaign_wiring.py        # one file
```

The registered opt-in groups are `gpu`, `model`, `network`, and `dependency`.
Run one with `python -m pytest -m <group>` only after provisioning its resource.
`release` is separate and release-blocking; it validates licensing plus
regenerated/package artifacts and is intentionally excluded from the default.

---

## Coverage map

| Subsystem | Test files |
| --- | --- |
| Roofline / physics / P0 | `test_rooflines.py`, `test_roofline.py`, `test_p0_sol.py`, `test_reward_physics.py`, `test_whitebox_reward.py`, `test_profile_reward.py`, `test_dense_reward.py`, `test_pmc_counters.py` |
| Reward ladder / integrity | `test_reward_stats.py`, `test_timing_integrity.py`, `test_paradigm_credit.py` |
| Correctness oracle | `test_verify_equivalence.py`, `test_verifier_determinism.py`, `test_verify_rigor.py` |
| Tasks / ops | `test_genops.py`, `test_vendor_ops.py`, `test_augment.py`, `test_data_scale_audit.py`, `test_arch_normalize.py`, `test_coverage.py` |
| Data factory | `test_data.py`, `test_parallel_datagen.py`, `test_evolve.py`, `test_assemble.py`, `test_mixing.py`, `test_rejection.py`, `test_hard_negatives.py`, `test_onpolicy.py`, `test_gold_wins.py`, `test_gen_repair_quality.py`, `test_gen_wins_convergent.py`, `test_repair_dpo.py`, `test_reverify.py`, `test_corpus_quality.py`, `test_curate.py`, `test_hygiene.py`, `test_grounded_reasoning.py`, `test_synth_agentic.py`, `test_preference_quality.py` |
| Open-ended curriculum | `test_openended_proposer.py`, `test_openended_task_space.py`, `test_openended_archive.py`, `test_openended_coevolve.py`, `test_openended_controller.py`, `test_coevolve_distill.py` |
| Policy / RL | `test_rl_core.py`, `test_policy.py`, `test_grpo_fsdp.py`, `test_grpo_distill_hook.py`, `test_dynamic_steps.py`, `test_midtrain.py`, `test_distributed.py`, `test_frontier_ops_wiring.py`, `test_deep_cot_contract.py` |
| Value model | `test_value.py`, `test_value_replay_train.py` |
| Agent / transforms | `test_agent.py`, `test_agent_transform_discover.py` |
| Eval / gates | `test_eval.py`, `test_generalization.py`, `test_retention.py`, `test_champion.py`, `test_korebench.py`, `test_vs_opus.py` |
| Infra | `test_campaign_wiring.py`, `test_obs.py`, `test_contract.py` |

`test_campaign_wiring.py` and `test_distributed.py` are the fastest confidence check that the orchestration and FSDP configuration are coherent. Resource-specific tests belong to an explicit marker group; the default remains CPU-only.
