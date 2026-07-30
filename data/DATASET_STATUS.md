# KORE-RL Pre-Midtrain Dataset — Build Status (2026-07-30)

## Built artifacts (Crusoe: /home/shasriva/Kore-RL/KORE/data/b05factory/)
- sft/multicap.jsonl : 24,235 rows (kernel wins/gold/repair + 4,000 agentic tool-use); held-out-clean
- dpo/pairs.jsonl    : 96,675 preference pairs (11,601 hard negatives = 12%); held-out-clean
- Backups: /home/shasriva/kore_dataset_*.tgz , /home/shasriva/kore_frontier_backup_*.tgz

## Provenance
- Frontier win corpus: ~805 tasks at 3 vendor-beating wins (before reverify 828 -> after reverify 447 -> after evolve ~805); 843 evolve wins; 3,000 gold wins minted.
- Agentic: 4,000 synth tool-use trajectories reconstructed from verified records (held-out-excluded).
- Held-out: 45 registry eval tasks + 34 quarantined leakage tasks EXCLUDED (verified 0 leakage in sft/dpo).
- Tier: kernel-focused (frontier). General-retention replay slice DEFERRED (requires HF egress from compute nodes, currently blocked; add via run_campaign build --use-hf once egress/dedicated-QoS fixed).

## Code changes this session
- kore/agent/tools.py: pmc tool returns rocprofv3-grounded PROFILE->DIAGNOSE->TRANSFORM (gated KORE_AGENTIC_GROUND).
- scripts/run_campaign.py: build read is campaign-mode-aware (dev = tolerant legacy_quarantine).
