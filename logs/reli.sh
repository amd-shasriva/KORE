#!/usr/bin/env bash
cd /root/Kore-rl/kore
for i in 1 2 3; do
  rm -rf runs_smoke/midtrain data_smoke/launch data_smoke/campaign_manifest.json
  HF_HUB_OFFLINE=1 KORE_LOG_COLOR=0 KORE_LOG_LEVEL=INFO KORE_RUN_DIR=logs/smoke PYTHONPATH=. \
    python scripts/run_campaign.py --model Qwen/Qwen3-14B --tasks rmsnorm_aiter,gemm_bf16 \
    --full-ft --stages midtrain --data-root data_smoke --midtrain-out runs_smoke/midtrain \
    > logs/reli_$i.log 2>&1
  rc=$?
  shards=$(ls runs_smoke/midtrain/*.safetensors 2>/dev/null | wc -l)
  warns=$(grep -c "not set to 'flash_attention_2'" logs/reli_$i.log)
  ckerr=$(grep -c "CheckpointError" logs/reli_$i.log)
  echo "run $i: exit=$rc shards=$shards sdpa_warns=$warns checkpoint_errors=$ckerr" >> logs/reli_summary.txt
done
echo "ALL DONE" >> logs/reli_summary.txt
