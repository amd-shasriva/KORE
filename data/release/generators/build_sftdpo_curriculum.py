import os, sys, json, traceback
os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HOME="/home/shasriva/.cache/huggingface", TOKENIZERS_PARALLELISM="false",
                  KORE_TOKENIZER_REVISION="40c069824f4251a91eefaf281ebe4c544efd3e18",
                  KORE_DECONTAM_BENCHMARK_ARTIFACT="/home/shasriva/kore_offline/benchmark_artifact.json",
                  KORE_SOURCE_METADATA="/home/shasriva/kore_offline/source_metadata.json")
KORE="/home/shasriva/Kore-RL/KORE"; sys.path.insert(0, KORE)
from pathlib import Path
DR=Path(KORE)/"data"/"b05factory"; REP="/home/shasriva/replay_hf"

# monkeypatch replay -> materialized
import kore.data.general_replay as gr
_c={}
def _load(kind,n,seed=0,use_hf=False,**kw):
    r=_c.get(kind)
    if r is None:
        p=f"{REP}/replay_{kind}.jsonl"; r=[json.loads(l) for l in open(p)] if os.path.exists(p) else []; _c[kind]=r
    return r[:n]
gr.load_general_replay=_load
import kore.data.assemble as asm
try: asm.load_general_replay=_load
except Exception: pass

# --- curriculum fold-in: all Tier 1-6 rows become the kernel_qa slice ---
import glob as _glob
_CUR=[]
for _p in ["/home/shasriva/kore_offline/curriculum_all.jsonl"]:
    try:
        for _l in open(_p):
            _l=_l.strip()
            if _l: _CUR.append(json.loads(_l))
    except Exception: pass
print("curriculum rows loaded:", len(_CUR), flush=True)
def _qa(tasks, teacher, n, seed=0, **kw):
    import random as _r; c=list(_CUR); _r.Random(seed).shuffle(c); return c[:max(0,int(n))]
import kore.data.gen_qa as _gq
_gq.generate_kernel_qa=_qa
try:
    asm.generate_kernel_qa=_qa
except Exception: pass


from kore.data.assemble import build_multicap_dataset, build_dpo_with_hard_negatives, summarize_multicap
from kore.data.build_datasets import dedup_by_source_hash, dedup_near_source
from kore.data.schemas import read_jsonl
from kore.data.teacher import make_teacher
from kore.data.rejection import stratified_rft_select
from kore.policy.configs import MultiCapSFTConfig
from kore.tasks.registry import all_tasks
try:
    from kore.tasks.registry import TRAIN_ARCH
except Exception:
    TRAIN_ARCH="gfx950"

# held-out eval ids
eval_ids=set()
try:
    from kore.tasks.registry import split_manifest_for_selection
    c=split_manifest_for_selection(all_tasks()); eval_ids=set(getattr(c,"eval_ids",[]) or [])
except Exception as e:
    print("split warn:", e, flush=True)
print(f"eval_ids={len(eval_ids)}", flush=True)

# mint gold wins + repair dpo (idempotent)
try:
    from kore.data.gold_wins import mint_gold_wins; mint_gold_wins(DR, cap=3000, seed=0, arch=TRAIN_ARCH)
    from kore.data.repair_dpo import mint_repair_dpo; mint_repair_dpo(DR, cap=8000, seed=0, arch=TRAIN_ARCH)
except Exception as e:
    print("mint warn:", e, flush=True)

raw=[]
for sub in ("repair","wins","groups"):
    d=DR/sub
    if d.exists():
        for p in sorted(d.glob("*.jsonl")):
            try: raw += read_jsonl(p, typed=True, mode="legacy_quarantine")
            except Exception as e: print("read warn",p.name,e, flush=True)
raw=dedup_by_source_hash(raw)
print(f"raw={len(raw)}", flush=True)

def rd(r): return r if isinstance(r,dict) else getattr(r,"__dict__",{})
def rtype(r):
    t=type(r).__name__.lower()
    if "win" in t: return "win"
    if "repair" in t: return "repair"
    if "group" in t or "ranked" in t: return "ranked_group"
    d=rd(r)
    if "candidates" in d or "preferences" in d: return "ranked_group"
    if "failure_class" in d or "broken_source" in d: return "repair"
    if "final_source" in d or "speedup" in d: return "win"
    return "other"
def hid(r):
    tid=rd(r).get("task_id"); return bool(tid and tid in eval_ids)

train=[r for r in raw if not hid(r)]
train_tasks=[t for t in all_tasks() if getattr(t,"task_id",None) not in eval_ids]
kernel_records=[r for r in train if rtype(r) in ("repair","win")]
group_records=[r for r in train if rtype(r)=="ranked_group"]
print(f"train={len(train)} kernel={len(kernel_records)} groups={len(group_records)} tasks={len(train_tasks)}", flush=True)

wins=[r for r in kernel_records if rtype(r)=="win"]; nonw=[r for r in kernel_records if rtype(r)!="win"]
if wins: wins=dedup_near_source(wins, per_fingerprint_cap=1)
kernel_records=nonw+wins
repairs=[r for r in kernel_records if rtype(r)=="repair"]; wins=[r for r in kernel_records if rtype(r)=="win"]
try:
    kept,rep=stratified_rft_select(wins, tau=1.0, per_task_frac_cap=0.34, seed=0); kernel_records=repairs+list(kept)
    print(f"rft kept={len(kept)}/{len(wins)} wins", flush=True)
except Exception as e:
    print("rft warn:", e, flush=True)

cfg=MultiCapSFTConfig()
if hasattr(cfg,"frac_kernel_qa"): cfg.frac_kernel_qa=0.20
print('calling build_multicap_dataset...', flush=True)
import traceback
try:
    rows=build_multicap_dataset(DR, train_tasks, make_teacher("stub"), cfg, total=95000, use_hf=True, kernel_records=kernel_records)
except BaseException as _e:
    print('BUILD_MULTICAP_EXC', repr(_e), flush=True); traceback.print_exc(); raise
(DR/"sft").mkdir(parents=True, exist_ok=True)
with open(DR/"sft"/"multicap.jsonl","w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
mix=summarize_multicap(rows)["fractions"]
print(f"SFT_OK rows={len(rows)} mix={mix}", flush=True)

print('calling build_dpo...', flush=True)
try:
    dpo=build_dpo_with_hard_negatives(DR, train_tasks, group_records=group_records, hard_target=0.12, seed=0)
except BaseException as _e:
    print('BUILD_DPO_EXC', repr(_e), flush=True); traceback.print_exc(); raise
(DR/"dpo").mkdir(parents=True, exist_ok=True)
with open(DR/"dpo"/"pairs.jsonl","w") as f:
    for r in dpo["rows"]: f.write(json.dumps(r)+"\n")
print(f"DPO_OK pairs={dpo['n_total']} hard={dpo['n_hard']} frac={dpo['n_hard']/max(1,dpo['n_total']):.3f}", flush=True)
print("DIRECT_DONE", flush=True)
