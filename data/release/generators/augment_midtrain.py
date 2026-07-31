import os, sys, json
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                  HF_HOME="/home/shasriva/.cache/huggingface")
sys.path.insert(0, "/home/shasriva/Kore-RL/KORE")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B",
        revision="40c069824f4251a91eefaf281ebe4c544efd3e18", local_files_only=True)

CORPUS = "/home/shasriva/Kore-RL/KORE/data/b05factory/midtrain/corpus.jsonl"
CUR = "/home/shasriva/kore_offline/curriculum_all.jsonl"
MAXTOK = 8192; MARGIN = 32
# explanatory tiers become pretraining docs (skip pure distractor/qa framing noise -> keep the knowledge)
KEEP = {"kernel_math", "kernel_concept", "kernel_reasoning", "kernel_evol", "kernel_qa"}


def chunk(text):
    ids = tok.encode(text, add_special_tokens=False)
    win = MAXTOK - MARGIN; out = []
    for i in range(0, len(ids), win):
        s = tok.decode(ids[i:i + win], skip_special_tokens=True).strip()
        if s:
            out.append(s)
    return out


# de-dup incoming chunks against themselves
import hashlib
seen = set(); added = 0; src_seen = 0
before = sum(1 for _ in open(CORPUS))
with open(CORPUS, "a") as f:
    for l in open(CUR):
        l = l.strip()
        if not l:
            continue
        d = json.loads(l); src = d.get("_source", "")
        if src not in KEEP:
            continue
        src_seen += 1
        msgs = d.get("messages", [])
        if len(msgs) < 3:
            continue
        text = msgs[-2].get("content", "") + "\n\n" + msgs[-1].get("content", "")
        for c in chunk(text):
            h = hashlib.sha256(c.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            f.write(json.dumps({"text": c, "source": "kernel_curriculum"}) + "\n"); added += 1
after = before + added
print("MIDTRAIN_AUG before=%d src_rows=%d appended_chunks=%d after=%d" % (before, src_seen, added, after))
