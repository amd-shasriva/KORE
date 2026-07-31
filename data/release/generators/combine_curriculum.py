import json, hashlib, collections, os
OUT = "/home/shasriva/kore_offline/curriculum_all.jsonl"
FILES = ["kernel_math", "kernel_qa", "kernel_concept", "kernel_reasoning", "kernel_evol", "kernel_distractor"]
seen = set(); rows = []; by = collections.Counter()
for name in FILES:
    p = "/home/shasriva/kore_offline/gen/" + name + ".jsonl"
    if not os.path.exists(p):
        continue
    for l in open(p):
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        msgs = d.get("messages", [])
        if len(msgs) < 3:
            continue
        asst = msgs[-1].get("content", "")
        if not isinstance(asst, str) or len(asst) < 40:
            continue
        q = msgs[-2].get("content", "")
        src = d.get("_source", name)
        h = (src, hashlib.sha256(q.encode()).hexdigest())
        if h in seen:
            continue
        seen.add(h); rows.append(d); by[src] += 1
tok = 0
with open(OUT, "w") as f:
    for d in rows:
        f.write(json.dumps(d) + "\n")
        for m in d["messages"]:
            c = m.get("content", "")
            if isinstance(c, str):
                tok += len(c) // 4
print("curriculum_all rows=" + str(len(rows)) + " by_source=" + str(dict(by)) +
      " approx_tokens=" + str(tok) + " (" + ("%.1f" % (tok / 1e6)) + "M) -> " + OUT)
