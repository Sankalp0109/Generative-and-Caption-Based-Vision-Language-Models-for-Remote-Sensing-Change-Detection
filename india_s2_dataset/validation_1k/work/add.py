import json, sys, time
sid, codes = sys.argv[1], sys.argv[2]
ids = json.load(open(f"sheets/{sid}.json"))["ids"]
C = {"L": "land_use", "S": "seasonal", "N": "none", "H": "atmospheric", "X": "unusable"}
ent = {}
for tok in [t.strip() for t in codes.split("|") if t.strip()]:
    i, rest = tok.split(":", 1); c, _, cap = rest.partition(":"); ent[int(i)] = (c.strip().upper(), cap.strip())
assert sorted(ent) == list(range(len(ids))), f"missing {set(range(len(ids))) - set(ent)}"
with open("new_labels.jsonl", "a") as f:
    for i, p in enumerate(ids):
        c, cap = ent[i]
        f.write(json.dumps({"patch_id": p, "label": C[c], "caption": cap or None, "sheet": sid,
                            "labeler": "claude-visual", "ts": round(time.time())}) + "\n")
print(sid, "saved", len(ids))
