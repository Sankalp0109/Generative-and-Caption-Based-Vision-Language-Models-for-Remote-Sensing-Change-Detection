import json

labels_file = "data/RSICC/claude_labels/labels.jsonl"
target_ids = []
with open("data/RSICC/claude_labels/sheets/s0001.json", 'r') as f:
    target_ids = json.load(f)['ids']

labels = {}
with open(labels_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['patch_id'] in target_ids:
            labels[data['patch_id']] = data

for i, tid in enumerate(target_ids):
    if tid in labels:
        print(f"#{i} {tid}: {labels[tid]['label']} - {labels[tid]['caption']}")
    else:
        print(f"#{i} {tid}: MISSING FROM LABELS")
