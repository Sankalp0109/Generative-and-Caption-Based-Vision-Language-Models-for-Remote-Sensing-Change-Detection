import json
from collections import Counter

labels_file = "data/RSICC/claude_labels/labels.jsonl"
patch_ids = []
with open(labels_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        patch_ids.append(data['patch_id'])

counts = Counter(patch_ids)
duplicates = {k: v for k, v in counts.items() if v > 1}
print(f"Duplicates: {duplicates}")
