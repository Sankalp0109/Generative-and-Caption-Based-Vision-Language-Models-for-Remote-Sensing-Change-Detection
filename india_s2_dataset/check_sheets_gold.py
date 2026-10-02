import json
import os
import glob

labels_file = "data/RSICC/claude_labels/labels.jsonl"
sheets_dir = "data/RSICC/claude_labels/sheets"
gold_file = "data/RSICC/gold100/visibility_labels.jsonl"

labels = {}
with open(labels_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        labels[data['patch_id']] = data

sheet_files = glob.glob(os.path.join(sheets_dir, "*.json"))
sheet_patch_ids = set()

for sf in sheet_files:
    with open(sf, 'r') as f:
        data = json.load(f)
        for patch_id in data['ids']:
            sheet_patch_ids.add(patch_id)

not_in_sheets = set(labels.keys()) - sheet_patch_ids
not_in_labels = sheet_patch_ids - set(labels.keys())

gold_patch_ids = set()
with open(gold_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        gold_patch_ids.add(data['patch_id'])

print("Are the 60 patches in labels but not in sheets exactly the gold patches?")
print(not_in_sheets == gold_patch_ids)

