import json
import os
import glob

labels_file = "data/RSICC/claude_labels/labels.jsonl"
sheets_dir = "data/RSICC/claude_labels/sheets"

labels = set()
with open(labels_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        labels.add(data['patch_id'])

sheet_files = glob.glob(os.path.join(sheets_dir, "*.json"))
sheet_patch_ids = set()

for sf in sheet_files:
    with open(sf, 'r') as f:
        data = json.load(f)
        for patch_id in data['ids']:
            sheet_patch_ids.add(patch_id)

missing_from_labels = sheet_patch_ids - labels
print(f"Missing from labels: {len(missing_from_labels)}")

missing_files = []
for patch_id in missing_from_labels:
    before = f"data/FINAL/accepted/{patch_id}/before.png"
    after = f"data/FINAL/accepted/{patch_id}/after.png"
    if not (os.path.exists(before) and os.path.exists(after)):
        missing_files.append(patch_id)

print(f"Missing from disk: {len(missing_files)}")
