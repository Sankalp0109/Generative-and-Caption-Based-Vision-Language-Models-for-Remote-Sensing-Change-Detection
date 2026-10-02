import json
import os
import glob

labels_file = "data/RSICC/claude_labels/labels.jsonl"
sheets_dir = "data/RSICC/claude_labels/sheets"

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

print(f"Total patch_ids in sheets: {len(sheet_patch_ids)}")
print(f"Patch ids in labels but not in sheets ({len(not_in_sheets)}):")
print(list(not_in_sheets)[:10])

print(f"\nPatch ids in sheets but not in labels ({len(not_in_labels)}):")
print(list(not_in_labels)[:10])

