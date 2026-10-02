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

# Check sheets
sheet_files = glob.glob(os.path.join(sheets_dir, "*.json"))
sheet_patch_ids = set()

results = {
    "missing_in_labels": [],
    "sheet_label_mismatch": []
}

for sf in sheet_files:
    with open(sf, 'r') as f:
        data = json.load(f)
        sheet_name = data['sheet']
        for patch_id in data['ids']:
            sheet_patch_ids.add(patch_id)
            if patch_id not in labels:
                results["missing_in_labels"].append(patch_id)
            else:
                if labels[patch_id]['sheet'] != sheet_name and labels[patch_id]['sheet'] != "visibility":
                    results["sheet_label_mismatch"].append((patch_id, labels[patch_id]['sheet'], sheet_name))

not_in_sheets = set(labels.keys()) - sheet_patch_ids
print(f"Total patch_ids in sheets: {len(sheet_patch_ids)}")
print(f"Patch ids in labels but not in sheets: {len(not_in_sheets)}")
print("Results:")
for k, v in results.items():
    print(f"{k}: {len(v)}")
