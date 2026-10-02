import json
import os
from collections import defaultdict

labels_file = "data/RSICC/claude_labels/labels.jsonl"
sheets_dir = "data/RSICC/claude_labels/sheets"
images_dir = "data/FINAL/accepted"

# Read labels
labels = {}
with open(labels_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        labels[data['patch_id']] = data

print(f"Total unique patch_ids in labels: {len(labels)}")

# Check conditions
results = {
    "missing_images": [],
    "missing_before": [],
    "missing_after": [],
    "caption_mismatch": [],
    "invalid_label": []
}

valid_labels = {"seasonal", "atmospheric", "none", "land_use"}

for patch_id, data in labels.items():
    label = data.get("label")
    caption = data.get("caption")
    
    if label not in valid_labels:
        results["invalid_label"].append((patch_id, label))
        
    if label == "land_use" and not caption:
        results["caption_mismatch"].append((patch_id, label, "Missing caption"))
    elif label != "land_use" and caption is not None:
        results["caption_mismatch"].append((patch_id, label, "Has caption but not land_use"))
        
    patch_dir = os.path.join(images_dir, patch_id)
    if not os.path.exists(patch_dir):
        results["missing_images"].append(patch_id)
    else:
        if not os.path.exists(os.path.join(patch_dir, "before.png")):
            results["missing_before"].append(patch_id)
        if not os.path.exists(os.path.join(patch_dir, "after.png")):
            results["missing_after"].append(patch_id)

print("Results:")
for k, v in results.items():
    print(f"{k}: {len(v)}")
    if len(v) > 0 and len(v) < 20:
        print(f"  {v}")
