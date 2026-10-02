import json

labels_file = "data/RSICC/claude_labels/labels.jsonl"
target_ids = ["madhya_pradesh_082_r3c2", "jammu_and_kashmir_011_r0c0", "rajasthan_t1_031_r2c0", "gujarat_017_r2c1", "jammu_and_kashmir_011_r2c1", "gujarat_t1_001_r3c3", "madhya_pradesh_t1_012_r3c0", "mizoram_005_r1c2", "bihar_003_r2c0", "meghalaya_002_r2c0", "madhya_pradesh_006_r0c0", "assam_t1_012_r0c2"]

labels = {}
with open(labels_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['patch_id'] in target_ids:
            labels[data['patch_id']] = data

for tid in target_ids:
    print(f"{tid}: {labels[tid]['label']} - {labels[tid]['caption']}")
