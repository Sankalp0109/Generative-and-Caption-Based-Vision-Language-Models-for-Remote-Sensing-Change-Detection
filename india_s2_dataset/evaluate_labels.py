import json

claude_file = "data/RSICC/claude_labels/labels.jsonl"
gold_file = "data/RSICC/gold100/visibility_labels.jsonl"

claude_labels = {}
with open(claude_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        # Keep the latest label as per instructions: "If a pair appears twice, the later line counts."
        claude_labels[data['patch_id']] = data['label']

gold_labels = {}
with open(gold_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        gold_labels[data['patch_id']] = data['label']

print(f"Total Claude labels: {len(claude_labels)}")
print(f"Total Gold labels: {len(gold_labels)}")

correct = 0
total = 0
mismatches = []

for patch_id, gold_label in gold_labels.items():
    if patch_id in claude_labels:
        total += 1
        if claude_labels[patch_id] == gold_label:
            correct += 1
        else:
            mismatches.append((patch_id, gold_label, claude_labels[patch_id]))

print(f"\nEvaluated: {total}")
if total > 0:
    print(f"Accuracy: {correct / total * 100:.2f}% ({correct}/{total})")

print("\nMismatches (Gold vs Claude):")
for m in mismatches:
    print(f"{m[0]}: Gold={m[1]}, Claude={m[2]}")
