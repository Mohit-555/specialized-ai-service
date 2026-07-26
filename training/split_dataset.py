import csv
import json
import os
import random
from collections import Counter
from sklearn.model_selection import train_test_split

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v1")
INPUT_FILE = os.path.join(DATA_DIR, "dataset_v1_raw.csv")

# Read dataset
rows = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Total rows: {len(rows)}")

# Check for duplicates
seen_texts = set()
unique_rows = []
dupes = 0
for r in rows:
    t = r["text"].strip()
    if t in seen_texts:
        dupes += 1
        continue
    seen_texts.add(t)
    unique_rows.append(r)

print(f"Removed {dupes} exact duplicates")
print(f"Unique rows: {len(unique_rows)}")

rows = unique_rows

# Stratified split
intents = [r["intent"] for r in rows]

# First split: train vs temp (validation + test)
train_rows, temp_rows = train_test_split(
    rows, test_size=0.35, random_state=42, stratify=intents, shuffle=True
)

# Second split: validation vs test from temp
temp_intents = [r["intent"] for r in temp_rows]
val_rows, test_rows = train_test_split(
    temp_rows, test_size=0.57, random_state=42, stratify=temp_intents, shuffle=True
)

# Check split sizes
print(f"Train: {len(train_rows)} ({len(train_rows)/len(rows)*100:.1f}%)")
print(f"Validation: {len(val_rows)} ({len(val_rows)/len(rows)*100:.1f}%)")
print(f"Test: {len(test_rows)} ({len(test_rows)/len(rows)*100:.1f}%)")

# Check each intent appears in every split
def check_intents(data, name):
    counts = Counter(r["intent"] for r in data)
    print(f"\n{name} intent distribution:")
    for intent in sorted(counts):
        print(f"  {intent}: {counts[intent]}")

check_intents(train_rows, "Train")
check_intents(val_rows, "Validation")
check_intents(test_rows, "Test")

# Check escalation distribution
def check_esc(data, name):
    counts = Counter(r["escalation_required"] for r in data)
    print(f"\n{name} escalation distribution: {dict(counts)}")

check_esc(train_rows, "Train")
check_esc(val_rows, "Validation")
check_esc(test_rows, "Test")

# Check for overlap
train_texts = set(r["text"].strip() for r in train_rows)
val_texts = set(r["text"].strip() for r in val_rows)
test_texts = set(r["text"].strip() for r in test_rows)

train_val_overlap = train_texts & val_texts
train_test_overlap = train_texts & test_texts
val_test_overlap = val_texts & test_texts

print(f"\nTrain-Val overlap: {len(train_val_overlap)}")
print(f"Train-Test overlap: {len(train_test_overlap)}")
print(f"Val-Test overlap: {len(val_test_overlap)}")

if train_val_overlap or train_test_overlap or val_test_overlap:
    print("WARNING: Overlap detected!")
else:
    print("No overlap between splits. ✓")

# Write splits
fieldnames = rows[0].keys()

for split_name, split_rows in [("train", train_rows), ("validation", val_rows), ("test", test_rows)]:
    output_file = os.path.join(DATA_DIR, f"{split_name}.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)
    print(f"Written {output_file} ({len(split_rows)} rows)")

# Write combined dataset info
dataset_info = {
    "dataset_version": "dataset-v1",
    "total_examples": len(rows),
    "train_count": len(train_rows),
    "validation_count": len(val_rows),
    "test_count": len(test_rows),
    "train_ratio": round(len(train_rows)/len(rows), 3),
    "validation_ratio": round(len(val_rows)/len(rows), 3),
    "test_ratio": round(len(test_rows)/len(rows), 3),
    "intents": sorted(set(intents)),
    "intent_distribution": dict(Counter(intents)),
    "train_intent_distribution": dict(Counter(r["intent"] for r in train_rows)),
    "validation_intent_distribution": dict(Counter(r["intent"] for r in val_rows)),
    "test_intent_distribution": dict(Counter(r["intent"] for r in test_rows)),
    "escalation_distribution": dict(Counter(r["escalation_required"] for r in rows)),
    "overlap_train_val": len(train_val_overlap),
    "overlap_train_test": len(train_test_overlap),
    "overlap_val_test": len(val_test_overlap),
    "random_seed": 42,
}
info_file = os.path.join(DATA_DIR, "dataset_v1_info.json")
with open(info_file, "w") as f:
    json.dump(dataset_info, f, indent=2)
print(f"Dataset info written to {info_file}")
