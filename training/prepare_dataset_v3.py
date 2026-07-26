#!/usr/bin/env python3
"""Prepare dataset-v3: validate, deduplicate, split, check quality, create benchmark."""

import csv
import json
import os
import sys
import random
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_V1 = os.path.join(BASE_DIR, "data", "v1")
DATA_DIR_V3 = os.path.join(BASE_DIR, "data", "v3")
os.makedirs(DATA_DIR_V3, exist_ok=True)

REQUIRED_INTENTS = [
    "general_question", "product_question", "pricing", "sales",
    "technical_support", "complaint", "refund", "account_issue",
    "human_request", "other"
]
VAL_INTENTS = set(REQUIRED_INTENTS)

random.seed(42)

# ── Load raw v3 dataset ──
INPUT_FILE = os.path.join(DATA_DIR_V3, "dataset_v3_raw.csv")
if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found. Run create_dataset_v3.py first.")
    sys.exit(1)

rows = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Loaded {len(rows)} raw examples")

# ── 1. Basic validation ──
print("\n=== BASIC VALIDATION ===")
errors = []
for i, r in enumerate(rows):
    text = r.get("text", "").strip()
    if not text:
        errors.append(f"Row {i}: empty text")
    intent = r.get("intent", "")
    if intent not in VAL_INTENTS:
        errors.append(f"Row {i}: invalid intent '{intent}'")
    esc = r.get("escalation_required", "").strip()
    if esc not in ("true", "false"):
        errors.append(f"Row {i}: invalid escalation '{esc}'")
    tags_raw = r.get("tags", "[]")
    try:
        json.loads(tags_raw)
    except json.JSONDecodeError:
        errors.append(f"Row {i}: invalid tags JSON '{tags_raw}'")

if errors:
    print(f"  ERRORS: {len(errors)}")
    for e in errors[:20]:
        print(f"    ✗ {e}")
    sys.exit(1)
else:
    print("  ✓ All rows valid")

# ── 2. Exact duplicate detection ──
print("\n=== EXACT DUPLICATE DETECTION ===")
seen_texts = {}
dupes = []
for i, r in enumerate(rows):
    t = r["text"].strip()
    if t in seen_texts:
        dupes.append((i, seen_texts[t], t[:80]))
    seen_texts[t] = i

if dupes:
    print(f"  Found {len(dupes)} duplicates:")
    for i, orig_idx, text in dupes[:10]:
        print(f"    Row {i} duplicate of row {orig_idx}: '{text}'")
    # Deduplicate (keep first occurrence)
    seen = set()
    unique_rows = []
    for r in rows:
        t = r["text"].strip()
        if t not in seen:
            seen.add(t)
            unique_rows.append(r)
    print(f"  Removed {len(rows) - len(unique_rows)} duplicates")
    rows = unique_rows
else:
    print("  ✓ No exact duplicates")
    unique_rows = rows

# ── 3. Near-duplicate detection using sentence similarity ──
print("\n=== NEAR-DUPLICATE DETECTION ===")
print("  Computing embeddings for near-dup check...")
texts = [r["text"].strip() for r in unique_rows]
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=256)

# Check each text against all others (sample if too many)
n = len(texts)
if n > 1000:
    print(f"  Large dataset ({n}), sampling subset for pairwise check")
    sample_indices = random.sample(range(n), min(1000, n))
else:
    sample_indices = list(range(n))

sim_matrix = cosine_similarity(embeddings[sample_indices])
near_dupes = []
for i_idx, i in enumerate(sample_indices):
    for j_idx in range(i_idx + 1, len(sample_indices)):
        j = sample_indices[j_idx]
        if sim_matrix[i_idx][j_idx] > 0.95:
            ti, tj = texts[i][:100], texts[j][:100]
            if ti != tj:  # skip exact matches already removed
                near_dupes.append((i, j, float(sim_matrix[i_idx][j_idx]), ti, tj))

if near_dupes:
    print(f"  Found {len(near_dupes)} near-duplicate pairs (cosine > 0.95):")
    for i, j, sim, ti, tj in near_dupes[:10]:
        print(f"    [{sim:.3f}] ({i}) '{ti}'")
        print(f"            ({j}) '{tj}'")
    print(f"  Flagging {len(near_dupes)} pairs for review (not auto-deleting)")
else:
    print("  ✓ No near-duplicates found (cosine > 0.95)")

# ── 4. Leakage check vs V1 test set ──
print("\n=== LEAKAGE CHECK vs V1 TEST SET ===")
v1_test = []
with open(os.path.join(DATA_DIR_V1, "test.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        v1_test.append(row["text"].strip())

leaks = []
v1_test_set = set(v1_test)
for i, r in enumerate(unique_rows):
    t = r["text"].strip()
    if t in v1_test_set:
        leaks.append((i, t[:100]))

if leaks:
    print(f"  WARNING: {len(leaks)} examples from V1 test set found in V3!")
    for i, t in leaks[:5]:
        print(f"    Row {i}: '{t}'")
else:
    print("  ✓ No leakage of V1 test data into V3")

# ── 5. Near-duplicate V1 test set in V3 ──
print("\n=== NEAR-DUPLICATE LEAKAGE vs V1 TEST ===")
v1_embeddings = embedder.encode(v1_test, show_progress_bar=True, batch_size=256)
v3_text_subset = [r["text"].strip() for r in unique_rows]
v3_embeddings = embedder.encode(v3_text_subset, show_progress_bar=True, batch_size=256)

leak_sim = cosine_similarity(v3_embeddings, v1_embeddings)
close_leaks = []
for i in range(len(v3_text_subset)):
    for j in range(len(v1_test)):
        if leak_sim[i][j] > 0.92:
            close_leaks.append((i, j, float(leak_sim[i][j]),
                                v3_text_subset[i][:80], v1_test[j][:80]))
            break

if close_leaks:
    print(f"  WARNING: {len(close_leaks)} V3 examples near-duplicate with V1 test (cosine > 0.92)")
    for i, j, sim, t3, t1 in close_leaks[:10]:
        print(f"    [{sim:.3f}] V3: '{t3}'")
        print(f"            V1 test: '{t1}'")
else:
    print("  ✓ No near-duplicate leakage from V1 test")

# ── 6. Class and escalation distribution ──
print("\n=== DISTRIBUTION ===")
intent_counts = Counter(r["intent"] for r in unique_rows)
esc_counts = Counter(r["escalation_required"] for r in unique_rows)
print(f"  Total examples: {len(unique_rows)}")
print(f"  Intent distribution:")
for c in REQUIRED_INTENTS:
    print(f"    {c}: {intent_counts.get(c, 0)}")
print(f"  Escalation: true={esc_counts.get('true', 0)} false={esc_counts.get('false', 0)}")

missing = set(REQUIRED_INTENTS) - set(intent_counts.keys())
if missing:
    print(f"  ERROR: Missing intents: {missing}")
    sys.exit(1)

# ── 7. Stratified split (preserving V1 test set structure) ──
print("\n=== SPLIT ===")
# First, split off a frozen benchmark set (not used for training)
# Then split remaining into train/validation/test (test mirrors V1 test size)

# Separate out V1 test examples to maintain them in V3 test
v1_test_texts = set(v1_test)
v1_test_rows = [r for r in unique_rows if r["text"].strip() in v1_test_texts]
new_rows = [r for r in unique_rows if r["text"].strip() not in v1_test_texts]

print(f"  V1 test examples preserved: {len(v1_test_rows)}")
print(f"  New/remaining examples: {len(new_rows)}")

# Split new examples: 70/15/15
new_intents = [r["intent"] for r in new_rows]
if len(new_rows) >= 100:
    train_new, temp_new = train_test_split(
        new_rows, test_size=0.30, random_state=42,
        stratify=new_intents, shuffle=True
    )
    temp_intents = [r["intent"] for r in temp_new]
    val_new, test_new = train_test_split(
        temp_new, test_size=0.50, random_state=42,
        stratify=temp_intents, shuffle=True
    )
else:
    train_new, val_new, test_new = new_rows, [], []

# Combine with preserved V1 test
train_rows = train_new
val_rows = val_new
test_rows = test_new + v1_test_rows

print(f"  Train: {len(train_rows)} ({len(train_rows)/len(unique_rows)*100:.1f}%)")
print(f"  Val:   {len(val_rows)} ({len(val_rows)/len(unique_rows)*100:.1f}%)")
print(f"  Test:  {len(test_rows)} ({len(test_rows)/len(unique_rows)*100:.1f}%)")

# ── 8. Create frozen benchmark set ──
print("\n=== FROZEN BENCHMARK ===")

# Extract difficult examples from test set for benchmark
# Strategy: pull hard examples from test to form a small standalone benchmark
benchmark_sources = []
test_texts_set = set(r["text"].strip() for r in test_rows)

# Identify hard examples by tag
for r in test_rows:
    tags = json.loads(r.get("tags", "[]"))
    if any(t in tags for t in ["confusion_pair", "hard_negative_escalation",
                                 "multi_intent", "noisy", "hinglish",
                                 "negation", "resolution_state"]):
        benchmark_sources.append(r)

# Sample ~100 examples for benchmark (balanced)
random.shuffle(benchmark_sources)
benchmark_rows = benchmark_sources[:min(150, len(benchmark_sources))]

# Add some normal/standard examples for balance
standard_bench = [r for r in test_rows if r not in benchmark_sources and
                  "standard" in json.loads(r.get("tags", "[]"))]
random.shuffle(standard_bench)
benchmark_rows += standard_bench[:min(50, len(standard_bench))]

random.shuffle(benchmark_rows)
print(f"  Benchmark set: {len(benchmark_rows)} examples")
print(f"  (Extracted from test, will NOT be used for training)")

benchmark_texts = set(r["text"].strip() for r in benchmark_rows)

# Remove benchmark from test_rows
test_rows = [r for r in test_rows if r["text"].strip() not in benchmark_texts]
print(f"  Remaining test: {len(test_rows)}")

# ── 9. Write all splits ──
print("\n=== WRITING SPLITS ===")
fieldnames = unique_rows[0].keys()

split_sets = [
    ("train", train_rows),
    ("validation", val_rows),
    ("test", test_rows),
    ("benchmark", benchmark_rows),
]

split_info = {}
for name, split_rows in split_sets:
    out_path = os.path.join(DATA_DIR_V3, f"{name}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(split_rows)
    print(f"  {name}.csv: {len(split_rows)} rows")
    split_info[name] = {
        "count": len(split_rows),
        "intents": dict(Counter(r["intent"] for r in split_rows)),
        "escalation": dict(Counter(r["escalation_required"] for r in split_rows)),
    }

# ── 10. Dataset info JSON ──
dataset_info = {
    "dataset_version": "dataset-v3",
    "total_examples": len(unique_rows),
    "v1_examples_carried_over": sum(1 for r in unique_rows if r["text"].strip() in v1_test_texts),
    "random_seed": 42,
    "splits": split_info,
    "intent_distribution": dict(intent_counts),
    "escalation_distribution": dict(esc_counts),
    "near_duplicates_found": len(near_dupes),
    "v1_test_leakage": len(leaks),
    "v1_test_near_duplicate_leakage": len(close_leaks),
}

info_path = os.path.join(DATA_DIR_V3, "dataset_v3_info.json")
with open(info_path, "w") as f:
    json.dump(dataset_info, f, indent=2, default=str)
print(f"\n  Dataset info written to {info_path}")

# ── 11. Benchmark info ──
bench_info = {
    "benchmark_version": "benchmark-v3",
    "description": "Frozen benchmark set for V3 evaluation. Do NOT use for training.",
    "total_examples": len(benchmark_rows),
    "intent_distribution": dict(Counter(r["intent"] for r in benchmark_rows)),
    "escalation_distribution": dict(Counter(r["escalation_required"] for r in benchmark_rows)),
    "tag_distribution": dict(Counter(
        t for r in benchmark_rows for t in json.loads(r.get("tags", "[]"))
    )),
}
bench_info_path = os.path.join(DATA_DIR_V3, "benchmark_v3_info.json")
with open(bench_info_path, "w") as f:
    json.dump(bench_info, f, indent=2)
print(f"  Benchmark info written to {bench_info_path}")

# ── Cross-split overlap check ──
print("\n=== CROSS-SPLIT OVERLAP ===")
all_texts = {name: set(r["text"].strip() for r in rows) for name, rows in split_sets}
pairs = [("train", "validation"), ("train", "test"), ("train", "benchmark"),
         ("validation", "test"), ("validation", "benchmark"), ("test", "benchmark")]
clean = True
for a, b in pairs:
    overlap = all_texts[a] & all_texts[b]
    if overlap:
        print(f"  ✗ {a}-{b}: {len(overlap)} overlapping texts!")
        clean = False
if clean:
    print("  ✓ All splits clean, no overlap")
else:
    print("  ERROR: Overlap detected!")
    sys.exit(1)

print("\n=== DATASET-V3 READY ===")
