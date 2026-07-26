#!/usr/bin/env python3
"""Validate the dataset splits for quality and consistency."""

import csv
import json
import os
import sys
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v1")

REQUIRED_INTENTS = [
    "general_question", "product_question", "pricing", "sales",
    "technical_support", "complaint", "refund", "account_issue",
    "human_request", "other"
]

VALID_INTENTS = set(REQUIRED_INTENTS)

def load_csv(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def validate_split(rows, split_name):
    errors = []
    warnings = []

    # Check missing/empty text
    for i, r in enumerate(rows):
        text = r.get("text", "").strip()
        if not text:
            errors.append(f"Row {i}: missing or empty text")

    # Check invalid intent
    for i, r in enumerate(rows):
        intent = r.get("intent", "")
        if intent not in VALID_INTENTS:
            errors.append(f"Row {i}: invalid intent '{intent}'")

    # Check invalid escalation label
    for i, r in enumerate(rows):
        esc = r.get("escalation_required", "")
        if esc not in ("true", "false"):
            errors.append(f"Row {i}: invalid escalation_required '{esc}'")

    # Check duplicates
    seen = set()
    for i, r in enumerate(rows):
        text = r.get("text", "").strip()
        if text in seen:
            warnings.append(f"Row {i}: duplicate text within {split_name}")
        seen.add(text)

    # Check intent distribution
    intent_counts = Counter(r["intent"] for r in rows)
    missing_intents = set(REQUIRED_INTENTS) - set(intent_counts.keys())
    if missing_intents:
        errors.append(f"Missing intents in {split_name}: {missing_intents}")

    # Check escalation distribution
    esc_counts = Counter(r["escalation_required"] for r in rows)
    if "true" not in esc_counts:
        warnings.append(f"No escalation-positive examples in {split_name}")
    if "false" not in esc_counts:
        warnings.append(f"No escalation-negative examples in {split_name}")

    return errors, warnings, intent_counts, esc_counts

def main():
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    splits = ["train", "validation", "test"]
    all_rows = {}
    all_errors = []
    all_warnings = []

    for split in splits:
        filepath = os.path.join(DATA_DIR, f"{split}.csv")
        if not os.path.exists(filepath):
            all_errors.append(f"{split}.csv not found!")
            continue
        rows = load_csv(filepath)
        all_rows[split] = rows
        errors, warnings, intent_counts, esc_counts = validate_split(rows, split)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

        print(f"\n--- {split.upper()} ({len(rows)} rows) ---")
        print(f"  Intent distribution: {dict(intent_counts)}")
        print(f"  Escalation distribution: {dict(esc_counts)}")
        if errors:
            print(f"  ERRORS ({len(errors)}):")
            for e in errors:
                print(f"    ✗ {e}")
        if warnings:
            print(f"  WARNINGS ({len(warnings)}):")
            for w in warnings:
                print(f"    ⚠ {w}")
        if not errors and not warnings:
            print("  ✓ Valid")

    # Cross-split overlap check
    if len(all_rows) == 3:
        print("\n--- CROSS-SPLIT OVERLAP ---")
        texts = {s: set(r["text"].strip() for r in rows) for s, rows in all_rows.items()}
        overlaps = {
            "train-val": texts["train"] & texts["validation"],
            "train-test": texts["train"] & texts["test"],
            "val-test": texts["validation"] & texts["test"],
        }
        for pair, overlap_texts in overlaps.items():
            if overlap_texts:
                all_errors.append(f"Overlap in {pair}: {len(overlap_texts)} shared texts")
                print(f"  ✗ {pair}: {len(overlap_texts)} overlapping texts")
                for t in list(overlap_texts)[:5]:
                    print(f"    - {t[:80]}...")
            else:
                print(f"  ✓ {pair}: clean")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Total errors: {len(all_errors)}")
    print(f"  Total warnings: {len(all_warnings)}")
    if all_errors:
        print("\n  ✗ VALIDATION FAILED")
        for e in all_errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print("\n  ✓ VALIDATION PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()
