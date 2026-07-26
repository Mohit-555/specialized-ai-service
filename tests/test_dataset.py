import csv
import json
import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v1")

REQUIRED_INTENTS = [
    "general_question", "product_question", "pricing", "sales",
    "technical_support", "complaint", "refund", "account_issue",
    "human_request", "other"
]


def load_csv(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


@pytest.fixture(scope="module")
def splits():
    return {
        name: load_csv(os.path.join(DATA_DIR, f"{name}.csv"))
        for name in ["train", "validation", "test"]
    }


def test_all_splits_exist():
    for name in ["train.csv", "validation.csv", "test.csv"]:
        path = os.path.join(DATA_DIR, name)
        assert os.path.exists(path), f"Missing {path}"


def test_no_empty_text(splits):
    for split_name, rows in splits.items():
        for i, r in enumerate(rows):
            assert r["text"].strip(), f"Empty text in {split_name} at row {i}"


def test_valid_intents(splits):
    valid = set(REQUIRED_INTENTS)
    for split_name, rows in splits.items():
        for i, r in enumerate(rows):
            assert r["intent"] in valid, f"Invalid intent '{r['intent']}' in {split_name} at row {i}"


def test_valid_escalation_labels(splits):
    for split_name, rows in splits.items():
        for i, r in enumerate(rows):
            assert r["escalation_required"] in ("true", "false"), \
                f"Invalid escalation '{r['escalation_required']}' in {split_name} at row {i}"


def test_no_duplicates_within_splits(splits):
    for split_name, rows in splits.items():
        seen = set()
        for r in rows:
            text = r["text"].strip()
            assert text not in seen, f"Duplicate in {split_name}: {text[:50]}..."
            seen.add(text)


def test_no_leakage(splits):
    texts = {name: set(r["text"].strip() for r in rows) for name, rows in splits.items()}
    assert not (texts["train"] & texts["validation"]), "Train-validation overlap"
    assert not (texts["train"] & texts["test"]), "Train-test overlap"
    assert not (texts["validation"] & texts["test"]), "Validation-test overlap"


def test_all_intents_in_all_splits(splits):
    for split_name, rows in splits.items():
        intents = set(r["intent"] for r in rows)
        for intent in REQUIRED_INTENTS:
            assert intent in intents, f"Missing intent '{intent}' in {split_name}"


def test_escalation_distribution(splits):
    for split_name, rows in splits.items():
        counts = Counter(r["escalation_required"] for r in rows)
        assert counts.get("true", 0) > 0, f"No escalation=true in {split_name}"
        assert counts.get("false", 0) > 0, f"No escalation=false in {split_name}"


def test_intent_distribution(splits):
    for split_name, rows in splits.items():
        intents = Counter(r["intent"] for r in rows)
        for intent in REQUIRED_INTENTS:
            assert intents[intent] > 0, f"Zero examples of '{intent}' in {split_name}"


def test_dataset_size():
    total = 0
    for name in ["train.csv", "validation.csv", "test.csv"]:
        path = os.path.join(DATA_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f) - 1  # minus header
        total += count
    assert 1900 <= total <= 3100, f"Dataset size {total} outside expected range 1900-3100"
