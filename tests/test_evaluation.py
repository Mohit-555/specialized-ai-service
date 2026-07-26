"""Tests for evaluation utilities shared by evaluate_v4 and compare_v2_v3_v4."""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.evaluate_v4 import (
    compute_ece, categorize_fn, make_serializable, SUBSET_TAGS, INTENT_CLASSES,
    ESC_THRESHOLDS,
)
from training.compare_v2_v3_v4 import (
    evaluate_escalation_thresholds, find_best_threshold,
)


def test_compute_ece_perfect():
    y_true = np.array([1, 1, 1, 1])
    y_prob = np.array([0.9, 0.85, 0.95, 0.8])
    ece = compute_ece(y_true, y_prob)
    assert 0.0 <= ece <= 1.0


def test_compute_ece_imperfect():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.9, 0.1])
    ece = compute_ece(y_true, y_prob)
    assert ece < 0.3


def test_compute_ece_all_same():
    y_true = np.array([1, 1, 1])
    y_prob = np.array([0.5, 0.5, 0.5])
    ece = compute_ece(y_true, y_prob)
    assert ece >= 0.0


def test_compute_ece_empty():
    with pytest.raises(ZeroDivisionError):
        compute_ece(np.array([]), np.array([]))


def test_categorize_fn_hinglish():
    cat = categorize_fn("mujhe bahut problem hai", [])
    assert cat == "hinglish"


def test_categorize_fn_time_persistent():
    cat = categorize_fn("I have been waiting for hours", [])
    assert cat == "time-persistent"


def test_categorize_fn_indirect_human():
    cat = categorize_fn("Can you help me with this issue", [])
    assert cat == "indirect human request"


def test_categorize_fn_security():
    cat = categorize_fn("This is a security vulnerability", [])
    assert cat == "security/account concern"


def test_categorize_fn_human_request_tag():
    cat = categorize_fn("Some random text", ["human_request"])
    assert cat == "indirect human request"


def test_categorize_fn_multintent_tag():
    cat = categorize_fn("Some text", ["multi_intent"])
    assert cat == "multi_intent"


def test_categorize_fn_account_tag():
    cat = categorize_fn("Some text", ["account"])
    assert cat == "security/account concern"


def test_categorize_fn_subtle():
    cat = categorize_fn("Just a normal query", [])
    assert cat == "subtle/unresolved"


def test_make_serializable_numpy_int():
    result = make_serializable(np.int64(42))
    assert result == 42
    assert isinstance(result, int)


def test_make_serializable_numpy_float():
    result = make_serializable(np.float64(3.14))
    assert result == 3.14
    assert isinstance(result, float)


def test_make_serializable_numpy_array():
    result = make_serializable(np.array([1, 2, 3]))
    assert result == [1, 2, 3]


def test_make_serializable_dict():
    data = {"a": np.int64(1), "b": np.float64(2.0)}
    result = make_serializable(data)
    assert result == {"a": 1, "b": 2.0}


def test_make_serializable_nested():
    data = {"x": [np.int64(1), {"y": np.float64(2.5)}]}
    result = make_serializable(data)
    assert result == {"x": [1, {"y": 2.5}]}


def test_subsets_defined():
    assert "hinglish" in SUBSET_TAGS
    assert "confusion_pair" in SUBSET_TAGS
    assert "multi_intent" in SUBSET_TAGS
    assert "noisy" in SUBSET_TAGS
    assert "negation" in SUBSET_TAGS
    assert "resolution_state" in SUBSET_TAGS
    assert "hard_negative_escalation" in SUBSET_TAGS


def test_intent_classes_complete():
    expected = [
        "account_issue", "complaint", "general_question", "human_request",
        "other", "pricing", "product_question", "refund", "sales", "technical_support",
    ]
    assert INTENT_CLASSES == expected


def test_esc_thresholds_range():
    assert len(ESC_THRESHOLDS) == 19
    assert ESC_THRESHOLDS[0] == 0.05
    assert ESC_THRESHOLDS[-1] == 0.95


def test_evaluate_v4_importable():
    from training.evaluate_v4 import evaluate_intent, evaluate_escalation, load_benchmark
    assert callable(evaluate_intent)
    assert callable(evaluate_escalation)
    assert callable(load_benchmark)


def test_compare_v2_v3_v4_importable():
    from training.compare_v2_v3_v4 import evaluate_escalation_thresholds, find_best_threshold, compute_ece
    assert callable(evaluate_escalation_thresholds)
    assert callable(find_best_threshold)
    assert callable(compute_ece)


def test_find_best_threshold():
    esc_eval = {str(t): {"f1": 0.5} for t in ESC_THRESHOLDS}
    esc_eval["0.15"]["f1"] = 0.85
    best_t, best_f1 = find_best_threshold(esc_eval)
    assert best_t == 0.15
    assert best_f1 == 0.85


def test_evaluate_escalation_thresholds():
    scores = np.array([0.1, 0.3, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    result = evaluate_escalation_thresholds(scores, labels)
    assert "0.5" in result
    assert "precision" in result["0.5"]


def test_benchmark_data_exists():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v3", "benchmark.csv")
    assert os.path.exists(path), "Benchmark CSV not found"


def test_benchmark_data_loadable():
    from training.evaluate_v4 import load_benchmark
    texts, intents, escs, tags = load_benchmark()
    assert len(texts) > 0
    assert len(intents) == len(texts)
    assert len(escs) == len(texts)
    assert len(tags) == len(texts)
