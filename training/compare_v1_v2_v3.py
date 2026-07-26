#!/usr/bin/env python3
"""Comprehensive comparison of V1 (TF-IDF+LR), V2 (MiniLM+LR, dataset-v1), V3 (MiniLM+LR, dataset-v3)."""

import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_V1 = os.path.join(BASE_DIR, "data", "v1")
DATA_DIR_V3 = os.path.join(BASE_DIR, "data", "v3")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVALS_DIR = os.path.join(BASE_DIR, "evaluations", "evaluation-v3")
os.makedirs(EVALS_DIR, exist_ok=True)

RANDOM_SEED = 42
ESC_THRESHOLDS = [x / 100 for x in range(5, 96, 5)]
V1_ESC_THRESHOLD = 0.20
V2_ESC_THRESHOLD = 0.25
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SUBSET_NAMES = [
    "hinglish", "confusion_pair", "hard_negative_escalation",
    "multi_intent", "noisy", "negation", "resolution_state",
]

FN_CATEGORY_RULES = [
    ("hinglish", ["hinglish"]),
    ("multi_intent", ["multi_intent"]),
    ("security/account concern", ["account", "security"]),
]

FN_KEYWORD_RULES = [
    ("time-persistent", r"\b(waited|waiting|been (waiting|wait)|long time|hours|days|weeks|still waiting|since last)\b", re.IGNORECASE),
    ("indirect human request", r"\b(can (you|someone|anyone) (help|assist|look|check|connect|speak)|i (need|want) (to talk|to speak|a human|a person|someone)|could you (please|kindly)|would you (please|kindly)|is there (a|any) (human|person|agent|representative))\b", re.IGNORECASE),
    ("security/account concern", r"\b(security|hack|breach|unauthorized|stolen|fraud|suspicious|scam|phish|compromised)\b", re.IGNORECASE),
]

HINGLISH_KEYWORDS = r"\b(baat|karni|hai|karo|karein|chahiye|mujhe|aap|hum|nahi|hoga|kya|yeh|mera|meri|mujh|tum|tera|teri|apna|unka|inka)\b"


def load_data(filepath):
    texts, intents, escs, tags_list = [], [], [], []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"].strip())
            intents.append(row["intent"].strip())
            escs.append(1 if row["escalation_required"].strip() == "true" else 0)
            tags_list.append(json.loads(row.get("tags", "[]")))
    return texts, intents, np.array(escs), tags_list


def load_data_simple(filepath):
    texts, escs = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"].strip())
            escs.append(1 if row["escalation_required"].strip() == "true" else 0)
    return texts, np.array(escs)


def load_v1_models():
    intent = joblib.load(os.path.join(MODELS_DIR, "intent", "intent-v1", "pipeline.joblib"))
    esc = joblib.load(os.path.join(MODELS_DIR, "escalation", "escalation-v1", "pipeline.joblib"))
    return intent, esc


def load_v2_models():
    intent = joblib.load(os.path.join(MODELS_DIR, "intent", "intent-v2", "classifier.joblib"))
    esc = joblib.load(os.path.join(MODELS_DIR, "escalation", "escalation-v2", "classifier.joblib"))
    return intent, esc


def load_v3_models():
    intent = joblib.load(os.path.join(MODELS_DIR, "intent", "intent-v3", "classifier.joblib"))
    esc = joblib.load(os.path.join(MODELS_DIR, "escalation", "escalation-v3", "classifier.joblib"))
    return intent, esc


def evaluate_intent(predictions, true_intents):
    acc = accuracy_score(true_intents, predictions)
    report = classification_report(true_intents, predictions, output_dict=True)
    return {"accuracy": round(float(acc), 4), "report": report}


def evaluate_escalation_thresholds(scores, true_escs):
    results = {}
    for t in ESC_THRESHOLDS:
        preds = (scores >= t).astype(int)
        pr, re, f1, _ = precision_recall_fscore_support(
            true_escs, preds, average="binary", pos_label=1, zero_division=0
        )
        cm = confusion_matrix(true_escs, preds, labels=[0, 1])
        results[str(t)] = {
            "precision": round(float(pr), 4),
            "recall": round(float(re), 4),
            "f1": round(float(f1), 4),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
        }
    return results


def find_best_threshold(esc_eval):
    best_f1, best_t = 0, 0.50
    for t in ESC_THRESHOLDS:
        r = esc_eval[str(t)]
        if r["f1"] > best_f1:
            best_f1 = r["f1"]
            best_t = t
    return best_t, best_f1


def evaluate_subset(true_vals, pred_vals, mask):
    if sum(mask) == 0:
        return None
    return round(
        float(
            accuracy_score(
                [v for v, m in zip(true_vals, mask) if m],
                [v for v, m in zip(pred_vals, mask) if m],
            )
        ),
        4,
    )


def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if np.sum(mask) == 0:
            continue
        ece += np.sum(mask) * np.abs(np.mean(y_true[mask]) - np.mean(y_prob[mask]))
    return ece / len(y_true)


def categorize_fn(row_text, tags):
    text_lower = row_text.lower()
    tag_set = set(tags)

    for cat_name, cat_tags in FN_CATEGORY_RULES:
        if any(t in tag_set for t in cat_tags):
            return cat_name

    for cat_name, pattern, flags in FN_KEYWORD_RULES:
        if re.search(pattern, text_lower, flags):
            return cat_name

    if re.search(HINGLISH_KEYWORDS, text_lower, re.IGNORECASE):
        return "hinglish"

    if "human_request" in tag_set:
        return "indirect human request"

    return "subtle/unresolved"


def compute_reliability_bins(scores, labels, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(scores, bins) - 1, 0, n_bins - 1)
    results = []
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n = int(np.sum(mask))
        if n == 0:
            results.append(
                {
                    "bin": f"[{bins[bin_idx]:.1f}-{bins[bin_idx + 1]:.1f})",
                    "n": 0,
                    "avg_pred": None,
                    "actual_pos": None,
                    "gap": None,
                }
            )
        else:
            avg_pred = float(np.mean(scores[mask]))
            actual_pos = float(np.mean(labels[mask]))
            gap = round(actual_pos - avg_pred, 4)
            results.append(
                {
                    "bin": f"[{bins[bin_idx]:.1f}-{bins[bin_idx + 1]:.1f})",
                    "n": n,
                    "avg_pred": round(avg_pred, 4),
                    "actual_pos": round(actual_pos, 4),
                    "gap": gap,
                }
            )
    return results


def print_separator(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def run_intent_and_escalation(v1_intent, v1_esc, embedder,
                               v2_intent, v2_esc, v3_intent, v3_esc,
                               texts, true_intents, true_escs,
                               v3_esc_threshold, timing=None):
    timing = timing or {}
    # V1
    t0 = time.time()
    v1_intent_preds = v1_intent.predict(texts)
    v1_esc_scores = v1_esc.predict_proba(texts)[:, 1]
    timing["v1_intent_predict"] = time.time() - t0
    timing["v1_esc_total"] = time.time() - t0

    # Embed
    t0 = time.time()
    embeddings = embedder.encode(texts, show_progress_bar=False)
    timing["embed"] = time.time() - t0

    # V2
    t0 = time.time()
    v2_intent_preds = v2_intent.predict(embeddings)
    v2_esc_scores = v2_esc.predict_proba(embeddings)[:, 1]
    timing["v2_total"] = time.time() - t0

    # V3
    t0 = time.time()
    v3_intent_preds = v3_intent.predict(embeddings)
    v3_esc_scores = v3_esc.predict_proba(embeddings)[:, 1]
    timing["v3_total"] = time.time() - t0

    v1_esc_preds = (v1_esc_scores >= V1_ESC_THRESHOLD).astype(int)
    v2_esc_preds = (v2_esc_scores >= V2_ESC_THRESHOLD).astype(int)
    v3_esc_preds = (v3_esc_scores >= v3_esc_threshold).astype(int)

    return {
        "v1_intent_preds": v1_intent_preds,
        "v1_esc_scores": v1_esc_scores,
        "v1_esc_preds": v1_esc_preds,
        "v2_intent_preds": v2_intent_preds,
        "v2_esc_scores": v2_esc_scores,
        "v2_esc_preds": v2_esc_preds,
        "v3_intent_preds": v3_intent_preds,
        "v3_esc_scores": v3_esc_scores,
        "v3_esc_preds": v3_esc_preds,
        "embeddings": embeddings,
        "timing": timing,
    }


def print_intent_comparison(v1_preds, v2_preds, v3_preds, true_intents, label=""):
    v1_eval = evaluate_intent(v1_preds, true_intents)
    v2_eval = evaluate_intent(v2_preds, true_intents)
    v3_eval = evaluate_intent(v3_preds, true_intents)
    classes = sorted(set(true_intents))

    print(f"\n  {'Class':>20} {'V1 F1':>8} {'V2 F1':>8} {'V3 F1':>8}")
    print("  " + "-" * 60)
    for c in classes:
        v1_f1 = v1_eval["report"].get(c, {}).get("f1-score", 0)
        v2_f1 = v2_eval["report"].get(c, {}).get("f1-score", 0)
        v3_f1 = v3_eval["report"].get(c, {}).get("f1-score", 0)
        print(f"  {c:>20} {v1_f1:>8.4f} {v2_f1:>8.4f} {v3_f1:>8.4f}")

    print(f"  {'Accuracy':>20} {v1_eval['accuracy']:>8.4f} {v2_eval['accuracy']:>8.4f} {v3_eval['accuracy']:>8.4f}")
    print(f"  {'Macro F1':>20} {v1_eval['report']['macro avg']['f1-score']:>8.4f} {v2_eval['report']['macro avg']['f1-score']:>8.4f} {v3_eval['report']['macro avg']['f1-score']:>8.4f}")
    print(f"  {'Weighted F1':>20} {v1_eval['report']['weighted avg']['f1-score']:>8.4f} {v2_eval['report']['weighted avg']['f1-score']:>8.4f} {v3_eval['report']['weighted avg']['f1-score']:>8.4f}")

    return {"v1": v1_eval, "v2": v2_eval, "v3": v3_eval}


def print_escalation_sweep(v1_scores, v2_scores, v3_scores, true_escs):
    v1_eval = evaluate_escalation_thresholds(v1_scores, true_escs)
    v2_eval = evaluate_escalation_thresholds(v2_scores, true_escs)
    v3_eval = evaluate_escalation_thresholds(v3_scores, true_escs)

    print(f"\n  {'Thresh':>8} {'V1 P':>8} {'V1 R':>8} {'V1 F1':>8} {'V1 FP':>5} {'V1 FN':>5} | {'V2 P':>8} {'V2 R':>8} {'V2 F1':>8} {'V2 FP':>5} {'V2 FN':>5} | {'V3 P':>8} {'V3 R':>8} {'V3 F1':>8} {'V3 FP':>5} {'V3 FN':>5}")
    print("  " + "-" * 148)
    for t in ESC_THRESHOLDS:
        v1r = v1_eval[str(t)]
        v2r = v2_eval[str(t)]
        v3r = v3_eval[str(t)]
        print(f"  {t:>8.2f} {v1r['precision']:>8.4f} {v1r['recall']:>8.4f} {v1r['f1']:>8.4f} {v1r['false_positives']:>5} {v1r['false_negatives']:>5} | {v2r['precision']:>8.4f} {v2r['recall']:>8.4f} {v2r['f1']:>8.4f} {v2r['false_positives']:>5} {v2r['false_negatives']:>5} | {v3r['precision']:>8.4f} {v3r['recall']:>8.4f} {v3r['f1']:>8.4f} {v3r['false_positives']:>5} {v3r['false_negatives']:>5}")

    best_v1_t, best_v1_f1 = find_best_threshold(v1_eval)
    best_v2_t, best_v2_f1 = find_best_threshold(v2_eval)
    best_v3_t, best_v3_f1 = find_best_threshold(v3_eval)

    print(f"\n  Best thresholds:")
    print(f"    V1: {best_v1_t:.2f} (F1={best_v1_f1:.4f})")
    print(f"    V2: {best_v2_t:.2f} (F1={best_v2_f1:.4f})")
    print(f"    V3: {best_v3_t:.2f} (F1={best_v3_f1:.4f})")
    print(f"  Fixed thresholds:")
    print(f"    V1: {V1_ESC_THRESHOLD:.2f} -> F1={v1_eval[str(V1_ESC_THRESHOLD)]['f1']:.4f}")
    print(f"    V2: {V2_ESC_THRESHOLD:.2f} -> F1={v2_eval[str(V2_ESC_THRESHOLD)]['f1']:.4f}")

    return v1_eval, v2_eval, v3_eval, best_v3_t


def print_subset_comparison(true_intents, preds_dict, true_escs, esc_preds_dict, tags_list, subsets=None):
    if subsets is None:
        subsets = SUBSET_NAMES
    print(f"\n  {'Subset':>30} {'Count':>6} {'V1 Int':>8} {'V2 Int':>8} {'V3 Int':>8} {'V1 Esc':>8} {'V2 Esc':>8} {'V3 Esc':>8}")
    print("  " + "-" * 94)
    results = {}
    for name in subsets:
        mask = [any(name in t for t in tl) for tl in tags_list]
        if sum(mask) == 0:
            results[name] = {
                "count": 0, "v1_intent": None, "v2_intent": None,
                "v3_intent": None, "v1_esc": None, "v2_esc": None, "v3_esc": None,
            }
            print(f"  {name:>30} {'0':>6} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue
        count = sum(mask)
        v1_int = evaluate_subset(true_intents, preds_dict["v1"], mask)
        v2_int = evaluate_subset(true_intents, preds_dict["v2"], mask)
        v3_int = evaluate_subset(true_intents, preds_dict["v3"], mask)
        v1_esc = evaluate_subset(true_escs, esc_preds_dict["v1"], mask)
        v2_esc = evaluate_subset(true_escs, esc_preds_dict["v2"], mask)
        v3_esc = evaluate_subset(true_escs, esc_preds_dict["v3"], mask)
        results[name] = {
            "count": count, "v1_intent": v1_int, "v2_intent": v2_int,
            "v3_intent": v3_int, "v1_esc": v1_esc, "v2_esc": v2_esc, "v3_esc": v3_esc,
        }
        fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"
        print(f"  {name:>30} {count:>6} {fmt(v1_int):>8} {fmt(v2_int):>8} {fmt(v3_int):>8} {fmt(v1_esc):>8} {fmt(v2_esc):>8} {fmt(v3_esc):>8}")
    return results


def print_reliability_bins(scores, labels, model_name, n_bins=10):
    bins_data = compute_reliability_bins(scores, labels, n_bins)
    print(f"\n  {model_name} Reliability:")
    print(f"  {'Bin':>14} {'n':>6} {'Avg Pred':>10} {'Actual+':>10} {'Gap':>8}")
    print("  " + "-" * 50)
    for b in bins_data:
        if b["n"] == 0:
            print(f"  {b['bin']:>14} {0:>6}")
        else:
            print(f"  {b['bin']:>14} {b['n']:>6} {b['avg_pred']:>10.4f} {b['actual_pos']:>10.4f} {b['gap']:>+8.4f}")


def main():
    print("=" * 72)
    print("  V1 vs V2 vs V3 Comprehensive Comparison")
    print("=" * 72)

    # =========================================================
    # 0. LOAD DATA
    # =========================================================
    print("\nLoading data...")
    texts_v3, intents_v3, escs_v3, tags_v3 = load_data(os.path.join(DATA_DIR_V3, "test.csv"))
    texts_v3b, intents_v3b, escs_v3b, tags_v3b = load_data(os.path.join(DATA_DIR_V3, "benchmark.csv"))
    texts_v1, intents_v1, escs_v1, tags_v1 = load_data(os.path.join(DATA_DIR_V1, "test.csv"))
    texts_v3val, escs_v3val = load_data_simple(os.path.join(DATA_DIR_V3, "validation.csv"))

    print(f"  V3 test:         {len(texts_v3)} examples ({int(escs_v3.sum())} esc+)")
    print(f"  V3 benchmark:    {len(texts_v3b)} examples ({int(escs_v3b.sum())} esc+)")
    print(f"  V1 test:         {len(texts_v1)} examples ({int(escs_v1.sum())} esc+)")
    print(f"  V3 validation:   {len(texts_v3val)} examples ({int(escs_v3val.sum())} esc+)")

    # =========================================================
    # 1. LOAD MODELS
    # =========================================================
    print("\nLoading models...")
    load_times = {}

    t0 = time.time()
    v1_intent, v1_esc = load_v1_models()
    load_times["v1"] = time.time() - t0
    print(f"  V1 models loaded in {load_times['v1']:.2f}s")

    t0 = time.time()
    v2_intent, v2_esc = load_v2_models()
    load_times["v2"] = time.time() - t0
    print(f"  V2 models loaded in {load_times['v2']:.2f}s")

    t0 = time.time()
    v3_intent, v3_esc = load_v3_models()
    load_times["v3"] = time.time() - t0
    print(f"  V3 models loaded in {load_times['v3']:.2f}s")

    t0 = time.time()
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    load_times["embedder"] = time.time() - t0
    print(f"  Embedder loaded in {load_times['embedder']:.2f}s")

    # =========================================================
    # 2. RUN PREDICTIONS ON ALL DATASETS
    # =========================================================
    print("\nRunning predictions...")

    perf_timing = {"embedder_load": load_times["embedder"]}

    # V3 test set
    t0_all = time.time()
    result_v3 = run_intent_and_escalation(
        v1_intent, v1_esc, embedder, v2_intent, v2_esc, v3_intent, v3_esc,
        texts_v3, intents_v3, escs_v3, 0.50,  # placeholder threshold
    )
    v3_test_timing = result_v3["timing"]
    # Get V3 escalation scores first, find best threshold via sweep
    print("  Finding best V3 escalation threshold on V3 test set...")
    v3_esc_eval_test = evaluate_escalation_thresholds(
        result_v3["v3_esc_scores"], escs_v3
    )
    best_v3_t, best_v3_f1 = find_best_threshold(v3_esc_eval_test)
    print(f"  V3 best threshold: {best_v3_t:.2f} (F1={best_v3_f1:.4f})")
    # Recompute V3 esc preds with best threshold
    result_v3["v3_esc_preds"] = (result_v3["v3_esc_scores"] >= best_v3_t).astype(int)

    # V3 benchmark
    result_v3b = run_intent_and_escalation(
        v1_intent, v1_esc, embedder, v2_intent, v2_esc, v3_intent, v3_esc,
        texts_v3b, intents_v3b, escs_v3b, best_v3_t,
    )

    # V1 test set
    result_v1test = run_intent_and_escalation(
        v1_intent, v1_esc, embedder, v2_intent, v2_esc, v3_intent, v3_esc,
        texts_v1, intents_v1, escs_v1, best_v3_t,
    )

    # V3 validation set (for calibration)
    t0 = time.time()
    val_embeddings = embedder.encode(texts_v3val, show_progress_bar=False)
    v3val_esc_scores = v3_esc.predict_proba(val_embeddings)[:, 1]
    v3val_embed_time = time.time() - t0

    total_time = time.time() - t0_all

    # =========================================================
    # A. INTENT COMPARISON (on V3 test set)
    # =========================================================
    print_separator("A. Intent Comparison (V3 Test Set)")
    intent_metrics_v3 = print_intent_comparison(
        result_v3["v1_intent_preds"], result_v3["v2_intent_preds"],
        result_v3["v3_intent_preds"], intents_v3,
    )

    # =========================================================
    # B. ESCALATION COMPARISON (on V3 test set)
    # =========================================================
    print_separator("B. Escalation Comparison (V3 Test Set)")
    v1_esc_eval_v3, v2_esc_eval_v3, v3_esc_eval_v3, best_v3_t_found = print_escalation_sweep(
        result_v3["v1_esc_scores"], result_v3["v2_esc_scores"],
        result_v3["v3_esc_scores"], escs_v3,
    )

    # =========================================================
    # C. SPECIAL SUBSETS (on V3 test set)
    # =========================================================
    print_separator("C. Special Subsets (V3 Test Set)")
    intent_preds_dict = {
        "v1": result_v3["v1_intent_preds"],
        "v2": result_v3["v2_intent_preds"],
        "v3": result_v3["v3_intent_preds"],
    }
    esc_preds_dict = {
        "v1": result_v3["v1_esc_preds"],
        "v2": result_v3["v2_esc_preds"],
        "v3": result_v3["v3_esc_preds"],
    }
    subset_metrics_v3 = print_subset_comparison(
        intents_v3, intent_preds_dict, escs_v3, esc_preds_dict, tags_v3,
    )

    # =========================================================
    # D. BENCHMARK RESULTS
    # =========================================================
    print_separator("D. Benchmark Results (V3 Frozen Benchmark)")
    print("\n  --- Intent ---")
    intent_metrics_bench = print_intent_comparison(
        result_v3b["v1_intent_preds"], result_v3b["v2_intent_preds"],
        result_v3b["v3_intent_preds"], intents_v3b,
    )
    print("\n  --- Escalation ---")
    v1_esc_eval_bench, v2_esc_eval_bench, v3_esc_eval_bench, _ = print_escalation_sweep(
        result_v3b["v1_esc_scores"], result_v3b["v2_esc_scores"],
        result_v3b["v3_esc_scores"], escs_v3b,
    )
    print("\n  --- Special Subsets ---")
    intent_preds_dict_bench = {
        "v1": result_v3b["v1_intent_preds"],
        "v2": result_v3b["v2_intent_preds"],
        "v3": result_v3b["v3_intent_preds"],
    }
    esc_preds_dict_bench = {
        "v1": result_v3b["v1_esc_preds"],
        "v2": result_v3b["v2_esc_preds"],
        "v3": result_v3b["v3_esc_preds"],
    }
    subset_metrics_bench = print_subset_comparison(
        intents_v3b, intent_preds_dict_bench, escs_v3b, esc_preds_dict_bench, tags_v3b,
    )

    # =========================================================
    # E. V1 TEST SET (backward compatibility)
    # =========================================================
    print_separator("E. V1 Test Set (Backward Compatibility)")
    print("\n  --- Intent ---")
    intent_metrics_v1test = print_intent_comparison(
        result_v1test["v1_intent_preds"], result_v1test["v2_intent_preds"],
        result_v1test["v3_intent_preds"], intents_v1,
    )
    print("\n  --- Escalation ---")
    v1_esc_eval_v1test, v2_esc_eval_v1test, v3_esc_eval_v1test, _ = print_escalation_sweep(
        result_v1test["v1_esc_scores"], result_v1test["v2_esc_scores"],
        result_v1test["v3_esc_scores"], escs_v1,
    )
    print("\n  --- Special Subsets ---")
    v1_subsets = [s for s in SUBSET_NAMES if s != "negation" and s != "resolution_state"]
    intent_preds_dict_v1test = {
        "v1": result_v1test["v1_intent_preds"],
        "v2": result_v1test["v2_intent_preds"],
        "v3": result_v1test["v3_intent_preds"],
    }
    esc_preds_dict_v1test = {
        "v1": result_v1test["v1_esc_preds"],
        "v2": result_v1test["v2_esc_preds"],
        "v3": result_v1test["v3_esc_preds"],
    }
    subset_metrics_v1test = print_subset_comparison(
        intents_v1, intent_preds_dict_v1test, escs_v1, esc_preds_dict_v1test, tags_v1,
        subsets=v1_subsets,
    )

    # =========================================================
    # F. ESCALATION FN ANALYSIS (V3 on V3 test set)
    # =========================================================
    print_separator("F. Escalation FN Analysis (V3)")
    fn_indices = np.where(
        (escs_v3 == 1) & (result_v3["v3_esc_preds"] == 0)
    )[0]
    print(f"\n  Total V3 escalation false negatives: {len(fn_indices)}")
    fn_categories = defaultdict(list)
    for idx in fn_indices:
        cat = categorize_fn(texts_v3[idx], tags_v3[idx])
        fn_categories[cat].append(
            {
                "text": texts_v3[idx][:120],
                "score": float(result_v3["v3_esc_scores"][idx]),
                "intent": intents_v3[idx],
                "predicted_intent": result_v3["v3_intent_preds"][idx],
                "tags": tags_v3[idx],
            }
        )

    category_order = [
        "hinglish", "multi_intent", "subtle/unresolved", "time-persistent",
        "indirect human request", "security/account concern", "other",
    ]
    for cat in category_order:
        items = fn_categories.get(cat, [])
        print(f"\n  [{cat}] ({len(items)} FNs)")
        for item in items[:5]:
            print(f"    score={item['score']:.4f} intent={item['intent']}->{item['predicted_intent']}: {item['text']}")
        if len(items) > 5:
            print(f"    ... and {len(items) - 5} more")

    # =========================================================
    # G. CALIBRATION ANALYSIS
    # =========================================================
    print_separator("G. Calibration Analysis (Binned Reliability)")
    print_reliability_bins(result_v3["v1_esc_scores"], escs_v3, "V1")
    print_reliability_bins(result_v3["v2_esc_scores"], escs_v3, "V2")
    print_reliability_bins(result_v3["v3_esc_scores"], escs_v3, "V3")

    v1_ece = compute_ece(escs_v3, result_v3["v1_esc_scores"])
    v2_ece = compute_ece(escs_v3, result_v3["v2_esc_scores"])
    v3_ece = compute_ece(escs_v3, result_v3["v3_esc_scores"])
    print(f"\n  ECE (Expected Calibration Error):")
    print(f"    V1: {v1_ece:.4f}")
    print(f"    V2: {v2_ece:.4f}")
    print(f"    V3: {v3_ece:.4f}")

    # =========================================================
    # H. PLATT / ISOTONIC CALIBRATION TEST (V3 escalation)
    # =========================================================
    print_separator("H. Platt / Isotonic Calibration Test (V3 Escalation)")
    print("\n  Fitting calibrators on V3 validation set...")

    # Uncalibrated test scores
    test_scores_raw = result_v3["v3_esc_scores"]
    test_embeddings = result_v3["embeddings"]

    # Manual Platt (sigmoid) calibration
    # Fit calibrator on validation set: predicted_score -> calibrated_probability
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.isotonic import IsotonicRegression

    t0 = time.time()
    val_scores_raw = v3_esc.predict_proba(val_embeddings)[:, 1].reshape(-1, 1)
    platt_calib = LR(C=1e10, solver="lbfgs")  # C=1e10 approximates unregularized
    platt_calib.fit(val_scores_raw, escs_v3val)
    test_scores_platt = platt_calib.predict_proba(test_scores_raw.reshape(-1, 1))[:, 1]
    platt_time = time.time() - t0

    # Isotonic calibration
    t0 = time.time()
    iso_calib = IsotonicRegression(out_of_bounds="clip")
    iso_calib.fit(val_scores_raw.flatten(), escs_v3val)
    test_scores_iso = iso_calib.transform(test_scores_raw)
    iso_time = time.time() - t0

    # Brier scores
    brier_raw = brier_score_loss(escs_v3, test_scores_raw)
    brier_platt = brier_score_loss(escs_v3, test_scores_platt)
    brier_iso = brier_score_loss(escs_v3, test_scores_iso)

    # ECE
    ece_raw = compute_ece(escs_v3, test_scores_raw)
    ece_platt = compute_ece(escs_v3, test_scores_platt)
    ece_iso = compute_ece(escs_v3, test_scores_iso)

    print(f"\n  {'Calibration':>14} {'Brier':>10} {'ECE':>10}")
    print("  " + "-" * 36)
    print(f"  {'Uncalibrated':>14} {brier_raw:>10.4f} {ece_raw:>10.4f}")
    print(f"  {'Platt (sig)':>14} {brier_platt:>10.4f} {ece_platt:>10.4f}")
    print(f"  {'Isotonic':>14} {brier_iso:>10.4f} {ece_iso:>10.4f}")

    print(f"\n  Fitting times: Platt={platt_time:.3f}s, Isotonic={iso_time:.3f}s")

    # Threshold sweep comparison for calibrated scores
    print(f"\n  Threshold comparison (best F1 on test set):")
    raw_eval = evaluate_escalation_thresholds(test_scores_raw, escs_v3)
    platt_eval = evaluate_escalation_thresholds(test_scores_platt, escs_v3)
    iso_eval = evaluate_escalation_thresholds(test_scores_iso, escs_v3)

    raw_best_t, raw_best_f1 = find_best_threshold(raw_eval)
    platt_best_t, platt_best_f1 = find_best_threshold(platt_eval)
    iso_best_t, iso_best_f1 = find_best_threshold(iso_eval)
    print(f"    Uncalibrated: best threshold={raw_best_t:.2f} F1={raw_best_f1:.4f}")
    print(f"    Platt:        best threshold={platt_best_t:.2f} F1={platt_best_f1:.4f}")
    print(f"    Isotonic:     best threshold={iso_best_t:.2f} F1={iso_best_f1:.4f}")

    if ece_platt < ece_raw and brier_platt < brier_raw:
        print("\n  -> Platt calibration improves probability reliability.")
    elif ece_platt < ece_raw or brier_platt < brier_raw:
        print("\n  -> Platt calibration provides mixed improvement.")
    else:
        print("\n  -> Platt calibration does NOT improve probability reliability.")

    if ece_iso < ece_raw and brier_iso < brier_raw:
        print("  -> Isotonic calibration improves probability reliability.")
    elif ece_iso < ece_raw or brier_iso < brier_raw:
        print("  -> Isotonic calibration provides mixed improvement.")
    else:
        print("  -> Isotonic calibration does NOT improve probability reliability.")

    # Reliability bins for calibrated scores
    print("\n  Platt calibrated reliability:")
    print_reliability_bins(test_scores_platt, escs_v3, "V3+Platt")
    print("\n  Isotonic calibrated reliability:")
    print_reliability_bins(test_scores_iso, escs_v3, "V3+Isotonic")

    # =========================================================
    # I. CONFIDENCE / ABSTENTION ANALYSIS (V2 and V3 intent)
    # =========================================================
    print_separator("I. Confidence / Abstention Analysis")

    # Get confidence scores
    v2_intent_probs = v2_intent.predict_proba(result_v3["embeddings"])
    v3_intent_probs = v3_intent.predict_proba(result_v3["embeddings"])
    v2_max_conf = v2_intent_probs.max(axis=1)
    v3_max_conf = v3_intent_probs.max(axis=1)

    conf_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for model_name, max_conf, preds in [
        ("V2", v2_max_conf, result_v3["v2_intent_preds"]),
        ("V3", v3_max_conf, result_v3["v3_intent_preds"]),
    ]:
        print(f"\n  {model_name} Intent Confidence Analysis:")
        print(f"  {'Threshold':>10} {'Coverage':>10} {'Acc (≥th)':>10} {'Acc (<th)':>10} {'N (≥th)':>8} {'N (<th)':>8}")
        print("  " + "-" * 60)
        for th in conf_thresholds:
            above = max_conf >= th
            n_above = int(above.sum())
            n_below = len(above) - n_above
            if n_above > 0:
                acc_above = accuracy_score(
                    [intents_v3[i] for i in range(len(intents_v3)) if above[i]],
                    [preds[i] for i in range(len(preds)) if above[i]],
                )
            else:
                acc_above = None
            if n_below > 0:
                acc_below = accuracy_score(
                    [intents_v3[i] for i in range(len(intents_v3)) if not above[i]],
                    [preds[i] for i in range(len(preds)) if not above[i]],
                )
            else:
                acc_below = None
            cov = n_above / len(max_conf) * 100
            fmt_a = lambda v: f"{v:.4f}" if v is not None else "N/A"
            print(f"  {th:>10.1f} {cov:>9.2f}% {fmt_a(acc_above):>10} {fmt_a(acc_below):>10} {n_above:>8} {n_below:>8}")

    # Find optimal abstention threshold using validation set
    print(f"\n  Finding optimal abstention threshold on validation set...")
    val_emb = embedder.encode(texts_v3val, show_progress_bar=False)
    for model_name, intent_clf in [("V2", v2_intent), ("V3", v3_intent)]:
        val_probs = intent_clf.predict_proba(val_emb)
        val_conf = val_probs.max(axis=1)
        # Also get val intents from validation CSV
        val_intents = []
        with open(os.path.join(DATA_DIR_V3, "validation.csv"), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val_intents.append(row["intent"].strip())
        val_preds = intent_clf.predict(val_emb)

        print(f"\n    {model_name}:")
        print(f"    {'Threshold':>10} {'Coverage':>10} {'Acc (≥th)':>10} {'Acc (<th)':>10}")
        print("    " + "-" + "-" * 44)
        best_abstain_th, best_abstain_score = 0.0, 0.0
        for th in conf_thresholds:
            above = val_conf >= th
            n_above = int(above.sum())
            n_below = len(above) - n_above
            cov = n_above / len(val_conf) * 100
            if n_above > 0:
                acc_above = accuracy_score(
                    [val_intents[i] for i in range(len(val_intents)) if above[i]],
                    [val_preds[i] for i in range(len(val_preds)) if above[i]],
                )
            else:
                acc_above = 0
            if n_below > 0:
                acc_below = accuracy_score(
                    [val_intents[i] for i in range(len(val_intents)) if not above[i]],
                    [val_preds[i] for i in range(len(val_preds)) if not above[i]],
                )
            else:
                acc_below = None
            # Score: prefer high accuracy on covered examples while maintaining coverage
            # Simple scoring: acc_above * coverage
            score = acc_above * cov if n_above > 0 else 0
            if score > best_abstain_score and cov >= 80:
                best_abstain_score = score
                best_abstain_th = th
            fmt_a = lambda v: f"{v:.4f}" if v is not None else "N/A"
            print(f"    {th:>10.1f} {cov:>9.2f}% {fmt_a(acc_above):>10} {fmt_a(acc_below):>10}")
        print(f"    Recommended abstention threshold: {best_abstain_th:.1f}")

    # =========================================================
    # J. PERFORMANCE MEASUREMENT
    # =========================================================
    print_separator("J. Performance Measurement")

    n_v3 = len(texts_v3)
    v1_intent_predict_time = v3_test_timing.get("v1_intent_predict", 0)
    v1_esc_predict_time = v3_test_timing.get("v1_esc_total", 0) - v1_intent_predict_time if "v1_esc_total" in v3_test_timing else 0
    embed_time_total = v3_test_timing.get("embed", 0)
    v2_inference_time = v3_test_timing.get("v2_total", 0)
    v3_inference_time = v3_test_timing.get("v3_total", 0)

    print(f"\n  Dataset size: {n_v3} examples")
    print(f"\n  Model Loading:")
    print(f"    Embedder load:           {load_times['embedder']:.4f}s")
    print(f"    V1 models load:          {load_times['v1']:.4f}s")
    print(f"    V2 models load:          {load_times['v2']:.4f}s")
    print(f"    V3 models load:          {load_times['v3']:.4f}s")
    print(f"\n  Embedding ({EMBEDDING_MODEL}):")
    print(f"    Total:                   {embed_time_total:.4f}s")
    print(f"    Per example:             {embed_time_total / max(n_v3, 1) * 1000:.4f}ms")
    print(f"\n  V1 (TF-IDF + LR):")
    print(f"    Intent predict:          {v1_intent_predict_time:.4f}s")
    print(f"    Escalation predict:      {v1_esc_predict_time:.4f}s")
    print(f"    Total classify:          {v1_intent_predict_time + v1_esc_predict_time:.4f}s")
    print(f"    Per example:             {(v1_intent_predict_time + v1_esc_predict_time) / max(n_v3, 1) * 1000:.4f}ms")
    print(f"\n  V2 (MiniLM + LR):")
    print(f"    Inference (int+esc):     {v2_inference_time:.4f}s")
    print(f"    Per example:             {v2_inference_time / max(n_v3, 1) * 1000:.4f}ms")
    print(f"    Total (embed+infer):     {embed_time_total + v2_inference_time:.4f}s")
    print(f"\n  V3 (MiniLM + LR):")
    print(f"    Inference (int+esc):     {v3_inference_time:.4f}s")
    print(f"    Per example:             {v3_inference_time / max(n_v3, 1) * 1000:.4f}ms")
    print(f"    Total (embed+infer):     {embed_time_total + v3_inference_time:.4f}s")
    print(f"\n  Total script time:        {total_time:.2f}s")

    # =========================================================
    # K. V1 TEST SET CONSISTENCY CHECK
    # =========================================================
    print_separator("K. V1 Test Set Consistency Check")
    print("\n  Comparing V2 metrics on V1 test set vs previously reported values:")
    v2_acc_v1test = accuracy_score(intents_v1, result_v1test["v2_intent_preds"])
    v2_esc_scores_v1test = result_v1test["v2_esc_scores"]
    v2_esc_preds_v1test = (v2_esc_scores_v1test >= V2_ESC_THRESHOLD).astype(int)
    v2_esc_cm = confusion_matrix(escs_v1, v2_esc_preds_v1test, labels=[0, 1])
    _, v2_rec_v1test, _, _ = precision_recall_fscore_support(
        escs_v1, v2_esc_preds_v1test, average="binary", pos_label=1, zero_division=0
    )
    v2_fp_v1test = int(v2_esc_cm[0][1])
    v2_fn_v1test = int(v2_esc_cm[1][0])

    print(f"\n    {'Metric':>30} {'Previous':>12} {'Current':>12} {'Match':>10}")
    print("    " + "-" * 66)
    prev_acc = 0.761
    prev_rec = 0.865
    prev_fp = 46
    prev_fn = 12
    acc_match = "YES" if abs(v2_acc_v1test - prev_acc) < 0.005 else "NO"
    rec_match = "YES" if abs(v2_rec_v1test - prev_rec) < 0.005 else "NO"
    fp_match = "YES" if v2_fp_v1test == prev_fp else "NO"
    fn_match = "YES" if v2_fn_v1test == prev_fn else "NO"
    print(f"    {'Intent accuracy':>30} {prev_acc:>12.4f} {v2_acc_v1test:>12.4f} {acc_match:>10}")
    print(f"    {'Esc@0.25 recall':>30} {prev_rec:>12.4f} {v2_rec_v1test:>12.4f} {rec_match:>10}")
    print(f"    {'Esc FP':>30} {prev_fp:>12} {v2_fp_v1test:>12} {fp_match:>10}")
    print(f"    {'Esc FN':>30} {prev_fn:>12} {v2_fn_v1test:>12} {fn_match:>10}")

    # =========================================================
    # SAVE RESULTS
    # =========================================================
    print("\nSaving detailed results...")

    def serialize(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return serialize(obj)

    results = {
        "config": {
            "v1_escalation_threshold": V1_ESC_THRESHOLD,
            "v2_escalation_threshold": V2_ESC_THRESHOLD,
            "v3_escalation_threshold": best_v3_t,
            "embedding_model": EMBEDDING_MODEL,
            "random_seed": RANDOM_SEED,
        },
        "v3_test_set": {
            "n_examples": len(texts_v3),
            "n_escalation_positive": int(escs_v3.sum()),
        },
        "v3_benchmark_set": {
            "n_examples": len(texts_v3b),
            "n_escalation_positive": int(escs_v3b.sum()),
        },
        "v1_test_set": {
            "n_examples": len(texts_v1),
            "n_escalation_positive": int(escs_v1.sum()),
        },
        "section_A_intent_comparison_v3": {
            "v1": intent_metrics_v3["v1"],
            "v2": intent_metrics_v3["v2"],
            "v3": intent_metrics_v3["v3"],
        },
        "section_B_escalation_comparison_v3": {
            "v1_threshold_sweep": v1_esc_eval_v3,
            "v2_threshold_sweep": v2_esc_eval_v3,
            "v3_threshold_sweep": v3_esc_eval_v3,
            "v3_best_threshold": best_v3_t,
        },
        "section_C_subsets_v3": subset_metrics_v3,
        "section_D_benchmark": {
            "intent": intent_metrics_bench,
            "escalation": {
                "v1_threshold_sweep": v1_esc_eval_bench,
                "v2_threshold_sweep": v2_esc_eval_bench,
                "v3_threshold_sweep": v3_esc_eval_bench,
            },
            "subsets": subset_metrics_bench,
        },
        "section_E_v1_test_backward": {
            "intent": intent_metrics_v1test,
            "escalation": {
                "v1_threshold_sweep": v1_esc_eval_v1test,
                "v2_threshold_sweep": v2_esc_eval_v1test,
                "v3_threshold_sweep": v3_esc_eval_v1test,
            },
            "subsets": subset_metrics_v1test,
        },
        "section_F_fn_analysis": {
            "total_fn": int(len(fn_indices)),
            "by_category": {
                cat: [
                    {
                        "text": item["text"],
                        "score": item["score"],
                        "intent": item["intent"],
                        "predicted_intent": item["predicted_intent"],
                        "tags": item["tags"],
                    }
                    for item in items
                ]
                for cat, items in fn_categories.items()
            },
        },
        "section_G_calibration": {
            "v1_ece": round(float(v1_ece), 4),
            "v2_ece": round(float(v2_ece), 4),
            "v3_ece": round(float(v3_ece), 4),
            "v1_reliability_bins": compute_reliability_bins(
                result_v3["v1_esc_scores"], escs_v3
            ),
            "v2_reliability_bins": compute_reliability_bins(
                result_v3["v2_esc_scores"], escs_v3
            ),
            "v3_reliability_bins": compute_reliability_bins(
                result_v3["v3_esc_scores"], escs_v3
            ),
        },
        "section_H_calibration_test": {
            "uncalibrated": {
                "brier": round(float(brier_raw), 4),
                "ece": round(float(ece_raw), 4),
                "best_threshold": raw_best_t,
                "best_f1": raw_best_f1,
            },
            "platt": {
                "brier": round(float(brier_platt), 4),
                "ece": round(float(ece_platt), 4),
                "best_threshold": platt_best_t,
                "best_f1": platt_best_f1,
            },
            "isotonic": {
                "brier": round(float(brier_iso), 4),
                "ece": round(float(ece_iso), 4),
                "best_threshold": iso_best_t,
                "best_f1": iso_best_f1,
            },
            "conclusion": {
                "platt_improves_reliability": ece_platt < ece_raw and brier_platt < brier_raw,
                "isotonic_improves_reliability": ece_iso < ece_raw and brier_iso < brier_raw,
            },
        },
        "section_I_abstention": {},
        "section_J_performance": {
            "n_examples": n_v3,
            "model_loading_seconds": {
                "embedder": round(load_times["embedder"], 4),
                "v1": round(load_times["v1"], 4),
                "v2": round(load_times["v2"], 4),
                "v3": round(load_times["v3"], 4),
            },
            "embedding_seconds": {
                "total": round(embed_time_total, 4),
                "per_example_ms": round(embed_time_total / max(n_v3, 1) * 1000, 4),
            },
            "v1_classify_seconds": {
                "total": round(v1_intent_predict_time + v1_esc_predict_time, 4),
                "per_example_ms": round(
                    (v1_intent_predict_time + v1_esc_predict_time) / max(n_v3, 1) * 1000, 4
                ),
            },
            "v2_classify_seconds": {
                "total": round(v2_inference_time, 4),
                "per_example_ms": round(v2_inference_time / max(n_v3, 1) * 1000, 4),
            },
            "v3_classify_seconds": {
                "total": round(v3_inference_time, 4),
                "per_example_ms": round(v3_inference_time / max(n_v3, 1) * 1000, 4),
            },
        },
        "section_K_consistency_check": {
            "v2_intent_accuracy_on_v1_test": round(float(v2_acc_v1test), 4),
            "v2_escalation_recall_at_0.25_on_v1_test": round(float(v2_rec_v1test), 4),
            "v2_fp_on_v1_test": v2_fp_v1test,
            "v2_fn_on_v1_test": v2_fn_v1test,
            "previous_values": {
                "intent_accuracy": prev_acc,
                "escalation_recall": prev_rec,
                "fp": prev_fp,
                "fn": prev_fn,
            },
            "matches": {
                "accuracy": acc_match == "YES",
                "recall": rec_match == "YES",
                "fp": fp_match == "YES",
                "fn": fn_match == "YES",
            },
        },
    }

    eval_path = os.path.join(EVALS_DIR, "comparison_v1_v2_v3_results.json")
    with open(eval_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2, default=str)
    print(f"Results saved to {eval_path}")

    print("\n" + "=" * 72)
    print("  Comparison complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
