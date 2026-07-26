#!/usr/bin/env python3
"""Compare V1 (TF-IDF) vs V2 (sentence embeddings) on the frozen test set."""

import csv
import json
import os
import sys
from collections import Counter

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "v1")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVALS_DIR = os.path.join(BASE_DIR, "evaluations", "evaluation-v2")
os.makedirs(EVALS_DIR, exist_ok=True)

ESC_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
V1_ESC_THRESHOLD = 0.20

def load_data():
    texts, intents, escs, tags_list = [], [], [], []
    with open(os.path.join(DATA_DIR, "test.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"].strip())
            intents.append(row["intent"].strip())
            escs.append(1 if row["escalation_required"].strip() == "true" else 0)
            tags_list.append(json.loads(row.get("tags", "[]")))
    return texts, intents, escs, tags_list

def load_v1_models():
    intent_pipeline = joblib.load(os.path.join(MODELS_DIR, "intent", "intent-v1", "pipeline.joblib"))
    esc_pipeline = joblib.load(os.path.join(MODELS_DIR, "escalation", "escalation-v1", "pipeline.joblib"))
    return intent_pipeline, esc_pipeline

def load_v2_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    intent_clf = joblib.load(os.path.join(MODELS_DIR, "intent", "intent-v2", "classifier.joblib"))
    esc_clf = joblib.load(os.path.join(MODELS_DIR, "escalation", "escalation-v2", "classifier.joblib"))
    return embedder, intent_clf, esc_clf

def evaluate_intent(predictions, true_intents, classes):
    acc = accuracy_score(true_intents, predictions)
    report = classification_report(true_intents, predictions, output_dict=True)
    return {"accuracy": round(float(acc), 4), "report": report}

def evaluate_escalation_thresholds(scores, true_escs):
    results = {}
    for t in ESC_THRESHOLDS:
        preds = (scores >= t).astype(int)
        pr, re, f1, _ = precision_recall_fscore_support(true_escs, preds, average="binary", pos_label=1, zero_division=0)
        cm = confusion_matrix(true_escs, preds, labels=[0, 1])
        results[str(t)] = {
            "precision": round(float(pr), 4),
            "recall": round(float(re), 4),
            "f1": round(float(f1), 4),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
        }
    return results

def evaluate_subset(true_vals, pred_vals, mask):
    if sum(mask) == 0:
        return None
    return round(float(accuracy_score([v for v, m in zip(true_vals, mask) if m], [v for v, m in zip(pred_vals, mask) if m])), 4)

def main():
    print("Loading data and models...")
    texts, true_intents, true_escs, tags_list = load_data()
    print(f"Test set: {len(texts)} examples")

    print("\n--- Loading V1 (TF-IDF) ---")
    v1_intent, v1_esc = load_v1_models()
    print("V1 models loaded")

    print("\n--- Loading V2 (Sentence Embeddings) ---")
    v2_embedder, v2_intent_clf, v2_esc_clf = load_v2_models()
    print("V2 models loaded")

    # --- V1 Predictions ---
    print("\nRunning V1 predictions...")
    v1_intent_preds = v1_intent.predict(texts)
    v1_intent_probs = v1_intent.predict_proba(texts)
    v1_esc_probs = v1_esc.predict_proba(texts)[:, 1]
    v1_esc_preds = (v1_esc_probs >= V1_ESC_THRESHOLD).astype(int)

    # --- V2 Predictions ---
    print("Running V2 predictions (embedding texts)...")
    embeddings = v2_embedder.encode(texts, show_progress_bar=True)
    v2_intent_preds = v2_intent_clf.predict(embeddings)
    v2_intent_probs = v2_intent_clf.predict_proba(embeddings)
    v2_esc_scores = v2_esc_clf.predict_proba(embeddings)[:, 1]

    # --- Intent Comparison ---
    print("\n" + "=" * 70)
    print("INTENT COMPARISON")
    print("=" * 70)
    v1_intent_eval = evaluate_intent(v1_intent_preds, true_intents, v1_intent.classes_)
    v2_intent_eval = evaluate_intent(v2_intent_preds, true_intents, v2_intent_clf.classes_)

    classes = sorted(set(true_intents))
    print(f"{'Class':>20} {'V1 F1':>8} {'V2 F1':>8} {'Change':>8}")
    print("-" * 48)
    for c in classes:
        v1_f1 = v1_intent_eval["report"].get(c, {}).get("f1-score", 0)
        v2_f1 = v2_intent_eval["report"].get(c, {}).get("f1-score", 0)
        change = v2_f1 - v1_f1
        print(f"{c:>20} {v1_f1:>8.4f} {v2_f1:>8.4f} {change:>+8.4f}")

    print(f"\n{'Accuracy':>20} {v1_intent_eval['accuracy']:>8.4f} {v2_intent_eval['accuracy']:>8.4f} {v2_intent_eval['accuracy']-v1_intent_eval['accuracy']:>+8.4f}")

    v1_macro = v1_intent_eval["report"]["macro avg"]["f1-score"]
    v2_macro = v2_intent_eval["report"]["macro avg"]["f1-score"]
    v1_weighted = v1_intent_eval["report"]["weighted avg"]["f1-score"]
    v2_weighted = v2_intent_eval["report"]["weighted avg"]["f1-score"]
    print(f"{'Macro F1':>20} {v1_macro:>8.4f} {v2_macro:>8.4f} {v2_macro-v1_macro:>+8.4f}")
    print(f"{'Weighted F1':>20} {v1_weighted:>8.4f} {v2_weighted:>8.4f} {v2_weighted-v1_weighted:>+8.4f}")

    # --- Escalation Comparison ---
    print("\n" + "=" * 70)
    print("ESCALATION COMPARISON")
    print("=" * 70)

    v1_esc_eval = evaluate_escalation_thresholds(v1_esc_probs, true_escs)
    v2_esc_eval = evaluate_escalation_thresholds(v2_esc_scores, true_escs)

    print(f"\n{'Threshold':>10} {'V1 Prec':>10} {'V1 Rec':>10} {'V1 F1':>10} {'V1 FP':>5} {'V1 FN':>5} | {'V2 Prec':>10} {'V2 Rec':>10} {'V2 F1':>10} {'V2 FP':>5} {'V2 FN':>5}")
    print("-" * 105)
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        v1r = v1_esc_eval[str(t)]
        v2r = v2_esc_eval[str(t)]
        print(f"{t:>10.2f} {v1r['precision']:>10.4f} {v1r['recall']:>10.4f} {v1r['f1']:>10.4f} {v1r['false_positives']:>5} {v1r['false_negatives']:>5} | {v2r['precision']:>10.4f} {v2r['recall']:>10.4f} {v2r['f1']:>10.4f} {v2r['false_positives']:>5} {v2r['false_negatives']:>5}")

    # Find best V2 threshold
    best_f1, best_t = 0, 0.20
    for t in ESC_THRESHOLDS:
        r = v2_esc_eval[str(t)]
        if r["f1"] > best_f1:
            best_f1 = r["f1"]
            best_t = t

    # Recommend V2 threshold: match or exceed V1 recall while minimizing FPs
    # V1@0.20: recall=0.82, FP=49. Pick V2 threshold with recall >=0.82 and lowest FP.
    v1_rec_at_20 = v1_esc_eval[str(V1_ESC_THRESHOLD)]["recall"]
    candidates = []
    for t in ESC_THRESHOLDS:
        r = v2_esc_eval[str(t)]
        if r["recall"] >= v1_rec_at_20:
            candidates.append((r["false_positives"], t))
    candidates.sort()
    v2_recommended = candidates[0][1] if candidates else 0.20

    print(f"\nV1 recommended threshold: {V1_ESC_THRESHOLD}")
    print(f"  Recall: {v1_esc_eval[str(V1_ESC_THRESHOLD)]['recall']:.4f}, FP: {v1_esc_eval[str(V1_ESC_THRESHOLD)]['false_positives']}, FN: {v1_esc_eval[str(V1_ESC_THRESHOLD)]['false_negatives']}")

    print(f"\nV2 recommended threshold: {v2_recommended}")
    print(f"  Recall: {v2_esc_eval[str(v2_recommended)]['recall']:.4f}, FP: {v2_esc_eval[str(v2_recommended)]['false_positives']}, FN: {v2_esc_eval[str(v2_recommended)]['false_negatives']}")

    print(f"\nV2 best F1 threshold: {best_t} (F1={best_f1:.4f})")

    # --- Subset Comparison ---
    print("\n" + "=" * 70)
    print("SPECIAL SUBSET COMPARISON")
    print("=" * 70)
    subsets = {
        "hinglish": [any("hinglish" in t for t in tl) for tl in tags_list],
        "confusion_pair": [any("confusion_pair" in t for t in tl) for tl in tags_list],
        "hard_negative_escalation": [any("hard_negative_escalation" in t for t in tl) for tl in tags_list],
        "multi_intent": [any("multi_intent" in t for t in tl) for tl in tags_list],
        "noisy": [any("noisy" in t for t in tl) for tl in tags_list],
    }
    print(f"\n{'Subset':>25} {'Count':>6} {'V1 Intent':>10} {'V2 Intent':>10} {'V1 Esc':>10} {'V2 Esc':>10}")
    print("-" * 75)
    for name, mask in subsets.items():
        count = sum(mask)
        v1_intent_acc = evaluate_subset(true_intents, v1_intent_preds, mask)
        v2_intent_acc = evaluate_subset(true_intents, v2_intent_preds, mask)
        v1_esc_acc = evaluate_subset(true_escs, v1_esc_preds, mask)
        v2_esc_preds_at_t = (v2_esc_scores >= v2_recommended).astype(int)
        v2_esc_acc = evaluate_subset(true_escs, v2_esc_preds_at_t, mask)
        print(f"{name:>25} {count:>6} {str(v1_intent_acc or 'N/A'):>10} {str(v2_intent_acc or 'N/A'):>10} {str(v1_esc_acc or 'N/A'):>10} {str(v2_esc_acc or 'N/A'):>10}")

    # --- Error Analysis ---
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS - V2 HIGH CONFIDENCE ERRORS")
    print("=" * 70)
    v2_classes = v2_intent_clf.classes_.tolist()
    errors = []
    for i in range(len(texts)):
        if true_intents[i] != v2_intent_preds[i] or true_escs[i] != int(v2_esc_scores[i] >= v2_recommended):
            errors.append({
                "message": texts[i][:200],
                "expected_intent": true_intents[i],
                "predicted_intent": v2_intent_preds[i],
                "intent_confidence": round(float(max(v2_intent_probs[i])), 4),
                "expected_escalation": bool(true_escs[i]),
                "predicted_escalation": bool(v2_esc_scores[i] >= v2_recommended),
                "escalation_probability": round(float(v2_esc_scores[i]), 4),
                "tags": tags_list[i],
            })
    # High confidence intent errors
    high_conf_errors = [e for e in errors if e["expected_intent"] != e["predicted_intent"] and e["intent_confidence"] > 0.7]
    print(f"\nHigh-confidence intent errors (conf > 0.7): {len(high_conf_errors)}")
    for e in high_conf_errors[:10]:
        print(f"  [{e['intent_confidence']:.2f}] {e['expected_intent']} -> {e['predicted_intent']}: {e['message'][:80]}...")

    # Confusion pairs
    from collections import Counter
    conf_pairs = Counter()
    for e in errors:
        if e["expected_intent"] != e["predicted_intent"]:
            conf_pairs[f"{e['expected_intent']} -> {e['predicted_intent']}"] += 1
    print(f"\nTop confusion pairs (V2):")
    for pair, count in conf_pairs.most_common(15):
        print(f"  {pair}: {count}")

    # --- Probability Distribution ---
    print("\n" + "=" * 70)
    print("PROBABILITY DISTRIBUTION - ESCALATION")
    print("=" * 70)
    true_escs_arr = np.array(true_escs)
    pos_probs_v1 = v1_esc_probs[true_escs_arr == 1]
    neg_probs_v1 = v1_esc_probs[true_escs_arr == 0]
    pos_probs_v2 = v2_esc_scores[true_escs_arr == 1]
    neg_probs_v2 = v2_esc_scores[true_escs_arr == 0]

    def percentile_stats(probs, name):
        if len(probs) == 0:
            return
        print(f"  {name}: mean={np.mean(probs):.4f}, median={np.median(probs):.4f}, "
              f"p25={np.percentile(probs, 25):.4f}, p75={np.percentile(probs, 75):.4f}")

    percentile_stats(pos_probs_v1, "V1 escalation-positive")
    percentile_stats(neg_probs_v1, "V1 escalation-negative")
    percentile_stats(pos_probs_v2, "V2 escalation-positive")
    percentile_stats(neg_probs_v2, "V2 escalation-negative")

    # Separation score: mean difference
    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0
    v1_sep = safe_mean(pos_probs_v1) - safe_mean(neg_probs_v1)
    v2_sep = safe_mean(pos_probs_v2) - safe_mean(neg_probs_v2)
    print(f"\n  V1 separation (mean pos - mean neg): {v1_sep:.4f}")
    print(f"  V2 separation (mean pos - mean neg): {v2_sep:.4f}")
    print(f"  V2 {'improves' if v2_sep > v1_sep else 'worsens'} separation by {abs(v2_sep - v1_sep):.4f}")

    # --- Save Results ---
    results = {
        "dataset_version": "dataset-v1",
        "v1": {
            "intent_model": "intent-v1",
            "escalation_model": "escalation-v1",
            "escalation_threshold": V1_ESC_THRESHOLD,
            "intent_metrics": {
                "accuracy": v1_intent_eval["accuracy"],
                "macro_f1": round(float(v1_macro), 4),
                "weighted_f1": round(float(v1_weighted), 4),
                "per_class_f1": {c: round(float(v1_intent_eval["report"][c]["f1-score"]), 4) for c in classes},
            },
            "escalation_metrics": v1_esc_eval,
        },
        "v2": {
            "intent_model": "intent-v2",
            "escalation_model": "escalation-v2",
            "embedding_model": "all-MiniLM-L6-v2",
            "escalation_threshold": v2_recommended,
            "intent_metrics": {
                "accuracy": v2_intent_eval["accuracy"],
                "macro_f1": round(float(v2_macro), 4),
                "weighted_f1": round(float(v2_weighted), 4),
                "per_class_f1": {c: round(float(v2_intent_eval["report"][c]["f1-score"]), 4) for c in classes},
            },
            "escalation_metrics": v2_esc_eval,
        },
        "comparison": {
            "intent_accuracy_change": round(v2_intent_eval["accuracy"] - v1_intent_eval["accuracy"], 4),
            "intent_macro_f1_change": round(v2_macro - v1_macro, 4),
            "escalation_v1_at_20": v1_esc_eval[str(V1_ESC_THRESHOLD)],
            "escalation_v2_recommended": v2_esc_eval[str(v2_recommended)],
        },
        "subsets": {},
    }
    for name, mask in subsets.items():
        results["subsets"][name] = {
            "count": int(sum(mask)),
            "v1_intent_accuracy": evaluate_subset(true_intents, v1_intent_preds, mask),
            "v2_intent_accuracy": evaluate_subset(true_intents, v2_intent_preds, mask),
        }

    eval_path = os.path.join(EVALS_DIR, "comparison_results.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {eval_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nIntent: V1 accuracy={v1_intent_eval['accuracy']:.4f} -> V2 accuracy={v2_intent_eval['accuracy']:.4f} ({v2_intent_eval['accuracy']-v1_intent_eval['accuracy']:+.4f})")
    print(f"Escalation: V1 @0.20 recall={v1_esc_eval[str(V1_ESC_THRESHOLD)]['recall']:.4f} -> V2 @{v2_recommended} recall={v2_esc_eval[str(v2_recommended)]['recall']:.4f}")
    print(f"Escalation FPs: V1={v1_esc_eval[str(V1_ESC_THRESHOLD)]['false_positives']} -> V2={v2_esc_eval[str(v2_recommended)]['false_positives']}")
    print(f"Escalation FNs: V1={v1_esc_eval[str(V1_ESC_THRESHOLD)]['false_negatives']} -> V2={v2_esc_eval[str(v2_recommended)]['false_negatives']}")

    print("\n\u2713 Comparison complete")

if __name__ == "__main__":
    main()
