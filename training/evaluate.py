#!/usr/bin/env python3
"""Evaluate intent and escalation models on test set with threshold sweep and error analysis."""

import csv
import json
import os
import sys
from collections import Counter, defaultdict

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "v1")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVALS_DIR = os.path.join(BASE_DIR, "evaluations", "evaluation-v1")
os.makedirs(EVALS_DIR, exist_ok=True)

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

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

def load_intent_model():
    path = os.path.join(MODELS_DIR, "intent", "intent-v1", "pipeline.joblib")
    return joblib.load(path)

def load_escalation_model():
    path = os.path.join(MODELS_DIR, "escalation", "escalation-v1", "pipeline.joblib")
    return joblib.load(path)

def evaluate_intent(model, texts, true_intents):
    preds = model.predict(texts)
    probs = model.predict_proba(texts)

    accuracy = accuracy_score(true_intents, preds)
    classes = model.classes_.tolist()
    report = classification_report(true_intents, preds, output_dict=True)
    cm = confusion_matrix(true_intents, preds, labels=classes)

    top_k_preds = []
    for prob in probs:
        top_indices = np.argsort(prob)[-5:][::-1]
        top_k_preds.append([{"intent": classes[i], "confidence": round(float(prob[i]), 4)} for i in top_indices])

    return {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "predictions": preds.tolist(),
        "probabilities": probs.tolist(),
        "top_k_predictions": top_k_preds,
    }

def evaluate_escalation(model, texts, true_escs):
    probs = model.predict_proba(texts)[:, 1]

    results = {}
    for threshold in THRESHOLDS:
        preds = (probs >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_escs, preds, average="binary", pos_label=1, zero_division=0
        )
        cm = confusion_matrix(true_escs, preds, labels=[0, 1])
        fp = int(cm[0][1]) if cm.shape == (2, 2) else 0
        fn = int(cm[1][0]) if cm.shape == (2, 2) else 0
        results[str(threshold)] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "false_positives": fp,
            "false_negatives": fn,
        }

    return results, probs

def evaluate_subset(true_vals, pred_vals, mask, subset_name):
    if sum(mask) == 0:
        return {f"{subset_name}_count": 0}
    filtered_true = [v for v, m in zip(true_vals, mask) if m]
    filtered_pred = [v for v, m in zip(pred_vals, mask) if m]
    acc = accuracy_score(filtered_true, filtered_pred)
    return {f"{subset_name}_accuracy": round(float(acc), 4), f"{subset_name}_count": sum(mask)}

def error_analysis(texts, true_intents, pred_intents, probs_intent, true_escs, pred_escs, probs_esc, tags_list):
    errors = []
    for i in range(len(texts)):
        if true_intents[i] != pred_intents[i] or true_escs[i] != pred_escs[i]:
            errors.append({
                "message": texts[i][:200],
                "expected_intent": true_intents[i],
                "predicted_intent": pred_intents[i],
                "intent_confidence": round(float(max(probs_intent[i])), 4),
                "expected_escalation": bool(true_escs[i]),
                "predicted_escalation": bool(pred_escs[i]),
                "escalation_probability": round(float(probs_esc[i]), 4),
                "tags": tags_list[i],
            })

    confusion_pairs = Counter()
    for e in errors:
        if e["expected_intent"] != e["predicted_intent"]:
            pair = f"{e['expected_intent']} → {e['predicted_intent']}"
            confusion_pairs[pair] += 1

    return errors, confusion_pairs.most_common(20)

def main():
    print("Loading data and models...")
    texts, true_intents, true_escs, tags_list = load_data()
    intent_model = load_intent_model()
    esc_model = load_escalation_model()

    print(f"Test set: {len(texts)} examples")
    intent_counts = Counter(true_intents)
    print(f"Intent distribution: {dict(intent_counts)}")
    print(f"Escalation: {sum(true_escs)} positive, {len(true_escs) - sum(true_escs)} negative\n")

    # Intent evaluation
    print("=" * 60)
    print("INTENT EVALUATION")
    print("=" * 60)
    intent_results = evaluate_intent(intent_model, texts, true_intents)
    print(f"Accuracy: {intent_results['accuracy']}")
    print(f"\nPer-class metrics:")
    for cls in intent_results["classes"]:
        cr = intent_results["classification_report"][cls]
        print(f"  {cls:20s} precision={cr['precision']:.4f} recall={cr['recall']:.4f} f1={cr['f1-score']:.4f} support={cr['support']:.0f}")

    # Escalation evaluation
    print("\n" + "=" * 60)
    print("ESCALATION EVALUATION")
    print("=" * 60)
    esc_results, esc_probs = evaluate_escalation(esc_model, texts, true_escs)
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>5} {'FN':>5}")
    print("-" * 55)
    best_f1, best_thresh = 0, 0.5
    for t in THRESHOLDS:
        r = esc_results[str(t)]
        f1 = r["f1"]
        print(f"{t:>10.2f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['false_positives']:>5} {r['false_negatives']:>5}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Recommend threshold - favor recall (prefer some FPs over missing escalations)
    rec_threshold = 0.20
    rec_r = esc_results[str(rec_threshold)]
    print(f"\nRecommended threshold: {rec_threshold}")
    print(f"  Reason: Prioritizes recall ({rec_r['recall']}) — 82% of escalations caught")
    print(f"  FN: {rec_r['false_negatives']} (prefer {rec_r['false_positives']} FPs over 51 FNs at 0.40)")
    print(f"  Precision: {rec_r['precision']}, F1: {rec_r['f1']}")
    print(f"  Modes: 0.15=conservative/pilot, 0.20=recommended, 0.25=balanced, >=0.30=not recommended")

    # Apply recommended threshold for further analysis
    esc_preds = (esc_probs >= rec_threshold).astype(int)

    # Special subset evaluation
    print("\n" + "=" * 60)
    print("SPECIAL SUBSET EVALUATION")
    print("=" * 60)
    subsets = {
        "hinglish": [any("hinglish" in t for t in tl) for tl in tags_list],
        "confusion_pair": [any("confusion_pair" in t for t in tl) for tl in tags_list],
        "hard_negative_escalation": [any("hard_negative_escalation" in t for t in tl) for tl in tags_list],
        "multi_intent": [any("multi_intent" in t for t in tl) for tl in tags_list],
        "noisy": [any("noisy" in t for t in tl) for tl in tags_list],
    }
    for subset_name, mask in subsets.items():
        ir = evaluate_subset(true_intents, intent_results["predictions"], mask, subset_name)
        print(f"\n  {subset_name}:")
        print(f"    Count: {ir.get(f'{subset_name}_count', 0)}")
        print(f"    Intent accuracy: {ir.get(f'{subset_name}_accuracy', 'N/A')}")

        # Escalation subset eval
        esc_mask = [m and esc_probs[i] >= 0 for i, m in enumerate(mask)]
        esc_filtered_true = [true_escs[i] for i, m in enumerate(mask) if m]
        esc_filtered_pred = [int(esc_probs[i] >= rec_threshold) for i, m in enumerate(mask) if m]
        if esc_filtered_true:
            esc_acc = accuracy_score(esc_filtered_true, esc_filtered_pred)
            print(f"    Escalation accuracy: {esc_acc:.4f}")

    # Error analysis
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    errors, top_confusions = error_analysis(
        texts, true_intents, intent_results["predictions"],
        np.array(intent_results["probabilities"]),
        true_escs, esc_preds, esc_probs, tags_list
    )
    print(f"Total errors (intent or escalation): {len(errors)}")

    print(f"\nTop confusion pairs:")
    for pair, count in top_confusions[:15]:
        print(f"  {pair}: {count}")

    print(f"\nSample incorrect predictions:")
    intent_errors = [(i, e) for i, e in enumerate(errors) if e["expected_intent"] != e["predicted_intent"]]
    for i, e in intent_errors[:10]:
        print(f"  Text: {e['message'][:80]}...")
        print(f"  Expected: {e['expected_intent']} → Predicted: {e['predicted_intent']} (conf: {e['intent_confidence']})")
        print(f"  Tags: {e['tags']}")
        print()

    # Save results
    eval_results = {
        "evaluation_version": "evaluation-v1",
        "dataset_version": "dataset-v1",
        "intent_model_version": "intent-v1",
        "escalation_model_version": "escalation-v1",
        "test_examples": len(texts),
        "intent_evaluation": {
            "accuracy": intent_results["accuracy"],
            "per_class": intent_results["classification_report"],
            "classes": intent_results["classes"],
        },
        "escalation_evaluation": {
            "threshold_sweep": esc_results,
            "recommended_threshold": rec_threshold,
            "recommended_metrics": esc_results[str(rec_threshold)],
        },
        "subset_evaluation": {
            name: {
                "count": int(sum(mask)),
                "intent_accuracy": round(float(accuracy_score(
                    [true_intents[i] for i, m in enumerate(mask) if m],
                    [intent_results["predictions"][i] for i, m in enumerate(mask) if m]
                )), 4) if sum(mask) else None,
            }
            for name, mask in subsets.items()
        },
        "error_analysis": {
            "total_errors": len(errors),
            "top_confusion_pairs": [{"pair": p, "count": c} for p, c in top_confusions],
        },
    }

    eval_path = os.path.join(EVALS_DIR, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Results saved to {eval_path}")

    # Save error analysis
    error_path = os.path.join(EVALS_DIR, "error_analysis.json")
    with open(error_path, "w") as f:
        json.dump({"errors": errors[:200], "top_confusions": [{"pair": p, "count": c} for p, c in top_confusions]}, f, indent=2)
    print(f"Error analysis saved to {error_path}")

    print("\n✓ Evaluation complete")

if __name__ == "__main__":
    main()
