#!/usr/bin/env python3
"""Evaluate V4 transformer models against the frozen V3 benchmark."""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support, brier_score_loss,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "v3")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVALS_DIR = os.path.join(BASE_DIR, "evaluations", "evaluation-v4")
os.makedirs(EVALS_DIR, exist_ok=True)

INTENT_CLASSES = [
    "account_issue", "complaint", "general_question", "human_request",
    "other", "pricing", "product_question", "refund", "sales", "technical_support",
]
INTENT_LABEL_MAP = {c: i for i, c in enumerate(INTENT_CLASSES)}
ESC_THRESHOLDS = [x / 100 for x in range(5, 96, 5)]

SUBSET_TAGS = {
    "hinglish": "hinglish",
    "confusion_pair": "confusion_pair",
    "multi_intent": "multi_intent",
    "noisy": "noisy",
    "negation": "negation",
    "resolution_state": "resolution_state",
    "hard_negative_escalation": "hard_negative_escalation",
}

FN_KEYWORD_RULES = [
    ("time-persistent", re.compile(r"\b(waited|waiting|been (waiting|wait)|long time|hours|days|weeks|still waiting|since last)\b", re.IGNORECASE)),
    ("indirect human request", re.compile(r"\b(can (you|someone|anyone) (help|assist|look|check|connect|speak)|i (need|want) (to talk|to speak|a human|a person|someone)|could you (please|kindly)|would you (please|kindly)|is there (a|any) (human|person|agent|representative))\b", re.IGNORECASE)),
    ("security/account concern", re.compile(r"\b(security|hack|breach|unauthorized|stolen|fraud|suspicious|scam|phish|compromised)\b", re.IGNORECASE)),
]

HINGLISH_KEYWORDS = re.compile(r"\b(baat|karni|hai|karo|karein|chahiye|mujhe|aap|hum|nahi|hoga|kya|yeh|mera|meri|mujh|tum|tera|teri|apna|unka|inka)\b")

CANDIDATE_INFO = {
    "en": {
        "name": "distilbert-base-uncased",
        "intent_dir": "intent-v4-en",
        "escalation_dir": "escalation-v4-en",
    },
    "multi": {
        "name": "distilbert-base-multilingual-cased",
        "intent_dir": "intent-v4-multi",
        "escalation_dir": "escalation-v4-multi",
    },
}


def load_benchmark():
    texts, intents, escs, tags_list = [], [], [], []
    with open(os.path.join(DATA_DIR, "benchmark.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"].strip())
            intents.append(row["intent"].strip())
            escs.append(1 if row["escalation_required"].strip() == "true" else 0)
            tags_list.append(json.loads(row.get("tags", "[]")))
    return texts, intents, np.array(escs), tags_list


def load_v4_intent_model(candidate):
    model_dir = os.path.join(MODELS_DIR, "intent", CANDIDATE_INFO[candidate]["intent_dir"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def load_v4_escalation_model(candidate):
    model_dir = os.path.join(MODELS_DIR, "escalation", CANDIDATE_INFO[candidate]["escalation_dir"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict_intent_batch(tokenizer, model, texts, batch_size=32):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


@torch.no_grad()
def predict_escalation_batch(tokenizer, model, texts, batch_size=32):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if np.sum(mask) == 0:
            continue
        ece += np.sum(mask) * abs(np.mean(y_true[mask]) - np.mean(y_prob[mask]))
    return ece / len(y_true)


def categorize_fn(row_text, tags):
    text_lower = row_text.lower()
    tag_set = set(tags)
    fn_category_rules = [
        ("hinglish", ["hinglish"]),
        ("multi_intent", ["multi_intent"]),
        ("security/account concern", ["account", "security"]),
    ]
    for cat_name, cat_tags in fn_category_rules:
        if any(t in tag_set for t in cat_tags):
            return cat_name
    for cat_name, pattern in FN_KEYWORD_RULES:
        if pattern.search(text_lower):
            return cat_name
    if HINGLISH_KEYWORDS.search(text_lower):
        return "hinglish"
    if "human_request" in tag_set:
        return "indirect human request"
    return "subtle/unresolved"


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def evaluate_intent(candidate):
    print(f"\n{'=' * 70}")
    print(f"  Intent V4 ({candidate.upper()}) Evaluation")
    print(f"{'=' * 70}")

    texts, true_intents, _, tags_list = load_benchmark()
    true_labels = np.array([INTENT_LABEL_MAP[i] for i in true_intents])

    print(f"\nLoading V4 intent model ({candidate})...")
    tokenizer, model = load_v4_intent_model(candidate)

    print(f"Running predictions on {len(texts)} benchmark examples...")
    probs = predict_intent_batch(tokenizer, model, texts)
    pred_labels = np.argmax(probs, axis=-1)
    pred_intents = [INTENT_CLASSES[i] for i in pred_labels]
    max_confs = probs.max(axis=1)

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    report = classification_report(true_labels, pred_labels, target_names=INTENT_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(INTENT_CLASSES))))

    print(f"\n  Accuracy:    {accuracy:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

    print(f"\n  Per-class metrics:")
    print(f"  {'Class':>25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print(f"  {'-' * 65}")
    for cls in INTENT_CLASSES:
        r = report.get(cls, {})
        print(f"  {cls:>25} {r.get('precision', 0):>10.4f} {r.get('recall', 0):>10.4f} {r.get('f1-score', 0):>10.4f} {r.get('support', 0):>8.0f}")

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = f"{'':>25}" + "".join(f"{c[:6]:>7}" for c in INTENT_CLASSES)
    print(f"  {header}")
    for i, cls in enumerate(INTENT_CLASSES):
        row = f"{cls:>25}" + "".join(f"{cm[i][j]:>7}" for j in range(len(INTENT_CLASSES)))
        print(f"  {row}")

    subset_results = {}
    print(f"\n  Subset performance:")
    print(f"  {'Subset':>30} {'Count':>6} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-' * 58}")
    for subset_name, tag in SUBSET_TAGS.items():
        mask = np.array([any(tag in t for t in tl) for tl in tags_list])
        count = int(mask.sum())
        if count == 0:
            subset_results[subset_name] = {"count": 0, "accuracy": None, "macro_f1": None}
            print(f"  {subset_name:>30} {count:>6} {'N/A':>10} {'N/A':>10}")
            continue
        sub_true = true_labels[mask]
        sub_pred = pred_labels[mask]
        sub_acc = accuracy_score(sub_true, sub_pred)
        sub_mf1 = f1_score(sub_true, sub_pred, average="macro", zero_division=0)
        subset_results[subset_name] = {
            "count": count,
            "accuracy": round(float(sub_acc), 4),
            "macro_f1": round(float(sub_mf1), 4),
        }
        print(f"  {subset_name:>30} {count:>6} {sub_acc:>10.4f} {sub_mf1:>10.4f}")

    confusion_pairs = Counter()
    high_conf_errors = []
    for i in range(len(texts)):
        if true_intents[i] != pred_intents[i]:
            pair = f"{true_intents[i]} -> {pred_intents[i]}"
            confusion_pairs[pair] += 1
            if max_confs[i] > 0.7:
                high_conf_errors.append({
                    "text": texts[i][:200],
                    "true_intent": true_intents[i],
                    "pred_intent": pred_intents[i],
                    "confidence": round(float(max_confs[i]), 4),
                    "tags": tags_list[i],
                })

    print(f"\n  Top confusion pairs:")
    for pair, count in confusion_pairs.most_common(10):
        print(f"    {pair}: {count}")

    print(f"\n  High-confidence incorrect predictions (conf > 0.7): {len(high_conf_errors)}")
    for err in high_conf_errors[:5]:
        print(f"    [{err['confidence']:.2f}] {err['true_intent']} -> {err['pred_intent']}: {err['text'][:80]}...")

    results = {
        "model": f"intent-v4-{candidate}",
        "dataset": "dataset-v3-benchmark",
        "n_examples": len(texts),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": {
            cls: {
                "precision": round(float(report[cls]["precision"]), 4),
                "recall": round(float(report[cls]["recall"]), 4),
                "f1": round(float(report[cls]["f1-score"]), 4),
                "support": int(report[cls]["support"]),
            }
            for cls in INTENT_CLASSES
        },
        "confusion_matrix": cm.tolist(),
        "classes": INTENT_CLASSES,
        "subsets": subset_results,
        "confusion_pairs": [{"pair": p, "count": c} for p, c in confusion_pairs.most_common(20)],
        "high_confidence_errors": high_conf_errors,
    }

    return results


def evaluate_escalation(candidate):
    print(f"\n{'=' * 70}")
    print(f"  Escalation V4 ({candidate.upper()}) Evaluation")
    print(f"{'=' * 70}")

    texts, _, true_escs, tags_list = load_benchmark()

    print(f"\nLoading V4 escalation model ({candidate})...")
    tokenizer, model = load_v4_escalation_model(candidate)

    print(f"Running predictions on {len(texts)} benchmark examples...")
    esc_probs = predict_escalation_batch(tokenizer, model, texts)

    threshold_results = {}
    best_f1, best_t, best_pr, best_re, best_fp, best_fn = 0, 0.50, 0, 0, 0, 0
    for t in ESC_THRESHOLDS:
        preds = (esc_probs >= t).astype(int)
        pr, re, f1, _ = precision_recall_fscore_support(true_escs, preds, average="binary", pos_label=1, zero_division=0)
        cm = confusion_matrix(true_escs, preds, labels=[0, 1])
        fp = int(cm[0][1]) if cm.shape == (2, 2) else 0
        fn = int(cm[1][0]) if cm.shape == (2, 2) else 0
        threshold_results[str(t)] = {
            "precision": round(float(pr), 4),
            "recall": round(float(re), 4),
            "f1": round(float(f1), 4),
            "false_positives": fp,
            "false_negatives": fn,
        }
        if f1 > best_f1:
            best_f1, best_t, best_pr, best_re, best_fp, best_fn = f1, t, pr, re, fp, fn

    print(f"\n  Threshold sweep:")
    print(f"  {'Thresh':>8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>6} {'FN':>6}")
    print(f"  {'-' * 54}")
    for t in ESC_THRESHOLDS:
        r = threshold_results[str(t)]
        print(f"  {t:>8.2f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['false_positives']:>6} {r['false_negatives']:>6}")

    print(f"\n  Recommended threshold: {best_t:.2f} (F1={best_f1:.4f}, P={best_pr:.4f}, R={best_re:.4f}, FP={best_fp}, FN={best_fn})")

    esc_preds = (esc_probs >= best_t).astype(int)

    ece = compute_ece(true_escs, esc_probs)
    brier = brier_score_loss(true_escs, esc_probs)
    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  Brier score:                      {brier:.4f}")

    subset_results = {}
    print(f"\n  Subset performance:")
    print(f"  {'Subset':>30} {'Count':>6} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'-' * 58}")
    for subset_name, tag in SUBSET_TAGS.items():
        mask = np.array([any(tag in t for t in tl) for tl in tags_list])
        count = int(mask.sum())
        if count == 0:
            subset_results[subset_name] = {"count": 0, "accuracy": None, "f1": None}
            print(f"  {subset_name:>30} {count:>6} {'N/A':>10} {'N/A':>10}")
            continue
        sub_true = true_escs[mask]
        sub_pred = esc_preds[mask]
        sub_acc = accuracy_score(sub_true, sub_pred)
        sub_f1 = f1_score(sub_true, sub_pred, pos_label=1, zero_division=0)
        subset_results[subset_name] = {
            "count": count,
            "accuracy": round(float(sub_acc), 4),
            "f1": round(float(sub_f1), 4),
        }
        print(f"  {subset_name:>30} {count:>6} {sub_acc:>10.4f} {sub_f1:>10.4f}")

    fn_indices = np.where((true_escs == 1) & (esc_preds == 0))[0]
    fn_categories = defaultdict(list)
    for idx in fn_indices:
        cat = categorize_fn(texts[idx], tags_list[idx])
        fn_categories[cat].append({
            "text": texts[idx][:200],
            "score": float(esc_probs[idx]),
            "tags": tags_list[idx],
        })

    print(f"\n  False negative categorization ({len(fn_indices)} total):")
    fn_cat_order = ["hinglish", "multi_intent", "subtle/unresolved", "time-persistent",
                     "indirect human request", "security/account concern"]
    for cat in fn_cat_order:
        items = fn_categories.get(cat, [])
        print(f"    {cat:30s}: {len(items)}")

    results = {
        "model": f"escalation-v4-{candidate}",
        "dataset": "dataset-v3-benchmark",
        "n_examples": len(texts),
        "n_escalation_positive": int(true_escs.sum()),
        "threshold_sweep": threshold_results,
        "recommended_threshold": best_t,
        "recommended_metrics": {
            "precision": round(float(best_pr), 4),
            "recall": round(float(best_re), 4),
            "f1": round(float(best_f1), 4),
            "false_positives": best_fp,
            "false_negatives": best_fn,
        },
        "ece": round(float(ece), 4),
        "brier_score": round(float(brier), 4),
        "subsets": subset_results,
        "fn_categories": {
            cat: [
                {"text": item["text"], "score": item["score"], "tags": item["tags"]}
                for item in items
            ]
            for cat, items in fn_categories.items()
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate V4 models against V3 benchmark")
    parser.add_argument("--task", choices=["intent", "escalation"], required=True)
    parser.add_argument("--candidate", choices=["en", "multi"], required=True)
    args = parser.parse_args()

    if args.task == "intent":
        results = evaluate_intent(args.candidate)
    else:
        results = evaluate_escalation(args.candidate)

    eval_path = os.path.join(EVALS_DIR, f"{results['model']}_evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2, default=str)
    print(f"\nResults saved to {eval_path}")

    print(f"\n{'=' * 70}")
    print(f"  Evaluation complete: {results['model']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
