#!/usr/bin/env python3
"""Side-by-side comparison of V2, V3, V4-en, V4-multi on the V3 frozen benchmark."""

import csv
import json
import os
import sys
from collections import Counter

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
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
SUBSET_TAGS = [
    "hinglish", "confusion_pair", "multi_intent", "noisy",
    "negation", "resolution_state", "hard_negative_escalation",
]
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_benchmark():
    texts, intents, escs, tags_list = [], [], [], []
    with open(os.path.join(DATA_DIR, "benchmark.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"].strip())
            intents.append(row["intent"].strip())
            escs.append(1 if row["escalation_required"].strip() == "true" else 0)
            tags_list.append(json.loads(row.get("tags", "[]")))
    return texts, intents, np.array(escs), tags_list


def load_v2_models():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    intent = joblib.load(os.path.join(MODELS_DIR, "intent", "intent-v2", "classifier.joblib"))
    esc = joblib.load(os.path.join(MODELS_DIR, "escalation", "escalation-v2", "classifier.joblib"))
    return embedder, intent, esc


def load_v3_models():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    intent = joblib.load(os.path.join(MODELS_DIR, "intent", "intent-v3", "classifier.joblib"))
    esc = joblib.load(os.path.join(MODELS_DIR, "escalation", "escalation-v3", "classifier.joblib"))
    return embedder, intent, esc


def load_v4_intent(candidate):
    model_dir = os.path.join(MODELS_DIR, "intent", f"intent-v4-{candidate}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def load_v4_escalation(candidate):
    model_dir = os.path.join(MODELS_DIR, "escalation", f"escalation-v4-{candidate}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict_v4_intent(tokenizer, model, texts, batch_size=32):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


@torch.no_grad()
def predict_v4_escalation(tokenizer, model, texts, batch_size=32):
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
            "false_positives": int(cm[0][1]) if cm.shape == (2, 2) else 0,
            "false_negatives": int(cm[1][0]) if cm.shape == (2, 2) else 0,
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


def main():
    print("=" * 80)
    print("  V2 vs V3 vs V4-en vs V4-multi Comparison")
    print("=" * 80)

    print("\nLoading benchmark data...")
    texts, true_intents, true_escs, tags_list = load_benchmark()
    n = len(texts)
    print(f"  Benchmark: {n} examples ({int(true_escs.sum())} esc+)")

    true_intent_labels = np.array([INTENT_LABEL_MAP[i] for i in true_intents])

    print("\nLoading models...")

    embedder_v2, v2_intent, v2_esc = load_v2_models()
    print("  V2 models loaded")

    embedder_v3, v3_intent, v3_esc = load_v3_models()
    print("  V3 models loaded")

    v4_en_tok_int, v4_en_int = load_v4_intent("en")
    v4_en_tok_esc, v4_en_esc = load_v4_escalation("en")
    print("  V4-en models loaded")

    v4_multi_tok_int, v4_multi_int = load_v4_intent("multi")
    v4_multi_tok_esc, v4_multi_esc = load_v4_escalation("multi")
    print("  V4-multi models loaded")

    print("\nRunning predictions...")

    embeddings_v2 = embedder_v2.encode(texts, show_progress_bar=False)
    v2_intent_preds = v2_intent.predict(embeddings_v2)
    v2_esc_scores = v2_intent.predict_proba(embeddings_v2)[:, 1] if hasattr(v2_esc, 'predict_proba') else v2_esc.predict_proba(embeddings_v2)[:, 1]

    embeddings_v3 = embedder_v3.encode(texts, show_progress_bar=False)
    v3_intent_preds = v3_intent.predict(embeddings_v3)
    v3_esc_scores = v3_esc.predict_proba(embeddings_v3)[:, 1]

    v4_en_intent_probs = predict_v4_intent(v4_en_tok_int, v4_en_int, texts)
    v4_en_intent_preds = np.argmax(v4_en_intent_probs, axis=-1)
    v4_en_esc_scores = predict_v4_escalation(v4_en_tok_esc, v4_en_esc, texts)

    v4_multi_intent_probs = predict_v4_intent(v4_multi_tok_int, v4_multi_int, texts)
    v4_multi_intent_preds = np.argmax(v4_multi_intent_probs, axis=-1)
    v4_multi_esc_scores = predict_v4_escalation(v4_multi_tok_esc, v4_multi_esc, texts)

    intent_preds = {
        "v2": v2_intent_preds,
        "v3": v3_intent_preds,
        "v4-en": v4_en_intent_preds,
        "v4-multi": v4_multi_intent_preds,
    }
    intent_probs = {
        "v2": None,
        "v3": None,
        "v4-en": v4_en_intent_probs,
        "v4-multi": v4_multi_intent_probs,
    }
    esc_scores = {
        "v2": v2_esc_scores,
        "v3": v3_esc_scores,
        "v4-en": v4_en_esc_scores,
        "v4-multi": v4_multi_esc_scores,
    }

    model_names = ["v2", "v3", "v4-en", "v4-multi"]

    # =========================================================
    # A. INTENT COMPARISON
    # =========================================================
    print("\n" + "=" * 80)
    print("  A. INTENT COMPARISON")
    print("=" * 80)

    print(f"\n  {'Metric':>25}", end="")
    for mn in model_names:
        print(f" {mn:>10}", end="")
    print()

    intent_metrics = {}
    for mn in model_names:
        preds = intent_preds[mn]
        acc = accuracy_score(true_intent_labels, preds)
        mf1 = f1_score(true_intent_labels, preds, average="macro", zero_division=0)
        wf1 = f1_score(true_intent_labels, preds, average="weighted", zero_division=0)
        report = classification_report(true_intent_labels, preds, target_names=INTENT_CLASSES, output_dict=True, zero_division=0)
        cm = confusion_matrix(true_intent_labels, preds, labels=list(range(len(INTENT_CLASSES))))
        intent_metrics[mn] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(mf1), 4),
            "weighted_f1": round(float(wf1), 4),
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
        }

    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        print(f"  {metric:>25}", end="")
        for mn in model_names:
            print(f" {intent_metrics[mn][metric]:>10.4f}", end="")
        print()

    print(f"\n  Per-class F1:")
    print(f"  {'Class':>25}", end="")
    for mn in model_names:
        print(f" {mn:>10}", end="")
    print()
    print(f"  {'-' * 65}")
    for cls in INTENT_CLASSES:
        print(f"  {cls:>25}", end="")
        for mn in model_names:
            f1_val = intent_metrics[mn]["per_class"][cls]["f1"]
            print(f" {f1_val:>10.4f}", end="")
        print()

    # Subset comparison
    print(f"\n  Subset intent accuracy:")
    print(f"  {'Subset':>30}", end="")
    for mn in model_names:
        print(f" {mn:>10}", end="")
    print()
    print(f"  {'-' * 75}")
    subset_intent_metrics = {}
    for tag in SUBSET_TAGS:
        mask = np.array([any(tag in t for t in tl) for tl in tags_list])
        count = int(mask.sum())
        if count == 0:
            print(f"  {tag:>30} {count:>6} {'N/A':>10}")
            continue
        print(f"  {tag:>30} {count:>6}", end="")
        subset_intent_metrics[tag] = {"count": count}
        for mn in model_names:
            sub_true = true_intent_labels[mask]
            sub_pred = intent_preds[mn][mask]
            sub_acc = accuracy_score(sub_true, sub_pred)
            subset_intent_metrics[tag][mn] = round(float(sub_acc), 4)
            print(f" {sub_acc:>10.4f}", end="")
        print()

    # =========================================================
    # B. ESCALATION COMPARISON
    # =========================================================
    print("\n" + "=" * 80)
    print("  B. ESCALATION COMPARISON")
    print("=" * 80)

    esc_eval = {}
    for mn in model_names:
        esc_eval[mn] = evaluate_escalation_thresholds(esc_scores[mn], true_escs)

    print(f"\n  Threshold sweep:")
    header = f"  {'Thresh':>8}"
    for mn in model_names:
        header += f" {mn+' P':>8} {mn+' R':>8} {mn+' F1':>8} {mn+' FP':>5} {mn+' FN':>5}"
    print(header)

    for t in ESC_THRESHOLDS:
        line = f"  {t:>8.2f}"
        for mn in model_names:
            r = esc_eval[mn][str(t)]
            line += f" {r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['false_positives']:>5} {r['false_negatives']:>5}"
        print(line)

    print(f"\n  Best thresholds:")
    for mn in model_names:
        best_t, best_f1_val = find_best_threshold(esc_eval[mn])
        r = esc_eval[mn][str(best_t)]
        print(f"    {mn:>10}: threshold={best_t:.2f} F1={best_f1_val:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")

    esc_metrics = {}
    for mn in model_names:
        bt, _ = find_best_threshold(esc_eval[mn])
        esc_preds_at_best = (esc_scores[mn] >= bt).astype(int)
        ece_val = compute_ece(true_escs, esc_scores[mn])
        brier_val = brier_score_loss(true_escs, esc_scores[mn])
        esc_metrics[mn] = {
            "best_threshold": bt,
            "best_metrics": esc_eval[mn][str(bt)],
            "ece": round(float(ece_val), 4),
            "brier_score": round(float(brier_val), 4),
        }

    print(f"\n  Calibration:")
    for mn in model_names:
        print(f"    {mn:>10}: ECE={esc_metrics[mn]['ece']:.4f}  Brier={esc_metrics[mn]['brier_score']:.4f}")

    print(f"\n  Subset escalation F1 (at best threshold):")
    print(f"  {'Subset':>30}", end="")
    for mn in model_names:
        print(f" {mn:>10}", end="")
    print()
    print(f"  {'-' * 75}")
    subset_esc_metrics = {}
    for tag in SUBSET_TAGS:
        mask = np.array([any(tag in t for t in tl) for tl in tags_list])
        count = int(mask.sum())
        if count == 0:
            print(f"  {tag:>30} {count:>6}")
            continue
        print(f"  {tag:>30} {count:>6}", end="")
        subset_esc_metrics[tag] = {"count": count}
        for mn in model_names:
            bt = esc_metrics[mn]["best_threshold"]
            esc_preds_at_best = (esc_scores[mn] >= bt).astype(int)
            sub_true = true_escs[mask]
            sub_pred = esc_preds_at_best[mask]
            sub_f1 = f1_score(sub_true, sub_pred, pos_label=1, zero_division=0)
            subset_esc_metrics[tag][mn] = round(float(sub_f1), 4)
            print(f" {sub_f1:>10.4f}", end="")
        print()

    # =========================================================
    # SAVE
    # =========================================================
    results = {
        "config": {
            "dataset": "dataset-v3-benchmark",
            "embedding_model": EMBEDDING_MODEL,
            "n_examples": n,
            "n_escalation_positive": int(true_escs.sum()),
        },
        "intent_comparison": intent_metrics,
        "escalation_comparison": esc_metrics,
        "escalation_threshold_sweep": esc_eval,
        "subset_intent_accuracy": subset_intent_metrics,
        "subset_escalation_f1": subset_esc_metrics,
    }

    eval_path = os.path.join(EVALS_DIR, "comparison_v2_v3_v4.json")
    with open(eval_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2, default=str)
    print(f"\nResults saved to {eval_path}")

    print("\n" + "=" * 80)
    print("  Comparison complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
