#!/usr/bin/env python3
"""Train escalation-v3 using sentence embeddings + Logistic Regression on dataset-v3."""

import csv, json, os, time
from datetime import datetime
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "v3")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_VERSION = "escalation-v3"
DATASET_VERSION = "dataset-v3"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RANDOM_SEED = 42

def load_csv(filepath):
    texts, labels = [], []
    with open(filepath, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"].strip())
            labels.append(1 if row["escalation_required"].strip() == "true" else 0)
    return texts, labels

def main():
    print(f"Training {MODEL_VERSION}")

    X_train_text, y_train = load_csv(os.path.join(DATA_DIR, "train.csv"))
    X_val_text, y_val = load_csv(os.path.join(DATA_DIR, "validation.csv"))

    print(f"Train: {len(X_train_text)} (esc+={sum(y_train)}), Val: {len(X_val_text)} (esc+={sum(y_val)})")

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    X_train = embedder.encode(X_train_text, show_progress_bar=True)
    X_val = embedder.encode(X_val_text, show_progress_bar=True)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs")
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Training time: {train_time:.2f}s")

    # Threshold sweep
    y_scores = clf.predict_proba(X_val)[:, 1]
    print(f"\nThreshold sweep on validation:")
    print(f"{'Thresh':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'FP':>5} {'FN':>5}")
    print("-" * 44)

    best_f1, best_t = 0, 0.25
    for t in [x/100 for x in range(5, 96, 5)]:
        preds = (y_scores >= t).astype(int)
        pr, re, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", pos_label=1, zero_division=0)
        cm = confusion_matrix(y_val, preds, labels=[0, 1])
        fp, fn = int(cm[0][1]), int(cm[1][0])
        print(f"{t:>8.2f} {pr:>8.4f} {re:>8.4f} {f1:>8.4f} {fp:>5} {fn:>5}")
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\nBest F1: {best_f1:.4f} at threshold {best_t:.2f}")

    model_dir = os.path.join(MODELS_DIR, "escalation", MODEL_VERSION)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "classifier.joblib")
    joblib.dump(clf, model_path)
    print(f"Saved to {model_path}")

    metadata = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "training_date": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "validation_escalation_positive": int(sum(y_val)),
        "validation_escalation_negative": int(len(y_val) - sum(y_val)),
        "best_threshold": best_t,
        "best_f1": round(float(best_f1), 4),
        "train_examples": len(X_train_text),
        "validation_examples": len(X_val_text),
        "classifier_params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"V escalation-v3 training complete")

if __name__ == "__main__":
    main()
