#!/usr/bin/env python3
"""Train intent-v3 using sentence embeddings + Logistic Regression on dataset-v3."""

import csv, json, os, time, sys
from datetime import datetime
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "v3")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_VERSION = "intent-v3"
DATASET_VERSION = "dataset-v3"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RANDOM_SEED = 42

def load_csv(filepath):
    texts, labels = [], []
    with open(filepath, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"].strip())
            labels.append(row["intent"].strip())
    return texts, labels

def main():
    print(f"Training {MODEL_VERSION}")
    print(f"Embedding: {EMBEDDING_MODEL}, Dataset: {DATASET_VERSION}")

    X_train_text, y_train = load_csv(os.path.join(DATA_DIR, "train.csv"))
    X_val_text, y_val = load_csv(os.path.join(DATA_DIR, "validation.csv"))

    print(f"Train: {len(X_train_text)}, Val: {len(X_val_text)}")
    print(f"Classes: {sorted(set(y_train))}")

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    start = time.time()
    X_train = embedder.encode(X_train_text, show_progress_bar=True)
    print(f"Train embeddings: {time.time()-start:.2f}s")
    X_val = embedder.encode(X_val_text, show_progress_bar=True)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs")
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)

    print(f"\nTraining time: {train_time:.2f}s")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(classification_report(y_val, y_pred))

    model_dir = os.path.join(MODELS_DIR, "intent", MODEL_VERSION)
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
        "classes": sorted(clf.classes_.tolist()),
        "validation_accuracy": round(float(accuracy), 4),
        "validation_classification_report": report,
        "train_examples": len(X_train_text),
        "validation_examples": len(X_val_text),
        "classifier_params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved")

    print(f"V intent-v3 training complete")


if __name__ == "__main__":
    main()
