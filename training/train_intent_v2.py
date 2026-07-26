#!/usr/bin/env python3
"""Train intent-v2 using sentence embeddings + Logistic Regression."""

import csv
import json
import os
import time
from datetime import datetime

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

RANDOM_SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v1")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

MODEL_VERSION = "intent-v2"
DATASET_VERSION = "dataset-v1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_csv(filepath):
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"].strip())
            labels.append(row["intent"].strip())
    return texts, labels


def main():
    print(f"Training {MODEL_VERSION}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Dataset: {DATASET_VERSION}")

    train_file = os.path.join(DATA_DIR, "train.csv")
    val_file = os.path.join(DATA_DIR, "validation.csv")

    X_train_text, y_train = load_csv(train_file)
    X_val_text, y_val = load_csv(val_file)

    print(f"Train: {len(X_train_text)} examples")
    print(f"Validation: {len(X_val_text)} examples")
    print(f"Classes: {sorted(set(y_train))}")

    # Load embedding model
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    start = time.time()
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embed_time = time.time() - start
    print(f"Embedding model loaded in {embed_time:.2f}s")

    # Embed texts
    print("Embedding training texts...")
    start = time.time()
    X_train = embedder.encode(X_train_text, show_progress_bar=True)
    train_embed_time = time.time() - start
    print(f"Training embeddings: {train_embed_time:.2f}s")

    print("Embedding validation texts...")
    X_val = embedder.encode(X_val_text, show_progress_bar=True)

    # Train classifier
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs")
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)

    print(f"\nTraining time: {train_time:.2f}s")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Save classifier
    model_dir = os.path.join(MODELS_DIR, "intent", MODEL_VERSION)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "classifier.joblib")
    joblib.dump(clf, model_path)
    print(f"Classifier saved to {model_path}")

    # Save embedder separately so it's not duplicated
    # In production, the embedder is loaded once and shared

    # Save metadata
    metadata = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "training_date": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "classes": sorted(clf.classes_.tolist()),
        "validation_accuracy": round(float(accuracy), 4),
        "validation_classification_report": report,
        "training_time_seconds": round(train_time + embed_time + train_embed_time, 2),
        "train_examples": len(X_train_text),
        "validation_examples": len(X_val_text),
        "classifier_params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
    }
    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    # Experiment tracking
    experiment = {
        "experiment_id": f"exp-{MODEL_VERSION}",
        "model_version": MODEL_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "dataset_version": DATASET_VERSION,
        "training_date": datetime.now().isoformat(),
        "lr_params": metadata["classifier_params"],
        "random_seed": RANDOM_SEED,
        "validation_accuracy": round(float(accuracy), 4),
    }
    exp_path = os.path.join(EXPERIMENTS_DIR, f"experiment_{MODEL_VERSION}.json")
    with open(exp_path, "w") as f:
        json.dump(experiment, f, indent=2)
    print(f"Experiment record saved to {exp_path}")

    print(f"\n\u2713 {MODEL_VERSION} training complete")
    return clf, embedder


if __name__ == "__main__":
    main()
