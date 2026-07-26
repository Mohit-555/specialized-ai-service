#!/usr/bin/env python3
"""Train the escalation binary classifier (TF-IDF + Logistic Regression)."""

import csv
import json
import os
import time
from datetime import datetime

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

RANDOM_SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v1")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

MODEL_VERSION = "escalation-v1"
DATASET_VERSION = "dataset-v1"

def load_csv(filepath):
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"].strip())
            labels.append(1 if row["escalation_required"].strip() == "true" else 0)
    return texts, labels

def main():
    print(f"Training {MODEL_VERSION}")
    print(f"Dataset: {DATASET_VERSION}")
    print(f"Random seed: {RANDOM_SEED}")

    train_file = os.path.join(DATA_DIR, "train.csv")
    val_file = os.path.join(DATA_DIR, "validation.csv")

    X_train, y_train = load_csv(train_file)
    X_val, y_val = load_csv(val_file)

    print(f"Train: {len(X_train)} examples (escalation: {sum(y_train)})")
    print(f"Validation: {len(X_val)} examples (escalation: {sum(y_val)})")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )),
    ])

    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=["no_escalation", "escalation"], output_dict=True)

    print(f"\nTraining time: {train_time:.2f}s")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["no_escalation", "escalation"]))

    model_dir = os.path.join(MODELS_DIR, "escalation", MODEL_VERSION)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    metadata = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "training_date": datetime.utcnow().isoformat(),
        "random_seed": RANDOM_SEED,
        "classes": ["no_escalation", "escalation"],
        "validation_accuracy": round(float(accuracy), 4),
        "validation_classification_report": report,
        "training_time_seconds": round(train_time, 2),
        "train_examples": len(X_train),
        "validation_examples": len(X_val),
        "pipeline": {
            "tfidf": {
                "max_features": 5000,
                "ngram_range": "(1, 2)",
                "sublinear_tf": True,
                "min_df": 2,
                "max_df": 0.95,
                "stop_words": "english",
            },
            "classifier": {
                "type": "LogisticRegression",
                "C": 1.0,
                "max_iter": 1000,
                "solver": "lbfgs",
            },
        },
    }
    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    experiment = {
        "experiment_id": f"exp-{MODEL_VERSION}",
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "training_date": datetime.utcnow().isoformat(),
        "tfidf_params": metadata["pipeline"]["tfidf"],
        "lr_params": metadata["pipeline"]["classifier"],
        "random_seed": RANDOM_SEED,
        "validation_accuracy": round(float(accuracy), 4),
        "timestamp": datetime.utcnow().isoformat(),
    }
    exp_path = os.path.join(EXPERIMENTS_DIR, f"experiment_{MODEL_VERSION}.json")
    with open(exp_path, "w") as f:
        json.dump(experiment, f, indent=2)
    print(f"Experiment record saved to {exp_path}")

    print(f"\n✓ {MODEL_VERSION} training complete")
    return pipeline

if __name__ == "__main__":
    main()
