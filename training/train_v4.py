#!/usr/bin/env python3
"""Train V4 transformer models for intent and escalation classification."""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from collections import Counter

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "v3")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_VERSION = "dataset-v3"
RANDOM_SEED = 42

CANDIDATE_INFO = {
    "en": {
        "name": "distilbert-base-uncased",
        "desc": "67M params, English, MIT license",
        "intent_dir": "intent-v4-en",
        "escalation_dir": "escalation-v4-en",
    },
    "multi": {
        "name": "distilbert-base-multilingual-cased",
        "desc": "134M params, 104 languages including Hindi, MIT license",
        "intent_dir": "intent-v4-multi",
        "escalation_dir": "escalation-v4-multi",
    },
}

INTENT_CLASSES = [
    "account_issue", "complaint", "general_question", "human_request",
    "other", "pricing", "product_question", "refund", "sales", "technical_support",
]
INTENT_LABEL_MAP = {c: i for i, c in enumerate(INTENT_CLASSES)}


def load_data(filepath, task):
    texts, labels = [], []
    with open(filepath, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"].strip())
            if task == "intent":
                labels.append(INTENT_LABEL_MAP[row["intent"].strip()])
            else:
                labels.append(1 if row["escalation_required"].strip() == "true" else 0)
    return texts, labels


def compute_intent_class_weights(train_labels):
    counts = Counter(train_labels)
    total = len(train_labels)
    num_classes = len(counts)
    weights = {c: total / (num_classes * counts[c]) for c in counts}
    weight_tensor = torch.tensor([weights[i] for i in sorted(weights.keys())], dtype=torch.float32)
    return weights, weight_tensor


def compute_escalation_pos_weight(train_labels):
    pos = sum(1 for l in train_labels if l == 1)
    neg = sum(1 for l in train_labels if l == 0)
    pos_weight_val = neg / pos if pos > 0 else 1.0
    return pos_weight_val


def print_class_distribution(labels, task, name):
    if task == "intent":
        idx_to_class = {v: k for k, v in INTENT_LABEL_MAP.items()}
        counts = Counter(labels)
        total = len(labels)
        print(f"  {name} distribution ({total} examples):")
        for c in sorted(INTENT_CLASSES):
            idx = INTENT_LABEL_MAP[c]
            cnt = counts.get(idx, 0)
            pct = cnt / total * 100 if total else 0
            print(f"    {c:25s} {cnt:5d} ({pct:5.1f}%)")
    else:
        pos = sum(1 for l in labels if l == 1)
        neg = sum(1 for l in labels if l == 0)
        total = len(labels)
        print(f"  {name} distribution ({total} examples):")
        print(f"    negative (0): {neg:5d} ({neg/total*100:5.1f}%)")
        print(f"    positive (1): {pos:5d} ({pos/total*100:5.1f}%)")


def compute_metrics_intent(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def compute_metrics_escalation(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probs >= 0.5).astype(int).flatten()
    acc = accuracy_score(labels, predictions)
    pr, re, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", pos_label=1, zero_division=0)
    return {"accuracy": acc, "precision": pr, "recall": re, "f1": f1}


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weight=None, pos_weight=None, task="intent", **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight
        self.pos_weight = pos_weight
        self.task = task

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.task == "intent":
            if self.class_weight is not None:
                weight = self.class_weight.to(logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=weight)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        else:
            if self.pos_weight is not None:
                pw = self.pos_weight.to(logits.device)
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(), labels.float())
        return (loss, outputs) if return_outputs else loss


def threshold_sweep(labels, probs, title="Threshold sweep"):
    print(f"\n  {title}:")
    print(f"  {'Thresh':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'FP':>5} {'FN':>5}")
    print(f"  {'-' * 44}")
    best_f1, best_t = 0.0, 0.50
    best_pr, best_re = 0.0, 0.0
    for t in [x / 100 for x in range(5, 96, 5)]:
        preds = (probs >= t).astype(int)
        pr, re, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1, zero_division=0)
        cm = None
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(labels, preds, labels=[0, 1])
        except Exception:
            cm = [[0, 0], [0, 0]]
        fp, fn = int(cm[0][1]), int(cm[1][0])
        print(f"  {t:>8.2f} {pr:>8.4f} {re:>8.4f} {f1:>8.4f} {fp:>5} {fn:>5}")
        if f1 > best_f1:
            best_f1, best_t, best_pr, best_re = f1, t, pr, re
    print(f"\n  Best F1: {best_f1:.4f} at threshold {best_t:.2f} (P={best_pr:.4f}, R={best_re:.4f})")
    return best_t, best_f1


def get_per_class_f1(labels, predictions, task):
    if task == "intent":
        idx_to_class = {v: k for k, v in INTENT_LABEL_MAP.items()}
        report = classification_report(labels, predictions, labels=list(range(len(INTENT_CLASSES))),
                                       target_names=INTENT_CLASSES, output_dict=True, zero_division=0)
        per_class = {}
        for cls_name in INTENT_CLASSES:
            if cls_name in report:
                per_class[cls_name] = round(float(report[cls_name]["f1-score"]), 4)
        return per_class
    else:
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        per_class = {"negative": round(float(report.get("0", {}).get("f1-score", 0)), 4),
                     "positive": round(float(report.get("1", {}).get("f1-score", 0)), 4)}
        return per_class


def train_model(task, candidate):
    set_seed(RANDOM_SEED)
    cinfo = CANDIDATE_INFO[candidate]
    model_name = cinfo["name"]

    version_prefix = "intent" if task == "intent" else "escalation"
    model_version = f"{version_prefix}-v4-{candidate}"
    output_dir = cinfo["intent_dir"] if task == "intent" else cinfo["escalation_dir"]

    if task == "intent":
        num_labels = len(INTENT_CLASSES)
    else:
        num_labels = 1  # binary with single logit

    save_path = os.path.join(MODELS_DIR, version_prefix, output_dir)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Training {model_version}")
    print(f"Base model: {model_name} ({cinfo['desc']})")
    print(f"Task: {task}")
    print(f"Output: {save_path}")
    print(f"{'=' * 70}\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading data...")
    train_texts, train_labels = load_data(os.path.join(DATA_DIR, "train.csv"), task)
    val_texts, val_labels = load_data(os.path.join(DATA_DIR, "validation.csv"), task)
    bench_texts, bench_labels = load_data(os.path.join(DATA_DIR, "benchmark.csv"), task)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_texts)}")
    print(f"  Validation: {len(val_texts)}")
    print(f"  Benchmark: {len(bench_texts)}")

    print(f"\nClass distribution:")
    print_class_distribution(train_labels, task, "Train")
    print_class_distribution(val_labels, task, "Validation")
    print_class_distribution(bench_labels, task, "Benchmark")

    # Setup class weights
    class_weight = None
    pos_weight = None
    if task == "intent":
        weights_dict, class_weight = compute_intent_class_weights(train_labels)
        print(f"\nIntent class weights:")
        idx_to_class = {v: k for k, v in INTENT_LABEL_MAP.items()}
        for idx in sorted(weights_dict.keys()):
            print(f"  {idx_to_class[idx]:25s} {weights_dict[idx]:.4f}")
    else:
        pos_weight_val = compute_escalation_pos_weight(train_labels)
        pos_weight = torch.tensor([pos_weight_val])
        print(f"\nEscalation pos_weight (neg/pos): {pos_weight_val:.4f}")

    print(f"\nTokenizing (max_seq_length=128)...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    bench_encodings = tokenizer(bench_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long if task == "intent" else torch.float)
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = TransformDataset(train_encodings, train_labels)
    val_dataset = TransformDataset(val_encodings, val_labels)
    bench_dataset = TransformDataset(bench_encodings, bench_labels)

    print(f"Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    metric_for_best = "accuracy" if task == "intent" else "f1"
    compute_metrics_fn = compute_metrics_intent if task == "intent" else compute_metrics_escalation

    training_args = TrainingArguments(
        output_dir=save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=6,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=87,
        logging_strategy="epoch",
        save_total_limit=3,
        remove_unused_columns=False,
        seed=RANDOM_SEED,
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weight=class_weight,
        pos_weight=pos_weight,
        task=task,
    )

    print(f"\nStarting training...")
    start_time = time.time()
    train_result = trainer.train()
    train_time_seconds = time.time() - start_time
    print(f"\nTraining completed in {train_time_seconds:.2f}s")

    training_stats = train_result.metrics
    epochs_trained = int(training_stats.get("epoch", 6))

    # Determine early stopping epoch
    early_stopped_at = epochs_trained
    for i in range(1, 7):
        key = f"eval_{metric_for_best}"
        # Check log history for best epoch
        if trainer.state.log_history:
            best_epoch = None
            best_val = -float("inf")
            for entry in trainer.state.log_history:
                if "epoch" in entry and f"eval_{metric_for_best}" in entry:
                    val = entry[f"eval_{metric_for_best}"]
                    if val > best_val:
                        best_val = val
                        best_epoch = int(entry["epoch"])
            if best_epoch is not None:
                early_stopped_at = best_epoch

    print(f"\nEvaluating on validation set...")
    val_preds = trainer.predict(val_dataset)
    if task == "intent":
        val_logits = val_preds.predictions
        val_predictions = np.argmax(val_logits, axis=-1)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        best_val_metric = val_accuracy
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"\nClassification report (validation):")
        print(classification_report(val_labels, val_predictions, labels=list(range(len(INTENT_CLASSES))),
                                    target_names=INTENT_CLASSES, zero_division=0))
        per_class_f1 = get_per_class_f1(val_labels, val_predictions, task)
    else:
        val_logits = val_preds.predictions
        val_probs = 1.0 / (1.0 + np.exp(-val_logits)).flatten()
        default_preds = (val_probs >= 0.5).astype(int)
        val_f1 = f1_score(val_labels, default_preds, pos_label=1, zero_division=0)
        val_accuracy = accuracy_score(val_labels, default_preds)
        best_val_metric = val_f1
        print(f"Validation accuracy (thresh=0.50): {val_accuracy:.4f}")
        print(f"Validation F1 (thresh=0.50): {val_f1:.4f}")
        best_t, best_f1_val = threshold_sweep(val_labels, val_probs, title="Validation threshold sweep")
        per_class_f1 = get_per_class_f1(val_labels, default_preds, task)
        val_predictions = default_preds

    print(f"\nEvaluating on benchmark set (final, no training decisions)...")
    bench_preds = trainer.predict(bench_dataset)
    if task == "intent":
        bench_logits = bench_preds.predictions
        bench_predictions = np.argmax(bench_logits, axis=-1)
        bench_accuracy = accuracy_score(bench_labels, bench_predictions)
        bench_f1_macro = f1_score(bench_labels, bench_predictions, average="macro", zero_division=0)
        print(f"Benchmark accuracy: {bench_accuracy:.4f}")
        print(f"Benchmark F1 (macro): {bench_f1_macro:.4f}")
        print(f"\nClassification report (benchmark):")
        print(classification_report(bench_labels, bench_predictions, labels=list(range(len(INTENT_CLASSES))),
                                    target_names=INTENT_CLASSES, zero_division=0))
    else:
        bench_logits = bench_preds.predictions
        bench_probs = 1.0 / (1.0 + np.exp(-bench_logits)).flatten()
        bench_default_preds = (bench_probs >= 0.5).astype(int)
        bench_accuracy = accuracy_score(bench_labels, bench_default_preds)
        bench_f1 = f1_score(bench_labels, bench_default_preds, pos_label=1, zero_division=0)
        print(f"Benchmark accuracy (thresh=0.50): {bench_accuracy:.4f}")
        print(f"Benchmark F1 (thresh=0.50): {bench_f1:.4f}")
        threshold_sweep(bench_labels, bench_probs, title="Benchmark threshold sweep")

    print(f"\nSaving model and tokenizer to {save_path}...")
    trainer.save_model()
    tokenizer.save_pretrained(save_path)
    print("Model and tokenizer saved.")

    metadata = {
        "model_version": model_version,
        "dataset_version": DATASET_VERSION,
        "base_model": model_name,
        "training_date": datetime.now().isoformat(),
        "num_epochs_trained": epochs_trained,
        "best_val_accuracy": round(float(val_accuracy), 4) if task == "intent" else round(float(val_accuracy), 4),
        "best_val_f1": round(float(best_val_metric), 4) if task == "escalation" else round(float(f1_score(val_labels, val_predictions, average="weighted", zero_division=0)), 4),
        "per_class_f1": per_class_f1,
        "train_examples": len(train_texts),
        "validation_examples": len(val_texts),
        "benchmark_examples": len(bench_texts),
        "training_time_seconds": round(train_time_seconds, 2),
        "early_stopped_at_epoch": early_stopped_at,
        "random_seed": RANDOM_SEED,
        "max_seq_length": 128,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 87,
        "num_epochs_config": 6,
        "early_stopping_patience": 2,
        "evaluation_strategy": "epoch",
    }

    if task == "intent":
        weights_dict_int = {INTENT_CLASSES[idx]: round(float(w), 4) for idx, w in weights_dict.items()}
        metadata["class_weights_used"] = weights_dict_int
        metadata["intent_classes"] = INTENT_CLASSES
    else:
        metadata["pos_weight_used"] = round(float(pos_weight_val), 4)
        metadata["best_threshold"] = round(float(best_t), 2)
        metadata["best_f1_at_best_threshold"] = round(float(best_f1_val), 4)

    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {os.path.join(save_path, 'metadata.json')}")

    print(f"\n{'=' * 70}")
    print(f"Training complete: {model_version}")
    print(f"Best val {metric_for_best}: {best_val_metric:.4f}")
    print(f"Training time: {train_time_seconds:.2f}s")
    print(f"{'=' * 70}\n")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train V4 transformer model")
    parser.add_argument("--task", choices=["intent", "escalation"], required=True,
                        help="Task to train: intent or escalation")
    parser.add_argument("--candidate", choices=["en", "multi"], required=True,
                        help="Model candidate: en (English) or multi (multilingual)")
    args = parser.parse_args()

    train_model(args.task, args.candidate)


if __name__ == "__main__":
    main()
