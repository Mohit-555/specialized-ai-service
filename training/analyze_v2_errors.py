#!/usr/bin/env python3
"""Comprehensive V2 error analysis before building dataset-v3."""

import csv
import json
import os
import sys
from collections import Counter

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "v1")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVALS_DIR = os.path.join(BASE_DIR, "evaluations", "evaluation-v2")
os.makedirs(EVALS_DIR, exist_ok=True)

V2_ESC_THRESHOLD = 0.25

INTENT_CATEGORIES = [
    "general_question", "product_question", "pricing", "sales",
    "technical_support", "complaint", "refund", "account_issue",
    "human_request", "other"
]


def load_data():
    texts, intents, escs, tags_list = [], [], [], []
    with open(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"].strip())
            intents.append(row["intent"].strip())
            escs.append(1 if row["escalation_required"].strip() == "true" else 0)
            tags_list.append(json.loads(row.get("tags", "[]")))
    return texts, intents, escs, tags_list


def main():
    print("=" * 70)
    print("V2 ERROR ANALYSIS FOR DATASET-V3 PLANNING")
    print("=" * 70)

    texts, true_intents, true_escs, tags_list = load_data()
    print(f"Test set: {len(texts)} examples")
    print(f"Escalation positive: {sum(true_escs)} / {len(true_escs)}")

    # Load V2 models
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    intent_clf = joblib.load(os.path.join(MODELS_DIR, "intent", "intent-v2", "classifier.joblib"))
    esc_clf = joblib.load(os.path.join(MODELS_DIR, "escalation", "escalation-v2", "classifier.joblib"))

    # Predict
    embeddings = embedder.encode(texts, show_progress_bar=True)
    pred_intents = intent_clf.predict(embeddings)
    intent_probs = intent_clf.predict_proba(embeddings)
    esc_scores = esc_clf.predict_proba(embeddings)[:, 1]
    pred_escs = (esc_scores >= V2_ESC_THRESHOLD).astype(int)

    # 1. Per-class error details
    print("\n" + "=" * 70)
    print("SECTION 1: PER-CLASS INTENT ERRORS")
    print("=" * 70)

    per_class_errors = {c: {"total": 0, "errors": 0, "confusions": Counter()} for c in INTENT_CATEGORIES}
    for i in range(len(texts)):
        c = true_intents[i]
        per_class_errors[c]["total"] += 1
        if pred_intents[i] != c:
            per_class_errors[c]["errors"] += 1
            per_class_errors[c]["confusions"][f"{c}->{pred_intents[i]}"] += 1

    for c in INTENT_CATEGORIES:
        info = per_class_errors[c]
        err_rate = info["errors"] / info["total"] * 100 if info["total"] > 0 else 0
        top_conf = info["confusions"].most_common(3)
        print(f"\n  {c} ({info['total']} examples): {info['errors']} errors ({err_rate:.1f}%)")
        for pair, count in top_conf:
            print(f"    {pair}: {count}")

    # 2. High-confidence wrong predictions
    print("\n" + "=" * 70)
    print("SECTION 2: HIGH-CONFIDENCE WRONG PREDICTIONS (conf > 0.7)")
    print("=" * 70)

    classes = intent_clf.classes_.tolist()
    high_conf_errors = []
    for i in range(len(texts)):
        if pred_intents[i] != true_intents[i]:
            conf = float(max(intent_probs[i]))
            if conf > 0.7:
                high_conf_errors.append({
                    "text": texts[i][:200],
                    "expected": true_intents[i],
                    "predicted": pred_intents[i],
                    "confidence": round(conf, 4),
                    "tags": tags_list[i],
                })

    print(f"\nTotal high-confidence intent errors: {len(high_conf_errors)}")
    for e in high_conf_errors:
        print(f"  [{e['confidence']:.4f}] {e['expected']} -> {e['predicted']}")
        print(f"    {e['text'][:120]}")
        print(f"    tags: {e['tags']}")

    # 3. Confusion matrix
    print("\n" + "=" * 70)
    print("SECTION 3: TOP CONFUSION PAIRS (ALL)")
    print("=" * 70)

    all_confusions = Counter()
    for i in range(len(texts)):
        if pred_intents[i] != true_intents[i]:
            all_confusions[f"{true_intents[i]} -> {pred_intents[i]}"] += 1

    for pair, count in all_confusions.most_common(25):
        print(f"  {pair}: {count}")

    # 4. Escalation error analysis
    print("\n" + "=" * 70)
    print(f"SECTION 4: ESCALATION FN/FP ANALYSIS (threshold={V2_ESC_THRESHOLD})")
    print("=" * 70)

    esc_fps = []
    esc_fns = []
    for i in range(len(texts)):
        if pred_escs[i] == 1 and true_escs[i] == 0:
            esc_fps.append({
                "text": texts[i][:200],
                "intent": true_intents[i],
                "pred_intent": pred_intents[i],
                "esc_score": round(float(esc_scores[i]), 4),
                "tags": tags_list[i],
            })
        elif pred_escs[i] == 0 and true_escs[i] == 1:
            esc_fns.append({
                "text": texts[i][:200],
                "intent": true_intents[i],
                "pred_intent": pred_intents[i],
                "esc_score": round(float(esc_scores[i]), 4),
                "tags": tags_list[i],
            })

    print(f"\nEscalation False Positives: {len(esc_fps)}")
    for e in esc_fps[:20]:
        print(f"  score={e['esc_score']:.4f} intent={e['intent']} pred={e['pred_intent']}")
        print(f"    {e['text'][:100]}")
        print(f"    tags: {e['tags']}")

    print(f"\nEscalation False Negatives: {len(esc_fns)}")
    for e in esc_fns[:20]:
        print(f"  score={e['esc_score']:.4f} intent={e['intent']} pred={e['pred_intent']}")
        print(f"    {e['text'][:100]}")
        print(f"    tags: {e['tags']}")

    # Categorize FNs
    print("\n--- FN Categories ---")
    fn_categories = Counter()
    for e in esc_fns:
        tags = e["tags"]
        text_lower = e["text"].lower()
        if any("hinglish" in t for t in tags):
            fn_categories["hinglish"] += 1
        elif any("multi_intent" in t for t in tags):
            fn_categories["multi_intent"] += 1
        elif any("noisy" in t for t in tags):
            fn_categories["noisy"] += 1
        elif e["intent"] == "human_request":
            fn_categories["human_request"] += 1
        elif "refund" in text_lower or "money" in text_lower or "charge" in text_lower:
            fn_categories["refund/charge"] += 1
        elif "wait" in text_lower or "week" in text_lower or "month" in text_lower or "time" in text_lower:
            fn_categories["time-persistent"] += 1
        elif "legal" in text_lower or "law" in text_lower or "sue" in text_lower or "chargeback" in text_lower:
            fn_categories["legal-threat"] += 1
        elif "hack" in text_lower or "unauthorized" in text_lower or "security" in text_lower:
            fn_categories["security"] += 1
        elif "manager" in text_lower or "supervisor" in text_lower or "senior" in text_lower:
            fn_categories["manager-request"] += 1
        else:
            fn_categories["other-subtle"] += 1

    for cat, count in fn_categories.most_common():
        print(f"  {cat}: {count}")

    # 5. Subset performance with detail
    print("\n" + "=" * 70)
    print("SECTION 5: SPECIAL SUBSET DETAILED ERRORS")
    print("=" * 70)

    subset_defs = {
        "hinglish": "hinglish",
        "confusion_pair": "confusion_pair",
        "hard_negative_escalation": "hard_negative_escalation",
        "multi_intent": "multi_intent",
        "noisy": "noisy",
    }

    for name, tag in subset_defs.items():
        mask = [any(tag in t for t in tl) for tl in tags_list]
        indices = [i for i, m in enumerate(mask) if m]
        if not indices:
            continue
        correct = sum(1 for i in indices if pred_intents[i] == true_intents[i])
        esc_correct = sum(1 for i in indices if pred_escs[i] == true_escs[i])
        print(f"\n  {name}: {len(indices)} examples, intent={correct}/{len(indices)} ({100*correct/len(indices):.1f}%), esc={esc_correct}/{len(indices)} ({100*esc_correct/len(indices):.1f}%)")

        # Show errors
        for i in indices[:8]:
            if pred_intents[i] != true_intents[i] or pred_escs[i] != true_escs[i]:
                tag = "ESC" if pred_escs[i] != true_escs[i] else "INT"
                print(f"    [{tag}] exp={true_intents[i]} pred={pred_intents[i]} esc={pred_escs[i]}({esc_scores[i]:.3f})")
                print(f"      {texts[i][:100]}")

    # 6. Escalation score distribution by intent
    print("\n" + "=" * 70)
    print("SECTION 6: ESCALATION SCORE DISTRIBUTION BY INTENT")
    print("=" * 70)

    for c in INTENT_CATEGORIES:
        idx = [i for i in range(len(texts)) if true_intents[i] == c]
        if not idx:
            continue
        scores = [esc_scores[i] for i in idx]
        esc_true = sum(1 for i in idx if true_escs[i] == 1)
        print(f"  {c} ({len(idx)} total, {esc_true} esc+): mean={np.mean(scores):.4f} median={np.median(scores):.4f}")

    # 7. Calibration (simple reliability)
    print("\n" + "=" * 70)
    print("SECTION 7: ESCALATION CALIBRATION (BINNED)")
    print("=" * 70)

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        idx = [j for j in range(len(texts)) if lo <= esc_scores[j] < hi]
        if not idx:
            continue
        avg_pred = np.mean([esc_scores[j] for j in idx])
        actual_pos = sum(1 for j in idx if true_escs[j] == 1) / len(idx)
        count = len(idx)
        print(f"  [{lo:.1f}-{hi:.1f}): n={count:3d}  avg_pred={avg_pred:.3f}  actual_pos={actual_pos:.3f}  gap={actual_pos-avg_pred:+.3f}")

    # 8. Summary
    print("\n" + "=" * 70)
    print("SECTION 8: SUMMARY FOR DATASET V3 PLANNING")
    print("=" * 70)

    intent_errors = sum(1 for i in range(len(texts)) if pred_intents[i] != true_intents[i])
    print(f"\nIntent accuracy: {(1 - intent_errors/len(texts))*100:.1f}%")
    print(f"Total intent errors: {intent_errors}")
    print(f"High-confidence intent errors: {len(high_conf_errors)}")
    print(f"Escalation FPs: {len(esc_fps)}, FNs: {len(esc_fns)}")

    # Save detailed errors for dataset generation reference
    errors_out = []
    for i in range(len(texts)):
        if pred_intents[i] != true_intents[i] or pred_escs[i] != true_escs[i]:
            errors_out.append({
                "text": texts[i],
                "true_intent": true_intents[i],
                "pred_intent": pred_intents[i],
                "intent_confidence": round(float(max(intent_probs[i])), 4),
                "true_esc": bool(true_escs[i]),
                "pred_esc": bool(pred_escs[i]),
                "esc_score": round(float(esc_scores[i]), 4),
                "tags": tags_list[i],
            })

    out_path = os.path.join(EVALS_DIR, "v2_error_analysis.json")
    with open(out_path, "w") as f:
        json.dump(errors_out, f, indent=2)
    print(f"\nDetailed errors saved to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
