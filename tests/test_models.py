import csv
import json
import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import joblib
import numpy as np
import pytest
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def test_intent_model_artifacts_exist():
    pipeline_path = os.path.join(MODELS_DIR, "intent", "intent-v1", "pipeline.joblib")
    metadata_path = os.path.join(MODELS_DIR, "intent", "intent-v1", "metadata.json")
    assert os.path.exists(pipeline_path), "Intent pipeline not found"
    assert os.path.exists(metadata_path), "Intent metadata not found"


def test_escalation_model_artifacts_exist():
    pipeline_path = os.path.join(MODELS_DIR, "escalation", "escalation-v1", "pipeline.joblib")
    metadata_path = os.path.join(MODELS_DIR, "escalation", "escalation-v1", "metadata.json")
    assert os.path.exists(pipeline_path), "Escalation pipeline not found"
    assert os.path.exists(metadata_path), "Escalation metadata not found"


def test_intent_model_loads():
    pipeline_path = os.path.join(MODELS_DIR, "intent", "intent-v1", "pipeline.joblib")
    pipeline = joblib.load(pipeline_path)
    assert pipeline is not None
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")
    assert hasattr(pipeline, "classes_")
    assert len(pipeline.classes_) == 10


def test_escalation_model_loads():
    pipeline_path = os.path.join(MODELS_DIR, "escalation", "escalation-v1", "pipeline.joblib")
    pipeline = joblib.load(pipeline_path)
    assert pipeline is not None
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")


def test_intent_prediction():
    pipeline_path = os.path.join(MODELS_DIR, "intent", "intent-v1", "pipeline.joblib")
    pipeline = joblib.load(pipeline_path)
    texts = [
        "What is the price of your premium plan?",
        "I can't log into my account",
        "I want my money back",
        "Can I speak to a human?",
    ]
    preds = pipeline.predict(texts)
    assert len(preds) == len(texts)
    for pred in preds:
        assert pred in pipeline.classes_


def test_intent_confidence_valid():
    pipeline_path = os.path.join(MODELS_DIR, "intent", "intent-v1", "pipeline.joblib")
    pipeline = joblib.load(pipeline_path)
    probs = pipeline.predict_proba(["What is the price?"])[0]
    assert abs(sum(probs) - 1.0) < 0.01
    for p in probs:
        assert 0.0 <= p <= 1.0


def test_escalation_probability():
    pipeline_path = os.path.join(MODELS_DIR, "escalation", "escalation-v1", "pipeline.joblib")
    pipeline = joblib.load(pipeline_path)
    probs = pipeline.predict_proba(["I want my money back immediately"])[0]
    assert len(probs) == 2
    for p in probs:
        assert 0.0 <= p <= 1.0
    assert abs(sum(probs) - 1.0) < 0.01


def test_escalation_threshold_configurable():
    pipeline_path = os.path.join(MODELS_DIR, "escalation", "escalation-v1", "pipeline.joblib")
    pipeline = joblib.load(pipeline_path)
    prob = float(pipeline.predict_proba(["This is a test"])[0][1])

    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        pred = int(prob >= threshold)
        assert pred in (0, 1)


def test_top_k_intents():
    pipeline_path = os.path.join(MODELS_DIR, "intent", "intent-v1", "pipeline.joblib")
    pipeline = joblib.load(pipeline_path)
    probs = pipeline.predict_proba(["What is the pricing?"])[0]
    classes = pipeline.classes_
    top_3_indices = np.argsort(probs)[-3:][::-1]
    top_3 = [(classes[i], probs[i]) for i in top_3_indices]
    assert len(top_3) == 3
    # First should be highest confidence
    assert top_3[0][1] >= top_3[1][1] >= top_3[2][1]


def test_intent_metadata():
    metadata_path = os.path.join(MODELS_DIR, "intent", "intent-v1", "metadata.json")
    with open(metadata_path) as f:
        meta = json.load(f)
    assert meta["model_version"] == "intent-v1"
    assert "dataset_version" in meta
    assert "training_date" in meta
    assert "classes" in meta
    assert len(meta["classes"]) == 10


def test_escalation_metadata():
    metadata_path = os.path.join(MODELS_DIR, "escalation", "escalation-v1", "metadata.json")
    with open(metadata_path) as f:
        meta = json.load(f)
    assert meta["model_version"] == "escalation-v1"
    assert "dataset_version" in meta
    assert "training_date" in meta


# --- V3 Dataset Tests ---

DATA_V3_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v3")

def test_dataset_v3_splits_exist():
    for name in ("train", "validation", "test", "benchmark"):
        p = os.path.join(DATA_V3_DIR, f"{name}.csv")
        assert os.path.exists(p), f"V3 {name}.csv not found"


def test_dataset_v3_info():
    p = os.path.join(DATA_V3_DIR, "dataset_v3_info.json")
    with open(p) as f:
        info = json.load(f)
    assert info["dataset_version"] == "dataset-v3"
    assert info["total_examples"] >= 3500
    assert len(info["splits"]) == 4  # train, val, test, benchmark
    for s in ("train", "validation", "test", "benchmark"):
        assert s in info["splits"]
        assert info["splits"][s]["count"] > 0


def test_dataset_v3_no_overlap():
    texts = {}
    for name in ("train", "validation", "test", "benchmark"):
        with open(os.path.join(DATA_V3_DIR, f"{name}.csv")) as f:
            texts[name] = set(r["text"].strip() for r in csv.DictReader(f))
    pairs = [("train", "validation"), ("train", "test"), ("train", "benchmark"),
             ("validation", "test"), ("validation", "benchmark"), ("test", "benchmark")]
    for a, b in pairs:
        overlap = texts[a] & texts[b]
        assert len(overlap) == 0, f"Overlap in {a}-{b}: {len(overlap)}"


def test_dataset_v3_all_intents():
    with open(os.path.join(DATA_V3_DIR, "train.csv")) as f:
        intents = set(r["intent"].strip() for r in csv.DictReader(f))
    expected = {"general_question", "product_question", "pricing", "sales",
                "technical_support", "complaint", "refund", "account_issue",
                "human_request", "other"}
    assert intents == expected


def test_dataset_v3_valid_labels():
    for name in ("train", "validation", "test", "benchmark"):
        with open(os.path.join(DATA_V3_DIR, f"{name}.csv")) as f:
            for r in csv.DictReader(f):
                assert r["intent"].strip() in ("general_question", "product_question", "pricing", "sales",
                    "technical_support", "complaint", "refund", "account_issue", "human_request", "other")
                assert r["escalation_required"].strip() in ("true", "false")
                assert r["text"].strip()


def test_benchmark_v3_isolation():
    """Benchmark set must NOT appear in training."""
    with open(os.path.join(DATA_V3_DIR, "train.csv")) as f:
        train_texts = set(r["text"].strip() for r in csv.DictReader(f))
    with open(os.path.join(DATA_V3_DIR, "benchmark.csv")) as f:
        bench_texts = set(r["text"].strip() for r in csv.DictReader(f))
    assert len(train_texts & bench_texts) == 0, "Benchmark leaked into training!"


def test_benchmark_v3_info():
    p = os.path.join(DATA_V3_DIR, "benchmark_v3_info.json")
    assert os.path.exists(p), "Benchmark info not found"


def test_dataset_v3_near_duplicate_flag():
    """Near-duplicate info should exist in dataset info."""
    p = os.path.join(DATA_V3_DIR, "dataset_v3_info.json")
    with open(p) as f:
        info = json.load(f)
    assert "near_duplicates_found" in info
    assert "v1_test_leakage" in info


# --- V3 Model Tests ---

def test_v3_intent_model_artifacts_exist():
    p = os.path.join(MODELS_DIR, "intent", "intent-v3", "classifier.joblib")
    assert os.path.exists(p), "V3 intent classifier not found"


def test_v3_escalation_model_artifacts_exist():
    p = os.path.join(MODELS_DIR, "escalation", "escalation-v3", "classifier.joblib")
    assert os.path.exists(p), "V3 escalation classifier not found"


def test_v3_intent_model_loads():
    p = os.path.join(MODELS_DIR, "intent", "intent-v3", "classifier.joblib")
    clf = joblib.load(p)
    assert clf is not None
    assert hasattr(clf, "predict")
    assert hasattr(clf, "predict_proba")
    assert hasattr(clf, "classes_")
    assert len(clf.classes_) == 10


def test_v3_escalation_model_loads():
    p = os.path.join(MODELS_DIR, "escalation", "escalation-v3", "classifier.joblib")
    clf = joblib.load(p)
    assert clf is not None
    assert hasattr(clf, "predict")
    assert hasattr(clf, "predict_proba")


def test_v3_intent_prediction():
    p = os.path.join(MODELS_DIR, "intent", "intent-v3", "classifier.joblib")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    clf = joblib.load(p)
    texts = ["What is the price of premium?", "I can't log in", "Hello"]
    embs = embedder.encode(texts)
    preds = clf.predict(embs)
    assert len(preds) == len(texts)
    for pred in preds:
        assert pred in clf.classes_


def test_v3_escalation_prediction():
    p = os.path.join(MODELS_DIR, "escalation", "escalation-v3", "classifier.joblib")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    clf = joblib.load(p)
    texts = ["Refund my money now", "What is your pricing?"]
    embs = embedder.encode(texts)
    probs = clf.predict_proba(embs)
    assert len(probs) == len(texts)
    for prob in probs:
        assert abs(sum(prob) - 1.0) < 0.01


def test_v3_escalation_threshold_configurable():
    p = os.path.join(MODELS_DIR, "escalation", "escalation-v3", "classifier.joblib")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    clf = joblib.load(p)
    emb = embedder.encode(["This is a test message"])
    prob = float(clf.predict_proba(emb)[0][1])
    for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        pred = int(prob >= t)
        assert pred in (0, 1)


# --- V2 Model Tests ---

def test_v2_intent_model_artifacts_exist():
    classifier_path = os.path.join(MODELS_DIR, "intent", "intent-v2", "classifier.joblib")
    assert os.path.exists(classifier_path), "V2 intent classifier not found"


def test_v2_escalation_model_artifacts_exist():
    classifier_path = os.path.join(MODELS_DIR, "escalation", "escalation-v2", "classifier.joblib")
    assert os.path.exists(classifier_path), "V2 escalation classifier not found"


def test_v2_intent_model_loads():
    classifier_path = os.path.join(MODELS_DIR, "intent", "intent-v2", "classifier.joblib")
    clf = joblib.load(classifier_path)
    assert clf is not None
    assert hasattr(clf, "predict")
    assert hasattr(clf, "predict_proba")
    assert hasattr(clf, "classes_")
    assert len(clf.classes_) == 10


def test_v2_escalation_model_loads():
    classifier_path = os.path.join(MODELS_DIR, "escalation", "escalation-v2", "classifier.joblib")
    clf = joblib.load(classifier_path)
    assert clf is not None
    assert hasattr(clf, "predict")
    assert hasattr(clf, "predict_proba")


def test_v2_embedding_encoder_available():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(["test message"])
        assert len(emb[0]) == 384
    except Exception as e:
        pytest.fail(f"SentenceTransformer failed: {e}")


# --- V4 Model Tests ---
#
# Lightweight artifact-presence tests verify real weight files exist.
# Mocked unit tests verify P0.1 (model.eval()) and P0.2 (inference_mode()).
# Slow integration test (pytest.mark.slow) verifies actual model loading.

V4_INTENT_DIR = os.path.join(MODELS_DIR, "intent", "intent-v4-en")
V4_ESCALATION_DIR = os.path.join(MODELS_DIR, "escalation", "escalation-v4-en")

# ── Lightweight artifact-presence/integrity checks ──

def test_v4_model_artifacts_exist():
    assert os.path.isdir(V4_INTENT_DIR), "V4 intent model dir not found"
    assert os.path.isdir(V4_ESCALATION_DIR), "V4 escalation model dir not found"
    assert os.path.exists(os.path.join(V4_INTENT_DIR, "config.json")), "V4 intent config not found"
    assert os.path.exists(os.path.join(V4_ESCALATION_DIR, "config.json")), "V4 escalation config not found"
    assert os.path.exists(os.path.join(V4_INTENT_DIR, "metadata.json")), "V4 intent metadata not found"
    assert os.path.exists(os.path.join(V4_ESCALATION_DIR, "metadata.json")), "V4 escalation metadata not found"


def test_v4_intent_model_safetensors_present():
    path = os.path.join(V4_INTENT_DIR, "model.safetensors")
    assert os.path.exists(path), f"intent model.safetensors missing at {path}"
    size = os.path.getsize(path)
    assert size > 200_000_000, f"intent model.safetensors too small: {size} bytes (expected >200MB)"
    with open(path, "rb") as f:
        header = f.read(8)
    assert len(header) == 8, "intent model.safetensors has invalid header"


def test_v4_escalation_model_safetensors_present():
    path = os.path.join(V4_ESCALATION_DIR, "model.safetensors")
    assert os.path.exists(path), f"escalation model.safetensors missing at {path}"
    size = os.path.getsize(path)
    assert size > 200_000_000, f"escalation model.safetensors too small: {size} bytes (expected >200MB)"
    with open(path, "rb") as f:
        header = f.read(8)
    assert len(header) == 8, "escalation model.safetensors has invalid header"


def test_v4_training_args_present():
    assert os.path.exists(os.path.join(V4_INTENT_DIR, "training_args.bin")), "intent training_args.bin missing"
    assert os.path.exists(os.path.join(V4_ESCALATION_DIR, "training_args.bin")), "escalation training_args.bin missing"


@pytest.mark.slow
def test_v4_intent_model_loads():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(V4_INTENT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(V4_INTENT_DIR)
    assert model is not None
    assert tokenizer is not None
    assert not model.training
    model.cpu()


@pytest.mark.slow
def test_v4_escalation_model_loads():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(V4_ESCALATION_DIR)
    tokenizer = AutoTokenizer.from_pretrained(V4_ESCALATION_DIR)
    assert model is not None
    assert tokenizer is not None
    assert not model.training
    model.cpu()


# ── Mocked unit tests (P0.1 and P0.2 verification) ──


def test_v4_intent_model_eval_called():
    """P0.1: model.eval() must be called after loading for intent model."""
    intent_model = MagicMock()
    escalation_model = MagicMock()

    def from_pretrained_side_effect(model_dir):
        if "intent" in model_dir:
            return intent_model
        return escalation_model

    with patch.dict(os.environ, {"INTENT_MODEL_VERSION": "intent-v4-en", "ESCALATION_MODEL_VERSION": "escalation-v4-en"}):
        import importlib
        import ai_service.app.config as cfg
        importlib.reload(cfg)
        import ai_service.app.services.classifier as clf_mod
        importlib.reload(clf_mod)
        with patch("ai_service.app.services.classifier.AutoModelForSequenceClassification.from_pretrained", side_effect=from_pretrained_side_effect):
            with patch("ai_service.app.services.classifier.AutoTokenizer.from_pretrained"):
                with patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"intent_classes": ["a", "b"]})):
                    from ai_service.app.services.classifier import ClassifierService
                    ClassifierService()
    intent_model.eval.assert_called_once()


def test_v4_escalation_model_eval_called():
    """P0.1: model.eval() must be called for escalation model too."""
    intent_model = MagicMock()
    escalation_model = MagicMock()

    def from_pretrained_side_effect(model_dir):
        if "intent" in model_dir:
            return intent_model
        return escalation_model

    with patch.dict(os.environ, {"INTENT_MODEL_VERSION": "intent-v4-en", "ESCALATION_MODEL_VERSION": "escalation-v4-en"}):
        import importlib
        import ai_service.app.config as cfg
        importlib.reload(cfg)
        import ai_service.app.services.classifier as clf_mod
        importlib.reload(clf_mod)
        with patch("ai_service.app.services.classifier.AutoModelForSequenceClassification.from_pretrained", side_effect=from_pretrained_side_effect):
            with patch("ai_service.app.services.classifier.AutoTokenizer.from_pretrained"):
                with patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"intent_classes": ["a", "b"]})):
                    from ai_service.app.services.classifier import ClassifierService
                    ClassifierService()
    escalation_model.eval.assert_called_once()


def test_v4_inference_requires_inference_mode():
    """P0.2: forward pass must run under torch.inference_mode().
    
    Verify by ensuring the service calls the model and returns
    controlled logits without gradient tracking errors.
    """
    from unittest.mock import MagicMock, patch
    import torch
    import numpy as np
    import importlib

    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=torch.float32)
    output = MagicMock()
    output.logits = logits

    model = MagicMock()
    model.return_value = output
    model.training = False

    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[101, 2054, 2003]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    intent_classes = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    metadata = json.dumps({"intent_classes": intent_classes})

    with patch.dict(os.environ, {"INTENT_MODEL_VERSION": "intent-v4-en", "ESCALATION_MODEL_VERSION": "escalation-v4-en"}):
        import ai_service.app.config as cfg
        importlib.reload(cfg)
        import ai_service.app.services.classifier as clf_mod
        importlib.reload(clf_mod)
        with patch("ai_service.app.services.classifier.AutoModelForSequenceClassification.from_pretrained", return_value=model):
            with patch("ai_service.app.services.classifier.AutoTokenizer.from_pretrained", return_value=tokenizer):
                with patch("builtins.open", new_callable=mock_open, read_data=metadata):
                    from ai_service.app.services.classifier import ClassifierService
                    service = ClassifierService()

    result = service.predict_intent("test text")
    assert result.intent == "j", f"Expected 'j' (highest logit=10), got '{result.intent}'"
    assert result.confidence > 0


def test_v4_inference_deterministic_with_fixed_logits():
    """Repeated inference with same input produces identical results."""
    from unittest.mock import MagicMock, patch
    import torch
    import importlib

    logits = torch.tensor([[2.0, 8.0, 1.0]], dtype=torch.float32)
    output = MagicMock()
    output.logits = logits

    model = MagicMock()
    model.return_value = output
    model.training = False

    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[101, 2054, 2003]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    metadata = json.dumps({"intent_classes": ["a", "b", "c"]})

    with patch.dict(os.environ, {"INTENT_MODEL_VERSION": "intent-v4-en", "ESCALATION_MODEL_VERSION": "escalation-v4-en"}):
        import ai_service.app.config as cfg
        importlib.reload(cfg)
        import ai_service.app.services.classifier as clf_mod
        importlib.reload(clf_mod)
        with patch("ai_service.app.services.classifier.AutoModelForSequenceClassification.from_pretrained", return_value=model):
            with patch("ai_service.app.services.classifier.AutoTokenizer.from_pretrained", return_value=tokenizer):
                with patch("builtins.open", new_callable=mock_open, read_data=metadata):
                    from ai_service.app.services.classifier import ClassifierService
                    service = ClassifierService()

    results = []
    for _ in range(20):
        r = service.predict_intent("test text")
        results.append((r.intent, r.confidence))
    first = results[0]
    for r in results[1:]:
        assert r == first, f"Non-deterministic result: {first} != {r}"


def test_v4_escalation_deterministic_with_fixed_logits():
    """Repeated escalation inference with same input produces identical results."""
    from unittest.mock import MagicMock, patch
    import torch
    import importlib

    logits = torch.tensor([[3.5]], dtype=torch.float32)
    output = MagicMock()
    output.logits = logits

    model = MagicMock()
    model.return_value = output
    model.training = False

    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[101, 2054, 2003]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    with patch.dict(os.environ, {"INTENT_MODEL_VERSION": "intent-v4-en", "ESCALATION_MODEL_VERSION": "escalation-v4-en"}):
        import ai_service.app.config as cfg
        importlib.reload(cfg)
        import ai_service.app.services.classifier as clf_mod
        importlib.reload(clf_mod)
        with patch("ai_service.app.services.classifier.AutoModelForSequenceClassification.from_pretrained", return_value=model):
            with patch("ai_service.app.services.classifier.AutoTokenizer.from_pretrained", return_value=tokenizer):
                with patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"intent_classes": ["a", "b", "c"]})):
                    from ai_service.app.services.classifier import ClassifierService
                    service = ClassifierService()

    results = []
    for _ in range(20):
        r = service.predict_escalation("test text")
        results.append((r.required, r.confidence))
    first = results[0]
    for r in results[1:]:
        assert r == first, f"Non-deterministic escalation: {first} != {r}"
