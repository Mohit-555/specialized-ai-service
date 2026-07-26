import json
import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai_service.app.main import app
from ai_service.app.services.classifier import ClassifierService


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data


def test_classify_endpoint(client):
    response = client.post(
        "/v1/classify",
        json={"text": "I was charged twice and want my money back."},
    )
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "intent_confidence" in data
    assert "escalation_required" in data
    assert "escalation_confidence" in data
    assert "model_version" in data
    assert "top_intents" in data
    assert isinstance(data["intent_confidence"], float)
    assert isinstance(data["escalation_confidence"], float)
    assert isinstance(data["escalation_required"], bool)
    assert 0.0 <= data["intent_confidence"] <= 1.0
    assert 0.0 <= data["escalation_confidence"] <= 1.0


def test_classify_empty_text(client):
    response = client.post("/v1/classify", json={"text": ""})
    assert response.status_code == 422


def test_classify_whitespace_text(client):
    response = client.post("/v1/classify", json={"text": "   "})
    assert response.status_code == 422


def test_classify_missing_text(client):
    response = client.post("/v1/classify", json={})
    assert response.status_code == 422


def test_classify_invalid_json(client):
    response = client.post(
        "/v1/classify",
        data="not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "intent", "intent-v1", "pipeline.joblib")), reason="Model not trained")
def test_intent_prediction_different_texts(client):
    test_cases = [
        ("What is the price?", "pricing"),
        ("I need help logging in", "account_issue"),
        ("I want a refund", "refund"),
    ]
    for text, expected_intent in test_cases:
        response = client.post("/v1/classify", json={"text": text})
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] is not None


def test_threshold_configurable_via_env(client):
    with patch.dict(os.environ, {"ESCALATION_THRESHOLD": "0.99"}, clear=True):
        response = client.post(
            "/v1/classify",
            json={"text": "Give me my money back right now."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "escalation_required" in data


def test_health_returns_model_info(client):
    response = client.get("/health")
    data = response.json()
    if data["models_loaded"]:
        assert data["intent_model"] is not None
        assert data["escalation_model"] is not None


def test_classify_response_schema(client):
    response = client.post(
        "/v1/classify",
        json={"text": "Hello, I need help."},
    )
    assert response.status_code == 200
    data = response.json()
    expected_keys = {"intent", "intent_confidence", "escalation_required", "escalation_confidence", "model_version", "top_intents"}
    assert expected_keys.issubset(data.keys())
    assert {"intent", "escalation"}.issubset(data["model_version"].keys())
    if data["top_intents"]:
        assert {"intent", "confidence"}.issubset(data["top_intents"][0].keys())
