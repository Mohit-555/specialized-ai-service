import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import inspect

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ai_service.app.config import (
    INTENT_MODEL_PATH,
    ESCALATION_MODEL_PATH,
    INTENT_MODEL_VERSION,
    ESCALATION_MODEL_VERSION,
    ESCALATION_THRESHOLD,
    TOP_K_INTENTS,
    IS_V2,
    IS_V4,
    EMBEDDING_MODEL_NAME,
    CLASSIFICATION_MAX_WORKERS,
    CLASSIFICATION_MAX_INFLIGHT,
)

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    intent: str
    confidence: float
    top_intents: list[dict]


@dataclass
class EscalationResult:
    required: bool
    confidence: float


@dataclass
class ClassifyResult:
    intent: str
    intent_confidence: float
    escalation_required: bool
    escalation_confidence: float
    top_intents: list[dict]
    model_version_intent: str
    model_version_escalation: str


class ClassifierService:
    def __init__(self):
        self.intent_model = None
        self.escalation_model = None
        self.intent_tokenizer = None
        self.escalation_tokenizer = None
        self.embedder = None
        self.intent_classes = None
        self.intent_model_version = INTENT_MODEL_VERSION
        self.escalation_model_version = ESCALATION_MODEL_VERSION
        self.is_v2 = IS_V2
        self.is_v4 = IS_V4
        max_workers = CLASSIFICATION_MAX_WORKERS
        max_inflight = CLASSIFICATION_MAX_INFLIGHT
        logger.info(f"Creating ThreadPoolExecutor with max_workers={max_workers}")
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"Creating admission semaphore with max_inflight={max_inflight}")
        self._semaphore = asyncio.BoundedSemaphore(max_inflight)
        self._load_models()

    @property
    def executor(self):
        return self._executor

    @property
    def semaphore(self):
        return self._semaphore

    def shutdown(self):
        self._executor.shutdown(wait=True)

    def _load_v4_model(self, model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        with open(f"{model_dir}/metadata.json") as f:
            metadata = json.load(f)
        classes = metadata.get("intent_classes")
        return model, tokenizer, classes

    @staticmethod
    def _accepts_token_type_ids(model):
        sig = inspect.signature(model.forward)
        return "token_type_ids" in sig.parameters

    def _load_models(self) -> None:
        try:
            if self.is_v4:
                logger.info(f"Loading V4 intent model from {INTENT_MODEL_PATH}...")
                self.intent_model, self.intent_tokenizer, self.intent_classes = self._load_v4_model(INTENT_MODEL_PATH)
                logger.info(f"V4 intent model loaded, classes={self.intent_classes}")
            else:
                if self.is_v2:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
                    self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
                    logger.info("Embedding model loaded")
                self.intent_model = joblib.load(INTENT_MODEL_PATH)
                logger.info(f"Intent model loaded from {INTENT_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load intent model: {e}")
            self.intent_model = None

        try:
            if self.is_v4:
                logger.info(f"Loading V4 escalation model from {ESCALATION_MODEL_PATH}...")
                self.escalation_model, self.escalation_tokenizer, _ = self._load_v4_model(ESCALATION_MODEL_PATH)
                logger.info("V4 escalation model loaded")
            else:
                self.escalation_model = joblib.load(ESCALATION_MODEL_PATH)
                logger.info(f"Escalation model loaded from {ESCALATION_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load escalation model: {e}")
            self.escalation_model = None

    @property
    def models_loaded(self) -> bool:
        return self.intent_model is not None and self.escalation_model is not None

    def _embed(self, texts: list[str]) -> np.ndarray | None:
        if not self.is_v2:
            return None
        return self.embedder.encode(texts)

    def _predict_proba_v4(self, model, tokenizer, text: str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        if "token_type_ids" in inputs and not self._accepts_token_type_ids(model):
            inputs.pop("token_type_ids")
        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits
        if logits.shape[-1] == 1:
            pos_prob = logits.sigmoid().item()
            return np.array([1 - pos_prob, pos_prob])
        return logits.softmax(dim=-1).detach().numpy()[0]

    def _predict_proba(self, model, text: str):
        if self.is_v2:
            emb = self.embedder.encode([text])
            return model.predict_proba(emb)[0]
        return model.predict_proba([text])[0]

    def predict_intent(self, text: str) -> IntentResult:
        if self.intent_model is None:
            raise RuntimeError("Intent model not loaded")

        if self.is_v4:
            probs = self._predict_proba_v4(self.intent_model, self.intent_tokenizer, text)
            classes = self.intent_classes
        else:
            probs = self._predict_proba(self.intent_model, text)
            classes = self.intent_model.classes_
        top_indices = np.argsort(probs)[-TOP_K_INTENTS:][::-1]

        top_intents = [
            {"intent": str(classes[i]), "confidence": round(float(probs[i]), 4)}
            for i in top_indices
        ]

        return IntentResult(
            intent=top_intents[0]["intent"],
            confidence=top_intents[0]["confidence"],
            top_intents=top_intents,
        )

    def predict_escalation(self, text: str) -> EscalationResult:
        if self.escalation_model is None:
            raise RuntimeError("Escalation model not loaded")

        if self.is_v4:
            probs = self._predict_proba_v4(self.escalation_model, self.escalation_tokenizer, text)
        else:
            probs = self._predict_proba(self.escalation_model, text)

        prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        return EscalationResult(
            required=prob >= ESCALATION_THRESHOLD,
            confidence=round(prob, 4),
        )

    def classify_sync(self, text: str) -> ClassifyResult:
        intent_result = self.predict_intent(text)
        escalation_result = self.predict_escalation(text)
        return ClassifyResult(
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            escalation_required=escalation_result.required,
            escalation_confidence=escalation_result.confidence,
            top_intents=intent_result.top_intents,
            model_version_intent=self.intent_model_version,
            model_version_escalation=self.escalation_model_version,
        )
