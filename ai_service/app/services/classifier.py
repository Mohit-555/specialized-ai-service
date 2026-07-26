import logging
from dataclasses import dataclass

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from ai_service.app.config import (
    INTENT_MODEL_PATH,
    ESCALATION_MODEL_PATH,
    INTENT_MODEL_VERSION,
    ESCALATION_MODEL_VERSION,
    ESCALATION_THRESHOLD,
    TOP_K_INTENTS,
    IS_V2,
    EMBEDDING_MODEL_NAME,
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


class ClassifierService:
    def __init__(self):
        self.intent_model = None
        self.escalation_model = None
        self.embedder = None
        self.intent_model_version = INTENT_MODEL_VERSION
        self.escalation_model_version = ESCALATION_MODEL_VERSION
        self.is_v2 = IS_V2
        self._load_models()

    def _load_models(self) -> None:
        try:
            if self.is_v2:
                logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
                self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
                logger.info("Embedding model loaded")
            self.intent_model = joblib.load(INTENT_MODEL_PATH)
            logger.info(f"Intent model loaded from {INTENT_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load intent model: {e}")
            self.intent_model = None

        try:
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

    def _predict_proba(self, model, text: str):
        if self.is_v2:
            emb = self.embedder.encode([text])
            return model.predict_proba(emb)[0]
        else:
            return model.predict_proba([text])[0]

    def _predict(self, model, text: str):
        if self.is_v2:
            emb = self.embedder.encode([text])
            return model.predict(emb)[0]
        else:
            return model.predict([text])[0]

    def predict_intent(self, text: str) -> IntentResult:
        if self.intent_model is None:
            raise RuntimeError("Intent model not loaded")

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

        probs = self._predict_proba(self.escalation_model, text)
        prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        return EscalationResult(
            required=prob >= ESCALATION_THRESHOLD,
            confidence=round(prob, 4),
        )
