from pydantic import BaseModel, Field, validator


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

    @validator("text")
    def text_must_be_valid(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be empty or whitespace only")
        return stripped


class IntentPrediction(BaseModel):
    intent: str
    confidence: float


class ModelVersion(BaseModel):
    intent: str
    escalation: str


class ClassifyResponse(BaseModel):
    intent: str
    intent_confidence: float
    escalation_required: bool
    escalation_confidence: float
    model_version: ModelVersion
    top_intents: list[IntentPrediction] = []


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    intent_model: str | None = None
    escalation_model: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
