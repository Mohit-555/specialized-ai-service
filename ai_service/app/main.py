import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_service.app.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
    HealthResponse,
    IntentPrediction,
    ModelVersion,
)
from ai_service.app.services.classifier import ClassifierService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

classifier_service: ClassifierService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier_service
    logger.info("Starting up...")
    classifier_service = ClassifierService()
    if classifier_service.models_loaded:
        logger.info("Models loaded successfully")
    else:
        logger.warning("Models failed to load")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="AI Classification Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"},
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    global classifier_service
    if classifier_service is None:
        return HealthResponse(
            status="not ready",
            models_loaded=False,
        )
    return HealthResponse(
        status="healthy" if classifier_service.models_loaded else "degraded",
        models_loaded=classifier_service.models_loaded,
        intent_model=classifier_service.intent_model_version if classifier_service.models_loaded else None,
        escalation_model=classifier_service.escalation_model_version if classifier_service.models_loaded else None,
    )


@app.post("/v1/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    global classifier_service
    if classifier_service is None or not classifier_service.models_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Service unavailable", "detail": "Models not loaded"},
        )

    text = request.text
    if len(text) > 5000:
        return JSONResponse(
            status_code=400,
            content={"error": "Bad request", "detail": "Text exceeds maximum length of 5000 characters"},
        )

    intent_result = classifier_service.predict_intent(text)
    escalation_result = classifier_service.predict_escalation(text)

    return ClassifyResponse(
        intent=intent_result.intent,
        intent_confidence=intent_result.confidence,
        escalation_required=escalation_result.required,
        escalation_confidence=escalation_result.confidence,
        model_version=ModelVersion(
            intent=classifier_service.intent_model_version,
            escalation=classifier_service.escalation_model_version,
        ),
        top_intents=[
            IntentPrediction(intent=t["intent"], confidence=t["confidence"])
            for t in intent_result.top_intents
        ],
    )
