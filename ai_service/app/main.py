import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_service.app.config import CLASSIFICATION_REQUEST_TIMEOUT
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
    classifier_service.shutdown()


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

    if classifier_service.semaphore.locked():
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service at capacity",
                "detail": "Too many concurrent classification requests. Try again later.",
            },
        )

    await classifier_service.semaphore.acquire()
    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                classifier_service.executor,
                classifier_service.classify_sync,
                text,
            ),
            timeout=CLASSIFICATION_REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Request timeout",
                "detail": f"Classification did not complete within {CLASSIFICATION_REQUEST_TIMEOUT}s.",
            },
        )
    finally:
        classifier_service.semaphore.release()

    return ClassifyResponse(
        intent=result.intent,
        intent_confidence=result.intent_confidence,
        escalation_required=result.escalation_required,
        escalation_confidence=result.escalation_confidence,
        model_version=ModelVersion(
            intent=result.model_version_intent,
            escalation=result.model_version_escalation,
        ),
        top_intents=[
            IntentPrediction(intent=t["intent"], confidence=t["confidence"])
            for t in result.top_intents
        ],
    )
