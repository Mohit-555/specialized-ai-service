import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INTENT_MODEL_VERSION = os.environ.get("INTENT_MODEL_VERSION", "intent-v1")
ESCALATION_MODEL_VERSION = os.environ.get("ESCALATION_MODEL_VERSION", "escalation-v1")

IS_V4 = "v4" in INTENT_MODEL_VERSION or "v4" in ESCALATION_MODEL_VERSION
IS_V2 = (INTENT_MODEL_VERSION.endswith("-v2") or ESCALATION_MODEL_VERSION.endswith("-v2")) and not IS_V4

CLASSIFICATION_MAX_WORKERS = int(os.environ.get("CLASSIFICATION_MAX_WORKERS", "2"))
CLASSIFICATION_MAX_INFLIGHT = int(os.environ.get("CLASSIFICATION_MAX_INFLIGHT", "6"))
CLASSIFICATION_REQUEST_TIMEOUT = float(os.environ.get("CLASSIFICATION_REQUEST_TIMEOUT", "30.0"))

def _model_path(version):
    if version.startswith("intent"):
        sub = "intent"
    elif version.startswith("escalation"):
        sub = "escalation"
    else:
        raise ValueError(f"Unknown model version: {version}")
    base = os.path.join(BASE_DIR, "models", sub, version)
    if "v4" in version:
        return base
    fname = "pipeline.joblib" if version == f"{sub}-v1" else "classifier.joblib"
    return os.path.join(base, fname)

INTENT_MODEL_PATH = _model_path(INTENT_MODEL_VERSION)
ESCALATION_MODEL_PATH = _model_path(ESCALATION_MODEL_VERSION)

ESCALATION_THRESHOLD = float(os.environ.get("ESCALATION_THRESHOLD", "0.20"))
MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", "5000"))
TOP_K_INTENTS = 3
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
