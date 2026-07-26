import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INTENT_MODEL_VERSION = os.environ.get("INTENT_MODEL_VERSION", "intent-v1")
ESCALATION_MODEL_VERSION = os.environ.get("ESCALATION_MODEL_VERSION", "escalation-v1")

def _model_path(version, is_embedding=False):
    if version.startswith("intent"):
        sub = "intent"
        if version == "intent-v1":
            fname = "pipeline.joblib"
        else:
            fname = "classifier.joblib" if not is_embedding else None
    elif version.startswith("escalation"):
        sub = "escalation"
        if version == "escalation-v1":
            fname = "pipeline.joblib"
        else:
            fname = "classifier.joblib" if not is_embedding else None
    else:
        raise ValueError(f"Unknown model version: {version}")
    return os.path.join(BASE_DIR, "models", sub, version, fname)

INTENT_MODEL_PATH = _model_path(INTENT_MODEL_VERSION)
ESCALATION_MODEL_PATH = _model_path(ESCALATION_MODEL_VERSION)

ESCALATION_THRESHOLD = float(os.environ.get("ESCALATION_THRESHOLD", "0.20"))
MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", "5000"))
TOP_K_INTENTS = 3
IS_V2 = INTENT_MODEL_VERSION.endswith("-v2") or ESCALATION_MODEL_VERSION.endswith("-v2")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
