# AI Classification Service

Multi-version ML classification service: V1 (TF-IDF + LR), V2 (sentence embeddings + LR, dataset-v1), V3 (sentence embeddings + LR, dataset-v3).

**Model Versions:** intent-v1/2/3, escalation-v1/2/3
**Dataset V1:** 2,004 examples | **Dataset V3:** 3,726 examples (10 intent classes)

---

## Model Selection

Switch between V1 and V2 via environment variables:

| Variable | Default | Options |
|----------|---------|---------|
| `INTENT_MODEL_VERSION` | `intent-v1` | `intent-v1`, `intent-v2` |
| `ESCALATION_MODEL_VERSION` | `escalation-v1` | `escalation-v1`, `escalation-v2` |

V2 uses `all-MiniLM-L6-v2` sentence embeddings + Logistic Regression. No fine-tuning.

```bash
# Run with V2
INTENT_MODEL_VERSION=intent-v2 ESCALATION_MODEL_VERSION=escalation-v2 \
  uvicorn ai_service.app.main:app --host 0.0.0.0 --port 8000
```

## Performance

### Intent (10-class)

| Metric | V1 | V2 | Change |
|--------|----|----|--------|
| Accuracy | 62.8% | 76.1% | **+13.3%** |
| Macro F1 | 61.2% | 75.5% | **+14.3%** |

### Escalation (binary)

V1 @ threshold 0.20 (recommended): recall 82.0%, FP 49, FN 16
V2 @ threshold 0.25 (recommended): recall 86.5%, FP 46, FN 12

V2 improves separation from 0.250 to 0.368 (+47%).

### Subset Comparison

| Subset | Count | V1 Intent | V2 Intent | V1 Esc | V2 Esc |
|--------|-------|-----------|-----------|--------|--------|
| hinglish | 77 | 66.2% | 71.4% | 85.7% | 81.8% |
| confusion_pair | 17 | 47.1% | 76.5% | 70.6% | 76.5% |
| hard_negative_escalation | 17 | 41.2% | 64.7% | 70.6% | 47.1% |
| multi_intent | 20 | 55.0% | 65.0% | 60.0% | 70.0% |
| noisy | 27 | 51.9% | 66.7% | 70.4% | 74.1% |

Full V1 vs V2 comparison: `python3 training/compare_v1_v2.py`

---

## Development

```bash
pip install -r requirements-training.txt
```

## Dataset

Schema: `text`, `intent`, `escalation_required`, `tags`

- 2004 examples across 10 intent classes
- Linguistic diversity: Standard English, Indian English, Hinglish, noisy text
- Special subsets: hinglish, confusion pairs, hard negatives, multi-intent, noisy
- Split: ~65/15/20 train/validation/test (stratified)

See `annotation_policy.md` for detailed annotation rules.

### Create dataset

```bash
python3 training/create_dataset.py
```

### Validate dataset

```bash
python3 training/validate_dataset.py
```

### Split dataset

```bash
python3 training/split_dataset.py
```

## Training

### V1 (TF-IDF + Logistic Regression)

```bash
python3 training/train_intent.py
python3 training/train_escalation.py
```

Output: `models/intent/intent-v1/pipeline.joblib`, `models/escalation/escalation-v1/pipeline.joblib`

### V2 (Sentence Embeddings + Logistic Regression)

```bash
python3 training/train_intent_v2.py
python3 training/train_escalation_v2.py
```

Output: `models/intent/intent-v2/classifier.joblib`, `models/escalation/escalation-v2/classifier.joblib`

## Evaluation

```bash
# V1 only
python3 training/evaluate.py

# V1 vs V2 comparison
python3 training/compare_v1_v2.py
```

Output: `evaluations/evaluation-v1/`, `evaluations/evaluation-v2/`

## Running API

### V1 (default)

```bash
export ESCALATION_THRESHOLD=0.20
uvicorn ai_service.app.main:app --host 0.0.0.0 --port 8000
```

### V2

```bash
export INTENT_MODEL_VERSION=intent-v2 ESCALATION_MODEL_VERSION=escalation-v2
uvicorn ai_service.app.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

**GET /health**

```json
{"status": "healthy", "models_loaded": true, "intent_model": "intent-v1", "escalation_model": "escalation-v1"}
```

**POST /v1/classify**

Request:
```json
{"text": "I was charged twice and want my money back."}
```

Response:
```json
{
  "intent": "refund",
  "intent_confidence": 0.94,
  "escalation_required": true,
  "escalation_confidence": 0.89,
  "model_version": {"intent": "intent-v1", "escalation": "escalation-v1"},
  "top_intents": [...]
}
```

## Docker

### Build

```bash
docker build -t ai-classification-service .
```

### Run (V1)

```bash
docker run -p 8000:8000 -e ESCALATION_THRESHOLD=0.20 ai-classification-service
```

### Run (V2)

```bash
docker run -p 8000:8000 \
  -e INTENT_MODEL_VERSION=intent-v2 \
  -e ESCALATION_MODEL_VERSION=escalation-v2 \
  ai-classification-service
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ESCALATION_THRESHOLD` | 0.20 | Escalation probability threshold |
| `MAX_INPUT_LENGTH` | 5000 | Maximum input text length |
| `INTENT_MODEL_VERSION` | intent-v1 | Intent model version (v1/v2/v3) |
| `ESCALATION_MODEL_VERSION` | escalation-v1 | Escalation model version (v1/v2/v3) |
| `EMBEDDING_MODEL_NAME` | all-MiniLM-L6-v2 | Sentence transformer model (V2/V3) |

## Model Versioning

Models are stored in versioned directories:
- `models/intent/intent-v1/`, `models/intent/intent-v2/`, `models/intent/intent-v3/`
- `models/escalation/escalation-v1/`, `models/escalation/escalation-v2/`, `models/escalation/escalation-v3/`

## Testing

```bash
python3 -m pytest tests/ -v
```

51 tests: dataset validation, API endpoints, model artifact loading (V1/V2/V3).
