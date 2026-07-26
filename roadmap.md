# Build a Standalone Specialized AI Classification Service

I want you to build a complete, independently deployable **specialized machine-learning classification service**.

This project must NOT depend on Django, Nexora, or any other existing application.

The goal is to build, train, evaluate, test, version, and serve our own small specialized ML models.

Do not use OpenAI, Claude, Gemini, or another LLM for inference.

Do not train a generative LLM.

For V1, use traditional machine learning:

* Python
* scikit-learn
* TF-IDF
* Logistic Regression
* FastAPI
* Pydantic
* pytest

The architecture must allow us to replace the models with more advanced models later without changing the API contract.

---

# 1. MODEL OBJECTIVES

Build two independent classifiers.

## Model A — Intent Classifier

Given a customer message, classify it into exactly one primary intent.

Initial intents:

* general_question
* product_question
* pricing
* sales
* technical_support
* complaint
* refund
* account_issue
* human_request
* other

Example:

Input:

"What is the price of your premium package?"

Output:

{
"intent": "pricing",
"confidence": 0.94
}

The model should also be capable of returning the top predicted intents and probabilities internally for debugging/evaluation.

---

# 2. ESCALATION CLASSIFIER

Build a separate binary classifier that predicts whether the message should require human attention.

Example:

Input:

"I was charged twice and I want my money back."

Output:

{
"escalation_required": true,
"escalation_confidence": 0.89
}

Intent and escalation MUST remain separate predictions.

For example:

"What is your refund policy?"

could be:

intent = refund
escalation_required = false

while:

"I've been waiting two weeks. Refund my payment immediately."

could be:

intent = refund
escalation_required = true

Do not implement escalation as simple intent-based rules.

---

# 3. ANNOTATION POLICY

Before generating the dataset, create a clear annotation policy.

Define precisely what each intent means and how ambiguous messages should be labeled.

Pay particular attention to boundaries such as:

* human_request vs sales
* human_request vs complaint
* complaint vs refund
* account_issue vs complaint
* account_issue vs pricing
* product_question vs general_question
* sales vs pricing
* other vs general_question

Define how to handle multi-intent messages.

For V1, each message must have one primary intent.

Document how the primary intent is selected when multiple intents are present.

Also define exactly what should and should not trigger escalation.

The dataset must follow this policy consistently.

---

# 4. DATASET

Create a high-quality dataset of approximately **2,000–3,000 examples**.

Do NOT create hundreds of trivial paraphrases.

Every example should contain:

* text
* intent
* escalation_required

Optional metadata is encouraged:

* language_style
* difficulty
* scenario_type
* notes

The dataset should contain realistic customer messages.

Include:

* Standard English
* Indian English
* Hinglish
* grammatical mistakes
* spelling mistakes
* very short messages
* long messages
* polite messages
* angry messages
* indirect requests
* negation
* ambiguous messages
* multi-intent messages
* hard negatives
* difficult intent boundaries

Examples of linguistic diversity:

"price?"

"how much does it cost"

"premium plan ka price kya hai"

"can someone explain the pricing?"

"payment kat gaya but account activate nahi hua"

"I don't want a refund, just explain your refund policy."

"your service is terrible but I don't need anyone to contact me"

"human se baat karni hai"

Avoid creating datasets where the model can solve classification purely from obvious keywords.

---

# 5. HARD NEGATIVES

Hard negatives are especially important.

Create messages containing escalation-related words that should NOT actually escalate.

Examples:

"I don't want a refund. I only want to know the refund policy."

"Nobody needs to contact me, I just have a question."

"My payment failed yesterday but it's working now."

"I was angry earlier but the issue has been resolved."

Also include messages that require escalation without obvious keywords.

The escalation classifier must learn context rather than simply:

refund → escalate

complaint → escalate

human → escalate

---

# 6. DATASET SPLITTING

Create strict:

* training set
* validation set
* test set

Use stratification where appropriate.

Every intent must appear in every split.

There must be ZERO exact duplicate leakage across splits.

Also check for obvious near-duplicate/paraphrase leakage where practical.

The test set must be meaningfully large.

Prefer approximately:

60–70% training
15–20% validation
20–25% test

Do not optimize directly against the final test set.

Include special test subsets/tags for:

* Hinglish
* confusion-pair examples
* escalation hard negatives
* multi-intent examples
* spelling/noisy examples

---

# 7. DATASET VALIDATION

Create a validation script that checks:

* missing text
* empty text
* invalid intent
* invalid escalation label
* duplicate messages
* class distribution
* train/validation/test overlap
* every intent exists in every split
* escalation-positive/negative distribution

Print a useful validation report.

Fail validation when serious dataset problems exist.

---

# 8. INTENT MODEL

Build the first intent baseline using:

TF-IDF
→ Logistic Regression

Use an sklearn Pipeline.

The model should produce:

* predicted intent
* confidence
* top-k predictions/probabilities

Persist the trained pipeline.

Do not retrain when the API starts.

Use explicit model versions, for example:

intent-v1

Store useful model metadata such as:

* model version
* training date
* dataset version
* classes
* important training configuration

---

# 9. ESCALATION MODEL

Build a separate binary classifier using:

TF-IDF
→ Logistic Regression

Return escalation probability.

The decision threshold MUST NOT be permanently embedded inside the trained model.

Make it configurable, for example:

ESCALATION_THRESHOLD=0.40

The API should determine:

escalation_required =
probability >= threshold

This allows threshold tuning without retraining.

---

# 10. MODEL TRAINING

Create reproducible training scripts.

Conceptually:

training/
train_intent.py
train_escalation.py

Set random seeds where appropriate.

Training should output:

* model artifact
* model metadata
* training summary

Keep training dependencies separate from runtime dependencies where practical.

The production inference code must not import training scripts.

---

# 11. INTENT EVALUATION

Evaluate against the untouched test set.

Report:

* accuracy
* macro precision
* macro recall
* macro F1
* weighted F1
* per-class precision
* per-class recall
* per-class F1
* support
* confusion matrix

Identify:

* worst-performing intents
* strongest intents
* major confusion pairs

Do NOT claim the model is production-ready based only on accuracy.

Save evaluation results as machine-readable JSON.

---

# 12. ESCALATION EVALUATION

Escalation safety is especially important.

Report:

* precision
* recall
* F1
* false positives
* false negatives
* confusion matrix

Run a threshold sweep.

At minimum evaluate:

0.20
0.25
0.30
0.35
0.40
0.45
0.50
0.55
0.60
0.65
0.70

Produce a table containing:

threshold
precision
recall
F1
false positives
false negatives

Missing a genuine escalation is considered more serious than unnecessarily escalating a safe message.

However, do NOT automatically choose the threshold that produces 100% recall if it produces an unusable false-positive rate.

Recommend a threshold based on the measured tradeoff and explain why.

---

# 13. SPECIAL SUBSET EVALUATION

Evaluate model performance separately on tagged difficult subsets:

* Hinglish
* confusion pairs
* hard-negative escalation
* multi-intent
* noisy/spelling-error messages

Report performance separately.

Example:

Overall intent accuracy: ...
Hinglish accuracy: ...
Confusion-pair accuracy: ...
Multi-intent accuracy: ...

For escalation:

Overall recall: ...
Hard-negative precision: ...
Hinglish recall: ...

This is important because high overall performance can hide weak edge cases.

---

# 14. ERROR ANALYSIS

Automatically produce an error-analysis report.

For every incorrect test prediction, record:

* message
* expected intent
* predicted intent
* confidence
* expected escalation
* predicted escalation
* escalation probability
* dataset tags

Group common failures.

Examples:

human_request → sales

account_issue → complaint

refund → complaint

product_question → general_question

Use this report to recommend Dataset V2/V3 improvements.

Do NOT silently modify test labels just because the model predicted them differently.

Flag potentially questionable labels separately for human review.

---

# 15. FASTAPI INFERENCE SERVICE

Build an independent FastAPI inference service.

Suggested structure:

ai_service/
app/
main.py
config.py
schemas.py
services/
classifier.py

```
models/
    intent/
    escalation/

training/
    train_intent.py
    train_escalation.py
    evaluate.py
    validate_dataset.py

data/
    v1/
        train.*
        validation.*
        test.*

tests/

requirements.txt
requirements-training.txt
Dockerfile
README.md
```

You may improve this structure if there is a clear engineering reason.

---

# 16. MODEL LOADING

Load trained models ONCE during application startup.

Do not:

* load models on every request
* retrain models during startup
* depend on training code during inference

Use FastAPI lifespan/startup handling appropriately.

If required model artifacts cannot be loaded, health status should clearly indicate the service is not ready.

---

# 17. HEALTH ENDPOINT

Implement:

GET /health

Return useful information such as:

{
"status": "healthy",
"models_loaded": true,
"intent_model": "intent-v1",
"escalation_model": "escalation-v1"
}

Do not expose sensitive filesystem or infrastructure details.

---

# 18. CLASSIFICATION ENDPOINT

Implement:

POST /v1/classify

Request:

{
"text": "I was charged twice and want my money back."
}

Response:

{
"intent": "refund",
"intent_confidence": 0.94,
"escalation_required": true,
"escalation_confidence": 0.89,
"model_version": {
"intent": "intent-v1",
"escalation": "escalation-v1"
}
}

Optionally support top intent predictions if useful:

{
"top_intents": [
{
"intent": "refund",
"confidence": 0.94
},
{
"intent": "complaint",
"confidence": 0.04
}
]
}

Keep the public API clean.

---

# 19. INPUT VALIDATION

Handle:

* missing text
* empty strings
* whitespace-only strings
* extremely long input
* invalid JSON
* unexpected fields where appropriate

Define a reasonable maximum input length.

Return proper HTTP status codes.

Never return Python stack traces to API consumers.

---

# 20. TESTS

Write meaningful pytest tests.

Dataset tests:

* no leakage
* valid labels
* required intents present
* duplicate detection
* split integrity

Model tests:

* model artifacts load
* intent prediction works
* confidence is valid
* escalation probability is valid
* configurable threshold works
* top-k predictions work

FastAPI tests:

* health endpoint
* classification endpoint
* empty input
* invalid input
* model unavailable
* response schema
* threshold behavior

Training/evaluation tests should not require external APIs.

No OpenAI/Anthropic/Gemini keys should be required anywhere.

---

# 21. PERFORMANCE

Benchmark inference.

Measure approximately:

* model startup/load time
* single-request inference latency
* repeated-request latency
* basic concurrent-request behavior

This is a small classifier and should be CPU-friendly.

Do not optimize prematurely, but identify obvious performance problems.

---

# 22. SECURITY

The service should:

* run without API secrets where possible
* not expose local paths
* validate input
* limit maximum message size
* avoid unsafe model deserialization practices where practical
* not expose stack traces
* not log unnecessary sensitive message contents

Document the security assumptions.

---

# 23. DOCKER

Create a production-oriented Dockerfile.

The container should contain only what is required for inference.

Avoid putting:

* raw training datasets
* unnecessary notebooks
* evaluation artifacts
* training-only dependencies

into the production image unless required.

The service should run independently.

---

# 24. DOCUMENTATION

Create a README covering:

## Development

How to install dependencies.

## Dataset

Dataset schema and annotation policy.

## Training

Exact commands to train both models.

## Evaluation

Exact commands to reproduce metrics.

## Running API

Exact command to start FastAPI.

## Docker

How to build and run the container.

## Configuration

Document variables such as:

ESCALATION_THRESHOLD

## API

Document `/health` and `/v1/classify`.

## Model Versioning

Explain how to create:

intent-v2
escalation-v2

without overwriting V1.

## Limitations

Clearly document weaknesses discovered during evaluation.

---

# 25. VERSIONING

Version all important artifacts independently:

Dataset:
dataset-v1

Intent:
intent-v1

Escalation:
escalation-v1

Evaluation:
evaluation-v1

Store which dataset version trained each model.

Never overwrite an existing production model artifact during experimentation.

---

# 26. EXPERIMENT TRACKING

Create a lightweight experiment record.

For each training run record:

* experiment ID
* dataset version
* model version
* TF-IDF parameters
* Logistic Regression parameters
* random seed
* validation metrics
* test metrics when final evaluation is performed
* escalation threshold recommendation
* timestamp

JSON is sufficient for V1.

Do not introduce MLflow or another large dependency unless there is a demonstrated need.

---

# 27. IMPORTANT ML RULES

Do not:

* use the test set for training
* repeatedly tune against the test set
* inflate the dataset using near-identical paraphrases
* report training accuracy as model quality
* hide poor metrics
* change labels merely to improve scores
* jump to transformers before establishing the baseline
* claim 100% reliability
* implement escalation using keyword rules disguised as ML
* use private customer data

Prefer:

data quality
→ reproducibility
→ evaluation
→ error analysis
→ improvement

over model complexity.

---

# 28. DEVELOPMENT PROCESS

Implement incrementally.

Phase 1:
Dataset + annotation policy + validation

Phase 2:
Intent + escalation training

Phase 3:
Evaluation + threshold analysis + error analysis

Phase 4:
FastAPI inference service

Phase 5:
Automated tests

Phase 6:
Docker + performance checks

Phase 7:
Documentation + final report

Run relevant tests after every phase.

Do not wait until the end to discover broken components.

---

# 29. COMPLETION CRITERIA

Do NOT mark the project complete merely because `/v1/classify` returns HTTP 200.

Completion requires:

* validated dataset
* strict train/validation/test separation
* reproducible training
* versioned model artifacts
* intent evaluation
* escalation evaluation
* threshold sweep
* special-subset evaluation
* error analysis
* working FastAPI service
* model loading at startup
* automated tests
* Docker support
* documentation
* reproducible metrics
* known limitations documented

Model quality and software completeness must be reported separately.

A technically complete API with a weak classifier must be described as:

"Engineering complete, model quality requires improvement."

Do not describe it as production-ready.

---

# 30. FINAL REPORT

At completion provide:

## Dataset

* total examples
* train count
* validation count
* test count
* intent distribution
* escalation distribution
* special subset counts

## Intent Model

* accuracy
* macro precision
* macro recall
* macro F1
* per-class F1
* major confusion pairs

## Escalation Model

* selected threshold
* precision
* recall
* F1
* false positives
* false negatives
* threshold comparison

## Difficult Cases

* Hinglish performance
* confusion-pair performance
* hard-negative performance
* multi-intent performance
* noisy-text performance

## Engineering

* files created
* architecture
* API endpoints
* model versions
* dataset version
* Docker status

## Testing

* tests added
* tests passing
* tests failing

## Performance

* startup time
* inference latency

## Limitations

Be explicit.

## Recommendation

Choose one:

* ready for controlled testing
* needs dataset improvement
* needs model improvement
* ready for production evaluation

Do not exaggerate model quality.

---

# Future Direction — DO NOT IMPLEMENT YET

Design the project so we can later add specialized models such as:

* answerability classifier
* sentiment classifier
* spam classifier
* urgency classifier
* lead-quality classifier
* conversation-topic classifier

We may also later compare:

* TF-IDF + Logistic Regression
* embedding-based classifiers
* sentence-transformer models
* compact transformer classifiers

But do NOT introduce those until the TF-IDF baseline has been properly evaluated.

Build the first system cleanly, measure it honestly, and make future model replacement straightforward.

---

# Continue Specialized ML Service — Correct V1 Threshold and Build V2

We have completed the V1 standalone classification service.

Current V1 architecture:

* TF-IDF + Logistic Regression intent classifier
* TF-IDF + Logistic Regression escalation classifier
* FastAPI inference service
* Dataset validation
* Training/evaluation pipeline
* Model versioning
* Docker
* 31/31 tests passing

## Important correction to V1 escalation threshold

The full held-out test-set threshold sweep is:

| Threshold | Precision | Recall |     F1 |  FP | FN |
| --------- | --------: | -----: | -----: | --: | -: |
| 0.05      |    0.2259 | 1.0000 | 0.3685 | 305 |  0 |
| 0.10      |    0.3175 | 0.9775 | 0.4793 | 187 |  2 |
| 0.15      |    0.4940 | 0.9326 | 0.6459 |  85 |  6 |
| 0.20      |    0.5984 | 0.8202 | 0.6919 |  49 | 16 |
| 0.25      |    0.6889 | 0.6966 | 0.6927 |  28 | 27 |
| 0.30      |    0.7746 | 0.6180 | 0.6875 |  16 | 34 |
| 0.35      |    0.8136 | 0.5393 | 0.6486 |  11 | 41 |
| 0.40      |    0.8261 | 0.4270 | 0.5630 |   8 | 51 |

The previous recommendation of 0.40 was not appropriate for our escalation objective.

At 0.40, 51 genuine escalation cases are missed.

For V1, change the default to:

ESCALATION_THRESHOLD=0.20

Reason:

* Recall = 82.02%
* Precision = 59.84%
* F1 = 0.6919
* FN = 16
* FP = 49

We explicitly prefer some unnecessary human reviews over silently missing genuine escalation cases.

Document:

* 0.15 = conservative/pilot mode
* 0.20 = V1 recommended safety-oriented default
* 0.25 = more balanced precision/recall
* >= 0.30 = not currently recommended for safety-sensitive escalation

The threshold must remain configurable.

Update tests and documentation accordingly.

Do NOT retrain V1 merely because the threshold changed.

---

# Freeze V1

Before starting V2, preserve the complete V1 baseline.

Do NOT overwrite:

* dataset-v1
* intent-v1
* escalation-v1
* V1 evaluation results
* V1 experiment metadata

Record the final V1 baseline:

Intent:

* Accuracy: 62.8%
* Macro F1: 0.61
* product_question F1: 0.38
* refund F1: 0.83

Escalation at threshold 0.20:

* Precision: 0.5984
* Recall: 0.8202
* F1: 0.6919
* FP: 49
* FN: 16

Difficult subsets:

* Hinglish intent accuracy: 66.2%
* Confusion pairs: 47.1%
* Multi-intent: 55.0%
* Noisy/spelling: 51.9%

These become the baseline V2 must beat.

---

# V2 Objective

The V1 error analysis shows that TF-IDF struggles with semantic distinctions.

Major problems include:

* complaint vs technical_support
* product_question vs general_question
* pricing vs general_question
* sales vs general_question
* multi-intent messages
* noisy language
* subtle escalation signals

Build V2 using semantic sentence embeddings.

Do NOT delete or replace V1.

We want an experimental comparison:

V1:
TF-IDF → Logistic Regression

versus

V2:
Sentence Embeddings → Classifier

---

# Step 1 — Dataset V2

Create `dataset-v2`.

Use V1 as a foundation but improve it based on V1 error analysis.

Focus especially on:

* complaint ↔ technical_support
* product_question ↔ general_question
* pricing ↔ general_question
* sales ↔ general_question
* human_request ↔ complaint/sales
* account_issue ↔ complaint
* refund ↔ complaint

Add more:

* hard negatives
* semantic paraphrases
* indirect requests
* negation
* multi-intent messages
* Hinglish
* Indian English
* spelling/noisy text
* subtle escalation cases

Do not simply inflate the dataset with trivial paraphrases.

Run all existing leakage and dataset validation checks.

---

# Step 2 — Preserve a Fair Benchmark

We need a fair V1 vs V2 comparison.

Do not evaluate V2 only on newly generated easy examples.

Create or preserve a frozen benchmark test set that neither V1 nor V2 trains on.

The benchmark must contain substantial numbers of:

* normal messages
* confusion-pair messages
* Hinglish
* noisy messages
* multi-intent messages
* escalation positives
* escalation hard negatives

Do not modify benchmark labels because a model disagrees with them.

Flag questionable annotations separately for manual review.

---

# Step 3 — Semantic Embeddings

Use a suitable lightweight sentence-transformer embedding model.

Requirements:

* CPU-compatible
* reasonable inference latency
* supports semantic similarity well
* suitable for English and preferably multilingual/Hinglish usage

Do not automatically choose the largest model.

Explain which embedding model you selected and why.

Record the exact model name/version.

Do NOT fine-tune the transformer yet.

First use it only as a feature extractor.

Architecture:

text
→ sentence embedding
→ classifier
→ probability

---

# Step 4 — Intent V2

Train:

Sentence Embeddings
→ Logistic Regression

for the same 10 intent classes.

Keep the public prediction contract compatible with V1:

{
"intent": "...",
"confidence": 0.0,
"top_intents": [...]
}

Version:

intent-v2

Record:

* embedding model
* classifier parameters
* dataset version
* random seed
* training timestamp

---

# Step 5 — Escalation V2

Train:

Sentence Embeddings
→ Logistic Regression

for escalation.

Version:

escalation-v2

Do not hardcode the threshold into the model.

Generate another full threshold sweep from at least:

0.05 through 0.95

with reasonable increments.

Report:

* precision
* recall
* F1
* FP
* FN

Do not automatically choose 0.50.

Recommend the operating threshold based on our preference for high escalation recall while maintaining a usable human-review workload.

---

# Step 6 — Compare V1 vs V2

Produce a direct benchmark table.

Intent:

| Metric                  | V1 | V2 | Change |
| ----------------------- | -: | -: | -----: |
| Accuracy                |    |    |        |
| Macro F1                |    |    |        |
| Weighted F1             |    |    |        |
| Hinglish accuracy       |    |    |        |
| Confusion-pair accuracy |    |    |        |
| Multi-intent accuracy   |    |    |        |
| Noisy-text accuracy     |    |    |        |

Also compare F1 for every individual intent.

Pay particular attention to:

* product_question
* general_question
* technical_support
* complaint
* sales
* pricing

---

# Step 7 — Compare Escalation

Compare V1 and V2 at their recommended operating thresholds.

Report:

| Metric    |     V1 | V2 |
| --------- | -----: | -: |
| Threshold |   0.20 |  ? |
| Precision | 0.5984 |  ? |
| Recall    | 0.8202 |  ? |
| F1        | 0.6919 |  ? |
| FP        |     49 |  ? |
| FN        |     16 |  ? |

Also compare both models at identical thresholds so we can understand whether V2 genuinely separates the classes better.

---

# Step 8 — Probability Analysis

Inspect the probability distributions for escalation-positive and escalation-negative examples.

Determine whether V2 creates better separation than V1.

Report statistics or histograms showing where:

* true escalation examples concentrate
* safe examples concentrate

We want to know whether V2 produces a cleaner decision boundary rather than merely shifting the optimal threshold.

---

# Step 9 — Error Analysis

Produce V2 error analysis.

Identify:

* remaining confusion pairs
* high-confidence wrong predictions
* missed escalations
* unnecessary escalations
* Hinglish failures
* multi-intent failures
* noisy-text failures

Pay special attention to **high-confidence incorrect predictions**.

These are more dangerous than uncertain errors.

---

# Step 10 — FastAPI Model Selection

Only after V2 evaluation:

Make the inference architecture capable of selecting model versions cleanly.

For example:

INTENT_MODEL_VERSION=intent-v1
ESCALATION_MODEL_VERSION=escalation-v1

or:

intent-v2
escalation-v2

Do not break the existing `/v1/classify` response contract.

Do not automatically make V2 the production/default model merely because it is newer.

Promote V2 only if evaluation justifies it.

---

# Step 11 — Performance Benchmark

V1 currently gives approximately:

* model load: ~2.3 seconds
* inference: ~1.3 ms/model

Sentence embeddings will be slower.

Measure V2:

* embedding model load time
* total startup time
* single-message inference latency
* repeated inference latency
* batch inference
* memory usage if practical

Compare V1 and V2.

Model quality improvement must justify the additional compute cost.

---

# Step 12 — Tests

Keep all existing tests passing.

Add tests for:

* V2 artifact loading
* embedding generation
* V2 intent prediction
* V2 escalation prediction
* threshold configuration
* V1/V2 model selection
* invalid model version
* API compatibility
* model metadata
* deterministic/reproducible behavior where applicable

No external paid AI API should be required.

---

# Important Rules

Do NOT:

* delete V1
* overwrite V1 artifacts
* claim V2 is better without benchmark evidence
* fine-tune a transformer yet
* use the frozen benchmark for training
* optimize repeatedly against the final benchmark
* choose threshold based only on precision
* choose threshold based only on F1
* hide false negatives
* change labels simply because V2 disagrees
* use OpenAI/Claude/Gemini for runtime classification

The objective is not to make V2 look better.

The objective is to determine whether semantic embeddings genuinely solve the weaknesses identified in V1.

---

# Final Report

At completion provide:

## V1 Baseline

Final frozen metrics.

## Dataset V2

Counts, distributions and improvements.

## Embedding Model

Exact model used and why.

## Intent V2

Complete metrics.

## Escalation V2

Complete threshold sweep and recommended threshold.

## V1 vs V2

Side-by-side comparison.

## Difficult Subsets

Hinglish, confusion pairs, multi-intent, noisy text, hard negatives.

## Performance

V1 vs V2 latency/startup/resource comparison.

## Error Analysis

Remaining weaknesses.

## Tests

Total passing/failing.

## Recommendation

Choose one:

* Keep V1
* Promote V2 for controlled testing
* V2 needs more dataset work
* Try a stronger classifier
* Consider transformer fine-tuning

Do not call the model production-ready unless the evidence genuinely supports that conclusion.
