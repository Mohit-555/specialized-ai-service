# Implementation Progress

## Phase 1: Dataset + Annotation Policy + Validation ✅
- [x] Project directory structure created
- [x] `.gitignore` added
- [x] Annotation policy defined (`annotation_policy.md`)
- [x] Dataset generation script (2004 examples)
- [x] Dataset validation script
- [x] Dataset files generated (train/validation/test splits)

## Phase 2: Model Training ✅
- [x] Intent model training script (`training/train_intent.py`)
- [x] Escalation model training script (`training/train_escalation.py`)
- [x] Trained intent-v1 model (TF-IDF + Logistic Regression)
- [x] Trained escalation-v1 model (TF-IDF + Logistic Regression)

## Phase 3: Evaluation ✅
- [x] Intent evaluation (accuracy: 62.8%)
- [x] Escalation evaluation + threshold sweep (recommended: 0.40)
- [x] Special subset evaluation
- [x] Error analysis (170 total errors, top confusions identified)

## Phase 4: FastAPI Service ✅
- [x] FastAPI application with lifespan model loading
- [x] `GET /health` endpoint
- [x] `POST /v1/classify` endpoint
- [x] Input validation + error handling
- [x] Configurable escalation threshold

## Phase 5: Automated Tests ✅
- [x] Dataset tests (11 tests) — leakage, validity, distribution
- [x] Model tests (11 tests) — loading, prediction, metadata
- [x] API tests (9 tests) — health, classify, validation, schema
- [x] **31/31 tests passing**

## Phase 6: Docker ✅
- [x] Production Dockerfile
- [x] Dependency separation (requirements.txt vs requirements-training.txt)
- [x] Performance verified (~2.3s load, ~1.3ms inference)

## Phase 7: Documentation ✅
- [x] `README.md` with setup, training, API, and Docker instructions
- [x] `annotation_policy.md` with detailed intent definitions
- [x] Final report below

---

## Final Report

### Dataset
| Metric | Value |
|--------|-------|
| Total examples | 2004 (1994 unique after dedup) |
| Train count | 1296 (65.0%) |
| Validation count | 300 (15.0%) |
| Test count | 398 (20.0%) |
| Intent classes | 10 |
| Escalation-positive | 401 (20.1%) |
| Hinglish examples | ~350 |
| Noisy/spelling errors | ~180 |
| Multi-intent | ~82 |
| Confusion pairs | ~67 |
| Escalation hard negatives | ~68 |

### Intent Model (intent-v1)
| Metric | Value |
|--------|-------|
| Accuracy | 62.8% |
| Macro avg F1 | 0.61 |
| Weighted avg F1 | 0.64 |
| Best class | refund (F1: 0.83) |
| Weakest class | product_question (F1: 0.38), other (F1: 0.48) |

**Top confusion pairs:**
- complaint → technical_support: 16
- product_question → general_question: 10
- pricing → general_question: 10
- general_question → technical_support: 9
- sales → general_question: 8

### Escalation Model (escalation-v1)
| Metric | Value (threshold=0.40) |
|--------|----------------------|
| Precision | 0.826 |
| Recall | 0.427 |
| F1 | 0.563 |
| False positives | 8 |
| False negatives | 51 |

**Threshold sweep recommendation: 0.40**
Balances acceptable recall and precision. Lower thresholds improve recall but introduce too many false positives.

### Difficult Cases
| Subset | Intent Accuracy | Count |
|--------|----------------|-------|
| Overall | 62.8% | 398 |
| Hinglish | 66.2% | 77 |
| Confusion pairs | 47.1% | 17 |
| Hard-negatives (escalation) | 41.2% | 17 |
| Multi-intent | 55.0% | 20 |
| Noisy/spelling errors | 51.9% | 27 |

### Engineering
| Component | Status |
|-----------|--------|
| Files created | ~25 |
| Architecture | FastAPI + sklearn pipelines |
| API endpoints | `GET /health`, `POST /v1/classify` |
| Model versions | intent-v1, escalation-v1 |
| Dataset version | dataset-v1 |
| Docker support | ✅ (Dockerfile) |

### Testing
- Tests added: 31
- Tests passing: 31
- Tests failing: 0

### Performance
| Metric | Value |
|--------|-------|
| Model load time | ~2.3s |
| Single inference (intent) | ~1.3ms |
| Single inference (escalation) | ~1.3ms |

### Limitations
1. Intent accuracy at 62.8% — TF-IDF baseline struggles with nuanced intent boundaries
2. Escalation recall is low (42.7% at recommended threshold) — many subtle escalation signals missed
3. Confusion between complaint↔technical_support is the largest error source
4. Multi-intent messages handled poorly (55% accuracy)
5. Confusion-pair examples have very low accuracy (47%)
6. Model relies on keyword patterns rather than semantic understanding
7. No sentence embeddings or transformer models used yet

### Recommendation
**Needs dataset improvement and model improvement** — Engineering complete, model quality requires improvement before production deployment.

---

## Phase 8: V2 Continuation — Sentence Embeddings ✅

### Changes from V1
- **Escalation threshold corrected from 0.40 → 0.20** (V1 threshold sweep originally recommended 0.40, but 0.20 gives 82% recall vs 43% at 0.40)
- **V2 uses `all-MiniLM-L6-v2` sentence embeddings** instead of TF-IDF, with the same Logistic Regression head
- **No transformer fine-tuning** — embeddings frozen, only LR weights learned
- **Model artifacts stored in versioned directories** (`intent-v2/`, `escalation-v2/`)

### V1 Escalation Threshold Correction
The original recommendation was 0.40 (best F1=0.56), but this only captured 43% of escalations. Changed to 0.20 which captures 82% of escalations (FP: 49, FN: 16) — prioritizing recall over precision since missing escalations is costlier than false positives.

### V2 Results

#### Intent (10-class)

| Metric | V1 | V2 | Change |
|--------|----|----|--------|
| Accuracy | 62.8% | 76.1% | **+13.3%** |
| Macro F1 | 61.2% | 75.5% | +14.3% |

#### Escalation

| Threshold | V1 Prec | V1 Rec | V1 F1 | V1 FP | V1 FN | V2 Prec | V2 Rec | V2 F1 | V2 FP | V2 FN |
|-----------|---------|--------|-------|-------|-------|---------|--------|-------|-------|-------|
| 0.15 | 0.494 | 0.933 | 0.646 | 85 | 6 | 0.500 | 0.944 | 0.654 | 84 | 5 |
| 0.20 | 0.598 | 0.820 | 0.692 | 49 | 16 | 0.544 | 0.910 | 0.681 | 68 | 8 |
| 0.25 | 0.689 | 0.697 | 0.693 | 28 | 27 | 0.626 | 0.865 | 0.726 | 46 | 12 |

V2 recommended: **0.25** (recall 86.5%, FP 46, FN 12) — beats V1 on all metrics at comparable recall.

#### Subset Improvements (V2 intent)

| Subset | V1 | V2 | Change |
|--------|----|----|--------|
| hinglish | 66.2% | 71.4% | +5.2% |
| confusion_pair | 47.1% | 76.5% | +29.4% |
| hard_negative_escalation | 41.2% | 64.7% | +23.5% |
| multi_intent | 55.0% | 65.0% | +10.0% |
| noisy | 51.9% | 66.7% | +14.8% |

#### Key Findings
- V2 improves intent accuracy across all 10 classes vs V1
- Largest gains: complaint (+0.31), product_question (+0.27), other (+0.23)
- Escalation separation improves from 0.250 (V1) to 0.368 (V2) — 47% better
- V2 still struggles with hard_negative_escalation (only 47.1% esc accuracy) — suggests need for harder training examples
- V2 @ 0.25 beats V1 @ 0.20 at same recall level with fewer FPs and FNs

### Updated Test Suite
- **Tests added:** 5 new V2-specific tests
- **Tests total:** 36
- **Tests passing:** 36

### API Model Selection
Service now supports runtime model selection via env vars:
- `INTENT_MODEL_VERSION` (intent-v1 / intent-v2 / intent-v3)
- `ESCALATION_MODEL_VERSION` (escalation-v1 / escalation-v2 / escalation-v3)
- `EMBEDDING_MODEL_NAME` (for V2/V3, default: all-MiniLM-L6-v2)

---

## Phase 9: Dataset V3 + Model V3 ✅

### 1. V2 Error Analysis
- **95 intent errors** (76.1% accuracy), only **3 high-confidence** (>0.7)
- Top confusions: product_question→general_question (11), tech_support↔complaint (14), pricing↔sales (8)
- **Escalation**: 46 FP, 12 FN at threshold 0.25
- Hard negatives only **47.1% esc accuracy** — biggest weakness
- **FNs**: 5 in hinglish, 5 subtle/unresolved, 1 time-persistent, 1 multi_intent
- **High-confidence errors**: sales→pricing ("Your pricing page is confusing..."), other→complaint (irony), general_question→refund ("How are refunds processed?")

### 2. Dataset V3
- **3,726 unique examples** (2,004 from V1 + 1,722 new)
- **317 exact duplicates removed** during validation
- **8 near-duplicate pairs flagged** (cosine >0.95, mostly missing trailing periods)
- **No cross-split leakage** — all splits verified clean
- **Special tags**: hinglish (730), noisy (419), hard_negative_escalation (185), multi_intent (205), confusion_pair (79), negation (67), resolution_state (76), escalation_positive (271)
- **Frozen benchmark**: 200 difficult examples isolated from training
- **Split**: 2,329 train / 499 validation / 698 test / 200 benchmark

### 3. Model V3 Training
- Same architecture: all-MiniLM-L6-v2 + Logistic Regression (no fine-tuning)
- **Intent-v3 validation**: 76.75% accuracy (vs V2 test at 76.1%)
- **Escalation-v3 validation best**: F1=0.732 at threshold 0.35

### 4. V1 vs V2 vs V3 Comparison

#### Intent (V3 Test Set, 698 examples)

| Metric | V1 | V2 | V3 |
|--------|----|----|----|
| Accuracy | 69.9% | 76.4% | 77.1% |
| Macro F1 | 66.7% | 74.4% | 74.4% |
| Weighted F1 | 69.1% | 76.1% | 76.7% |

Per-class F1 (V3):

| Class | V1 | V2 | V3 |
|-------|----|----|----|
| account_issue | 0.723 | 0.777 | **0.820** |
| complaint | 0.535 | 0.662 | **0.698** |
| general_question | 0.684 | **0.772** | 0.755 |
| human_request | 0.704 | **0.746** | 0.654 |
| other | 0.375 | **0.683** | 0.632 |
| pricing | 0.752 | 0.823 | **0.846** |
| product_question | 0.576 | **0.588** | 0.585 |
| refund | **0.900** | 0.892 | 0.889 |
| sales | 0.674 | 0.693 | **0.742** |
| technical_support | 0.751 | 0.804 | **0.816** |

#### Escalation (V3 Test Set @ threshold 0.25)

| Metric | V1 | V2 | V3 |
|--------|----|----|----|
| Precision | 0.717 | 0.633 | 0.648 |
| Recall | 0.707 | 0.850 | 0.843 |
| F1 | 0.712 | **0.726** | **0.733** |
| FP | 39 | 69 | 64 |
| FN | 41 | 21 | 22 |

V3 best escalation: F1=0.733 @ threshold 0.25

#### Special Subsets (Intent Accuracy)

| Subset | Count | V1 | V2 | V3 |
|--------|-------|----|----|----|
| hinglish | 107 | 71.0% | 68.2% | 68.2% |
| confusion_pair | 15 | 73.3% | 66.7% | 66.7% |
| hard_negative_esc | 24 | 45.8% | 66.7% | **70.8%** |
| multi_intent | 29 | 69.0% | 65.5% | 65.5% |
| noisy | 55 | 63.6% | 70.9% | **74.6%** |
| negation | 8 | 62.5% | 37.5% | 50.0% |
| resolution_state | 9 | 44.4% | 55.6% | **66.7%** |

#### Special Subsets (Escalation Accuracy)

| Subset | V1 | V2 | V3 |
|--------|----|----|----|
| hinglish | 78.5% | 80.4% | **83.2%** |
| hard_negative_esc | 58.3% | 41.7% | **66.7%** |
| noisy | 80.0% | 80.0% | **83.6%** |
| negation | 37.5% | 25.0% | 37.5% |
| resolution_state | 66.7% | 44.4% | 55.6% |

#### Backward Compatibility (V1 Test Set, 398 examples)

| Metric | Previously | Current V2 | Current V3 |
|--------|-----------|-----------|-----------|
| Intent Accuracy | 76.1% | 76.1% | **76.4%** |
| Escalation F1 @0.25 | 0.726 | 0.726 | **0.732** |
| Escalation Recall @0.25 | 86.5% | 86.5% | **87.6%** |
| Escalation FP @0.25 | 46 | 46 | 46 |
| Escalation FN @0.25 | 12 | 12 | **11** |

No regression on V1 test set.

### 5. Escalation FN Analysis (V3, 22 total)
- **hinglish**: 5 FNs (low confidence on Hinglish escalation signals)
- **multi_intent**: 2 FNs (conflicting signals)
- **subtle/unresolved**: 14 FNs (main category — indirect escalation signals without strong keywords)
- **time-persistent**: 1 FN (waited long but no strong frustration keywords)

### 6. Probability Calibration

| Model | ECE | Brier |
|-------|-----|-------|
| V1 | 0.108 | 0.111 |
| V2 | 0.061 | 0.086 |
| **V3** | **0.059** | **0.084** |

V3 is the best-calibrated model. All models show overconfidence at mid-range (0.3-0.5).

#### Calibration improvement test (V3 only)

| Calibration | Brier | ECE | Best F1 | Best Threshold |
|------------|-------|-----|---------|---------------|
| Uncalibrated | 0.0844 | 0.0587 | 0.7329 | 0.25 |
| Platt (sigmoid) | 0.0848 | 0.0453 | 0.7398 | 0.15 |
| Isotonic | 0.0812 | 0.0306 | 0.7307 | 0.25 |

**Result**: Both calibration methods reduce ECE (better probability reliability) but:
- Platt shifts best threshold from 0.25→0.15 (too aggressive)
- Isotonic barely changes F1
- **Neither is recommended for production** — the F1 gain is negligible and the complexity isn't justified

### 7. Confidence / Abstention Analysis

**V3 on validation at threshold 0.5**: Keep 65.3%, keep-accuracy 89.0%, reject-accuracy 53.8%
**V3 on test at threshold 0.5**: Keep 64.2%, keep-accuracy 89.3%, reject-accuracy 55.2%

**V2 on test at threshold 0.5**: Keep 58.5%, keep-accuracy 90.0%, reject-accuracy 57.2%

V3 keeps more messages at high confidence (64% vs 58%) with similar high-confidence accuracy (~89%).

At threshold 0.7: V3 keeps 40.1% with 95.0% accuracy on kept predictions.

### 8. Performance

| Operation | Latency |
|-----------|---------|
| Model loading (V2/V3) | ~0.01s (classifier only) |
| Embedding model load | ~9.5s (once) |
| Embed + classify (1 msg) | ~19ms |
| Embed (10 msgs batch) | ~41ms total = 4ms/msg |

V3 is architecturally identical to V2 — same embedding model, same LR classifier. Performance is identical.

### Summary

| Aspect | V1 | V2 vs V1 | V3 vs V2 |
|--------|----|----------|----------|
| Architecture | TF-IDF+LR | Emb+LR | Emb+LR (same) |
| Dataset | 2,004 | 2,004 | 3,726 |
| Intent Accuracy | 62.8% | **+13.3%** | **+0.7%** |
| Escalation F1 | 0.692 | **+0.034** | **+0.007** |
| Hard-neg esc acc | — | 41.7% | **66.7%** |
| Calibration ECE | 0.108 | 0.061 | **0.059** |
| Tests | 31 | 36 | **51** |

### Recommendation

**Do NOT promote V3 to default.**

V3 improves over V2 in important ways:
- Hard-negative escalation accuracy: 41.7% → **66.7%** (+25pp)
- Intent accuracy on backward-compat V1 test: 76.1% → **76.4%** (no regression)
- Escalation recall at same threshold: 86.5% → **87.6%**
- Better calibration (ECE 0.061 → 0.059)
- Frozen benchmark enables disciplined evaluation

But the improvements are **modest** for the data added:
- 3,726 examples (86% more data) → only +0.7% accuracy
- Per-class F1 actually *decreased* for general_question (0.772→0.755), human_request (0.746→0.654), other (0.683→0.632), refund (0.892→0.889), product_question (0.588→0.585)
- Escalation improvement is primarily in hard-negative handling, not overall metrics

**Keep V2 as default**. The data added in V3 helps specific subsets but doesn't justify a model version bump for the existing pipeline. However, V3's frozen benchmark is valuable for future evaluation.

### Decision: V4 Transformer Fine-Tuning

**Yes, the MiniLM + LR architecture is reaching a performance plateau.**

Evidence:
1. **86% more data** (2,004→3,726) yielded only **+0.7% accuracy**
2. **Escalation F1 barely moved** (0.726→0.733)
3. **Per-class F1 regressed for 5/10 classes** despite more data
4. The **22 false negatives** that remain are fundamentally semantic: the model can't distinguish "I've waited 2 weeks" (should escalate) from "My account was locked but it's fixed" (should not escalate) because the difference is in the *temporal/contextual* meaning, not keyword presence
5. **Hinglish FNs** (5/22) suggest the embedding model lacks coverage for code-mixed Hindi-English — fine-tuning on code-mixed text would help
6. **Subtle/unresolved FNs** (14/22) are cases where a human understands the implied urgency but the embedding doesn't encode it strongly enough

**Proposed V4 experiment** (do NOT implement without approval):
- Fine-tune a compact transformer (e.g., `distilbert-base-uncased` or `MiniLM-L6-v2`) on the intent+escalation task
- Add hinge-style training or contrastive learning for hard negatives
- Expected gains:
  - Intent: +5-8% over V3 (to ~82-85%)
  - Escalation hard negatives: +15-20pp over V3
  - Hinglish handling: significant improvement with code-mixed data
- Risk: increased inference latency (~50-100ms vs current ~19ms)

### Final V3 Files Created
| File | Purpose |
|------|---------|
| `training/create_dataset_v3.py` | Dataset V3 generator (4,043 raw) |
| `training/prepare_dataset_v3.py` | Validate, dedup, split, create benchmark |
| `data/v3/` | Dataset V3 splits + frozen benchmark |
| `training/train_intent_v3.py` | Intent-v3 training |
| `training/train_escalation_v3.py` | Escalation-v3 training (+ threshold sweep) |
| `training/compare_v1_v2_v3.py` | Comprehensive V1/V2/V3 comparison (11 analysis sections) |
| `training/analyze_v2_errors.py` | V2 error analysis used for V3 planning |
| `models/intent/intent-v3/` | V3 intent model + metadata |
| `models/escalation/escalation-v3/` | V3 escalation model + metadata |
| `evaluations/evaluation-v3/` | Evaluation results |
| `tests/test_models.py` | +15 new tests (51 total) |
